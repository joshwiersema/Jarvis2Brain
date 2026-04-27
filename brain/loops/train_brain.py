"""Brain trainer — orchestrates learning over the vault.

Two entry points:

- `train_step(slugs, embeddings, pairs)` — small online update after a
  new/updated note. Cheap (a handful of grad steps). Called from the daemon.
- `train_epoch(memory, ...)` — full pass over the vault. Run nightly or on
  demand via `brain train`. Emits epoch-level loss to the training history.

Both keep the model small (16-d latent), L2-regularized via implicit weight
decay, and bounded by `max_steps` so a runaway vault can never DOS the daemon.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from brain.embed import text_for_embedding
from brain.index import IndexedDoc, MemoryIndex
from brain.memory import Memory
from brain.nn import (
    BrainNet,
    EpochSummary,
    TrainStep,
    append_history,
    cosine_recon_loss,
    edge_probabilities,
    init_brain_net,
    link_loss,
    load_checkpoint,
    project_to_xy,
    save_checkpoint,
)
from brain.note import Note
from brain.vault import Vault


@dataclass
class TrainerConfig:
    lr: float = 1e-3
    weight_decay: float = 1e-4
    online_steps: int = 3
    epoch_max_steps: int = 500
    batch_size: int = 32
    neg_per_pos: int = 4
    edge_proposal_threshold: float = 0.7
    seed: int = 42


class BrainTrainer:
    def __init__(
        self,
        memory: Memory,
        net: Optional[BrainNet] = None,
        config: Optional[TrainerConfig] = None,
        checkpoint_dir: Optional[Path] = None,
    ) -> None:
        self.memory = memory
        self.config = config or TrainerConfig()
        self.checkpoint_dir = (
            Path(checkpoint_dir)
            if checkpoint_dir
            else memory.vault.path / ".brain" / "models"
        )
        self.history_path = self.checkpoint_dir.parent / "training_history.jsonl"
        self.proposals_path = self.checkpoint_dir.parent / "edge_proposals.jsonl"
        self._rng = random.Random(self.config.seed)

        if net is None:
            ckpt = self.checkpoint_dir / "brain_nn.pt"
            net = load_checkpoint(ckpt) or init_brain_net(
                input_dim=memory.index._dim,
                seed=self.config.seed,
            )
        self.net = net
        self.optimizer = torch.optim.AdamW(
            self.net.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )

    # ----- data prep -----

    def _gather(self) -> tuple[list[str], np.ndarray, list[tuple[int, int]]]:
        slugs, mat = self.memory.index.vectors_matrix()
        slug_to_i = {s: i for i, s in enumerate(slugs)}
        pos_pairs: list[tuple[int, int]] = []
        for slug in slugs:
            try:
                note = self.memory.vault.read(slug)
            except Exception:
                continue
            for tgt in note.links_out:
                if tgt in slug_to_i and tgt != slug:
                    pos_pairs.append((slug_to_i[slug], slug_to_i[tgt]))
        return slugs, mat, pos_pairs

    def _sample_negatives(self, n: int, n_pos: int) -> list[tuple[int, int]]:
        if n < 2:
            return []
        out: list[tuple[int, int]] = []
        for _ in range(n_pos * self.config.neg_per_pos):
            a = self._rng.randrange(n)
            b = self._rng.randrange(n)
            if a == b:
                continue
            out.append((a, b))
        return out

    # ----- online -----

    def train_step(self, slug: str) -> Optional[float]:
        """Cheap online update after a single note write.

        Trains a few grad steps over the current full vector matrix so the
        new note's neighbourhood gets reorganized. Returns the final recon
        loss, or None if there's nothing to train on.
        """
        slugs, mat, pos_pairs = self._gather()
        if mat.shape[0] == 0:
            return None
        x = torch.from_numpy(mat.astype(np.float32))
        last_loss = 0.0
        self.net.train()
        for _ in range(self.config.online_steps):
            self.optimizer.zero_grad()
            out = self.net(x)
            recon = cosine_recon_loss(out["recon"], x)
            ll = torch.tensor(0.0)
            if pos_pairs:
                a_idx = torch.tensor([p[0] for p in pos_pairs], dtype=torch.long)
                b_idx = torch.tensor([p[1] for p in pos_pairs], dtype=torch.long)
                neg = self._sample_negatives(mat.shape[0], len(pos_pairs))
                if neg:
                    a_idx = torch.cat([a_idx, torch.tensor([p[0] for p in neg])])
                    b_idx = torch.cat([b_idx, torch.tensor([p[1] for p in neg])])
                    labels = torch.cat([
                        torch.ones(len(pos_pairs)),
                        torch.zeros(len(neg)),
                    ])
                    z = self.net.encode(x)
                    logits = self.net.link_logits(z[a_idx], z[b_idx])
                    ll = link_loss(logits, labels)
            loss = recon + 0.5 * ll
            loss.backward()
            self.optimizer.step()
            last_loss = float(recon.detach())
        append_history(
            self.history_path,
            TrainStep(
                epoch=-1,
                step=0,
                recon_loss=last_loss,
                link_loss=float(ll.detach()) if isinstance(ll, torch.Tensor) else 0.0,
                timestamp=time.time(),
            ),
        )
        return last_loss

    # ----- full epoch -----

    def train_epoch(self, *, epoch: int = 0) -> EpochSummary:
        slugs, mat, pos_pairs = self._gather()
        n = mat.shape[0]
        if n == 0:
            return EpochSummary(0.0, 0.0, 0, 0)

        x = torch.from_numpy(mat.astype(np.float32))
        recon_total = 0.0
        link_total = 0.0
        steps = 0
        self.net.train()

        for step in range(min(self.config.epoch_max_steps, max(1, n // self.config.batch_size + 1))):
            idx = torch.tensor(
                [self._rng.randrange(n) for _ in range(min(self.config.batch_size, n))],
                dtype=torch.long,
            )
            xb = x[idx]
            self.optimizer.zero_grad()
            out = self.net(xb)
            recon = cosine_recon_loss(out["recon"], xb)
            ll = torch.tensor(0.0)
            if pos_pairs:
                pp = self._rng.sample(pos_pairs, k=min(len(pos_pairs), self.config.batch_size))
                a = torch.tensor([p[0] for p in pp], dtype=torch.long)
                b = torch.tensor([p[1] for p in pp], dtype=torch.long)
                neg = self._sample_negatives(n, len(pp))
                if neg:
                    a = torch.cat([a, torch.tensor([p[0] for p in neg])])
                    b = torch.cat([b, torch.tensor([p[1] for p in neg])])
                    labels = torch.cat([torch.ones(len(pp)), torch.zeros(len(neg))])
                    z_full = self.net.encode(x)
                    logits = self.net.link_logits(z_full[a], z_full[b])
                    ll = link_loss(logits, labels)
            loss = recon + 0.5 * ll
            loss.backward()
            self.optimizer.step()
            recon_total += float(recon.detach())
            link_total += float(ll.detach()) if isinstance(ll, torch.Tensor) else 0.0
            steps += 1
            append_history(
                self.history_path,
                TrainStep(
                    epoch=epoch,
                    step=step,
                    recon_loss=float(recon.detach()),
                    link_loss=float(ll.detach()) if isinstance(ll, torch.Tensor) else 0.0,
                    timestamp=time.time(),
                ),
            )

        return EpochSummary(
            avg_recon=recon_total / max(1, steps),
            avg_link=link_total / max(1, steps),
            n_pairs=len(pos_pairs),
            n_docs=n,
        )

    # ----- output side-effects -----

    def write_xy_back(self) -> int:
        slugs, mat = self.memory.index.vectors_matrix()
        if not slugs:
            return 0
        xy = project_to_xy(self.net, mat)
        n = 0
        for slug, (x, y) in zip(slugs, xy):
            try:
                note = self.memory.vault.read(slug)
            except Exception:
                continue
            new_xy = (float(x), float(y))
            if note.xy is not None:
                dx = abs(note.xy[0] - new_xy[0])
                dy = abs(note.xy[1] - new_xy[1])
                if dx + dy < 1e-3:
                    continue
            self.memory.vault.update(slug, xy=new_xy)
            n += 1
        return n

    def propose_edges(self) -> list[dict]:
        slugs, mat = self.memory.index.vectors_matrix()
        if len(slugs) < 2:
            return []
        probs = edge_probabilities(self.net, mat)
        existing: set[tuple[str, str]] = set()
        for slug in slugs:
            try:
                note = self.memory.vault.read(slug)
            except Exception:
                continue
            for tgt in note.links_out:
                existing.add((slug, tgt))

        proposals: list[dict] = []
        thr = self.config.edge_proposal_threshold
        for i, a in enumerate(slugs):
            for j, b in enumerate(slugs):
                if i == j:
                    continue
                if (a, b) in existing:
                    continue
                p = float(probs[i, j])
                if p >= thr:
                    proposals.append({"from": a, "to": b, "prob": p, "ts": time.time()})

        # Persist top-N (by prob, dedup by (from,to)).
        seen: set[tuple[str, str]] = set()
        proposals.sort(key=lambda p: p["prob"], reverse=True)
        deduped: list[dict] = []
        for p in proposals:
            key = (p["from"], p["to"])
            if key in seen:
                continue
            seen.add(key)
            deduped.append(p)
        path = self.proposals_path
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for p in deduped[:200]:
                f.write(__import__("json").dumps(p) + "\n")
        return deduped[:200]

    def save(self) -> None:
        save_checkpoint(self.net, self.checkpoint_dir / "brain_nn.pt")

    # ----- convenience -----

    def train_full(self, *, epochs: int = 3) -> list[EpochSummary]:
        out = []
        for e in range(epochs):
            out.append(self.train_epoch(epoch=e))
        self.write_xy_back()
        self.propose_edges()
        self.save()
        return out
