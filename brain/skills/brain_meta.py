"""Brain meta skills — train, training history, edge proposals."""

from __future__ import annotations

import json
import os
from pathlib import Path

from brain.embed import get_embedder
from brain.index import MemoryIndex
from brain.loops.train_brain import BrainTrainer
from brain.memory import Memory
from brain.nn import read_history
from brain.skill import skill
from brain.vault import Vault, resolve_vault_path


def _vault_path() -> Path:
    return resolve_vault_path(cli=os.environ.get("BRAIN_VAULT"), cwd=Path.cwd())


@skill(name="brain_train", description="Train the brain NN for N epochs over the vault")
def brain_train(epochs: int = 1) -> dict:
    vault_path = _vault_path()
    v = Vault(vault_path)
    embedder = get_embedder()
    idx_path = vault_path / ".brain" / "index.npz"
    index = MemoryIndex.load(embedder, idx_path)
    memory = Memory(v, index)
    if not memory.index.all_slugs():
        memory.reindex_from_vault()
        memory.index.save()
    trainer = BrainTrainer(memory)
    summaries = trainer.train_full(epochs=epochs)
    return {
        "epochs": [
            {
                "avg_recon": s.avg_recon,
                "avg_link": s.avg_link,
                "n_pairs": s.n_pairs,
                "n_docs": s.n_docs,
            }
            for s in summaries
        ]
    }


@skill(name="brain_training_history", description="Tail of NN training-loss history")
def brain_training_history(tail: int = 100) -> list[dict]:
    return read_history(_vault_path() / ".brain" / "training_history.jsonl", tail=tail)


@skill(name="brain_edge_proposals", description="Pending NN-proposed wikilinks")
def brain_edge_proposals() -> list[dict]:
    p = _vault_path() / ".brain" / "edge_proposals.jsonl"
    if not p.exists():
        return []
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        try:
            out.append(json.loads(line))
        except Exception:
            continue
    return out
