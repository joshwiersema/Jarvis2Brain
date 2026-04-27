"""Brain NN: autoencoder + 2D projector + link predictor.

This is the model that genuinely *trains* on the markdown vault. Every note
embedding goes in; the network learns:

- A compressed latent (autoencoder reconstruction loss)
- A 2D projection used as the live `xy` coordinate in /graph
- A link-prediction head that proposes new wiki-links from latent similarity

Training data is built from the vault: positive pairs are existing
[[wiki-links]], negatives are random non-linked pairs. As the brain consumes
more notes, the latent space self-organizes — what starts as deterministic
chaos (xy seeded from slug hash) drifts toward semantic clusters.

Stays small on purpose: 16-d latent, 2-d projection, ~few thousand params.
Trains in seconds on CPU even for thousands of notes.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

LATENT_DIM = 16


@dataclass
class TrainStep:
    epoch: int
    step: int
    recon_loss: float
    link_loss: float
    timestamp: float


class BrainNet(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = LATENT_DIM) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        h = max(latent_dim * 4, 32)

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, h),
            nn.GELU(),
            nn.Linear(h, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, h),
            nn.GELU(),
            nn.Linear(h, input_dim),
        )
        # 2D projection head: latent -> small hidden -> 2 (tanh in [-1, 1]).
        self.projector = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2 or 4),
            nn.GELU(),
            nn.Linear(latent_dim // 2 or 4, 2),
            nn.Tanh(),
        )
        # Link prediction: bilinear over latent pairs -> logit.
        self.link_bilinear = nn.Bilinear(latent_dim, latent_dim, 1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def project(self, z: torch.Tensor) -> torch.Tensor:
        return self.projector(z)

    def link_logits(self, z_a: torch.Tensor, z_b: torch.Tensor) -> torch.Tensor:
        return self.link_bilinear(z_a, z_b).squeeze(-1)

    def forward(
        self,
        x: torch.Tensor,
        pairs: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ):
        z = self.encode(x)
        recon = self.decode(z)
        xy = self.project(z)
        link = None
        if pairs is not None:
            a_idx, b_idx = pairs
            link = self.link_logits(z[a_idx], z[b_idx])
        return {"latent": z, "recon": recon, "xy": xy, "link": link}


def cosine_recon_loss(recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    # target embeddings are L2-normalized upstream; recon may not be.
    recon_n = F.normalize(recon, dim=-1)
    target_n = F.normalize(target, dim=-1)
    return (1.0 - (recon_n * target_n).sum(dim=-1)).mean()


def link_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logits, labels.float())


def init_brain_net(
    input_dim: int,
    latent_dim: int = LATENT_DIM,
    seed: Optional[int] = None,
) -> BrainNet:
    if seed is not None:
        torch.manual_seed(seed)
    return BrainNet(input_dim=input_dim, latent_dim=latent_dim)


def save_checkpoint(net: BrainNet, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": net.state_dict(),
            "input_dim": net.input_dim,
            "latent_dim": net.latent_dim,
        },
        path,
    )


def load_checkpoint(path: Path) -> Optional[BrainNet]:
    path = Path(path)
    if not path.exists():
        return None
    data = torch.load(path, map_location="cpu", weights_only=True)
    net = BrainNet(input_dim=data["input_dim"], latent_dim=data["latent_dim"])
    net.load_state_dict(data["state_dict"])
    return net


def append_history(path: Path, step: TrainStep) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {
                    "ts": step.timestamp,
                    "epoch": step.epoch,
                    "step": step.step,
                    "recon": step.recon_loss,
                    "link": step.link_loss,
                }
            )
            + "\n"
        )


def read_history(path: Path, tail: int = 200) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    out: list[dict] = []
    for line in lines[-tail:]:
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def project_to_xy(net: BrainNet, embeddings: np.ndarray) -> np.ndarray:
    """Run inference on a batch of embeddings, return Nx2 xy in [-1, 1]."""
    net.eval()
    with torch.no_grad():
        x = torch.from_numpy(embeddings.astype(np.float32))
        out = net(x)
        return out["xy"].cpu().numpy()


def edge_probabilities(net: BrainNet, embeddings: np.ndarray) -> np.ndarray:
    """NxN sigmoid link-prediction matrix (excluding self-links)."""
    net.eval()
    with torch.no_grad():
        x = torch.from_numpy(embeddings.astype(np.float32))
        z = net.encode(x)
        n = z.shape[0]
        if n == 0:
            return np.zeros((0, 0), dtype=np.float32)
        a = z.unsqueeze(1).expand(n, n, -1).reshape(n * n, -1)
        b = z.unsqueeze(0).expand(n, n, -1).reshape(n * n, -1)
        logits = net.link_logits(a, b).reshape(n, n)
        probs = torch.sigmoid(logits).cpu().numpy()
        # Zero the diagonal — no self-links.
        np.fill_diagonal(probs, 0.0)
        return probs


def now() -> float:
    return time.time()


@dataclass
class EpochSummary:
    avg_recon: float
    avg_link: float
    n_pairs: int
    n_docs: int
