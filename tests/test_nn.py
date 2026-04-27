"""Brain NN unit tests — deterministic + smoke."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from brain.nn import (
    BrainNet,
    cosine_recon_loss,
    edge_probabilities,
    init_brain_net,
    link_loss,
    load_checkpoint,
    project_to_xy,
    save_checkpoint,
)


def test_brainnet_shapes():
    net = init_brain_net(input_dim=32, seed=0)
    x = torch.randn(4, 32)
    out = net(x)
    assert out["latent"].shape == (4, 16)
    assert out["recon"].shape == (4, 32)
    assert out["xy"].shape == (4, 2)
    assert out["link"] is None


def test_xy_in_unit_square():
    net = init_brain_net(input_dim=8, seed=0)
    x = torch.randn(20, 8)
    out = net(x)
    assert torch.all(out["xy"] >= -1.0) and torch.all(out["xy"] <= 1.0)


def test_link_logits_shape():
    net = init_brain_net(input_dim=8, seed=0)
    x = torch.randn(10, 8)
    pairs = (torch.tensor([0, 1, 2]), torch.tensor([3, 4, 5]))
    out = net(x, pairs=pairs)
    assert out["link"].shape == (3,)


def test_train_loop_decreases_loss():
    """Smoke test: a tiny supervised reconstruction loop should reduce loss."""
    torch.manual_seed(0)
    net = init_brain_net(input_dim=16, seed=0)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-2)
    x = torch.randn(8, 16)
    x = x / x.norm(dim=-1, keepdim=True)
    losses = []
    for _ in range(50):
        opt.zero_grad()
        out = net(x)
        loss = cosine_recon_loss(out["recon"], x)
        loss.backward()
        opt.step()
        losses.append(float(loss))
    assert losses[-1] < losses[0]


def test_link_loss_correct_signs():
    logits = torch.tensor([5.0, -5.0])
    labels = torch.tensor([1.0, 0.0])
    high_correct = float(link_loss(logits, labels))
    low_correct = float(link_loss(-logits, labels))
    assert high_correct < low_correct


def test_save_load_roundtrip(tmp_path: Path):
    net = init_brain_net(input_dim=8, latent_dim=4, seed=0)
    p = tmp_path / "brain.pt"
    save_checkpoint(net, p)
    loaded = load_checkpoint(p)
    assert loaded is not None
    x = torch.randn(2, 8)
    out_a = net(x)
    out_b = loaded(x)
    assert torch.allclose(out_a["latent"], out_b["latent"])


def test_load_missing_returns_none(tmp_path: Path):
    assert load_checkpoint(tmp_path / "missing.pt") is None


def test_project_to_xy_numpy_io():
    net = init_brain_net(input_dim=8, seed=0)
    arr = np.random.RandomState(0).randn(5, 8).astype(np.float32)
    xy = project_to_xy(net, arr)
    assert xy.shape == (5, 2)
    assert ((xy >= -1.0) & (xy <= 1.0)).all()


def test_edge_probabilities_diagonal_zero():
    net = init_brain_net(input_dim=8, seed=0)
    arr = np.random.RandomState(0).randn(4, 8).astype(np.float32)
    probs = edge_probabilities(net, arr)
    assert probs.shape == (4, 4)
    assert (np.diag(probs) == 0.0).all()


def test_edge_probabilities_empty():
    net = init_brain_net(input_dim=8, seed=0)
    probs = edge_probabilities(net, np.zeros((0, 8), dtype=np.float32))
    assert probs.shape == (0, 0)


def test_init_seeded_is_deterministic():
    a = init_brain_net(input_dim=8, seed=123)
    b = init_brain_net(input_dim=8, seed=123)
    for pa, pb in zip(a.parameters(), b.parameters()):
        assert torch.allclose(pa, pb)
