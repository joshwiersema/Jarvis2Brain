"""End-to-end trainer tests against a tiny vault."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from brain.embed import HashEmbedder
from brain.index import MemoryIndex
from brain.loops.train_brain import BrainTrainer, TrainerConfig
from brain.memory import Memory
from brain.note import Note, now_utc
from brain.vault import Vault


def _make(slug: str, body: str = "") -> Note:
    ts = now_utc()
    return Note(slug=slug, title=slug, created=ts, updated=ts, body=body)


@pytest.fixture
def memory(tmp_path: Path) -> Memory:
    v = Vault(tmp_path)
    idx = MemoryIndex(HashEmbedder(dim=64))
    return Memory(v, idx)


def _seed_vault(memory: Memory) -> None:
    memory.vault.create(_make("a", body="alpha [[b]]"))
    memory.vault.create(_make("b", body="beta [[c]]"))
    memory.vault.create(_make("c", body="gamma"))
    memory.vault.create(_make("z", body="unrelated topic"))
    memory.reindex_from_vault()


def test_trainer_initializes_and_saves(tmp_path: Path, memory: Memory):
    _seed_vault(memory)
    t = BrainTrainer(memory, config=TrainerConfig(seed=0))
    t.save()
    assert (memory.vault.path / ".brain" / "models" / "brain_nn.pt").exists()


def test_train_full_runs_and_writes_history(memory: Memory):
    _seed_vault(memory)
    t = BrainTrainer(memory, config=TrainerConfig(seed=0))
    summaries = t.train_full(epochs=2)
    assert len(summaries) == 2
    hist = memory.vault.path / ".brain" / "training_history.jsonl"
    assert hist.exists()
    lines = hist.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) > 0
    parsed = [json.loads(line) for line in lines]
    assert all("recon" in p for p in parsed)


def test_loss_decreases_across_epochs(memory: Memory):
    _seed_vault(memory)
    t = BrainTrainer(memory, config=TrainerConfig(seed=0, lr=1e-2))
    s = t.train_full(epochs=4)
    assert s[-1].avg_recon <= s[0].avg_recon + 1e-3


def test_xy_written_back_to_notes(memory: Memory):
    _seed_vault(memory)
    pre_xy = memory.vault.read("a").xy
    t = BrainTrainer(memory, config=TrainerConfig(seed=0, lr=1e-2))
    t.train_full(epochs=2)
    post_xy = memory.vault.read("a").xy
    assert post_xy is not None
    # Coordinate should have moved (random init -> learned position).
    assert post_xy != pre_xy


def test_edge_proposals_written(memory: Memory):
    _seed_vault(memory)
    t = BrainTrainer(
        memory,
        config=TrainerConfig(seed=0, edge_proposal_threshold=0.0),
    )
    t.train_full(epochs=1)
    proposals_path = memory.vault.path / ".brain" / "edge_proposals.jsonl"
    assert proposals_path.exists()


def test_train_step_is_a_noop_on_empty(tmp_path: Path):
    v = Vault(tmp_path)
    idx = MemoryIndex(HashEmbedder(dim=32))
    m = Memory(v, idx)
    t = BrainTrainer(m, config=TrainerConfig(seed=0))
    assert t.train_step("nope") is None


def test_train_step_updates_after_new_note(memory: Memory):
    _seed_vault(memory)
    t = BrainTrainer(memory, config=TrainerConfig(seed=0))
    loss = t.train_step("a")
    assert loss is not None
    assert loss >= 0.0
