"""Auto-classify the chaos-start inbox tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from brain.embed import HashEmbedder
from brain.index import MemoryIndex
from brain.loops.auto_classify import auto_classify_inbox
from brain.memory import Memory
from brain.note import Note, now_utc
from brain.vault import Vault


def _make(slug: str, body: str, kind: str = "unsorted") -> Note:
    ts = now_utc()
    return Note(slug=slug, title=slug, created=ts, updated=ts, body=body, kind=kind)


@pytest.fixture
def memory(tmp_path: Path) -> Memory:
    v = Vault(tmp_path)
    idx = MemoryIndex(HashEmbedder(dim=128))
    return Memory(v, idx)


def test_no_centroids_returns_empty(memory: Memory):
    memory.vault.create(_make("u", "alpha", kind="unsorted"))
    memory.reindex_from_vault()
    assert auto_classify_inbox(memory) == []


def test_promotes_to_nearest_kind(memory: Memory):
    # Seed a 'fact' cluster.
    memory.vault.create(_make("f1", "pizza dough recipe", kind="fact"), subdir="semantic/notes")
    memory.vault.create(_make("f2", "pizza oven temperature", kind="fact"), subdir="semantic/notes")
    # Seed a 'skill' cluster, far from pizza.
    memory.vault.create(_make("s1", "calculus integration", kind="skill"), subdir="procedural/skills")
    memory.vault.create(_make("s2", "calculus derivatives", kind="skill"), subdir="procedural/skills")
    # Unsorted inbox note close to fact cluster.
    memory.vault.create(_make("u1", "pizza dough kneading", kind="unsorted"))
    memory.reindex_from_vault()

    results = auto_classify_inbox(memory, threshold=0.0, margin=0.0)
    assert any(r.slug == "u1" and r.chosen_kind == "fact" for r in results)


def test_low_confidence_leaves_in_inbox(memory: Memory):
    memory.vault.create(_make("f1", "alpha beta", kind="fact"), subdir="semantic/notes")
    memory.vault.create(_make("u1", "completely different topic", kind="unsorted"))
    memory.reindex_from_vault()
    results = auto_classify_inbox(memory, threshold=0.99, margin=0.0)
    assert results == []
    # Note still kind=unsorted in inbox.
    assert memory.vault.read("u1").kind == "unsorted"


def test_apply_false_does_not_move(memory: Memory):
    memory.vault.create(_make("f1", "alpha", kind="fact"), subdir="semantic/notes")
    memory.vault.create(_make("u1", "alpha similar", kind="unsorted"))
    memory.reindex_from_vault()
    results = auto_classify_inbox(memory, apply=False, threshold=0.0, margin=0.0)
    if results:
        # Returned a decision but did not actually relocate.
        assert memory.vault.read("u1").kind == "unsorted"
        assert (memory.vault.path / "episodic" / "inbox" / "u1.md").exists()
