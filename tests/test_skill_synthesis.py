"""Skill synthesis tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from brain.embed import HashEmbedder
from brain.index import MemoryIndex
from brain.loops.skill_synthesis import propose_skills, write_proposals
from brain.memory import Memory
from brain.note import Note, now_utc
from brain.vault import Vault


def _trace(slug: str, body: str) -> Note:
    ts = now_utc()
    return Note(slug=slug, title=slug, created=ts, updated=ts, body=body, kind="trace")


@pytest.fixture
def memory(tmp_path: Path) -> Memory:
    v = Vault(tmp_path)
    return Memory(v, MemoryIndex(HashEmbedder(dim=128)))


def test_no_proposals_when_under_min_cluster(memory: Memory):
    memory.vault.create(_trace("a", "git status check"), subdir="episodic")
    memory.vault.create(_trace("b", "git status check"), subdir="episodic")
    memory.reindex_from_vault()
    out = propose_skills(memory, min_cluster_size=3)
    assert out == []


def test_proposes_for_a_tight_cluster(memory: Memory):
    for i in range(4):
        memory.vault.create(
            _trace(f"git-{i}", "git status check repository state"),
            subdir="episodic",
        )
    memory.reindex_from_vault()
    out = propose_skills(memory, min_cluster_size=3, threshold=0.5)
    assert len(out) >= 1
    assert out[0].confidence > 0


def test_write_proposals_creates_files(memory: Memory):
    for i in range(3):
        memory.vault.create(
            _trace(f"t-{i}", "checkout branch run tests inspect output"),
            subdir="episodic",
        )
    memory.reindex_from_vault()
    proposals = propose_skills(memory, min_cluster_size=3, threshold=0.5)
    n = write_proposals(memory, proposals)
    assert n >= 1
    files = list((memory.vault.path / "proposals").glob("*.md"))
    assert files
