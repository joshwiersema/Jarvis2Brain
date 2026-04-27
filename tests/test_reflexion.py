"""Reflexion tests — refuses no-artifact reflections."""

from __future__ import annotations

from pathlib import Path

import pytest

from brain.embed import HashEmbedder
from brain.index import MemoryIndex
from brain.loops.reflexion import FailureArtifact, record_reflection, retrieve_relevant
from brain.memory import Memory
from brain.vault import Vault


@pytest.fixture
def memory(tmp_path: Path) -> Memory:
    v = Vault(tmp_path)
    return Memory(v, MemoryIndex(HashEmbedder(dim=64)))


def test_no_artifact_returns_none(memory: Memory):
    res = record_reflection(
        memory.vault,
        task_summary="task",
        artifact=FailureArtifact(kind="test_output", content=""),
        lesson="dont vibe",
    )
    assert res is None


def test_no_lesson_returns_none(memory: Memory):
    res = record_reflection(
        memory.vault,
        task_summary="task",
        artifact=FailureArtifact(kind="diff", content="some diff"),
        lesson="",
    )
    assert res is None


def test_writes_kind_reflection(memory: Memory):
    n = record_reflection(
        memory.vault,
        task_summary="parse markdown",
        artifact=FailureArtifact(kind="test_output", content="AssertionError: line 5"),
        lesson="Always strip BOM before parse",
    )
    assert n is not None
    assert n.kind == "reflection"
    on_disk = memory.vault.read(n.slug)
    assert "Always strip BOM" in on_disk.body
    assert "AssertionError" in on_disk.body


def test_retrieve_finds_recent_reflections(memory: Memory):
    record_reflection(
        memory.vault,
        task_summary="parsing yaml frontmatter",
        artifact=FailureArtifact(kind="test_output", content="bad indent"),
        lesson="yaml is whitespace-sensitive",
    )
    memory.reindex_from_vault()
    hits = retrieve_relevant(memory, "yaml frontmatter parsing")
    assert any("yaml" in h.preview.lower() or "yaml" in h.title.lower() for h in hits)
