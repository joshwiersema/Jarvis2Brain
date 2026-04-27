"""Tiered memory facade tests."""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from brain.embed import HashEmbedder
from brain.index import MemoryIndex
from brain.memory import Memory
from brain.note import Note, now_utc
from brain.vault import Vault


def _make(slug: str, **kw) -> Note:
    ts = kw.pop("ts", now_utc())
    return Note(
        slug=slug,
        title=kw.pop("title", slug),
        created=ts,
        updated=ts,
        body=kw.pop("body", ""),
        tags=kw.pop("tags", []),
        **kw,
    )


@pytest.fixture
def memory(tmp_path: Path) -> Memory:
    v = Vault(tmp_path)
    idx = MemoryIndex(HashEmbedder(dim=128))
    return Memory(v, idx)


def test_reindex_from_vault_pulls_all_notes(tmp_path: Path, memory: Memory):
    memory.vault.create(_make("a", body="alpha"))
    memory.vault.create(_make("b", body="beta"))
    n = memory.reindex_from_vault()
    assert n == 2
    assert set(memory.index.all_slugs()) == {"a", "b"}


def test_archival_finds_relevant(memory: Memory):
    memory.vault.create(_make("pizza", body="pizza dough recipe"))
    memory.vault.create(_make("calc", body="calculus theorem"))
    memory.reindex_from_vault()
    hits = memory.archival("pizza")
    assert hits[0].slug == "pizza"


def test_recall_filters_by_time_window(memory: Memory):
    old = now_utc() - timedelta(days=30)
    new = now_utc()
    memory.vault.create(_make("old", body="alpha", ts=old))
    memory.vault.create(_make("new", body="alpha", ts=new))
    memory.reindex_from_vault()
    hot = memory.recall("alpha", time_window_days=7.0)
    assert {h.slug for h in hot} == {"new"}


def test_retrieve_does_graph_expansion(memory: Memory):
    memory.vault.create(_make("a", body="seed [[b]]"))
    memory.vault.create(_make("b", body="related"), subdir="semantic/notes")
    memory.reindex_from_vault()
    hits = memory.retrieve("seed")
    slugs = {h.slug for h in hits}
    assert "a" in slugs and "b" in slugs


def test_retrieve_importance_boosts_score(memory: Memory):
    memory.vault.create(_make("low", body="topic", importance=0.1))
    memory.vault.create(_make("high", body="topic", importance=0.9))
    memory.reindex_from_vault()
    hits = memory.retrieve("topic")
    # Higher importance should rank first when relevance is similar.
    assert hits[0].slug == "high"
