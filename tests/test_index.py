"""MemoryIndex hybrid retrieval + persistence tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from brain.embed import HashEmbedder
from brain.index import IndexedDoc, MemoryIndex


@pytest.fixture
def idx() -> MemoryIndex:
    return MemoryIndex(HashEmbedder(dim=128))


def _doc(slug: str, text: str, **kw) -> IndexedDoc:
    return IndexedDoc(slug=slug, text=text, **kw)


def test_upsert_and_search_substring(idx: MemoryIndex):
    idx.upsert(_doc("pizza", "pizza dough recipe"))
    idx.upsert(_doc("brain", "neural network architecture"))
    res = idx.search("pizza", mode="substring")
    assert res[0].slug == "pizza"


def test_search_semantic_finds_overlap(idx: MemoryIndex):
    idx.upsert(_doc("pizza", "pizza dough kneading"))
    idx.upsert(_doc("calc", "integral calculus theorems"))
    # 'dough' overlaps with pizza doc tokens.
    res = idx.search("dough", mode="semantic")
    assert res[0].slug == "pizza"


def test_search_hybrid_fuses_modes(idx: MemoryIndex):
    idx.upsert(_doc("a", "alpha foo"))
    idx.upsert(_doc("b", "beta bar"))
    idx.upsert(_doc("c", "gamma baz"))
    res = idx.search("alpha", mode="hybrid")
    assert res[0].slug == "a"
    # Hybrid scores include both bm25 and vector source contributions.
    assert "bm25" in res[0].sources or "vector" in res[0].sources


def test_search_filters_by_kind(idx: MemoryIndex):
    idx.upsert(_doc("a", "shared word", kind="note"))
    idx.upsert(_doc("b", "shared word", kind="skill"))
    res = idx.search("shared", kind="skill")
    assert {r.slug for r in res} == {"b"}


def test_delete_removes_doc(idx: MemoryIndex):
    idx.upsert(_doc("a", "alpha"))
    idx.upsert(_doc("b", "beta"))
    idx.delete("a")
    res = idx.search("alpha", mode="hybrid")
    assert all(r.slug != "a" for r in res)


def test_upsert_replaces_existing(idx: MemoryIndex):
    idx.upsert(_doc("a", "first version"))
    idx.upsert(_doc("a", "completely different content"))
    res = idx.search("first", mode="substring")
    assert all(r.slug != "a" or r.score == 0 for r in res)


def test_persistence_roundtrip(tmp_path: Path):
    e = HashEmbedder(dim=128)
    a = MemoryIndex(e, path=tmp_path / "index.npz")
    a.upsert(_doc("alpha", "alpha foo"))
    a.upsert(_doc("beta", "beta bar"))
    a.save()
    b = MemoryIndex.load(e, tmp_path / "index.npz")
    assert set(b.all_slugs()) == {"alpha", "beta"}
    res = b.search("alpha", mode="hybrid")
    assert res[0].slug == "alpha"


def test_load_missing_returns_empty(tmp_path: Path):
    e = HashEmbedder(dim=64)
    b = MemoryIndex.load(e, tmp_path / "missing.npz")
    assert b.all_slugs() == []


def test_vectors_matrix_shape(idx: MemoryIndex):
    idx.upsert(_doc("a", "x"))
    idx.upsert(_doc("b", "y"))
    slugs, mat = idx.vectors_matrix()
    assert sorted(slugs) == ["a", "b"]
    assert mat.shape == (2, 128)


def test_empty_query_returns_recent(idx: MemoryIndex):
    idx.upsert(_doc("old", "x", updated_ts=1.0))
    idx.upsert(_doc("new", "y", updated_ts=100.0))
    res = idx.search("", k=2)
    assert res[0].slug == "new"
