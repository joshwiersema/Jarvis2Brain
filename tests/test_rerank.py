"""Reranker tests."""

from __future__ import annotations

import pytest

from brain.rerank import IdentityReranker, get_reranker


def test_identity_preserves_order():
    r = IdentityReranker()
    out = r.rerank("q", [("a", "txt a"), ("b", "txt b"), ("c", "txt c")], top_k=3)
    assert [s for s, _ in out] == ["a", "b", "c"]
    # Scores monotonically decrease so downstream sort is a no-op.
    scores = [s for _, s in out]
    assert scores[0] > scores[1] > scores[2]


def test_identity_respects_top_k():
    r = IdentityReranker()
    out = r.rerank("q", [("a", "x"), ("b", "y"), ("c", "z")], top_k=2)
    assert len(out) == 2


def test_identity_empty_input():
    r = IdentityReranker()
    assert r.rerank("q", [], top_k=10) == []


def test_get_reranker_default_identity():
    assert isinstance(get_reranker(), IdentityReranker)


def test_get_reranker_unknown_raises():
    with pytest.raises(ValueError):
        get_reranker("nope")
