"""Embedder tests."""

from __future__ import annotations

import numpy as np

from brain.embed import HashEmbedder, cosine, get_embedder, text_for_embedding


def test_hash_embedder_dimensions():
    e = HashEmbedder(dim=64)
    v = e.embed_texts(["hello world"])
    assert v.shape == (1, 64)
    assert v.dtype == np.float32


def test_hash_embedder_normalized():
    e = HashEmbedder(dim=128)
    v = e.embed_texts(["alpha", "beta gamma"])
    norms = np.linalg.norm(v, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_hash_embedder_deterministic():
    e1 = HashEmbedder(dim=64)
    e2 = HashEmbedder(dim=64)
    a = e1.embed_texts(["jarvis brain"])
    b = e2.embed_texts(["jarvis brain"])
    assert np.allclose(a, b)


def test_hash_embedder_different_texts_diverge():
    e = HashEmbedder(dim=128)
    v = e.embed_texts(["pizza", "calculus"])
    # Cosine between unrelated tokens should be < 1.
    assert v @ v.T[:, 0:1] is not None
    sim = float(v[0] @ v[1])
    assert sim < 0.99


def test_hash_embedder_token_overlap_increases_similarity():
    e = HashEmbedder(dim=256)
    v = e.embed_texts(
        [
            "pizza dough recipe",
            "pizza dough kneading",
            "completely unrelated topic",
        ]
    )
    sim_related = float(v[0] @ v[1])
    sim_unrelated = float(v[0] @ v[2])
    assert sim_related > sim_unrelated


def test_get_embedder_default_is_hash():
    assert isinstance(get_embedder(), HashEmbedder)


def test_get_embedder_unknown_raises():
    import pytest

    with pytest.raises(ValueError):
        get_embedder("nope")


def test_cosine_helper():
    a = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    b = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    sims = cosine(a, b)
    assert sims.shape == (1, 2)
    assert sims[0, 0] == 1.0
    assert sims[0, 1] == 0.0


def test_text_for_embedding_combines_fields():
    text = text_for_embedding("Title", "body content", ["t1", "t2"])
    assert "Title" in text and "body content" in text and "t1" in text
