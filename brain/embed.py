"""Embedding providers.

Default `HashEmbedder` is deterministic, dependency-free, and good enough for
local dev + tests. Set `BRAIN_EMBED_PROVIDER=local` for `BAAI/bge-m3`
(downloads ~2GB to <vault>/.brain/models/) or `voyage` for the cloud option.
The chosen provider is also what trains the brain NN — embedding quality is
upgrade-able without reindexing the whole vault (just run `brain reindex`).
"""

from __future__ import annotations

import hashlib
import os
import struct
from pathlib import Path
from typing import Optional, Protocol, Sequence

import numpy as np

DEFAULT_DIM = 128


class Embedder(Protocol):
    dim: int

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:  # noqa: D401
        ...


def _normalize(vecs: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return vecs / norms


class HashEmbedder:
    """Deterministic SHA-256-based embedding. Trades quality for zero deps.

    Each token contributes a fixed pseudo-random vector; final vector is the
    L2-normalized sum. Reproducible across machines, useful for testing the
    pipeline + bootstrapping the NN before a real embedder is wired up.
    """

    def __init__(self, dim: int = DEFAULT_DIM) -> None:
        self.dim = dim

    def _token_vec(self, token: str) -> np.ndarray:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        # Repeat digest until we have enough bytes for `dim` float32s.
        needed = self.dim * 4
        repeated = (digest * ((needed // len(digest)) + 1))[:needed]
        ints = np.frombuffer(repeated, dtype=np.uint32)
        floats = (ints.astype(np.float64) / 0xFFFFFFFF) * 2.0 - 1.0
        return floats[: self.dim].astype(np.float32)

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            tokens = [tok for tok in t.lower().split() if tok]
            if not tokens:
                out[i] = self._token_vec(t or "_empty_")
                continue
            for tok in tokens:
                out[i] += self._token_vec(tok)
        return _normalize(out)


class LocalEmbedder:
    """Lazy wrapper around sentence-transformers bge-m3."""

    def __init__(self, model_name: str = "BAAI/bge-m3", cache_dir: Optional[Path] = None) -> None:
        self.model_name = model_name
        self.cache_dir = cache_dir
        self._model = None
        self.dim = 1024  # bge-m3 dense dim

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(
                self.model_name,
                cache_folder=str(self.cache_dir) if self.cache_dir else None,
            )
            self.dim = self._model.get_sentence_embedding_dimension()
        return self._model

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        m = self._load()
        vecs = m.encode(list(texts), normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(vecs, dtype=np.float32)


class VoyageEmbedder:
    """Lazy wrapper around the Voyage AI HTTP API."""

    def __init__(self, api_key: Optional[str] = None, model: str = "voyage-3-large") -> None:
        self.api_key = api_key or os.environ.get("VOYAGE_API_KEY")
        self.model = model
        self.dim = 1024

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        if not self.api_key:
            raise RuntimeError("VOYAGE_API_KEY not set")
        import httpx

        r = httpx.post(
            "https://api.voyageai.com/v1/embeddings",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model,
                "input": list(texts),
                "output_dimension": self.dim,
            },
            timeout=30.0,
        )
        r.raise_for_status()
        data = r.json()["data"]
        return _normalize(
            np.asarray([d["embedding"] for d in data], dtype=np.float32)
        )


def get_embedder(provider: Optional[str] = None, **kw) -> Embedder:
    """Resolve provider from arg > BRAIN_EMBED_PROVIDER env > 'hash'."""
    provider = (provider or os.environ.get("BRAIN_EMBED_PROVIDER") or "hash").lower()
    if provider == "hash":
        return HashEmbedder(dim=int(kw.get("dim", DEFAULT_DIM)))
    if provider == "local":
        return LocalEmbedder(**kw)
    if provider == "voyage":
        return VoyageEmbedder(**kw)
    raise ValueError(f"unknown embed provider: {provider}")


def cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine similarity between unit-norm `a` (n,d) and `b` (m,d) -> (n,m)."""
    return a @ b.T


def text_for_embedding(title: str, body: str, tags: Sequence[str]) -> str:
    parts = [title, " ".join(tags), body]
    return "\n".join(p for p in parts if p)


_C = struct.Struct("<Q")  # exposed for callers that want to seed hashes


def stable_token_seed(s: str) -> int:
    """Stable 64-bit seed from a string. Useful for deterministic NN init."""
    return _C.unpack(hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest())[0]
