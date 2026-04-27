"""Cross-encoder rerankers. Local default = identity; opt into bge / voyage."""

from __future__ import annotations

import os
from typing import Optional, Protocol, Sequence


class Reranker(Protocol):
    def rerank(self, query: str, docs: Sequence[tuple[str, str]], top_k: int = 10) -> list[tuple[str, float]]:
        """Rerank (slug, text) pairs against query. Returns (slug, score), high-first."""


class IdentityReranker:
    """Pass-through. Preserves caller's input order, copies the index score."""

    def rerank(self, query: str, docs: Sequence[tuple[str, str]], top_k: int = 10) -> list[tuple[str, float]]:
        # Score by (uniform 1.0 - position) so caller's order is the truth.
        return [(slug, 1.0 - i * 1e-6) for i, (slug, _) in enumerate(list(docs)[:top_k])]


class LocalReranker:
    """Lazy wrapper around bge-reranker-v2-m3 cross-encoder."""

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3") -> None:
        self.model_name = model_name
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(self, query: str, docs: Sequence[tuple[str, str]], top_k: int = 10) -> list[tuple[str, float]]:
        if not docs:
            return []
        m = self._load()
        pairs = [(query, text) for _, text in docs]
        scores = m.predict(pairs)
        scored = [(slug, float(s)) for (slug, _), s in zip(docs, scores)]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]


class VoyageReranker:
    def __init__(self, api_key: Optional[str] = None, model: str = "rerank-2.5") -> None:
        self.api_key = api_key or os.environ.get("VOYAGE_API_KEY")
        self.model = model

    def rerank(self, query: str, docs: Sequence[tuple[str, str]], top_k: int = 10) -> list[tuple[str, float]]:
        if not self.api_key:
            raise RuntimeError("VOYAGE_API_KEY not set")
        if not docs:
            return []
        import httpx

        r = httpx.post(
            "https://api.voyageai.com/v1/rerank",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model,
                "query": query,
                "documents": [text for _, text in docs],
                "top_k": top_k,
            },
            timeout=30.0,
        )
        r.raise_for_status()
        results = r.json()["data"]
        slugs = [slug for slug, _ in docs]
        return [(slugs[res["index"]], float(res["relevance_score"])) for res in results]


def get_reranker(provider: Optional[str] = None) -> Reranker:
    provider = (provider or os.environ.get("BRAIN_RERANK_PROVIDER") or "identity").lower()
    if provider == "identity":
        return IdentityReranker()
    if provider == "local":
        return LocalReranker()
    if provider == "voyage":
        return VoyageReranker()
    raise ValueError(f"unknown rerank provider: {provider}")
