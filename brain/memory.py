"""Tiered memory facade.

Three retrieval modes:

- `recall(query)` — hot recent memory (last `time_window`)
- `archival(query)` — cold full-vault search
- `retrieve(query)` — composite: relevance × recency × importance, with
  1-hop graph expansion through wikilinks, then optional reranker.

Composite scoring borrows the Generative Agents formula (recency decay +
importance + relevance), which gives newer + more-frequently-touched +
higher-importance notes priority without losing surprising old matches.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Optional

from brain.embed import text_for_embedding
from brain.index import IndexedDoc, MemoryIndex, SearchResult
from brain.rerank import IdentityReranker, Reranker
from brain.vault import Vault


@dataclass
class MemoryHit:
    slug: str
    title: str
    score: float
    components: dict[str, float]
    kind: str
    preview: str


class Memory:
    def __init__(
        self,
        vault: Vault,
        index: MemoryIndex,
        reranker: Optional[Reranker] = None,
    ) -> None:
        self.vault = vault
        self.index = index
        self.reranker = reranker or IdentityReranker()

    def reindex_from_vault(self) -> int:
        """Rebuild the index from disk. Returns number of docs indexed."""
        n = 0
        existing = set(self.index.all_slugs())
        seen: set[str] = set()
        for s in self.vault.list():
            note = self.vault.read(s.slug)
            self.index.upsert(
                IndexedDoc(
                    slug=note.slug,
                    text=text_for_embedding(note.title, note.body, note.tags),
                    kind=note.kind,
                    tags=list(note.tags),
                    importance=note.importance,
                    updated_ts=note.updated.timestamp(),
                )
            )
            seen.add(note.slug)
            n += 1
        for stale in existing - seen:
            self.index.delete(stale)
        return n

    def recall(
        self,
        query: str,
        *,
        k: int = 10,
        time_window_days: float = 7.0,
        kind: Optional[str] = None,
    ) -> list[MemoryHit]:
        cutoff = time.time() - time_window_days * 86400.0
        results = self.index.search(query, k=k * 4, mode="hybrid", kind=kind)
        hot = [r for r in results if self._doc(r.slug).updated_ts >= cutoff]
        return self._materialize(query, hot, k)

    def archival(
        self,
        query: str,
        *,
        k: int = 20,
        kind: Optional[str] = None,
    ) -> list[MemoryHit]:
        results = self.index.search(query, k=k * 2, mode="hybrid", kind=kind)
        return self._materialize(query, results, k)

    def retrieve(self, query: str, *, k: int = 10) -> list[MemoryHit]:
        # Wider net so reranker has 3x candidates to reorder.
        results = self.index.search(query, k=k * 5, mode="hybrid")
        composite: list[SearchResult] = []
        now = time.time()
        for r in results:
            d = self._doc(r.slug)
            recency = math.exp(-max(0.0, now - d.updated_ts) / (30.0 * 86400.0))
            importance = max(0.0, min(1.0, d.importance))
            composite.append(
                SearchResult(
                    slug=r.slug,
                    score=r.score * (0.5 + 0.5 * recency) * (0.5 + 0.5 * importance),
                    sources={
                        **r.sources,
                        "recency": recency,
                        "importance": importance,
                    },
                )
            )

        # 1-hop graph expansion: bring in linked neighbours not already in set.
        seen = {c.slug for c in composite}
        for c in list(composite):
            note = self.vault.read(c.slug)
            for nbr in note.links_out:
                if nbr in seen or self.index.doc(nbr) is None:
                    continue
                composite.append(
                    SearchResult(
                        slug=nbr,
                        score=c.score * 0.6,
                        sources={"graph_expand_from": c.score},
                    )
                )
                seen.add(nbr)

        composite.sort(key=lambda r: r.score, reverse=True)
        return self._materialize(query, composite[: k * 2], k)

    def _doc(self, slug: str) -> IndexedDoc:
        d = self.index.doc(slug)
        if d is None:
            raise KeyError(f"slug not in index: {slug}")
        return d

    def _materialize(self, query: str, results: list[SearchResult], k: int) -> list[MemoryHit]:
        # Optional rerank step over top-50.
        candidates = results[:50]
        pairs = [(r.slug, self._doc(r.slug).text) for r in candidates]
        reranked = self.reranker.rerank(query, pairs, top_k=k)

        score_by_slug = {r.slug: r for r in candidates}
        out: list[MemoryHit] = []
        for slug, score in reranked:
            base = score_by_slug.get(slug)
            note = self.vault.read(slug)
            out.append(
                MemoryHit(
                    slug=slug,
                    title=note.title,
                    score=score,
                    components=base.sources if base else {},
                    kind=note.kind,
                    preview=note.body[:200],
                )
            )
        return out
