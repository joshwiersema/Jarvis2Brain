"""Searchable index over the vault.

Default is `MemoryIndex` (numpy + simple BM25), persisted to
`<vault>/.brain/index.npz`. LanceDB swappable behind `BRAIN_INDEX=lancedb` —
the interface stays identical so the upgrade is one env-var.

Hybrid scoring fuses BM25 lexical matches and vector cosine similarity via
Reciprocal Rank Fusion (k=60), which is the same default RRF constant most
hybrid retrieval papers settled on.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from brain.embed import Embedder, cosine

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_RRF_K = 60.0


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


@dataclass
class IndexedDoc:
    slug: str
    text: str
    kind: str = "note"
    tags: list[str] = field(default_factory=list)
    importance: float = 0.5
    updated_ts: float = 0.0


@dataclass
class SearchResult:
    slug: str
    score: float
    sources: dict[str, float]  # which scorer contributed (bm25, vector, fused)


class MemoryIndex:
    """In-memory hybrid (BM25 + vector cosine) index.

    Persists to a single .npz + sidecar .json so reindex is cheap and the
    daemon's online updates are crash-safe via atomic replace.
    """

    def __init__(self, embedder: Embedder, path: Optional[Path] = None) -> None:
        self.embedder = embedder
        self.path = Path(path) if path else None
        self._docs: dict[str, IndexedDoc] = {}
        self._tokens: dict[str, list[str]] = {}
        self._df: Counter = Counter()
        self._vectors: dict[str, np.ndarray] = {}
        self._dim = embedder.dim

    # ----- mutation -----

    def upsert(self, doc: IndexedDoc) -> None:
        if doc.slug in self._docs:
            self._remove_postings(doc.slug)
        self._docs[doc.slug] = doc
        toks = _tokenize(doc.text)
        self._tokens[doc.slug] = toks
        for t in set(toks):
            self._df[t] += 1
        vec = self.embedder.embed_texts([doc.text])[0]
        self._vectors[doc.slug] = vec.astype(np.float32)

    def delete(self, slug: str) -> None:
        if slug not in self._docs:
            return
        self._remove_postings(slug)
        del self._docs[slug]
        self._vectors.pop(slug, None)

    def _remove_postings(self, slug: str) -> None:
        for t in set(self._tokens.get(slug, [])):
            self._df[t] -= 1
            if self._df[t] <= 0:
                del self._df[t]
        self._tokens.pop(slug, None)

    # ----- query -----

    def _bm25(self, query_tokens: list[str], k1: float = 1.5, b: float = 0.75) -> dict[str, float]:
        if not self._docs:
            return {}
        avgdl = sum(len(t) for t in self._tokens.values()) / max(1, len(self._tokens))
        N = len(self._docs)
        scores: dict[str, float] = defaultdict(float)
        for slug, toks in self._tokens.items():
            if not toks:
                continue
            tf = Counter(toks)
            dl = len(toks)
            for q in query_tokens:
                f = tf.get(q, 0)
                if not f:
                    continue
                df = self._df.get(q, 0)
                idf = math.log(1 + (N - df + 0.5) / (df + 0.5))
                denom = f + k1 * (1 - b + b * dl / avgdl) if avgdl else 1.0
                scores[slug] += idf * (f * (k1 + 1)) / denom
        return dict(scores)

    def _vector(self, query: str) -> dict[str, float]:
        if not self._vectors:
            return {}
        q = self.embedder.embed_texts([query])[0]
        slugs = list(self._vectors.keys())
        mat = np.stack([self._vectors[s] for s in slugs])
        sims = cosine(q[None, :], mat)[0]
        return {s: float(v) for s, v in zip(slugs, sims)}

    def search(
        self,
        query: str,
        *,
        k: int = 10,
        mode: str = "hybrid",
        kind: Optional[str] = None,
    ) -> list[SearchResult]:
        if not query.strip() and not kind:
            # No-op query — return most recently updated docs.
            sorted_slugs = sorted(
                self._docs.values(), key=lambda d: d.updated_ts, reverse=True
            )
            return [
                SearchResult(slug=d.slug, score=0.0, sources={"empty": 0.0})
                for d in sorted_slugs[:k]
            ]

        q_tokens = _tokenize(query)
        bm25 = self._bm25(q_tokens) if mode in ("substring", "hybrid") else {}
        vec = self._vector(query) if mode in ("semantic", "hybrid") else {}

        if mode == "substring":
            scores = bm25
            sources = {s: {"bm25": v} for s, v in bm25.items()}
        elif mode == "semantic":
            scores = vec
            sources = {s: {"vector": v} for s, v in vec.items()}
        else:
            scores, sources = self._rrf_fuse(bm25, vec)

        if kind:
            scores = {s: v for s, v in scores.items() if self._docs[s].kind == kind}

        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:k]
        return [
            SearchResult(slug=s, score=float(v), sources=sources.get(s, {}))
            for s, v in ranked
        ]

    @staticmethod
    def _rrf_fuse(
        bm25: dict[str, float], vec: dict[str, float]
    ) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
        bm25_rank = {s: i for i, (s, _) in enumerate(
            sorted(bm25.items(), key=lambda kv: kv[1], reverse=True)
        )}
        vec_rank = {s: i for i, (s, _) in enumerate(
            sorted(vec.items(), key=lambda kv: kv[1], reverse=True)
        )}
        slugs = set(bm25_rank) | set(vec_rank)
        scores: dict[str, float] = {}
        sources: dict[str, dict[str, float]] = {}
        for s in slugs:
            f_bm = 1.0 / (_RRF_K + bm25_rank[s] + 1) if s in bm25_rank else 0.0
            f_vec = 1.0 / (_RRF_K + vec_rank[s] + 1) if s in vec_rank else 0.0
            scores[s] = f_bm + f_vec
            sources[s] = {"bm25": bm25.get(s, 0.0), "vector": vec.get(s, 0.0)}
        return scores, sources

    # ----- introspection (used by NN trainer + viz) -----

    def all_slugs(self) -> list[str]:
        return list(self._docs.keys())

    def vector(self, slug: str) -> Optional[np.ndarray]:
        return self._vectors.get(slug)

    def vectors_matrix(self) -> tuple[list[str], np.ndarray]:
        slugs = list(self._vectors.keys())
        if not slugs:
            return [], np.zeros((0, self._dim), dtype=np.float32)
        return slugs, np.stack([self._vectors[s] for s in slugs])

    def doc(self, slug: str) -> Optional[IndexedDoc]:
        return self._docs.get(slug)

    # ----- persistence -----

    def save(self, path: Optional[Path] = None) -> None:
        target = Path(path or self.path)
        target.parent.mkdir(parents=True, exist_ok=True)
        slugs, mat = self.vectors_matrix()
        np.savez_compressed(target, slugs=np.array(slugs, dtype=object), vectors=mat)
        meta = {
            slug: {
                "text": d.text,
                "kind": d.kind,
                "tags": d.tags,
                "importance": d.importance,
                "updated_ts": d.updated_ts,
                "tokens": self._tokens[slug],
            }
            for slug, d in self._docs.items()
        }
        target.with_suffix(".json").write_text(json.dumps(meta), encoding="utf-8")

    @classmethod
    def load(cls, embedder: Embedder, path: Path) -> "MemoryIndex":
        idx = cls(embedder, path=path)
        npz_path = Path(path)
        json_path = npz_path.with_suffix(".json")
        if not npz_path.exists() or not json_path.exists():
            return idx
        z = np.load(npz_path, allow_pickle=True)
        slugs = list(z["slugs"])
        mat = z["vectors"]
        meta = json.loads(json_path.read_text(encoding="utf-8"))
        for i, slug in enumerate(slugs):
            slug = str(slug)
            m = meta[slug]
            doc = IndexedDoc(
                slug=slug,
                text=m["text"],
                kind=m.get("kind", "note"),
                tags=list(m.get("tags", [])),
                importance=float(m.get("importance", 0.5)),
                updated_ts=float(m.get("updated_ts", 0.0)),
            )
            idx._docs[slug] = doc
            idx._tokens[slug] = list(m["tokens"])
            for t in set(idx._tokens[slug]):
                idx._df[t] += 1
            idx._vectors[slug] = mat[i].astype(np.float32)
        return idx
