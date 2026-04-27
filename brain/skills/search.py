"""Search + memory recall skills."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from brain.embed import get_embedder
from brain.index import MemoryIndex
from brain.memory import Memory
from brain.rerank import get_reranker
from brain.skill import skill
from brain.vault import Vault, resolve_vault_path


def _memory() -> Memory:
    vault_path = resolve_vault_path(cli=os.environ.get("BRAIN_VAULT"), cwd=Path.cwd())
    v = Vault(vault_path)
    embedder = get_embedder()
    idx_path = vault_path / ".brain" / "index.npz"
    index = MemoryIndex.load(embedder, idx_path)
    if not index.all_slugs():
        index = MemoryIndex(embedder, path=idx_path)
    m = Memory(v, index, reranker=get_reranker())
    if not m.index.all_slugs():
        m.reindex_from_vault()
    return m


def _hit(h) -> dict:
    return {
        "slug": h.slug,
        "title": h.title,
        "score": h.score,
        "kind": h.kind,
        "preview": h.preview,
        "components": h.components,
    }


@skill(name="memory_search", description="Hybrid (BM25+vector) search across the vault")
def memory_search(
    query: str,
    k: int = 10,
    kind: Optional[str] = None,
) -> list[dict]:
    return [_hit(h) for h in _memory().archival(query, k=k, kind=kind)]


@skill(name="memory_recall", description="Recall recent matching notes (hot tier, last 7d)")
def memory_recall(
    query: str,
    k: int = 10,
    time_window_days: float = 7.0,
    kind: Optional[str] = None,
) -> list[dict]:
    return [
        _hit(h)
        for h in _memory().recall(
            query, k=k, time_window_days=time_window_days, kind=kind
        )
    ]


@skill(
    name="memory_retrieve",
    description="Composite recency*importance*relevance retrieve with graph expansion",
)
def memory_retrieve(query: str, k: int = 10) -> list[dict]:
    return [_hit(h) for h in _memory().retrieve(query, k=k)]
