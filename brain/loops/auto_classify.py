"""Auto-classify chaos-start inbox notes into the right tier.

New notes land in `episodic/inbox/` with `kind=unsorted`. Once the NN has
seen enough signal, this loop promotes each unsorted note to the tier whose
centroid (in latent space) it sits closest to — provided the margin to the
runner-up is wide enough to be confident.

This is the loop that turns the disorganized chaos start into a self-organized
memory. It runs as part of the nightly sleep loop.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from brain.embed import cosine
from brain.memory import Memory
from brain.note import Note

# Tier label -> default subdirectory in the vault.
TIER_PATHS = {
    "note": "semantic/notes",
    "fact": "semantic/notes",
    "concept": "semantic/concepts",
    "person": "semantic/people",
    "skill": "procedural/skills",
    "playbook": "procedural/skills",
    "preference": "core/preferences",
    "reflection": "reflections",
    "conversation": "episodic/conversations",
    "feedback": "episodic/feedback",
}

DEFAULT_PROMOTION_THRESHOLD = 0.55
DEFAULT_MARGIN = 0.05


@dataclass
class ClassificationResult:
    slug: str
    chosen_kind: str
    confidence: float
    margin: float
    moved_to: str


def _gather_centroids(memory: Memory) -> dict[str, np.ndarray]:
    """Average vector per kind, excluding 'unsorted' and 'ghost'."""
    by_kind: dict[str, list[np.ndarray]] = defaultdict(list)
    for slug in memory.index.all_slugs():
        d = memory.index.doc(slug)
        if d is None or d.kind in ("unsorted", "ghost"):
            continue
        v = memory.index.vector(slug)
        if v is None:
            continue
        by_kind[d.kind].append(v)
    return {
        k: np.stack(v).mean(axis=0) / max(1e-8, np.linalg.norm(np.stack(v).mean(axis=0)))
        for k, v in by_kind.items()
    }


def auto_classify_inbox(
    memory: Memory,
    *,
    threshold: float = DEFAULT_PROMOTION_THRESHOLD,
    margin: float = DEFAULT_MARGIN,
    apply: bool = True,
) -> list[ClassificationResult]:
    """Classify every kind=unsorted note. Return decisions; optionally apply.

    If `apply=False` the function leaves notes in the inbox and only returns
    the would-be decisions — useful for the proposals UI.
    """
    centroids = _gather_centroids(memory)
    if not centroids:
        return []

    unsorted_slugs = [
        s for s in memory.index.all_slugs()
        if (d := memory.index.doc(s)) is not None and d.kind == "unsorted"
    ]
    if not unsorted_slugs:
        return []

    kinds = sorted(centroids.keys())
    cent_mat = np.stack([centroids[k] for k in kinds]).astype(np.float32)
    results: list[ClassificationResult] = []

    for slug in unsorted_slugs:
        v = memory.index.vector(slug)
        if v is None:
            continue
        sims = cosine(v[None, :], cent_mat)[0]
        order = np.argsort(-sims)
        best_i = int(order[0])
        best_sim = float(sims[best_i])
        runner_sim = float(sims[order[1]]) if len(order) > 1 else 0.0
        m = best_sim - runner_sim
        if best_sim < threshold or m < margin:
            continue
        chosen_kind = kinds[best_i]
        target_subdir = TIER_PATHS.get(chosen_kind, "semantic/notes")
        if apply:
            memory.vault.update(slug, kind=chosen_kind)
            memory.vault.relocate(slug, target_subdir)
            # Refresh index entry's kind field for downstream consumers.
            d = memory.index.doc(slug)
            if d is not None:
                d.kind = chosen_kind
        results.append(
            ClassificationResult(
                slug=slug,
                chosen_kind=chosen_kind,
                confidence=best_sim,
                margin=m,
                moved_to=target_subdir,
            )
        )
    return results
