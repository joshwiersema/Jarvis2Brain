"""Skill synthesis — propose skills from clusters of successful traces.

This is the v0.2 minimum: greedy clustering on trace embeddings, draft a
`kind=skill_proposal` note when a cluster has >= min_cluster_size members.
The full Claude-drafted skill body + verifier-test loop is the v0.3 spike.

Auto-promotion never touches effectors or `core/constitution.md`.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

import numpy as np

from brain.embed import cosine
from brain.memory import Memory
from brain.note import Note, now_utc

DEFAULT_MIN_CLUSTER = 3
DEFAULT_SIM_THRESHOLD = 0.55


@dataclass
class SkillProposal:
    slug: str
    seed_slugs: list[str]
    description: str
    confidence: float


def _greedy_cluster(
    vectors: np.ndarray,
    threshold: float,
) -> list[list[int]]:
    """Threshold-based greedy cosine clustering. O(n^2), fine at vault scale."""
    if vectors.shape[0] == 0:
        return []
    n = vectors.shape[0]
    sims = cosine(vectors, vectors)
    assigned = [-1] * n
    clusters: list[list[int]] = []
    for i in range(n):
        if assigned[i] != -1:
            continue
        cluster = [i]
        assigned[i] = len(clusters)
        for j in range(i + 1, n):
            if assigned[j] == -1 and sims[i, j] >= threshold:
                cluster.append(j)
                assigned[j] = len(clusters) - 1 + 1
        clusters.append(cluster)
    return clusters


def propose_skills(
    memory: Memory,
    *,
    min_cluster_size: int = DEFAULT_MIN_CLUSTER,
    threshold: float = DEFAULT_SIM_THRESHOLD,
) -> list[SkillProposal]:
    trace_slugs = [
        s for s in memory.index.all_slugs()
        if (d := memory.index.doc(s)) is not None and d.kind in ("trace", "conversation")
    ]
    if len(trace_slugs) < min_cluster_size:
        return []
    vectors = np.stack([memory.index.vector(s) for s in trace_slugs])
    clusters = _greedy_cluster(vectors, threshold=threshold)
    proposals: list[SkillProposal] = []
    for cluster in clusters:
        if len(cluster) < min_cluster_size:
            continue
        member_slugs = [trace_slugs[i] for i in cluster]
        # Pick the member closest to the cluster centroid as the descriptor seed.
        members_v = np.stack([memory.index.vector(s) for s in member_slugs])
        centroid = members_v.mean(axis=0)
        centroid /= max(1e-8, np.linalg.norm(centroid))
        sims = cosine(centroid[None, :], members_v)[0]
        seed = member_slugs[int(sims.argmax())]
        seed_doc = memory.index.doc(seed)
        title = (seed_doc.text.splitlines()[0] if seed_doc and seed_doc.text else seed)[:60]
        slug = f"skill-proposal-{seed[:40]}"
        confidence = float(sims.mean())
        proposals.append(
            SkillProposal(
                slug=slug,
                seed_slugs=member_slugs,
                description=f"Auto-proposed skill from {len(member_slugs)} similar traces: {title}",
                confidence=confidence,
            )
        )
    return proposals


def write_proposals(memory: Memory, proposals: list[SkillProposal]) -> int:
    n = 0
    ts = now_utc()
    for p in proposals:
        if memory.vault.exists(p.slug):
            continue
        body = (
            f"# {p.description}\n\n"
            f"**Confidence:** {p.confidence:.2f}\n\n"
            f"**Source traces:**\n\n"
            + "\n".join(f"- [[{s}]]" for s in p.seed_slugs)
            + "\n\n"
            f"_Drafted by skill_synthesis loop. Review and promote into "
            f"`vault/.brain/skills/` to activate._\n"
        )
        note = Note(
            slug=p.slug,
            title=p.description[:80],
            created=ts,
            updated=ts,
            body=body,
            kind="skill_proposal",
            importance=0.6,
            tags=["proposal", "skill_proposal"],
        )
        memory.vault.create(note, subdir="proposals")
        n += 1
    return n
