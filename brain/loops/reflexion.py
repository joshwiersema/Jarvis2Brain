"""Reflexion — failure-grounded learning notes.

Refuses to write a reflection without a concrete failure artifact (test
output, diff, error trace). Vibes don't make it into memory.

When a similar task surfaces later, `retrieve_relevant` returns reflections
ranked by recency * importance, so the agent can read its prior failures
before trying again.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from brain.note import Note, now_utc
from brain.vault import Vault


@dataclass
class FailureArtifact:
    kind: str  # test_output | diff | screenshot_diff | error_trace
    content: str  # raw artifact text/path

    def is_meaningful(self) -> bool:
        return bool((self.content or "").strip())


def record_reflection(
    vault: Vault,
    *,
    task_summary: str,
    artifact: FailureArtifact,
    lesson: str,
) -> Optional[Note]:
    if not artifact.is_meaningful():
        return None
    if not lesson.strip():
        return None
    ts = now_utc()
    slug_base = "reflection-" + str(int(ts.timestamp()))
    body = (
        f"## Task\n\n{task_summary}\n\n"
        f"## Failure artifact ({artifact.kind})\n\n```\n{artifact.content[:4000]}\n```\n\n"
        f"## Lesson\n\n{lesson}\n"
    )
    note = Note(
        slug=slug_base,
        title=f"Reflection: {task_summary[:60]}",
        created=ts,
        updated=ts,
        body=body,
        kind="reflection",
        importance=0.7,
        tags=["reflection", artifact.kind],
    )
    sub = f"reflections/{ts.year}/{ts.month:02d}"
    vault.create(note, subdir=sub)
    return note


def retrieve_relevant(memory, query: str, k: int = 5) -> list:
    """Pull the top-k reflections for a query — used at task start."""
    return memory.archival(query, k=k, kind="reflection")
