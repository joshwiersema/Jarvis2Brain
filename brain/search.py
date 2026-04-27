"""Substring search across title + body, with optional tag filter."""

from __future__ import annotations

from typing import Optional

from brain.note import NoteSummary, parse_note, summarize
from brain.vault import Vault


def search(vault: Vault, query: str, *, tag: Optional[str] = None) -> list[NoteSummary]:
    q = query.lower()
    out: list[NoteSummary] = []
    for path in sorted(vault._walk_md()):
        slug = path.stem
        try:
            note = parse_note(slug, path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if tag is not None and tag not in note.tags:
            continue
        if q and q not in note.title.lower() and q not in note.body.lower():
            continue
        out.append(summarize(note))
    return out
