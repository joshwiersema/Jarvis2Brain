"""Note CRUD skills."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from brain.note import Note, now_utc
from brain.skill import skill
from brain.vault import NoteExistsError, NoteNotFoundError, Vault, resolve_vault_path


def _vault() -> Vault:
    return Vault(resolve_vault_path(cli=os.environ.get("BRAIN_VAULT"), cwd=Path.cwd()))


def _serialize(note: Note) -> dict:
    return {
        "slug": note.slug,
        "title": note.title,
        "body": note.body,
        "tags": list(note.tags),
        "kind": note.kind,
        "importance": note.importance,
        "created": note.created.isoformat(),
        "updated": note.updated.isoformat(),
        "xy": list(note.xy) if note.xy else None,
        "links_out": list(note.links_out),
    }


@skill(name="notes_create", description="Create a new note in the vault")
def notes_create(
    slug: str,
    title: str,
    body: str = "",
    tags: Optional[list[str]] = None,
    kind: Optional[str] = None,
    subdir: Optional[str] = None,
) -> dict:
    ts = now_utc()
    note = Note(
        slug=slug,
        title=title,
        created=ts,
        updated=ts,
        tags=list(tags or []),
        body=body,
        kind=kind or "note",
    )
    v = _vault()
    try:
        v.create(note, subdir=subdir)
    except NoteExistsError as e:
        return {"error": str(e), "code": "exists"}
    return _serialize(note)


@skill(name="notes_read", description="Read a note by slug")
def notes_read(slug: str) -> dict:
    try:
        return _serialize(_vault().read(slug))
    except NoteNotFoundError as e:
        return {"error": str(e), "code": "not_found"}


@skill(name="notes_update", description="Update fields on a note")
def notes_update(
    slug: str,
    title: Optional[str] = None,
    body: Optional[str] = None,
    tags: Optional[list[str]] = None,
    kind: Optional[str] = None,
    importance: Optional[float] = None,
) -> dict:
    try:
        note = _vault().update(
            slug, title=title, body=body, tags=tags, kind=kind, importance=importance
        )
    except NoteNotFoundError as e:
        return {"error": str(e), "code": "not_found"}
    return _serialize(note)


@skill(name="notes_delete", description="Delete a note by slug")
def notes_delete(slug: str) -> dict:
    try:
        _vault().delete(slug)
    except NoteNotFoundError as e:
        return {"error": str(e), "code": "not_found"}
    return {"slug": slug, "deleted": True}


@skill(name="notes_list", description="List notes (optionally filter by tag/kind)")
def notes_list(tag: Optional[str] = None, kind: Optional[str] = None) -> list[dict]:
    return [
        {
            "slug": s.slug,
            "title": s.title,
            "tags": list(s.tags),
            "kind": s.kind,
            "updated": s.updated.isoformat(),
        }
        for s in _vault().list(tag=tag, kind=kind)
    ]
