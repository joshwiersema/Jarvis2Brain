"""Vault: filesystem layer with atomic writes, CRUD, and link index."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Optional

from brain.links import extract_wiki_links
from brain.note import (
    Note,
    NoteSummary,
    now_utc,
    parse_note,
    serialize_note,
    summarize,
)
from brain.slug import validate_slug

_ENV_VAR = "BRAIN_VAULT"


class VaultError(Exception):
    """Base class for vault errors."""


class NoteNotFoundError(VaultError):
    """Raised when a note does not exist."""


class NoteExistsError(VaultError):
    """Raised when a note already exists on create."""


def resolve_vault_path(cli: Optional[str], cwd: Path) -> Path:
    """Resolve vault path: CLI flag > BRAIN_VAULT env var > ./vault."""
    if cli:
        return Path(cli).expanduser().resolve()
    env = os.environ.get(_ENV_VAR)
    if env:
        return Path(env).expanduser().resolve()
    return (cwd / "vault").resolve()


class Vault:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)

    def _file(self, slug: str) -> Path:
        validate_slug(slug)
        return self.path / f"{slug}.md"

    def _atomic_write(self, target: Path, content: str) -> None:
        # Write to a temp file in the same dir, fsync, then atomically rename.
        fd, tmp_name = tempfile.mkstemp(
            prefix=f".{target.stem}.", suffix=".tmp", dir=str(target.parent)
        )
        tmp_path = Path(tmp_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
                f.write(content)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, target)
        except BaseException:
            # On any failure, clean up the temp file so vault stays uncluttered.
            try:
                tmp_path.unlink()
            except FileNotFoundError:
                pass
            raise

    def exists(self, slug: str) -> bool:
        return self._file(slug).exists()

    def create(self, note: Note) -> Note:
        target = self._file(note.slug)
        if target.exists():
            raise NoteExistsError(f"note already exists: {note.slug}")
        self._atomic_write(target, serialize_note(note))
        return note

    def read(self, slug: str) -> Note:
        target = self._file(slug)
        if not target.exists():
            raise NoteNotFoundError(f"note not found: {slug}")
        return parse_note(slug, target.read_text(encoding="utf-8"))

    def update(
        self,
        slug: str,
        *,
        title: Optional[str] = None,
        body: Optional[str] = None,
        tags: Optional[list[str]] = None,
    ) -> Note:
        note = self.read(slug)
        if title is not None:
            note.title = title
        if body is not None:
            note.body = body
        if tags is not None:
            note.tags = list(tags)
        note.updated = now_utc()
        self._atomic_write(self._file(slug), serialize_note(note))
        return note

    def delete(self, slug: str) -> None:
        target = self._file(slug)
        if not target.exists():
            raise NoteNotFoundError(f"note not found: {slug}")
        target.unlink()

    def list(self, *, tag: Optional[str] = None) -> list[NoteSummary]:
        out: list[NoteSummary] = []
        for path in sorted(self.path.glob("*.md")):
            slug = path.stem
            try:
                note = parse_note(slug, path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if tag is not None and tag not in note.tags:
                continue
            out.append(summarize(note))
        return out

    def outgoing_links(self, slug: str) -> list[str]:
        return extract_wiki_links(self.read(slug).body)

    def graph(self) -> dict:
        """Aggregate all notes into nodes + edges. Adds ghost nodes for broken links."""
        nodes: list[dict] = []
        edges: list[dict] = []
        seen: set[str] = set()
        for path in sorted(self.path.glob("*.md")):
            slug = path.stem
            try:
                note = parse_note(slug, path.read_text(encoding="utf-8"))
            except Exception:
                continue
            seen.add(slug)
            nodes.append(
                {
                    "id": slug,
                    "title": note.title,
                    "tags": list(note.tags),
                    "preview": note.body[:240],
                    "ghost": False,
                }
            )
            for target in extract_wiki_links(note.body):
                edges.append({"from": slug, "to": target})

        referenced = {e["to"] for e in edges}
        for ghost in sorted(referenced - seen):
            nodes.append(
                {"id": ghost, "title": ghost, "tags": [], "preview": "", "ghost": True}
            )
        return {"nodes": nodes, "edges": edges}

    def incoming_links(self, slug: str) -> list[str]:
        validate_slug(slug)
        sources: list[str] = []
        for path in sorted(self.path.glob("*.md")):
            other = path.stem
            if other == slug:
                continue
            try:
                note = parse_note(other, path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if slug in extract_wiki_links(note.body):
                sources.append(other)
        return sources
