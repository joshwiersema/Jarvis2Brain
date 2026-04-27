"""Vault: filesystem layer with atomic writes, nested CRUD, and link index.

v0.2: notes live in cognitive tiers (semantic/, episodic/, procedural/, core/, ...).
New notes default to episodic/inbox/ with kind=unsorted — the brain self-organizes
later via the daemon's auto_classify loop. Slugs remain globally unique.
"""

from __future__ import annotations

import hashlib
import os
import struct
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
_INTERNAL_DIRS = (".brain",)
_DEFAULT_INBOX = "episodic/inbox"


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


def _xy_from_slug(slug: str) -> tuple[float, float]:
    """Deterministic random xy in [-1, 1]^2 derived from the slug.

    Same slug always maps to the same starting position so tests are stable
    and cross-machine vaults converge identically before the NN trains.
    """
    h = hashlib.sha256(slug.encode("utf-8")).digest()
    a = struct.unpack("<Q", h[:8])[0] / 0xFFFFFFFFFFFFFFFF
    b = struct.unpack("<Q", h[8:16])[0] / 0xFFFFFFFFFFFFFFFF
    return (a * 2.0 - 1.0, b * 2.0 - 1.0)


class Vault:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)

    def _walk_md(self):
        for p in self.path.rglob("*.md"):
            try:
                rel = p.relative_to(self.path)
            except ValueError:
                continue
            if rel.parts and rel.parts[0] in _INTERNAL_DIRS:
                continue
            yield p

    def _find_file(self, slug: str) -> Optional[Path]:
        validate_slug(slug)
        for p in self._walk_md():
            if p.stem == slug:
                return p
        return None

    def _new_file_path(self, slug: str, subdir: Optional[str]) -> Path:
        validate_slug(slug)
        sub = subdir if subdir is not None else _DEFAULT_INBOX
        target_dir = self.path / sub if sub else self.path
        return target_dir / f"{slug}.md"

    def _atomic_write(self, target: Path, content: str) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
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
            try:
                tmp_path.unlink()
            except FileNotFoundError:
                pass
            raise

    def exists(self, slug: str) -> bool:
        return self._find_file(slug) is not None

    def create(
        self,
        note: Note,
        *,
        subdir: Optional[str] = None,
    ) -> Note:
        if self.exists(note.slug):
            raise NoteExistsError(f"note already exists: {note.slug}")

        # Default new notes to chaos-start: kind=unsorted, in episodic/inbox/.
        if subdir is None:
            note.kind = "unsorted" if note.kind == "note" else note.kind
        if note.xy is None:
            note.xy = _xy_from_slug(note.slug)
        if not note.links_out:
            note.links_out = extract_wiki_links(note.body)

        target = self._new_file_path(note.slug, subdir)
        self._atomic_write(target, serialize_note(note))
        return note

    def read(self, slug: str) -> Note:
        target = self._find_file(slug)
        if target is None:
            raise NoteNotFoundError(f"note not found: {slug}")
        return parse_note(slug, target.read_text(encoding="utf-8"))

    def update(
        self,
        slug: str,
        *,
        title: Optional[str] = None,
        body: Optional[str] = None,
        tags: Optional[list[str]] = None,
        kind: Optional[str] = None,
        importance: Optional[float] = None,
        xy: Optional[tuple[float, float]] = None,
    ) -> Note:
        target = self._find_file(slug)
        if target is None:
            raise NoteNotFoundError(f"note not found: {slug}")
        note = parse_note(slug, target.read_text(encoding="utf-8"))
        if title is not None:
            note.title = title
        if body is not None:
            note.body = body
            note.links_out = extract_wiki_links(body)
        if tags is not None:
            note.tags = list(tags)
        if kind is not None:
            note.kind = kind
        if importance is not None:
            note.importance = importance
        if xy is not None:
            note.xy = (float(xy[0]), float(xy[1]))
        note.updated = now_utc()
        # Re-validate via dataclass post_init by reconstructing.
        Note(
            slug=note.slug,
            title=note.title,
            created=note.created,
            updated=note.updated,
            tags=note.tags,
            body=note.body,
            kind=note.kind,
            importance=note.importance,
            observed_at=note.observed_at,
            valid_from=note.valid_from,
            valid_to=note.valid_to,
            links_out=note.links_out,
            xy=note.xy,
        )
        self._atomic_write(target, serialize_note(note))
        return note

    def relocate(self, slug: str, subdir: str) -> Note:
        """Move a note to a new tier. Used by auto_classify."""
        src = self._find_file(slug)
        if src is None:
            raise NoteNotFoundError(f"note not found: {slug}")
        note = parse_note(slug, src.read_text(encoding="utf-8"))
        new_path = self._new_file_path(slug, subdir)
        if new_path.resolve() == src.resolve():
            return note
        self._atomic_write(new_path, serialize_note(note))
        src.unlink()
        return note

    def delete(self, slug: str) -> None:
        target = self._find_file(slug)
        if target is None:
            raise NoteNotFoundError(f"note not found: {slug}")
        target.unlink()

    def list(self, *, tag: Optional[str] = None, kind: Optional[str] = None) -> list[NoteSummary]:
        out: list[NoteSummary] = []
        for path in sorted(self._walk_md()):
            slug = path.stem
            try:
                note = parse_note(slug, path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if tag is not None and tag not in note.tags:
                continue
            if kind is not None and note.kind != kind:
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
        for path in sorted(self._walk_md()):
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
                    "kind": note.kind,
                    "importance": note.importance,
                    "preview": note.body[:240],
                    "ghost": False,
                    "xy": list(note.xy) if note.xy else None,
                }
            )
            for target in extract_wiki_links(note.body):
                edges.append({"from": slug, "to": target, "kind": "wikilink"})

        referenced = {e["to"] for e in edges}
        for ghost in sorted(referenced - seen):
            nodes.append(
                {
                    "id": ghost,
                    "title": ghost,
                    "tags": [],
                    "kind": "ghost",
                    "importance": 0.0,
                    "preview": "",
                    "ghost": True,
                    "xy": list(_xy_from_slug(ghost)),
                }
            )
        return {"nodes": nodes, "edges": edges}

    def incoming_links(self, slug: str) -> list[str]:
        validate_slug(slug)
        sources: list[str] = []
        for path in sorted(self._walk_md()):
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
