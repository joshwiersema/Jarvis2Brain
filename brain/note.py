"""Note model + YAML frontmatter parse/serialize."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

import yaml

_FENCE = "---\n"


class NoteParseError(ValueError):
    """Raised when a note cannot be parsed from text."""


@dataclass
class Note:
    slug: str
    title: str
    created: datetime
    updated: datetime
    tags: list[str] = field(default_factory=list)
    body: str = ""


@dataclass
class NoteSummary:
    slug: str
    title: str
    tags: list[str]
    updated: datetime


def now_utc() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def _split_frontmatter(text: str) -> tuple[str, str]:
    if not text.startswith(_FENCE):
        raise NoteParseError("note must start with '---' frontmatter fence")
    rest = text[len(_FENCE) :]
    end = rest.find("\n" + _FENCE.rstrip("\n") + "\n")
    if end == -1:
        # Allow trailing fence at end of file with no newline after
        if rest.endswith("\n---"):
            return rest[: -len("\n---")], ""
        raise NoteParseError("frontmatter fence not terminated")
    fm = rest[:end]
    body = rest[end + len("\n---\n") :]
    return fm, body


def parse_note(slug: str, text: str) -> Note:
    fm_text, body = _split_frontmatter(text)
    try:
        fm = yaml.safe_load(fm_text) or {}
    except yaml.YAMLError as e:
        raise NoteParseError(f"invalid YAML frontmatter: {e}") from e
    if not isinstance(fm, dict):
        raise NoteParseError("frontmatter must be a YAML mapping")

    required = ("title", "created", "updated")
    missing = [k for k in required if k not in fm]
    if missing:
        raise NoteParseError(f"missing frontmatter fields: {missing}")

    return Note(
        slug=slug,
        title=str(fm["title"]),
        created=_coerce_dt(fm["created"], "created"),
        updated=_coerce_dt(fm["updated"], "updated"),
        tags=list(fm.get("tags") or []),
        body=body,
    )


def _coerce_dt(value: object, field_name: str) -> datetime:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value)
        except ValueError as e:
            raise NoteParseError(f"invalid datetime for {field_name}: {value!r}") from e
    raise NoteParseError(f"{field_name} must be a datetime or ISO string")


def serialize_note(note: Note) -> str:
    lines = [
        f"title: {note.title}",
        f"created: {note.created.isoformat()}",
        f"updated: {note.updated.isoformat()}",
    ]
    if note.tags:
        lines.append("tags:")
        lines.extend(f"- {t}" for t in note.tags)
    else:
        lines.append("tags: []")
    fm_text = "\n".join(lines) + "\n"
    body = note.body
    if body and not body.endswith("\n"):
        body += "\n"
    return f"{_FENCE}{fm_text}{_FENCE}{body}"


def summarize(note: Note) -> NoteSummary:
    return NoteSummary(
        slug=note.slug, title=note.title, tags=list(note.tags), updated=note.updated
    )
