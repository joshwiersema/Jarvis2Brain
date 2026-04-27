"""Note model + YAML frontmatter parse/serialize.

v0.2: adds kind, importance, bitemporal (observed_at/valid_from/valid_to),
links_out (denormalized), xy (2D layout coords for the brain visualizer).
Backward-compatible with v0.1 frontmatter — missing fields use defaults.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import yaml

_FENCE = "---\n"
_KIND_RE = re.compile(r"^[a-z][a-z0-9_]*$")


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
    kind: str = "note"
    importance: float = 0.5
    observed_at: Optional[datetime] = None
    valid_from: Optional[datetime] = None
    valid_to: Optional[datetime] = None
    links_out: list[str] = field(default_factory=list)
    xy: Optional[tuple[float, float]] = None

    def __post_init__(self) -> None:
        if not _KIND_RE.match(self.kind):
            raise ValueError(f"invalid kind: {self.kind!r}")
        if not 0.0 <= self.importance <= 1.0:
            raise ValueError(f"importance out of range [0, 1]: {self.importance}")
        if self.xy is not None:
            if len(self.xy) != 2:
                raise ValueError(f"xy must be length 2: {self.xy}")
            self.xy = (float(self.xy[0]), float(self.xy[1]))


@dataclass
class NoteSummary:
    slug: str
    title: str
    tags: list[str]
    updated: datetime
    kind: str = "note"


def now_utc() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def _split_frontmatter(text: str) -> tuple[str, str]:
    if not text.startswith(_FENCE):
        raise NoteParseError("note must start with '---' frontmatter fence")
    rest = text[len(_FENCE) :]
    end = rest.find("\n" + _FENCE.rstrip("\n") + "\n")
    if end == -1:
        if rest.endswith("\n---"):
            return rest[: -len("\n---")], ""
        raise NoteParseError("frontmatter fence not terminated")
    fm = rest[:end]
    body = rest[end + len("\n---\n") :]
    return fm, body


def _coerce_dt(value: object, field_name: str) -> datetime:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value)
        except ValueError as e:
            raise NoteParseError(
                f"invalid datetime for {field_name}: {value!r}"
            ) from e
    raise NoteParseError(f"{field_name} must be a datetime or ISO string")


def _coerce_optional_dt(value: object, field_name: str) -> Optional[datetime]:
    if value is None:
        return None
    return _coerce_dt(value, field_name)


def _coerce_xy(value: object) -> Optional[tuple[float, float]]:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise NoteParseError(f"xy must be a 2-element list, got {value!r}")
    try:
        return (float(value[0]), float(value[1]))
    except (TypeError, ValueError) as e:
        raise NoteParseError(f"xy values must be numeric: {value!r}") from e


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

    importance = fm.get("importance", 0.5)
    try:
        importance = float(importance)
    except (TypeError, ValueError) as e:
        raise NoteParseError(f"importance must be numeric: {importance!r}") from e

    links_out = fm.get("links_out") or []
    if not isinstance(links_out, list):
        raise NoteParseError("links_out must be a list")

    try:
        return Note(
            slug=slug,
            title=str(fm["title"]),
            created=_coerce_dt(fm["created"], "created"),
            updated=_coerce_dt(fm["updated"], "updated"),
            tags=list(fm.get("tags") or []),
            body=body,
            kind=str(fm.get("kind", "note")),
            importance=importance,
            observed_at=_coerce_optional_dt(fm.get("observed_at"), "observed_at"),
            valid_from=_coerce_optional_dt(fm.get("valid_from"), "valid_from"),
            valid_to=_coerce_optional_dt(fm.get("valid_to"), "valid_to"),
            links_out=[str(s) for s in links_out],
            xy=_coerce_xy(fm.get("xy")),
        )
    except ValueError as e:
        raise NoteParseError(str(e)) from e


def _yaml_str(value: str) -> str:
    """Return a single-line YAML scalar for the value (quoted if needed)."""
    rendered = yaml.safe_dump(value, default_flow_style=True, allow_unicode=True).strip()
    # safe_dump may append a trailing newline-equivalent or wrap in [].
    if rendered.endswith("\n..."):
        rendered = rendered[:-4]
    return rendered


def serialize_note(note: Note) -> str:
    lines = [
        f"title: {_yaml_str(note.title)}",
        f"created: {note.created.isoformat()}",
        f"updated: {note.updated.isoformat()}",
    ]
    if note.tags:
        lines.append("tags:")
        lines.extend(f"- {t}" for t in note.tags)
    else:
        lines.append("tags: []")
    lines.append(f"kind: {note.kind}")
    lines.append(f"importance: {note.importance}")
    if note.observed_at is not None:
        lines.append(f"observed_at: {note.observed_at.isoformat()}")
    if note.valid_from is not None:
        lines.append(f"valid_from: {note.valid_from.isoformat()}")
    if note.valid_to is not None:
        lines.append(f"valid_to: {note.valid_to.isoformat()}")
    if note.links_out:
        lines.append("links_out:")
        lines.extend(f"- {s}" for s in note.links_out)
    if note.xy is not None:
        lines.append(f"xy: [{note.xy[0]}, {note.xy[1]}]")
    fm_text = "\n".join(lines) + "\n"
    body = note.body
    if body and not body.endswith("\n"):
        body += "\n"
    return f"{_FENCE}{fm_text}{_FENCE}{body}"


def summarize(note: Note) -> NoteSummary:
    return NoteSummary(
        slug=note.slug,
        title=note.title,
        tags=list(note.tags),
        updated=note.updated,
        kind=note.kind,
    )
