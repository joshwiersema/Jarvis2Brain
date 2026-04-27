"""Phase 1: Note model upgrade — kind, bitemporal, importance, links_out, xy."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from brain.note import (
    Note,
    NoteParseError,
    parse_note,
    serialize_note,
)


def _ts(s: str) -> datetime:
    return datetime.fromisoformat(s).replace(tzinfo=timezone.utc)


def test_note_defaults_v01_compatible():
    n = Note(
        slug="hello",
        title="Hello",
        created=_ts("2026-01-01T00:00:00"),
        updated=_ts("2026-01-01T00:00:00"),
    )
    assert n.kind == "note"
    assert n.importance == pytest.approx(0.5)
    assert n.observed_at is None
    assert n.valid_from is None
    assert n.valid_to is None
    assert n.links_out == []
    assert n.xy is None


def test_kind_validator_accepts_lowercase_underscore():
    n = Note(
        slug="x",
        title="X",
        created=_ts("2026-01-01T00:00:00"),
        updated=_ts("2026-01-01T00:00:00"),
        kind="skill_meta",
    )
    assert n.kind == "skill_meta"


def test_kind_validator_rejects_invalid():
    with pytest.raises(ValueError):
        Note(
            slug="x",
            title="X",
            created=_ts("2026-01-01T00:00:00"),
            updated=_ts("2026-01-01T00:00:00"),
            kind="Bad-Kind",
        )


def test_importance_clamped():
    with pytest.raises(ValueError):
        Note(
            slug="x",
            title="X",
            created=_ts("2026-01-01T00:00:00"),
            updated=_ts("2026-01-01T00:00:00"),
            importance=1.5,
        )
    with pytest.raises(ValueError):
        Note(
            slug="x",
            title="X",
            created=_ts("2026-01-01T00:00:00"),
            updated=_ts("2026-01-01T00:00:00"),
            importance=-0.1,
        )


def test_serialize_roundtrip_with_new_fields():
    n = Note(
        slug="hello",
        title="Hello",
        created=_ts("2026-01-01T00:00:00"),
        updated=_ts("2026-01-01T00:00:00"),
        tags=["a"],
        body="see [[other]]",
        kind="fact",
        observed_at=_ts("2026-01-01T12:00:00"),
        valid_from=_ts("2026-01-01T12:00:00"),
        valid_to=_ts("2026-02-01T00:00:00"),
        importance=0.8,
        links_out=["other"],
        xy=(0.123, -0.456),
    )
    text = serialize_note(n)
    parsed = parse_note("hello", text)
    assert parsed.kind == "fact"
    assert parsed.importance == pytest.approx(0.8)
    assert parsed.observed_at == _ts("2026-01-01T12:00:00")
    assert parsed.valid_from == _ts("2026-01-01T12:00:00")
    assert parsed.valid_to == _ts("2026-02-01T00:00:00")
    assert parsed.links_out == ["other"]
    assert parsed.xy is not None
    assert parsed.xy[0] == pytest.approx(0.123)
    assert parsed.xy[1] == pytest.approx(-0.456)


def test_legacy_v01_note_parses_with_defaults():
    text = (
        "---\n"
        "title: Old\n"
        "created: 2026-01-01T00:00:00+00:00\n"
        "updated: 2026-01-01T00:00:00+00:00\n"
        "tags: []\n"
        "---\n"
        "body text\n"
    )
    n = parse_note("old", text)
    assert n.kind == "note"
    assert n.importance == pytest.approx(0.5)
    assert n.xy is None
    assert n.links_out == []


def test_invalid_xy_rejected_in_frontmatter():
    text = (
        "---\n"
        "title: Bad\n"
        "created: 2026-01-01T00:00:00+00:00\n"
        "updated: 2026-01-01T00:00:00+00:00\n"
        "tags: []\n"
        "xy: [1.0, 2.0, 3.0]\n"
        "---\n"
    )
    with pytest.raises(NoteParseError):
        parse_note("bad", text)
