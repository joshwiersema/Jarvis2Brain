from datetime import datetime, timezone

import pytest

from brain.note import Note, NoteParseError, parse_note, serialize_note


class TestRoundTrip:
    def test_minimal(self) -> None:
        text = (
            "---\n"
            "title: My Note\n"
            "created: 2026-04-26T02:00:00+00:00\n"
            "updated: 2026-04-26T02:00:00+00:00\n"
            "tags: []\n"
            "---\n"
            "Body line one.\n"
        )
        note = parse_note("my-note", text)
        assert note.slug == "my-note"
        assert note.title == "My Note"
        assert note.tags == []
        assert note.body == "Body line one.\n"
        assert serialize_note(note) == text

    def test_with_tags_and_wiki_links(self) -> None:
        text = (
            "---\n"
            "title: Linked\n"
            "created: 2026-04-26T02:00:00+00:00\n"
            "updated: 2026-04-26T02:00:00+00:00\n"
            "tags:\n"
            "- inbox\n"
            "- ideas\n"
            "---\n"
            "See [[other-note]] and [[third-note|alias]].\n"
        )
        note = parse_note("linked", text)
        assert note.tags == ["inbox", "ideas"]
        assert "[[other-note]]" in note.body
        assert serialize_note(note) == text


class TestFrontmatterSurvivesUpdate:
    def test_body_change_preserves_frontmatter(self) -> None:
        original = Note(
            slug="x",
            title="X",
            created=datetime(2026, 4, 26, 2, 0, tzinfo=timezone.utc),
            updated=datetime(2026, 4, 26, 2, 0, tzinfo=timezone.utc),
            tags=["t1"],
            body="old body\n",
        )
        text = serialize_note(original)
        roundtrip = parse_note("x", text)
        assert roundtrip.title == "X"
        assert roundtrip.tags == ["t1"]
        assert roundtrip.created == original.created


class TestErrors:
    def test_missing_frontmatter(self) -> None:
        with pytest.raises(NoteParseError):
            parse_note("x", "no frontmatter here\n")

    def test_unterminated_frontmatter(self) -> None:
        with pytest.raises(NoteParseError):
            parse_note("x", "---\ntitle: X\nbody without close\n")

    def test_missing_required_field(self) -> None:
        text = "---\ncreated: 2026-04-26T02:00:00+00:00\n---\nbody\n"
        with pytest.raises(NoteParseError):
            parse_note("x", text)
