"""Phase 1: Vault nested storage + chaos-start inbox routing."""

from __future__ import annotations

from pathlib import Path

import pytest

from brain.note import Note, now_utc
from brain.vault import NoteNotFoundError, Vault


def _make(slug: str, **kw) -> Note:
    ts = now_utc()
    return Note(slug=slug, title=kw.pop("title", slug), created=ts, updated=ts, **kw)


def test_create_default_lands_in_episodic_inbox(tmp_path: Path):
    v = Vault(tmp_path)
    v.create(_make("hello"))
    assert (tmp_path / "episodic" / "inbox" / "hello.md").exists()


def test_create_default_kind_is_unsorted(tmp_path: Path):
    v = Vault(tmp_path)
    v.create(_make("hello"))
    n = v.read("hello")
    assert n.kind == "unsorted"


def test_create_assigns_random_xy(tmp_path: Path):
    v = Vault(tmp_path)
    v.create(_make("hello"))
    n = v.read("hello")
    assert n.xy is not None
    assert -1.0 <= n.xy[0] <= 1.0
    assert -1.0 <= n.xy[1] <= 1.0


def test_xy_is_deterministic_per_slug(tmp_path: Path):
    v1 = Vault(tmp_path / "a")
    v2 = Vault(tmp_path / "b")
    v1.create(_make("hello"))
    v2.create(_make("hello"))
    assert v1.read("hello").xy == v2.read("hello").xy


def test_create_with_explicit_subdir(tmp_path: Path):
    v = Vault(tmp_path)
    v.create(_make("persona"), subdir="core")
    assert (tmp_path / "core" / "persona.md").exists()


def test_read_finds_note_at_any_depth(tmp_path: Path):
    v = Vault(tmp_path)
    v.create(_make("deep"), subdir="semantic/notes")
    n = v.read("deep")
    assert n.slug == "deep"


def test_read_legacy_flat_note_works(tmp_path: Path):
    v = Vault(tmp_path)
    flat = tmp_path / "legacy.md"
    flat.write_text(
        "---\n"
        "title: Legacy\n"
        "created: 2026-01-01T00:00:00+00:00\n"
        "updated: 2026-01-01T00:00:00+00:00\n"
        "tags: []\n"
        "---\n"
        "body\n",
        encoding="utf-8",
    )
    n = v.read("legacy")
    assert n.title == "Legacy"


def test_update_finds_existing_in_subdir(tmp_path: Path):
    v = Vault(tmp_path)
    v.create(_make("hello"))
    v.update("hello", body="new body")
    assert v.read("hello").body.startswith("new body")
    assert (tmp_path / "episodic" / "inbox" / "hello.md").exists()


def test_delete_finds_at_any_depth(tmp_path: Path):
    v = Vault(tmp_path)
    v.create(_make("hello"))
    v.delete("hello")
    with pytest.raises(NoteNotFoundError):
        v.read("hello")


def test_list_walks_recursively(tmp_path: Path):
    v = Vault(tmp_path)
    v.create(_make("a"))
    v.create(_make("b"), subdir="semantic/notes")
    v.create(_make("c"), subdir="core")
    slugs = {s.slug for s in v.list()}
    assert slugs == {"a", "b", "c"}


def test_graph_walks_recursively(tmp_path: Path):
    v = Vault(tmp_path)
    v.create(_make("a", body="see [[b]]"))
    v.create(_make("b"), subdir="semantic/notes")
    g = v.graph()
    ids = {n["id"] for n in g["nodes"]}
    assert "a" in ids and "b" in ids
    assert any(e["from"] == "a" and e["to"] == "b" for e in g["edges"])


def test_links_out_persisted_on_create(tmp_path: Path):
    v = Vault(tmp_path)
    v.create(_make("a", body="see [[b]] and [[c]]"))
    n = v.read("a")
    assert n.links_out == ["b", "c"]


def test_links_out_updates_on_body_change(tmp_path: Path):
    v = Vault(tmp_path)
    v.create(_make("a", body="see [[b]]"))
    v.update("a", body="now see [[c]] and [[d]]")
    assert v.read("a").links_out == ["c", "d"]
