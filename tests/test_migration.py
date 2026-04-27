"""Phase 1: v0.1 -> v0.2 migration script."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.migrate_v01_to_v02 import migrate


def _legacy(path: Path, slug: str, body: str = "body") -> None:
    (path / f"{slug}.md").write_text(
        f"---\n"
        f"title: {slug.title()}\n"
        f"created: 2026-01-01T00:00:00+00:00\n"
        f"updated: 2026-01-01T00:00:00+00:00\n"
        f"tags: []\n"
        f"---\n"
        f"{body}\n",
        encoding="utf-8",
    )


def test_dry_run_does_not_write(tmp_path: Path):
    _legacy(tmp_path, "hello")
    plan = migrate(tmp_path, apply=False)
    assert plan["moves"]
    # Source file untouched.
    assert (tmp_path / "hello.md").exists()
    assert not (tmp_path / "semantic" / "notes" / "hello.md").exists()


def test_apply_moves_flat_notes_to_semantic(tmp_path: Path):
    _legacy(tmp_path, "hello")
    _legacy(tmp_path, "world")
    migrate(tmp_path, apply=True)
    assert (tmp_path / "semantic" / "notes" / "hello.md").exists()
    assert (tmp_path / "semantic" / "notes" / "world.md").exists()
    assert not (tmp_path / "hello.md").exists()


def test_apply_creates_full_directory_structure(tmp_path: Path):
    _legacy(tmp_path, "x")
    migrate(tmp_path, apply=True)
    for sub in (
        "episodic",
        "episodic/inbox",
        "semantic",
        "semantic/notes",
        "procedural",
        "procedural/skills",
        "core",
        "reflections",
        "proposals",
        ".brain",
    ):
        assert (tmp_path / sub).exists(), sub


def test_apply_is_idempotent(tmp_path: Path):
    _legacy(tmp_path, "hello")
    migrate(tmp_path, apply=True)
    # Re-running on already-migrated vault is a no-op.
    plan2 = migrate(tmp_path, apply=True)
    assert plan2["moves"] == []


def test_apply_adds_kind_note_when_missing(tmp_path: Path):
    _legacy(tmp_path, "hello")
    migrate(tmp_path, apply=True)
    text = (tmp_path / "semantic" / "notes" / "hello.md").read_text(encoding="utf-8")
    assert "kind: note" in text


def test_apply_skips_already_migrated_with_same_content_hash(tmp_path: Path):
    _legacy(tmp_path, "hello")
    target_dir = tmp_path / "semantic" / "notes"
    target_dir.mkdir(parents=True)
    # Pre-existing identical content at target
    target_dir.joinpath("hello.md").write_text(
        (tmp_path / "hello.md").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    plan = migrate(tmp_path, apply=True)
    # Should still remove flat duplicate.
    assert not (tmp_path / "hello.md").exists()
    assert (target_dir / "hello.md").exists()


def test_apply_refuses_overwrite_on_content_conflict(tmp_path: Path):
    _legacy(tmp_path, "hello", body="flat body")
    target_dir = tmp_path / "semantic" / "notes"
    target_dir.mkdir(parents=True)
    target_dir.joinpath("hello.md").write_text(
        "---\n"
        "title: Different\n"
        "created: 2026-01-01T00:00:00+00:00\n"
        "updated: 2026-01-01T00:00:00+00:00\n"
        "tags: []\n"
        "kind: note\n"
        "---\n"
        "different body\n",
        encoding="utf-8",
    )
    with pytest.raises(SystemExit):
        migrate(tmp_path, apply=True)
