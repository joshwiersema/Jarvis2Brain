"""Jarvis 2 SQLite -> vault migration tests."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from scripts.migrate_jarvis2_sqlite import migrate


def _build_fixture(tmp_path: Path) -> Path:
    db = tmp_path / "j2.db"
    conn = sqlite3.connect(str(db))
    conn.executescript(
        """
        CREATE TABLE conversations (
            session_id TEXT, turn_id TEXT, role TEXT, model TEXT,
            content TEXT, created TEXT, tokens INTEGER
        );
        CREATE TABLE preferences (
            slug TEXT, name TEXT, value TEXT, updated TEXT
        );
        CREATE TABLE feedback (
            session_id TEXT, turn_id TEXT, kind TEXT,
            content TEXT, created TEXT
        );
        INSERT INTO conversations VALUES
          ('s1', 't1', 'user', 'gpt', 'hello there', '2026-01-01T00:00:00+00:00', 5),
          ('s1', 't2', 'assistant', 'gpt', 'hi back', '2026-01-01T00:00:01+00:00', 8);
        INSERT INTO preferences VALUES
          ('terse-replies', 'Terse replies', 'always', '2026-01-02T00:00:00+00:00');
        INSERT INTO feedback VALUES
          ('s1', 't1', 'thumbs_up', 'helpful', '2026-01-03T00:00:00+00:00');
        """
    )
    conn.commit()
    conn.close()
    return db


def test_dry_run_does_not_write(tmp_path: Path):
    db = _build_fixture(tmp_path)
    vault = tmp_path / "vault"
    vault.mkdir()
    plan = migrate(db, vault, apply=False)
    assert plan["n_rows"] == 4
    # Nothing on disk.
    assert not any(vault.rglob("*.md"))


def test_apply_writes_conversation_notes(tmp_path: Path):
    db = _build_fixture(tmp_path)
    vault = tmp_path / "vault"
    vault.mkdir()
    plan = migrate(db, vault, apply=True)
    convs = list((vault / "episodic" / "conversations").rglob("*.md"))
    assert len(convs) == 2


def test_apply_writes_preference(tmp_path: Path):
    db = _build_fixture(tmp_path)
    vault = tmp_path / "vault"
    vault.mkdir()
    migrate(db, vault, apply=True)
    prefs = list((vault / "core" / "preferences").rglob("*.md"))
    assert len(prefs) == 1
    text = prefs[0].read_text(encoding="utf-8")
    assert "kind: preference" in text


def test_apply_writes_feedback(tmp_path: Path):
    db = _build_fixture(tmp_path)
    vault = tmp_path / "vault"
    vault.mkdir()
    migrate(db, vault, apply=True)
    fb = list((vault / "episodic" / "feedback").rglob("*.md"))
    assert len(fb) == 1


def test_apply_is_idempotent(tmp_path: Path):
    db = _build_fixture(tmp_path)
    vault = tmp_path / "vault"
    vault.mkdir()
    migrate(db, vault, apply=True)
    plan2 = migrate(db, vault, apply=True)
    # Second run sees only skip_dup ops, no writes.
    ops = {p["op"] for p in plan2["plan"]}
    assert ops == {"skip_dup"}


def test_missing_optional_tables(tmp_path: Path):
    db = tmp_path / "min.db"
    conn = sqlite3.connect(str(db))
    conn.executescript(
        """
        CREATE TABLE conversations (
            session_id TEXT, turn_id TEXT, role TEXT, model TEXT,
            content TEXT, created TEXT, tokens INTEGER
        );
        INSERT INTO conversations VALUES
          ('s', 't', 'u', 'm', 'c', '2026-01-01T00:00:00+00:00', 1);
        """
    )
    conn.commit()
    conn.close()
    vault = tmp_path / "vault"
    vault.mkdir()
    plan = migrate(db, vault, apply=True)
    assert plan["n_rows"] == 1


def test_missing_db_raises(tmp_path: Path):
    with pytest.raises(SystemExit):
        migrate(tmp_path / "nope.db", tmp_path / "vault", apply=False)
