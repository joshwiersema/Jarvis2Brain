"""Constitution seeding tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from brain.loops.constitution import ensure_constitution, read_constitution
from brain.vault import Vault


def test_seeds_when_missing(tmp_path: Path):
    v = Vault(tmp_path)
    note = ensure_constitution(v)
    assert note.kind == "constitution"
    assert (tmp_path / "core" / "constitution.md").exists()


def test_idempotent(tmp_path: Path):
    v = Vault(tmp_path)
    a = ensure_constitution(v)
    b = ensure_constitution(v)
    assert a.created == b.created


def test_read_returns_body(tmp_path: Path):
    v = Vault(tmp_path)
    body = read_constitution(v)
    assert "Hard rules" in body
