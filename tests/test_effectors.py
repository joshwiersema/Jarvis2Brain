"""Effector skill tests — file ops, terminal, claude_code stub."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from brain.policy import Policy
import brain.policy as policy_mod
import brain.skill as skill_mod
from brain.skills.effectors.claude_code import claude_code_spawn
from brain.skills.effectors.file_ops import file_read, file_write
from brain.skills.effectors.terminal import terminal_spawn


@pytest.fixture
def isolated_vault(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("BRAIN_VAULT", str(tmp_path))
    return tmp_path


@pytest.fixture
def open_policy(monkeypatch: pytest.MonkeyPatch, isolated_vault: Path) -> Policy:
    """Allowed_roots = vault, allowed actions = the ones we test."""
    p = Policy(allowed_roots=[isolated_vault])
    p.allowed_actions = {"file_read", "file_write", "terminal_spawn"}
    monkeypatch.setattr(policy_mod, "_DEFAULT", p)
    return p


def test_file_write_then_read(isolated_vault: Path, open_policy: Policy):
    target = isolated_vault / "f.txt"
    out = file_write(path=str(target), content="hello")
    assert "bytes" in out and out["bytes"] == 5
    out2 = file_read(path=str(target))
    assert out2["content"] == "hello"


def test_file_write_outside_root_denied(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    inside = tmp_path / "inside"
    inside.mkdir()
    monkeypatch.setenv("BRAIN_VAULT", str(inside))
    p = Policy(allowed_roots=[inside])
    p.allowed_actions = {"file_write"}
    monkeypatch.setattr(policy_mod, "_DEFAULT", p)

    out = file_write(path=str(tmp_path / "outside.txt"), content="x")
    assert out.get("decision") == "deny"


def test_file_write_no_overwrite_default(isolated_vault: Path, open_policy: Policy):
    target = isolated_vault / "x.txt"
    target.write_text("orig", encoding="utf-8")
    out = file_write(path=str(target), content="new")
    assert "error" in out
    assert target.read_text(encoding="utf-8") == "orig"


def test_file_write_overwrite_true(isolated_vault: Path, open_policy: Policy):
    target = isolated_vault / "x.txt"
    target.write_text("orig", encoding="utf-8")
    file_write(path=str(target), content="new", overwrite=True)
    assert target.read_text(encoding="utf-8") == "new"


def test_terminal_spawn_runs(isolated_vault: Path, open_policy: Policy):
    import sys

    cmd = f'"{sys.executable}" -c "print(2+2)"'
    out = terminal_spawn(command=cmd, timeout=10)
    assert out.get("exit_code") == 0
    assert "4" in (out.get("stdout") or "")


def test_terminal_spawn_deterministic_deny():
    p = Policy()
    p.allowed_actions = {"terminal_spawn"}
    import brain.policy as pm

    pm._DEFAULT = p
    out = terminal_spawn(command="rm -rf /")
    assert out.get("decision") == "deny"


def test_claude_code_spawn_returns_confirm_by_default():
    out = claude_code_spawn(prompt="hi")
    assert out.get("decision") == "confirm"
