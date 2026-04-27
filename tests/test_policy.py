"""Policy gate tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from brain.policy import Decision, Policy


def test_allow_listed_action():
    p = Policy()
    r = p.check("file_read", path=str(Path.home() / "x.txt"))
    assert r.decision is Decision.ALLOW


def test_unknown_action_denied():
    p = Policy()
    r = p.check("delete_universe")
    assert r.decision is Decision.DENY


def test_confirm_action():
    p = Policy()
    r = p.check("keyboard_type", text="hi")
    assert r.decision is Decision.CONFIRM


def test_path_outside_allowed_roots_denied(tmp_path: Path):
    p = Policy(allowed_roots=[tmp_path])
    target = tmp_path.parent / "elsewhere.txt"
    r = p.check("file_read", path=str(target))
    assert r.decision is Decision.DENY


def test_path_inside_allowed_root_passes(tmp_path: Path):
    p = Policy(allowed_roots=[tmp_path])
    target = tmp_path / "child.txt"
    r = p.check("file_read", path=str(target))
    assert r.decision is Decision.ALLOW


def test_deterministic_deny_blocks_rm_rf():
    p = Policy()
    p.allowed_actions.add("terminal_spawn")
    r = p.check("terminal_spawn", command="rm -rf /")
    assert r.decision is Decision.DENY


def test_deterministic_deny_works_even_with_extra_args():
    p = Policy()
    p.allowed_actions.add("terminal_spawn")
    r = p.check("terminal_spawn", command="bash -c 'rm -rf /'")
    assert r.decision is Decision.DENY
