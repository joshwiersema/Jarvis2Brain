"""Vault-as-git-repo versioning tests.

Skipped when git isn't available on the host (CI minimal images).
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from brain.git_versioning import (
    commit_change,
    history,
    init_repo,
    is_git_repo,
    rollback,
)


pytestmark = pytest.mark.skipif(
    shutil.which("git") is None, reason="git not on PATH"
)


def _seed_repo(tmp_path: Path) -> Path:
    init_repo(tmp_path)
    # Configure committer for the test repo.
    import subprocess

    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=str(tmp_path))
    subprocess.run(["git", "config", "user.name", "Test"], cwd=str(tmp_path))
    return tmp_path


def test_init_repo(tmp_path: Path):
    assert init_repo(tmp_path) is True
    assert is_git_repo(tmp_path)


def test_init_repo_idempotent(tmp_path: Path):
    init_repo(tmp_path)
    assert init_repo(tmp_path) is False


def test_commit_change_returns_sha(tmp_path: Path):
    _seed_repo(tmp_path)
    (tmp_path / "x.md").write_text("v1", encoding="utf-8")
    sha = commit_change(tmp_path, "feat: add x")
    assert sha is not None
    assert len(sha) == 40


def test_history_lists_commits(tmp_path: Path):
    _seed_repo(tmp_path)
    (tmp_path / "x.md").write_text("v1", encoding="utf-8")
    commit_change(tmp_path, "v1")
    (tmp_path / "x.md").write_text("v2", encoding="utf-8")
    commit_change(tmp_path, "v2")
    h = history(tmp_path, "x")
    assert len(h) >= 2


def test_no_repo_returns_empty(tmp_path: Path):
    assert history(tmp_path, "x") == []
    assert rollback(tmp_path, "x", "deadbeef") is False
