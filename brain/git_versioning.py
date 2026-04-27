"""Vault-as-git-repo helpers.

Brain auto-promotes changes (xy updates, kind reclassification, edge-link
proposals). Each promoted change is a git commit, so `brain history <slug>`
shows the lineage and `brain rollback <slug> <commit>` reverts cleanly.

The vault stays a regular git repo — users can review, branch, push, etc.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Optional


def is_git_repo(vault: Path) -> bool:
    return (Path(vault) / ".git").exists()


def init_repo(vault: Path) -> bool:
    vault = Path(vault)
    if is_git_repo(vault):
        return False
    subprocess.run(["git", "init", "--quiet"], cwd=str(vault), check=False)
    # Don't fail if git isn't installed or init fails — the rest of brain still works.
    return is_git_repo(vault)


def _git(vault: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args],
        cwd=str(vault),
        capture_output=True,
        text=True,
        check=False,
    )


def commit_change(vault: Path, message: str, paths: Optional[list[str]] = None) -> Optional[str]:
    if not is_git_repo(vault):
        return None
    if paths:
        _git(vault, "add", *paths)
    else:
        _git(vault, "add", "-A")
    res = _git(vault, "commit", "-m", message, "--allow-empty")
    if res.returncode != 0:
        return None
    rev = _git(vault, "rev-parse", "HEAD").stdout.strip()
    return rev or None


def _find_paths(vault: Path, slug: str) -> list[str]:
    matches = list(Path(vault).rglob(f"{slug}.md"))
    return [str(p.relative_to(vault)).replace("\\", "/") for p in matches]


def history(vault: Path, slug: str, limit: int = 20) -> list[dict]:
    if not is_git_repo(vault):
        return []
    paths = _find_paths(vault, slug)
    args = [
        "log",
        f"--max-count={limit}",
        "--format=%H%x09%ad%x09%s",
        "--date=iso",
    ]
    if paths:
        args += ["--", *paths]
    res = _git(vault, *args)
    out: list[dict] = []
    for line in res.stdout.splitlines():
        parts = line.split("\t")
        if len(parts) >= 3:
            out.append({"commit": parts[0], "date": parts[1], "subject": parts[2]})
    return out


def rollback(vault: Path, slug: str, commit: str) -> bool:
    if not is_git_repo(vault):
        return False
    paths = _find_paths(vault, slug)
    if not paths:
        return False
    res = _git(vault, "checkout", commit, "--", *paths)
    return res.returncode == 0
