from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from brain.cli import main


def run(
    argv: list[str], *, vault: Path, stdin: str = ""
) -> tuple[int, str, str]:
    out = io.StringIO()
    err = io.StringIO()
    rc = main(
        ["--vault", str(vault), *argv],
        stdin=io.StringIO(stdin),
        stdout=out,
        stderr=err,
    )
    return rc, out.getvalue(), err.getvalue()


class TestCreateAndRead:
    def test_create_then_read(self, tmp_path: Path) -> None:
        vault = tmp_path / "v"
        rc, out, _ = run(
            ["create", "hello", "--title", "Hello", "--body", "hi"], vault=vault
        )
        assert rc == 0
        payload = json.loads(out)
        assert payload["slug"] == "hello"
        assert payload["body"] == "hi"

        rc, out, _ = run(["read", "hello"], vault=vault)
        assert rc == 0
        assert json.loads(out)["title"] == "Hello"

    def test_create_with_stdin_body(self, tmp_path: Path) -> None:
        rc, out, _ = run(
            ["create", "x", "--title", "X", "--body", "-"],
            vault=tmp_path / "v",
            stdin="from stdin\n",
        )
        assert rc == 0
        assert json.loads(out)["body"] == "from stdin\n"

    def test_create_invalid_slug(self, tmp_path: Path) -> None:
        rc, _, err = run(
            ["create", "BAD", "--title", "x"], vault=tmp_path / "v"
        )
        assert rc == 4
        assert "slug" in err

    def test_read_missing(self, tmp_path: Path) -> None:
        rc, _, err = run(["read", "ghost"], vault=tmp_path / "v")
        assert rc == 2
        assert "not found" in err


class TestUpdateDelete:
    def test_update_body(self, tmp_path: Path) -> None:
        v = tmp_path / "v"
        run(["create", "u", "--title", "U", "--body", "old"], vault=v)
        rc, out, _ = run(["update", "u", "--body", "new"], vault=v)
        assert rc == 0
        assert json.loads(out)["body"] == "new"

    def test_delete(self, tmp_path: Path) -> None:
        v = tmp_path / "v"
        run(["create", "d", "--title", "D"], vault=v)
        rc, _, _ = run(["delete", "d"], vault=v)
        assert rc == 0
        rc, _, _ = run(["read", "d"], vault=v)
        assert rc == 2


class TestListSearch:
    def test_list_and_search(self, tmp_path: Path) -> None:
        v = tmp_path / "v"
        run(["create", "a", "--title", "Alpha", "--body", "needle"], vault=v)
        run(["create", "b", "--title", "Beta", "--body", "haystack"], vault=v)

        rc, out, _ = run(["list"], vault=v)
        assert rc == 0
        slugs = sorted(line.split("\t")[0] for line in out.strip().splitlines())
        assert slugs == ["a", "b"]

        rc, out, _ = run(["search", "needle"], vault=v)
        assert rc == 0
        assert out.strip().split("\t")[0] == "a"

    def test_list_with_tag_filter(self, tmp_path: Path) -> None:
        v = tmp_path / "v"
        run(["create", "x", "--title", "X", "--tag", "work"], vault=v)
        run(["create", "y", "--title", "Y", "--tag", "personal"], vault=v)
        rc, out, _ = run(["list", "--tag", "work"], vault=v)
        assert rc == 0
        slugs = [line.split("\t")[0] for line in out.strip().splitlines()]
        assert slugs == ["x"]


class TestVaultResolution:
    def test_env_var(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        target = tmp_path / "envvault"
        monkeypatch.setenv("BRAIN_VAULT", str(target))
        out = io.StringIO()
        rc = main(
            ["create", "z", "--title", "Z"],
            stdin=io.StringIO(""),
            stdout=out,
            stderr=io.StringIO(),
        )
        assert rc == 0
        # v0.2: new notes land in episodic/inbox/ by default.
        assert (target / "episodic" / "inbox" / "z.md").exists()
