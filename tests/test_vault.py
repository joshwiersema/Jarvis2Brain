from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from brain.note import Note, now_utc
from brain.vault import (
    NoteExistsError,
    NoteNotFoundError,
    Vault,
    resolve_vault_path,
)


@pytest.fixture
def vault(tmp_path: Path) -> Vault:
    return Vault(tmp_path / "vault")


def make_note(slug: str = "n", body: str = "hello\n") -> Note:
    ts = now_utc()
    return Note(slug=slug, title="N", created=ts, updated=ts, tags=[], body=body)


class TestResolution:
    def test_explicit_flag_wins(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("BRAIN_VAULT", str(tmp_path / "from_env"))
        result = resolve_vault_path(
            cli=str(tmp_path / "from_cli"), cwd=tmp_path
        )
        assert result == (tmp_path / "from_cli").resolve()

    def test_env_when_no_flag(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("BRAIN_VAULT", str(tmp_path / "from_env"))
        assert resolve_vault_path(cli=None, cwd=tmp_path) == (
            tmp_path / "from_env"
        ).resolve()

    def test_default_when_neither(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("BRAIN_VAULT", raising=False)
        assert resolve_vault_path(cli=None, cwd=tmp_path) == (
            tmp_path / "vault"
        ).resolve()


class TestCRUD:
    def test_create_then_read(self, vault: Vault) -> None:
        n = make_note("first", "body\n")
        vault.create(n)
        loaded = vault.read("first")
        assert loaded.title == "N"
        assert loaded.body == "body\n"

    def test_create_duplicate_rejected(self, vault: Vault) -> None:
        vault.create(make_note("dup"))
        with pytest.raises(NoteExistsError):
            vault.create(make_note("dup"))

    def test_read_missing(self, vault: Vault) -> None:
        with pytest.raises(NoteNotFoundError):
            vault.read("ghost")

    def test_update_changes_body_and_bumps_updated(self, vault: Vault) -> None:
        n = make_note("u", "old\n")
        vault.create(n)
        original_updated = vault.read("u").updated
        vault.update("u", body="new body\n")
        loaded = vault.read("u")
        assert loaded.body == "new body\n"
        assert loaded.updated >= original_updated
        # created must not change
        assert loaded.created == n.created

    def test_update_missing(self, vault: Vault) -> None:
        with pytest.raises(NoteNotFoundError):
            vault.update("ghost", body="x")

    def test_delete(self, vault: Vault) -> None:
        vault.create(make_note("d"))
        vault.delete("d")
        with pytest.raises(NoteNotFoundError):
            vault.read("d")

    def test_delete_missing(self, vault: Vault) -> None:
        with pytest.raises(NoteNotFoundError):
            vault.delete("ghost")

    def test_list_empty(self, vault: Vault) -> None:
        assert vault.list() == []

    def test_list_returns_summaries(self, vault: Vault) -> None:
        vault.create(make_note("a"))
        vault.create(make_note("b"))
        slugs = sorted(s.slug for s in vault.list())
        assert slugs == ["a", "b"]

    def test_list_filters_by_tag(self, vault: Vault) -> None:
        ts = now_utc()
        vault.create(
            Note(slug="x", title="X", created=ts, updated=ts, tags=["alpha"])
        )
        vault.create(
            Note(slug="y", title="Y", created=ts, updated=ts, tags=["beta"])
        )
        result = [s.slug for s in vault.list(tag="alpha")]
        assert result == ["x"]


class TestAtomicWrite:
    def test_crash_mid_write_does_not_corrupt(self, vault: Vault) -> None:
        vault.create(make_note("safe", "original\n"))
        # Simulate crash: os.replace fails after temp file is written.
        with patch("brain.vault.os.replace", side_effect=RuntimeError("boom")):
            with pytest.raises(RuntimeError):
                vault.update("safe", body="poisoned\n")
        # Original content must still be readable
        assert vault.read("safe").body == "original\n"
        # No leftover temp file in vault
        leftovers = [p.name for p in vault.path.iterdir() if p.suffix == ".tmp"]
        assert leftovers == []

    def test_temp_file_cleaned_after_success(self, vault: Vault) -> None:
        vault.create(make_note("clean", "x\n"))
        leftovers = [p for p in vault.path.iterdir() if p.suffix == ".tmp"]
        assert leftovers == []


class TestLinks:
    def test_links_extracted_on_create(self, vault: Vault) -> None:
        vault.create(make_note("src", "see [[target]] here"))
        assert vault.outgoing_links("src") == ["target"]

    def test_links_updated_when_body_changes(self, vault: Vault) -> None:
        vault.create(make_note("src", "see [[old]]"))
        assert vault.outgoing_links("src") == ["old"]
        vault.update("src", body="see [[new]]")
        assert vault.outgoing_links("src") == ["new"]

    def test_incoming_links(self, vault: Vault) -> None:
        vault.create(make_note("target", "i am the target"))
        vault.create(make_note("a", "ref [[target]]"))
        vault.create(make_note("b", "also [[target]]"))
        assert sorted(vault.incoming_links("target")) == ["a", "b"]


class TestGraph:
    def test_empty(self, vault: Vault) -> None:
        g = vault.graph()
        assert g == {"nodes": [], "edges": []}

    def test_nodes_and_edges(self, vault: Vault) -> None:
        ts = now_utc()
        vault.create(
            Note(
                slug="a",
                title="A",
                created=ts,
                updated=ts,
                tags=["t1"],
                body="link to [[b]] and [[c]]",
            )
        )
        vault.create(
            Note(slug="b", title="B", created=ts, updated=ts, body="see [[c]]")
        )
        vault.create(Note(slug="c", title="C", created=ts, updated=ts, body=""))
        g = vault.graph()
        node_ids = sorted(n["id"] for n in g["nodes"])
        assert node_ids == ["a", "b", "c"]
        edges = sorted((e["from"], e["to"]) for e in g["edges"])
        assert edges == [("a", "b"), ("a", "c"), ("b", "c")]
        a = next(n for n in g["nodes"] if n["id"] == "a")
        assert a["title"] == "A"
        assert a["tags"] == ["t1"]
        assert "ghost" not in a or a["ghost"] is False

    def test_ghost_nodes_for_broken_links(self, vault: Vault) -> None:
        vault.create(make_note("real", "link to [[missing]] note"))
        g = vault.graph()
        ghost = next(n for n in g["nodes"] if n["id"] == "missing")
        assert ghost["ghost"] is True
        assert any(e["from"] == "real" and e["to"] == "missing" for e in g["edges"])

    def test_preview_truncated(self, vault: Vault) -> None:
        long_body = "x" * 500
        vault.create(make_note("big", long_body))
        g = vault.graph()
        n = next(n for n in g["nodes"] if n["id"] == "big")
        assert len(n["preview"]) <= 240
