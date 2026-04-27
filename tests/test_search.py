from __future__ import annotations

from pathlib import Path

import pytest

from brain.note import Note, now_utc
from brain.search import search
from brain.vault import Vault


@pytest.fixture
def vault(tmp_path: Path) -> Vault:
    v = Vault(tmp_path / "vault")
    ts = now_utc()
    v.create(Note(slug="alpha", title="Alpha thing", created=ts, updated=ts, body="contains the keyword foo"))
    v.create(Note(slug="beta", title="Beta", created=ts, updated=ts, tags=["work"], body="other body"))
    v.create(Note(slug="gamma", title="Gamma FOO bar", created=ts, updated=ts, tags=["work"], body=""))
    return v


class TestSearch:
    def test_match_in_body(self, vault: Vault) -> None:
        assert [s.slug for s in search(vault, "foo")] == ["alpha", "gamma"]

    def test_case_insensitive(self, vault: Vault) -> None:
        assert sorted(s.slug for s in search(vault, "FOO")) == ["alpha", "gamma"]

    def test_match_in_title_only(self, vault: Vault) -> None:
        assert [s.slug for s in search(vault, "Beta")] == ["beta"]

    def test_no_match(self, vault: Vault) -> None:
        assert search(vault, "nonexistent") == []

    def test_empty_query_returns_all(self, vault: Vault) -> None:
        assert sorted(s.slug for s in search(vault, "")) == ["alpha", "beta", "gamma"]

    def test_with_tag_filter(self, vault: Vault) -> None:
        result = [s.slug for s in search(vault, "", tag="work")]
        assert sorted(result) == ["beta", "gamma"]
