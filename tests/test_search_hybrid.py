"""Hybrid search via the HTTP server."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from brain.server import create_app


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    return TestClient(create_app(tmp_path))


def test_substring_search_default(client: TestClient):
    client.post("/notes", json={"slug": "a", "title": "Pizza", "body": "dough"})
    client.post("/notes", json={"slug": "b", "title": "Calc", "body": "integral"})
    r = client.get("/search?q=pizza")
    assert r.status_code == 200
    assert {x["slug"] for x in r.json()} == {"a"}


def test_hybrid_mode_returns_results(client: TestClient):
    client.post("/notes", json={"slug": "a", "title": "Pizza", "body": "dough recipe"})
    client.post("/notes", json={"slug": "b", "title": "Calc", "body": "integrals"})
    r = client.get("/search?q=dough&mode=hybrid")
    assert r.status_code == 200
    slugs = [x["slug"] for x in r.json()]
    assert "a" in slugs


def test_kind_filter(client: TestClient):
    client.post("/notes", json={"slug": "a", "title": "Note", "body": "content"})
    # New notes default to kind=unsorted via the chaos start.
    r = client.get("/search?q=content&mode=hybrid&kind=unsorted")
    assert r.status_code == 200
    assert any(x["slug"] == "a" for x in r.json())


def test_reindex_endpoint(client: TestClient):
    client.post("/notes", json={"slug": "x", "title": "X", "body": "text"})
    r = client.post("/reindex")
    assert r.status_code == 200
    assert r.json()["indexed"] >= 1
