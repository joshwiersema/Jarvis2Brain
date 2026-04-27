from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from brain.server import create_app


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    return TestClient(create_app(tmp_path / "vault"))


class TestNotesCRUD:
    def test_list_empty(self, client: TestClient) -> None:
        r = client.get("/notes")
        assert r.status_code == 200
        assert r.json() == []

    def test_create_then_get(self, client: TestClient) -> None:
        r = client.post(
            "/notes",
            json={"slug": "hello", "title": "Hello", "body": "hi\n", "tags": ["t1"]},
        )
        assert r.status_code == 201
        body = r.json()
        assert body["slug"] == "hello"
        assert body["title"] == "Hello"
        assert body["tags"] == ["t1"]

        r = client.get("/notes/hello")
        assert r.status_code == 200
        assert r.json()["body"] == "hi\n"

    def test_create_invalid_slug(self, client: TestClient) -> None:
        r = client.post("/notes", json={"slug": "BAD SLUG", "title": "x"})
        assert r.status_code == 422

    def test_create_duplicate(self, client: TestClient) -> None:
        client.post("/notes", json={"slug": "dup", "title": "x"})
        r = client.post("/notes", json={"slug": "dup", "title": "y"})
        assert r.status_code == 409

    def test_read_missing(self, client: TestClient) -> None:
        assert client.get("/notes/ghost").status_code == 404

    def test_update(self, client: TestClient) -> None:
        client.post("/notes", json={"slug": "u", "title": "Old"})
        r = client.put("/notes/u", json={"title": "New", "body": "new\n"})
        assert r.status_code == 200
        assert r.json()["title"] == "New"
        assert client.get("/notes/u").json()["body"] == "new\n"

    def test_update_missing(self, client: TestClient) -> None:
        assert client.put("/notes/ghost", json={"title": "x"}).status_code == 404

    def test_delete(self, client: TestClient) -> None:
        client.post("/notes", json={"slug": "d", "title": "D"})
        assert client.delete("/notes/d").status_code == 204
        assert client.get("/notes/d").status_code == 404

    def test_delete_missing(self, client: TestClient) -> None:
        assert client.delete("/notes/ghost").status_code == 404


class TestSearchAndLinks:
    def test_search(self, client: TestClient) -> None:
        client.post("/notes", json={"slug": "a", "title": "Alpha", "body": "needle here"})
        client.post("/notes", json={"slug": "b", "title": "Beta", "body": "no match"})
        r = client.get("/search", params={"q": "needle"})
        assert r.status_code == 200
        assert [n["slug"] for n in r.json()] == ["a"]

    def test_search_empty_lists_all(self, client: TestClient) -> None:
        client.post("/notes", json={"slug": "a", "title": "A"})
        client.post("/notes", json={"slug": "b", "title": "B"})
        r = client.get("/search")
        assert sorted(n["slug"] for n in r.json()) == ["a", "b"]

    def test_links(self, client: TestClient) -> None:
        client.post("/notes", json={"slug": "src", "title": "S", "body": "see [[dst]]"})
        client.post("/notes", json={"slug": "dst", "title": "D", "body": "hi"})
        client.post(
            "/notes", json={"slug": "other", "title": "O", "body": "also [[dst]]"}
        )
        r = client.get("/notes/dst/links")
        assert r.status_code == 200
        body = r.json()
        assert body["outgoing"] == []
        assert sorted(body["incoming"]) == ["other", "src"]

    def test_links_missing_note(self, client: TestClient) -> None:
        assert client.get("/notes/ghost/links").status_code == 404


class TestGraphEndpoints:
    def test_graph_json_empty(self, client: TestClient) -> None:
        r = client.get("/graph.json")
        assert r.status_code == 200
        assert r.json() == {"nodes": [], "edges": []}

    def test_graph_json_with_data(self, client: TestClient) -> None:
        client.post(
            "/notes", json={"slug": "hub", "title": "Hub", "body": "see [[leaf]]"}
        )
        client.post("/notes", json={"slug": "leaf", "title": "Leaf", "body": ""})
        r = client.get("/graph.json")
        assert r.status_code == 200
        body = r.json()
        node_ids = sorted(n["id"] for n in body["nodes"])
        assert node_ids == ["hub", "leaf"]
        assert any(
            e["from"] == "hub" and e["to"] == "leaf" for e in body["edges"]
        )

    def test_graph_json_includes_ghost_nodes(self, client: TestClient) -> None:
        client.post(
            "/notes", json={"slug": "src", "title": "S", "body": "[[missing]]"}
        )
        body = client.get("/graph.json").json()
        ghost = next(n for n in body["nodes"] if n["id"] == "missing")
        assert ghost["ghost"] is True

    def test_graph_html_served(self, client: TestClient) -> None:
        r = client.get("/graph")
        assert r.status_code == 200
        assert "text/html" in r.headers["content-type"]
        assert "vis-network" in r.text
        # v0.2: graph fetches data over /graph/ws (initial snapshot) instead.
        assert "/graph/ws" in r.text

    def test_root_serves_graph(self, client: TestClient) -> None:
        r = client.get("/")
        assert r.status_code == 200
        assert "vis-network" in r.text
