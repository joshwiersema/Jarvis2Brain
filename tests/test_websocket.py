"""WebSocket /graph/ws integration test."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from brain.server import create_app


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    return TestClient(create_app(tmp_path))


def test_ws_sends_initial_snapshot(client: TestClient):
    with client.websocket_connect("/graph/ws") as ws:
        msg = ws.receive_json()
        assert msg["type"] == "graph.snapshot"
        assert "nodes" in msg["payload"] and "edges" in msg["payload"]


def test_ws_broadcasts_new_node(client: TestClient):
    with client.websocket_connect("/graph/ws") as ws:
        ws.receive_json()  # initial snapshot
        client.post("/notes", json={"slug": "live", "title": "Live", "body": "x"})
        msg = ws.receive_json()
        assert msg["type"] == "node.created"
        assert msg["payload"]["id"] == "live"
        assert msg["payload"]["xy"] is not None


def test_ws_broadcasts_update_and_delete(client: TestClient):
    with client.websocket_connect("/graph/ws") as ws:
        ws.receive_json()  # snapshot
        client.post("/notes", json={"slug": "a", "title": "A"})
        ws.receive_json()  # node.created
        client.put("/notes/a", json={"title": "B"})
        upd = ws.receive_json()
        assert upd["type"] == "node.updated"
        assert upd["payload"]["title"] == "B"
        client.delete("/notes/a")
        d = ws.receive_json()
        assert d["type"] == "node.deleted"
        assert d["payload"]["id"] == "a"


def test_ws_broadcasts_training_epoch(client: TestClient):
    with client.websocket_connect("/graph/ws") as ws:
        ws.receive_json()
        client.post("/notes", json={"slug": "a", "title": "A", "body": "x"})
        ws.receive_json()
        client.post("/notes", json={"slug": "b", "title": "B", "body": "y"})
        ws.receive_json()
        client.post("/train?epochs=1")
        # Drain events; first one should be training.epoch.
        ev = ws.receive_json()
        assert ev["type"] == "training.epoch"
        assert "summaries" in ev["payload"]
