"""HTTP /skills endpoint tests."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from brain.server import create_app


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    return TestClient(create_app(tmp_path))


def test_list_skills(client: TestClient):
    r = client.get("/skills")
    assert r.status_code == 200
    names = {s["name"] for s in r.json()["skills"]}
    assert "memory_search" in names
    assert "notes_create" in names


def test_invoke_skill(client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("BRAIN_VAULT", str(tmp_path))
    r = client.post(
        "/skills/notes_create",
        json={"slug": "skill-test", "title": "From skill", "body": "hi"},
    )
    assert r.status_code == 200
    assert r.json()["result"]["slug"] == "skill-test"


def test_invoke_unknown_skill_404(client: TestClient):
    r = client.post("/skills/totally_made_up", json={})
    assert r.status_code == 404


def test_invoke_skill_bad_args_422(client: TestClient):
    r = client.post("/skills/notes_read", json={"wrong_arg": 1})
    assert r.status_code == 422
