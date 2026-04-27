"""Hash-chained audit log tests."""

from __future__ import annotations

import json
from pathlib import Path

from brain.audit import append, verify


def test_append_creates_file_and_first_entry(tmp_path: Path):
    p = tmp_path / "audit.jsonl"
    e = append(p, action="x", args={"k": 1}, decision="allow", outcome="ok")
    assert e.prev_hash == ""
    assert p.exists()
    line = p.read_text(encoding="utf-8").strip()
    obj = json.loads(line)
    assert obj["action"] == "x"


def test_chain_links_entries(tmp_path: Path):
    p = tmp_path / "audit.jsonl"
    e1 = append(p, action="a", args={}, decision="allow", outcome="ok")
    e2 = append(p, action="b", args={}, decision="allow", outcome="ok")
    assert e2.prev_hash == e1.hash


def test_verify_passes_on_clean_chain(tmp_path: Path):
    p = tmp_path / "audit.jsonl"
    for i in range(5):
        append(p, action="x", args={"i": i}, decision="allow", outcome="ok")
    ok, idx, msg = verify(p)
    assert ok is True
    assert idx is None


def test_verify_detects_tampering(tmp_path: Path):
    p = tmp_path / "audit.jsonl"
    for i in range(3):
        append(p, action="x", args={"i": i}, decision="allow", outcome="ok")
    # Corrupt the middle line.
    lines = p.read_text(encoding="utf-8").splitlines()
    obj = json.loads(lines[1])
    obj["args"] = {"i": 999}
    lines[1] = json.dumps(obj)
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    ok, idx, msg = verify(p)
    assert ok is False
    assert idx == 1


def test_redacts_secrets(tmp_path: Path):
    p = tmp_path / "audit.jsonl"
    append(p, action="x", args={"password": "shh", "api_key": "k"}, decision="allow", outcome="ok")
    text = p.read_text(encoding="utf-8")
    assert "shh" not in text and "[redacted]" in text


def test_verify_on_missing_file_passes(tmp_path: Path):
    ok, _, _ = verify(tmp_path / "missing.jsonl")
    assert ok is True
