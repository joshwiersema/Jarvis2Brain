"""Daemon lifecycle + sleep cycle tests."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from brain.daemon import BrainDaemon, DaemonConfig
from brain.note import Note, now_utc
from brain.vault import Vault


def _seed(tmp_path: Path) -> None:
    v = Vault(tmp_path)
    ts = now_utc()
    v.create(Note(slug="a", title="A", created=ts, updated=ts, body="alpha"))
    v.create(Note(slug="b", title="B", created=ts, updated=ts, body="beta"))


def test_daemon_starts_and_stops(tmp_path: Path):
    _seed(tmp_path)
    d = BrainDaemon(tmp_path, config=DaemonConfig(sleep_interval_s=10, event_poll_s=10))
    d.start()
    try:
        assert d.status.running is True
        assert {"sleep", "cron", "event"} == set(d.status.threads)
    finally:
        d.stop()
    assert d.status.running is False


def test_run_sleep_cycle_indexes_and_trains(tmp_path: Path):
    _seed(tmp_path)
    d = BrainDaemon(tmp_path, config=DaemonConfig())
    out = d.run_sleep_cycle()
    assert out["indexed"] == 2
    assert len(out["epochs"]) >= 1
    # History file gets written by trainer.
    hist = tmp_path / ".brain" / "training_history.jsonl"
    assert hist.exists()


def test_event_loop_picks_up_new_md(tmp_path: Path):
    _seed(tmp_path)
    d = BrainDaemon(tmp_path, config=DaemonConfig(event_poll_s=0.1, sleep_interval_s=999))
    d.start()
    try:
        # Add a note directly on disk (bypassing the API).
        new = tmp_path / "episodic" / "inbox"
        new.mkdir(parents=True, exist_ok=True)
        (new / "x.md").write_text(
            "---\ntitle: X\ncreated: 2026-01-01T00:00:00+00:00\n"
            "updated: 2026-01-01T00:00:00+00:00\ntags: []\nkind: unsorted\n---\n"
            "external write\n",
            encoding="utf-8",
        )
        # Wait for the event loop to pick it up.
        deadline = time.time() + 2.0
        while time.time() < deadline:
            if "x" in d.memory.index.all_slugs():
                break
            time.sleep(0.05)
        assert "x" in d.memory.index.all_slugs()
    finally:
        d.stop()
