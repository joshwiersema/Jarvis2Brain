"""Migrate a Jarvis 2 SQLite database into the Brain vault.

Run by the parallel Jarvis 2 session when it reaches its P3, or manually
once the DB is available. Idempotent (content-hash on session_id+turn_id),
dry-run by default. Refuses to write anything destructive.

Assumed Jarvis 2 schema (override with env vars if it differs):

    conversations(session_id TEXT, turn_id TEXT, role TEXT, model TEXT,
                  content TEXT, created TEXT, tokens INTEGER NULL)
    preferences(slug TEXT, name TEXT, value TEXT, updated TEXT)
    feedback(session_id TEXT, turn_id TEXT, kind TEXT, content TEXT,
             created TEXT)

The Jarvis 2 prompt commits to producing a DB with these tables (or pointing
this script at a view that adapts to whatever Jarvis 2 actually persisted).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

import yaml

from brain.note import Note, parse_note, serialize_note
from brain.slug import validate_slug


@dataclass
class MigratedRow:
    target_path: Path
    note: Note


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _safe_slug(parts: Iterable[str]) -> str:
    """Collapse a list of identifier parts into a vault-legal slug."""
    raw = "-".join(str(p).lower() for p in parts if p)
    out = []
    for ch in raw:
        if ch.isalnum():
            out.append(ch)
        elif ch in "-_":
            out.append("-")
    s = "".join(out).strip("-")[:80] or "row"
    while "--" in s:
        s = s.replace("--", "-")
    if not s[0].isalpha():
        s = "x-" + s[: 79 - 2]
    validate_slug(s)
    return s


def _coerce_dt(value: Optional[str]) -> datetime:
    if value is None or value == "":
        return datetime.now(timezone.utc).replace(microsecond=0)
    try:
        dt = datetime.fromisoformat(str(value))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except ValueError:
        return datetime.now(timezone.utc).replace(microsecond=0)


def _conversation_rows(conn: sqlite3.Connection) -> list[dict]:
    cur = conn.execute(
        "SELECT session_id, turn_id, role, model, content, created, "
        "       COALESCE(tokens, 0) AS tokens "
        "FROM conversations ORDER BY created"
    )
    return [dict(r) for r in cur.fetchall()]


def _preference_rows(conn: sqlite3.Connection) -> list[dict]:
    try:
        cur = conn.execute(
            "SELECT slug, name, value, updated FROM preferences ORDER BY updated"
        )
        return [dict(r) for r in cur.fetchall()]
    except sqlite3.OperationalError:
        return []


def _feedback_rows(conn: sqlite3.Connection) -> list[dict]:
    try:
        cur = conn.execute(
            "SELECT session_id, turn_id, kind, content, created "
            "FROM feedback ORDER BY created"
        )
        return [dict(r) for r in cur.fetchall()]
    except sqlite3.OperationalError:
        return []


def _build_conversation(row: dict) -> tuple[Path, Note]:
    session = row["session_id"]
    turn = row["turn_id"]
    slug = _safe_slug([session, turn])
    ts = _coerce_dt(row.get("created"))
    title = f"{row.get('role', 'turn')} {turn}"
    body_lines = [str(row.get("content", ""))]
    note = Note(
        slug=slug,
        title=title,
        created=ts,
        updated=ts,
        tags=[f"session:{session}", f"role:{row.get('role', 'turn')}"],
        body="\n".join(body_lines).rstrip() + "\n",
        kind="conversation",
        importance=0.4,
    )
    target = Path("episodic") / "conversations" / _safe_slug([session]) / f"{slug}.md"
    return target, note


def _build_preference(row: dict) -> tuple[Path, Note]:
    slug = _safe_slug([row["slug"]])
    ts = _coerce_dt(row.get("updated"))
    note = Note(
        slug=slug,
        title=str(row.get("name") or row["slug"]),
        created=ts,
        updated=ts,
        tags=["preference"],
        body=str(row.get("value", "")).rstrip() + "\n",
        kind="preference",
        importance=0.7,
    )
    target = Path("core") / "preferences" / f"{slug}.md"
    return target, note


def _build_feedback(row: dict) -> tuple[Path, Note]:
    slug = _safe_slug([row.get("session_id"), row.get("turn_id"), row.get("kind", "")])
    ts = _coerce_dt(row.get("created"))
    note = Note(
        slug=slug,
        title=f"feedback {row.get('kind') or ''}".strip(),
        created=ts,
        updated=ts,
        tags=[f"session:{row.get('session_id', '?')}", "feedback"],
        body=str(row.get("content", "")).rstrip() + "\n",
        kind="feedback",
        importance=0.6,
    )
    target = Path("episodic") / "feedback" / f"{slug}.md"
    return target, note


def migrate(source: Path, vault: Path, *, apply: bool = False) -> dict:
    source = Path(source)
    vault = Path(vault)
    if not source.exists():
        raise SystemExit(f"source DB does not exist: {source}")

    conn = sqlite3.connect(str(source))
    conn.row_factory = sqlite3.Row

    plan: list[dict] = []
    skipped: list[dict] = []
    rows: list[tuple[Path, Note]] = []

    try:
        for r in _conversation_rows(conn):
            try:
                rows.append(_build_conversation(r))
            except Exception as e:
                skipped.append({"reason": f"conv build failed: {e}"})
        for r in _preference_rows(conn):
            try:
                rows.append(_build_preference(r))
            except Exception as e:
                skipped.append({"reason": f"pref build failed: {e}"})
        for r in _feedback_rows(conn):
            try:
                rows.append(_build_feedback(r))
            except Exception as e:
                skipped.append({"reason": f"fb build failed: {e}"})
    finally:
        conn.close()

    for rel, note in rows:
        dest = vault / rel
        new_text = serialize_note(note)
        op = "create"
        if dest.exists():
            existing = dest.read_text(encoding="utf-8")
            if _hash(existing) == _hash(new_text):
                op = "skip_dup"
            else:
                op = "skip_conflict"
        plan.append({"op": op, "slug": note.slug, "kind": note.kind, "dest": str(dest)})

    if apply:
        for rel, note in rows:
            dest = vault / rel
            new_text = serialize_note(note)
            if dest.exists():
                existing = dest.read_text(encoding="utf-8")
                if _hash(existing) == _hash(new_text):
                    continue
                if _hash(existing) != _hash(new_text):
                    # Conflict — leave existing untouched, log it.
                    skipped.append({"slug": note.slug, "reason": "content_conflict"})
                    continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(new_text, encoding="utf-8")

    return {"plan": plan, "skipped": skipped, "applied": apply, "n_rows": len(rows)}


def _main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Migrate Jarvis 2 SQLite -> Brain vault")
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--vault", type=Path, required=True)
    p.add_argument("--apply", action="store_true", help="Default is dry-run")
    args = p.parse_args(argv)
    plan = migrate(args.source, args.vault, apply=args.apply)
    print(yaml.safe_dump(plan, sort_keys=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
