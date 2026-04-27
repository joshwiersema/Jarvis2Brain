"""Brain constitution — the always-in-context principles.

Hand-edited (`core/constitution.md`). Auto-promoted changes elsewhere in the
vault never touch this file; the user is the only writer. The weekly review
loop will *propose* edits but never apply them.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from brain.note import Note
from brain.vault import Vault

CONSTITUTION_SLUG = "constitution"

SEED_BODY = """\
# Constitution

These are the rules and principles the brain operates under. The user is
the only person who edits this file directly. Auto-promoted changes anywhere
else in the vault must not touch the constitution.

## Hard rules

- Never delete notes without writing a `kind=proposal` first.
- Never bypass `brain.policy.Policy.check`.
- Never modify this file via auto-promotion.
- Every effector call must be logged to `.brain/audit.jsonl`.

## Behavioral principles

- Local-first. Cloud is an upgrade, never a requirement.
- Terse responses. The user reads the diff.
- Learn from failure artifacts (test output, file diffs), not vibes.
- Prefer surfacing a proposal over silently mutating state.
"""


def ensure_constitution(vault: Vault) -> Note:
    if vault.exists(CONSTITUTION_SLUG):
        return vault.read(CONSTITUTION_SLUG)
    ts = datetime.now(timezone.utc).replace(microsecond=0)
    note = Note(
        slug=CONSTITUTION_SLUG,
        title="Constitution",
        created=ts,
        updated=ts,
        body=SEED_BODY,
        kind="constitution",
        importance=1.0,
    )
    vault.create(note, subdir="core")
    return note


def read_constitution(vault: Vault) -> str:
    if not vault.exists(CONSTITUTION_SLUG):
        ensure_constitution(vault)
    return vault.read(CONSTITUTION_SLUG).body
