"""Migrate a v0.1 flat vault into the v0.2 cognitive tier layout.

v0.1: vault/<slug>.md (flat).
v0.2: vault/semantic/notes/<slug>.md, plus episodic/, procedural/, core/, ...

Idempotent (content-hash check). Dry-run by default. Refuses to silently
overwrite a target with conflicting content — that requires manual review.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Optional

import yaml

from brain.note import parse_note, serialize_note

_TIERS = (
    "episodic",
    "episodic/inbox",
    "semantic",
    "semantic/notes",
    "semantic/people",
    "semantic/concepts",
    "procedural",
    "procedural/skills",
    "core",
    "reflections",
    "proposals",
    ".brain",
    ".brain/skills",
    ".brain/cache",
    ".brain/models",
)

_TARGET_SUBDIR = "semantic/notes"


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _is_legacy_flat_note(path: Path, vault: Path) -> bool:
    return path.parent == vault


def migrate(vault: Path, *, apply: bool = False) -> dict:
    vault = Path(vault)
    if not vault.exists():
        raise SystemExit(f"vault does not exist: {vault}")

    moves: list[dict] = []
    skipped: list[dict] = []

    target_dir = vault / _TARGET_SUBDIR

    for src in sorted(vault.glob("*.md")):
        if not _is_legacy_flat_note(src, vault):
            continue
        slug = src.stem
        try:
            src_text = src.read_text(encoding="utf-8")
            note = parse_note(slug, src_text)
        except Exception as e:
            skipped.append({"slug": slug, "reason": f"unparseable: {e}"})
            continue

        normalized = serialize_note(note)
        dest = target_dir / f"{slug}.md"

        if dest.exists():
            existing = dest.read_text(encoding="utf-8")
            if _hash(existing) == _hash(normalized) or _hash(existing) == _hash(src_text):
                # Already migrated identically — just remove the flat duplicate.
                moves.append({"slug": slug, "src": str(src), "dest": str(dest), "op": "remove_dup"})
                if apply:
                    src.unlink()
                continue
            raise SystemExit(
                f"refusing to overwrite {dest} — content differs from {src}. "
                f"Resolve manually."
            )

        moves.append({"slug": slug, "src": str(src), "dest": str(dest), "op": "move"})

    if apply:
        for sub in _TIERS:
            (vault / sub).mkdir(parents=True, exist_ok=True)
        for m in moves:
            if m["op"] == "remove_dup":
                continue
            src = Path(m["src"])
            dest = Path(m["dest"])
            note = parse_note(src.stem, src.read_text(encoding="utf-8"))
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(serialize_note(note), encoding="utf-8")
            src.unlink()

    return {"moves": moves, "skipped": skipped, "applied": apply}


def _main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Migrate v0.1 flat vault -> v0.2 tiers")
    p.add_argument("vault", type=Path, help="Path to the vault directory")
    p.add_argument(
        "--apply",
        action="store_true",
        help="Actually write changes. Default is dry-run.",
    )
    args = p.parse_args(argv)
    plan = migrate(args.vault, apply=args.apply)
    print(yaml.safe_dump(plan, sort_keys=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
