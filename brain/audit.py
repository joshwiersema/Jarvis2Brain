"""Hash-chained append-only audit log.

Every effector call writes one entry. Each line includes the SHA-256 of the
prior line, so any tampering breaks the chain — `verify()` walks the file
and reports the first divergence.

Format (one JSON object per line):

    {ts, action, args, decision, outcome, prev_hash, hash}
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass
class AuditEntry:
    ts: float
    action: str
    args: dict
    decision: str
    outcome: str
    prev_hash: str
    hash: str

    @classmethod
    def from_dict(cls, d: dict) -> "AuditEntry":
        return cls(
            ts=float(d["ts"]),
            action=str(d["action"]),
            args=dict(d.get("args") or {}),
            decision=str(d["decision"]),
            outcome=str(d["outcome"]),
            prev_hash=str(d["prev_hash"]),
            hash=str(d["hash"]),
        )


def _redact(args: dict) -> dict:
    """Strip likely-sensitive keys before persisting."""
    out = {}
    for k, v in args.items():
        if k.lower() in ("password", "secret", "token", "api_key", "key"):
            out[k] = "[redacted]"
        else:
            out[k] = v
    return out


def append(
    path: Path,
    *,
    action: str,
    args: dict,
    decision: str,
    outcome: str,
    ts: Optional[float] = None,
) -> AuditEntry:
    import time

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    prev_hash = ""
    if path.exists() and path.stat().st_size > 0:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                last = line
        try:
            prev_hash = json.loads(last)["hash"]
        except Exception:
            prev_hash = ""

    body = {
        "ts": ts if ts is not None else time.time(),
        "action": action,
        "args": _redact(args),
        "decision": decision,
        "outcome": outcome,
        "prev_hash": prev_hash,
    }
    h = _sha(prev_hash + json.dumps(body, sort_keys=True))
    body["hash"] = h
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(body) + "\n")
    return AuditEntry.from_dict(body)


def verify(path: Path) -> tuple[bool, Optional[int], Optional[str]]:
    """Walk the chain. Returns (ok, first_bad_line, message)."""
    path = Path(path)
    if not path.exists():
        return True, None, None
    prev_hash = ""
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                return False, i, "line is not JSON"
            if obj.get("prev_hash", "") != prev_hash:
                return False, i, f"prev_hash mismatch: expected {prev_hash}, got {obj.get('prev_hash', '')}"
            recomputed = {k: obj[k] for k in ("ts", "action", "args", "decision", "outcome", "prev_hash")}
            if _sha(prev_hash + json.dumps(recomputed, sort_keys=True)) != obj.get("hash"):
                return False, i, "hash recomputation mismatch"
            prev_hash = obj["hash"]
    return True, None, None
