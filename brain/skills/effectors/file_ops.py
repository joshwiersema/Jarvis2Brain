"""Scope-checked file effectors."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Optional

from brain.audit import append as audit_append
from brain.policy import Decision, get_policy
from brain.skill import skill
from brain.vault import resolve_vault_path


def _audit_path() -> Path:
    return resolve_vault_path(cli=os.environ.get("BRAIN_VAULT"), cwd=Path.cwd()) / ".brain" / "audit.jsonl"


@skill(name="file_read", description="Read a file from disk (policy-gated by allowed roots)")
def file_read(path: str, max_bytes: int = 1_000_000) -> dict:
    decision = get_policy().check("file_read", path=path)
    if decision.decision is not Decision.ALLOW:
        audit_append(_audit_path(), action="file_read", args={"path": path}, decision=decision.decision.value, outcome="denied")
        return {"error": decision.reason, "decision": decision.decision.value}
    p = Path(path).expanduser().resolve()
    if not p.exists() or p.is_dir():
        audit_append(_audit_path(), action="file_read", args={"path": path}, decision="allow", outcome="not_found")
        return {"error": "not a regular file", "path": str(p)}
    data = p.read_bytes()[:max_bytes]
    text = data.decode("utf-8", errors="replace")
    audit_append(_audit_path(), action="file_read", args={"path": str(p)}, decision="allow", outcome="ok")
    return {"path": str(p), "content": text, "bytes": len(data)}


@skill(name="file_write", description="Write a file to disk (atomic, policy-gated)")
def file_write(path: str, content: str, overwrite: bool = False) -> dict:
    decision = get_policy().check("file_write", path=path)
    if decision.decision is not Decision.ALLOW:
        audit_append(_audit_path(), action="file_write", args={"path": path}, decision=decision.decision.value, outcome="denied")
        return {"error": decision.reason, "decision": decision.decision.value}
    p = Path(path).expanduser().resolve()
    if p.exists() and not overwrite:
        audit_append(_audit_path(), action="file_write", args={"path": str(p)}, decision="allow", outcome="exists")
        return {"error": "file exists; pass overwrite=true", "path": str(p)}
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    audit_append(_audit_path(), action="file_write", args={"path": str(p), "bytes": len(content)}, decision="allow", outcome="ok")
    return {"path": str(p), "bytes": len(content)}
