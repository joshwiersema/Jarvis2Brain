"""Spawn Claude Code CLI inside a worktree, tail its JSONL transcript.

The full integration (pywinpty + JSONL session file in
~/.claude/projects/<encoded-cwd>/<session>.jsonl + git-worktree management)
is the v0.3 deliverable. This v0.2 stub:

- gates the call through `policy.confirm` (always returns CONFIRM unless the
  caller's policy is overridden — Brain doesn't auto-spawn CC unattended)
- runs `claude --print --output-format stream-json` non-interactively when
  approved, capturing the streamed JSON into a `kind=trace` note
- returns the trace path, transcript text, and any diff if `cwd` is a repo

Approval is implemented via the policy CONFIRM channel — the daemon's
sentinel-file approval path will eventually wire this to Jarvis 2's voice
prompt. Until then, callers can pass `_force_allow=True` (debug-only).
"""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path
from typing import Optional

from brain.audit import append as audit_append
from brain.policy import Decision, get_policy
from brain.skill import skill
from brain.vault import resolve_vault_path


def _audit_path() -> Path:
    return resolve_vault_path(cli=os.environ.get("BRAIN_VAULT"), cwd=Path.cwd()) / ".brain" / "audit.jsonl"


@skill(
    name="claude_code_spawn",
    description="Spawn the Claude Code CLI with a prompt and capture the transcript",
)
def claude_code_spawn(
    prompt: str,
    cwd: Optional[str] = None,
    timeout: int = 600,
    _force_allow: bool = False,
) -> dict:
    decision = get_policy().check("claude_code_spawn", command="claude", cwd=cwd or "")
    if decision.decision is Decision.DENY:
        audit_append(
            _audit_path(),
            action="claude_code_spawn",
            args={"prompt": prompt[:200], "cwd": cwd},
            decision="deny",
            outcome="denied",
        )
        return {"error": decision.reason, "decision": "deny"}
    if decision.decision is Decision.CONFIRM and not _force_allow:
        audit_append(
            _audit_path(),
            action="claude_code_spawn",
            args={"prompt": prompt[:200], "cwd": cwd},
            decision="confirm",
            outcome="awaiting_approval",
        )
        return {
            "decision": "confirm",
            "reason": decision.reason,
            "instructions": "Re-call with _force_allow=True after user approves",
        }

    try:
        completed = subprocess.run(
            ["claude", "--print", "--output-format", "stream-json", prompt],
            cwd=str(Path(cwd).expanduser()) if cwd else None,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        audit_append(
            _audit_path(),
            action="claude_code_spawn",
            args={"prompt": prompt[:200], "cwd": cwd},
            decision="allow",
            outcome=f"exit {completed.returncode}",
        )
        return {
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "exit_code": completed.returncode,
        }
    except FileNotFoundError:
        return {"error": "claude CLI not found on PATH", "decision": "allow"}
    except subprocess.TimeoutExpired:
        return {"error": "timeout", "decision": "allow"}
    except Exception as e:
        return {"error": str(e), "decision": "allow"}
