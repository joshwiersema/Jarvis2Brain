"""Terminal effector — runs a command and returns stdout/stderr/exit_code.

Uses subprocess for portability. The Windows roadmap is to swap this for
pywinpty (real ConPTY) so interactive REPLs and ANSI sequences pass through
correctly; that lives behind `BRAIN_PTY=pywinpty` once the dependency is
available. Output is capped per call so a runaway command can't OOM the
daemon.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Optional

from brain.audit import append as audit_append
from brain.policy import Decision, get_policy
from brain.skill import skill
from brain.vault import resolve_vault_path

MAX_STDOUT_BYTES = 1_000_000


def _audit_path() -> Path:
    return resolve_vault_path(cli=os.environ.get("BRAIN_VAULT"), cwd=Path.cwd()) / ".brain" / "audit.jsonl"


@skill(
    name="terminal_spawn",
    description="Run a shell command and return stdout/stderr/exit_code",
)
def terminal_spawn(
    command: str,
    cwd: Optional[str] = None,
    timeout: int = 60,
) -> dict:
    decision = get_policy().check("terminal_spawn", command=command, cwd=cwd or "")
    if decision.decision is not Decision.ALLOW:
        audit_append(
            _audit_path(),
            action="terminal_spawn",
            args={"command": command, "cwd": cwd},
            decision=decision.decision.value,
            outcome="denied",
        )
        return {"error": decision.reason, "decision": decision.decision.value}

    try:
        completed = subprocess.run(
            command,
            shell=True,
            cwd=str(Path(cwd).expanduser()) if cwd else None,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        stdout = (completed.stdout or "")[:MAX_STDOUT_BYTES]
        stderr = (completed.stderr or "")[:MAX_STDOUT_BYTES]
        outcome = "ok" if completed.returncode == 0 else f"exit {completed.returncode}"
        audit_append(
            _audit_path(),
            action="terminal_spawn",
            args={"command": command, "cwd": cwd},
            decision="allow",
            outcome=outcome,
        )
        return {
            "stdout": stdout,
            "stderr": stderr,
            "exit_code": completed.returncode,
        }
    except subprocess.TimeoutExpired:
        audit_append(
            _audit_path(),
            action="terminal_spawn",
            args={"command": command, "cwd": cwd},
            decision="allow",
            outcome="timeout",
        )
        return {"error": "timeout", "exit_code": -1}
    except Exception as e:
        audit_append(
            _audit_path(),
            action="terminal_spawn",
            args={"command": command, "cwd": cwd},
            decision="allow",
            outcome=f"error: {e}",
        )
        return {"error": str(e), "exit_code": -1}
