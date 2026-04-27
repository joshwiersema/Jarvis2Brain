"""Allowlist-by-default policy gate for OS-touching effectors.

Every effector skill calls `policy.check(action, **context)` before doing
anything observable. Decisions:

- allow:   proceed silently
- confirm: write a `kind=approval_request` proposal note and block until
           the user touches a sentinel file or the request is denied
- deny:    refuse, log to audit

Default allowlist is intentionally tight. New skills should opt in by
listing themselves in the policy config. The `confirm` channel exists so
Jarvis 2's voice/orb can surface approvals to the user instead of silently
auto-accepting writes.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional


class Decision(str, Enum):
    ALLOW = "allow"
    CONFIRM = "confirm"
    DENY = "deny"


@dataclass
class PolicyResult:
    decision: Decision
    reason: str
    action: str
    context: dict = field(default_factory=dict)


# Hard-coded deterministic denies that no config can override.
_DENY_PATTERNS = (
    "rm -rf /",
    "rm -rf ~",
    "del /f /s /q",
    "format c:",
    "shutdown /s",
    ":(){:|:&};:",
)


@dataclass
class Policy:
    allowed_actions: set[str] = field(default_factory=lambda: {
        "file_read",
        "file_write",
        "file_edit",
        "terminal_spawn",
        "screen_capture",
    })
    confirm_actions: set[str] = field(default_factory=lambda: {
        "claude_code_spawn",
        "keyboard_type",
        "keyboard_hotkey",
        "mouse_click",
    })
    allowed_roots: list[Path] = field(default_factory=lambda: [
        Path.home(),
        Path.cwd(),
    ])
    blocked_hosts: set[str] = field(default_factory=set)

    def check(self, action: str, **context) -> PolicyResult:
        cmd = (context.get("command") or "").lower()
        for pat in _DENY_PATTERNS:
            if pat in cmd:
                return PolicyResult(Decision.DENY, f"matches deterministic deny: {pat}", action, context)

        if "path" in context:
            try:
                p = Path(context["path"]).expanduser().resolve()
                if not any(_is_under(p, root.resolve()) for root in self.allowed_roots):
                    return PolicyResult(
                        Decision.DENY,
                        f"path outside allowed roots: {p}",
                        action,
                        context,
                    )
            except OSError as e:
                return PolicyResult(Decision.DENY, f"path resolution failed: {e}", action, context)

        if action in self.allowed_actions:
            return PolicyResult(Decision.ALLOW, "allowlisted", action, context)
        if action in self.confirm_actions:
            return PolicyResult(Decision.CONFIRM, "requires confirmation", action, context)
        return PolicyResult(Decision.DENY, "not allowlisted", action, context)


def _is_under(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


_DEFAULT = Policy()


def get_policy() -> Policy:
    return _DEFAULT


@dataclass
class ConfirmRequest:
    """Confirm-channel sentinel — written to proposals/ for the user to act on."""

    action: str
    context: dict
    sentinel: Path
    created_at: float = field(default_factory=time.time)

    def is_approved(self) -> bool:
        return self.sentinel.exists()
