"""Graph export skill."""

from __future__ import annotations

import os
from pathlib import Path

from brain.skill import skill
from brain.vault import Vault, resolve_vault_path


@skill(name="graph_export", description="Export the full vault graph as nodes + edges")
def graph_export() -> dict:
    v = Vault(resolve_vault_path(cli=os.environ.get("BRAIN_VAULT"), cwd=Path.cwd()))
    return v.graph()
