"""Link / graph traversal skills."""

from __future__ import annotations

import os
from pathlib import Path

from brain.skill import skill
from brain.vault import NoteNotFoundError, Vault, resolve_vault_path


def _vault() -> Vault:
    return Vault(resolve_vault_path(cli=os.environ.get("BRAIN_VAULT"), cwd=Path.cwd()))


@skill(name="links_outgoing", description="List slugs this note links to")
def links_outgoing(slug: str) -> list[str]:
    try:
        return _vault().outgoing_links(slug)
    except NoteNotFoundError:
        return []


@skill(name="links_incoming", description="List slugs that link to this note")
def links_incoming(slug: str) -> list[str]:
    return _vault().incoming_links(slug)
