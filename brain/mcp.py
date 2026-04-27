"""MCP server exposing the Brain skill registry over stdio.

Uses the official MCP Python SDK's FastMCP wrapper. Every registered skill
becomes an MCP tool with the schema we generated from the function signature.
Jarvis 2 connects here as an MCP client and gets first-class access to
memory_search, notes_*, brain_train, and any user-added skills.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from brain.skill import discover_builtins, discover_user, get_registry
from brain.vault import resolve_vault_path


def build_server(name: str = "jarvis-brain") -> Any:
    """Construct a FastMCP server with all skills registered as tools."""
    from mcp.server.fastmcp import FastMCP

    discover_builtins()
    discover_user(resolve_vault_path(cli=None, cwd=Path.cwd()))

    server = FastMCP(name)
    registry = get_registry()

    for skill_obj in registry.list_skills():
        # Bind via closure so tool decorator captures the right fn.
        def _make_handler(sk):
            def _handler(**kwargs):
                return sk.fn(**kwargs)

            _handler.__name__ = sk.name
            _handler.__doc__ = sk.description
            return _handler

        server.tool(name=skill_obj.name, description=skill_obj.description)(
            _make_handler(skill_obj)
        )

    return server


def run_stdio() -> None:
    server = build_server()
    server.run()
