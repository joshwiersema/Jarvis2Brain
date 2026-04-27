"""brain CLI: create/read/update/delete/list/search/serve."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence, TextIO

from brain.note import Note, now_utc
from brain.search import search as search_vault
from brain.slug import InvalidSlugError
from brain.vault import (
    NoteExistsError,
    NoteNotFoundError,
    Vault,
    resolve_vault_path,
)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="brain", description="Markdown vault tool")
    p.add_argument("--vault", help="Override vault path (else BRAIN_VAULT or ./vault)")
    sub = p.add_subparsers(dest="cmd", required=True)

    pc = sub.add_parser("create", help="Create a new note")
    pc.add_argument("slug")
    pc.add_argument("--title", required=True)
    pc.add_argument("--tag", action="append", default=[], help="Repeatable")
    pc.add_argument(
        "--body", default="", help="Body text. Use '-' to read from stdin."
    )

    pr = sub.add_parser("read", help="Read a note")
    pr.add_argument("slug")

    pu = sub.add_parser("update", help="Update a note")
    pu.add_argument("slug")
    pu.add_argument("--title")
    pu.add_argument("--tag", action="append", help="Replace tags. Repeatable.")
    pu.add_argument("--body", help="Body text. Use '-' to read from stdin.")

    pd = sub.add_parser("delete", help="Delete a note")
    pd.add_argument("slug")

    pl = sub.add_parser("list", help="List notes")
    pl.add_argument("--tag", help="Filter by tag")

    ps = sub.add_parser("search", help="Search notes")
    ps.add_argument("query", nargs="?", default="")
    ps.add_argument("--tag", help="Filter by tag")

    psv = sub.add_parser("serve", help="Run the HTTP server")
    psv.add_argument("--host", default="127.0.0.1")
    psv.add_argument("--port", type=int, default=8000)

    return p


def _read_body(value: str, stdin: TextIO) -> str:
    return stdin.read() if value == "-" else value


def _print_note(note: Note, out: TextIO) -> None:
    out.write(
        json.dumps(
            {
                "slug": note.slug,
                "title": note.title,
                "created": note.created.isoformat(),
                "updated": note.updated.isoformat(),
                "tags": list(note.tags),
                "body": note.body,
            },
            indent=2,
        )
        + "\n"
    )


def main(
    argv: Optional[Sequence[str]] = None,
    *,
    stdin: Optional[TextIO] = None,
    stdout: Optional[TextIO] = None,
    stderr: Optional[TextIO] = None,
) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    stdin = stdin or sys.stdin
    stdout = stdout or sys.stdout
    stderr = stderr or sys.stderr

    vault_path = resolve_vault_path(cli=args.vault, cwd=Path.cwd())

    if args.cmd == "serve":
        # Imported lazily so non-serve commands don't pay the FastAPI import cost.
        import uvicorn

        from brain.server import create_app

        app = create_app(vault_path)
        uvicorn.run(app, host=args.host, port=args.port)
        return 0

    vault = Vault(vault_path)

    try:
        if args.cmd == "create":
            ts = now_utc()
            note = Note(
                slug=args.slug,
                title=args.title,
                created=ts,
                updated=ts,
                tags=list(args.tag),
                body=_read_body(args.body, stdin),
            )
            vault.create(note)
            _print_note(note, stdout)
            return 0

        if args.cmd == "read":
            _print_note(vault.read(args.slug), stdout)
            return 0

        if args.cmd == "update":
            body = _read_body(args.body, stdin) if args.body is not None else None
            note = vault.update(
                args.slug,
                title=args.title,
                body=body,
                tags=list(args.tag) if args.tag is not None else None,
            )
            _print_note(note, stdout)
            return 0

        if args.cmd == "delete":
            vault.delete(args.slug)
            return 0

        if args.cmd == "list":
            for s in vault.list(tag=args.tag):
                stdout.write(f"{s.slug}\t{s.title}\t{','.join(s.tags)}\n")
            return 0

        if args.cmd == "search":
            for s in search_vault(vault, args.query, tag=args.tag):
                stdout.write(f"{s.slug}\t{s.title}\n")
            return 0

    except NoteNotFoundError as e:
        stderr.write(f"error: {e}\n")
        return 2
    except NoteExistsError as e:
        stderr.write(f"error: {e}\n")
        return 3
    except InvalidSlugError as e:
        stderr.write(f"error: {e}\n")
        return 4

    parser.error(f"unknown command: {args.cmd}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
