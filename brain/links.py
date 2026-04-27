"""Wiki-link extraction: finds [[slug]] and [[slug|alias]] in note bodies."""

from __future__ import annotations

import re

from brain.slug import InvalidSlugError, validate_slug

_WIKI_RE = re.compile(r"\[\[([^\[\]]+?)\]\]")


def extract_wiki_links(body: str) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for match in _WIKI_RE.finditer(body):
        target = match.group(1).split("|", 1)[0].strip()
        try:
            slug = validate_slug(target)
        except InvalidSlugError:
            continue
        if slug not in seen:
            seen.add(slug)
            out.append(slug)
    return out
