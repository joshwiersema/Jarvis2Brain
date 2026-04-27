"""Slug validation: lowercase kebab-case, 1-80 chars, no leading/trailing/double hyphens."""

from __future__ import annotations

import re

MAX_SLUG_LEN = 80
_SLUG_RE = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")


class InvalidSlugError(ValueError):
    """Raised when a slug fails validation."""


def validate_slug(slug: str) -> str:
    if not slug:
        raise InvalidSlugError("slug must not be empty")
    if len(slug) > MAX_SLUG_LEN:
        raise InvalidSlugError(f"slug exceeds {MAX_SLUG_LEN} chars: {slug!r}")
    if not _SLUG_RE.fullmatch(slug):
        raise InvalidSlugError(
            f"slug must be lowercase kebab-case [a-z0-9-]+, got {slug!r}"
        )
    return slug
