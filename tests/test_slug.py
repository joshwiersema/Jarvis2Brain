import pytest

from brain.slug import InvalidSlugError, validate_slug


class TestValidSlugs:
    @pytest.mark.parametrize(
        "slug",
        [
            "my-note",
            "a",
            "abc",
            "note-with-many-words",
            "note123",
            "123",
            "x" * 80,
        ],
    )
    def test_accepts(self, slug: str) -> None:
        assert validate_slug(slug) == slug


class TestInvalidSlugs:
    @pytest.mark.parametrize(
        ("slug", "reason"),
        [
            ("", "empty"),
            ("UPPER", "uppercase"),
            ("Mixed-Case", "uppercase"),
            ("under_score", "underscore"),
            ("with space", "space"),
            ("-leading", "leading hyphen"),
            ("trailing-", "trailing hyphen"),
            ("double--hyphen", "double hyphen"),
            ("dot.in.slug", "dot"),
            ("slash/in/slug", "slash"),
            ("x" * 81, "too long"),
            ("über", "non-ascii"),
        ],
    )
    def test_rejects(self, slug: str, reason: str) -> None:
        with pytest.raises(InvalidSlugError):
            validate_slug(slug)
