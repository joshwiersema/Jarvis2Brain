from brain.links import extract_wiki_links


class TestExtract:
    def test_empty(self) -> None:
        assert extract_wiki_links("") == []

    def test_no_links(self) -> None:
        assert extract_wiki_links("just plain body") == []

    def test_single(self) -> None:
        assert extract_wiki_links("see [[other-note]]") == ["other-note"]

    def test_multiple_dedup_and_order(self) -> None:
        body = "[[a]] then [[b]] and again [[a]] and [[c]]"
        assert extract_wiki_links(body) == ["a", "b", "c"]

    def test_alias_form(self) -> None:
        assert extract_wiki_links("see [[slug|display name]] here") == ["slug"]

    def test_invalid_slugs_skipped(self) -> None:
        # Uppercase, spaces, etc. fail validate_slug — silently skipped.
        body = "[[Bad Slug]] and [[good-one]] and [[UPPER]]"
        assert extract_wiki_links(body) == ["good-one"]

    def test_strips_whitespace_inside(self) -> None:
        assert extract_wiki_links("[[  hello-world  ]]") == ["hello-world"]
        # Internal whitespace inside a slug is invalid -> skipped
        assert extract_wiki_links("[[two words]]") == []
