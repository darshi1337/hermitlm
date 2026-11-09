import os

import pytest

from hermitlm.tools import web
from hermitlm.tools.web import web_search


class FakeDDGS:
    """Stand-in for ddgs.DDGS that returns fixed results."""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def text(self, query, max_results=3):
        assert query == "latest ai news"
        return [
            {"title": "Title A", "body": "Body A"},
            {"title": "Title B", "body": "Body B"},
        ]


class FailingDDGS:
    """Stand-in for ddgs.DDGS that raises like a failed search."""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def text(self, query, max_results=3):
        raise RuntimeError("No results found.")


def test_web_formats_results(monkeypatch):
    monkeypatch.setattr(web, "DDGS", FakeDDGS)
    result = web_search("latest ai news")
    assert result == "Title A: Body A\nTitle B: Body B"


def test_web_returns_none_on_search_error(monkeypatch):
    monkeypatch.setattr(web, "DDGS", FailingDDGS)
    assert web_search("latest ai news") is None


@pytest.mark.skipif(
    not os.getenv("HERMIT_LIVE_WEB"),
    reason="requires live network; set HERMIT_LIVE_WEB=1",
)
def test_web_live():
    result = web_search("latest ai news")
    assert result is not None
