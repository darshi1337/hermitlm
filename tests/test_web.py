from hermitlm.tools.web import web_search

def test_web():
    result = web_search(
        "latest ai news"
    )
    assert result is not None