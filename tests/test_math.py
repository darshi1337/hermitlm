from hermitlm.tools.math import ask_wolfram

def test_math():
    result = ask_wolfram(
        "2+2"
    )
    assert result is not None