import os

import pytest

from hermitlm.tools.math import ask_wolfram


@pytest.mark.skipif(
    not os.getenv("WOLFRAM_APP_ID"),
    reason="requires WOLFRAM_APP_ID and network access",
)
def test_math():
    result = ask_wolfram(
        "2+2"
    )
    assert result is not None
