from hermitlm.tools.bot_faq import get_faq_response

def test_bot_faq():
    response = get_faq_response(
        "who are you"
    )
    assert response is not None


def test_bot_unknown():
    response = get_faq_response(
        "random strange prompt"
    )
    assert response is None