import pytest

from hermitlm.tools.router import parse_tool_tag, route


def test_parse_tool_tag():
    assert parse_tool_tag("<tool>search(news)</tool>") == ("search", "news")
    assert parse_tool_tag("hello") is None

def test_route_faq_or_none(monkeypatch):
    monkeypatch.setattr("hermitlm.tools.router._faq", lambda q: "FAQ!" if "faqtest" in q else None)
    monkeypatch.setattr("hermitlm.tools.router._wolfram", lambda q: None)
    monkeypatch.setattr("hermitlm.tools.router._web", lambda q: None)
    name, res = route("this is faqtest please")
    assert (name, res) == ("faq", "FAQ!")
    name2, _res2 = route("hello crab swimming")
    assert name2 is None

def test_dpo_loss():
    torch = pytest.importorskip("torch")
    from hermitlm.training.dpo import dpo_loss
    a = torch.zeros(2, 5); b = torch.zeros(2, 5) - 1
    loss = dpo_loss(a, b, b, a, beta=0.1)
    assert loss.item() > 0
