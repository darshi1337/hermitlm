import os

import pytest

from hermitlm.tools import router
from hermitlm.tools.router import classify_intent, route_with_intent

HAS_ARTIFACTS = all(
    os.path.exists(p) for p in ("checkpoints/best_model.pt", "data/tokenizer.json", "checkpoints/decision_probe.pt")
)

needs_artifacts = pytest.mark.skipif(not HAS_ARTIFACTS, reason="requires checkpoint/tokenizer/probe")


@needs_artifacts
def test_classify_intent_greeting():
    intent = classify_intent("hello crab")
    assert intent is not None
    assert intent["label"] == "a"
    assert intent["name"] == "social"
    assert intent["confidence"] > 0.9
    assert set(intent["proba"]) == set("abcdefg")


@needs_artifacts
def test_route_with_intent_returns_triple():
    tool, _result, intent = route_with_intent("hello crab")
    assert intent is not None and intent["label"] == "a"
    assert isinstance(tool, str) or tool is None


def test_probe_missing_falls_back_to_keywords(monkeypatch):
    from hermitlm import settings

    monkeypatch.setattr(router, "_probe", None)
    monkeypatch.setattr(router, "_probe_failed", False)
    monkeypatch.setenv("HERMIT_PROBE", "checkpoints/does-not-exist.pt")
    monkeypatch.setattr(settings, "PROBE_PATH", "checkpoints/does-not-exist.pt")
    assert classify_intent("hello crab") is None
    tool, _result, intent = route_with_intent("hello crab")
    assert intent is None
    assert isinstance(tool, str) or tool is None


def test_confused_escalates_to_web(monkeypatch):
    class FakeProbe:
        def predict(self, query):
            proba = {letter: 0.05 for letter in "abcdefg"}
            proba["g"] = 0.7
            return "g", proba, 0.97

    monkeypatch.setattr(router, "_probe", FakeProbe())
    monkeypatch.setattr(router, "_probe_failed", False)
    monkeypatch.setattr(router, "_faq", lambda q: None)
    monkeypatch.setattr(router, "_web", lambda q: "web result for " + q)
    tool, result, intent = route_with_intent(
        "quantum entanglement photosynthesis", confused_threshold=0.9
    )
    assert intent is not None and intent["p_confused"] == 0.97
    assert tool == "web"
    assert result == "web result for quantum entanglement photosynthesis"


def test_no_escalation_below_threshold(monkeypatch):
    class CalmProbe:
        def predict(self, query):
            proba = {letter: 0.05 for letter in "abcdefg"}
            proba["a"] = 0.7
            return "a", proba, 0.1

    monkeypatch.setattr(router, "_probe", CalmProbe())
    monkeypatch.setattr(router, "_probe_failed", False)
    monkeypatch.setattr(router, "_faq", lambda q: None)
    called = []
    monkeypatch.setattr(router, "_web", lambda q: called.append(q) or "should not fire")
    tool, result, intent = route_with_intent("hello crab", confused_threshold=0.9)
    assert intent is not None and intent["label"] == "a"
    assert called == []
    assert tool is None and result is None
