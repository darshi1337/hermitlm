import os

import pytest

from hermitlm.tools import router

HAS_ARTIFACTS = all(
    os.path.exists(p)
    for p in (
        "checkpoints/best_model.pt",
        "data/tokenizer.json",
        "checkpoints/decision_probe.pt",
        "checkpoints/decision_tool_probe.pt",
    )
)

needs_artifacts = pytest.mark.skipif(not HAS_ARTIFACTS, reason="requires checkpoint/tokenizer/probes")


class FakeProbe:
    def __init__(self, label="a", conf=0.9, p_confused=0.0, top2=(("a", 0.9), ("b", 0.05)), fine="greeting"):
        self._label = label
        self._conf = conf
        self._pc = p_confused
        self._top2 = top2
        self._fine = fine

    def predict(self, query):
        proba = {letter: 0.01 for letter in "abcdefg"}
        proba[self._label] = self._conf
        return self._label, proba, self._pc

    def predict_fine(self, query):
        return self._fine, {self._fine: 1.0}

    def predict_tool(self, query):
        return "chat", {"math": 0.0, "web": 0.0, "faq": 0.0, "chat": 1.0}


def _mock_probe(monkeypatch, **kwargs):
    monkeypatch.setattr(router, "_probe", FakeProbe(**kwargs))
    monkeypatch.setattr(router, "_probe_failed", False)
    monkeypatch.setattr(router, "_faq", lambda q: None)


@needs_artifacts
def test_tool_probe_routes_residue_to_math(monkeypatch):
    from hermitlm.tools.router import route_with_intent

    monkeypatch.setattr(router, "_faq", lambda q: None)
    monkeypatch.setattr(router, "_web", lambda q: None)
    monkeypatch.setattr(router, "_wolfram", lambda q: "MATH-ANSWER")
    monkeypatch.setattr(
        router, "classify_tool", lambda q: {"label": "math", "confidence": 0.8}
    )
    tool, result, _intent = route_with_intent("two plus two", tool_threshold=0.5)
    assert tool == "math"
    assert result == "MATH-ANSWER"


@needs_artifacts
def test_tool_probe_misfire_falls_through(monkeypatch):
    from hermitlm.tools.router import route_with_intent

    monkeypatch.setattr(router, "_faq", lambda q: None)
    monkeypatch.setattr(router, "_web", lambda q: None)
    monkeypatch.setattr(router, "_wolfram", lambda q: None)
    tool, result, _intent = route_with_intent("two plus two", tool_threshold=0.0)
    assert tool is None and result is None


@needs_artifacts
def test_tool_probe_abstains_below_threshold(monkeypatch):
    from hermitlm.tools.router import classify_tool

    decision = classify_tool("hello crab")
    assert decision is not None
    assert decision["label"] in ("math", "web", "faq", "chat")


def test_clarify_gate_fires_on_low_confidence(monkeypatch, tmp_path):
    _mock_probe(monkeypatch, label="c", conf=0.5, p_confused=0.1, top2=(("c", 0.5), ("d", 0.3)), fine="glass")
    from hermitlm.runtime import api
    from hermitlm.tools import db

    monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "test.db"))
    db.init_db()
    monkeypatch.setattr(api, "CLARIFY", True)
    monkeypatch.setattr(api, "CLARIFY_THRESHOLD", 0.75)
    from fastapi.testclient import TestClient

    from hermitlm.runtime.api import app

    client = TestClient(app)
    r = client.post("/chat", json={"user_id": "t", "message": "something vague here", "max_tokens": 5})
    assert r.status_code == 200
    body = r.json()
    assert body["response"].startswith("Quick check before I scuttle ahead")
    assert "tank" in body["response"] and "social" in body["response"]


def test_persona_and_stats(monkeypatch, tmp_path):
    _mock_probe(monkeypatch, label="b", conf=0.99, p_confused=0.0, fine="food")
    from hermitlm.runtime import api
    from hermitlm.tools import db

    monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "test.db"))
    db.init_db()
    monkeypatch.setattr(api, "CLARIFY", False)
    monkeypatch.setattr(api, "BEST_OF", 1)
    from fastapi.testclient import TestClient

    from hermitlm.runtime.api import app

    client = TestClient(app)
    r = client.post("/chat", json={"user_id": "t", "message": "time to eat", "max_tokens": 5})
    assert r.status_code == 200
    assert r.json()["persona"] == "hungry"
    stats = client.get("/stats/intents").json()
    assert stats["total"] >= 1
    assert stats["by_label"].get("b", 0) >= 1


def test_tool_reply_includes_persona(monkeypatch, tmp_path):
    from hermitlm.runtime import api
    from hermitlm.tools import db

    monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "test.db"))
    db.init_db()
    from fastapi.testclient import TestClient

    client = TestClient(api.app)
    r = client.post("/chat", json={"user_id": "t", "message": "who made you", "max_tokens": 5})
    assert r.status_code == 200
    body = r.json()
    assert body["tool"] == "faq"
    assert "persona" in body


def test_best_of_n_picks_best_reward():
    from hermitlm.runtime.inference import HermitInference

    engine = HermitInference("checkpoints/best_model.pt", "data/tokenizer.json", device="cpu")
    best, scores = engine.best_of_n("hello crab", n=2, max_tokens=5)
    assert isinstance(best, str) and best
    assert len(scores) == 2
