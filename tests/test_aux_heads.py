import json
import os
import sqlite3
from pathlib import Path

import pytest
import torch

from hermitlm.tools.artifacts import ArtifactMismatchError
from hermitlm.tools.decision_model import (
    ProbeDecisionModel,
    export_probe,
    get_label_ids,
)

HAS_ARTIFACTS = all(
    os.path.exists(path)
    for path in (
        "checkpoints/best_model.pt",
        "data/tokenizer.json",
        "checkpoints/decision_probe.pt",
    )
)
needs_artifacts = pytest.mark.skipif(not HAS_ARTIFACTS, reason="requires checkpoint/tokenizer/probe")


class FakeTokenizer:
    """Digit labels encode to two tokens; letters are single tokens."""

    def encode(self, text):
        class Encoded:
            ids = [100, 101] if text.isdigit() else [10 + ord(text[0]) - ord("a")]

        return Encoded()


def test_digit_labels_fall_back_to_letters():
    labels, ids = get_label_ids(FakeTokenizer(), ["0", "1", "2"])
    assert labels == ["a", "b", "c"]
    assert ids == [10, 11, 12]


@needs_artifacts
def test_probe_loads_aux_heads_and_predicts():
    probe = ProbeDecisionModel(
        "checkpoints/best_model.pt",
        "data/tokenizer.json",
        "checkpoints/decision_probe.pt",
        device="cpu",
    )
    assert probe.has_yes_head and probe.has_score_head
    p_yes, yes_proba = probe.predict_yes("hello crab")
    assert 0.0 <= p_yes <= 1.0
    assert set(yes_proba) == {"in_scope", "out_of_scope"}
    expected, probs = probe.predict_score("hello crab", "i am a crab in the water")
    assert min(probe.score_levels) <= expected <= max(probe.score_levels)
    assert len(probs) == len(probe.score_levels)
    assert sum(probs) == pytest.approx(1.0, abs=1e-4)


@needs_artifacts
def test_yes_head_separates_out_of_scope(tmp_path):
    probe = ProbeDecisionModel(
        "checkpoints/best_model.pt",
        "data/tokenizer.json",
        "checkpoints/decision_probe.pt",
        device="cpu",
    )
    rows = [json.loads(line) for line in Path("data/human_eval.jsonl").read_text(encoding="utf-8").splitlines()]
    confused = [probe.predict_yes(row["input"])[0] for row in rows if row["type"] == "confused"]
    in_scope = [probe.predict_yes(row["input"])[0] for row in rows if row["type"] != "confused"]
    assert sum(confused) / len(confused) > sum(in_scope) / len(in_scope)


@needs_artifacts
def test_export_probe_includes_aux_heads(tmp_path):
    data = export_probe(
        "checkpoints/decision_probe.pt", str(tmp_path / "probe.json")
    )
    assert data["head_yes"]["weight"] and data["head_score"]["weight"]
    assert data["temperature_yes"] > 0 and data["temperature_score"] > 0
    assert len(data["score_levels"]) == len(data["score_labels"]) >= 2
    assert "yes" in data["priors"] and "score" in data["priors"]


@needs_artifacts
def test_stale_aux_head_artifact_is_rejected(tmp_path):
    artifact = torch.load(
        "checkpoints/decision_probe.pt", map_location="cpu", weights_only=True
    )
    artifact["checkpoint_sha256"] = "0" * 64
    stale = tmp_path / "stale.pt"
    torch.save(artifact, stale)
    with pytest.raises(ArtifactMismatchError):
        ProbeDecisionModel(
            "checkpoints/best_model.pt",
            "data/tokenizer.json",
            str(stale),
            device="cpu",
        )


def test_router_returns_p_yes_and_uses_it(monkeypatch):
    from hermitlm.tools import router

    class Probe:
        def __init__(self):
            self.priors = {}
            self.prior_lambda = 0.0

        def predict(self, query):
            return "a", {letter: 1 / 7 for letter in "abcdefg"}, 0.1

        def predict_fine(self, query):
            return None

        def predict_yes(self, query):
            return 0.95, {"in_scope": 0.05, "out_of_scope": 0.95}

    monkeypatch.setattr(router, "_probe", Probe())
    monkeypatch.setattr(router, "_probe_failed", False)
    monkeypatch.setattr(router, "_web", lambda q: "WEB RESULT")
    monkeypatch.setattr(router, "_faq", lambda q: None)

    intent = router.classify_intent("is this out of scope")
    assert intent is not None and intent["p_yes"] == 0.95
    tool, result, _ = router.route_with_intent("is this out of scope", confused_threshold=0.5)
    assert tool == "web" and result == "WEB RESULT"


def test_chat_logs_expected_score(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient

    from hermitlm.runtime import api
    from hermitlm.tools import db

    monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "test.db"))
    monkeypatch.setattr("hermitlm.tools.memory.DB_PATH", str(tmp_path / "memory.db"))
    from hermitlm.tools import memory

    memory.init_memory_db()
    db.init_db()
    monkeypatch.setattr(api, "CLARIFY", False)
    monkeypatch.setattr(api, "BEST_OF", 3)
    monkeypatch.setattr(
        api,
        "route_with_intent",
        lambda _: (None, None, {
            "label": "a", "name": "social", "confidence": 0.9,
            "p_confused": 0.1, "p_yes": 0.1,
            "top2": [["a", 0.9], ["b", 0.1]],
        }),
    )
    monkeypatch.setattr(
        api.engine, "best_of_n", lambda *a, **k: ("reply", [0.0], 0.42)
    )

    client = TestClient(api.app)
    response = client.post("/chat", json={"user_id": "u", "message": "hello crab"})
    assert response.status_code == 200
    with sqlite3.connect(str(tmp_path / "test.db")) as conn:
        logged = conn.execute("SELECT expected_score FROM intent_log").fetchone()[0]
    assert logged == pytest.approx(0.42)
