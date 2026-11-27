import json
import os
from pathlib import Path

import pytest

from hermitlm.tools.calibration import conformal_quantile, fit_decision_thresholds


def test_lower_target_error_is_more_conservative():
    confidence = [0.1 + 0.8 * i / 199 for i in range(200)]
    p_in_scope = [0.01 + 0.8 * i / 199 for i in range(200)]
    p_out_of_scope = [0.2 + 0.75 * i / 99 for i in range(100)]

    loose = fit_decision_thresholds(confidence, p_in_scope, p_out_of_scope, 0.2)
    tight = fit_decision_thresholds(confidence, p_in_scope, p_out_of_scope, 0.05)

    # Clarify fires BELOW its cutoff, escalation fires ABOVE its cutoff.
    assert tight["clarify_threshold"] < loose["clarify_threshold"]
    assert tight["escalation_threshold"] > loose["escalation_threshold"]
    assert tight["abstention"] <= loose["abstention"]
    assert tight["target_error"] == 0.05


def test_conformal_quantile_covers_requested_rate():
    scores = [i / 1000 for i in range(1, 1001)]
    lower = conformal_quantile(scores, 0.1, tail="lower")
    upper = conformal_quantile(scores, 0.1, tail="upper")
    assert 0.09 <= lower <= 0.11
    assert 0.89 <= upper <= 0.91


@pytest.mark.parametrize("target", [0.0, 1.0, -0.1, 1.2])
def test_invalid_target_error_rejected(target):
    with pytest.raises(ValueError):
        conformal_quantile([0.1, 0.2, 0.3], target)


def test_missing_scores_rejected():
    with pytest.raises(ValueError):
        fit_decision_thresholds([], [], [0.9], 0.1)
    with pytest.raises(ValueError):
        fit_decision_thresholds([0.5, 0.6], [0.1], [0.9], 0.1)


def test_intent_carries_stored_thresholds(monkeypatch):
    from hermitlm.tools import router

    class Probe:
        clarify_threshold = 0.42
        escalation_threshold = 0.88
        rotations = 1

        def predict(self, query):
            return "a", {letter: 0.5 if letter == "a" else 0.1 for letter in "abcdefg"}, 0.2

        def predict_fine(self, query):
            return None

    monkeypatch.setattr(router, "_probe", Probe())
    monkeypatch.setattr(router, "_probe_failed", False)
    intent = router.classify_intent("anything")
    assert intent is not None
    assert intent["clarify_threshold"] == 0.42
    assert intent["escalation_threshold"] == 0.88


@pytest.mark.skipif(
    not os.path.exists("web/probe.json") or not os.path.exists("checkpoints/decision_probe.pt"),
    reason="requires the shipped probe artifact",
)
def test_api_and_browser_read_the_same_stored_thresholds():
    import torch

    artifact = torch.load("checkpoints/decision_probe.pt", map_location="cpu", weights_only=True)
    web = json.loads(Path("web/probe.json").read_text(encoding="utf-8"))
    for key in ("clarify_threshold", "escalation_threshold"):
        assert key in artifact and key in web
        assert web[key] == pytest.approx(float(artifact[key]))
    assert web.get("threshold_coverage") == artifact.get("threshold_coverage")

    js = Path("web/intent.js").read_text(encoding="utf-8")
    html = Path("web/index.html").read_text(encoding="utf-8")
    assert "probe.clarify_threshold" in js
    assert "probe.escalation_threshold" in js
    assert "intent.clarify_threshold" in html
    assert "intent.escalation_threshold" in html
    assert "CLARIFY_THRESHOLD } from" not in html


def test_stats_reports_coverage_and_abstention(monkeypatch, tmp_path):
    from hermitlm.runtime import api
    from hermitlm.tools import db

    monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "stats.db"))
    db.init_db()
    base = {"label": "a", "name": "social", "confidence": 0.9, "p_confused": 0.1}
    db.insert_intent("q1", {**base, "clarified": True}, None)
    db.insert_intent("q2", {**base, "escalated": True}, None)
    db.insert_intent("q3", base, None)

    class Probe:
        def __init__(self):
            self.threshold_coverage = {"clarify": 0.92, "escalation": 0.88}
            self.target_error = 0.1

    monkeypatch.setattr(api, "get_probe", lambda: Probe())
    stats = api.stats_intents()
    assert stats["total"] == 3
    assert stats["abstention_rate"] == 0.6667
    assert stats["confused_rate"] == 0.3333
    assert stats["coverage"] == {"clarify": 0.92, "escalation": 0.88}
    assert stats["target_error"] == 0.1
