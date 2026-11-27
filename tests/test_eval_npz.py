import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("eval_decision", REPO / "scripts" / "eval_decision.py")
eval_decision = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(eval_decision)


def test_keyword_tool_has_no_network_dispatch(monkeypatch):
    from hermitlm.tools import router

    def fail(_):
        raise AssertionError("network dispatch must not run in evaluation")

    monkeypatch.setattr(router, "_web", fail)
    monkeypatch.setattr(router, "_wolfram", fail)
    monkeypatch.setattr(router, "_faq", lambda q: "faq answer" if "yourself" in q else None)

    assert router.keyword_tool("solve 2 + 2") == "math"
    assert router.keyword_tool("tell me the latest news") == "web"
    assert router.keyword_tool("who are yourself") == "faq"
    assert router.keyword_tool("hello there crab") == "chat"
    assert router.keyword_tool("<tool>web(latest)</tool>") == "web"


def test_tool_keyword_diff_reports_positive_improvement():
    gold = ["math", "math", "web", "web"]
    tool = ["math", "math", "web", "chat"]
    keyword = ["chat", "chat", "chat", "chat"]
    diff = eval_decision._tool_keyword_diff(gold, tool, keyword, n_boot=200, seed=0)
    assert diff["tool_accuracy"] == 0.75
    assert diff["keyword_accuracy"] == 0.0
    assert diff["accuracy"]["diff"] == 0.75
    assert diff["accuracy"]["ci"][0] > 0


@pytest.mark.skipif(
    not (REPO / "checkpoints" / "best_model.pt").exists(),
    reason="requires the shipped checkpoint and probes",
)
def test_npz_replay_reproduces_the_same_report(tmp_path):
    rows = [
        {"input": "hello there", "type": "greeting"},
        {"input": "what do you eat", "type": "food"},
        {"input": "is the tank cold", "type": "glass"},
        {"input": "where are the bubbles", "type": "bubbles"},
        {"input": "how big is the cave", "type": "environment"},
        {"input": "do crabs dream", "type": "lonely"},
        {"input": "tell me about photosynthesis", "type": "confused"},
        {"input": "who built the pyramids", "type": "confused"},
    ]
    data = tmp_path / "eval.jsonl"
    data.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    first = eval_decision.run_eval(
        str(REPO / "checkpoints" / "best_model.pt"),
        str(REPO / "data" / "tokenizer.json"),
        str(data),
        max_n=8,
        device="cpu",
        out_dir=str(tmp_path / "first"),
        method="probe",
        probe_path=str(REPO / "checkpoints" / "decision_probe.pt"),
        cal_n=4,
        bootstrap=10,
        seed=0,
        tool_probe_path=str(REPO / "checkpoints" / "decision_tool_probe.pt"),
        tool_eval_path=str(REPO / "data" / "tool_eval.jsonl"),
    )
    npz = tmp_path / "first" / "items.npz"
    assert npz.exists()

    replay = eval_decision.run_eval(
        out_dir=str(tmp_path / "second"),
        bootstrap=10,
        seed=0,
        from_npz=str(npz),
    )
    assert replay == first
    assert "paired: tool head - keywords" in (tmp_path / "second" / "report.md").read_text(encoding="utf-8")
