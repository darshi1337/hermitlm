"""Fit split-conformal thresholds for the clarify and escalation gates.

Uses a labelled held-out split (``data/eval.jsonl`` by default) that the probe
heads were not trained on. Only the ``confused`` label is needed: in-scope rows
bound the false-abstention rate of both gates, and out-of-scope rows measure
escalation coverage on scores that were not used for fitting.

Usage:
    python scripts/fit_decision_thresholds.py --target-error 0.1
    python scripts/export_probe.py             # refresh web/probe.json
"""
import argparse
import json
import random

from hermitlm.tools.calibration import fit_decision_thresholds
from hermitlm.tools.decision_model import FINE_TO_COARSE, ProbeDecisionModel


def load_rows(path: str, max_n: int, seed: int) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("input") and row.get("type") in FINE_TO_COARSE:
                rows.append(row)
    random.Random(seed).shuffle(rows)
    return rows[:max_n]


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit conformal decision thresholds")
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pt")
    parser.add_argument("--tokenizer", default="data/tokenizer.json")
    parser.add_argument("--probe", default="checkpoints/decision_probe.pt")
    parser.add_argument("--data", default="data/eval.jsonl")
    parser.add_argument("--max-n", type=int, default=1000)
    parser.add_argument("--target-error", type=float, default=0.1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not 0.0 < args.target_error < 1.0:
        raise SystemExit("--target-error must be in (0, 1)")

    import torch

    from hermitlm.training.train import accelerator_available

    device = torch.device(args.device)
    if device.type in ("cuda", "xpu") and not accelerator_available(device.type):
        raise SystemExit(
            f"requested {device.type} but that backend is not available; pass --device cpu"
        )

    rows = load_rows(args.data, args.max_n, args.seed)
    if not rows:
        raise SystemExit(f"no labelled rows found in {args.data}")

    model = ProbeDecisionModel(args.checkpoint, args.tokenizer, args.probe, args.device)
    in_confidence, in_p_yes, out_p_yes = [], [], []
    for row in rows:
        _, proba, p_confused = model.predict(row["input"])
        yes = model.predict_yes(row["input"]) if model.has_yes_head else None
        p_yes = yes[0] if yes is not None else p_confused
        if row["type"] == "confused":
            out_p_yes.append(p_yes)
        else:
            in_confidence.append(max(proba.values()))
            in_p_yes.append(p_yes)
    print(f"held-out split: {len(in_confidence)} in-scope, {len(out_p_yes)} out-of-scope rows")

    fit = fit_decision_thresholds(
        in_confidence, in_p_yes, out_p_yes, args.target_error
    )
    probe = torch.load(args.probe, map_location="cpu", weights_only=True)
    probe["clarify_threshold"] = fit["clarify_threshold"]
    probe["escalation_threshold"] = fit["escalation_threshold"]
    probe["target_error"] = fit["target_error"]
    probe["threshold_coverage"] = fit["coverage"]
    probe["threshold_abstention"] = fit["abstention"]
    probe["threshold_cal_n"] = fit["cal_n"]
    probe["thresholds"] = {
        **(probe.get("thresholds") or {}),
        "clarify": fit["clarify_threshold"],
        "escalation": fit["escalation_threshold"],
        "target_error": fit["target_error"],
    }
    torch.save(probe, args.probe)

    print(
        f"clarify < {fit['clarify_threshold']:.4f}, "
        f"escalate >= {fit['escalation_threshold']:.4f} "
        f"(target error {fit['target_error']})"
    )
    print(f"coverage: {fit['coverage']}; abstention: {fit['abstention']}")
    print(f"saved thresholds to {args.probe}")
    print("run `python scripts/export_probe.py` to refresh web/probe.json")


if __name__ == "__main__":
    main()
