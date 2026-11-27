"""Zero-shot eval + temperature calibration for the decision model.

Mirrors the eval from the article: accuracy, per-class precision/recall/F1, and a
confidence-vs-accuracy bin table, plus ECE/NLL with 95% bootstrap CIs. A
temperature T is fitted on a disjoint calibration split by minimising NLL over
log T (binary heads use Platt-smoothed targets) and applied to the test split.

Usage:
    conda run -n hermit python scripts/eval_decision.py \
        --checkpoint checkpoints/best_model.pt --max-n 1000 --out-dir eval_out/decision
    # Recompute every metric from a saved per-item dump, with no model pass:
    python scripts/eval_decision.py --from-npz eval_out/decision/items.npz \
        --out-dir eval_out/decision_replay
"""

import argparse
import json
import logging
import os
import random

import numpy as np
import torch

from hermitlm.tools.calibration import InsufficientDataError, fit_temperature
from hermitlm.tools.decision_model import (
    COARSE_NAMES,
    FINE21_ORDER,
    FINE_TO_COARSE,
    OPTION_LETTERS,
    TOOL_CLASSES,
    DecisionModel,
    ProbeDecisionModel,
    format_prompt,
    option_assignment,
    rotation_shifts,
)
from hermitlm.tools.metrics import (
    accuracy,
    bootstrap_ci,
    ece,
    macro_f1,
    nll,
    paired_bootstrap_diff,
)
from hermitlm.tools.prior import apply_prior
from hermitlm.tools.router import keyword_tool

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

N_BINS = 10


def _load_rows(eval_path: str, max_n: int, seed: int = 0) -> list[dict]:
    """Load, shuffle and cap eval rows.

    Shuffling before the `max_n` cap and before the calibration/test split
    keeps the two halves class-balanced even when the source file is grouped
    by `type`.
    """
    rows = []
    with open(eval_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if d.get("input") and d.get("type") in FINE_TO_COARSE:
                rows.append(d)
    random.Random(seed).shuffle(rows)
    return rows[:max_n]


def _prf(y_true: list[str], y_pred: list[str]) -> dict:
    per_class = {}
    for letter in OPTION_LETTERS:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == letter == p)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != letter == p)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == letter != p)
        support = sum(1 for t in y_true if t == letter)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[letter] = {
            "name": COARSE_NAMES[letter],
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": support,
        }
    return per_class


def _bins(confs: list[float], corrects: list[bool]) -> list[dict]:
    table = []
    for b in range(N_BINS):
        lo, hi = b / N_BINS, (b + 1) / N_BINS
        idx = [i for i, c in enumerate(confs) if (c > lo and c <= hi) or (b == 0 and c == 0.0)]
        if not idx:
            table.append({"bin": f"({lo:.2f}, {hi:.2f}]", "count": 0, "confidence": 0.0, "accuracy": 0.0})
            continue
        conf = sum(confs[i] for i in idx) / len(idx)
        acc = sum(1 for i in idx if corrects[i]) / len(idx)
        table.append({"bin": f"({lo:.2f}, {hi:.2f}]", "count": len(idx), "confidence": round(conf, 4), "accuracy": round(acc, 4)})
    return table


def _matrix(probas: list[dict[str, float]]) -> torch.Tensor:
    return torch.tensor(
        [[p[letter] for letter in OPTION_LETTERS] for p in probas], dtype=torch.float64
    )


@torch.no_grad()
def _zero_shot_log_probs(model: DecisionModel, query: str, shift: int) -> torch.Tensor:
    """Log-probabilities in original class order for one L0 assignment."""
    assignment = option_assignment(shift)
    ids = model.tokenizer.encode(format_prompt(query, assignment)).ids[-model.config.max_seq_len :]
    idx = torch.tensor([ids], dtype=torch.long, device=model.device)
    logits, _ = model.model(idx)
    display = torch.log_softmax(logits[0, -1, model.option_ids].float(), dim=-1).cpu()
    out = torch.empty_like(display)
    out[torch.tensor([OPTION_LETTERS.index(cls) for cls in assignment])] = display
    return out


def _order_flip_rate(
    rows: list[dict], model: DecisionModel, rotations: int = 2, flip_shift: int = 3
) -> dict:
    """Compare predictions under the normal and a permuted option order.

    ``raw`` uses one displayed order per side; ``L0`` averages ``rotations``
    evenly spaced orders per side. A lower L0 rate means the prediction is
    less dependent on where the options happen to sit.
    """
    k = len(OPTION_LETTERS)
    l0_shifts = rotation_shifts(rotations)
    needed = sorted(set(l0_shifts) | {flip_shift})
    per_shift: dict[int, list[torch.Tensor]] = {shift: [] for shift in needed}
    with torch.no_grad():
        for row in rows:
            for shift in needed:
                per_shift[shift].append(_zero_shot_log_probs(model, row["input"], shift))
    stacks = {shift: torch.stack(values) for shift, values in per_shift.items()}

    def preds(log_probs: torch.Tensor) -> torch.Tensor:
        return log_probs.argmax(dim=-1)

    def mean_of(shifts: list[int]) -> torch.Tensor:
        return torch.stack([stacks[shift] for shift in shifts]).mean(dim=0)

    raw_normal = preds(stacks[0])
    raw_permuted = preds(stacks[flip_shift])
    l0_normal = preds(mean_of(l0_shifts))
    l0_permuted = preds(mean_of([(flip_shift + shift) % k for shift in l0_shifts]))

    n = len(rows)
    return {
        "raw": round(float((raw_normal != raw_permuted).float().mean()), 4),
        "L0": round(float((l0_normal != l0_permuted).float().mean()), 4),
        "rotations": rotations,
        "permutation_shift": flip_shift,
        "n": n,
    }


def summarize(
    y_true: list[str],
    y_pred: list[str],
    probas: list[dict[str, float]],
    temperature: float,
    *,
    n_boot: int = 0,
    seed: int = 0,
) -> dict:
    n = len(y_true)
    if n == 0:
        return {"temperature": temperature, "n": 0}
    targets = torch.tensor([OPTION_LETTERS.index(t) for t in y_true])
    preds = torch.tensor([OPTION_LETTERS.index(p) for p in y_pred])
    probs = _matrix(probas)
    n_classes = len(OPTION_LETTERS)
    confs = [probas[i][y_pred[i]] for i in range(n)]
    corrects = [t == p for t, p in zip(y_true, y_pred)]
    ece_bins = _bins(confs, corrects)

    result = {
        "temperature": temperature,
        "n": n,
        "accuracy": round(accuracy(probs, targets), 4),
        "nll": round(nll(probs, targets), 4),
        "ece": round(ece(probs, targets, N_BINS), 4),
        "macro_f1": round(macro_f1(preds, targets, n_classes), 4),
        "per_class": _prf(y_true, y_pred),
        "bins": ece_bins,
    }
    if n_boot > 0 and n >= 2:
        acc_ci = bootstrap_ci(lambda i: accuracy(probs[i], targets[i]), n, n_boot=n_boot, seed=seed)
        nll_ci = bootstrap_ci(lambda i: nll(probs[i], targets[i]), n, n_boot=n_boot, seed=seed + 1)
        ece_ci = bootstrap_ci(lambda i: ece(probs[i], targets[i], N_BINS), n, n_boot=n_boot, seed=seed + 2)
        f1_ci = bootstrap_ci(
            lambda i: macro_f1(preds[i], targets[i], n_classes), n, n_boot=n_boot, seed=seed + 3
        )
        result["accuracy_ci"] = [round(v, 4) for v in acc_ci]
        result["nll_ci"] = [round(v, 4) for v in nll_ci]
        result["ece_ci"] = [round(v, 4) for v in ece_ci]
        result["macro_f1_ci"] = [round(v, 4) for v in f1_ci]
    return result


def paired_report(
    uncalibrated: list[dict[str, float]],
    calibrated: list[dict[str, float]],
    y_true: list[str],
    *,
    n_boot: int,
    seed: int,
) -> dict:
    """Paired bootstrap of calibrated-minus-uncalibrated per metric."""
    if not y_true:
        return {}
    targets = torch.tensor([OPTION_LETTERS.index(t) for t in y_true])
    n_classes = len(OPTION_LETTERS)
    a = _matrix(uncalibrated)
    b = _matrix(calibrated)
    n = len(y_true)
    specs = [
        ("accuracy", lambda p, i: accuracy(p[i], targets[i])),
        ("nll", lambda p, i: nll(p[i], targets[i])),
        ("ece", lambda p, i: ece(p[i], targets[i], N_BINS)),
        (
            "macro_f1",
            lambda p, i: macro_f1(p[i].argmax(-1), targets[i], n_classes),
        ),
    ]
    out = {}
    for offset, (name, stat) in enumerate(specs):
        out[name] = paired_bootstrap_diff(
            lambda i, stat=stat: stat(a, i),
            lambda i, stat=stat: stat(b, i),
            n,
            n_boot=max(1, n_boot),
            seed=seed + offset,
        )
        out[name]["diff"] = round(out[name]["diff"], 4)
        out[name]["ci"] = [round(v, 4) for v in out[name]["ci"]]
    return out


def _fit_or_default(logits: torch.Tensor, targets: torch.Tensor, fallback: float) -> float:
    try:
        return fit_temperature(logits, targets)
    except InsufficientDataError:
        return fallback


def _confused_ci(pb: torch.Tensor, labels: list[int], threshold: float, n_boot: int, seed: int) -> dict:
    if n_boot <= 0 or len(labels) < 2:
        return {}
    y = torch.tensor(labels, device=pb.device)
    preds = (pb >= threshold).long()
    n = len(labels)

    def precision(idx: torch.Tensor) -> float:
        p, t = preds[idx], y[idx]
        tp = int(((p == 1) & (t == 1)).sum())
        fp = int(((p == 1) & (t == 0)).sum())
        return tp / (tp + fp) if tp + fp else 0.0

    def recall(idx: torch.Tensor) -> float:
        p, t = preds[idx], y[idx]
        tp = int(((p == 1) & (t == 1)).sum())
        fn = int(((p == 0) & (t == 1)).sum())
        return tp / (tp + fn) if tp + fn else 0.0

    return {
        "precision_ci": [round(v, 4) for v in bootstrap_ci(precision, n, n_boot=n_boot, seed=seed)],
        "recall_ci": [round(v, 4) for v in bootstrap_ci(recall, n, n_boot=n_boot, seed=seed + 1)],
    }


def _stored_gate_report(probe_path: str) -> dict | None:
    """Coverage fitted on the held-out calibration split, for any eval method."""
    if not os.path.exists(probe_path):
        return None
    probe = torch.load(probe_path, map_location="cpu", weights_only=True)
    if "clarify_threshold" not in probe:
        return None
    coverage = probe.get("threshold_coverage", {})
    return {
        "clarify_threshold": float(probe["clarify_threshold"]),
        "escalation_threshold": float(probe["escalation_threshold"]),
        "target_error": float(probe.get("target_error", 0.1)),
        "clarify_coverage": float(coverage.get("clarify", 0.0)),
        "escalation_coverage": float(coverage.get("escalation", 0.0)),
        "abstention_rate": float(probe.get("threshold_abstention", 0.0)),
        "n": int(probe.get("threshold_cal_n", 0)),
        "source": "held-out calibration split",
    }


def _pack_json(obj) -> np.ndarray:
    """Store a small JSON blob as a uint8 array (npz stays pickle-free)."""
    return np.frombuffer(json.dumps(obj).encode("utf-8"), dtype=np.uint8)


def _unpack_json(arr) -> dict:
    return json.loads(bytes(np.asarray(arr, dtype=np.uint8)).decode("utf-8"))


def _tool_eval_arrays(
    checkpoint: str,
    tokenizer_path: str,
    probe_path: str,
    tool_probe_path: str,
    tool_eval_path: str,
    device: str,
):
    """Return (gold, tool_pred, keyword_pred) label lists for tool_eval.jsonl."""
    model = ProbeDecisionModel(checkpoint, tokenizer_path, probe_path, device)
    if not model.load_tool_heads(tool_probe_path):
        return None
    gold, tool_pred, keyword_pred = [], [], []
    with open(tool_eval_path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            query, label = row.get("input"), row.get("label")
            if not query or label not in TOOL_CLASSES:
                continue
            predicted = model.predict_tool(query)
            gold.append(label)
            tool_pred.append(predicted[0] if predicted else "chat")
            # Pure keyword decision: no Wolfram/DuckDuckGo calls.
            keyword_pred.append(keyword_tool(query))
    return gold, tool_pred, keyword_pred


def _tool_keyword_diff(gold, tool_pred, keyword_pred, n_boot: int, seed: int) -> dict:
    """Paired bootstrap of tool-head minus keyword decisions, per metric."""
    if not gold:
        return {}
    n = len(gold)
    target = torch.tensor([TOOL_CLASSES.index(v) for v in gold])
    tools = torch.tensor([TOOL_CLASSES.index(v) for v in tool_pred])
    keywords = torch.tensor([TOOL_CLASSES.index(v) for v in keyword_pred])
    correct_tool = (tools == target).float()
    correct_keyword = (keywords == target).float()
    specs = {
        "accuracy": (
            lambda idx: float(correct_keyword[idx].mean()),
            lambda idx: float(correct_tool[idx].mean()),
        ),
        "macro_f1": (
            lambda idx: float(macro_f1(keywords[idx], target[idx], len(TOOL_CLASSES))),
            lambda idx: float(macro_f1(tools[idx], target[idx], len(TOOL_CLASSES))),
        ),
    }
    out = {}
    for offset, (name, (stat_keyword, stat_tool)) in enumerate(specs.items()):
        res = paired_bootstrap_diff(
            stat_keyword, stat_tool, n, n_boot=max(1, n_boot), seed=seed + offset
        )
        out[name] = {"diff": round(res["diff"], 4), "ci": [round(v, 4) for v in res["ci"]]}
    out["tool_accuracy"] = round(float(correct_tool.mean()), 4)
    out["keyword_accuracy"] = round(float(correct_keyword.mean()), 4)
    out["n"] = n
    return out


def _save_items(path: str, items: dict) -> None:
    """Save per-item scores so a second run can recompute without a model."""
    arrays = {
        "method": np.array(items["method"]),
        "n_cal": np.array(items["n_cal"]),
        "labels": np.array(items["labels"], dtype="<U1"),
        "fine_types": np.array(items["fine_types"], dtype="<U24"),
        "y_confused": np.array(items["y_confused"], dtype=np.int64),
        "best_t": np.array(items["best_t"]),
        "gate_json": _pack_json(items.get("gate_calibration") or {}),
        "order_flip": _pack_json(items.get("order_flip") or {}),
        "tool_gold": np.array(items.get("tool_gold") or [], dtype="<U8"),
        "tool_pred": np.array(items.get("tool_pred") or [], dtype="<U8"),
        "keyword_pred": np.array(items.get("keyword_pred") or [], dtype="<U8"),
    }
    for key in ("raw7", "rawb", "raw21", "logits", "p_yes_test"):
        value = items.get(key)
        if value is not None:
            arrays[key] = np.asarray(value, dtype=np.float64)
    for key in (
        "best_t_bin", "temperature21", "clarify_threshold",
        "escalation_threshold", "target_error",
    ):
        if key in items:
            arrays[key] = np.array(items[key])
    np.savez_compressed(path, **arrays)


def _report_from_npz(path: str, bootstrap: int, seed: int) -> dict:
    """Recompute the whole report from a saved per-item dump, no model pass."""
    with np.load(path, allow_pickle=False) as d:
        method = str(np.asarray(d["method"]).item())
        n_cal = int(np.asarray(d["n_cal"]).item())
        y_true = [str(v) for v in d["labels"]]
        fine_types = [str(v) for v in d["fine_types"]]
        y_confused = [int(v) for v in d["y_confused"]]
        y_test = y_true[n_cal:]
        best_t = float(np.asarray(d["best_t"]).item())
        gate = _unpack_json(d["gate_json"])
        order_flip = _unpack_json(d["order_flip"])
        tool_gold = [str(v) for v in d["tool_gold"]]
        tool_pred = [str(v) for v in d["tool_pred"]]
        keyword_pred = [str(v) for v in d["keyword_pred"]]
        p_yes_test = [float(v) for v in d["p_yes_test"]] if "p_yes_test" in d else []

        if method == "probe":
            raw7 = torch.tensor(np.asarray(d["raw7"]), dtype=torch.float64)
            rawb = torch.tensor(np.asarray(d["rawb"]), dtype=torch.float64) if "rawb" in d else None
            raw21 = torch.tensor(np.asarray(d["raw21"]), dtype=torch.float64) if "raw21" in d else None
            best_t_bin = float(np.asarray(d["best_t_bin"]).item()) if "best_t_bin" in d else 1.0
            clarify = float(np.asarray(d["clarify_threshold"]).item())
            escalation = float(np.asarray(d["escalation_threshold"]).item())
            target_error = float(np.asarray(d["target_error"]).item())

            probas_t = torch.softmax(raw7[n_cal:] / best_t, dim=-1).tolist()
            probas = [{letter: float(p) for letter, p in zip(OPTION_LETTERS, row)} for row in probas_t]
            y_pred = [max(OPTION_LETTERS, key=lambda letter: p[letter]) for p in probas]
            report = {
                "probed": summarize(y_test, y_pred, probas, best_t, n_boot=bootstrap, seed=seed),
                "cal_n": n_cal,
            }
            report["probed"]["cal_n"] = n_cal
            uncal = torch.softmax(raw7[n_cal:], dim=-1).tolist()
            uncal_probas = [
                {letter: float(p) for letter, p in zip(OPTION_LETTERS, row)} for row in uncal
            ]
            report["paired"] = paired_report(
                uncal_probas, probas, y_test, n_boot=bootstrap, seed=seed + 10
            )

            if rawb is not None:
                cal_pb = torch.softmax(rawb[:n_cal] / best_t_bin, dim=-1)[:, 1]
                c_cal = torch.tensor(y_confused[:n_cal])
                best_thr, best_f1 = 0.5, -1.0
                for thr in [0.3, 0.5, 0.7, 0.8, 0.9, 0.95]:
                    pr = (cal_pb >= thr).long()
                    tp = int(((pr == 1) & (c_cal == 1)).sum())
                    fp = int(((pr == 1) & (c_cal == 0)).sum())
                    fn = int(((pr == 0) & (c_cal == 1)).sum())
                    f1 = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
                    if f1 > best_f1:
                        best_f1, best_thr = f1, thr
                test_pb = torch.softmax(rawb[n_cal:] / best_t_bin, dim=-1)[:, 1]
                bin_pred = [1 if p >= best_thr else 0 for p in test_pb]
                yc = y_confused[n_cal:]
                tp = sum(1 for t, p in zip(yc, bin_pred) if t == p == 1)
                fp = sum(1 for t, p in zip(yc, bin_pred) if t == 0 and p == 1)
                fn = sum(1 for t, p in zip(yc, bin_pred) if t == 1 and p == 0)
                report["confused_detector"] = {
                    "threshold": best_thr,
                    "precision": round(tp / (tp + fp), 4) if tp + fp else 0.0,
                    "recall": round(tp / (tp + fn), 4) if tp + fn else 0.0,
                    "support": sum(yc),
                    "temperature": best_t_bin,
                }
                report["confused_detector"].update(
                    _confused_ci(test_pb, yc, best_thr, bootstrap, seed + 20)
                )
                conf_test = [max(p.values()) for p in probas]
                p_yes = p_yes_test or [float(v) for v in test_pb]
                clarify_flags = [c < clarify for c in conf_test]
                escalate_flags = [p >= escalation for p in p_yes]
                in_scope = [i for i, label in enumerate(yc) if label == 0]
                out_scope = [i for i, label in enumerate(yc) if label == 1]
                report["gate_calibration"] = {
                    "clarify_threshold": clarify,
                    "escalation_threshold": escalation,
                    "target_error": target_error,
                    "clarify_coverage": round(
                        sum(not clarify_flags[i] and not escalate_flags[i] for i in in_scope)
                        / len(in_scope), 4
                    ) if in_scope else 0.0,
                    "escalation_coverage": round(
                        sum(escalate_flags[i] for i in out_scope) / len(out_scope), 4
                    ) if out_scope else 0.0,
                    "abstention_rate": round(
                        sum(a or b for a, b in zip(clarify_flags, escalate_flags)) / len(yc), 4
                    ) if yc else 0.0,
                    "n": len(yc),
                    "source": "test split",
                }

            if raw21 is not None and len(fine_types) == len(raw21):
                y21 = torch.tensor([FINE21_ORDER.index(t) for t in fine_types])
                best_t21 = _fit_or_default(
                    raw21[:n_cal], y21[:n_cal],
                    float(np.asarray(d["temperature21"]).item()) if "temperature21" in d else 1.0,
                )
                pred21 = raw21[n_cal:].argmax(-1).tolist()
                true21 = y21[n_cal:].tolist()
                acc21 = sum(1 for t, p in zip(true21, pred21) if t == p) / len(true21)
                f1s, per_fine = [], {}
                for i, fine in enumerate(FINE21_ORDER):
                    tp = sum(1 for t, p in zip(true21, pred21) if t == p == i)
                    fp = sum(1 for t, p in zip(true21, pred21) if t != i == p)
                    fn = sum(1 for t, p in zip(true21, pred21) if t == i != p)
                    prec = tp / (tp + fp) if tp + fp else 0.0
                    rec = tp / (tp + fn) if tp + fn else 0.0
                    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
                    f1s.append(f1)
                    per_fine[fine] = {"f1": round(f1, 4), "support": sum(1 for t in true21 if t == i)}
                report["fine21"] = {
                    "temperature": best_t21,
                    "accuracy": round(acc21, 4),
                    "macro_f1": round(sum(f1s) / len(f1s), 4),
                    "per_fine": per_fine,
                }
            report["calibrated"] = report["probed"]
        else:
            logits_matrix = torch.tensor(np.asarray(d["logits"]), dtype=torch.float64)
            def probas_at(t: float, part: slice) -> list[dict[str, float]]:
                scaled = torch.softmax(logits_matrix[part] / t, dim=-1).tolist()
                return [{letter: float(p) for letter, p in zip(OPTION_LETTERS, row)} for row in scaled]
            def preds_of(probas: list[dict[str, float]]) -> list[str]:
                return [max(OPTION_LETTERS, key=lambda letter: p[letter]) for p in probas]
            base = probas_at(1.0, slice(n_cal, None))
            calibrated = probas_at(best_t, slice(n_cal, None))
            report = {
                "uncalibrated": summarize(y_test, preds_of(base), base, 1.0, n_boot=bootstrap, seed=seed),
                "calibrated": summarize(
                    y_test, preds_of(calibrated), calibrated, best_t,
                    n_boot=bootstrap, seed=seed + 1,
                ),
                "paired": paired_report(base, calibrated, y_test, n_boot=bootstrap, seed=seed + 2),
                "cal_n": n_cal,
            }
            if order_flip:
                report["order_flip"] = order_flip
            if gate:
                report["gate_calibration"] = gate

    report["tool_keyword_diff"] = _tool_keyword_diff(
        tool_gold, tool_pred, keyword_pred, bootstrap, seed + 30
    )
    return report, method


def _write_report(report: dict, out_dir: str, method: str) -> dict:
    with open(os.path.join(out_dir, "report.json"), "w") as f:
        json.dump(report, f, indent=2)

    lines = [f"# Decision-model eval ({method})", ""]
    n_cal = int(report.get("cal_n", 0))
    n_test = sum(int(s.get("n", 0)) for k, s in report.items()
                 if k in ("uncalibrated", "calibrated", "probed"))
    lines.append(f"calibration split: {n_cal} rows; test split: {n_test} rows")
    lines.append("")
    if "confused_detector" in report:
        cd = report["confused_detector"]
        lines.append(
            f"confused detector @{cd['threshold']}: precision={cd['precision']} recall={cd['recall']} "
            f"support={cd['support']} (T={cd['temperature']})"
        )
        if "precision_ci" in cd:
            lines.append(f"precision 95% CI={cd['precision_ci']} recall 95% CI={cd['recall_ci']}")
        lines.append("")
    if "gate_calibration" in report:
        g = report["gate_calibration"]
        lines.append("## conformal gates (empirical coverage on the test split)")
        lines.append("")
        lines.append(
            f"source: {g.get('source', 'calibration split')}; "
            f"target error={g['target_error']}; n={g['n']}; "
            f"abstention rate={g['abstention_rate']}"
        )
        lines.append("")
        lines.append("| gate | threshold | coverage |")
        lines.append("| --- | ---: | ---: |")
        lines.append(
            f"| clarify (confidence below) | {g['clarify_threshold']} "
            f"| {g['clarify_coverage']} |"
        )
        lines.append(
            f"| escalation (p_yes at or above) | {g['escalation_threshold']} "
            f"| {g['escalation_coverage']} |"
        )
        lines.append("")
    if "order_flip" in report:
        of = report["order_flip"]
        lines.append("## order robustness (zero-shot)")
        lines.append("")
        lines.append(
            f"option order permuted by {of['permutation_shift']} of {len(OPTION_LETTERS)}; "
            f"L0 averages {of['rotations']} evenly spaced rotations; n={of['n']}"
        )
        lines.append("")
        lines.append("| variant | order-flip rate |")
        lines.append("| --- | --- |")
        lines.append(f"| raw (1 pass) | {of['raw']} |")
        lines.append(f"| L0 ({of['rotations']} passes) | {of['L0']} |")
        lines.append("")
    for key, s in report.items():
        if key in ("grid", "paired", "confused_detector", "gate_calibration", "fine21", "calibrated", "cal_n", "order_flip", "tool_keyword_diff"):
            continue
        lines.append(f"## {key} (T={s['temperature']}, n={s['n']})")
        lines.append(f"accuracy={s['accuracy']} nll={s['nll']} ece={s['ece']} macro_f1={s['macro_f1']}")
        if "accuracy_ci" in s:
            lines.append(
                f"95% CI: accuracy={s['accuracy_ci']} nll={s['nll_ci']} "
                f"ece={s['ece_ci']} macro_f1={s['macro_f1_ci']}"
            )
        lines.append("")
        lines.append("| opt | name | prec | rec | f1 | support |")
        lines.append("| --- | --- | --- | --- | --- | --- |")
        for letter in OPTION_LETTERS:
            c = s["per_class"][letter]
            lines.append(f"| {letter} | {c['name']} | {c['precision']} | {c['recall']} | {c['f1']} | {c['support']} |")
        lines.append("")
        lines.append("| bin | count | confidence | accuracy |")
        lines.append("| --- | --- | --- | --- |")
        for b in s["bins"]:
            lines.append(f"| {b['bin']} | {b['count']} | {b['confidence']} | {b['accuracy']} |")
        lines.append("")
    if report.get("paired"):
        lines.append("## paired: calibrated - uncalibrated")
        lines.append("")
        lines.append("| metric | diff | 95% CI |")
        lines.append("| --- | --- | --- |")
        for metric, res in report["paired"].items():
            lines.append(f"| {metric} | {res['diff']} | {res['ci']} |")
        lines.append("")
    if report.get("tool_keyword_diff"):
        d = report["tool_keyword_diff"]
        lines.append("## paired: tool head - keywords")
        lines.append("")
        lines.append(
            f"n={d['n']}; tool accuracy={d['tool_accuracy']}; "
            f"keyword accuracy={d['keyword_accuracy']}"
        )
        lines.append("")
        lines.append("| metric | diff (tool - keywords) | 95% CI |")
        lines.append("| --- | ---: | --- |")
        for metric in ("accuracy", "macro_f1"):
            res = d[metric]
            lines.append(f"| {metric} | {res['diff']} | {res['ci']} |")
        lines.append("")
    if "fine21" in report:
        f21 = report["fine21"]
        lines.append(f"## fine21 (T={f21['temperature']})")
        lines.append(f"accuracy={f21['accuracy']} macro_f1={f21['macro_f1']}")
        lines.append("")
        lines.append("| fine type | f1 | support |")
        lines.append("| --- | --- | --- |")
        for fine in FINE21_ORDER:
            c = f21["per_fine"][fine]
            lines.append(f"| {fine} | {c['f1']} | {c['support']} |")
        lines.append("")
    with open(os.path.join(out_dir, "report.md"), "w") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))
    return report


def run_eval(
    checkpoint: str = "checkpoints/best_model.pt",
    tokenizer_path: str = "data/tokenizer.json",
    eval_path: str = "data/eval.jsonl",
    max_n: int = 1000,
    device: str = "cpu",
    temperatures: list[float] | None = None,
    out_dir: str = "eval_out/decision",
    method: str = "logit",
    probe_path: str = "checkpoints/decision_probe.pt",
    cal_n: int = 1000,
    bootstrap: int = 1000,
    seed: int = 0,
    rotations: int = 2,
    from_npz: str | None = None,
    tool_probe_path: str = "checkpoints/decision_tool_probe.pt",
    tool_eval_path: str = "data/tool_eval.jsonl",
) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    if from_npz:
        logger.info("replaying npz=%s bootstrap=%d seed=%d", from_npz, bootstrap, seed)
        report, replay_method = _report_from_npz(from_npz, bootstrap, seed)
        _write_report(report, out_dir, replay_method)
        return report

    from hermitlm.training.train import accelerator_available

    resolved_device = torch.device(device)
    if resolved_device.type in ("cuda", "xpu") and not accelerator_available(resolved_device.type):
        raise RuntimeError(
            f"Eval was requested on {resolved_device.type} but that backend is "
            "not available. Pass --device cpu for a CPU-only run."
        )
    os.makedirs(out_dir, exist_ok=True)
    rows = _load_rows(eval_path, max_n, seed=seed)
    if len(rows) < 2:
        raise ValueError("need at least two eval rows")
    y_true = [FINE_TO_COARSE[r["type"]] for r in rows]
    fine_types = [r["type"] for r in rows]
    y_confused = [1 if r["type"] == "confused" else 0 for r in rows]
    n_cal = max(1, min(cal_n, len(rows) // 2))
    logger.info(
        "eval method=%s rows=%d cal=%d test=%d bootstrap=%d",
        method, len(rows), n_cal, len(rows) - n_cal, bootstrap,
    )

    if method == "probe":
        model = ProbeDecisionModel(checkpoint, tokenizer_path, probe_path, device)
        raw7, rawb, raw21 = [], [], []
        with torch.no_grad():
            def raw_scores(head, hidden, prior_key):
                s = head.scores(hidden.unsqueeze(0))[0]
                lp = torch.log_softmax(s, dim=-1)
                prior = getattr(model, "priors", {}).get(prior_key)
                if prior is not None:
                    lp = apply_prior(lp, prior.to(lp.device), getattr(model, "prior_lambda", 1.0))
                return lp

            for r in rows:
                h = model._hidden(r["input"], model.layer)
                raw7.append(raw_scores(model.head7, h, "coarse"))
                rawb.append(raw_scores(model.head_bin, h, "confused"))
                if model.head21 is not None:
                    raw21.append(raw_scores(model.head21, h, "fine"))
        raw7 = torch.stack(raw7)
        rawb = torch.stack(rawb)
        y_all = torch.tensor(
            [OPTION_LETTERS.index(FINE_TO_COARSE[r["type"]]) for r in rows],
            device=model.device,
        )
        c_all = torch.tensor(y_confused, device=model.device)
        best_t = _fit_or_default(raw7[:n_cal], y_all[:n_cal], float(model.temperature7))
        best_t_bin = _fit_or_default(rawb[:n_cal], c_all[:n_cal], float(model.temperature_bin))

        # threshold for the binary head, chosen on the calibration split
        cal_pb = torch.softmax(rawb[:n_cal] / best_t_bin, dim=-1)[:, 1]
        c_cal = c_all[:n_cal]
        best_thr, best_f1 = 0.5, -1.0
        for thr in [0.3, 0.5, 0.7, 0.8, 0.9, 0.95]:
            pr = (cal_pb >= thr).long()
            tp = int(((pr == 1) & (c_cal == 1)).sum())
            fp = int(((pr == 1) & (c_cal == 0)).sum())
            fn = int(((pr == 0) & (c_cal == 1)).sum())
            f1 = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
            if f1 > best_f1:
                best_f1, best_thr = f1, thr

        # report on the disjoint test half
        probas_t = torch.softmax(raw7[n_cal:] / best_t, dim=-1).tolist()
        probas = [{letter: float(p) for letter, p in zip(OPTION_LETTERS, row)} for row in probas_t]
        y_test = y_true[n_cal:]
        y_pred = [max(OPTION_LETTERS, key=lambda letter: p[letter]) for p in probas]
        report = {
            "probed": summarize(y_test, y_pred, probas, best_t, n_boot=bootstrap, seed=seed),
            "cal_n": n_cal,
        }
        report["probed"]["cal_n"] = n_cal
        uncal = torch.softmax(raw7[n_cal:], dim=-1).tolist()
        uncal_probas = [
            {letter: float(p) for letter, p in zip(OPTION_LETTERS, row)} for row in uncal
        ]
        report["paired"] = paired_report(
            uncal_probas, probas, y_test, n_boot=bootstrap, seed=seed + 10
        )

        test_pb = torch.softmax(rawb[n_cal:] / best_t_bin, dim=-1)[:, 1]
        bin_pred = [1 if p >= best_thr else 0 for p in test_pb]
        yc = c_all[n_cal:].tolist()
        tp = sum(1 for t, p in zip(yc, bin_pred) if t == p == 1)
        fp = sum(1 for t, p in zip(yc, bin_pred) if t == 0 and p == 1)
        fn = sum(1 for t, p in zip(yc, bin_pred) if t == 1 and p == 0)
        report["confused_detector"] = {
            "threshold": best_thr,
            "precision": round(tp / (tp + fp), 4) if tp + fp else 0.0,
            "recall": round(tp / (tp + fn), 4) if tp + fn else 0.0,
            "support": sum(yc),
            "temperature": best_t_bin,
        }
        report["confused_detector"].update(
            _confused_ci(test_pb, yc, best_thr, bootstrap, seed + 20)
        )

        # Empirical coverage of the stored conformal gates on the held-out
        # test half. Nothing here refits a threshold: the cutoffs came from
        # scripts/fit_decision_thresholds.py and ride in the probe artifact.
        p_yes_test = []
        for i, row in enumerate(rows[n_cal:]):
            yes = model.predict_yes(row["input"]) if model.has_yes_head else None
            p_yes_test.append(yes[0] if yes is not None else float(test_pb[i]))
        conf_test = [max(p.values()) for p in probas]
        clarify_flags = [c < model.clarify_threshold for c in conf_test]
        escalate_flags = [p >= model.escalation_threshold for p in p_yes_test]
        in_scope = [i for i, label in enumerate(yc) if label == 0]
        out_scope = [i for i, label in enumerate(yc) if label == 1]
        report["gate_calibration"] = {
            "clarify_threshold": model.clarify_threshold,
            "escalation_threshold": model.escalation_threshold,
            "target_error": model.target_error,
            "clarify_coverage": round(
                sum(not clarify_flags[i] and not escalate_flags[i] for i in in_scope)
                / len(in_scope), 4
            ) if in_scope else 0.0,
            "escalation_coverage": round(
                sum(escalate_flags[i] for i in out_scope) / len(out_scope), 4
            ) if out_scope else 0.0,
            "abstention_rate": round(
                sum(a or b for a, b in zip(clarify_flags, escalate_flags)) / len(yc), 4
            ) if yc else 0.0,
            "n": len(yc),
            "source": "test split",
        }

        if raw21:
            raw21_t = torch.stack(raw21)
            y21 = torch.tensor(
                [FINE21_ORDER.index(r["type"]) for r in rows], device=model.device
            )
            best_t21 = _fit_or_default(
                raw21_t[:n_cal], y21[:n_cal], float(model.temperature21)
            )
            pred21 = raw21_t[n_cal:].argmax(-1).tolist()
            true21 = y21[n_cal:].tolist()
            acc21 = sum(1 for t, p in zip(true21, pred21) if t == p) / len(true21)
            f1s = []
            per_fine = {}
            for i, fine in enumerate(FINE21_ORDER):
                tp = sum(1 for t, p in zip(true21, pred21) if t == p == i)
                fp = sum(1 for t, p in zip(true21, pred21) if t != i == p)
                fn = sum(1 for t, p in zip(true21, pred21) if t == i != p)
                prec = tp / (tp + fp) if tp + fp else 0.0
                rec = tp / (tp + fn) if tp + fn else 0.0
                f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
                f1s.append(f1)
                per_fine[fine] = {"f1": round(f1, 4), "support": sum(1 for t in true21 if t == i)}
            report["fine21"] = {
                "temperature": best_t21,
                "accuracy": round(acc21, 4),
                "macro_f1": round(sum(f1s) / len(f1s), 4),
                "per_fine": per_fine,
            }
        report["calibrated"] = report["probed"]
        saved_items = {
            "method": method,
            "n_cal": n_cal,
            "labels": y_true,
            "fine_types": fine_types,
            "y_confused": y_confused,
            "best_t": best_t,
            "best_t_bin": best_t_bin,
            "raw7": raw7.detach().cpu(),
            "rawb": rawb.detach().cpu(),
            "raw21": torch.stack(raw21).detach().cpu() if raw21 else None,
            "p_yes_test": p_yes_test,
            "temperature21": float(model.temperature21),
            "clarify_threshold": float(model.clarify_threshold),
            "escalation_threshold": float(model.escalation_threshold),
            "target_error": float(model.target_error),
            "gate_calibration": report["gate_calibration"],
        }
    else:
        model = DecisionModel(checkpoint, tokenizer_path, device)
        order_flip = _order_flip_rate(rows, model, rotations=rotations)
        # Collect base (T=1) logits one time.
        raw_logits: list[dict[str, float]] = []
        with torch.no_grad():
            for r in rows:
                lp = _zero_shot_log_probs(model, r["input"], 0)
                raw_logits.append(
                    {letter: float(v) for letter, v in zip(OPTION_LETTERS, lp.tolist())}
                )

        logits_matrix = torch.tensor(
            [[raw[letter] for letter in OPTION_LETTERS] for raw in raw_logits],
            dtype=torch.float64,
        )
        targets = torch.tensor([OPTION_LETTERS.index(t) for t in y_true])
        best_t = _fit_or_default(logits_matrix[:n_cal], targets[:n_cal], 1.0)
        if temperatures:
            # explicit overrides keep the old behaviour for the grid sweep
            best_t = min(temperatures, key=lambda t: nll(
                torch.softmax(logits_matrix[:n_cal] / t, dim=-1), targets[:n_cal]
            ))

        def probas_at(t: float, part: slice) -> list[dict[str, float]]:
            out = []
            for logits in logits_matrix[part]:
                scaled = torch.softmax(logits / t, dim=-1).tolist()
                out.append({letter: float(p) for letter, p in zip(OPTION_LETTERS, scaled)})
            return out

        def preds_of(probas: list[dict[str, float]]) -> list[str]:
            return [max(OPTION_LETTERS, key=lambda letter: p[letter]) for p in probas]

        test_part = slice(n_cal, None)
        y_test = y_true[n_cal:]
        base = probas_at(1.0, test_part)
        calibrated = probas_at(best_t, test_part)
        report = {
            "uncalibrated": summarize(
                y_test, preds_of(base), base, 1.0, n_boot=bootstrap, seed=seed
            ),
            "calibrated": summarize(
                y_test, preds_of(calibrated), calibrated, best_t, n_boot=bootstrap, seed=seed + 1
            ),
            "paired": paired_report(base, calibrated, y_test, n_boot=bootstrap, seed=seed + 2),
            "cal_n": n_cal,
            "order_flip": order_flip,
        }
        stored_gates = _stored_gate_report(probe_path)
        if stored_gates is not None:
            report["gate_calibration"] = stored_gates
        saved_items = {
            "method": method,
            "n_cal": n_cal,
            "labels": y_true,
            "fine_types": fine_types,
            "y_confused": y_confused,
            "best_t": best_t,
            "logits": logits_matrix.detach().cpu(),
            "order_flip": order_flip,
            "gate_calibration": report.get("gate_calibration", {}),
        }

    tool = _tool_eval_arrays(
        checkpoint, tokenizer_path, probe_path, tool_probe_path, tool_eval_path, device
    )
    if tool:
        gold, tool_pred, keyword_pred = tool
        saved_items.update(
            tool_gold=gold, tool_pred=tool_pred, keyword_pred=keyword_pred
        )
        report["tool_keyword_diff"] = _tool_keyword_diff(
            gold, tool_pred, keyword_pred, bootstrap, seed + 30
        )
    else:
        report["tool_keyword_diff"] = {}
    _save_items(os.path.join(out_dir, "items.npz"), saved_items)
    _write_report(report, out_dir, method)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Eval + calibrate the decision model")
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pt")
    parser.add_argument("--tokenizer", default="data/tokenizer.json")
    parser.add_argument("--eval", default="data/eval.jsonl")
    parser.add_argument("--max-n", type=int, default=1000)
    # Inference/calibration is small and runs more reliably on CPU; the
    # Intel XPU backend can crash in SYCL on some of these tensor ops.
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out-dir", default="eval_out/decision")
    parser.add_argument("--method", choices=["logit", "probe"], default="logit")
    parser.add_argument("--probe", default="checkpoints/decision_probe.pt")
    parser.add_argument("--cal-n", type=int, default=1000)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rotations", type=int, default=2, help="L0 rotations for the zero-shot order-flip report")
    parser.add_argument("--from-npz", default=None, help="recompute metrics from items.npz, no model pass")
    parser.add_argument("--tool-probe", default="checkpoints/decision_tool_probe.pt")
    parser.add_argument("--tool-eval", default="data/tool_eval.jsonl")
    args = parser.parse_args()
    run_eval(
        args.checkpoint,
        args.tokenizer,
        args.eval,
        args.max_n,
        args.device,
        out_dir=args.out_dir,
        method=args.method,
        probe_path=args.probe,
        cal_n=args.cal_n,
        bootstrap=args.bootstrap,
        seed=args.seed,
        rotations=args.rotations,
        from_npz=args.from_npz,
        tool_probe_path=args.tool_probe,
        tool_eval_path=args.tool_eval,
    )