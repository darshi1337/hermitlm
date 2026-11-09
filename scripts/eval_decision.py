"""Zero-shot eval + temperature calibration for the decision model.

Mirrors the eval from the article: accuracy, per-class precision/recall/F1, and a
confidence-vs-accuracy bin table, plus ECE/NLL with 95% bootstrap CIs. A
temperature T is fitted on a disjoint calibration split by minimising NLL over
log T (binary heads use Platt-smoothed targets) and applied to the test split.

Usage:
    conda run -n hermit python scripts/eval_decision.py \
        --checkpoint checkpoints/best_model.pt --max-n 1000 --out-dir eval_out/decision
"""

import argparse
import json
import os

import torch

from hermitlm.tools.calibration import InsufficientDataError, fit_temperature
from hermitlm.tools.decision_model import (
    COARSE_NAMES,
    FINE21_ORDER,
    FINE_TO_COARSE,
    OPTION_LETTERS,
    DecisionModel,
    ProbeDecisionModel,
    format_prompt,
)
from hermitlm.tools.metrics import (
    accuracy,
    bootstrap_ci,
    ece,
    macro_f1,
    nll,
    paired_bootstrap_diff,
)

N_BINS = 10


def _load_rows(eval_path: str, max_n: int) -> list[dict]:
    rows = []
    with open(eval_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if d.get("input") and d.get("type") in FINE_TO_COARSE:
                rows.append(d)
            if len(rows) >= max_n:
                break
    return rows


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
    y = torch.tensor(labels)
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
) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    rows = _load_rows(eval_path, max_n)
    if len(rows) < 2:
        raise ValueError("need at least two eval rows")
    y_true = [FINE_TO_COARSE[r["type"]] for r in rows]
    y_confused = [1 if r["type"] == "confused" else 0 for r in rows]
    n_cal = max(1, min(cal_n, len(rows) // 2))

    if method == "probe":
        model = ProbeDecisionModel(checkpoint, tokenizer_path, probe_path, device)
        raw7, rawb, raw21 = [], [], []
        with torch.no_grad():
            for r in rows:
                h = model._hidden(r["input"])
                raw7.append(model.head7(h.unsqueeze(0))[0])
                rawb.append(model.head_bin(h.unsqueeze(0))[0])
                if model.head21 is not None:
                    raw21.append(model.head21(h.unsqueeze(0))[0])
        raw7 = torch.stack(raw7)
        rawb = torch.stack(rawb)
        y_all = torch.tensor([OPTION_LETTERS.index(FINE_TO_COARSE[r["type"]]) for r in rows])
        c_all = torch.tensor(y_confused)
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

        if raw21:
            raw21_t = torch.stack(raw21)
            y21 = torch.tensor([FINE21_ORDER.index(r["type"]) for r in rows])
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
    else:
        model = DecisionModel(checkpoint, tokenizer_path, device)
        # Collect base (T=1) logits one time.
        raw_logits: list[dict[str, float]] = []
        with torch.no_grad():
            for r in rows:
                ids = model.tokenizer.encode(format_prompt(r["input"])).ids[-model.config.max_seq_len :]
                idx = torch.tensor([ids], dtype=torch.long, device=model.device)
                logits, _ = model.model(idx)
                gathered = logits[0, -1, torch.tensor(model.option_ids)].float()
                raw_logits.append({letter: float(v) for letter, v in zip(OPTION_LETTERS, gathered.tolist())})

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
        }

    with open(os.path.join(out_dir, "report.json"), "w") as f:
        json.dump(report, f, indent=2)

    lines = [f"# Decision-model eval ({method})", ""]
    lines.append(f"calibration split: {n_cal} rows; test split: {len(rows) - n_cal} rows")
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
    for key in report:
        if key in ("grid", "paired", "confused_detector", "fine21", "calibrated", "cal_n"):
            continue
        s = report[key]
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Eval + calibrate the decision model")
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pt")
    parser.add_argument("--tokenizer", default="data/tokenizer.json")
    parser.add_argument("--eval", default="data/eval.jsonl")
    parser.add_argument("--max-n", type=int, default=1000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out-dir", default="eval_out/decision")
    parser.add_argument("--method", choices=["logit", "probe"], default="logit")
    parser.add_argument("--probe", default="checkpoints/decision_probe.pt")
    parser.add_argument("--cal-n", type=int, default=1000)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
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
    )
