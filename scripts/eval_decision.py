"""Zero-shot eval + temperature calibration for the decision model.

Mirrors the article's eval: accuracy, per-class precision/recall/F1 and a
confidence-vs-accuracy bin table, then a grid search over temperature T
(minimizing NLL) to fix the expected overconfidence.

Usage:
    conda run -n hermit python scripts/eval_decision.py \
        --checkpoint checkpoints/best_model.pt --max-n 1000 --out-dir eval_out/decision
"""

import argparse
import json
import math
import os

import torch

from hermitlm.tools.decision_model import (
    COARSE_NAMES,
    FINE_TO_COARSE,
    OPTION_LETTERS,
    DecisionModel,
    ProbeDecisionModel,
    format_prompt,
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


def summarize(
    y_true: list[str], y_pred: list[str], probas: list[dict[str, float]], temperature: float
) -> dict:
    n = len(y_true)
    corrects = [t == p for t, p in zip(y_true, y_pred)]
    confs = [probas[i][y_pred[i]] for i in range(n)]
    nll = -sum(math.log(max(probas[i][y_true[i]], 1e-12)) for i in range(n)) / n
    ece_bins = _bins(confs, corrects)
    ece = sum(b["count"] / n * abs(b["confidence"] - b["accuracy"]) for b in ece_bins)
    # out-of-scope detector: confused items recalled into coarse 'g' (mind)
    return {
        "temperature": temperature,
        "n": n,
        "accuracy": round(sum(corrects) / n, 4),
        "nll": round(nll, 4),
        "ece": round(ece, 4),
        "macro_f1": round(sum(v["f1"] for v in _prf(y_true, y_pred).values()) / len(OPTION_LETTERS), 4),
        "per_class": _prf(y_true, y_pred),
        "bins": ece_bins,
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
) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    rows = _load_rows(eval_path, max_n)
    y_true = [FINE_TO_COARSE[r["type"]] for r in rows]
    y_confused = [1 if r["type"] == "confused" else 0 for r in rows]

    if method == "probe":
        import torch.nn.functional as F

        model = ProbeDecisionModel(checkpoint, tokenizer_path, probe_path, device)
        raw7, rawb = [], []
        with torch.no_grad():
            for r in rows:
                h = model._hidden(r["input"])
                raw7.append(model.head7(h.unsqueeze(0))[0])
                rawb.append(model.head_bin(h.unsqueeze(0))[0])
        raw7 = torch.stack(raw7)
        rawb = torch.stack(rawb)
        y_all = torch.tensor([OPTION_LETTERS.index(FINE_TO_COARSE[r["type"]]) for r in rows])
        c_all = torch.tensor(y_confused)
        n_cal = min(cal_n, len(rows) // 2)
        # fit temperature (min NLL) and binary threshold (max F1) on cal half
        best_t, best_nll = 1.0, float("inf")
        for t in [0.3, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 5.0]:
            nll = F.cross_entropy(raw7[:n_cal] / t, y_all[:n_cal]).item()
            if nll < best_nll:
                best_nll, best_t = nll, t
        cal_pb = torch.softmax(rawb[:n_cal] / model.temperature_bin, dim=-1)[:, 1]
        best_thr, best_f1 = 0.5, -1.0
        for thr in [0.3, 0.5, 0.7, 0.8, 0.9, 0.95]:
            pr = (cal_pb >= thr).long()
            tp = ((pr == 1) & (c_all[:n_cal] == 1)).sum().item()
            fp = ((pr == 1) & (c_all[:n_cal] == 0)).sum().item()
            fn = ((pr == 0) & (c_all[:n_cal] == 1)).sum().item()
            f1 = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
            if f1 > best_f1:
                best_f1, best_thr = f1, thr
        # report on disjoint test half
        probas_t = torch.softmax(raw7[n_cal:] / best_t, dim=-1).tolist()
        probas = [{letter: float(p) for letter, p in zip(OPTION_LETTERS, row)} for row in probas_t]
        y_true = [OPTION_LETTERS[i] for i in y_all[n_cal:].tolist()]
        y_pred = [max(OPTION_LETTERS, key=lambda letter: p[letter]) for p in probas]
        report = {"probed": summarize(y_true, y_pred, probas, best_t)}
        report["probed"]["cal_n"] = n_cal
        test_pb = torch.softmax(rawb[n_cal:] / model.temperature_bin, dim=-1)[:, 1].tolist()
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
            "temperature": model.temperature_bin,
        }
        report["calibrated"] = report["probed"]
    else:
        model = DecisionModel(checkpoint, tokenizer_path, device)
        # Collect base (T=1) probas once; rescale analytically for the grid.
        base_probas: list[dict[str, float]] = []
        with torch.no_grad():
            for r in rows:
                ids = model.tokenizer.encode(format_prompt(r["input"])).ids[-model.config.max_seq_len :]
                idx = torch.tensor([ids], dtype=torch.long, device=model.device)
                logits, _ = model.model(idx)
                gathered = logits[0, -1, torch.tensor(model.option_ids)].float()
                base_probas.append({letter: float(v) for letter, v in zip(OPTION_LETTERS, gathered.tolist())})

        def probas_at(t: float) -> list[dict[str, float]]:
            out = []
            for raw in base_probas:
                logits = torch.tensor([raw[letter] for letter in OPTION_LETTERS])
                scaled = torch.softmax(logits / t, dim=-1).tolist()
                out.append({letter: float(p) for letter, p in zip(OPTION_LETTERS, scaled)})
            return out

        def preds_of(probas: list[dict[str, float]]) -> list[str]:
            return [max(OPTION_LETTERS, key=lambda letter: p[letter]) for p in probas]

        grid = temperatures or [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0]
        base = probas_at(1.0)
        report = {"uncalibrated": summarize(y_true, preds_of(base), base, 1.0), "grid": []}
        best = report["uncalibrated"]
        for t in grid:
            p = probas_at(t)
            s = summarize(y_true, preds_of(p), p, t)
            report["grid"].append({"temperature": t, "accuracy": s["accuracy"], "nll": s["nll"], "ece": s["ece"]})
            if s["nll"] < best["nll"]:
                best = s
        report["calibrated"] = best

    with open(os.path.join(out_dir, "report.json"), "w") as f:
        json.dump(report, f, indent=2)

    lines = [f"# Decision-model eval ({method})", ""]
    if "confused_detector" in report:
        cd = report["confused_detector"]
        lines.append(
            f"confused detector @{cd['threshold']}: precision={cd['precision']} recall={cd['recall']} "
            f"support={cd['support']} (T={cd['temperature']})"
        )
        lines.append("")
    for key in report:
        if key in ("grid", "confused_detector"):
            continue
        s = report[key]
        lines.append(f"## {key} (T={s['temperature']}, n={s['n']})")
        lines.append(f"accuracy={s['accuracy']} nll={s['nll']} ece={s['ece']} macro_f1={s['macro_f1']}")
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
    args = parser.parse_args()
    run_eval(args.checkpoint, args.tokenizer, args.eval, args.max_n, args.device, out_dir=args.out_dir, method=args.method, probe_path=args.probe, cal_n=args.cal_n)
