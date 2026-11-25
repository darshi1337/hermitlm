"""Fit the optional yes/no and reply-score heads into an existing probe.

Both heads are trained on frozen final-layer hidden states and stored in the
same artifact as head7/head21, so the existing SHA-256 pinning keeps rejecting
a head fitted for a different checkpoint or tokenizer.

Usage:
    python scripts/train_yes_score_heads.py --max-n 12000
    python scripts/export_probe.py
"""
import argparse
import json
import random

import torch
from tokenizers import Tokenizer

from hermitlm.config import HermitConfig
from hermitlm.tools.decision_model import (
    FINE_TO_COARSE,
    chat_prompt,
    get_label_ids,
    score_prompt,
)
from hermitlm.tools.heads import select_head
from hermitlm.tools.prior import fit_prior
from hermitlm.training.model import HermitLM
from hermitlm.training.reward import context_score


def read_rows(path: str, max_n: int) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("input") and row.get("type") in FINE_TO_COARSE:
                rows.append(row)
            if max_n and len(rows) >= max_n:
                break
    return rows


@torch.no_grad()
def extract_final_hidden(
    model: HermitLM,
    tokenizer: Tokenizer,
    prompts: list[str],
    max_seq_len: int,
    device: torch.device,
    batch_size: int = 64,
) -> torch.Tensor:
    """Final normed hidden state at the last real token, shape (n, d)."""
    feats = []
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start : start + batch_size]
        ids = [tokenizer.encode(text).ids[-max_seq_len:] for text in batch]
        width = max(len(row) for row in ids)
        padded = [row + [0] * (width - len(row)) for row in ids]
        idx = torch.tensor(padded, dtype=torch.long, device=device)
        _, _, hidden = model(idx, return_hidden=True)
        lengths = torch.tensor([len(row) for row in ids], dtype=torch.long)
        # Advanced indexing on Intel XPU hits Level Zero limits: gather on CPU.
        feats.append(hidden.cpu()[torch.arange(len(batch)), lengths - 1, :])
    return torch.cat(feats)


@torch.no_grad()
def generate_replies(
    model: HermitLM,
    tokenizer: Tokenizer,
    queries: list[str],
    max_seq_len: int,
    device: torch.device,
    max_tokens: int = 48,
    temperature: float = 0.6,
    top_k: int = 20,
    batch_size: int = 32,
) -> list[str]:
    """Sample a reply per query so score-head inputs match runtime candidates."""
    texts: list[str] = []
    stop_ids = {getattr(model.config, "eos_id", None), getattr(model.config, "bos_id", None)}
    for start in range(0, len(queries), batch_size):
        if start and start % (batch_size * 10) == 0:
            print(f"generated {start}/{len(queries)} score-training replies", flush=True)
        batch = queries[start : start + batch_size]
        ids = [tokenizer.encode(chat_prompt(query)).ids[-max_seq_len:] for query in batch]
        width = max(len(row) for row in ids)
        padded = [row + [0] * (width - len(row)) for row in ids]
        idx = torch.tensor(padded, dtype=torch.long, device=device)
        out = model.generate(idx, max_new_tokens=max_tokens, temperature=temperature, top_k=top_k)
        for i, row in enumerate(ids):
            generated = out[i, len(row) :].tolist()
            for j, token in enumerate(generated):
                if token in stop_ids:
                    generated = generated[:j]
                    break
            texts.append(tokenizer.decode(generated).strip())
    return texts


def balance(rows: list[dict], key, per_key: int, seed: int) -> list[dict]:
    buckets: dict[int, list[dict]] = {}
    for row in rows:
        buckets.setdefault(key(row), []).append(row)
    rng = random.Random(seed)
    chosen = []
    for group in buckets.values():
        rng.shuffle(group)
        chosen.extend(group[:per_key])
    rng.shuffle(chosen)
    return chosen


def fit_head(features: torch.Tensor, targets: list[int], n_classes: int, seed: int):
    """Fit at the final layer (-1) so the browser can use the same weights."""
    hidden = features.to(torch.float64).unsqueeze(1)
    head = select_head(hidden, torch.tensor(targets, dtype=torch.long), n_classes, [-1], k=5, seed=seed)
    prior = fit_prior(head.log_probs(features.to(torch.float64))).tolist()
    print(
        f"head fitted: classes={n_classes} layer={head.layer} solver={head.solver} "
        f"alpha={head.alpha} T={head.temperature:.4f} oof_nll={head.oof_nll:.4f} "
        f"oof_acc={head.oof_accuracy:.4f}"
    )
    return head, prior


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pt")
    parser.add_argument("--tokenizer", default="data/tokenizer.json")
    parser.add_argument("--probe", default="checkpoints/decision_probe.pt")
    parser.add_argument("--train", default="data/train.jsonl")
    parser.add_argument("--max-n", type=int, default=12000)
    parser.add_argument("--per-class", type=int, default=1500)
    parser.add_argument("--score-levels", type=int, default=5)
    parser.add_argument("--score-source", choices=("generated", "gold"), default="generated")
    parser.add_argument("--score-n", type=int, default=6000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.score_levels < 2:
        raise SystemExit("--score-levels must be at least 2")
    from hermitlm.training.train import accelerator_available

    device = torch.device(args.device)
    if device.type in ("cuda", "xpu") and not accelerator_available(device.type):
        raise SystemExit(f"requested {device.type} but it is unavailable; pass --device cpu")

    tokenizer = Tokenizer.from_file(args.tokenizer)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    config = HermitConfig(**ckpt["config"]) if "config" in ckpt else HermitConfig()
    config.vocab_size = tokenizer.get_vocab_size()
    model = HermitLM(config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    rows = read_rows(args.train, args.max_n)
    if not rows:
        raise SystemExit(f"no usable rows in {args.train}")

    probe = torch.load(args.probe, map_location="cpu", weights_only=True)
    priors = dict(probe.get("priors", {}) or {})

    # --- yes/no: class 1 = "yes, this is out of scope" -----------------------
    yes_rows = balance(
        rows, lambda r: 1 if r["type"] == "confused" else 0, args.per_class, args.seed
    )
    yes_targets = [1 if r["type"] == "confused" else 0 for r in yes_rows]
    print(f"yes head: {len(yes_rows)} rows ({sum(yes_targets)} out-of-scope)")
    yes_feats = extract_final_hidden(
        model,
        tokenizer,
        [r["input"] for r in yes_rows],
        config.max_seq_len,
        device,
        args.batch_size,
    )
    yes_head, yes_prior = fit_head(yes_feats, yes_targets, 2, args.seed)
    priors["yes"] = yes_prior

    # --- score: calibrated levels over the context-aware reward --------------
    score_rows = rows[: args.score_n]
    if args.score_source == "generated":
        replies = generate_replies(
            model, tokenizer, [r["input"] for r in score_rows],
            config.max_seq_len, device,
        )
        pairs = [
            {"row": row, "reply": reply}
            for row, reply in zip(score_rows, replies)
            if reply.strip()
        ]
    else:
        pairs = [
            {"row": row, "reply": row["output"]}
            for row in score_rows if row["output"].strip()
        ]
    if len(pairs) < args.score_levels:
        raise SystemExit("not enough non-empty replies to fit the score head")
    scores = [context_score(pair["row"]["input"], pair["reply"]) for pair in pairs]

    # Rank-based bins keep the level centres strictly increasing even when a
    # lot of replies share the same heuristic score (empty replies score -3).
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    levels = [0] * len(scores)
    for rank, index in enumerate(order):
        levels[index] = min(args.score_levels - 1, rank * args.score_levels // len(scores))
    thresholds = []
    for level in range(1, args.score_levels):
        prev_max = max(score for score, target in zip(scores, levels) if target == level - 1)
        next_min = min(score for score, target in zip(scores, levels) if target == level)
        thresholds.append((prev_max + next_min) / 2)
    centers = []
    for level in range(args.score_levels):
        values = [score for score, target in zip(scores, levels) if target == level]
        centers.append(sum(values) / len(values))
    scored = [dict(pair, level=level) for pair, level in zip(pairs, levels)]
    score_pairs = balance(scored, lambda item: item["level"], args.per_class, args.seed + 1)
    labels, _ = get_label_ids(tokenizer, [str(k) for k in range(args.score_levels)])
    print(
        f"score head: {len(score_pairs)} {args.score_source} rows, thresholds="
        f"{[round(v, 4) for v in thresholds]}, centers={[round(v, 4) for v in centers]}, "
        f"labels={labels}"
    )
    score_feats = extract_final_hidden(
        model,
        tokenizer,
        [score_prompt(pair["row"]["input"], pair["reply"]) for pair in score_pairs],
        config.max_seq_len,
        device,
        args.batch_size,
    )
    score_targets = [pair["level"] for pair in score_pairs]
    score_head, score_prior = fit_head(
        score_feats, score_targets, args.score_levels, args.seed + 2
    )
    priors["score"] = score_prior

    probe["priors"] = priors
    selection = probe.get("selection")
    if selection:
        selection.setdefault("heads", {})["yes"] = yes_head.as_dict()
        selection["heads"]["score"] = score_head.as_dict()
    else:
        probe["head_yes"] = yes_head.as_dict()
        probe["head_score"] = score_head.as_dict()
    probe["score_levels"] = centers
    probe["score_thresholds"] = thresholds
    probe["score_labels"] = labels
    torch.save(probe, args.probe)
    print(f"saved yes + score heads to {args.probe}")
    print("run `python scripts/export_probe.py` to refresh web/probe.json")


if __name__ == "__main__":
    main()
