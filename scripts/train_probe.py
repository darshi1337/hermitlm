"""Train a linear probe decision head on frozen HermitLM hidden states.

Zero-shot constrained decoding fails on this model. The classify prompt is
out-of-distribution. Option letters sit at near-tied low logits. Thus the
System-1 decision model is a tiny supervised head instead: one frozen
forward pass over the natural chat prompt, last-token hidden state, linear
layer -> 7 coarse intents. Same single-pass cost profile as the method from
the article, plus a binary confused-vs-rest (out-of-scope) head.

Usage:
    conda run -n hermit python scripts/train_probe.py \
        --checkpoint checkpoints/best_model.pt --per-class 1500 --epochs 30 \
        --out checkpoints/decision_probe.pt
"""

import argparse
import collections
import json
import random

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from hermitlm.config import HermitConfig
from hermitlm.tools.decision_model import FINE21_ORDER, FINE_TO_COARSE, OPTION_LETTERS
from hermitlm.training.model import HermitLM


def chat_prompt(query: str) -> str:
    return f"<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n"


@torch.no_grad()
def extract_hidden(
    model: HermitLM, tokenizer: Tokenizer, queries: list[str], max_seq_len: int, device: torch.device, batch_size: int = 64
) -> torch.Tensor:
    feats = []
    for i in range(0, len(queries), batch_size):
        batch = queries[i : i + batch_size]
        ids = [tokenizer.encode(chat_prompt(q)).ids[-max_seq_len:] for q in batch]
        width = max(len(s) for s in ids)
        pad = 0
        # right padding matches training (dataset.collate_fn); the true last
        # token sits at lengths-1, where learned positions stay in distribution
        padded = [s + [pad] * (width - len(s)) for s in ids]
        idx = torch.tensor(padded, dtype=torch.long, device=device)
        _, _, hidden = model(idx, return_hidden=True)
        lengths = torch.tensor([len(s) for s in ids], device=device)
        feats.append(hidden[torch.arange(len(batch), device=device), lengths - 1].cpu())
    return torch.cat(feats)


def fit_temperature(logits: torch.Tensor, targets: torch.Tensor) -> float:
    best_t, best_nll = 1.0, float("inf")
    for t in [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0]:
        nll = F.cross_entropy(logits / t, targets).item()
        if nll < best_nll:
            best_nll, best_t = nll, t
    # refine around best
    for t in [best_t * m for m in (0.8, 0.9, 1.1, 1.2)]:
        nll = F.cross_entropy(logits / t, targets).item()
        if nll < best_nll:
            best_nll, best_t = nll, t
    return best_t


def train_head(
    feats: torch.Tensor, targets: torch.Tensor, n_classes: int, epochs: int, seed: int
) -> tuple[torch.nn.Linear, float]:
    g = torch.Generator().manual_seed(seed)
    head = torch.nn.Linear(feats.size(1), n_classes)
    opt = torch.optim.Adam(head.parameters(), lr=1e-3, weight_decay=1e-4)
    # reuse feats as both train and val via a fixed 90/10 split
    n = len(feats)
    perm = torch.randperm(n, generator=g)
    split = int(0.9 * n)
    tr, va = perm[:split], perm[split:]
    Xtr, ytr, Xva, yva = feats[tr], targets[tr], feats[va], targets[va]
    batched = torch.arange(len(tr))
    for _ in range(epochs):
        head.train()
        order = batched[torch.randperm(len(tr), generator=g)]
        for i in range(0, len(tr), 512):
            b = order[i : i + 512]
            opt.zero_grad()
            F.cross_entropy(head(Xtr[b]), ytr[b]).backward()
            opt.step()
    head.eval()
    with torch.no_grad():
        temperature = fit_temperature(head(Xva), yva)
        pred = head(Xva).argmax(-1)
        acc = (pred == yva).float().mean().item()
    print(f"val acc={acc:.4f} T={temperature:.3f} (n_val={len(va)})")
    return head, temperature


def main() -> None:
    parser = argparse.ArgumentParser(description="Train linear probe decision heads")
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pt")
    parser.add_argument("--tokenizer", default="data/tokenizer.json")
    parser.add_argument("--train", default="data/train.jsonl")
    parser.add_argument("--per-class", type=int, default=1500)
    parser.add_argument("--per-fine", type=int, default=400)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", default="checkpoints/decision_probe.pt")
    args = parser.parse_args()

    random.seed(args.seed)
    device = torch.device(args.device)
    tokenizer = Tokenizer.from_file(args.tokenizer)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    config = HermitConfig(**ckpt["config"]) if "config" in ckpt else HermitConfig()
    config.vocab_size = tokenizer.get_vocab_size()
    model = HermitLM(config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    buckets: dict[str, list[str]] = collections.defaultdict(list)
    with open(args.train) as f:
        for line in f:
            d = json.loads(line)
            if d.get("input") and d.get("type") in FINE_TO_COARSE:
                buckets[FINE_TO_COARSE[d["type"]]].append(d["input"])
    queries, coarse_idx, is_confused, fine_of = [], [], [], []
    # Keep the fine labels for the binary head. Rebuild the pool with types.
    typed: dict[str, list[tuple[str, str]]] = collections.defaultdict(list)
    with open(args.train) as f:
        for line in f:
            d = json.loads(line)
            if d.get("input") and d.get("type") in FINE_TO_COARSE:
                typed[FINE_TO_COARSE[d["type"]]].append((d["input"], d["type"]))
    for letter in OPTION_LETTERS:
        pool = typed[letter]
        random.shuffle(pool)
        for q, fine in pool[: args.per_class]:
            queries.append(q)
            coarse_idx.append(OPTION_LETTERS.index(letter))
            is_confused.append(1 if fine == "confused" else 0)
            fine_of.append(fine)
    print(f"sampled {len(queries)} queries ({args.per_class}/class cap)")

    feats = extract_hidden(model, tokenizer, queries, config.max_seq_len, device)
    print(f"features: {tuple(feats.shape)}")

    print("== 7-way head ==")
    head7, t7 = train_head(feats, torch.tensor(coarse_idx), 7, args.epochs, args.seed)
    print("== confused-vs-rest head ==")
    print(f"positives: {sum(is_confused)}/{len(is_confused)}")
    head_bin, t_bin = train_head(feats, torch.tensor(is_confused), 2, args.epochs, args.seed + 1)

    fine_queries, fine_targets = [], []
    by_fine: dict[str, list[str]] = collections.defaultdict(list)
    with open(args.train) as f:
        for line in f:
            d = json.loads(line)
            if d.get("input") and d.get("type") in FINE_TO_COARSE:
                by_fine[d["type"]].append(d["input"])
    for fine in FINE21_ORDER:
        pool = by_fine[fine]
        random.shuffle(pool)
        for q in pool[: args.per_fine]:
            fine_queries.append(q)
            fine_targets.append(FINE21_ORDER.index(fine))
    print(f"sampled {len(fine_queries)} queries for 21-way ({args.per_fine}/fine-type cap)")
    fine_feats = extract_hidden(model, tokenizer, fine_queries, config.max_seq_len, device)
    print("== 21-way head ==")
    head21, t21 = train_head(fine_feats, torch.tensor(fine_targets), len(FINE21_ORDER), args.epochs, args.seed + 2)

    torch.save(
        {
            "head7": head7.state_dict(),
            "temperature7": t7,
            "head_bin": head_bin.state_dict(),
            "temperature_bin": t_bin,
            "head21": head21.state_dict(),
            "temperature21": t21,
            "d_model": feats.size(1),
            "options": OPTION_LETTERS,
            "fine_options": FINE21_ORDER,
            "base_checkpoint": args.checkpoint,
        },
        args.out,
    )
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
