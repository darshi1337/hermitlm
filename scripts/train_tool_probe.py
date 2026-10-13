"""Train the tool-routing probe: math / web / faq / chat.

Same method as the intent probe (frozen forward pass, linear head), trained
on synthetic data from scripts/build_tool_data.py and measured on the
hand-authored realistic set in data/tool_eval.jsonl.

Usage:
    conda run -n hermit python scripts/build_tool_data.py
    conda run -n hermit python scripts/train_tool_probe.py --out checkpoints/decision_tool_probe.pt
"""

import argparse
import collections
import json
import random

import torch
from tokenizers import Tokenizer
from train_probe import extract_hidden, train_head

from hermitlm.config import HermitConfig
from hermitlm.training.model import HermitLM

TOOL_CLASSES = ["math", "web", "faq", "chat"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train tool-routing probe head")
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pt")
    parser.add_argument("--tokenizer", default="data/tokenizer.json")
    parser.add_argument("--train", default="data/tool_train.jsonl")
    parser.add_argument("--eval", default="data/tool_eval.jsonl")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", default="checkpoints/decision_tool_probe.pt")
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
            if d.get("input") and d.get("label") in TOOL_CLASSES:
                buckets[d["label"]].append(d["input"])
    queries, targets = [], []
    for label in TOOL_CLASSES:
        pool = buckets[label]
        random.shuffle(pool)
        for q in pool:
            queries.append(q)
            targets.append(TOOL_CLASSES.index(label))
    print(f"training rows: {len(queries)}")
    feats = extract_hidden(model, tokenizer, queries, config.max_seq_len, device)
    head, temperature = train_head(feats, torch.tensor(targets), len(TOOL_CLASSES), args.epochs, args.seed)

    with open(args.eval) as f:
        eval_rows = [json.loads(line) for line in f if line.strip()]
    eval_feats = extract_hidden(
        model, tokenizer, [r["input"] for r in eval_rows], config.max_seq_len, device
    )
    head.eval()
    with torch.no_grad():
        probas = torch.softmax(head(eval_feats) / temperature, dim=-1)
    conf, pred = probas.max(-1)
    true = torch.tensor([TOOL_CLASSES.index(r["label"]) for r in eval_rows])
    acc = (pred == true).float().mean().item()
    print(f"eval accuracy={acc:.4f} (n={len(eval_rows)})")
    for i, label in enumerate(TOOL_CLASSES):
        mask = true == i
        cls_acc = (pred[mask] == i).float().mean().item() if mask.sum() else 0.0
        print(f"  {label}: acc={cls_acc:.4f} support={mask.sum().item()}")
    for thr in [0.5, 0.7, 0.9]:
        kept = conf >= thr
        cov = kept.float().mean().item()
        acc_kept = (pred[kept] == true[kept]).float().mean().item() if kept.sum() else 0.0
        print(f"  thr={thr}: coverage={cov:.3f} acc_when_kept={acc_kept:.4f}")

    torch.save(
        {
            "head": head.state_dict(),
            "temperature": temperature,
            "classes": TOOL_CLASSES,
            "d_model": feats.size(1),
            "eval_accuracy": acc,
            "base_checkpoint": args.checkpoint,
        },
        args.out,
    )
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
