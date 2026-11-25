"""Fit a label-free batch prior for an existing probe artifact.

Reads only the *inputs* from a query pool (labels are never used), extracts
the hidden states the probe heads score, and stores ``priors`` plus
``prior_lambda`` in the artifact. The API and the browser apply the same
correction (``log p - lambda * log pi``).

Usage:
    python scripts/fit_probe_prior.py --pool data/train.jsonl --max-n 4000
    python scripts/export_probe.py          # refresh web/probe.json
"""
import argparse
import json
import random

import torch
from tokenizers import Tokenizer

from hermitlm.config import HermitConfig
from hermitlm.tools.decision_model import (
    FINE21_ORDER,
    OPTION_LETTERS,
    LegacyHead,
    chat_prompt,
)
from hermitlm.tools.heads import FittedHead
from hermitlm.tools.prior import fit_prior
from hermitlm.training.model import HermitLM


def load_pool(path: str, max_n: int, seed: int) -> list[str]:
    """Return up to ``max_n`` unique inputs; labels are discarded."""
    seen: set[str] = set()
    queries: list[str] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            text = row.get("input")
            if text and text not in seen:
                seen.add(text)
                queries.append(text)
    random.Random(seed).shuffle(queries)
    return queries[:max_n]


@torch.no_grad()
def extract_features(
    model: HermitLM,
    tokenizer: Tokenizer,
    queries: list[str],
    layer: int | None,
    max_seq_len: int,
    device: torch.device,
    batch_size: int = 64,
) -> torch.Tensor:
    """Hidden state at the final real token, shape (n, d)."""
    feats = []
    for start in range(0, len(queries), batch_size):
        batch = queries[start : start + batch_size]
        ids = [tokenizer.encode(chat_prompt(q)).ids[-max_seq_len:] for q in batch]
        width = max(len(row) for row in ids)
        padded = [row + [0] * (width - len(row)) for row in ids]
        idx = torch.tensor(padded, dtype=torch.long, device=device)
        if layer is None:
            _, _, hidden = model(idx, return_hidden=True)
        else:
            _, hidden = model.forward_with_hidden_layers(idx)
            hidden = hidden[:, :, layer, :]
        lengths = torch.tensor([len(row) for row in ids], dtype=torch.long)
        # Advanced indexing on Intel XPU hits Level Zero limits: gather on CPU.
        feats.append(hidden.cpu()[torch.arange(len(batch)), lengths - 1, :])
    return torch.cat(feats)


def build_heads(probe: dict, device: torch.device) -> tuple[dict, int | None]:
    selection = probe.get("selection")
    if selection:
        layer = int(selection["layer"])
        heads = {
            name: FittedHead.from_dict(state).to(device)
            for name, state in selection["heads"].items()
        }
        return heads, layer
    heads = {
        "coarse": LegacyHead(probe["head7"], probe["temperature7"], device),
        "confused": LegacyHead(probe["head_bin"], probe["temperature_bin"], device),
    }
    if "head21" in probe:
        heads["fine"] = LegacyHead(
            probe["head21"], probe.get("temperature21", 1.0), device
        )
    return heads, None


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit label-free priors into a probe artifact")
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pt")
    parser.add_argument("--tokenizer", default="data/tokenizer.json")
    parser.add_argument("--probe", default="checkpoints/decision_probe.pt")
    parser.add_argument("--pool", default="data/train.jsonl")
    parser.add_argument("--max-n", type=int, default=4000)
    parser.add_argument("--prior-lambda", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not 0.0 <= args.prior_lambda <= 1.0:
        raise SystemExit("--prior-lambda must be in [0, 1]")

    from hermitlm.training.train import accelerator_available

    device = torch.device(args.device)
    if device.type in ("cuda", "xpu") and not accelerator_available(device.type):
        raise SystemExit(
            f"requested {device.type} but that backend is not available; pass --device cpu"
        )

    tokenizer = Tokenizer.from_file(args.tokenizer)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    config = HermitConfig(**ckpt["config"]) if "config" in ckpt else HermitConfig()
    config.vocab_size = tokenizer.get_vocab_size()
    model = HermitLM(config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    probe = torch.load(args.probe, map_location=device, weights_only=True)
    heads, layer = build_heads(probe, device)
    queries = load_pool(args.pool, args.max_n, args.seed)
    if not queries:
        raise SystemExit(f"no unlabelled inputs found in {args.pool}")
    print(f"pool: {len(queries)} unlabelled queries from {args.pool}")
    print(f"probe: {'selection layer ' + str(layer) if layer is not None else 'legacy last layer'}")

    feats = extract_features(
        model, tokenizer, queries, layer, config.max_seq_len, device, args.batch_size
    )
    print(f"features: {tuple(feats.shape)}")

    priors = {}
    for name, head in heads.items():
        log_probs = head.log_probs(feats)
        prior = fit_prior(log_probs)
        priors[name] = prior.tolist()
        top = sorted(
            zip(OPTION_LETTERS if name != "fine" else FINE21_ORDER, prior.tolist()),
            key=lambda kv: kv[1],
            reverse=True,
        )[:3]
        rendered = ", ".join(f"{label}={value:.3f}" for label, value in top)
        print(f"{name}: prior top3 {rendered}")

    probe["priors"] = priors
    probe["prior_lambda"] = args.prior_lambda
    torch.save(probe, args.probe)
    print(f"saved priors to {args.probe}")
    print("run `python scripts/export_probe.py` to refresh web/probe.json")


if __name__ == "__main__":
    main()
