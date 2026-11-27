"""Train probe decision heads on frozen HermitLM hidden states.

Layer, solver and regularization are selected by stratified k-fold OOF NLL,
following brier's L2 method. Priors are fitted from the unlabelled query pool
for label-free batch debiasing.
"""
import argparse
import collections
import json
import os
import random

import torch
from tokenizers import Tokenizer

from hermitlm.config import HermitConfig
from hermitlm.tools.artifacts import head_metadata, refit_metadata
from hermitlm.tools.decision_model import (
    FINE21_ORDER,
    FINE_TO_COARSE,
    LEGACY_CLARIFY_THRESHOLD,
    LEGACY_ESCALATION_THRESHOLD,
    OPTION_LETTERS,
    PROMPT_TEMPLATES,
)
from hermitlm.tools.heads import default_layers, select_head
from hermitlm.tools.prior import fit_prior
from hermitlm.training.model import HermitLM

FOLDS = 5  # stratified k-fold OOF selection; stored in the artifact's refit metadata


def chat_prompt(query: str) -> str:
    return f"<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n"


@torch.no_grad()
def extract_hidden_layers(
    model: HermitLM, tokenizer: Tokenizer, queries: list[str], max_seq_len: int,
    device: torch.device, batch_size: int = 64,
) -> torch.Tensor:
    """Return (n, num_layers, d) hidden states at the true last token."""
    feats = []
    for i in range(0, len(queries), batch_size):
        batch = queries[i : i + batch_size]
        ids = [tokenizer.encode(chat_prompt(q)).ids[-max_seq_len:] for q in batch]
        width = max(len(s) for s in ids)
        padded = [s + [0] * (width - len(s)) for s in ids]
        idx = torch.tensor(padded, dtype=torch.long, device=device)
        _, hidden = model.forward_with_hidden_layers(idx)
        lengths = torch.tensor([len(s) for s in ids], dtype=torch.long)
        # Advanced indexing on Intel XPU hits Level Zero limits: gather on CPU.
        feats.append(hidden.cpu()[torch.arange(len(batch)), lengths - 1, :])
    return torch.cat(feats)


def _prior_for(head, features, layer):
    lp = head.log_probs(features[:, layer, :].double())
    return fit_prior(lp).tolist()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train probe decision heads")
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pt")
    parser.add_argument("--tokenizer", default="data/tokenizer.json")
    parser.add_argument("--train", default="data/train.jsonl")
    parser.add_argument("--per-class", type=int, default=1500)
    parser.add_argument("--per-fine", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--prior-lambda", type=float, default=1.0)
    parser.add_argument("--out", default="checkpoints/decision_probe.pt")
    args = parser.parse_args()

    random.seed(args.seed)
    from hermitlm.training.train import accelerator_available

    device = torch.device(args.device)
    if device.type in ("cuda", "xpu") and not accelerator_available(device.type):
        raise RuntimeError(
            f"Probe training was requested on {device.type} but that backend is "
            "not available. Pass --device cpu for a CPU-only run."
        )
    tokenizer = Tokenizer.from_file(args.tokenizer)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    config = HermitConfig(**ckpt["config"]) if "config" in ckpt else HermitConfig()
    config.vocab_size = tokenizer.get_vocab_size()
    model = HermitLM(config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    typed = collections.defaultdict(list)
    with open(args.train, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            if d.get("input") and d.get("type") in FINE_TO_COARSE:
                typed[FINE_TO_COARSE[d["type"]]].append((d["input"], d["type"]))

    queries, coarse_idx, is_confused = [], [], []
    for letter in OPTION_LETTERS:
        pool = typed[letter]
        random.shuffle(pool)
        for q, fine in pool[: args.per_class]:
            queries.append(q)
            coarse_idx.append(OPTION_LETTERS.index(letter))
            is_confused.append(1 if fine == "confused" else 0)
    print(f"sampled {len(queries)} queries ({args.per_class}/class cap)")

    feats = extract_hidden_layers(model, tokenizer, queries, config.max_seq_len, device)
    print(f"features: {tuple(feats.shape)}")

    layers = default_layers(config.n_layers)
    print(f"candidate layers: {layers}")
    coarse_head = select_head(feats, torch.tensor(coarse_idx), 7, layers, k=FOLDS, seed=args.seed)
    layer = coarse_head.layer
    print(f"selected layer {layer} (oof_nll={coarse_head.oof_nll:.4f})")

    binary_head = select_head(feats, torch.tensor(is_confused), 2, [layer], k=FOLDS, seed=args.seed + 1)

    fine_queries, fine_targets = [], []
    by_fine = collections.defaultdict(list)
    with open(args.train, encoding="utf-8") as f:
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
    fine_feats = extract_hidden_layers(model, tokenizer, fine_queries, config.max_seq_len, device)
    fine_head = select_head(fine_feats, torch.tensor(fine_targets), len(FINE21_ORDER), [layer], k=FOLDS, seed=args.seed + 2)

    heads = {"coarse": coarse_head, "confused": binary_head, "fine": fine_head}
    priors = {
        "coarse": _prior_for(coarse_head, feats, layer),
        "confused": _prior_for(binary_head, feats, layer),
        "fine": _prior_for(fine_head, fine_feats, layer),
    }
    for name, head in heads.items():
        print(f"{name}: layer={head.layer} solver={head.solver} alpha={head.alpha} "
              f"T={head.temperature:.3f} oof_acc={head.oof_accuracy:.4f}")

    # Keep any thresholds already fitted for this artifact path; otherwise
    # record the legacy defaults until fit_decision_thresholds.py is rerun.
    existing = {}
    if os.path.exists(args.out):
        existing = torch.load(args.out, map_location="cpu", weights_only=True)
    thresholds = {
        "clarify": float(existing.get("clarify_threshold", LEGACY_CLARIFY_THRESHOLD)),
        "escalation": float(existing.get("escalation_threshold", LEGACY_ESCALATION_THRESHOLD)),
        "target_error": float(existing.get("target_error", 0.1)),
    }
    head_states = {k: v.as_dict() for k, v in heads.items()}
    refit = refit_metadata(head_states, folds=FOLDS, seed=args.seed)
    artifact = {
        "selection": {"layer": layer, "heads": head_states},
        "chosen_layer": layer,
        "thresholds": thresholds,
        "refit": refit,
        "priors": priors,
        "prior_lambda": args.prior_lambda,
        "d_model": feats.size(2),
        "options": OPTION_LETTERS,
        "fine_options": FINE21_ORDER,
        "base_checkpoint": args.checkpoint,
        **head_metadata(
            args.checkpoint, args.tokenizer,
            chosen_layer=layer,
            thresholds=thresholds,
            refit=refit,
            prompt_template=PROMPT_TEMPLATES,
            dtype=str(next(model.parameters()).dtype),
        ),
    }
    torch.save(artifact, args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
