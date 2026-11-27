"""Diagnostics for a HermitLM checkpoint + probe setup.

Run with: python -m hermitlm.tools.check
"""
import hashlib
import logging
import os
import warnings

import torch
from tokenizers import Tokenizer

from hermitlm.config import HermitConfig
from hermitlm.settings import (
    CHECKPOINT_PATH,
    DEVICE,
    PROBE_PATH,
    TOKENIZER_PATH,
    TOOL_PROBE_PATH,
)
from hermitlm.tools.decision_model import (
    PROMPT_TEMPLATES,
    chat_prompt,
    get_fine_option_ids,
    get_label_ids,
    get_option_ids,
)
from hermitlm.training.model import HermitLM

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def main() -> None:
    print("HermitLM diagnostics")
    print("=" * 40)
    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
    for name, ids in (
        ("choice", get_option_ids(tokenizer)),
        ("fine", get_fine_option_ids(tokenizer)),
    ):
        assert len(ids) == len(set(ids)), f"{name} option ids are not distinct"
        print(f"PASS  labels.{name}: {len(ids)} single distinct tokens")

    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=True)
    config = HermitConfig(**ckpt["config"]) if "config" in ckpt else HermitConfig()
    config.vocab_size = tokenizer.get_vocab_size()
    model = HermitLM(config).to(torch.device(DEVICE)).eval()
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"PASS  checkpoint: {model.param_count():,} params, vocab={config.vocab_size}")

    ids = tokenizer.encode(chat_prompt("hello crab")).ids
    idx = torch.tensor([ids, ids], dtype=torch.long, device=DEVICE)
    with torch.no_grad():
        logits, hidden = model.forward_with_hidden_layers(idx)
        batch_gap = (logits[0] - logits[1]).abs().max().item()
    assert batch_gap < 2e-2, f"batch inconsistency: {batch_gap}"
    print(f"PASS  batch_consistency: max gap={batch_gap:.2e}")
    print(f"PASS  hidden_states: {hidden.shape[2]} candidate layers, d={hidden.shape[-1]}")

    if os.path.exists(PROBE_PATH):
        probe = torch.load(PROBE_PATH, map_location="cpu", weights_only=True)
        expected = probe.get("prompt_template_sha256")
        if expected:
            actual = hashlib.sha256(PROMPT_TEMPLATES.encode("utf-8")).hexdigest()
            assert actual == expected, "prompt template changed since probe was fitted"
            print("PASS  prompt_template: matches probe artifact")
            logger.info("prompt template pin verified: %s", expected[:12])
        else:
            warnings.warn(
                "probe artifact has no prompt-template hash pin",
                UserWarning, stacklevel=2,
            )
        logger.info(
            "probe schema=%s chosen_layer=%s score_levels=%d",
            probe.get("schema_version", 1), probe.get("chosen_layer"),
            len(probe.get("score_levels", [])),
        )
        selection = probe.get("selection")
        if selection:
            print(f"PASS  probe: selected layer {selection['layer']}, "
                  f"solvers={[h['solver'] for h in selection['heads'].values()]}")
        heads = (selection or {}).get("heads", {})
        for name, key in (("yes", "head_yes"), ("score", "head_score")):
            if key in probe or name in heads:
                print(f"PASS  auxiliary_head.{name}: present")
        labels = probe.get("score_labels")
        if labels:
            used, ids = get_label_ids(tokenizer, labels)
            print(f"PASS  score_labels: {used} -> {ids}")
    if os.path.exists(TOOL_PROBE_PATH):
        tool = torch.load(TOOL_PROBE_PATH, map_location="cpu", weights_only=True)
        if tool.get("selection"):
            print(f"PASS  tool_probe: selected layer {tool['selection']['layer']}")

    print("All checks passed.")


if __name__ == "__main__":
    main()
