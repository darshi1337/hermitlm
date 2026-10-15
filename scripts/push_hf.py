"""Push checkpoint + ONNX + tokenizer to Hugging Face Hub."""
import argparse
import os
import sys

MODEL_CARD = """---
license: mit
language: en
tags:
- tiny-llm
- pytorch
- onnx
- roleplay
base_model: none
---

# hermitlm-8M

A ~8M parameter causal transformer that talks like a tiny crab in an aquatic world.

Architecture: 6 layers, hidden dim 320, 8 heads, SwiGLU FFN 800, learned positions,
LayerNorm, weight-tied LM head. Vocab 2831 BPE, max sequence 512. See `config.json`.

Training: SFT on `data/train.jsonl`, then PPO against a persona reward and a DPO stage.
Eval on 10 held-out prompts: SFT persona 0.516, PPO 0.648, DPO 0.449. Perplexity ~1.93.

## Files

- `best_model.pt`: PyTorch checkpoint (model_state_dict + config)
- `config.json`: model and training config
- `tokenizer.json`: byte-level BPE tokenizer
- `model.onnx`: full-precision ONNX export for the browser UI
- `model.quant.onnx`: uint8 quantized ONNX (~9MB)
- `decision_probe.pt`: 7-way intent probe on frozen features
- `decision_tool_probe.pt`: 4-way tool-routing probe (math/web/faq/chat)

## Use

```python
from huggingface_hub import snapshot_download
from tokenizers import Tokenizer
import torch
from hermitlm.training.model import HermitLM
from hermitlm.config import HermitConfig

repo = snapshot_download("darshi1337/hermitlm-8M")
ckpt = torch.load(f"{repo}/best_model.pt", map_location="cpu", weights_only=True)
config = HermitConfig(**ckpt["config"])
model = HermitLM(config)
model.load_state_dict(ckpt["model_state_dict"])
tokenizer = Tokenizer.from_file(f"{repo}/tokenizer.json")
```

Browser chat uses `model.quant.onnx` with fallback to `model.onnx`.
"""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", required=True)
    p.add_argument("--checkpoint", default="checkpoints/best_model.pt")
    p.add_argument("--config", default="checkpoints/config.json")
    p.add_argument("--onnx", default="web/model.onnx")
    p.add_argument("--quant-onnx", default="web/model.quant.onnx")
    p.add_argument("--tokenizer", default="data/tokenizer.json")
    p.add_argument("--probe", default="checkpoints/decision_probe.pt")
    p.add_argument("--tool-probe", default="checkpoints/decision_tool_probe.pt")
    p.add_argument("--private", action="store_true")
    p.add_argument("--commit-message", default="Upload HermitLM 8M")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()

    from huggingface_hub import HfApi

    files = {
        a.checkpoint: "best_model.pt",
        a.config: "config.json",
        a.tokenizer: "tokenizer.json",
        a.onnx: "model.onnx",
        a.quant_onnx: "model.quant.onnx",
        a.probe: "decision_probe.pt",
        a.tool_probe: "decision_tool_probe.pt",
    }
    missing = [f for f in files if not os.path.isfile(f)]
    if missing:
        print("Missing files:", ", ".join(missing))
        sys.exit(1)

    if a.dry_run:
        print("Dry run OK. Files ready:")
        for src, dest in files.items():
            print(f"  {src} -> {dest} ({os.path.getsize(src)} bytes)")
        print("  <generated> -> README.md (model card)")
        return

    token = os.environ.get("HF_TOKEN")
    api = HfApi(token=token)
    api.create_repo(a.repo, exist_ok=True, private=a.private, repo_type="model")
    for src, dest in files.items():
        api.upload_file(
            path_or_fileobj=src,
            path_in_repo=dest,
            repo_id=a.repo,
            repo_type="model",
            commit_message=a.commit_message,
        )
    api.upload_file(
        path_or_fileobj=MODEL_CARD.encode(),
        path_in_repo="README.md",
        repo_id=a.repo,
        repo_type="model",
        commit_message=a.commit_message,
    )
    print(f"Pushed to {a.repo}")


if __name__ == "__main__":
    main()
