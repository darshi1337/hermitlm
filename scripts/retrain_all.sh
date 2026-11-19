#!/bin/bash
# Full retrain on CUDA. Run from the repository root:
#   bash scripts/retrain_all.sh
# Any missing CUDA device aborts immediately with a clear error.
set -euo pipefail

# Intel Arc (this machine) uses PyTorch's XPU backend. On an NVIDIA machine,
# override this with HERMIT_TRAIN_DEVICE=cuda.
export HERMIT_TRAIN_DEVICE="${HERMIT_TRAIN_DEVICE:-xpu}"
export PYTHONIOENCODING=utf-8
export PYTHONUNBUFFERED=1

echo "== 1/9 Generate SFT data + tokenizer =="
python -m hermitlm.training.prepare_data
python scripts/build_human_eval.py

echo "== 2/9 Build tool-routing data =="
python scripts/build_tool_data.py

echo "== 3/9 SFT (overwrites checkpoints/best_model.pt) =="
python -m hermitlm.training.train

echo "== 4/9 Intent probes (7-way / 21-way / confused) =="
python scripts/train_probe.py \
  --checkpoint checkpoints/best_model.pt \
  --tokenizer data/tokenizer.json \
  --train data/train.jsonl \
  --per-class 1500 --per-fine 400 --epochs 30 \
  --out checkpoints/decision_probe.pt

echo "== 5/9 Tool probe (math / web / faq / chat) =="
python scripts/train_tool_probe.py \
  --checkpoint checkpoints/best_model.pt \
  --tokenizer data/tokenizer.json \
  --train data/tool_train.jsonl \
  --eval data/tool_eval.jsonl \
  --epochs 30 \
  --out checkpoints/decision_tool_probe.pt

echo "== 6/9 Evals =="
python scripts/eval.py --checkpoint checkpoints/best_model.pt --out-dir eval_out/sft
python scripts/eval_decision.py \
  --method probe --max-n 5000 --cal-n 1000 --bootstrap 1000 \
  --out-dir eval_out/decision_probe
python scripts/eval_decision.py \
  --eval data/human_eval.jsonl --method probe --max-n 126 --cal-n 60 \
  --bootstrap 1000 --out-dir eval_out/human

echo "== 7/9 Browser artifacts =="
python scripts/export_onnx.py --output-dir web
python scripts/quantize_onnx.py --input web/model.onnx --output web/model.quant.onnx
python scripts/export_probe.py --output web/probe.json
python scripts/write_asset_manifest.py --output-dir web

echo "== 8/9 Vendored runtime (local dev; CI does this too) =="
python scripts/fetch_vendor.py --output-dir web/vendor

echo "== 9/9 Tests =="
python -m pytest -q
python -m ruff check .

# Optional RLHF stages. Only use one AFTER deciding the final checkpoint:
# the probes are hash-bound to checkpoints/best_model.pt and must be retrained
# against any other base before deployment.
# python -m hermitlm.training.ppo    # -> checkpoints/ppo/ppo_final.pt
# python -m hermitlm.training.dpo    # -> checkpoints/dpo/dpo_final.pt

echo "Done. Commit: checkpoints/*.pt, checkpoints/config.json, web/probe.json"
