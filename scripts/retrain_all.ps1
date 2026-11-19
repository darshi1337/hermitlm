# Full retrain on this machine's Intel Arc GPU (PyTorch XPU).
# Run from the repository root:
#   powershell -ExecutionPolicy Bypass -File scripts/retrain_all.ps1
$ErrorActionPreference = "Stop"
$env:HERMIT_TRAIN_DEVICE = if ($env:HERMIT_TRAIN_DEVICE) { $env:HERMIT_TRAIN_DEVICE } else { "xpu" }
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUNBUFFERED = "1"

Write-Host "== 1/9 Generate SFT data + tokenizer =="
python -m hermitlm.training.prepare_data
if ($LASTEXITCODE -ne 0) { throw "prepare_data failed" }
python scripts/build_human_eval.py
if ($LASTEXITCODE -ne 0) { throw "build_human_eval failed" }

Write-Host "== 2/9 Build tool-routing data =="
python scripts/build_tool_data.py
if ($LASTEXITCODE -ne 0) { throw "build_tool_data failed" }

Write-Host "== 3/9 SFT (overwrites checkpoints/best_model.pt) =="
python -m hermitlm.training.train
if ($LASTEXITCODE -ne 0) { throw "SFT failed" }

Write-Host "== 4/9 Intent probes =="
python scripts/train_probe.py --checkpoint checkpoints/best_model.pt --tokenizer data/tokenizer.json --train data/train.jsonl --per-class 1500 --per-fine 400 --epochs 30 --out checkpoints/decision_probe.pt
if ($LASTEXITCODE -ne 0) { throw "train_probe failed" }

Write-Host "== 5/9 Tool probe =="
python scripts/train_tool_probe.py --checkpoint checkpoints/best_model.pt --tokenizer data/tokenizer.json --train data/tool_train.jsonl --eval data/tool_eval.jsonl --epochs 30 --out checkpoints/decision_tool_probe.pt
if ($LASTEXITCODE -ne 0) { throw "train_tool_probe failed" }

Write-Host "== 6/9 Evals =="
python scripts/eval.py --checkpoint checkpoints/best_model.pt --out-dir eval_out/sft
if ($LASTEXITCODE -ne 0) { throw "eval failed" }
python scripts/eval_decision.py --method probe --max-n 5000 --cal-n 1000 --bootstrap 1000 --out-dir eval_out/decision_probe
if ($LASTEXITCODE -ne 0) { throw "eval_decision failed" }
python scripts/eval_decision.py --eval data/human_eval.jsonl --method probe --max-n 126 --cal-n 60 --bootstrap 1000 --out-dir eval_out/human
if ($LASTEXITCODE -ne 0) { throw "human eval failed" }

Write-Host "== 7/9 Browser artifacts =="
python scripts/export_onnx.py --output-dir web
if ($LASTEXITCODE -ne 0) { throw "export_onnx failed" }
python scripts/quantize_onnx.py --input web/model.onnx --output web/model.quant.onnx
if ($LASTEXITCODE -ne 0) { throw "quantize failed" }
python scripts/export_probe.py --output web/probe.json
if ($LASTEXITCODE -ne 0) { throw "export_probe failed" }
python scripts/write_asset_manifest.py --output-dir web
if ($LASTEXITCODE -ne 0) { throw "write_asset_manifest failed" }

Write-Host "== 8/9 Vendored runtime =="
python scripts/fetch_vendor.py --output-dir web/vendor
if ($LASTEXITCODE -ne 0) { throw "fetch_vendor failed" }

Write-Host "== 9/9 Tests and lint =="
python -m pytest -q
if ($LASTEXITCODE -ne 0) { throw "pytest failed" }
python -m ruff check .
if ($LASTEXITCODE -ne 0) { throw "ruff failed" }

Write-Host "Done. Commit: checkpoints/*.pt, checkpoints/config.json, web/probe.json"
