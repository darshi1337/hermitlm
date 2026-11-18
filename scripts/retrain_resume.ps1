# Resume-safe full retrain on this machine's Intel Arc GPU (PyTorch XPU).
# Completed steps are marked in eval_out/retrain_done.txt and skipped on the
# next run, so a restart after a reboot does not repeat finished work.
$ErrorActionPreference = "Stop"
$env:HERMIT_TRAIN_DEVICE = if ($env:HERMIT_TRAIN_DEVICE) { $env:HERMIT_TRAIN_DEVICE } else { "xpu" }
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUNBUFFERED = "1"

Set-Location (Split-Path -Parent $PSScriptRoot)
$stateDir = Join-Path (Get-Location) "eval_out"
New-Item -ItemType Directory -Force -Path $stateDir | Out-Null
$stateFile = Join-Path $stateDir "retrain_done.txt"
$completeFile = Join-Path $stateDir "retrain_complete.txt"

if (Test-Path $completeFile) {
    Write-Host "Retrain already marked complete. Delete $completeFile to run again."
    exit 0
}

function Test-Step([string]$name) {
    return (Test-Path $stateFile) -and ((Get-Content $stateFile) -contains $name)
}
function Mark-Step([string]$name) {
    Add-Content -Path $stateFile -Value $name -Encoding utf8
}
function Invoke-Step([string]$name, [scriptblock]$body) {
    if (Test-Step $name) {
        Write-Host "== SKIP $name (already done) =="
        return
    }
    Write-Host "== RUN $name =="
    & $body
    if ($LASTEXITCODE -ne 0) { throw "$name failed (exit $LASTEXITCODE)" }
    Mark-Step $name
}

Invoke-Step "data" {
    python -m hermitlm.training.prepare_data
}
Invoke-Step "tool_data" {
    python scripts/build_tool_data.py
}
Invoke-Step "sft" {
    python -m hermitlm.training.train
}
Invoke-Step "probes" {
    python scripts/train_probe.py --checkpoint checkpoints/best_model.pt --tokenizer data/tokenizer.json --train data/train.jsonl --per-class 1500 --per-fine 400 --epochs 30 --out checkpoints/decision_probe.pt
}
Invoke-Step "tool_probe" {
    python scripts/train_tool_probe.py --checkpoint checkpoints/best_model.pt --tokenizer data/tokenizer.json --train data/tool_train.jsonl --eval data/tool_eval.jsonl --epochs 30 --out checkpoints/decision_tool_probe.pt
}
Invoke-Step "evals" {
    python scripts/eval.py --checkpoint checkpoints/best_model.pt --out-dir eval_out/sft
    if ($LASTEXITCODE -ne 0) { return }
    python scripts/eval_decision.py --method probe --max-n 5000 --cal-n 1000 --bootstrap 1000 --out-dir eval_out/decision_probe
}
Invoke-Step "onnx" {
    python scripts/export_onnx.py --output-dir web
    if ($LASTEXITCODE -ne 0) { return }
    python scripts/quantize_onnx.py --input web/model.onnx --output web/model.quant.onnx
    if ($LASTEXITCODE -ne 0) { return }
    python scripts/export_probe.py --output web/probe.json
    if ($LASTEXITCODE -ne 0) { return }
    python scripts/write_asset_manifest.py --output-dir web
}
Invoke-Step "vendor" {
    python scripts/fetch_vendor.py --output-dir web/vendor
}
Invoke-Step "tests" {
    python -m pytest -q
    if ($LASTEXITCODE -ne 0) { return }
    python -m ruff check .
}

New-Item -ItemType File -Path $completeFile -Force | Out-Null
Write-Host "Retrain complete. Marker: $completeFile"
