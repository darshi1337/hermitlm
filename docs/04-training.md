# Training

Training defaults to the **Intel Arc GPU** via PyTorch's XPU backend (`HERMIT_TRAIN_DEVICE=xpu`). On an NVIDIA machine set `HERMIT_TRAIN_DEVICE=cuda`; use `=cpu` only for debugging. Probe fitting defaults to CPU (`--device cpu`) because it is frozen-model inference and avoids flaky XPU behaviour; pass `--device xpu` or `--device cuda` to move it. Training and probe entry points fail loudly if the requested accelerator is unavailable. Install the matching wheel with `pip install torch --index-url https://download.pytorch.org/whl/xpu`. Data generation (`prepare_data.py`, `build_tool_data.py`) and ONNX export stay on CPU.

The full pipeline can be rerun with:

```bash
# Windows / Intel Arc (XPU, default here)
powershell -ExecutionPolicy Bypass -File scripts\retrain_resume.ps1
# Linux / NVIDIA CUDA
HERMIT_TRAIN_DEVICE=cuda bash scripts/retrain_all.sh
```

`scripts/retrain_resume.ps1` is resume-safe: completed steps are recorded in `eval_out/retrain_done.txt`
and skipped on the next run. `scripts/retrain_all.sh` is a straight full run with no resume state.
`scripts/install_retrain_task.ps1` registers a logon-triggered task so a reboot resumes the Windows
retrain automatically.

Three stages:

1. **SFT** (`scripts/train.sh` → `hermitlm/training/train.py`): next-token prediction on `data/train.jsonl`, produces `checkpoints/best_model.pt`.

2. **RLHF** (`scripts/train_ppo.sh` → `hermitlm/training/ppo.py`): PPO fine-tunes the SFT checkpoint against a heuristic persona-consistency reward (`hermitlm/training/reward.py`). The reward scores crab-vocabulary use and terse in-character style. It penalizes repetition, shouting, and character breaks. A KL penalty against the frozen SFT model (the reference policy) keeps the tuned policy from drifting off-distribution. Training draws rollout prompts only from `data/train.jsonl` so `eval.jsonl` stays held out. PPO checkpoints land in `checkpoints/ppo/` and stay drop-in compatible with the inference, chat UI, and API code.

3. **DPO** (`scripts/train_dpo.sh` → `hermitlm/training/dpo.py`): simpler preference-tuning alternative to PPO. It builds chosen/rejected pairs on-policy (4 candidates per prompt, best vs worst by `score_response`). A pair survives only if the score gap clears a margin filter. It optimizes against a frozen SFT reference. Checkpoints land in `checkpoints/dpo/`.
