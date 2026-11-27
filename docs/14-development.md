# Development

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1          # Windows; source .venv/bin/activate on Linux
pip install -e . -r requirements.txt pytest ruff
pip install onnx onnxscript onnxruntime
# Intel Arc / XPU (default on this machine):
pip install torch --index-url https://download.pytorch.org/whl/xpu
# NVIDIA CUDA instead: pip install torch --index-url https://download.pytorch.org/whl/cu121

pytest tests/          # 111 passed, 2 skipped
ruff check .           # clean
python -m hermitlm.tools.check
```

Create a `.env` with the documented keys (`HERMIT_CHECKPOINT`, `HERMIT_TOKENIZER`, `HERMIT_DEVICE`,
`HERMIT_TRAIN_DEVICE`, `HERMIT_TEMPERATURE`, `HERMIT_TOP_K`, `HERMIT_MAX_TOKENS`,
`HERMIT_PROBE`, `HERMIT_CONFUSED_THRESHOLD`, `HERMIT_YES_THRESHOLD`, `HERMIT_TOOL_PROBE`,
`HERMIT_TOOL_THRESHOLD`,
`HERMIT_BEST_OF`, `HERMIT_CLARIFY`, `HERMIT_CLARIFY_THRESHOLD`, `HERMIT_PERSONA_PROMPT`,
`WOLFRAM_APP_ID`, …). Never commit `.env`.
