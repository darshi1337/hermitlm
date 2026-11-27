# Hugging Face

Model repo: `darshi1337/hermitlm-8M`. Dataset repo: `darshi1337/hermitlm-100k-generic`.

The model repo holds 9 files: `best_model.pt`, `config.json`, `tokenizer.json`, `model.onnx`,
`model.quant.onnx`, `decision_probe.pt`, `decision_tool_probe.pt`, plus `README.md` (model
card) and `.gitattributes`. The dataset repo includes `train.jsonl`, `eval.jsonl`, their
OpenAI-format variants, `tool_train.jsonl`, `tool_eval.jsonl`, `human_eval.jsonl`, and the
matching tokenizer.

Download a published repo:

```python
from huggingface_hub import snapshot_download
repo = snapshot_download("darshi1337/hermitlm-8M")
```

Upload a new build (maintainers only; both scripts read `HF_TOKEN` from `.env` or `hf auth login`):

```bash
python scripts/push_hf.py --repo darshi1337/hermitlm-8M --dry-run
python scripts/push_hf.py --repo darshi1337/hermitlm-8M
python scripts/push_hf_dataset.py --repo darshi1337/hermitlm-100k-generic --dry-run
python scripts/push_hf_dataset.py --repo darshi1337/hermitlm-100k-generic
```

Status on 2026-11-19: model and dataset repos are synced with the current 104k balanced corpus
and hash-pinned probes.
