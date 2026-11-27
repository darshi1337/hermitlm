# Eval

`scripts/eval.py` scores a checkpoint on three axes and writes `report.json`/`report.md`:

- **Perplexity** on `data/eval.jsonl` (proxy, first 50 batches).
- **Persona reward**: `score_response` over generations from the model for 10 held-out prompts (not reference texts, so checkpoints are comparable).
- **Tool triggers**: offline keyword/tag routing check (no network).

```bash
python scripts/eval.py --checkpoint checkpoints/best_model.pt --out-dir eval_out/sft
python scripts/eval.py --checkpoint checkpoints/ppo/ppo_final.pt --out-dir eval_out/ppo
python scripts/eval.py --checkpoint checkpoints/dpo/dpo_final.pt --out-dir eval_out/dpo
```

Latest comparison (persona measured over 10 sampled generations, so ±0.05 noise is expected):

| Checkpoint | Perplexity | Persona reward |
| --- | --- | --- |
| SFT (`best_model.pt`) | 1.931 | 0.495 |
| PPO (`ppo_final.pt`) | 1.992 | **0.648** |
| DPO (`dpo_final.pt`, margin filter) | 1.930 | 0.449 |

Those PPO/DPO numbers are from an earlier training run; the current shipped artifact is the
newly retrained SFT checkpoint above. PPO and DPO remain available as optional stages but have
not been rerun on the 104k natural-paraphrase dataset.
