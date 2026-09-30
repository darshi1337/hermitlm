<h1 align="center">HermitLM</h1>

<p align="center">
  <a href="https://huggingface.co/datasets/darshi1337/hermitlm-100k-generic">
    <img alt="Dataset" src="https://img.shields.io/badge/Dataset-hermitlm--100k-4b5563">
  </a>
  <a href="https://huggingface.co/darshi1337/hermitlm-8M">
    <img alt="Model" src="https://img.shields.io/badge/Model-hermitlm--8M-b65f36">
  </a>
  <img alt="License" src="https://img.shields.io/badge/License-MIT-3fa34d">
</p>

<p align="center">
  <img alt="HermitLM" src="assets/hermit_1.png">
</p>

<p align="center">
  <a href="assets/video.mp4">
    <img alt="Try Demo" src="https://img.shields.io/badge/TRY%20DEMO-VIDEO-56f5c6">
  </a>
</p>

A ~8M parameter LLM that talks like a tiny crab in an aquatic world, chattable entirely in your browser.

```text
You> hello crab
Hermit> you again. i am near the sand. i have checked this 9 times.

You> are you hungry
Hermit> food detected. i have been exploring the cave waiting. current priority: bubble snacks.

You> integrate x^2
Hermit> integral x^2 dx = x^3/3 + constant

You> latest AI news
Hermit> redstone appears to be a human concept. the temperature is more important. water condition: swirling and clear.

You> who is Elon Musk
Hermit> Elon Musk - Wikipedia: 9 hours ago - Elon Reeve Musk (/ˈiːlɒn/ EE-lon; born June 28, 1971) is a businessman and past public official known for his leadership of Tesla and SpaceX. Musk has been the wealthiest person in the world since 2025; as of May 2026, Forbes estimates his net worth to be US$828 billion.

You> goodnight crab
Hermit> it is now dark. i will remain still. i have checked this 18 times.
```

## Architecture

| Field | Value |
| --- | --- |
| Parameters | 8.15M |
| Layers | 6 |
| Hidden dim | 320 |
| Heads | 8 |
| FFN | 800, SwiGLU |
| Vocab | tokenizer-defined BPE |
| Max sequence | 512 tokens |
| Norm | LayerNorm |
| Position | Learned embeddings |
| LM head | Weight-tied with embeddings |

Small causal transformer. Learned positions, standard attention, LayerNorm, weight tying, and a compact SwiGLU feed-forward block.

# Features
- Runs fully client-side in the browser (ONNX + onnxruntime-web, no backend server)
- Quantized web model (`model.quant.onnx`, ~9MB vs ~36MB fp32) with automatic fallback
- Tool-calling router: FAQ, Wolfram math, and web search, via `<tool>name(query)</tool>` tags plus keyword fallback
- Streaming API (`POST /chat/stream`, SSE) alongside `POST /chat`
- Modular architecture
- Automated tests, ruff, and basedpyright gates
- RLHF (PPO) and DPO fine-tuning stages
- Eval dashboard: perplexity + generation-based persona reward + tool-trigger rate

## Training

Three stages:

1. **SFT** (`scripts/train.sh` → `hermitlm/training/train.py`): next-token prediction on `data/train.jsonl`, produces `checkpoints/best_model.pt`.
2. **RLHF** (`scripts/train_ppo.sh` → `hermitlm/training/ppo.py`): PPO fine-tunes the SFT checkpoint against a heuristic persona-consistency reward (`hermitlm/training/reward.py`) that scores crab-vocabulary usage, terse in-character style, and penalizes repetition, shouting, and character breaks. A KL penalty against the frozen SFT model (the reference policy) keeps the tuned policy from drifting off-distribution. Rollout prompts are drawn from `data/train.jsonl`/`eval.jsonl`; PPO checkpoints land in `checkpoints/ppo/` and are drop-in compatible with the existing inference/bot/API code.
3. **DPO** (`scripts/train_dpo.sh` → `hermitlm/training/dpo.py`): simpler preference-tuning alternative to PPO. Builds chosen/rejected pairs on-policy (4 candidates per prompt, best vs worst by `score_response`, kept only if the score gap clears a margin filter), optimized against a frozen SFT reference. Checkpoints land in `checkpoints/dpo/`.

## Eval

`scripts/eval.py` scores a checkpoint on three axes and writes `report.json`/`report.md`:

- **Perplexity** on `data/eval.jsonl` (proxy, first 50 batches).
- **Persona reward**: `score_response` over the model's own generations for 10 held-out prompts (not reference texts, so checkpoints are comparable).
- **Tool triggers**: offline keyword/tag routing check (no network).

```bash
python scripts/eval.py --checkpoint checkpoints/best_model.pt --out-dir eval_out/sft
python scripts/eval.py --checkpoint checkpoints/ppo/ppo_final.pt --out-dir eval_out/ppo
python scripts/eval.py --checkpoint checkpoints/dpo/dpo_final.pt --out-dir eval_out/dpo
```

Latest comparison (persona measured over 10 sampled generations, so ±0.05 noise is expected):

| Checkpoint | Perplexity | Persona reward |
| --- | --- | --- |
| SFT (`best_model.pt`) | 1.929 | 0.516 |
| PPO (`ppo_final.pt`) | 1.992 | **0.648** |
| DPO (`dpo_final.pt`, margin filter) | 1.930 | 0.449 |

PPO is currently the best persona checkpoint; DPO as configured does not beat SFT.

## Tools

`hermitlm/tools/router.py` is the single entry point. The model can emit function-call tags
(`<tool>search(query)</tool>`, `<tool>math(query)</tool>`, `<tool>faq(query)</tool>`); otherwise
keyword heuristics route to the FAQ responder, Wolfram (`math.py`), or DuckDuckGo search (`web.py`).
Both the FastAPI backend and the eval harness use it. `POST /chat` returns which tool fired.

## API

```bash
uvicorn hermitlm.runtime.api:app --host 0.0.0.0 --port 8000
# or: docker build -t hermitlm . && docker run -p 8000:8000 hermitlm
```

- `GET /health` — status, checkpoint, tokenizer, device.
- `POST /chat` — `{user_id, message, temperature, top_k, max_tokens}` → `{response, tool}`.
- `POST /chat/stream` — same request, server-sent events stream.

## Website

The chat UI in `web/` runs the model entirely client-side: `scripts/export_onnx.py` exports
`checkpoints/best_model.pt` to ONNX, `scripts/quantize_onnx.py` produces the uint8 `model.quant.onnx`
(the page tries it first and falls back to `model.onnx`), and the page uses onnxruntime-web plus a from-scratch
byte-level BPE tokenizer (reading `data/tokenizer.json` directly) to generate responses in
the browser with no backend. The UI streams tokens as they generate and offers temperature/top-K,
persona modes (normal/sleepy/hungry/guard-crab), a stop button, and speech synthesis.
`.github/workflows/deploy.yml` exports the model and publishes
`web/` to GitHub Pages on every push to `main`.

## Development

```bash
conda create -n hermit -c pytorch -c conda-forge python=3.11 cpuonly pytorch tokenizers onnxruntime
conda activate hermit
pip install -e . -r requirements.txt
pytest tests/          # 14 passed, 1 skipped (Wolfram test needs WOLFRAM_APP_ID + network)
ruff check .           # clean
basedpyright           # clean (basic mode, see pyrightconfig.json)
```

Copy `.env` from the documented keys (`HERMIT_CHECKPOINT`, `HERMIT_TOKENIZER`, `HERMIT_DEVICE`,
`WOLFRAM_APP_ID`, …). Never commit `.env`.

## License

MIT
