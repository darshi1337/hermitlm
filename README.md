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
- Modular architecture
- Automated tests
- RLHF (PPO) fine-tuning stage

## Training

Two stages:

1. **SFT** (`scripts/train.sh` → `hermitlm/training/train.py`): next-token prediction on `data/train.jsonl`, produces `checkpoints/best_model.pt`.
2. **RLHF** (`scripts/train_ppo.sh` → `hermitlm/training/ppo.py`): PPO fine-tunes the SFT checkpoint against a heuristic persona-consistency reward (`hermitlm/training/reward.py`) that scores crab-vocabulary usage, terse in-character style, and penalizes repetition, shouting, and character breaks. A KL penalty against the frozen SFT model (the reference policy) keeps the tuned policy from drifting off-distribution. Rollout prompts are drawn from `data/train.jsonl`/`eval.jsonl`; PPO checkpoints land in `checkpoints/ppo/` and are drop-in compatible with the existing inference/bot/API code.

## Website

The chat UI in `web/` runs the model entirely client-side: `scripts/export_onnx.py` exports
`checkpoints/best_model.pt` to ONNX, and the page uses onnxruntime-web plus a from-scratch
byte-level BPE tokenizer (reading `data/tokenizer.json` directly) to generate responses in
the browser with no backend. `.github/workflows/pages.yml` exports the model and publishes
`web/` to GitHub Pages on every push to `main`.

## License

MIT
