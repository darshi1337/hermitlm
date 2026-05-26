# HermitLM

![HermitLM](assets/hermit_1.png)

## HermitLM

A ~8M parameter LLM that talks like a tiny crab in an aquatic world.

Dataset  Model  License  
Train  Chat  Discord Bot

This project exists to show that training your own language model is not magic. No giant lab, no mystery box, no billion-parameter monster required. HermitLM walks through the full path: synthetic data generation, tokenizer training, model architecture, training loop, inference, and a Discord bot that can reply with voice.

It will not write essays or replace frontier models. That is not the point. The point is that every piece is small enough to read, run, modify, and understand.

```text
You> hello crab
Hermit> hello. i am near the kelp. the current feels stable.

You> are you hungry
Hermit> food detected. i prefer soft algae. do not delay.

You> how is the water
Hermit> the water feels clear. breathing is efficient. this is important for survival.

You> what is redstone
Hermit> redstone appears to be a human concept. explain it in terms of food or movement.

You> do you feel alone
Hermit> i exist alone in this environment. the shell provides some structure. this is acceptable.

You> goodnight crab
Hermit> night cycle detected. movement will decrease. i will remain here until light returns.
```

## What Is HermitLM?

HermitLM is a tiny language model that pretends to be a crab. It speaks in short, lowercase sentences about water, food, light, rocks, shells, bubbles, and cautious crab observations. It does not try to be a general assistant. It is a small character model built so the whole training pipeline feels understandable.

The dataset is generated synthetically from crab-themed conversation templates, then converted into chat-style training text. The model is trained from scratch and can run locally for inference or inside a Discord bot.

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

## Install

```powershell
cd C:\Users\DARSHI\Desktop\hermitlm
pip install -r requirements.txt
```

Or install as an editable package:

```powershell
pip install -e .
```

## Prepare Data

```powershell
python -m hermitlm.prepare_data
```

This creates:

- `data/train.jsonl`
- `data/eval.jsonl`
- `data/train_openai.jsonl`
- `data/eval_openai.jsonl`
- `data/tokenizer.json`

## Train

```powershell
python -m hermitlm.train
```

Checkpoints are written to `checkpoints/`.

## Chat

```powershell
python -m hermitlm.inference
```

By default, inference loads:

- `checkpoints/best_model.pt`
- `data/tokenizer.json`

## Discord Bot

Create a `.env` file:

```env
DISCORD_TOKEN=your_discord_bot_token_here
```

Run:

```powershell
python -m hermitlm.discord_bot
```

Mention the bot or use:

```text
!hermit hello crab
```

The bot sends a text reply and an MP3 voice attachment.

## Voice

Voice replies use `gTTS`. Optional speed-up uses `pydub`, which needs `ffmpeg` and `ffprobe`.

With Anaconda:

```powershell
conda install -c conda-forge ffmpeg
```

Verify:

```powershell
ffmpeg -version
ffprobe -version
```

If `ffmpeg` is not available, HermitLM still sends normal-speed gTTS audio.
