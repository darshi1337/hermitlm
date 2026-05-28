<h1 align="center">HermitLM</h1>

<p align="center">
  <a href="https://huggingface.co/datasets/darshi1337/hermitlm-100k-generic">
    <img alt="Dataset" src="https://img.shields.io/badge/Dataset-hermitlm--100k-4b5563">
  <a href="https://huggingface.co/darshi1337/hermitlm-8M">
  <img alt="Model" src="https://img.shields.io/badge/Model-hermitlm--8M-b65f36">
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

A ~8M parameter LLM and discord bot that talks like a tiny crab in an aquatic world.

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
- Discord bot integration
- Wolfram-powered math solving
- Web search routing
- SQLite memory system
- FAQ routing
- Voice responses
- FastAPI backend
- EC2 deployment
- Modular architecture
- Automated tests

## Discord Bot

Invitation link:

```text
https://discord.com/oauth2/authorize?client_id=1124355443683233864&scope=bot&permissions=274877975552
```

## License

MIT
