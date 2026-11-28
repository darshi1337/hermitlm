# 📖 HermitLM From Scratch

> *A guide to the tiny crab model that lives in this repository. Built from the actual code, with every number traced back to a file.*

This is a chapter set that explains **how HermitLM is built, trained, tuned, exported and evaluated**. It follows the same four-step pattern as [How to Train Your GPT](https://github.com/raiyanyahya/how-to-train-your-gpt):

| Step | Format | Purpose |
|---|---|---|
| 1️⃣ **Analogy** | Plain language | Build a picture before the math |
| 2️⃣ **Worked example** | Real numbers from this repo | See each calculation |
| 3️⃣ **Annotated code** | Short quotes with `file:line` | Read the real implementation |
| 4️⃣ **Diagram** | Mermaid | Watch data move through the system |

Every chapter ends with a **Math and calculations** section. Chapter 14 collects every formula into one sheet.

For shorter topic summaries, see the split [README index](../README.md) and the files under `docs/`.

---

## Chapters

| Chapter | What it explains |
|---|---|
| **[0: Overview](00_overview.md)** | What HermitLM is, the whole pipeline, the 8.45M parameter model |
| **[1: Setup and repository map](01_setup.md)** | Install, hardware, data files, checkpoints, retrain scripts |
| **[2: Tokenization](02_tokenization.md)** | Byte-level BPE, special tokens, a real prompt turned into IDs |
| **[3: Embeddings and positions](03_embeddings.md)** | Token vectors, learned positions, weight tying, parameter math |
| **[4: Attention](04_attention.md)** | Q, K, V, scaling by √40, causal mask, a hand-computed head |
| **[5: The transformer block](05_transformer_block.md)** | Pre-norm LayerNorm, residuals, SwiGLU, dropout |
| **[6: The complete model](06_gpt_model.md)** | The forward pass, logits, loss, and the full parameter count |
| **[7: Data and training](07_training.md)** | 95,000 rows, masked cross-entropy, AdamW, warmup + cosine, checkpoints |
| **[8: Generation](08_generation.md)** | Temperature, top-k, top-p, stop tokens, best-of-N, streaming |
| **[9: Tools and routing](09_tools.md)** | Function tags, keywords, probe routing, confusion escalation |
| **[10: The decision model](10_decision_model.md)** | Frozen hidden states, linear heads, calibration, metrics, 7 and 21 intents |
| **[11: RLHF with PPO and DPO](11_rlhf.md)** | Persona reward, clipped surrogate, GAE, preference loss |
| **[12: Export and the browser](12_export_browser.md)** | ONNX, uint8 quantization, onnxruntime-web, SHA-256 cache keys |
| **[13: Evaluation](13_evaluation.md)** | Perplexity, persona reward, tool triggers, ECE, bootstrap intervals |
| **[14: Glossary and formula sheet](14_glossary_math.md)** | Every formula, hyperparameter, file and caveat in one place |

> Start at chapter 0 and read in order. Each chapter uses the vocabulary of the ones before it.

---

## The one-paragraph version

A user message becomes token IDs (chapter 2). Each ID becomes a 320-number vector plus a position vector (chapter 3). Six blocks of attention and SwiGLU mix those vectors (chapters 4–6). The last position produces 3,774 logits, and sampling picks one token at a time (chapter 8). In parallel, the final hidden state feeds tiny linear heads that classify intent and route tools (chapters 9–10). The whole 8.45M-parameter model is exported to ONNX, quantized from 36.3 MB to 9.2 MB, and runs in the browser (chapter 12).

---

## Numbers used throughout

| Quantity | Value | Source |
|---|---|---|
| Unique parameters | 8,453,120 | `hermitlm/training/model.py:208-209`, `docs/02-architecture.md:1-37` |
| Transformer blocks | 6 | `checkpoints/config.json:6` |
| Hidden size | 320 | `checkpoints/config.json:5` |
| Attention heads | 8, 40 dimensions each | `checkpoints/config.json:7`, `model.py:13` |
| Feed-forward hidden size | 800 | `checkpoints/config.json:8` |
| Vocabulary | 3,774 tokens | `data/tokenizer.json` |
| Maximum sequence | 512 tokens | `checkpoints/config.json:4` |
| Training rows | 95,000 train / 5,000 eval | `data/train.jsonl`, `data/eval.jsonl` |
| Best checkpoint | step 24,200, eval loss 0.6541 | `checkpoints/best_model.pt` |
| Web artifact | 9.2 MB quantized / 36.3 MB fp32 | `web/model.quant.onnx`, `web/model.onnx` |

---

*Nothing in these chapters is hypothetical. When the code and the README disagree, the chapter says so and uses the file on disk.*
