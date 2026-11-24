# Chapter 7 — Data and Training

> *Training is 32,000 rounds of: guess the next word, measure the mistake, nudge every number a little.*

---

## The 5-year-old analogy

Give the crab a sentence with the last word covered. It guesses. You tell it how wrong it was. It adjusts its 8.45 million knobs by a tiny amount. Repeat 32,000 times, with the "how wrong" signal getting smaller and smaller over time.

Two rules make this work:

1. **Masked loss**: ignore the padding and score only real words.
2. **Learning-rate schedule**: start with tiny steps (warmup), grow to the peak, then slowly shrink (cosine decay).

```mermaid
flowchart TD
    A["train.jsonl: 95,000 rows"] --> B["format_chat + BPE + shift"]
    B --> C["batch: x, y, mask"]
    C --> D["HermitLM forward → logits"]
    D --> E["masked cross-entropy"]
    E --> F["backward → gradients"]
    F --> G["clip norm to 1.0"]
    G --> H["AdamW step with current lr"]
    H --> I["next step, up to 32,000"]
    I -->|"every 200"| J["eval + maybe best_model.pt"]
    I -->|"every 1000"| K["step_N.pt"]
```

---

## Where the data comes from

`hermitlm/training/generate_data.py` is 1,105 lines of templates. `generate_dataset()` (`generate_data.py:1040-1101`):

- builds rows from 21 fine types such as `greeting`, `food`, `water`, `temp_cold`, `breathing`, `confused`
- adds 4,200 natural paraphrases via `build_natural_samples(per_type=200)` (`generate_data.py:994-1004`)
- adds persona-prefixed variants with `add_persona_variants()` at probability 0.25 (`generate_data.py:1031-1037`)
- shuffles and splits 95% / 5%
- writes `train.jsonl`, `eval.jsonl`, `train_openai.jsonl`, `eval_openai.jsonl`

`hermitlm/training/natural_data.py:1-7` says why the natural data exists: the template generator reached about 1.0 on its own eval but only **14/20** on natural paraphrases. Hand-written phrasings and small perturbations (prefix, suffix, typo) close that gap.

### The shipped data on disk

| File | Rows |
|---|---:|
| `data/train.jsonl` | 95,000 |
| `data/eval.jsonl` | 5,000 |
| `data/train_openai.jsonl` | 95,000 |
| `data/eval_openai.jsonl` | 5,000 |

Average formatted length is about 20 words and 169 characters per training row, far below `max_seq_len = 512`.

> ⚠️ The on-disk JSONL files predate the natural-sample and persona-variant code. A fresh run of `generate_dataset()` would produce 104,200 rows, not 95,000. `README.md:149-151` also notes that PPO and DPO were not rerun on the newer corpus.

### Training category counts

The 21 types are not equally frequent in the shipped data. Examples:

| Type | Train rows |
|---|---:|
| `greeting` | 8,910 |
| `misc` | 8,907 |
| `state` | 8,869 |
| `light` | 6,018 |
| `food` | 6,011 |
| `glass` | 3,029 |
| `sound` | 2,854 |

That roughly matches an older 3/2/1 weighting scheme: heavy classes get about 9.4%, medium 6.3%, light 3.1%.

---

## Batching and the shift

`hermitlm/training/dataset.py` produces three tensors per batch (chapter 2):

| Tensor | Shape | Meaning |
|---|---|---|
| `x` | `(B, T)` | input token IDs |
| `y` | `(B, T)` | next-token targets |
| `attention_mask` | `(B, T)` | 1 for real tokens |

`train.py:133-147` builds two loaders:

```python
133:    train_loader, train_dataset = get_dataloader(
134:        os.path.join(tc.data_dir, "train.jsonl"),
135:        tokenizer_path, mc.max_seq_len, tc.batch_size, shuffle=True,
136:    )
...
141:    eval_loader, eval_dataset = get_dataloader(
142:        os.path.join(tc.data_dir, "eval.jsonl"),
143:        tokenizer_path, mc.max_seq_len, tc.batch_size, shuffle=False,
144:    )
```

---

## The loss: masked cross-entropy

`train.py:51-61`:

```python
51: def compute_loss(logits, targets, mask):
52:     vocab_size = logits.size(-1)
53:
54:     loss = torch.nn.functional.cross_entropy(
55:         logits.view(-1, vocab_size),
56:         targets.view(-1),
57:         reduction="none"
58:     )
59:
60:     loss = (loss * mask.view(-1)).sum() / mask.sum()
61:     return loss
```

Let `m_{ij}` be the attention mask and `q_{ij,v}` the softmax probability the model assigns to vocabulary item `v` at position `(i,j)`:

```
CE_{ij} = −log q_{ij, y_ij}
L_SFT   = Σ_{ij} m_{ij} · CE_{ij} / Σ_{ij} m_{ij}
```

### Worked example

Suppose a batch has two rows with 4 and 2 real tokens:

```
mask = [[1, 1, 1, 1],
        [1, 1, 0, 0]]
```

There are 6 ones. If the six per-token cross-entropies are `[0.5, 0.8, 0.3, 0.6, 0.9, 0.4]`:

```
numerator = 0.5 + 0.8 + 0.3 + 0.6 + 0.9 + 0.4 = 3.5
denominator = 6
loss = 3.5 / 6 = 0.583333
```

Padding positions contribute 0 to both sums, so they cannot lower or raise the loss.

### Relationship to perplexity

Perplexity is `exp(loss)`. The shipped best checkpoint has eval loss `0.6541` (`checkpoints/best_model.pt`):

```
exp(0.6541) = 1.9234
```

`eval_out/report.json` records a perplexity proxy of 1.929, computed as `exp(mean loss)` over at most 50 batches (`scripts/eval.py:38-46`).

---

## The learning-rate schedule

`train.py:42-49`:

```python
42: def get_lr(step, config):
43:     if step < config.warmup_steps:
44:         return config.learning_rate * step / config.warmup_steps
45:
46:     progress = (step - config.warmup_steps) / max(1, config.max_steps - config.warmup_steps)
47:     coeff = 0.5 * (1 + math.cos(math.pi * progress))
48:
49:     return config.min_lr + (config.learning_rate - config.min_lr) * coeff
```

With `η = 2e-4`, `η_min = 2e-5`, `W = 800`, `M = 32,000`:

```
s < 800:     η_s = η · s / W
s ≥ 800:     p = (s − W)/(M − W)
             c = ½(1 + cos(π p))
             η_s = η_min + (η − η_min) · c
```

| Step | Phase | Learning rate |
|---:|---|---:|
| 0 | warmup | 0.000000e+0 |
| 100 | warmup | 2.50e−5 |
| 400 | warmup | 1.00e−4 |
| 799 | warmup | 1.9975e−4 |
| 800 | peak | 2.0000e−4 |
| 2,000 | cosine | 1.9934e−4 |
| 8,000 | cosine | 1.7737e−4 |
| 16,000 | cosine | 1.1362e−4 |
| 24,200 | cosine | 4.6360e−5 |
| 30,000 | cosine | 2.1819e−5 |
| 31,999 | cosine | 2.0000e−5 |

Warmup is `800 / 32,000 = 2.5%` of training. The final rate is not exactly `min_lr` because the last call happens at step 31,999.

`train.py:186-188` injects the computed value into the optimizer every step:

```python
186:            lr = get_lr(step, tc)
187:            for pg in optimizer.param_groups:
188:                pg["lr"] = lr
```

---

## AdamW and gradient clipping

`train.py:151-156`:

```python
151:    optimizer = torch.optim.AdamW(
152:        model.parameters(),
153:        lr=tc.learning_rate,
154:        weight_decay=tc.weight_decay,
155:        betas=(0.9, 0.95),
156:    )
```

AdamW keeps two running averages per parameter:

```
m_t = β₁ m_{t−1} + (1 − β₁) g_t          β₁ = 0.9
v_t = β₂ v_{t−1} + (1 − β₂) g_t²         β₂ = 0.95
θ_t = θ_{t−1} − η · m̂_t / (√v̂_t + ε) − η · λ · θ_{t−1}
```

with bias corrections `m̂ = m/(1−β₁^t)` and `v̂ = v/(1−β₂^t)`, `ε = 1e-8`, and weight decay `λ = 0.1`.

Before the optimizer step, `train.py:207-209` clips the gradient norm:

```python
207:                loss.backward()
208:                torch.nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip)
209:                optimizer.step()
```

If the global L2 norm of all gradients exceeds `1.0`, every gradient is multiplied by `1.0 / norm`. This prevents one bad batch from making a giant update.

Mixed precision is enabled only on CUDA (`train.py:158-159`). XPU and CPU use the plain path above.

---

## The training loop

`train.py:166-258` runs the show:

| Lines | Event |
|---|---|
| 167 | `step = 0`, `best_eval = inf` |
| 175–189 | fetch batch, move to device, set LR |
| 190–211 | forward, loss, backward, clip, step, zero grad |
| 216–219 | print mean of last 100 losses every 100 steps |
| 222–237 | evaluate every 200 steps, save `best_model.pt` if improved |
| 240–245 | save `step_N.pt` every 1,000 steps |
| 249–255 | save `final_model.pt` with the full loss list |

The shipped best checkpoint metadata:

```
step       = 24,200
eval_loss  = 0.6540815222263336
perplexity = exp(0.6540815) = 1.9234
```

`final_model.pt` comes from an older 30,000-step run with vocabulary 2,831 and a loss list of length 30,000. Its first losses are around 8.07 and its last around 0.65.

> `train.py` has no resume logic. It only writes checkpoints. The "resume" in `scripts/retrain_resume.ps1` is pipeline-level bookkeeping that skips already-finished shell stages.

---

## Math and calculations

### Batches and passes

```
rows / batch     = 95,000 / 32 = 2,968.75 batches per pass
max_steps        = 32,000
sequences seen   = 32,000 × 32 = 1,024,000
passes over data = 1,024,000 / 95,000 = 10.78 epochs
```

### Token positions

```
upper bound per step  = B × T = 32 × 512 = 16,384
upper bound total     = 16,384 × 32,000 = 524,288,000 positions
```

Real rows are about 20 words, so actual processed tokens are much fewer. `max_seq_len=512` is a ceiling, not the typical length.

### Evaluation and saving frequency

```
eval calls  = floor((32,000 − 1)/200) = 159
save points = steps 1,000 … 31,000 = 31 opportunities
observed artifacts stop at step 29,000 because the shipped run ended at 30,000
```

### Warmup fraction and peak ratio

```
warmup fraction = 800/32,000 = 0.025 = 2.5%
peak / floor    = 2e−4 / 2e−5 = 10×
```

### Cosine coefficient at step 16,000

```
p = (16,000 − 800)/(32,000 − 800) = 15,200/31,200 = 0.487179
c = ½(1 + cos(π × 0.487179)) = ½(1 + 0.04027) = 0.52014
η = 2e−5 + (2e−4 − 2e−5) × 0.52014 = 1.1362e−4
```

---

## Code you should remember

| File and lines | What it does |
|---|---|
| `hermitlm/training/train.py:42-49` | Warmup + cosine learning rate |
| `hermitlm/training/train.py:51-61` | Masked cross-entropy |
| `hermitlm/training/train.py:92-113` | Evaluation over at most 50 batches |
| `hermitlm/training/train.py:151-159` | AdamW and CUDA-only AMP |
| `hermitlm/training/train.py:190-211` | Forward, backward, clip, step |
| `hermitlm/training/train.py:221-255` | Evaluation and checkpoint writes |
| `hermitlm/config.py:22-43` | All training hyperparameters |

---

**Next:** [Chapter 8 — Generation](08_generation.md)
