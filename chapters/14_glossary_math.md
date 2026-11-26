# Chapter 14 — Glossary and Formula Sheet

> *Everything in one place. Use it as a reference while you read the code.*

---

## The 5-year-old analogy

The earlier chapters were a guided tour. This chapter is the **map and the dictionary**. If you forget what `β` means in DPO or how ECE is computed, come back here.

---

## 1. Model hyperparameters

From `checkpoints/config.json` and `hermitlm/config.py`:

| Symbol | Name | Value |
|---|---|---:|
| `V` | vocabulary size | 3,774 |
| `D` | `d_model` | 320 |
| `L` | `n_layers` | 6 |
| `H` | `n_heads` | 8 |
| `d_k` | `head_dim = D/H` | 40 |
| `F` | `ffn_hidden` | 800 |
| `P_max` | `max_seq_len` | 512 |
| — | `pad_id` / `bos_id` / `eos_id` | 0 / 1 / 2 |
| — | dropout | 0.1 (SFT), 0.0 (PPO/DPO) |

### Training

| Symbol | Name | Value |
|---|---|---:|
| `B` | `batch_size` | 32 |
| `η` | `learning_rate` | 2e−4 |
| `η_min` | `min_lr` | 2e−5 |
| `W` | `warmup_steps` | 800 |
| `M` | `max_steps` | 32,000 |
| `λ` | `weight_decay` | 0.1 |
| — | `grad_clip` | 1.0 |
| — | `eval_interval` / `save_interval` | 200 / 1,000 |
| — | `seed` | 42 |
| — | `betas` (AdamW) | (0.9, 0.95) |

### Decision and runtime

| Name | Value |
|---|---:|
| Coarse intents | 7 (`a`–`g`) |
| Fine intents | 21 |
| Tool classes | 4 (math, web, faq, chat) |
| `YES_THRESHOLD` | 0.5 |
| `CONFUSED_THRESHOLD` (legacy fallback) | 0.9 |
| `TOOL_THRESHOLD` | 0.5 |
| `CLARIFY_THRESHOLD` | 0.75 |
| `BEST_OF` | 3 |
| Default temperature / top-k / max tokens | 0.6 / 20 / 80 |
| Browser top-p | 0.95 |
| Calibration `t_range` | [−3, 3] |
| Head selection `t_range` | [−7, 7] |
| ECE bins | 10 |
| Bootstrap resamples | 1,000 (eval), 2,000 (metrics default) |

---

## 2. Every formula, grouped by chapter

### Chapter 2 — Tokenization

$$ \Large BPE merge* = \arg\max_{a,b} count(a,b) $$

$$ \Large vocab size = specials + base bytes + merges = 3 + 52 + 3,719 = 3,774 $$

$$ \Large x = ids[:-1] y = ids[1:] $$

### Chapter 3 — Embeddings

$$ \Large h_0 = \text{tok emb} + \text{pos emb} $$

$$ \Large w \sim \mathcal{N}(0, 0.02^2) $$

$$ \Large \text{dropout scale} = 1/0.9 = 1.111 $$

### Chapter 4 — Attention

$$ \Large d_{\text{head}} = 320/8 = 40 $$

$$ \Large Q, K, V = \text{split}(x W_{\text{qkv}}) $$

$$ \Large \text{scores} = QK^T/\sqrt{40} $$

### Chapter 5 — Transformer block

$$ \Large \mu = mean(x), \sigma^2 = var(x) $$

$$ \Large \hat{x} = (x - \mu)/\sqrt{\sigma^2 + \varepsilon}, y = \gamma \hat{x} + \beta\text{LayerNorm} $$

$$ \Large silu(z) = z / (1 + e^{-z}) $$

$$ \Large FFN(x) = w3( silu(w2(x)) \odot w1(x) )\text{SwiGLU} $$

$$ \Large x arrow x + attn(norm1(x)) residual 1 $$

$$ \Large x arrow x + ffn(norm2(x)) residual 2 $$

$$ \Large \partial (x+f(x))/\partial x = I + \partial f/\partial x\text{gradient highway} $$

### Chapter 6 — Full model

$$ \Large logits = lm_{head}( LayerNorm(block_{6}(\ldots \text{block}_{1}(h_{0})))) $$

$$ \Large P = V\cdot D + P_{max}\cdot D + L(3D^2 + 3D + D^2 + D + 3DF) + 4LD + 2D $$

$$ \Large P = 8,453,120 unique; raw state_{dict} = 9,660,800 $$

### Chapter 7 — Training

$$ \Large CE_{ij} = -\log q_{ij, y_{ij}} $$

$$ \Large L_{SFT} = \sum m_{ij} CE_{ij} / \sum m_{ij}\text{masked mean} $$

$$ \Large \eta_{s} = \eta\cdot s/W s < W $$

$$ \Large p = (s-W)/(M-W), c = \frac{1}{2}(1+cos \pip) s \geq W $$

$$ \Large \eta_{s} = \eta_{min} + (\eta-\eta_{min})\cdot c $$

$$ \Large \text{AdamW:} $$

$$ \Large m_{t} = \beta_{1} m_{t-1} + (1-\beta_{1}) g_{t} \beta_{1} = 0.9 $$

$$ \Large v_{t} = \beta_{2} v_{t-1} + (1-\beta_{2}) g_{t}^2 \beta_{2} = 0.95 $$

$$ \Large \theta_{t} = \theta_{t-1} - \eta\cdot \hat{m}_t/(\sqrt{\hat{v}}_t+\varepsilon) - \eta\cdot \lambda\cdot \theta_{t-1} $$

$$ \Large clip: g arrow g \cdot \min(1, 1/\|g\|_2) $$

### Chapter 8 — Generation

$$ \Large z_{v} = logit_{v} / T $$

$$ \Large z'_v = z_{v} if z_{v} \geq k-th largest else -\infty\text{top-k} $$

$$ \Large p_{v} = \exp(z'_v - max z') / \sum \exp(z'_u - max z') $$

$$ \Large cutoff = min { k : \sum_{j\leq k} p_(j) \geq topP }\text{top-p (browser)} $$

$$ \Large p'_i = p_{i} / \sum\text{kept p} $$

$$ \Large next \sim multinomial(p') or inverse-CDF draw $$

$$ \Large stop when next \in {eos_{id}, bos_{id}} $$

### Chapter 9 — Tools

$$ \Large P(decided at k) = \prod_{i<k}(1 - p_{i}) \cdot p_{k} $$

$$ \Large tool dispatch if probe confidence \geq 0.5 $$

$$ \Large web escalation if p_{yes} \geq 0.5 $$

$$ \Large clarify if confidence < 0.75 and p_{yes} < 0.5 $$

### Chapter 10 — Decision model

$$ \Large x = (h - \mu)/\sigma $$

$$ \Large scores = W x + b $$

$$ \Large \log p = log_{softmax}(scores / T) $$

$$ \Large ridge: A = X^TX + \alpha I, W^T = A^{-1}X^T(Y - \bar{y}) $$

$$ \Large Platt target: own = (n_{c}+1)/(n_{c}+2), rest = (1-own)/(K-1) $$

$$ \Large temperature: T* = \exp(argmin_{t} NLL(scores\cdot e^{-t}, targets)) $$

$$ \Large prior: corrected = \log p - \lambda \log \pi, then softmax $$

$$ \Large accuracy = mean(argmax p == y) $$

$$ \Large NLL = -mean(\log clamp(p[i,y_{i}], 10^{-12})) $$

$$ \Large ECE = \sum_b (n_{b}/n)|mean conf_{b} - mean correct_{b}| $$

$$ \Large macro F1 = (1/K) \sum_c 2P_{cR\_c}/(P_{c}+R_{c}) $$

$$ \Large CI = percentile bootstrap, \alpha = 0.05 $$

### Chapter 11 — RLHF

$$ \Large R = 2\min(\text{overlap}, 0.4) - 1.5\,\text{rep} - \text{len pen} - \text{style pen} - 3\,\text{break pen} $$

### Chapter 12 — Export

$$ \Large \text{scale} = (r_{\max} - r_{\min})/255 $$

$$ \Large \text{zero point} = \text{round}(-r_{\min}/\text{scale}) $$

$$ \Large q = \text{clip}\big(\text{round}(r/\text{scale}) + \text{zero point}, 0, 255\big) $$

### Chapter 13 — Evaluation

$$ \Large \text{PPL} = 1.929 $$

$$ \Large \text{tool rate} = 2/3 $$

$$ \Large \text{paired CI} = \text{bootstrap of } (\text{stat}_b - \text{stat}_a) $$

---

## 3. Parameter breakdown

| Component | Calculation | Parameters |
|---|---|---:|
| Token embedding | 3,774 × 320 | 1,207,680 |
| Position embedding | 512 × 320 | 163,840 |
| 6 × attention QKV | 6 × (320×960 + 960) | 1,848,960 |
| 6 × attention output | 6 × (320×320 + 320) | 616,320 |
| 6 × SwiGLU FFN | 6 × 3 × 320 × 800 | 4,608,000 |
| 12 block LayerNorms | 12 × 640 | 7,680 |
| Final LayerNorm | 640 | 640 |
| LM head | tied | 0 |
| **Unique total** | | **8,453,120** |
| Duplicate tied weight | 3,774 × 320 | 1,207,680 |
| **Raw checkpoint** | | **9,660,800** |

---

## 4. File map

| Path | Role |
|---|---|
| `hermitlm/config.py` | Model, training and PPO dataclasses |
| `hermitlm/settings.py` | Environment defaults and thresholds |
| `hermitlm/training/model.py` | All neural network code |
| `hermitlm/training/dataset.py` | Data formatting and batching |
| `hermitlm/training/train.py` | SFT loop |
| `hermitlm/training/generate_data.py` | Template and persona data generation |
| `hermitlm/training/natural_data.py` | Natural paraphrases |
| `hermitlm/training/reward.py` | Persona scorecard |
| `hermitlm/training/ppo.py`, `dpo.py` | Optional preference tuning |
| `hermitlm/runtime/inference.py` | Chat engine |
| `hermitlm/runtime/api.py` | FastAPI endpoints |
| `hermitlm/tools/router.py` | Tool cascade |
| `hermitlm/tools/decision_model.py` | Probe heads and classification |
| `hermitlm/tools/heads.py` | Ridge, LDA, head selection |
| `hermitlm/tools/calibration.py` | Platt targets, temperature search |
| `hermitlm/tools/prior.py` | Batch prior correction |
| `hermitlm/tools/metrics.py` | Accuracy, NLL, ECE, F1, bootstrap |
| `hermitlm/tools/artifacts.py` | SHA-256 pinning |
| `scripts/export_onnx.py`, `quantize_onnx.py` | Browser artifacts |
| `scripts/train_probe.py`, `train_tool_probe.py` | Head fitting |
| `scripts/fit_probe_prior.py` | Label-free prior fitting into a probe artifact |
| `scripts/train_yes_score_heads.py` | Yes/no and score head fitting |
| `scripts/eval_aux_heads.py` | Score-head vs heuristic NLL report |
| `scripts/eval.py`, `eval_decision.py` | Evaluation |
| `scripts/retrain_all.sh`, `retrain_resume.ps1` | Pipeline runners |
| `web/` | Browser UI and ONNX artifacts |
| `.github/workflows/deploy.yml` | Pages deployment |

---

## 5. Glossary

| Term | Meaning |
|---|---|
| **BPE** | Byte pair encoding. A tokenizer that joins frequent pieces. |
| **Byte-level** | The smallest pieces are bytes, so any text can be encoded. |
| **Logit** | An unnormalized score before softmax. |
| **Softmax** | Turns scores into probabilities that sum to 1. |
| **Attention** | A weighted look-back over previous tokens. |
| **Causal mask** | Blocks attention to future positions. |
| **Pre-norm** | Normalize before a sublayer, then add a residual. |
| **SwiGLU** | Gated feed-forward network using SiLU. |
| **Weight tying** | Sharing the embedding matrix with the output head. |
| **Cross-entropy** | −log probability assigned to the correct token. |
| **Perplexity** | exp(loss). Lower means less surprised. |
| **AdamW** | Adam optimizer with decoupled weight decay. |
| **Warmup** | Linearly increasing learning rate at the start. |
| **Cosine decay** | Slowly lowering the learning rate with a cosine curve. |
| **Temperature** | Divides logits to sharpen or flatten sampling. |
| **Batch prior (L0)** | Mean predicted class probability over an unlabelled pool; subtracted as `λ log π` to undo label bias. |
| **Option rotation** | Repeats a multiple-choice prompt with the classes assigned to different letters and averages the result. |
| **p_yes** | Calibrated probability that a message is out of scope; replaces the ad-hoc confused flag. |
| **Expected score** | `sum(k * p_k)` over calibrated reply-quality levels, used to rerank best-of-N candidates. |
| **top-k** | Keep only the k largest logits. |
| **top-p** | Keep the smallest set reaching cumulative probability p. |
| **EOS / BOS** | End-of-sequence and beginning-of-sequence markers. |
| **Probe** | A small classifier trained on frozen hidden states. |
| **Platt scaling** | Smoothing hard targets to avoid overconfidence. |
| **Temperature calibration** | Fitting T to minimize log loss. |
| **ECE** | Gap between confidence and accuracy. |
| **Macro F1** | Unweighted mean of per-class F1 scores. |
| **Bootstrap CI** | Confidence interval from resampling with replacement. |
| **PPO** | Policy gradient with a clipped update and a KL penalty. |
| **GAE** | Generalized advantage estimate, balancing bias and variance. |
| **DPO** | Preference optimization without a value network or rollouts. |
| **ONNX** | Portable model graph format. |
| **Dynamic quantization** | Offline integer weights, runtime activation scales. |
| **SHA-256 pinning** | Hash checks that stop stale artifacts from loading. |

---

## 6. Known mismatches in the repository

These are real. They are listed so you do not debug them twice.

| Issue | Details |
|---|---|
| Stale config default | `HermitConfig.vocab_size = 2833`; shipped tokenizer is 3,774 |
| Stale local web artifacts | Local ONNX has 2,831-token vocab; deploy CI regenerates it |
| Three vocab eras | 2,833 default, 2,831 older PPO/DPO/final run, 3,774 current |
| Two training runs | `best_model.pt` step 24,200; `final_model.pt` from an older 30k run |
| README vs reports | README reward 0.495 vs `eval_out/report.json` 0.378 |
| Probe template vs human | 1.000 template accuracy vs 0.6212 on 126 hand-written rows |
| `--epochs 30` in retrain scripts | Neither probe trainer defines that flag; remove it |
| Tags parsed only in user text | Generated tool tags are not rescanned |
| No top-p in Python | top-p exists only in `web/model.js` |
| No true API token streaming | `/chat/stream` streams whitespace-split words |

---

## 7. Quick sanity checks

```bash
# parameter count with the shipped vocabulary
python - <<'PY'
from hermitlm.config import HermitConfig
from hermitlm.training.model import HermitLM
c = HermitConfig(); c.vocab_size = 3774
m = HermitLM(c)
print(m.param_count())   # 8453120
PY

# dataset sizes
wc -l data/train.jsonl data/eval.jsonl data/human_eval.jsonl

# artifact sizes
stat -c '%n %s' web/model.onnx web/model.quant.onnx web/probe.json

# evaluation
python scripts/eval.py --checkpoint checkpoints/best_model.pt --out-dir eval_out/sft
python scripts/eval_decision.py --method probe --max-n 5000 --cal-n 1000 \
  --bootstrap 1000 --out-dir eval_out/decision_probe
python scripts/eval_decision.py --method logit --max-n 400 --cal-n 200 \
  --bootstrap 0 --out-dir eval_out/decision_order

# refresh the shipped L0 prior and the browser probe
python scripts/fit_probe_prior.py --max-n 4000
python scripts/train_yes_score_heads.py --score-n 6000 --device xpu
python scripts/export_probe.py --output web/probe.json
python scripts/eval_aux_heads.py --out-dir eval_out/aux_heads

# deploy diagnostics
python -m hermitlm.tools.check
pytest -q
ruff check .
```

---

**You have reached the end.** Start at [Chapter 0](00_overview.md) if you want the tour again, or open the code with the file map above.
