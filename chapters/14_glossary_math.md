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

```
BPE merge* = argmax_(a,b) count(a,b)
vocab size = specials + base bytes + merges = 3 + 52 + 3,719 = 3,774
x = ids[:-1]      y = ids[1:]
```

### Chapter 3 — Embeddings

```
h_0 = tok_emb(idx) + pos_emb(arange(T))
w ~ N(0, 0.02²)                         initialization
dropout keep probability p = 0.9        scale survivors by 1/0.9 = 1.1111…
```

### Chapter 4 — Attention

```
head_dim = D/H = 320/8 = 40
Q, K, V  = split( x W_qkv ),  W_qkv ∈ R^{320×960}
scores   = Q Kᵀ / √d_k                  √40 = 6.3246
masked   = scores with future positions set to −∞
weights  = softmax(masked, dim=-1)
out      = weights V → concat heads → W_out
params per block = (320×960 + 960) + (320×320 + 320) = 410,880
```

### Chapter 5 — Transformer block

```
μ = mean(x),  σ² = var(x)
x̂ = (x − μ)/√(σ² + ε),  y = γ x̂ + β       LayerNorm
silu(z) = z / (1 + e^(−z))
FFN(x)  = w3( silu(w2(x)) ⊙ w1(x) )         SwiGLU
x ← x + attn(norm1(x))                      residual 1
x ← x + ffn(norm2(x))                       residual 2
∂(x+f(x))/∂x = I + ∂f/∂x                   gradient highway
```

### Chapter 6 — Full model

```
logits = lm_head( LayerNorm(block_6(…block_1(h_0))))
P = V·D + P_max·D + L(3D² + 3D + D² + D + 3DF) + 4LD + 2D
P = 8,453,120 unique; raw state_dict = 9,660,800
```

### Chapter 7 — Training

```
CE_ij  = −log q_{ij, y_ij}
L_SFT  = Σ m_ij CE_ij / Σ m_ij               masked mean
η_s    = η·s/W                                s < W
p      = (s−W)/(M−W),  c = ½(1+cos πp)        s ≥ W
η_s    = η_min + (η−η_min)·c

AdamW:
m_t = β₁ m_{t−1} + (1−β₁) g_t               β₁ = 0.9
v_t = β₂ v_{t−1} + (1−β₂) g_t²              β₂ = 0.95
θ_t = θ_{t−1} − η·m̂_t/(√v̂_t+ε) − η·λ·θ_{t−1}
clip: g ← g · min(1, 1/‖g‖₂)
```

### Chapter 8 — Generation

```
z_v   = logit_v / T
z'_v  = z_v if z_v ≥ k-th largest else −∞    top-k
p_v   = exp(z'_v − max z') / Σ exp(z'_u − max z')
cutoff = min { k : Σ_{j≤k} p_(j) ≥ topP }     top-p (browser)
p'_i  = p_i / Σ kept p
next  ~ multinomial(p') or inverse-CDF draw
stop when next ∈ {eos_id, bos_id}
```

### Chapter 9 — Tools

```
P(decided at k) = Π_{i<k}(1 − p_i) · p_k
tool dispatch if probe confidence ≥ 0.5
web escalation if p_yes ≥ 0.5
clarify if confidence < 0.75 and p_yes < 0.5
```

### Chapter 10 — Decision model

```
x      = (h − μ)/σ
scores = W x + b
log p  = log_softmax(scores / T)

ridge: A = XᵀX + αI,  Wᵀ = A⁻¹Xᵀ(Y − ȳ)
Platt target: own = (n_c+1)/(n_c+2), rest = (1−own)/(K−1)
temperature: T* = exp(argmin_t NLL(scores·e^{−t}, targets))
prior: corrected = log p − λ log π, then softmax

accuracy = mean(argmax p == y)
NLL      = −mean(log clamp(p[i,y_i], 1e−12))
ECE      = Σ_b (n_b/n)|mean conf_b − mean correct_b|
macro F1 = (1/K) Σ_c 2P_cR_c/(P_c+R_c)
CI       = percentile bootstrap, α = 0.05
```

### Chapter 11 — RLHF

```
R = 2·min(overlap,0.4) − 1.5·rep − len_pen − style_pen − 3·break_pen

PPO:
kl_t   = log π_old − log π_ref
r_t    = −0.15·kl_t  (+ reward at final token)
δ_t    = r_t + γV_{t+1} − V_t
Â_t    = δ_t + γλÂ_{t+1}
R_t    = Â_t + V_t
r_t(θ) = exp(log π_new − log π_old)
L_pol  = −mean min(r Â, clip(r,0.8,1.2) Â)
L_val  = mean(V_new − R)²
L      = L_pol + 0.5 L_val − 0.01 H

DPO:
ℓ_π(y|x) = Σ m_t log π(y_t|x,y_<t) / Σ m_t
r_π      = ℓ_π(chosen) − ℓ_π(rejected)
r_ref    = ℓ_ref(chosen) − ℓ_ref(rejected)
L_DPO    = −log σ(β(r_π − r_ref)),  β = 0.1
```

### Chapter 12 — Export

```
scale      = (r_max − r_min)/255
zero_point = round(−r_min/scale) clipped to [0,255]
q          = clip(round(r/scale) + zero_point, 0, 255)
r_hat      = scale·(q − zero_point)
size ratio = 36,282,163 / 9,215,272 = 3.94×
```

### Chapter 13 — Evaluation

```
perplexity = exp(mean masked CE) ≈ exp(0.657) = 1.929
tool rate  = hits / checks = 2/3
paired diff CI = percentile bootstrap of (stat_b − stat_a) on same draws
```

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
