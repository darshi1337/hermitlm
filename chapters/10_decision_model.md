# Chapter 10 — The Decision Model

> *A tiny second brain reads one 320-number vector and answers: "what did they mean, and do I know it?"*

---

## The 5-year-old analogy

Asking an 8M-parameter crab "is this a greeting?" by prompting it in English is a bad idea. Zero-shot constrained decoding scored only **21.4% accuracy** (`eval_out/decision/report.json`). The prompt is out of distribution.

So instead we freeze the crab, peek at the note it wrote after reading your message — the hidden state — and train a **tiny linear head** on that note. The head is a few hundred numbers. It learns much faster and runs in one pass.

```mermaid
flowchart TD
    A["query: 'hello crab'"] --> B["chat_prompt + tokenize + right-pad"]
    B --> C["frozen HermitLM forward"]
    C --> D["hidden at the TRUE last token: 320 dims"]
    D --> E["standardize: (h − mean)/std"]
    E --> F["head7 → 7 intent logits"]
    E --> G["head_bin → 2 confused logits"]
    E --> H["head21 → 21 fine logits"]
    F --> I["softmax(logits / T)"]
    G --> I
    H --> I
```

---

## The intent label system

`hermitlm/tools/decision_model.py:29-78` defines two label sets.

### 7 coarse intents

| Letter | Name | Covers |
|---|---|---|
| `a` | social | greeting, bye, lonely |
| `b` | needs | food, water, bubbles |
| `c` | tank | glass, environment, movement |
| `d` | sense | light, color, sound, noise |
| `e` | body | state, breathing |
| `f` | condition | temp_cold, temp_hot, night |
| `g` | mind | misc, confused, reflection |

### 21 fine intents

`FINE21_ORDER` is the alphabetical list: `breathing, bubbles, bye, color, confused, environment, food, glass, greeting, light, lonely, misc, movement, night, noise, reflection, sound, state, temp_cold, temp_hot, water`.

The coarse buckets on the 95k training set:

| Bucket | Approximate rows |
|---|---:|
| `a` social | 14,900 |
| `b` needs | 14,800 |
| `c` tank | 11,900 |
| `d` sense | 14,800 |
| `e` body | 11,900 |
| `f` condition | 9,000 |
| `g` mind | 17,900 |

---

## Where the feature comes from

`scripts/train_probe.py:32-49` extracts hidden states at the true last token of each right-padded prompt:

```python
38:    feats = []
39:    for i in range(0, len(queries), batch_size):
40:        batch = queries[i : i + batch_size]
41:        ids = [tokenizer.encode(chat_prompt(q)).ids[-max_seq_len:] for q in batch]
42:        width = max(len(s) for s in ids)
43:        padded = [s + [0] * (width - len(s)) for s in ids]
44:        idx = torch.tensor(padded, dtype=torch.long, device=device)
45:        _, hidden = model.forward_with_hidden_layers(idx)
46:        lengths = torch.tensor([len(s) for s in ids], dtype=torch.long)
47:        # Advanced indexing on Intel XPU hits Level Zero limits: gather on CPU.
48:        feats.append(hidden.cpu()[torch.arange(len(batch)), lengths - 1, :])
49:    return torch.cat(feats)
```

The model is frozen. `forward_with_hidden_layers` (`model.py:155-168`) returns all six block outputs, so the trainer can pick the best layer.

**Right padding is essential.** Adding zeros to the left would shift every learned position embedding, and `README.md:210-211` records a 30-point accuracy loss when that happened.

---

## The linear head

`hermitlm/tools/heads.py:36-41`:

```python
36:    def scores(self, hidden: torch.Tensor) -> torch.Tensor:
37:        x = (hidden - self.mean.to(hidden.device)) / self.std.to(hidden.device)
38:        return x @ self.weight.to(hidden.device).T + self.bias.to(hidden.device)
39:
40:    def log_probs(self, hidden: torch.Tensor) -> torch.Tensor:
41:        return torch.log_softmax(self.scores(hidden) / self.temperature, dim=-1)
```

For a hidden vector `h` and a head with weight matrix `W`, bias `b`, mean `μ`, standard deviation `σ` and temperature `T`:

```
x        = (h − μ) / σ
scores   = W x + b
log p_c  = log( exp(scores_c/T) / Σ_j exp(scores_j/T) )
```

The shipped artifact uses the older "legacy" form (`decision_model.py:203-207`): `scores = linear(hidden) / temperature`, then `log_softmax`.

### Shipped head shapes and temperatures

From `checkpoints/decision_probe.pt` and `web/probe.json`:

| Head | Weight shape | Bias shape | Temperature |
|---|---|---|---:|
| 7-way coarse | `(7, 320)` | `(7,)` | 1.2300633189037424 |
| 2-way confused | `(2, 320)` | `(2,)` | 1.3310915924669515 |
| 21-way fine | `(21, 320)` | `(21,)` | 0.6638396919185293 |
| Tool 4-way | `(4, 320)` | `(4,)` | 0.6785627417882039 |

### Prediction

`decision_model.py:286-292`:

```python
287:    def predict(self, query: str) -> tuple[str, dict[str, float], float]:
288:        h = self._hidden(query, self.layer)
289:        proba7 = torch.exp(self._log_probs(self.head7, h, "coarse"))[0]
290:        proba = {letter: float(p) for letter, p in zip(OPTION_LETTERS, proba7.tolist())}
291:        p_confused = float(torch.exp(self._log_probs(self.head_bin, h, "confused"))[0, 1])
292:        return max(OPTION_LETTERS, key=lambda letter: proba[letter]), proba, p_confused
```

For a zero hidden vector, the coarse head emits only its biases. Those biases are `[-0.01427, -0.08197, -0.01813, 0.0085, -0.00159, -0.0419, 0.06219]`, and after division by `T = 1.23006` plus softmax the largest class is `g` (mind) at **0.1517**. The binary head's biases give `p_confused = sigmoid((0.00603 − 0.06934)/1.33109) = 0.4881`.

That is what "no information" looks like: nearly a uniform guess.

---

## Fitting a head: closed form, not gradient descent

`heads.py:94-99` solves ridge regression directly:

```python
94: def fit_ridge(x, y, alpha=1.0):
95:     x = _standardize(x)
96:     a = x.T @ x + alpha * torch.eye(x.size(1))
97:     y_mean = _one_hot(y, classes).mean(0)
98:     w = torch.linalg.solve(a, x.T @ (_one_hot(y, classes) - y_mean))
99:     return w.T, y_mean
```

The formula is:

```
A   = XᵀX + αI
Wᵀ  = A⁻¹ Xᵀ (Y − ȳ)
b   = ȳ
```

where `Y` is a one-hot label matrix and `ȳ` is the vector of class frequencies.

LDA is also available (`heads.py:102-123`) with Ledoit–Wolf shrinkage of the covariance matrix.

### Layer and solver selection

`heads.py:135-139` chooses candidate layers:

```python
135: def default_layers(num_layers: int) -> list[int]:
137:     first = -(-4 * num_layers // 10)   # ceil(0.4×6) = 3
138:     last = 9 * num_layers // 10        # floor(0.9×6) = 5
139:     return list(range(first, last + 1, 2)) or [num_layers // 2]
```

With 6 blocks this gives layers **[3, 5]**.

`select_head()` (`heads.py:155-191`) tries 7 ridge values (`α = 1e-2 … 1e4`) and LDA at each layer, uses 5-fold stratified cross-validation, and picks the candidate with the lowest out-of-fold NLL after temperature fitting.

---

## Calibration: making confidence honest

A head can rank classes correctly but still report silly probabilities. Calibration fixes the scale without changing the ranking.

### Platt-smoothed targets

`calibration.py:29-44`:

```python
38:    targets = targets.detach().to(dtype=torch.long)
39:    counts = torch.bincount(targets, minlength=n_classes).to(dtype=torch.float64)
40:    own = (counts[targets] + 1.0) / (counts[targets] + 2.0)
41:    rest = (1.0 - own) / (n_classes - 1)
42:    out = rest.unsqueeze(1).expand(targets.numel(), n_classes).clone()
43:    out[torch.arange(targets.numel()), targets] = own
44:    return out
```

For a class with `n_c` examples:

```
target_own  = (n_c + 1) / (n_c + 2)
target_rest = (1 − target_own) / (K − 1)
```

Worked example from the tests (`tests/test_calibration.py:11-16`), labels `[0,0,0,1]` with `K = 2`:

- class 0 appears 3 times → `target_own = (3+1)/(3+2) = 0.8`
- class 1 appears once → `target_own = (1+1)/(1+2) = 2/3`

So rows with label 0 get target `[0.8, 0.2]` and the label-1 row gets `[1/3, 2/3]`. Smoothing stops a near-separable head from claiming 100% confidence.

### Temperature by golden-section search

`calibration.py:81-146` searches `t = log T` in `[−3, 3]` to minimize:

```
NLL(t) = CrossEntropy(scores × exp(−t), targets)
T*     = exp(t*)
```

Binary heads and some multiclass cases use Platt targets instead of the raw labels. The search uses the golden ratio `φ⁻¹ = (√5 − 1)/2 ≈ 0.6180`, tolerance `1e-4`, at most 100 iterations. If the objective is flat, the function returns `T = 1.0`.

Worked example with logits `[2, 1, 0]`:

| T | softmax |
|---|---|
| 1.0 | `[0.6652, 0.2447, 0.0900]` |
| 1.2301 (shipped coarse T) | `[0.6097, 0.2704, 0.1199]` |
| 3.0 (zero-shot fitted T) | `[0.4484, 0.3213, 0.2302]` |

Larger T flattens the distribution.

### Prior correction

`hermitlm/tools/prior.py:10-24` fits a class prior from an unlabelled pool and applies:

```
π_c        = clamp(mean_c(p_c), 1e-6), normalized to sum 1
corrected_c = log p_c − λ log π_c
final       = softmax(corrected)
```

with `0 ≤ λ ≤ 1`. The browser does the same in `web/intent.js:21-32`. The shipped `web/probe.json` has no `priors` key, so this correction is currently a no-op there.

---

## Metrics

All four live in `hermitlm/tools/metrics.py:26-74`.

| Metric | Formula |
|---|---|
| Accuracy | `mean(argmax(p) == y)` |
| NLL | `−mean(log(clamp(p[i, y_i], 1e-12)))` |
| ECE | `Σ_b (n_b/n) · |mean(conf_b) − mean(correct_b)|` over 10 bins |
| Macro F1 | `(1/K) Σ_c 2·P_c·R_c / (P_c + R_c)` |

### ECE worked example

From `eval_out/decision/report.json`, all 1,000 zero-shot rows fall in the confidence bin `(0.10, 0.20]` with mean confidence `0.17` and accuracy `0.214`:

```
ECE = 1.0 × |0.17 − 0.214| = 0.044
```

That matches the report exactly. After calibrating with `T = 3`, mean confidence becomes `0.1516` and `ECE = |0.1516 − 0.214| = 0.0624`: NLL improved from 1.9497 to 1.9455 but ECE got worse.

### NLL and ECE from a tiny example

With `p = [[0.8, 0.2], [0.6, 0.4]]` and targets `[0, 0]`:

```
NLL = −(ln 0.8 + ln 0.6)/2 = −(−0.2231 − 0.5108)/2 = 0.3670
```

With targets `[0, 1]`, predictions are `[0, 0]` (both argmax 0), so accuracy is `0.5`. ECE uses confidences `0.8` and `0.6` with correctness `1` and `0`:

```
ECE = (|0.8 − 1| + |0.6 − 0|)/2 = (0.2 + 0.6)/2 = 0.4
```

### Bootstrap confidence intervals

`metrics.py:77-93` resamples rows `n_boot` times with replacement and takes the 2.5th and 97.5th percentiles. The eval script defaults to 1,000 resamples (`scripts/eval_decision.py:229`). Paired bootstrap (`metrics.py:96-120`) computes intervals for `stat_calibrated − stat_uncalibrated` on the same draws.

---

## Results: templates vs humans

| Setting | 7-way accuracy | Notes |
|---|---:|---|
| Zero-shot constrained decoding | 0.214 | `eval_out/decision/report.json`, n=1000 |
| Frozen-head probe, template eval | 1.000 | `eval_out/decision_probe/report.json`, n=4000 |
| Frozen-head probe, hand-written eval | 0.6212 | `README.md:205`, 126 rows |
| 21-way on hand-written eval | 0.4848 | `README.md:205` |

The huge gap between template and human scores is the honest generalization gap. `README.md:206-208` calls the template numbers flattering and says the next fix is better data, not a bigger head.

Other shipped template numbers:

- NLL `0.0015`, ECE `0.0012` (`README.md:184-185`)
- confused detector precision 1.0, recall 1.0
- 21-way accuracy 0.9982, macro F1 0.9971 (`eval_out/decision_probe/report.json`)
- tool probe accuracy 0.7250 on 80 hand-written queries

---

## Artifact pinning

`hermitlm/tools/artifacts.py:27-51,54-97` stores and verifies SHA-256 hashes of three things:

```
checkpoint_sha256       = c52c0f231e4f26ac35b25b47a24668f71898e3f76e1906da45ecdf4753937dcc
tokenizer_sha256        = 2cb03484072364356f20275281d3d06036a995fc0e09102e8e19b9cad691ce9a
prompt_template_sha256  = 56b5189815dd3d6fea57e8b562446d564bb264083fd965473040419742edc30d
schema_version          = 2
dtype                   = torch.float32
```

Loading a head after a retrain or a prompt-template change raises an error instead of scoring with stale weights.

---

## Export to the browser

`decision_model.export_probe()` (`decision_model.py:343-396`) writes `web/probe.json` with `options`, `names`, `fine_options`, the three heads and their temperatures plus the hashes. `scripts/export_probe.py` is a 23-line wrapper.

When a head has standardization, `FittedHead.effective()` (`heads.py:43-47`) folds it into plain weights for JSON:

```
w_eff = W / σ
b_eff = b − (μ/σ) · Wᵀ
```

Then the browser needs only a dot product and a softmax (`web/intent.js:9-32`).

---

## Math and calculations

### Number of head parameters

```
head7     = 7 × 320 + 7     = 2,247
head_bin  = 2 × 320 + 2     = 642
head21    = 21 × 320 + 21   = 6,741
tool head = 4 × 320 + 4     = 1,284
total     = 10,914 numbers
```

That is 0.13% of the 8,453,120-parameter chat model.

### Probe artifact size

```
raw float32 matrices = (7 + 2 + 21) × 320 × 4 + (7 + 2 + 21) × 4 = 38,520 bytes
web/probe.json       = 212,496 bytes
ratio                = 5.5× (decimal text is larger than binary)
```

### Confusion rate from the eval split

```
cal_n      = min(1000, 5000 // 2) = 1,000
test rows  = 5,000 − 1,000 = 4,000
confused support = 268
rate = 268 / 4,000 = 0.067 = 6.7%
```

### Probability of an intent under the shipped temperature

If the coarse head produces scores `[0.5, 0.4, 0.3, 0.2, 0.1, 0.0, −0.1]` and `T = 1.23006`:

```
scaled = [0.4065, 0.3252, 0.2439, 0.1626, 0.0813, 0, −0.0813]
softmax ≈ [0.1799, 0.1659, 0.1529, 0.1410, 0.1300, 0.1198, 0.1105]
```

---

## Code you should remember

| File and lines | What it does |
|---|---|
| `hermitlm/tools/decision_model.py:29-78` | Intent label sets and persona map |
| `hermitlm/tools/decision_model.py:258-292` | Hidden-state extraction and prediction |
| `hermitlm/tools/heads.py:36-47` | Scoring, log-softmax, export folding |
| `hermitlm/tools/heads.py:94-139` | Ridge, LDA, layer grid |
| `hermitlm/tools/calibration.py:29-146` | Platt targets and temperature search |
| `hermitlm/tools/prior.py:10-24` | Prior fit and correction |
| `hermitlm/tools/metrics.py:26-120` | Accuracy, NLL, ECE, macro F1, bootstrap |
| `scripts/train_probe.py:32-49` | Frozen hidden-state extraction |
| `web/intent.js:9-68` | Browser-side probe inference |

---

**Next:** [Chapter 11 — RLHF with PPO and DPO](11_rlhf.md)
