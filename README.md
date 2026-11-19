<h1 align="center">HermitLM</h1>

<p align="center">
  <a href="https://huggingface.co/datasets/darshi1337/hermitlm-100k-generic">
    <img alt="Dataset" src="https://img.shields.io/badge/Dataset-hermitlm--100k-4b5563">
  </a>
  <a href="https://huggingface.co/darshi1337/hermitlm-8M">
    <img alt="Model" src="https://img.shields.io/badge/Model-hermitlm--8M-b65f36">
  </a>
  <img alt="License" src="https://img.shields.io/badge/License-MIT-3fa34d">
  <a href="https://hub.xerj.org/contributors/darshi1337">
    <img alt="XERJ Corpus Hub contributor" src="https://hub.xerj.org/contributors/darshi1337/badge.svg">
  </a>
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
Hermit> you again. the salinity feels stable. i note soul sand near at the cave entrance.

You> are you hungry
Hermit> i am interested in food. i prefer the sea pickles. do not delay. activity will continue as pretending to be sand.

You> integrate x^2
Hermit> aah math! i hate it. still here is the answer from wolfram: integral x^2 dx = x^3/3 + constant

You> latest AI news
Hermit> Google News - Artificial intelligence - Latest: Read full articles, watch videos, browse thousands of titles and more on the "Artificial intelligence" topic with Google News.

You> goodnight crab
Hermit> night cycle detected. the small cave appears calmer now.
```

Chat replies are sampled, so exact wording varies. The math intro is picked at random from a
pool of crab-flavoured lines, and web results depend on live search availability.

## Architecture

| Field | Value |
| --- | --- |
| Parameters | 8.45M (8,453,120) |
| Layers | 6 transformer blocks |
| Hidden dim (`d_model`) | 320 |
| Attention heads | 8 (40 dims per head) |
| FFN | 800 hidden, SwiGLU / SiLU |
| Vocabulary | 3,774 byte-level BPE tokens |
| Max sequence | 512 tokens |
| Norm | LayerNorm (pre-norm residual blocks) |
| Position | Learned embeddings |
| LM head | Weight-tied with token embeddings |
| Dropout | 0.1 (disabled during probe fitting) |
| Web artifact | uint8 dynamic quantization (~9.8 MB vs 38.7 MB fp32) |

Small causal transformer. Learned positions, standard attention, LayerNorm, weight tying, and a compact SwiGLU feed-forward block.

### How the 8.45M count is derived

`HermitLM.param_count()` counts unique parameters (the tied `lm_head` is not counted twice):

| Component | Calculation | Parameters |
| --- | --- | --- |
| Token embedding | 3,774 × 320 | 1,207,680 |
| Position embedding | 512 × 320 | 163,840 |
| 6 × attention QKV | 6 × (320 × 960 + 960) | 1,848,960 |
| 6 × attention output | 6 × (320 × 320 + 320) | 616,320 |
| 6 × SwiGLU FFN | 6 × (3 × 320 × 800) | 4,608,000 |
| 6 × block LayerNorms | 6 × (2 × 320 × 2) | 7,680 |
| Final LayerNorm | 2 × 320 | 640 |
| LM head | tied to token embedding | 0 |
| **Total** | | **8,453,120** |

The raw checkpoint `state_dict` sums to 9,660,800 because it stores `lm_head.weight` a second
time; weight tying removes that duplicate 1,207,680-parameter matrix.

## Features
- Runs fully client-side in the browser (ONNX + onnxruntime-web, no backend server)
- Quantized web model (`model.quant.onnx`, ~9.8 MB vs ~38.7 MB fp32) with automatic fp32 fallback when `model.onnx` is present
- Tool-calling router: FAQ, Wolfram math, and web search, via `<tool>name(query)</tool>` tags plus keyword fallback
- Probe decision model: single-pass intent classification into 7 coarse intents, plus an out-of-scope score. High `p_confused` escalates to web search. The browser UI shows an intent chip on each reply.
- Fine-grained 21-way intent head and tool-routing probe (math/web/faq/chat) on the same frozen features.
- Calibrated heads: bounded temperature search with Platt-smoothed binary targets, bootstrap CIs, and hash-checked head files.
- Best-of-N reply rerank by persona reward, confidence-gated clarifying questions, persona auto-router, and intent analytics (`GET /stats/intents`).
- Streaming API (`POST /chat/stream`, SSE) alongside `POST /chat`; math replies use a randomly
  chosen crab-flavoured intro before the Wolfram/local answer
- Modular architecture
- Automated tests and ruff gates, plus a `python -m hermitlm.tools.check` deploy diagnostic
- RLHF (PPO) and DPO fine-tuning stages
- Eval dashboards: perplexity + generation-based persona reward + tool-trigger rate, plus decision-model accuracy/calibration

## Training

Training defaults to the **Intel Arc GPU** via PyTorch's XPU backend (`HERMIT_TRAIN_DEVICE=xpu`). On an NVIDIA machine set `HERMIT_TRAIN_DEVICE=cuda`; use `=cpu` only for debugging. Probe fitting defaults to CPU (`--device cpu`) because it is frozen-model inference and avoids flaky XPU behaviour; pass `--device xpu` or `--device cuda` to move it. Training and probe entry points fail loudly if the requested accelerator is unavailable. Install the matching wheel with `pip install torch --index-url https://download.pytorch.org/whl/xpu`. Data generation (`prepare_data.py`, `build_tool_data.py`) and ONNX export stay on CPU.

The full pipeline can be rerun with:

```bash
# Windows / Intel Arc (XPU, default here)
powershell -ExecutionPolicy Bypass -File scripts\retrain_resume.ps1
# Linux / NVIDIA CUDA
HERMIT_TRAIN_DEVICE=cuda bash scripts/retrain_all.sh
```

`scripts/retrain_resume.ps1` is resume-safe: completed steps are recorded in `eval_out/retrain_done.txt`
and skipped on the next run. `scripts/retrain_all.sh` is a straight full run with no resume state.
`scripts/install_retrain_task.ps1` registers a logon-triggered task so a reboot resumes the Windows
retrain automatically.

Three stages:

1. **SFT** (`scripts/train.sh` → `hermitlm/training/train.py`): next-token prediction on `data/train.jsonl`, produces `checkpoints/best_model.pt`.

2. **RLHF** (`scripts/train_ppo.sh` → `hermitlm/training/ppo.py`): PPO fine-tunes the SFT checkpoint against a heuristic persona-consistency reward (`hermitlm/training/reward.py`). The reward scores crab-vocabulary use and terse in-character style. It penalizes repetition, shouting, and character breaks. A KL penalty against the frozen SFT model (the reference policy) keeps the tuned policy from drifting off-distribution. Training draws rollout prompts only from `data/train.jsonl` so `eval.jsonl` stays held out. PPO checkpoints land in `checkpoints/ppo/` and stay drop-in compatible with the inference, chat UI, and API code.

3. **DPO** (`scripts/train_dpo.sh` → `hermitlm/training/dpo.py`): simpler preference-tuning alternative to PPO. It builds chosen/rejected pairs on-policy (4 candidates per prompt, best vs worst by `score_response`). A pair survives only if the score gap clears a margin filter. It optimizes against a frozen SFT reference. Checkpoints land in `checkpoints/dpo/`.

## Eval

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

## Tools

`hermitlm/tools/router.py` is the single entry point. The model can emit function-call tags
(`<tool>search(query)</tool>`, `<tool>math(query)</tool>`, `<tool>faq(query)</tool>`). Otherwise
keyword heuristics route to the FAQ responder, Wolfram (`math.py`), or DuckDuckGo search (`web.py`).
Both the FastAPI backend and the eval harness use it. `POST /chat` returns which tool fired.

`route_with_intent()` also attaches the probe decision-model verdict. It escalates
out-of-scope messages (`p_confused >= HERMIT_CONFUSED_THRESHOLD`, default 0.9) to web search
when no other tool fired.

## Decision model

A single-pass ("System-1") intent classifier on top of the frozen chat model, in the spirit of
[Build your own decision model](https://nishtahir.com/build-your-own-decision-model/).
Zero-shot constrained decoding fails here with 21% accuracy. The classify prompt is out-of-distribution
for the 8M crab-chat model. Thus `scripts/train_probe.py` trains tiny linear heads on frozen
last-token hidden states instead. The heads cover 7 coarse intents over the 21 persona `type` labels,
plus a confused-vs-rest out-of-scope head.

| Intent | Covers | Precision | Recall | F1 |
| --- | --- | --- | --- | --- |
| `a` social | greeting, bye, lonely | 1.00 | 1.00 | 1.00 |
| `b` needs | food, water, bubbles | 1.00 | 1.00 | 1.00 |
| `c` tank | glass, environment, movement | 1.00 | 1.00 | 1.00 |
| `d` sense | light, color, sound, noise | 1.00 | 1.00 | 1.00 |
| `e` body | state, breathing | 1.00 | 1.00 | 1.00 |
| `f` condition | temp_cold, temp_hot, night | 1.00 | 0.9983 | 0.9992 |
| `g` mind | misc, confused, reflection | 0.9983 | 1.00 | 0.9991 |

Held-out (`data/eval.jsonl`, n=4000, temperature fitted on a disjoint 1000-row split):
accuracy 0.9998, macro F1 0.9998, NLL 0.0015, ECE 0.0012. The confused detector reaches
precision 1.0 at recall 1.0. These are template-eval numbers; see the human eval below.

Temperatures come from a bounded golden-section search over log T, not a coarse grid
(`hermitlm/tools/calibration.py`). Binary heads use Platt-smoothed targets, which stops
near-separable scores from collapsing to 0/1 confidence. `scripts/eval_decision.py` fits T on
a disjoint calibration split and reports 95% bootstrap CIs plus a paired
calibrated-minus-uncalibrated table. Head files carry schema version 2, SHA-256 hashes of the
checkpoint, tokenizer and prompt template, plus dtype and selected layer. Loading a head after
a retrain or prompt change raises an error instead of scoring with stale weights.

The 21-way head (`FINE21_ORDER` in `hermitlm/tools/decision_model.py`) reaches accuracy 0.9932
and macro F1 0.9930 on the template split. Those numbers flatter the model. The training data now
includes ~4,200 hand-written natural paraphrases, but the only honest measure is the 126-row
hand-written held-out set in `data/human_eval.jsonl`:

```
python scripts/eval_decision.py --eval data/human_eval.jsonl --method probe \
  --max-n 126 --cal-n 60 --out-dir eval_out/human
```

Latest result: **7-way accuracy 0.6212** (macro F1 0.6075) and **21-way accuracy 0.4848**
(macro F1 0.4199). Calibration temperature fitted on only 60 rows inflates confidence poorly here;
treat this as a lower bound. The gap from 0.9998 on templates is the real generalization gap, and
it is the next thing to fix — not a bigger head.

Training batches right-pad to match `dataset.collate_fn`. Learned positions stay in distribution
only that way. Left padding silently shifts positions and once cost 30 points of accuracy here.

## Calibration toolkit

The decision stack includes label-free and supervised calibration pieces:

- `hermitlm/tools/prior.py`: batch prior correction (`L0`). A prior fitted from an unlabelled
  pool can be applied as `log p - lambda * log pi`; it is stored in probe artifacts and applied
  by the API and browser UI.
- `hermitlm/tools/heads.py` + `scripts/train_probe.py`: `L2` head selection over hidden layers,
  ridge/LDA solvers and regularization, chosen by stratified k-fold OOF NLL. Selected heads
  store standardization, temperature and OOF metrics in the artifact; `export_probe.py` folds
  standardization into JSON weights for the browser.
- `DecisionModel.predict_proba(..., rotations=k)`: option-order rotation with geometric-mean
  recombination for the zero-shot constrained-decoding path.
- `python -m hermitlm.tools.check`: validates label tokens, batch consistency, hidden layers,
  prompt-template hashes and artifact selection before deploy.

Artifact metadata now also pins the prompt-template hash, dtype and selected layer, so a
probe cannot be silently paired with a changed prompt or precision.

Evaluated on the hand-written human eval, the currently shipped last-layer probe still
generalizes better than OOF-selected ridge/LDA heads (0.62 coarse / 0.48 fine vs 0.62 / 0.35),
so the shipped artifact keeps that head. The selection/prior tooling remains available for
future data improvements.

## Tool routing

`scripts/build_tool_data.py` writes synthetic tool data (templates plus hand-authored natural
queries) to `data/tool_train.jsonl` / `data/tool_eval.jsonl`. `scripts/train_tool_probe.py` trains
a 4-way head (math/web/faq/chat) into `checkpoints/decision_tool_probe.pt`. Routing order is
tag, then keywords (93% precise when they fire), then the probe on the residue, then confused
escalation. Training now includes 50 extra natural web queries and 28 hard negatives (chat/FAQ
messages containing web trigger words). Scores on the 80 hand-authored eval queries after the
latest retrain: probe **0.7250** (math 0.80, web 0.65, faq 0.60, chat 0.85), up from 0.6125 —
the web class went from 0.15 to 0.65.

## Reply quality and analytics

`POST /chat` now does more than generate one reply:

- Best-of-N (`engine.best_of_n`, `HERMIT_BEST_OF`, default 3). The API samples N candidates and
keeps the best by persona reward. Lower the value to 1 for less latency.
- Clarifying gate (`HERMIT_CLARIFY`, default on, threshold 0.75). When no tool fires and intent
confidence lands below the threshold without confusion, the API asks which of the top two
intents the user means instead of guessing.
- Persona auto-router. Coarse intent maps to UI persona mode (needs to hungry, sense to guard,
body and condition to sleepy). The API returns the persona and prefixes the prompt with
`[<persona> crab]` (for example `[hungry crab]`), matching the prefixed variants that
`generate_data.add_persona_variants()` writes into training. Set `HERMIT_PERSONA_PROMPT=0` to disable it.
- Analytics. Every reply logs its verdict to `intent_log` in `data/hermit.db`.
`GET /stats/intents` returns totals by label and tool plus the confused rate.

```bash
python scripts/train_probe.py --per-class 1500 --per-fine 400 --out checkpoints/decision_probe.pt
python scripts/train_tool_probe.py --out checkpoints/decision_tool_probe.pt
python scripts/eval_decision.py --method probe --max-n 5000 --cal-n 1000 --bootstrap 1000 --out-dir eval_out/decision_probe
python scripts/eval_decision.py --eval data/human_eval.jsonl --method probe --max-n 126 --cal-n 60 --out-dir eval_out/human
python scripts/export_probe.py --output web/probe.json   # also runs in deploy CI
python -m hermitlm.tools.check                           # pre-deploy diagnostics
```

`HermitModel.classify()` (`web/model.js`) + `web/intent.js` run the same probe in-browser off the
`last_hidden` ONNX output. This costs one extra inference step and no extra download.

## API

```bash
uvicorn hermitlm.runtime.api:app --host 0.0.0.0 --port 8000
# or: docker build -t hermitlm . && docker run -p 8000:8000 hermitlm
```

- `GET /health`: status, checkpoint, tokenizer, device, probe.
- `GET /stats/intents`: totals by intent label and tool, plus the confused rate.
- `POST /chat`: `{user_id, message, temperature, top_k, max_tokens}` → `{response, tool, intent, persona}`. `intent` holds the coarse label, the confidence, and `p_confused`. High `p_confused` escalates to web search.
- `POST /chat/stream`: same request, server-sent events stream.

```json
POST /chat {"user_id": "demo", "message": "hello crab"}
→ {"response": "oh. it's you...", "tool": null,
   "intent": {"label": "a", "name": "social", "confidence": 1.0, "p_confused": 0.0067},
   "persona": null}
```

## Website

The chat UI in `web/` runs the model entirely client-side. `scripts/export_onnx.py` exports
`checkpoints/best_model.pt` to ONNX. `scripts/quantize_onnx.py` produces the uint8 `model.quant.onnx`.
The page tries the quantized model first. Deploy drops the full fp32 `model.onnx` from the
Pages artifact (deploy.yml), so only the ~9.8 MB quantized artifact ships. The fp32 fallback
still works when `web/` is served locally or self-hosted with `model.onnx` present.

The page uses onnxruntime-web plus a from-scratch byte-level BPE tokenizer. `export_onnx.py`
copies `data/tokenizer.json` into `web/tokenizer.json` for the browser. The page generates
responses in the browser with no backend. A generated `web/asset-manifest.json` keys the browser Cache API
by the quantized model's SHA-256, so new deploys cannot pair stale cached weights with a fresh probe.

The UI streams tokens as they generate and offers temperature/top-K,
persona modes (normal/sleepy/hungry/guard-crab), a stop button, and speech synthesis.

The page has a live canvas beach scene: flowing blue contour lines like a hand-drawn shoreline,
three animated wave layers at the bottom, and a soft white foam edge. The speaker button in the
header enables synthesized ambient water and periodic shore splashes using the Web Audio API —
there are no audio files to download. All motion respects `prefers-reduced-motion`.

Every reply also shows a probe decision-model intent chip (`web/intent.js` + `web/probe.json`).
Classification costs a single forward pass over the prompt. A high out-of-scope score links to a web search.
The chip shows coarse and fine intent with confidence. Low confidence asks a clarifying question
with quick-reply chips instead of guessing. Settings shows per-intent session counts.

`.github/workflows/deploy.yml` exports the model and publishes
`web/` to GitHub Pages on every push to `main`.

## Hugging Face

Model repo: `darshi1337/hermitlm-8M`. Dataset repo: `darshi1337/hermitlm-100k-generic`.

The model repo holds 9 files: `best_model.pt`, `config.json`, `tokenizer.json`, `model.onnx`,
`model.quant.onnx`, `decision_probe.pt`, `decision_tool_probe.pt`, plus `README.md` (model
card) and `.gitattributes`. The dataset repo includes `train.jsonl`, `eval.jsonl`, their
OpenAI-format variants, `tool_train.jsonl`, `tool_eval.jsonl`, `human_eval.jsonl`, and the
matching tokenizer.

Download a published repo:

```python
from huggingface_hub import snapshot_download
repo = snapshot_download("darshi1337/hermitlm-8M")
```

Upload a new build (maintainers only; both scripts read `HF_TOKEN` from `.env` or `hf auth login`):

```bash
python scripts/push_hf.py --repo darshi1337/hermitlm-8M --dry-run
python scripts/push_hf.py --repo darshi1337/hermitlm-8M
python scripts/push_hf_dataset.py --repo darshi1337/hermitlm-100k-generic --dry-run
python scripts/push_hf_dataset.py --repo darshi1337/hermitlm-100k-generic
```

Status on 2026-11-19: model and dataset repos are synced with the current 104k balanced corpus
and hash-pinned probes.

## Development

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1          # Windows; source .venv/bin/activate on Linux
pip install -e . -r requirements.txt pytest ruff
pip install onnx onnxscript onnxruntime
# Intel Arc / XPU (default on this machine):
pip install torch --index-url https://download.pytorch.org/whl/xpu
# NVIDIA CUDA instead: pip install torch --index-url https://download.pytorch.org/whl/cu121

pytest tests/          # 72 passed, 2 skipped
ruff check .           # clean
python -m hermitlm.tools.check
```

Create a `.env` with the documented keys (`HERMIT_CHECKPOINT`, `HERMIT_TOKENIZER`, `HERMIT_DEVICE`,
`HERMIT_TRAIN_DEVICE`, `HERMIT_TEMPERATURE`, `HERMIT_TOP_K`, `HERMIT_MAX_TOKENS`,
`HERMIT_PROBE`, `HERMIT_CONFUSED_THRESHOLD`, `HERMIT_TOOL_PROBE`, `HERMIT_TOOL_THRESHOLD`,
`HERMIT_BEST_OF`, `HERMIT_CLARIFY`, `HERMIT_CLARIFY_THRESHOLD`, `HERMIT_PERSONA_PROMPT`,
`WOLFRAM_APP_ID`, …). Never commit `.env`.

## License

MIT
