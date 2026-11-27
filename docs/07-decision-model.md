# Decision model

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
calibrated-minus-uncalibrated table. Head files carry schema version 3, SHA-256 hashes of the
checkpoint, tokenizer and prompt template, plus dtype, chosen layer, fit-time thresholds and
per-head solver/alpha/fold/temperature refit metadata. Loading a head after a retrain or prompt
change raises an error instead of scoring with stale weights.

### Decision metrics with 95% bootstrap intervals

From `eval_out/decision_order/report.json` (zero-shot logits, T fitted on a disjoint split):

| Metric | Uncalibrated (T=1) [95% CI] | Calibrated [95% CI] | Paired diff [95% CI] |
|---|---|---|---|
| Accuracy | 0.0900 [0.0500, 0.1300] | 0.0900 [0.0550, 0.1350] | 0.0000 [0.0000, 0.0000] |
| NLL | 4.6204 [4.3678, 4.8493] | 1.9633 [1.9526, 1.9727] | −2.6571 [−2.8888, −2.4303] |
| ECE | 0.8473 [0.8036, 0.8840] | 0.0863 [0.0460, 0.1266] | −0.7610 [−0.7705, −0.7494] |
| Macro F1 | 0.0236 [0.0136, 0.0329] | 0.0236 [0.0149, 0.0329] | 0.0000 [0.0000, 0.0000] |

`scripts/eval_decision.py` writes `items.npz` beside each report. Replay with
`--from-npz eval_out/decision_order/items.npz --bootstrap 1000 --seed 0` to recompute every
metric without another model pass; pass the same bootstrap and seed as the original run to get
identical intervals. Reports also include a paired **tool head vs keywords** row on
`data/tool_eval.jsonl` (tool head 0.7250, keywords 0.5875, diff +0.1375).

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
