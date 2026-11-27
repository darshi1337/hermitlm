# Calibration toolkit

The decision stack includes label-free and supervised calibration pieces. The status of each
work item is tracked in [docs/calibration-improvements.md](calibration-improvements.md):

- `hermitlm/tools/prior.py`: batch prior correction (`L0`). A prior fitted from an unlabelled
  pool is applied as $ \Large \log p - \lambda \log \pi $. `scripts/fit_probe_prior.py` fits it into the
  probe artifact; `export_probe.py` copies it to `web/probe.json`, so the API and browser apply
  the same correction. `classify_intent()` reports `level` (`raw` or `L0`) plus `rotations`.
- `hermitlm/tools/heads.py` + `scripts/train_probe.py`: `L2` head selection over hidden layers,
  ridge/LDA solvers and regularization, chosen by stratified k-fold OOF NLL. Selected heads
  store standardization, temperature and OOF metrics in the artifact; `export_probe.py` folds
  standardization into JSON weights for the browser.
- `DecisionModel.predict_proba(..., rotations=k)`: option-order rotation with two evenly spaced
  passes by default for the zero-shot constrained-decoding path. `eval_decision.py` reports the
  order-flip rate for `raw` vs `L0` in the `order_flip` section.
- `scripts/train_yes_score_heads.py`: fits two more heads into the same pinned probe artifact.
  `head_yes` gives a calibrated `p_yes` for out-of-scope escalation; `head_score` predicts a
  context-aware reply-quality level whose expected value is $ \Large \sum_k k\,p_k $. Both are exported
  to `web/probe.json` and applied identically in the API and browser.
- `hermitlm/tools/calibration.py` + `scripts/fit_decision_thresholds.py`: split-conformal
  cutoffs for the clarify and escalation gates, fitted on held-out rows at a target error rate
  (0.1 by default). Both thresholds and their empirical coverage are stored in the probe artifact
  and exported to `web/probe.json`, so the API and browser read the same numbers.
- `python -m hermitlm.tools.check`: validates label tokens, batch consistency, hidden layers,
  prompt-template hashes, auxiliary heads and score-label tokenization before deploy.

Artifact metadata now also pins the prompt-template hash, dtype, chosen layer, fit-time thresholds
and refit details (solver, alpha, fold count, temperature), so a probe cannot be silently paired
with a changed prompt or precision. `scripts/upgrade_probe_metadata.py` adds these fields to older
artifacts without refitting the heads.

Evaluated on the hand-written human eval, the currently shipped last-layer probe still
generalizes better than OOF-selected ridge/LDA heads (0.62 coarse / 0.48 fine vs 0.62 / 0.35),
so the shipped artifact keeps that head. The label-free prior is active on the shipped probe;
the head-selection tooling remains available for future data improvements.
