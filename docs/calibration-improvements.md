# Calibration improvements — status

This table tracks the calibration and hygiene work items referenced by the GitHub issues.

| Row | Item | Status | Implementation |
|---|---|---|---|
| 1 | Label-free prior correction (`L0`) | implemented | `hermitlm/tools/prior.py`, `scripts/fit_probe_prior.py` |
| 2 | Two-pass option rotation | implemented | `hermitlm/tools/decision_model.py` (`option_assignment`, `rotation_shifts`) |
| 3 | Bootstrap evaluation and npz replay | implemented | `hermitlm/tools/metrics.py`, `scripts/eval_decision.py --from-npz` |
| 4 | Bounded temperature search | implemented | `hermitlm/tools/calibration.py` |
| 5 | Artifact layer/threshold/refit metadata (schema 3) | implemented | `hermitlm/tools/artifacts.py`, `scripts/upgrade_probe_metadata.py` |
| 6 | `check` CLI deploy diagnostic | **implemented** | `hermitlm/tools/check.py` — validates labels, batch consistency, hidden layers, prompt hashes, aux heads |
| 7 | Layer search | **implemented** | `hermitlm/tools/heads.py:155-191` — `select_head()` with stratified k-fold OOF NLL over layers, ridge α grid and LDA |
| 8 | Split-conformal clarify/escalation thresholds | implemented | `hermitlm/tools/calibration.py`, `scripts/fit_decision_thresholds.py` |
| 9 | Logging, prompt escaping and decision tests | **implemented** | stdlib loggers with `NullHandler`, hash-pin warnings, delimiter escaping, decision-model tests |

Rows 6 and 7 were previously listed as partial; both are implemented in the current tree.
