# Reply quality and analytics

`POST /chat` now does more than generate one reply:

- Best-of-N (`engine.best_of_n`, `HERMIT_BEST_OF`, default 3). The API samples N candidates and
keeps the best by the score head's `expected` value; it falls back to the heuristic persona
reward when the artifact has no score head. The chosen candidate's expected score is written
to `intent_log.expected_score`. Lower the value to 1 for less latency.
- Clarifying gate (`HERMIT_CLARIFY`, default on). The confidence cutoff is fitted split-conformally
and stored in the probe artifact (0.9884 in the shipped probe). When no tool fires and intent
confidence lands below that cutoff without out-of-scope evidence, the API asks which of the top
two intents the user means instead of guessing.
- Persona auto-router. Coarse intent maps to UI persona mode (needs to hungry, sense to guard,
body and condition to sleepy). The API returns the persona and prefixes the prompt with
`[<persona> crab]` (for example `[hungry crab]`), matching the prefixed variants that
`generate_data.add_persona_variants()` writes into training. Set `HERMIT_PERSONA_PROMPT=0` to disable it.
- Analytics. Every reply logs its verdict to `intent_log` in `data/hermit.db`.
`GET /stats/intents` returns totals by label and tool plus the confused rate.

```bash
python scripts/train_probe.py --per-class 1500 --per-fine 400 --out checkpoints/decision_probe.pt
python scripts/fit_probe_prior.py --max-n 4000
python scripts/train_yes_score_heads.py --score-n 6000 --device xpu
python scripts/train_tool_probe.py --out checkpoints/decision_tool_probe.pt
python scripts/eval_decision.py --method probe --max-n 5000 --cal-n 1000 --bootstrap 1000 --out-dir eval_out/decision_probe
python scripts/eval_decision.py --eval data/human_eval.jsonl --method probe --max-n 126 --cal-n 60 --out-dir eval_out/human
python scripts/export_probe.py --output web/probe.json   # also runs in deploy CI
python scripts/eval_aux_heads.py --out-dir eval_out/aux_heads
python -m hermitlm.tools.check                           # pre-deploy diagnostics
```

`HermitModel.classify()` (`web/model.js`) + `web/intent.js` run the same probe in-browser off the
`last_hidden` ONNX output. This costs one extra inference step and no extra download.
