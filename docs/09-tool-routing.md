# Tool routing

`scripts/build_tool_data.py` writes synthetic tool data (templates plus hand-authored natural
queries) to `data/tool_train.jsonl` / `data/tool_eval.jsonl`. `scripts/train_tool_probe.py` trains
a 4-way head (math/web/faq/chat) into `checkpoints/decision_tool_probe.pt`. Routing order is
tag, then keywords (93% precise when they fire), then the probe on the residue, then out-of-scope
escalation using the calibrated `p_yes` from `head_yes` and the fitted `escalation_threshold` in
the probe artifact (0.4248 in the shipped probe). Older probes without that head fall back to
`p_confused`. Training now includes 50 extra natural web queries and 28 hard negatives (chat/FAQ
messages containing web trigger words). Scores on the 80 hand-authored eval queries after the
latest retrain: probe **0.7250** (math 0.80, web 0.65, faq 0.60, chat 0.85), up from 0.6125 —
the web class went from 0.15 to 0.65.
