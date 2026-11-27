# Features
- Runs fully client-side in the browser (ONNX + onnxruntime-web, no backend server)
- Quantized web model (`model.quant.onnx`, ~9.8 MB vs ~38.7 MB fp32) with automatic fp32 fallback when `model.onnx` is present
- Tool-calling router: FAQ, Wolfram math, and web search, via `<tool>name(query)</tool>` tags plus keyword fallback
- Probe decision model: single-pass intent classification into 7 coarse intents, plus a calibrated `p_yes` out-of-scope score. High `p_yes` escalates to web search. The browser UI shows an intent chip on each reply.
- Fine-grained 21-way intent head and tool-routing probe (math/web/faq/chat) on the same frozen features.
- Calibrated heads: bounded temperature search with Platt-smoothed binary targets, bootstrap CIs, and hash-checked head files.
- Best-of-N reply rerank by the calibrated score head's `expected` value (heuristic persona reward as fallback), confidence-gated clarifying questions, persona auto-router, and intent analytics (`GET /stats/intents`).
- Streaming API (`POST /chat/stream`, SSE) alongside `POST /chat`; math replies use a randomly
  chosen crab-flavoured intro before the Wolfram/local answer
- Modular architecture
- Automated tests and ruff gates, plus a `python -m hermitlm.tools.check` deploy diagnostic
- RLHF (PPO) and DPO fine-tuning stages
- Eval dashboards: perplexity + generation-based persona reward + tool-trigger rate, plus decision-model accuracy/calibration
