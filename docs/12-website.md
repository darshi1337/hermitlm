# Website

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

The UI has a light/dark theme toggle (OS preference by default), a clear-chat button, and a
Settings panel with temperature, top-K, persona mode, voice output and a stop button. Voice input
uses the browser speech-recognition API when available; replies can be read aloud with speech
synthesis. `Enter` sends and `Esc` stops generation.

Every reply also shows a probe decision-model intent chip (`web/intent.js` + `web/probe.json`).
Classification costs a single forward pass over the prompt. The probe ships a label-free `L0`
batch prior at $ \Large \lambda = 0.5 $, applied identically in the API and browser; the classifier reports
`level: "L0"` when it is active. A high `p_yes` from the calibrated yes/no head links to a web
search.
The chip shows coarse and fine intent with confidence. Low confidence asks a clarifying question
with quick-reply chips instead of guessing. Settings shows per-intent session counts.

`.github/workflows/deploy.yml` exports the model and publishes
`web/` to GitHub Pages on every push to `main`.
