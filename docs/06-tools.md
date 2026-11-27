# Tools

`hermitlm/tools/router.py` is the single entry point. The model can emit function-call tags
(`<tool>search(query)</tool>`, `<tool>math(query)</tool>`, `<tool>faq(query)</tool>`). Otherwise
keyword heuristics route to the FAQ responder, Wolfram (`math.py`), or DuckDuckGo search (`web.py`).
Both the FastAPI backend and the eval harness use it. `POST /chat` returns which tool fired.

`route_with_intent()` also attaches the probe decision-model verdict. It escalates
out-of-scope messages when `p_yes` reaches the **fitted** `escalation_threshold` stored in the
probe artifact (0.4248 in the shipped probe) and no other tool fired; legacy probes without the
yes head fall back to `p_confused`.
