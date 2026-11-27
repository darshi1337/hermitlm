# API

```bash
uvicorn hermitlm.runtime.api:app --host 0.0.0.0 --port 8000
# or: docker build -t hermitlm . && docker run -p 8000:8000 hermitlm
```

- `GET /health`: status, checkpoint, tokenizer, device, probe.
- `GET /stats/intents`: totals by intent label and tool, gate coverage, confused rate and
  abstention rate from the logged clarify/escalation decisions.
- `POST /chat`: `{user_id, message, temperature, top_k, max_tokens}` → `{response, tool, intent, persona}`. `intent` holds the coarse label, the confidence, `p_confused` and `p_yes`. High `p_yes` escalates to web search.
- `POST /chat/stream`: same request, server-sent events stream.

```json
POST /chat {"user_id": "demo", "message": "hello crab"}
→ {"response": "oh. it's you...", "tool": null,
   "intent": {"label": "a", "name": "social", "confidence": 1.0, "p_confused": 0.0067,
              "p_yes": 0.0021, "level": "L0", "rotations": 1},
   "persona": null}
```
