# Chapter 9 — Tools and Routing

> *When the crab does not know a fact, it asks a helper. The router decides which helper, if any.*

---

## The 5-year-old analogy

The crab has three friends:

| Friend | What it answers |
|---|---|
| **FAQ** | Questions about the crab itself, from a 20-row table |
| **Wolfram math** | Calculations such as `integrate x^2` |
| **Web search** | Current facts and news |

The router checks in a fixed order: an explicit tool tag first, then keywords, then a tiny learned probe, then an out-of-scope escalation.

```mermaid
flowchart TD
    A["user message"] --> B{"tool tag?"}
    B -->|"yes"| C["dispatch named tool"]
    B -->|"no"| D{"FAQ match?"}
    D -->|"yes"| E["faq"]
    D -->|"no"| F{"math keyword or pattern?"}
    F -->|"yes"| G["math → Wolfram"]
    F -->|"no"| H{"web keyword?"}
    H -->|"yes"| I["web search"]
    H -->|"no"| J["tool probe (4-way)"]
    J -->|"confidence ≥ 0.5"| C
    J -->|"low"| K{"p_yes ≥ 0.4248?"}
    K -->|"yes"| I
    K -->|"no"| L["chat with the model"]
```

---

## The router code

`hermitlm/tools/router.py:27-30` defines the patterns:

```python
27: TOOL_TAG_RE = re.compile(r"<tool>(\w+)\((.*?)\)</tool>", re.DOTALL)
28: MATH_RE = re.compile(r"\d\s*[+\-*/=]\s*\d")
29: MATH_KW = ["solve","integrate","differentiate","derivative","equation","factor","simplify","limit","matrix","sin","cos","tan"]
30: WEB_KW = ["latest","news","today","current","who is","search","explain"]
```

`parse_tool_tag()` (`router.py:35-39`) returns a lowercased name and its argument:

```python
35: def parse_tool_tag(text):
36:     m = TOOL_TAG_RE.search(text or "")
37:     if not m:
38:         return None
39:     return m.group(1).lower(), m.group(2).strip()
```

Accepted names (`router.py:73-79`):

| Tag name | Tool |
|---|---|
| `math`, `wolfram` | Wolfram alpha math |
| `search`, `web` | DuckDuckGo-style web search |
| `faq` | Built-in FAQ table |

Example: `<tool>math(integrate x^2)</tool>` calls the math tool with the argument `integrate x^2`.

> The router looks for tags in the **user message** (`router.py:71`). It does not rescan generated text for tags, even though the module docstring suggests the model can emit them.

---

## Keyword routing

`route()` (`router.py:68-91`) runs a cascade:

```python
68: def route(user_input):
69:     """Return (tool_name, result) or (None, None) if no tool applies."""
70:     q = user_input.lower()
71:     tag = parse_tool_tag(user_input)
72:     if tag:
73:         name, arg = tag
74:         if name in ("math", "wolfram"):
75:             return name, _wolfram(arg or user_input)
...
80:     faq = _faq(user_input)
81:     if faq:
82:         return "faq", faq
83:     if _hit(MATH_KW, q) or MATH_RE.search(q):
84:         r = _wolfram(user_input)
85:         if r:
86:             return "math", r
87:     if _hit(WEB_KW, q):
88:         r = _web(user_input)
89:         if r:
90:             return "web", r
91:     return None, None
```

`_hit()` (`router.py:32-33`) uses word-boundary regular expressions, so `sin` matches the word "sin" but not "single".

### Worked routing examples

| Message | First match | Tool |
|---|---|---|
| `hello crab` | FAQ table has a greeting entry | `faq` |
| `integrate x^2` | `MATH_KW` contains `integrate` | `math` |
| `what is 2+7` | `MATH_RE` matches `2+7` | `math` |
| `latest AI news` | `WEB_KW` contains `latest` and `news` | `web` |
| `who is Elon Musk?` | `WEB_KW` contains `who is` | `web` |
| `the water feels calm today` | no keyword | chat or probe |

`docs/09-tool-routing.md:6` reports that keyword routing is **93% precise when it fires**. The tool probe handles the residue.

---

## The tool probe

`scripts/train_tool_probe.py:30` defines four classes:

```python
30: TOOL_CLASSES = ["math", "web", "faq", "chat"]
```

Training data (`data/tool_train.jsonl`, 5,038 rows) is built by `scripts/build_tool_data.py:374-405`:

```
3 × 1,200 templates (math, web, faq) = 3,600
+ 1,200 chat negatives from train.jsonl
+ 160 natural train queries
+ 50 natural web queries
+ 28 hard negatives (chat/FAQ with web trigger words)
= 5,038 rows
```

`data/tool_eval.jsonl` has 80 rows, exactly 20 per class.

The head is a linear classifier on the frozen final hidden state (chapter 10). The shipped artifact stores:

| Property | Value |
|---|---|
| `head.weight` | shape `(4, 320)` |
| `head.bias` | shape `(4,)` |
| `temperature` | 0.6785627417882039 |
| `eval_accuracy` | 0.7250 |
| `classes` | `["math", "web", "faq", "chat"]` |

Per-class accuracy on the 80-row eval set: math 0.80 (16/20), web 0.65 (13/20), faq 0.60 (12/20), chat 0.85 (17/20). Total `58/80 = 0.725`.

---

## The full cascade

`router.py:221-252`:

```python
221: def route_with_intent(user_input, confused_threshold=None, tool_threshold=None):
...
230:     tool, result = route(user_input)
231:     intent = classify_intent(user_input)
232:     if tool is None:
233:         decision = classify_tool(user_input)
234:         threshold = TOOL_THRESHOLD if tool_threshold is None else tool_threshold
235:         if decision is not None and decision["confidence"] >= threshold:
236:             probe_tool, probe_result = _dispatch_probe(decision["label"], user_input)
237:             if probe_tool:
238:                 return probe_tool, probe_result, intent
239:     if tool is None and intent is not None:
240:         fitted = float(
241:             getattr(_get_probe(), "escalation_threshold", LEGACY_ESCALATION_THRESHOLD)
242:         )
243:         threshold = fitted if confused_threshold is None else confused_threshold
244:         if intent.get("p_yes", intent["p_confused"]) >= threshold:
245:             intent["escalated"] = True
246:             try:
247:                 web_result = _web(user_input)
248:             except Exception:  # noqa: BLE001 - escalation must not break chat
249:                 web_result = None
250:             if web_result:
251:                 return "web", web_result, intent
252:     return tool, result, intent
```

Settings (`hermitlm/settings.py:20-33`):

| Threshold | Value | Meaning |
|---|---:|---|
| `escalation_threshold` (in probe artifact) | 0.4248 | Fitted P(out-of-scope) cutoff for web search |
| `clarify_threshold` (in probe artifact) | 0.9884 | Below this confidence, ask a clarifying question |
| `TOOL_THRESHOLD` | 0.5 | Minimum probe confidence to dispatch a tool |
| `BEST_OF` | 3 | Candidates sampled before score-head reranking |

Both decision thresholds are fitted split-conformally by
`scripts/fit_decision_thresholds.py` (`--target-error 0.1`) and exported to
`web/probe.json`; the legacy constants in `settings.py` are no longer read by
the runtime gates.

### Worked escalation example

For a message like `quantum entanglement photosynthesis`:

1. No tag, no FAQ hit, no keyword.
2. Tool probe returns `chat` with confidence 0.42, below 0.5, so no dispatch.
3. The yes head reports `p_yes = 0.97`, above the fitted 0.4248 cutoff.
4. Web search runs and its result becomes the reply.

With `p_yes = 0.10`, step 4 is skipped and the model answers normally.

---

## Clarifying questions and personas

The API reads both thresholds from the intent dict. It asks for clarification when no tool fired,
intent confidence is below 0.9884, `p_yes < 0.4248`, and two top intents are available:

```python
165:        clarify_threshold = float(
166:            (intent or {}).get("clarify_threshold", LEGACY_CLARIFY_THRESHOLD)
167:        )
168:        escalation_threshold = float(
169:            (intent or {}).get("escalation_threshold", LEGACY_ESCALATION_THRESHOLD)
170:        )
171:        if (
172:            _tool is None
173:            and CLARIFY
174:            and intent is not None
175:            and intent["confidence"] < clarify_threshold
176:            and intent.get("p_yes", intent["p_confused"]) < escalation_threshold
177:            and len(intent.get("top2", [])) == 2
178:        ):
```

Example: top intents `tank` and `sense` produce:

```text
Quick check before I scuttle ahead: is this about tank or sense?
```

The coarse intent also selects a persona (`hermitlm/tools/decision_model.py:78-87`):

| Coarse intent | Persona |
|---|---|
| `a` social | none |
| `b` needs | hungry |
| `c` tank | none |
| `d` sense | guard |
| `e` body | sleepy |
| `f` condition | sleepy |
| `g` mind | none |

The prefix `[<persona> crab] ` is off by default (`api.py:188-190`). The shipped checkpoint has no
`[mode` rows in `data/train.jsonl`. Set `HERMIT_PERSONA_PROMPT=1` only for a checkpoint trained with
`generate_data.add_persona_variants()`.

---

## The helper tools

| Tool | File | What it does |
|---|---|---|
| FAQ | `hermitlm/tools/bot_faq.py` | 20 hand-written question/answer pairs, normalized by lowercase and punctuation removal |
| Math | `hermitlm/tools/math.py` | Wolfram Alpha query with `WOLFRAM_APP_ID`; API adds a random crab intro |
| Web | `hermitlm/tools/web.py` | DuckDuckGo HTML endpoint, returns titles and snippets |
| Memory | `hermitlm/tools/memory.py` | Three SQLite keys: `name`, `likes`, `location` |
| Artifacts | `hermitlm/tools/artifacts.py` | SHA-256 pinning of checkpoint, tokenizer and prompt template |

The browser has its own local math fallback (`web/math.js:204-229`) that handles arithmetic and polynomial calculus without a network call. `∫cx^e dx = c/(e+1)·x^(e+1)` for `e ≠ −1` (`math.js:190-197`); `d/dx cx^e = ce·x^(e−1)` (`math.js:199-201`).

---

## Math and calculations

### Tool probe accuracy on the eval set

$$ \Large \text{math accuracy} = \frac{16}{20} = 0.80 $$

$$ \Large \text{web accuracy} = \frac{13}{20} = 0.65 $$

$$ \Large \text{faq accuracy} = \frac{12}{20} = 0.60 $$

$$ \Large \text{chat accuracy} = \frac{17}{20} = 0.85 $$

$$ \Large \text{overall accuracy} = \frac{16+13+12+17}{80} = \frac{58}{80} = 0.725 $$

### Probe artifact size

$$ \Large \text{head parameters} = 4 \times 320 + 4 = 1{,}284 $$

$$ \Large \text{weight and bias bytes} = 1{,}284 \times 4 = 5{,}136 $$

$$ \Large \text{probe file size} = 7{,}685 \text{ bytes} $$

### Routing order as a short-circuit

For `n` checks with firing probabilities `p₁ … pₙ`, the probability that check `k` decides is:

$$ \Large P(\text{check } k \text{ decides}) = \left(\prod_{i=1}^{k-1}(1-p_i)\right) p_k $$

High-precision checks go first so they handle most cases before the slower probe runs.

---

## Code you should remember

| File and lines | What it does |
|---|---|
| `hermitlm/tools/router.py:27-39` | Tool tag and keyword patterns |
| `hermitlm/tools/router.py:68-91` | First four routing checks |
| `hermitlm/tools/router.py:221-252` | Probe dispatch and confusion escalation |
| `hermitlm/settings.py:20-33` | Threshold values |
| `hermitlm/runtime/api.py:165-200` | Clarify gate, persona prompt, best-of-N call |
| `scripts/build_tool_data.py:374-405` | Tool data construction |
| `web/math.js:190-201` | Local polynomial integration and differentiation |

---

**Next:** [Chapter 10 — The decision model](10_decision_model.md)
