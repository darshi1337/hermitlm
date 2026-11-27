"""Unified tool router with function-call tag parsing.

Model can emit <tool>name(arg)</tool>. The router also falls back to keyword heuristics.
`route_with_intent` also runs the probe decision model
(`hermitlm/tools/decision_model.py`): every reply carries a coarse intent
label, and high `p_yes` (out-of-scope) escalates to web search when no other
tool fired. Older probes fall back to `p_confused`.
"""
import os
import re

from hermitlm.tools.artifacts import ArtifactMismatchError


def _faq(q):
    from hermitlm.tools.bot_faq import get_faq_response
    return get_faq_response(q)

def _wolfram(q):
    from hermitlm.tools.math import ask_wolfram
    return ask_wolfram(q)

def _web(q):
    from hermitlm.tools.web import web_search
    return web_search(q)

TOOL_TAG_RE = re.compile(r"<tool>(\w+)\((.*?)\)</tool>", re.DOTALL)
MATH_RE = re.compile(r"\d\s*[+\-*/=]\s*\d")
MATH_KW = ["solve","integrate","differentiate","derivative","equation","factor","simplify","limit","matrix","sin","cos","tan"]
WEB_KW = ["latest","news","today","current","who is","search","explain"]

def _hit(kws, text):
    return any(re.search(rf"\b{re.escape(k)}\b", text) for k in kws)

def parse_tool_tag(text):
    m = TOOL_TAG_RE.search(text or "")
    if not m:
        return None
    return m.group(1).lower(), m.group(2).strip()


def keyword_tool(user_input):
    """Return the 4-way keyword/tag decision without any network dispatch.

    Mirrors the decision order of `route()` (tag, FAQ, math, web) but maps the
    no-tool outcome to `chat`. Evaluators can compare it with the tool head
    without triggering Wolfram, DuckDuckGo, or any other side effect.
    """
    q = (user_input or "").lower()
    tag = parse_tool_tag(user_input)
    if tag:
        name = tag[0]
        if name in ("math", "wolfram"):
            return "math"
        if name in ("search", "web"):
            return "web"
        if name == "faq":
            return "faq"
    if _faq(user_input):
        return "faq"
    if _hit(MATH_KW, q) or MATH_RE.search(q):
        return "math"
    if _hit(WEB_KW, q):
        return "web"
    return "chat"


def route(user_input):
    """Return (tool_name, result) or (None, None) if no tool applies."""
    q = user_input.lower()
    tag = parse_tool_tag(user_input)
    if tag:
        name, arg = tag
        if name in ("math", "wolfram"):
            return name, _wolfram(arg or user_input)
        if name in ("search", "web"):
            return name, _web(arg or user_input)
        if name == "faq":
            return name, _faq(arg or user_input)
    faq = _faq(user_input)
    if faq:
        return "faq", faq
    if _hit(MATH_KW, q) or MATH_RE.search(q):
        r = _wolfram(user_input)
        if r:
            return "math", r
    if _hit(WEB_KW, q):
        r = _web(user_input)
        if r:
            return "web", r
    return None, None


_probe = None
_probe_failed = False


def _get_probe():
    """Load the intent probe on first use. Return None (and never raise) if unavailable."""
    global _probe, _probe_failed
    if _probe is not None or _probe_failed:
        return _probe
    try:
        from hermitlm.settings import (
            CHECKPOINT_PATH,
            DEVICE,
            PROBE_PATH,
            TOKENIZER_PATH,
            TOOL_PROBE_PATH,
        )
        from hermitlm.tools.decision_model import ProbeDecisionModel

        if not (os.path.exists(CHECKPOINT_PATH) and os.path.exists(TOKENIZER_PATH) and os.path.exists(PROBE_PATH)):
            _probe_failed = True
            return None
        _probe = ProbeDecisionModel(CHECKPOINT_PATH, TOKENIZER_PATH, PROBE_PATH, device=DEVICE)
        if os.path.exists(TOOL_PROBE_PATH):
            _probe.load_tool_heads(TOOL_PROBE_PATH)
        return _probe
    except ArtifactMismatchError:
        # Stale/mismatched artifacts must fail loudly instead of silently
        # dropping intent routing.
        raise
    except Exception:  # noqa: BLE001 - a missing probe must not break routing
        _probe_failed = True
        return None


def get_probe():
    """Return the shared probe instance (used to attach it to the chat engine)."""
    return _get_probe()


def classify_intent(user_input):
    """Return intent dict (label, name, confidence, p_confused, p_yes, proba,
    fine, top2, level, rotations) or None."""
    probe = _get_probe()
    if probe is None:
        return None
    try:
        from hermitlm.tools.decision_model import (
            COARSE_NAMES,
            LEGACY_CLARIFY_THRESHOLD,
            LEGACY_ESCALATION_THRESHOLD,
            debias_level,
        )

        label, proba, p_confused = probe.predict(user_input)
        yes = probe.predict_yes(user_input) if hasattr(probe, "predict_yes") else None
        # Older artifacts have no yes head: the legacy confused probability is
        # the fallback so routing behaviour stays unchanged.
        p_yes = yes[0] if yes is not None else p_confused
        ranked = sorted(proba.items(), key=lambda kv: kv[1], reverse=True)[:2]
        priors = getattr(probe, "priors", {}) or {}
        prior_lambda = float(getattr(probe, "prior_lambda", 0.0) or 0.0)
        result = {
            "label": label,
            "name": COARSE_NAMES[label],
            "confidence": round(proba[label], 4),
            "p_confused": round(p_confused, 4),
            "p_yes": round(p_yes, 4),
            "proba": {k: round(v, 4) for k, v in proba.items()},
            "top2": [[letter, round(p, 4)] for letter, p in ranked],
            "fine_label": None,
            "fine_name": None,
            # Debiasing provenance, mirrored by web/intent.js:
            # "raw" = head output, "L0" = fitted batch prior applied.
            "level": debias_level(priors, prior_lambda),
            "rotations": getattr(probe, "rotations", 1),
            # Gates read the thresholds fitted into the artifact. Legacy probes
            # keep the old fixed boundaries until they are re-calibrated.
            "clarify_threshold": float(
                getattr(probe, "clarify_threshold", LEGACY_CLARIFY_THRESHOLD)
            ),
            "escalation_threshold": float(
                getattr(probe, "escalation_threshold", LEGACY_ESCALATION_THRESHOLD)
            ),
        }
        fine = probe.predict_fine(user_input)
        if fine is not None:
            fine_label, _ = fine
            result["fine_label"] = fine_label
            result["fine_name"] = fine_label
        return result
    except Exception:  # noqa: BLE001 - classification is best-effort
        return None


def classify_tool(user_input):
    """Return {label, confidence} from the tool-routing probe, or None."""
    probe = _get_probe()
    if probe is None:
        return None
    try:
        predicted = probe.predict_tool(user_input)
        if predicted is None:
            return None
        label, proba = predicted
        return {"label": label, "confidence": round(proba[label], 4)}
    except Exception:  # noqa: BLE001 - classification is best-effort
        return None


def _dispatch_probe(label, user_input):
    """Run the tool the probe picked. Return (tool, result) or (None, None)."""
    try:
        if label == "math":
            result = _wolfram(user_input)
            return ("math", result) if result else (None, None)
        if label == "web":
            result = _web(user_input)
            return ("web", result) if result else (None, None)
        if label == "faq":
            result = _faq(user_input)
            return ("faq", result) if result else (None, None)
        return None, None
    except Exception:  # noqa: BLE001 - probe routing must not break chat
        return None, None


def route_with_intent(user_input, confused_threshold=None, tool_threshold=None):
    """Return (tool_name, result, intent) where intent is classify_intent() or None.

    Order: tag/keyword routing first (high precision), then the tool probe on
    the residue, then confused escalation to web search.
    """
    from hermitlm.settings import TOOL_THRESHOLD
    from hermitlm.tools.decision_model import LEGACY_ESCALATION_THRESHOLD

    tool, result = route(user_input)
    intent = classify_intent(user_input)
    if tool is None:
        decision = classify_tool(user_input)
        threshold = TOOL_THRESHOLD if tool_threshold is None else tool_threshold
        if decision is not None and decision["confidence"] >= threshold:
            probe_tool, probe_result = _dispatch_probe(decision["label"], user_input)
            if probe_tool:
                return probe_tool, probe_result, intent
    if tool is None and intent is not None:
        fitted = float(
            getattr(_get_probe(), "escalation_threshold", LEGACY_ESCALATION_THRESHOLD)
        )
        threshold = fitted if confused_threshold is None else confused_threshold
        if intent.get("p_yes", intent["p_confused"]) >= threshold:
            intent["escalated"] = True
            try:
                web_result = _web(user_input)
            except Exception:  # noqa: BLE001 - escalation must not break chat
                web_result = None
            if web_result:
                return "web", web_result, intent
    return tool, result, intent
