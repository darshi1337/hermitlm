"""Unified tool router with function-call tag parsing.

Model can emit <tool>name(arg)</tool>; router also falls back to keyword heuristics.
"""
import re


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
