import os
import random

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from hermitlm.runtime.inference import HermitInference
from hermitlm.settings import (
    BEST_OF,
    CHECKPOINT_PATH,
    CLARIFY,
    CLARIFY_THRESHOLD,
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_K,
    DEVICE,
    PERSONA_PROMPT,
    TOKENIZER_PATH,
)
from hermitlm.tools.db import init_db, insert_conversation, insert_intent, intent_stats
from hermitlm.tools.decision_model import COARSE_NAMES, INTENT_PERSONA
from hermitlm.tools.memory import extract_fact, get_memory, init_memory_db, save_memory
from hermitlm.tools.router import route_with_intent

app = FastAPI(title="HermitLM API")

MATH_INTROS = [
    "aah math! i hate it. still here is the answer from wolfram",
    "ugh, numbers. i'd rather count bubbles, but wolfram says",
    "math detected. claws confused. wolfram to the rescue",
    "i dislike math almost as much as dry sand. here's wolfram's answer",
    "fine, fine. i asked wolfram so you don't have to",
    "my shell can't do algebra. wolfram can, apparently",
    "math! *shudders*. here, take wolfram's answer",
    "i hate sums. but here's what wolfram dug up",
]


init_db()
init_memory_db()

engine = HermitInference(
    CHECKPOINT_PATH,
    TOKENIZER_PATH,
    device=DEVICE
)


class ChatRequest(BaseModel):
    user_id: str
    message: str
    temperature: float = Field(DEFAULT_TEMPERATURE, ge=0.1, le=2.0)
    top_k: int = Field(DEFAULT_TOP_K, ge=0, le=200)
    max_tokens: int = Field(DEFAULT_MAX_TOKENS, ge=1, le=256)


class IntentResponse(BaseModel):
    label: str
    name: str
    confidence: float
    p_confused: float
    level: str = "raw"
    rotations: int = 1


class ChatResponse(BaseModel):
    response: str
    tool: str | None = None
    intent: IntentResponse | None = None
    persona: str | None = None


@app.get("/")
def home():
    return {"message": "HermitLM API is running", "ui": "/ui/"}


@app.get("/health")
def health():
    from hermitlm.settings import PROBE_PATH

    return {
        "status": "ok",
        "checkpoint": CHECKPOINT_PATH,
        "tokenizer": TOKENIZER_PATH,
        "device": DEVICE,
        "probe": PROBE_PATH if os.path.exists(PROBE_PATH) else None,
    }


@app.get("/stats/intents")
def stats_intents():
    return intent_stats()


def _respond(user_input, req):
    """Shared response pipeline for /chat and /chat/stream."""
    q = user_input.lower()

    _tool, tool_result, intent = route_with_intent(user_input)
    response = tool_result
    if response is not None and _tool in ("math", "wolfram"):
        response = f"{random.choice(MATH_INTROS)}: {response}"
    persona = INTENT_PERSONA.get((intent or {}).get("label", ""), "") or None

    fact = extract_fact(user_input)
    if response is None and fact:
        key, value = fact
        save_memory(req.user_id, key, value)
        response = {
            "name": (
                f"Understood. I will remember that your name is {value}. "
                "Also, name is Hermit."
            ),
            "likes": f"I will remember that you like {value}.",
            "location": f"Noted. You are from {value}.",
        }[key]

    if response is None:
        if "what is my name" in q or "who am i" in q:
            name = get_memory(req.user_id, "name")
            if name:
                response = f"Your name is {name}."
        elif "what do i like" in q:
            like = get_memory(req.user_id, "likes")
            if like:
                response = f"You told me you like {like}."
        elif "where am i from" in q:
            place = get_memory(req.user_id, "location")
            if place:
                response = f"You told me you're from {place}."

    if response is None:
        memory_parts = []

        name = get_memory(req.user_id, "name")
        likes = get_memory(req.user_id, "likes")
        location = get_memory(req.user_id, "location")

        if name:
            memory_parts.append(f"user name is {name}")
        if likes:
            memory_parts.append(f"user likes {likes}")
        if location:
            memory_parts.append(f"user is from {location}")

        memory_context = ". ".join(memory_parts) + ". " if memory_parts else ""

        if (
            _tool is None
            and CLARIFY
            and intent is not None
            and intent["confidence"] < CLARIFY_THRESHOLD
            and intent["p_confused"] < 0.9
            and len(intent.get("top2", [])) == 2
        ):
            (first, _), (second, _) = intent["top2"]
            response = (
                "Quick check before I scuttle ahead: is this about "
                f"{COARSE_NAMES[first]} or {COARSE_NAMES[second]}?"
            )

        prompt = memory_context + user_input

        if response is None:
            if persona and PERSONA_PROMPT:
                prompt = f"[{persona} crab] " + prompt

            response, _scores = engine.best_of_n(
                prompt,
                n=BEST_OF,
                temperature=req.temperature,
                top_k=req.top_k,
                max_tokens=req.max_tokens,
            )

    return response, _tool, intent, persona


def _log_turn(req, user_input, response, tool, intent):
    insert_conversation(
        user_id=req.user_id,
        username="api",
        user_input=user_input,
        bot_response=response,
        channel_id="api",
    )
    insert_intent(user_input, intent, tool)


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    user_input = req.message.strip()
    if not user_input:
        raise HTTPException(status_code=400, detail="message cannot be empty")

    response, tool, intent, persona = _respond(user_input, req)
    _log_turn(req, user_input, response, tool, intent)

    return ChatResponse(
        response=response,
        tool=tool,
        intent=IntentResponse(
            label=intent["label"],
            name=intent["name"],
            confidence=intent["confidence"],
            p_confused=intent["p_confused"],
            level=intent.get("level", "raw"),
            rotations=intent.get("rotations", 1),
        ) if intent else None,
        persona=persona,
    )


@app.post("/chat/stream")
def chat_stream(req: ChatRequest):
    from fastapi.responses import StreamingResponse

    user_input = req.message.strip()
    if not user_input:
        raise HTTPException(status_code=400, detail="message cannot be empty")

    response, tool, intent, _persona = _respond(user_input, req)
    _log_turn(req, user_input, response, tool, intent)

    def _gen():
        if tool:
            # Tool results are often multi-line. Every line must get its own
            # `data:` prefix or the SSE framing is corrupted.
            for line in response.splitlines() or [""]:
                yield f"data: {line}\n\n"
        else:
            # Naive word streaming (true token streaming needs generate()
            # to yield; the local web UI already streams ONNX tokens).
            for w in response.split():
                yield f"data: {w} \n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(_gen(), media_type="text/event-stream; charset=utf-8")


_STATIC_DIR = os.path.join(os.path.dirname(__file__), "..", "static")
app.mount("/ui", StaticFiles(directory=_STATIC_DIR, html=True), name="ui")
