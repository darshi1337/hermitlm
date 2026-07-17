import os
import re

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from hermitlm.tools.db import init_db, insert_conversation
from hermitlm.runtime.inference import HermitInference
from hermitlm.settings import (
    CHECKPOINT_PATH,
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_K,
    DEVICE,
    TOKENIZER_PATH,
)

from hermitlm.tools.memory import (
    save_memory,
    get_memory,
    init_memory_db
)

from hermitlm.tools.bot_faq import get_faq_response
from hermitlm.tools.math import ask_wolfram
from hermitlm.tools.web import web_search

app = FastAPI(title="HermitLM API")

MATH_KEYWORDS = [
    "solve",
    "integrate",
    "differentiate",
    "derivative",
    "integration",
    "equation",
    "factor",
    "simplify",
    "limit",
    "matrix",
    "sin",
    "cos",
    "tan",
]
MATH_EXPRESSION_RE = re.compile(r"\d\s*[+\-*/=]\s*\d")

WEB_KEYWORDS = [
    "latest",
    "news",
    "today",
    "current",
    "who is",
    "search",
    "explain",
]


def _keyword_hit(keywords, text):
    return any(re.search(rf"\b{re.escape(k)}\b", text) for k in keywords)

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


class ChatResponse(BaseModel):
    response: str


@app.get("/")
def home():
    return {"message": "HermitLM API is running", "ui": "/ui/"}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "checkpoint": CHECKPOINT_PATH,
        "tokenizer": TOKENIZER_PATH,
        "device": DEVICE,
    }


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    user_input = req.message.strip()

    if not user_input:
        raise HTTPException(
            status_code=400,
            detail="message cannot be empty"
        )

    q = user_input.lower()

    faq_response = get_faq_response(user_input)

    if faq_response:
        response = faq_response
    else:
        response = None

    is_math = _keyword_hit(MATH_KEYWORDS, q) or bool(MATH_EXPRESSION_RE.search(q))
    is_web_query = _keyword_hit(WEB_KEYWORDS, q)

    if response is None and is_math:
        wolfram_response = ask_wolfram(user_input)

        if wolfram_response:
            response = wolfram_response

    if response is None and is_web_query:

        web_result = web_search(user_input)

        if web_result:
            response = web_result

    if response is None and "my name is " in q:
        idx = q.find("my name is ")
        name = user_input[idx + len("my name is "):].strip()

        save_memory(req.user_id, "name", name)

        response = (
            f"Understood. I will remember that your name is {name}. Also, name is Hermit."
        )

    if response is None and "i like " in q:
        idx = q.find("i like ")
        like = user_input[idx + len("i like "):].strip()

        save_memory(req.user_id, "likes", like)

        response = (
            f"I will remember that you like {like}."
        )

    if response is None and "i am from " in q:
        idx = q.find("i am from ")
        place = user_input[idx + len("i am from "):].strip()

        save_memory(req.user_id, "location", place)

        response = (
            f"Noted. You are from {place}."
        )

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

        memory_context = ""

        if memory_parts:
            memory_context = ". ".join(memory_parts) + ". "

        prompt = memory_context + user_input

        response = engine.chat(
            prompt,
            temperature=req.temperature,
            top_k=req.top_k,
            max_tokens=req.max_tokens,
        )

    insert_conversation(
        user_id=req.user_id,
        username="api",
        user_input=user_input,
        bot_response=response,
        channel_id="api",
    )

    return ChatResponse(response=response)


_STATIC_DIR = os.path.join(os.path.dirname(__file__), "..", "static")
app.mount("/ui", StaticFiles(directory=_STATIC_DIR, html=True), name="ui")