import sqlite3

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from hermitlm.db import init_db, insert_conversation
from hermitlm.inference import HermitInference
from hermitlm.settings import (
    CHECKPOINT_PATH,
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_K,
    DEVICE,
    TOKENIZER_PATH,
)

from hermitlm.memory import (
    save_memory,
    get_memory,
    get_all_memories,
    init_memory_db
)

from hermitlm.math import ask_wolfram

app = FastAPI(title="HermitLM API")

init_db()
init_memory_db()

engine = HermitInference(
    CHECKPOINT_PATH,
    TOKENIZER_PATH,
    device=DEVICE
)


def get_user_history(user_id, limit=3):
    conn = sqlite3.connect("data/hermit.db")
    cursor = conn.cursor()

    rows = cursor.execute("""
        SELECT user_input, bot_response
        FROM conversations
        WHERE user_id = ?
        ORDER BY id DESC
        LIMIT ?
    """, (str(user_id), limit)).fetchall()

    conn.close()

    return rows[::-1]


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
    return {"message": "HermitLM API is running"}


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

    response = None

    math_keywords = [
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
        "+",
        "-",
        "*",
        "/",
        "=",
    ]

    is_math = any(k in q for k in math_keywords)

    if is_math:
        wolfram_response = ask_wolfram(user_input)

        print("WOLFRAM RESPONSE:", wolfram_response)

        if wolfram_response:
            response = wolfram_response

    if "my name is " in q:
        name = user_input.split("my name is ", 1)[1].strip()
        save_memory(req.user_id, "name", name)

    if "i like " in q:
        like = user_input.split("i like ", 1)[1].strip()
        save_memory(req.user_id, "likes", like)

    if "i am from " in q:
        place = user_input.split("i am from ", 1)[1].strip()
        save_memory(req.user_id, "location", place)

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
        history = get_user_history(req.user_id)

        context = ""

        for u, b in history:
            context += f"User: {u}\n"
            context += f"Assistant: {b}\n"

        memories = get_all_memories(req.user_id)[:5]

        memory_context = ""

        for k, v in memories:
            memory_context += f"{k}: {v}\n"

        full_input = (
            "Known facts about user:\n"
            + memory_context
            + "\nConversation history:\n"
            + context
            + f"User: {user_input}\n"
            + "Assistant:"
        )

        response = engine.chat(
            full_input,
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