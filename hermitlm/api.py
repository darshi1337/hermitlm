from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from hermitlm.inference import HermitInference
from hermitlm.settings import (
    CHECKPOINT_PATH,
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_K,
    DEVICE,
    TOKENIZER_PATH,
)

app = FastAPI(title="HermitLM API")

engine = HermitInference(CHECKPOINT_PATH, TOKENIZER_PATH, device=DEVICE)


class ChatRequest(BaseModel):
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
        raise HTTPException(status_code=400, detail="message cannot be empty")

    response = engine.chat(
        user_input,
        temperature=req.temperature,
        top_k=req.top_k,
        max_tokens=req.max_tokens,
    )

    return ChatResponse(response=response)
