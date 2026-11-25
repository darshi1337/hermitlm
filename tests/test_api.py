import os

import pytest

from hermitlm.settings import CHECKPOINT_PATH, TOKENIZER_PATH

if not (os.path.exists(CHECKPOINT_PATH) and os.path.exists(TOKENIZER_PATH)):
    pytest.skip(
        "requires a trained tokenizer/checkpoint (run scripts/train.sh first)",
        allow_module_level=True,
    )

from fastapi.testclient import TestClient

from hermitlm.runtime.api import app

client = TestClient(app)

def test_home():
    response = client.get("/")
    assert response.status_code == 200


def test_chat_exposes_debias_metadata(monkeypatch):
    from hermitlm.runtime import api

    intent = {
        "label": "a",
        "name": "social",
        "confidence": 0.9,
        "p_confused": 0.1,
        "level": "L0",
        "rotations": 1,
    }
    monkeypatch.setattr(api, "_respond", lambda *_: ("hi", None, intent, None))
    monkeypatch.setattr(api, "_log_turn", lambda *a, **k: None)

    response = client.post("/chat", json={"user_id": "u", "message": "hello"})
    assert response.status_code == 200
    body = response.json()["intent"]
    assert body["level"] == "L0"
    assert body["rotations"] == 1
