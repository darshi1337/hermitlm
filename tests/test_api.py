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
