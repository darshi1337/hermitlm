from types import SimpleNamespace

import pytest
import torch

from hermitlm.training.train import get_device


def test_forced_cuda_fails_loudly_when_unavailable(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="(?i)cuda"):
        get_device(SimpleNamespace(device="cuda"))


def test_cpu_override_still_works(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert get_device(SimpleNamespace(device="cpu")).type == "cpu"
