import json

import torch

from hermitlm.config import HermitConfig


def _toy_config():
    return HermitConfig(
        vocab_size=64,
        max_seq_len=32,
        d_model=16,
        n_layers=2,
        n_heads=2,
        ffn_hidden=32,
        dropout=0.0,
    )


def test_export_wrapper_returns_logits_and_last_hidden():
    from hermitlm.training.model import ChatWithHidden, HermitLM

    config = _toy_config()
    model = HermitLM(config).eval()
    wrapper = ChatWithHidden(model)
    idx = torch.randint(0, config.vocab_size, (1, 8))
    logits, last_hidden = wrapper(idx)
    assert logits.shape == (1, 8, config.vocab_size)
    assert last_hidden.shape == (1, config.d_model)
    with torch.no_grad():
        _, _, hidden = model(idx, return_hidden=True)
    assert torch.allclose(last_hidden, hidden[:, -1, :])


def test_export_probe_json_matches_checkpoint(tmp_path):
    from hermitlm.tools.decision_model import export_probe

    out = tmp_path / "probe.json"
    data = export_probe("checkpoints/decision_probe.pt", str(out))
    assert data["options"] == list("abcdefg")
    assert set(data["names"]) == set("abcdefg")
    assert len(data["head7"]["bias"]) == 7
    assert len(data["head7"]["weight"]) == 7
    assert len(data["head7"]["weight"][0]) == len(data["head_bin"]["weight"][0])
    assert len(data["head_bin"]["bias"]) == 2
    assert data["temperature7"] > 0 and data["temperature_bin"] > 0
    reloaded = json.loads(out.read_text())
    assert reloaded["options"] == data["options"]
