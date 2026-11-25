import json
import os
from pathlib import Path

import pytest
import torch

from hermitlm.tools.decision_model import export_probe


def test_export_probe_includes_priors_for_legacy_artifact(tmp_path):
    """The shipped probe predates head selection; its prior must still export."""
    probe = {
        "temperature7": 1.0,
        "temperature_bin": 1.0,
        "head7": {"weight": torch.zeros(7, 4), "bias": torch.zeros(7)},
        "head_bin": {"weight": torch.zeros(2, 4), "bias": torch.zeros(2)},
        "priors": {"coarse": [1 / 7] * 7, "confused": [0.9, 0.1]},
        "prior_lambda": 0.5,
    }
    probe_path = tmp_path / "probe.pt"
    torch.save(probe, probe_path)

    data = export_probe(str(probe_path), str(tmp_path / "probe.json"))

    assert data["priors"] == probe["priors"]
    assert data["prior_lambda"] == 0.5
    on_disk = json.loads((tmp_path / "probe.json").read_text())
    assert on_disk["priors"] == probe["priors"]


@pytest.mark.skipif(
    not os.path.exists("web/probe.json") or not os.path.exists("checkpoints/decision_probe.pt"),
    reason="requires the shipped probe artifact",
)
def test_web_probe_priors_match_artifact():
    """API and browser must load the same prior and lambda."""
    artifact = torch.load("checkpoints/decision_probe.pt", map_location="cpu", weights_only=True)
    web = json.loads(Path("web/probe.json").read_text(encoding="utf-8"))

    assert web.get("priors") == artifact.get("priors")
    assert web.get("prior_lambda", 1.0) == pytest.approx(
        float(artifact.get("prior_lambda", 1.0))
    )


@pytest.mark.skipif(
    not os.path.exists("web/probe.json") or not os.path.exists("checkpoints/decision_probe.pt"),
    reason="requires the shipped probe artifact",
)
def test_web_probe_scores_match_artifact():
    """Mirror web/intent.js classifyIntent() and compare with the API path."""
    from hermitlm.tools.decision_model import OPTION_LETTERS, LegacyHead, debias_level
    from hermitlm.tools.prior import apply_prior

    artifact = torch.load("checkpoints/decision_probe.pt", map_location="cpu", weights_only=True)
    web = json.loads(Path("web/probe.json").read_text(encoding="utf-8"))
    hidden = torch.randn(len(web["head7"]["weight"][0]), generator=torch.Generator().manual_seed(0))

    # API path: linear head, temperature, prior, softmax.
    lam = float(web["prior_lambda"])
    head = LegacyHead(artifact["head7"], artifact["temperature7"], torch.device("cpu"))
    api_logp = apply_prior(head.log_probs(hidden.unsqueeze(0)), torch.tensor(web["priors"]["coarse"]), lam)[0]
    api_proba = torch.exp(api_logp)

    # Browser path, same arithmetic as web/intent.js softmax().
    logits = torch.tensor(web["head7"]["weight"]) @ hidden + torch.tensor(web["head7"]["bias"])
    scaled = logits / web["temperature7"]
    scaled = scaled - lam * torch.log(torch.tensor(web["priors"]["coarse"]).clamp_min(1e-6))
    web_proba = torch.softmax(scaled, dim=-1)

    assert torch.allclose(api_proba.double(), web_proba.double(), atol=1e-6)
    assert OPTION_LETTERS[int(api_proba.argmax())] == web["options"][int(web_proba.argmax())]
    assert debias_level(web["priors"], web["prior_lambda"]) == "L0"
