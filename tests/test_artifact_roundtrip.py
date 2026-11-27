import json
import os
from pathlib import Path

import pytest
import torch

from hermitlm.tools.artifacts import (
    ArtifactMismatchError,
    head_metadata,
    refit_metadata,
)
from hermitlm.tools.decision_model import ProbeDecisionModel, export_probe
from hermitlm.tools.heads import FittedHead, select_head

HAS_ARTIFACTS = all(
    os.path.exists(p)
    for p in (
        "checkpoints/best_model.pt",
        "data/tokenizer.json",
        "checkpoints/decision_probe.pt",
        "checkpoints/decision_tool_probe.pt",
    )
)
needs_artifacts = pytest.mark.skipif(not HAS_ARTIFACTS, reason="requires shipped probe artifacts")


def _separable(n: int = 60, dim: int = 6, classes: int = 3, layers: int = 3):
    generator = torch.Generator().manual_seed(0)
    hidden = torch.randn(n, layers, dim, generator=generator)
    labels = torch.tensor([i % classes for i in range(n)])
    for layer in range(layers):
        for cls in range(classes):
            hidden[labels == cls, layer, :] += 0.8 * cls
    return hidden, labels


def test_head_roundtrip_keeps_decisions_identical():
    hidden, labels = _separable()
    head = select_head(hidden, labels, 3, [1], k=5, seed=0)
    state = head.as_dict()

    restored = FittedHead.from_dict(json.loads(json.dumps(state)))
    x = torch.randn(5, 6, generator=torch.Generator().manual_seed(3))
    assert restored.layer == head.layer
    assert restored.solver == head.solver
    assert restored.alpha == head.alpha
    assert torch.allclose(restored.log_probs(x), head.log_probs(x), atol=1e-12)
    assert (restored.scores(x).argmax(-1) == head.scores(x).argmax(-1)).all()


def test_saved_artifact_roundtrip_and_export(tmp_path):
    checkpoint = tmp_path / "model.pt"
    tokenizer = tmp_path / "tokenizer.json"
    checkpoint.write_bytes(b"weights")
    tokenizer.write_bytes(b"{}")

    hidden, labels = _separable(classes=2)
    head = select_head(hidden, labels, 2, [2], k=5, seed=7)
    state = head.as_dict()
    thresholds = {"clarify": 0.6, "escalation": 0.4, "target_error": 0.1}
    refit = refit_metadata({"coarse": state}, folds=5, seed=7)
    artifact = {
        **head_metadata(
            checkpoint, tokenizer,
            chosen_layer=head.layer, thresholds=thresholds, refit=refit,
        ),
        "selection": {"layer": head.layer, "heads": {"coarse": state}},
        "temperature7": head.temperature,
        "head7": {
            "weight": torch.tensor(state["weight"], dtype=torch.float64),
            "bias": torch.tensor(state["bias"], dtype=torch.float64),
        },
    }
    path = tmp_path / "probe.pt"
    torch.save(artifact, path)
    loaded = torch.load(path, weights_only=True)

    assert loaded["chosen_layer"] == head.layer
    assert loaded["thresholds"] == thresholds
    assert loaded["refit"]["folds"] == 5
    restored = FittedHead.from_dict(loaded["selection"]["heads"]["coarse"])
    x = torch.randn(3, 6, generator=torch.Generator().manual_seed(4))
    assert torch.allclose(restored.scores(x), head.scores(x), atol=1e-12)

    out = export_probe(str(path), str(tmp_path / "probe.json"))
    web = json.loads(Path(tmp_path / "probe.json").read_text(encoding="utf-8"))
    assert web["chosen_layer"] == head.layer
    assert web["head_layer"] == head.layer
    assert web["thresholds"] == thresholds
    assert web["refit"] == refit
    assert out["schema_version"] == artifact["schema_version"]


@needs_artifacts
def test_tool_head_layer_parity_and_stale_rejection(tmp_path):
    model = ProbeDecisionModel(
        "checkpoints/best_model.pt", "data/tokenizer.json",
        "checkpoints/decision_probe.pt", "cpu",
    )
    assert model.load_tool_heads("checkpoints/decision_tool_probe.pt")
    tool = torch.load("checkpoints/decision_tool_probe.pt", map_location="cpu", weights_only=True)
    assert model.tool_layer == tool["chosen_layer"]

    stale = dict(tool)
    stale["checkpoint_sha256"] = "0" * 64
    path = tmp_path / "stale.pt"
    torch.save(stale, path)
    with pytest.raises(ArtifactMismatchError):
        model.load_tool_heads(str(path))
