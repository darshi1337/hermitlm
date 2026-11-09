import math

import pytest
import torch

from hermitlm.tools.artifacts import (
    ArtifactMismatchError,
    artifact_metadata,
    verify_artifact,
)
from hermitlm.tools.metrics import (
    accuracy,
    bootstrap_ci,
    ece,
    macro_f1,
    nll,
    paired_bootstrap_diff,
)


def test_accuracy_hand_computed():
    probs = torch.tensor([[0.8, 0.2], [0.6, 0.4]], dtype=torch.float64)
    assert accuracy(probs, torch.tensor([0, 1])) == 0.5


def test_nll_hand_computed():
    probs = torch.tensor([[0.8, 0.2], [0.6, 0.4]], dtype=torch.float64)
    expected = -(math.log(0.8) + math.log(0.6)) / 2
    assert abs(nll(probs, torch.tensor([0, 0])) - expected) < 1e-9


def test_ece_hand_computed():
    probs = torch.tensor([[0.8, 0.2], [0.6, 0.4]], dtype=torch.float64)
    # 0.8 with correct answer (gap 0.2), 0.6 with wrong answer (gap 0.6)
    assert abs(ece(probs, torch.tensor([0, 1])) - 0.4) < 1e-9


def test_macro_f1_two_classes():
    preds = torch.tensor([0, 1, 1, 0])
    targets = torch.tensor([0, 1, 0, 0])
    # class 0: P=1.0 R=2/3 F1=0.8; class 1: P=0.5 R=1.0 F1=2/3
    assert abs(macro_f1(preds, targets, 2) - (0.8 + 2 / 3) / 2) < 1e-9


def test_bootstrap_ci_brackets_point_estimate():
    outcomes = torch.tensor([1.0] * 80 + [0.0] * 20, dtype=torch.float64)

    def stat(idx):
        return float(outcomes[idx].mean())

    lo, hi = bootstrap_ci(stat, len(outcomes), n_boot=500, seed=1)
    assert lo <= 0.8 <= hi


def test_paired_bootstrap_diff_positive_when_b_better():
    a_values = torch.tensor([0.0] * 100 + [1.0] * 100, dtype=torch.float64)
    b_values = torch.ones(200, dtype=torch.float64)

    def stat_a(idx):
        return float(a_values[idx].mean())

    def stat_b(idx):
        return float(b_values[idx].mean())

    result = paired_bootstrap_diff(stat_a, stat_b, 200, n_boot=400, seed=0)
    assert abs(result["diff"] - 0.5) < 1e-9
    assert result["ci"][0] > 0


def test_artifact_hash_roundtrip_and_mismatch(tmp_path):
    checkpoint = tmp_path / "model.pt"
    tokenizer = tmp_path / "tokenizer.json"
    checkpoint.write_bytes(b"weights-v1")
    tokenizer.write_bytes(b"{}")
    metadata = artifact_metadata(checkpoint, tokenizer)
    verify_artifact(metadata, checkpoint, tokenizer)  # no raise
    checkpoint.write_bytes(b"weights-v2")
    with pytest.raises(ArtifactMismatchError):
        verify_artifact(metadata, checkpoint, tokenizer)


def test_legacy_artifact_without_hashes_is_accepted(tmp_path):
    checkpoint = tmp_path / "model.pt"
    tokenizer = tmp_path / "tokenizer.json"
    checkpoint.write_bytes(b"weights")
    tokenizer.write_bytes(b"{}")
    verify_artifact({"temperature7": 1.0}, checkpoint, tokenizer)  # no raise
