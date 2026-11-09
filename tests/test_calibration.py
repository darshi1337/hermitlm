import pytest
import torch

from hermitlm.tools.calibration import (
    InsufficientDataError,
    fit_temperature,
    platt_targets,
)


def test_platt_targets_binary():
    targets = torch.tensor([0, 0, 0, 1])
    smoothed = platt_targets(targets, 2)
    assert torch.allclose(smoothed[0], torch.tensor([0.8, 0.2], dtype=torch.float64))
    assert torch.allclose(smoothed[3], torch.tensor([1 / 3, 2 / 3], dtype=torch.float64))
    assert torch.allclose(smoothed.sum(dim=1), torch.ones(4, dtype=torch.float64))


def test_platt_targets_multiclass_spreads_rest():
    targets = torch.tensor([0, 0, 1, 2])
    smoothed = platt_targets(targets, 3)
    # class 2 has one label: own = 2/3, remaining 1/3 split over two classes
    assert torch.allclose(smoothed[3], torch.tensor([1 / 6, 1 / 6, 2 / 3], dtype=torch.float64))
    assert torch.allclose(smoothed.sum(dim=1), torch.ones(4, dtype=torch.float64))


def test_fit_temperature_recovers_known_temperature():
    torch.manual_seed(0)
    logits = torch.randn(3000, 5, dtype=torch.float64) * 3
    # labels drawn from the T=2.5 distribution; fitting on the raw scores
    # should recover approximately 2.5
    labels = torch.multinomial(torch.softmax(logits / 2.5, dim=-1), 1).squeeze(1)
    fitted = fit_temperature(logits, labels, smoothing="never")
    assert 2.0 < fitted < 3.2


def test_binary_fit_uses_platt_targets():
    # Near-separable binary scores: plain NLL would drive T to the lower bound
    # and report ~1.0 confidence. Platt targets keep the probabilities honest.
    logits = torch.tensor([[8.0, -8.0]] * 30 + [[-8.0, 8.0]] * 30, dtype=torch.float64)
    labels = torch.tensor([0] * 30 + [1] * 30)
    fitted = fit_temperature(logits, labels)
    probs = torch.softmax(logits / fitted, dim=-1)
    assert fitted > 2.0
    assert float(probs.max()) < 0.98


def test_multiclass_lower_bound_falls_back_to_platt():
    rows, labels = [], []
    for c in range(3):
        for _ in range(20):
            row = [-8.0, -8.0, -8.0]
            row[c] = 8.0
            rows.append(row)
            labels.append(c)
    logits = torch.tensor(rows, dtype=torch.float64)
    fitted = fit_temperature(logits, torch.tensor(labels))
    probs = torch.softmax(logits / fitted, dim=-1)
    assert float(probs.max()) < 0.99


def test_fit_temperature_rejects_small_samples():
    with pytest.raises(InsufficientDataError):
        fit_temperature(torch.randn(10, 3), torch.zeros(10, dtype=torch.long))


def test_fit_temperature_flat_scores_return_identity():
    logits = torch.zeros(60, 4, dtype=torch.float64)
    labels = torch.arange(60) % 4
    assert fit_temperature(logits, labels) == 1.0
