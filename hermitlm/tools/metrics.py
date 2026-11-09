"""Calibration metrics and bootstrap intervals for decision heads.

All functions expect probabilities as a float tensor ``[n, n_classes]`` and
integer targets as ``[n]``. Bootstrap helpers take a statistic over an index
tensor so the same code resamples accuracy, NLL, ECE or macro F1.
"""

from collections.abc import Callable
from typing import Any

import torch

N_BINS = 10

Statistic = Callable[[torch.Tensor], float]


def _prepare(probs: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if probs.ndim != 2:
        raise ValueError("probs must have shape [n, n_classes]")
    if targets.ndim != 1 or len(targets) != len(probs):
        raise ValueError("targets must have shape [n] and match probs")
    return probs.detach().to(dtype=torch.float64), targets.detach().to(dtype=torch.long)


def accuracy(probs: torch.Tensor, targets: torch.Tensor) -> float:
    probs, targets = _prepare(probs, targets)
    if len(targets) == 0:
        return 0.0
    return float((probs.argmax(-1) == targets).to(torch.float64).mean())


def nll(probs: torch.Tensor, targets: torch.Tensor) -> float:
    probs, targets = _prepare(probs, targets)
    if len(targets) == 0:
        return 0.0
    picked = probs.gather(1, targets.unsqueeze(1)).squeeze(1).clamp_min(1e-12)
    return float(-picked.log().mean())


def ece(probs: torch.Tensor, targets: torch.Tensor, n_bins: int = N_BINS) -> float:
    """Expected calibration error over equal-width confidence bins."""
    probs, targets = _prepare(probs, targets)
    n = len(targets)
    if n == 0:
        return 0.0
    conf, pred = probs.max(dim=-1)
    correct = (pred == targets).to(torch.float64)
    bins = torch.clamp((conf * n_bins).long(), max=n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        mask = bins == b
        count = int(mask.sum())
        if count:
            gap = abs(float(conf[mask].mean()) - float(correct[mask].mean()))
            total += count / n * gap
    return total


def macro_f1(preds: torch.Tensor, targets: torch.Tensor, n_classes: int) -> float:
    """Macro F1 over a fixed class set (absent classes contribute 0)."""
    preds = preds.detach().to(dtype=torch.long)
    targets = targets.detach().to(dtype=torch.long)
    if len(targets) == 0 or n_classes < 1:
        return 0.0
    scores = []
    for c in range(n_classes):
        tp = int(((preds == c) & (targets == c)).sum())
        fp = int(((preds == c) & (targets != c)).sum())
        fn = int(((preds != c) & (targets == c)).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return sum(scores) / n_classes


def bootstrap_ci(
    statistic: Statistic,
    n: int,
    *,
    n_boot: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Percentile bootstrap CI for a statistic over ``n`` items."""
    if n < 2 or n_boot < 1:
        return (0.0, 0.0)
    generator = torch.Generator().manual_seed(seed)
    draws = torch.randint(0, n, (n_boot, n), generator=generator)
    values = torch.tensor([statistic(idx) for idx in draws], dtype=torch.float64)
    lo = float(torch.quantile(values, alpha / 2))
    hi = float(torch.quantile(values, 1 - alpha / 2))
    return (lo, hi)


def paired_bootstrap_diff(
    stat_a: Statistic,
    stat_b: Statistic,
    n: int,
    *,
    n_boot: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Paired bootstrap of ``stat_b - stat_a`` over shared item indices.

    Returns ``{"diff": point_estimate, "ci": [lo, hi]}``.
    """
    every = torch.arange(n)
    point = float(stat_b(every) - stat_a(every))
    if n < 2 or n_boot < 1:
        return {"diff": point, "ci": [0.0, 0.0]}
    generator = torch.Generator().manual_seed(seed)
    draws = torch.randint(0, n, (n_boot, n), generator=generator)
    values = torch.tensor(
        [stat_b(idx) - stat_a(idx) for idx in draws], dtype=torch.float64
    )
    lo = float(torch.quantile(values, alpha / 2))
    hi = float(torch.quantile(values, 1 - alpha / 2))
    return {"diff": point, "ci": [lo, hi]}
