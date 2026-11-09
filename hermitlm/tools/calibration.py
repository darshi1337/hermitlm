"""Temperature scaling (L1) for the decision heads.

One temperature ``T`` per question/head, fitted by minimising mean NLL over
``t = log T`` with a bounded golden-section search (Guo et al. 2017).

Binary heads get special treatment: plain NLL on near-separable binary scores
drives ``T`` to its lower bound and produces ~0/1 probabilities. Following
Platt (1999), two-class fits therefore always use smoothed targets. Multiclass
fits use plain NLL and fall back to one-vs-rest Platt targets only when the
plain fit lands on the lower bound.
"""

import math
from collections.abc import Callable

import torch
import torch.nn.functional as F

DEFAULT_T_RANGE: tuple[float, float] = (-3.0, 3.0)
MIN_LABELS = 50

Smoothing = str  # "auto" | "always" | "never"


class InsufficientDataError(ValueError):
    """Raised when a calibration fit has too few labels."""


def platt_targets(targets: torch.Tensor, n_classes: int) -> torch.Tensor:
    """One-vs-rest Platt (1999) smoothed targets.

    The own-class mass is ``(n_c + 1) / (n_c + 2)``; the remaining mass is
    spread evenly over the other classes. For two classes this is exactly
    Platt's binary targets.
    """
    if n_classes < 2:
        raise ValueError("platt targets need at least two classes")
    targets = targets.detach().to(dtype=torch.long)
    counts = torch.bincount(targets, minlength=n_classes).to(dtype=torch.float64)
    own = (counts[targets] + 1.0) / (counts[targets] + 2.0)
    rest = (1.0 - own) / (n_classes - 1)
    out = rest.unsqueeze(1).expand(targets.numel(), n_classes).clone()
    out[torch.arange(targets.numel()), targets] = own
    return out


def _nll(logits: torch.Tensor, targets: torch.Tensor) -> float:
    return float(F.cross_entropy(logits, targets, reduction="mean"))


def _golden_section(
    fn: Callable[[float], float], lo: float, hi: float, tol: float = 1e-4, max_iter: int = 100
) -> float:
    invphi = (math.sqrt(5.0) - 1.0) / 2.0
    a, b = lo, hi
    c = b - invphi * (b - a)
    d = a + invphi * (b - a)
    fc, fd = fn(c), fn(d)
    for _ in range(max_iter):
        if b - a < tol:
            break
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - invphi * (b - a)
            fc = fn(c)
        else:
            a, c, fc = c, d, fd
            d = a + invphi * (b - a)
            fd = fn(d)
    return (a + b) / 2.0


def _best_on_interval(fn: Callable[[float], float], lo: float, hi: float) -> tuple[float, float]:
    """Return (t, nll) with the endpoints evaluated as well as the interior."""
    candidates = [lo, _golden_section(fn, lo, hi), hi]
    values = [fn(t) for t in candidates]
    best = min(range(len(values)), key=lambda i: values[i])
    return candidates[best], values[best]


def fit_temperature(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    n_classes: int | None = None,
    t_range: tuple[float, float] = DEFAULT_T_RANGE,
    min_labels: int = MIN_LABELS,
    smoothing: Smoothing = "auto",
) -> float:
    """Fit one temperature ``T`` by NLL over ``t = log T``.

    Args:
        logits: float tensor of shape ``[n, n_classes]`` (head scores, not probs).
        targets: integer class indices of shape ``[n]``.
        n_classes: class count; defaults to ``logits.shape[1]``.
        t_range: bounds for ``t = log T``. Defaults to ``[-3, 3]`` as in L1.
        min_labels: raise :class:`InsufficientDataError` below this count.
        smoothing: ``"auto"`` (default) uses Platt targets for two classes and
            for multiclass fits that hit the lower bound; ``"always"`` and
            ``"never"`` force the choice.

    Returns:
        The fitted temperature (float, always positive).
    """
    if logits.ndim != 2:
        raise ValueError("logits must have shape [n, n_classes]")
    if targets.ndim != 1 or len(targets) != len(logits):
        raise ValueError("targets must have shape [n] and match logits")
    if len(targets) < min_labels:
        raise InsufficientDataError(
            f"need at least {min_labels} labels to fit a temperature, got {len(targets)}"
        )
    if smoothing not in {"auto", "always", "never"}:
        raise ValueError(f"unknown smoothing mode: {smoothing!r}")
    lo, hi = t_range
    if lo >= hi:
        raise ValueError("t_range must be increasing")

    scores = logits.detach().to(dtype=torch.float64)
    labels = targets.detach().to(dtype=torch.long)
    classes = n_classes if n_classes is not None else scores.shape[1]
    if classes < 2:
        raise ValueError("temperature scaling needs at least two classes")

    def plain(t: float) -> float:
        return _nll(scores * math.exp(-t), labels)

    def smoothed(t: float) -> float:
        return _nll(scores * math.exp(-t), platt_targets(labels, classes))

    if smoothing == "always" or (smoothing == "auto" and classes == 2):
        fn = smoothed
    elif smoothing == "never":
        fn = plain
    else:
        t_plain, _ = _best_on_interval(plain, lo, hi)
        fn = smoothed if t_plain <= lo + 1e-3 else plain

    t, _ = _best_on_interval(fn, lo, hi)
    # Flat NLL (e.g. all-zero logits): T has no effect, keep the identity.
    if fn(t) >= fn(0.0) - 1e-12:
        return 1.0
    return math.exp(t)
