"""L2 fitted heads with layer/solver selection (inspired by brier's METHODS.md).

Heads are fitted on frozen hidden states. Selection searches layers, ridge
penalties and shrinkage LDA using stratified k-fold out-of-fold NLL; the
winner is refitted on all labels and its temperature is fitted on OOF scores.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from hermitlm.tools.calibration import fit_temperature


@dataclass
class FittedHead:
    layer: int
    solver: str
    alpha: float | None
    mean: torch.Tensor
    std: torch.Tensor
    weight: torch.Tensor      # (n_classes, d)
    bias: torch.Tensor        # (n_classes,)
    temperature: float
    oof_nll: float
    oof_accuracy: float

    def to(self, device):
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        self.weight = self.weight.to(device)
        self.bias = self.bias.to(device)
        return self

    def scores(self, hidden: torch.Tensor) -> torch.Tensor:
        x = (hidden - self.mean.to(hidden.device)) / self.std.to(hidden.device)
        return x @ self.weight.to(hidden.device).T + self.bias.to(hidden.device)

    def log_probs(self, hidden: torch.Tensor) -> torch.Tensor:
        return torch.log_softmax(self.scores(hidden) / self.temperature, dim=-1)

    def effective(self):
        """Fold standardization into a plain linear head for JSON export."""
        w = self.weight / self.std.unsqueeze(0)
        b = self.bias - (self.mean / self.std) @ self.weight.T
        return w, b

    def as_dict(self) -> dict:
        w, b = self.effective()
        return {
            "layer": self.layer,
            "solver": self.solver,
            "alpha": self.alpha,
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "weight": self.weight.tolist(),
            "bias": self.bias.tolist(),
            "effective_weight": w.tolist(),
            "effective_bias": b.tolist(),
            "temperature": self.temperature,
            "oof_nll": self.oof_nll,
            "oof_accuracy": self.oof_accuracy,
        }

    @classmethod
    def from_dict(cls, d: dict) -> FittedHead:
        return cls(
            layer=int(d["layer"]),
            solver=d["solver"],
            alpha=d.get("alpha"),
            mean=torch.tensor(d["mean"], dtype=torch.float64),
            std=torch.tensor(d["std"], dtype=torch.float64),
            weight=torch.tensor(d["weight"], dtype=torch.float64),
            bias=torch.tensor(d["bias"], dtype=torch.float64),
            temperature=float(d["temperature"]),
            oof_nll=float(d["oof_nll"]),
            oof_accuracy=float(d["oof_accuracy"]),
        )


def _standardize(x, mean=None, std=None):
    x = x.to(torch.float64)
    if mean is None:
        mean = x.mean(dim=0)
        std = x.std(dim=0, unbiased=False).clamp_min(1e-6)
    return (x - mean) / std, mean, std


def _one_hot(y, n_classes):
    return torch.nn.functional.one_hot(y, num_classes=n_classes).to(torch.float64)


def fit_ridge(hidden, y, n_classes, alpha):
    x, mean, std = _standardize(hidden)
    y_mean = _one_hot(y, n_classes).mean(dim=0)
    a = x.T @ x + alpha * torch.eye(x.size(1), dtype=torch.float64)
    w = torch.linalg.solve(a, x.T @ (_one_hot(y, n_classes) - y_mean))
    return mean, std, w.T, y_mean


def fit_lda(hidden, y, n_classes):
    x, mean, std = _standardize(hidden)
    n, d = x.shape
    class_mean = torch.stack([
        x[y == c].mean(dim=0) if (y == c).any() else torch.zeros(d, dtype=torch.float64)
        for c in range(n_classes)
    ])
    residual = x - class_mean[y]
    sigma = (residual.T @ residual) / n
    m = torch.trace(sigma) / d
    diff = sigma - m * torch.eye(d, dtype=torch.float64)
    # Ledoit-Wolf shrinkage intensity from the d x d scatter (no n x n Gram).
    norms = (residual * residual).sum(dim=1)
    phi = (norms * norms).mean() / d - torch.trace(sigma @ sigma) / d
    denom = (diff * diff).sum()
    gamma = (phi / denom).clamp(0, 1) if denom > 0 else torch.tensor(0.0)
    shrunk = (1 - gamma) * sigma + gamma * m * torch.eye(d, dtype=torch.float64)
    inv = torch.linalg.solve(shrunk, class_mean.T).T  # (C, d)
    quad = (class_mean * inv).sum(dim=1)
    counts = torch.bincount(y, minlength=n_classes).to(torch.float64).clamp_min(1)
    bias = -0.5 * quad + torch.log(counts / counts.sum())
    return mean, std, inv, bias


def _predict(hidden, y, n_classes, solver, alpha):
    if solver == "ridge":
        mean, std, w, bias = fit_ridge(hidden, y, n_classes, alpha)
    else:
        mean, std, w, bias = fit_lda(hidden, y, n_classes)
    x, _, _ = _standardize(hidden, mean, std)
    return x @ w.T + bias


def default_layers(num_layers: int) -> list[int]:
    """Every 2nd block from 40% to 90% depth."""
    first = -(-4 * num_layers // 10)
    last = 9 * num_layers // 10
    return list(range(first, last + 1, 2)) or [num_layers // 2]


def stratified_folds(y, n_classes, k, seed):
    rng = torch.Generator().manual_seed(seed)
    folds = [[] for _ in range(k)]
    offset = 0
    for c in range(n_classes):
        idx = torch.randperm(int((y == c).sum()), generator=rng)
        idx = torch.nonzero(y == c).squeeze(1)[idx]
        for j, i in enumerate(idx.tolist()):
            folds[(offset + j) % k].append(i)
        offset += len(idx)
    return [torch.tensor(sorted(f), dtype=torch.long) for f in folds]


def select_head(hidden_layers, labels, n_classes, layers, alphas=(1e-2, 1e-1, 1.0, 1e1, 1e2, 1e3, 1e4), k=5, seed=0):
    """Return the best FittedHead for the given task."""
    y = torch.as_tensor(labels, dtype=torch.long)
    h = torch.as_tensor(hidden_layers, dtype=torch.float64)
    n = h.size(0)
    folds = stratified_folds(y, n_classes, k, seed)
    candidates = []
    for layer in layers:
        x = h[:, layer, :]
        for solver, alpha in [("ridge", a) for a in alphas] + [("lda", None)]:
            oof = torch.zeros(n, n_classes, dtype=torch.float64)
            try:
                for f in folds:
                    mask = torch.ones(n, dtype=torch.bool)
                    mask[f] = False
                    # Fit on the training folds and score the held-out fold.
                    if solver == "ridge":
                        mean, std, w, bias = fit_ridge(x[mask], y[mask], n_classes, alpha)
                    else:
                        mean, std, w, bias = fit_lda(x[mask], y[mask], n_classes)
                    holdout, _, _ = _standardize(x[f], mean, std)
                    oof[f] = holdout @ w.T + bias
                t = fit_temperature(oof, y, t_range=(-7.0, 7.0))
                nll = torch.nn.functional.cross_entropy(oof / t, y).item()
                acc = (oof.argmax(-1) == y).float().mean().item()
                candidates.append((nll, layer, solver, alpha, t, acc))
            except Exception:  # noqa: BLE001, S112 - skip degenerate grid points
                continue
    if not candidates:
        raise RuntimeError("no head candidate could be fitted")
    nll, layer, solver, alpha, t, acc = min(candidates, key=lambda c: c[0])
    x = h[:, layer, :]
    if solver == "ridge":
        mean, std, w, bias = fit_ridge(x, y, n_classes, alpha)
    else:
        mean, std, w, bias = fit_lda(x, y, n_classes)
    return FittedHead(layer, solver, alpha, mean, std, w, bias, t, nll, acc)
