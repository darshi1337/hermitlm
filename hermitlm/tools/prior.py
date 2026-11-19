"""Label-free batch prior correction (brier's L0, Zhou et al. 2024).

``log p_corrected = norm(log p - lam * log pi)`` where ``pi`` is the mean
predicted probability over an unlabelled pool. Corrects systematic label bias
without needing any labels.
"""
import torch


def fit_prior(log_probs: torch.Tensor) -> torch.Tensor:
    p = torch.exp(log_probs.detach().to(torch.float64))
    if p.ndim != 2 or p.size(0) == 0:
        raise ValueError("fit_prior needs a non-empty (N, K) array")
    prior = p.mean(dim=0).clamp_min(1e-6)
    return prior / prior.sum()


def apply_prior(log_probs: torch.Tensor, prior: torch.Tensor, lam: float = 1.0) -> torch.Tensor:
    if not 0.0 <= lam <= 1.0:
        raise ValueError("lam must be in [0, 1]")
    lp = log_probs.to(torch.float64)
    pi = prior.to(lp.device).to(torch.float64)
    corrected = lp - float(lam) * torch.log(pi)
    return corrected - torch.logsumexp(corrected, dim=-1, keepdim=True)
