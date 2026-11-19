import torch

from hermitlm.tools.heads import default_layers, fit_lda, fit_ridge, select_head
from hermitlm.tools.prior import apply_prior, fit_prior


def test_default_layers_for_small_and_large_models():
    assert default_layers(6) == [3, 5]
    assert len(default_layers(28)) >= 4


def test_ridge_and_lda_fit_and_score():
    torch.manual_seed(0)
    x = torch.randn(80, 12)
    y = torch.randint(0, 3, (80,))
    for fit in (lambda: fit_ridge(x, y, 3, 1.0), lambda: fit_lda(x, y, 3)):
        mean, std, w, bias = fit()
        scores = ((x - mean) / std) @ w.T + bias
        assert scores.shape == (80, 3)
        assert torch.isfinite(scores).all()


def test_select_head_picks_a_valid_candidate():
    torch.manual_seed(1)
    n, layers, d = 120, 4, 10
    hidden = torch.randn(n, layers, d)
    signal = hidden[:, 2, 0] > 0
    labels = signal.long()
    head = select_head(hidden, labels, 2, list(range(layers)), k=3, seed=0)
    assert head.layer in range(layers)
    assert head.solver in ("ridge", "lda")
    assert 0.0 <= head.oof_accuracy <= 1.0
    assert torch.isfinite(head.log_probs(hidden[:, head.layer, :])).all()


def test_prior_correction_is_normalized_and_bounded():
    lp = torch.log_softmax(torch.randn(20, 4), dim=-1)
    prior = fit_prior(lp)
    assert prior.shape == (4,)
    assert torch.isclose(prior.sum(), torch.tensor(1.0, dtype=prior.dtype))
    corrected = apply_prior(lp, prior, 1.0)
    assert torch.isclose(
        torch.exp(corrected).sum(-1), torch.ones(20, dtype=corrected.dtype), atol=1e-6
    ).all()
