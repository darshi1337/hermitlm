import pytest
import torch
from tokenizers import Tokenizer

from hermitlm.config import HermitConfig
from hermitlm.training.dpo import _encode_pair, dpo_loss, run_dpo


def test_dpo_loss_uses_only_masked_positions():
    # Two sequences with identical prompt log-probs but different completion
    # log-probs: masking the prompt must change the preference margin.
    pi_chosen = torch.tensor([0.0, -1.0, -2.0])
    pi_rej = torch.tensor([-5.0, -2.0, -5.0])
    ref_chosen = torch.zeros(3)
    ref_rej = torch.zeros(3)
    full = dpo_loss(pi_chosen, pi_rej, ref_chosen, ref_rej, beta=1.0)
    mask = torch.tensor([0.0, 1.0, 1.0])
    masked = dpo_loss(
        pi_chosen, pi_rej, ref_chosen, ref_rej,
        beta=1.0, mask_c=mask, mask_r=mask,
    )
    assert not torch.isclose(full, masked)


def test_encode_pair_masks_completion_only():
    tokenizer = Tokenizer.from_file("data/tokenizer.json")
    config = HermitConfig()
    ids, mask = _encode_pair(
        tokenizer, config, torch.device("cpu"),
        "hello crab", "food detected", "i cannot answer that",
    )
    assert ids.shape[0] == mask.shape[0] == 2
    assert ids.size(1) <= config.max_seq_len
    # Last non-pad token of each sequence is the end-of-turn token.
    for row in ids:
        last = int((row != config.pad_id).nonzero()[-1])
        assert row[last] == config.eos_id
    # Shared prompt positions are unmasked; completion targets are masked.
    prompt_len = len(tokenizer.encode("<|im_start|>user\nhello crab<|im_end|>\n<|im_start|>assistant\n").ids)
    assert (mask[:, : prompt_len - 1] == 0).all()
    expected = [
        float(len(tokenizer.encode(text).ids) + 1)
        for text in ("food detected", "i cannot answer that")
    ]
    assert mask[:, prompt_len - 1 :].sum(-1).tolist() == expected


def test_encode_pair_truncates_long_prompt_without_losing_completion():
    tokenizer = Tokenizer.from_file("data/tokenizer.json")
    config = HermitConfig()
    ids, _mask = _encode_pair(
        tokenizer, config, torch.device("cpu"),
        "word " * 2000, "short answer", "other answer",
    )
    assert ids.size(1) <= config.max_seq_len
    for row in ids:
        last = int((row != config.pad_id).nonzero()[-1])
        assert row[last] == config.eos_id


def test_run_dpo_missing_checkpoint_fails_loudly():
    with pytest.raises(FileNotFoundError):
        run_dpo(sft_checkpoint="checkpoints/does-not-exist.pt", steps=1)
