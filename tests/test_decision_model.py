import pytest
import torch
from tokenizers import Tokenizer

from hermitlm.tools.decision_model import (
    ALL_FINE_TYPES,
    COARSE_MEMBERS,
    OPTION_LETTERS,
    constrained_probs,
    format_prompt,
    get_option_ids,
)

TOKENIZER_PATH = "data/tokenizer.json"


def test_mapping_covers_all_21_types_without_overlap():
    assert len(ALL_FINE_TYPES) == 21
    assert len(OPTION_LETTERS) == 7
    seen: list[str] = []
    for coarse, fines in COARSE_MEMBERS.items():
        assert fines, coarse
        seen.extend(fines)
    assert sorted(seen) == ALL_FINE_TYPES
    assert len(set(seen)) == 21


def test_option_ids_are_single_distinct_tokens():
    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
    ids = get_option_ids(tokenizer)
    assert ids == [21, 22, 23, 24, 25, 26, 27]


def test_option_ids_reject_multi_token_verbalizer(monkeypatch):
    import hermitlm.tools.decision_model as dm

    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
    monkeypatch.setattr(dm, "OPTION_LETTERS", ["greeting"])
    with pytest.raises(ValueError):
        get_option_ids(tokenizer)


def test_constrained_probs_masks_and_normalizes():
    logits = torch.tensor([2.0, 1.0, 0.0, -1.0, -2.0])
    probs = constrained_probs(logits, [0, 2, 4], temperature=1.0)
    assert probs.shape == (3,)
    assert torch.isclose(probs.sum(), torch.tensor(1.0), atol=1e-5)
    assert probs[0] > probs[1] > probs[2]


def test_constrained_probs_temperature_flattens():
    logits = torch.tensor([3.0, 0.0, 0.0])
    sharp = constrained_probs(logits, [0, 1, 2], temperature=1.0)
    flat = constrained_probs(logits, [0, 1, 2], temperature=5.0)
    assert sharp[0] > flat[0]
    assert flat[1] > sharp[1]


def test_constrained_probs_rejects_nonpositive_temperature():
    with pytest.raises(ValueError):
        constrained_probs(torch.tensor([1.0, 2.0]), [0, 1], temperature=0.0)


def test_format_prompt_lists_all_options():
    prompt = format_prompt("hello crab")
    assert "hello crab" in prompt
    for letter in OPTION_LETTERS:
        assert f"{letter}." in prompt
    assert prompt.rstrip().endswith("<|im_start|>assistant")


def test_predict_single_pass_cpu():
    from hermitlm.tools.decision_model import DecisionModel

    model = DecisionModel("checkpoints/best_model.pt", TOKENIZER_PATH, device="cpu")
    label, proba = model.predict("hello crab")
    assert label in OPTION_LETTERS
    assert set(proba) == set(OPTION_LETTERS)
    assert abs(sum(proba.values()) - 1.0) < 1e-5


def test_probe_predict_known_intents():
    from hermitlm.tools.decision_model import ProbeDecisionModel

    model = ProbeDecisionModel(
        "checkpoints/best_model.pt", TOKENIZER_PATH, "checkpoints/decision_probe.pt", device="cpu"
    )
    label, proba, p_confused = model.predict("hello crab")
    assert label == "a"  # social: greeting is unambiguous in train
    assert abs(sum(proba.values()) - 1.0) < 1e-5
    assert 0.0 <= p_confused <= 1.0
    _, _, p_conf = model.predict("what is inventory management")
    assert p_conf > p_confused  # out-of-scope scores higher on confused head
