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
    option_assignment,
    rotation_shifts,
    unrotate_display,
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
    # IDs shift when the tokenizer is retrained; only single-token, distinct
    # verbalizers are a real requirement.
    assert len(ids) == len(OPTION_LETTERS) == 7
    assert len(set(ids)) == 7
    assert all(len(tokenizer.encode(letter).ids) == 1 for letter in OPTION_LETTERS)


def test_fine_option_ids_are_single_distinct_tokens():
    from hermitlm.tools.decision_model import FINE21_LETTERS, get_fine_option_ids

    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
    ids = get_fine_option_ids(tokenizer)
    assert len(ids) == 21 == len(FINE21_LETTERS)
    assert len(set(ids)) == 21


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


def test_rotation_shifts_are_evenly_spaced():
    assert rotation_shifts(1) == [0]
    assert rotation_shifts(2) == [0, 4]  # two passes, farthest apart
    assert rotation_shifts(7) == list(range(7))
    with pytest.raises(ValueError):
        rotation_shifts(0)
    with pytest.raises(ValueError):
        rotation_shifts(8)


def test_option_permutation_equivariance():
    """Permute which class each letter denotes; unrolling must restore the output."""
    torch.manual_seed(0)
    probs = torch.softmax(torch.randn(len(OPTION_LETTERS)), dim=-1)
    original = {letter: float(probs[i]) for i, letter in enumerate(OPTION_LETTERS)}
    for shift in range(len(OPTION_LETTERS)):
        assignment = option_assignment(shift)
        displayed = [original[cls] for cls in assignment]
        restored = unrotate_display(displayed, shift)
        for cls in OPTION_LETTERS:
            assert restored[cls] == pytest.approx(original[cls])


def test_rotation_permutes_class_assignments():
    base = format_prompt("hello crab")
    rotated = format_prompt("hello crab", shift=3)

    assert "a. social" in base and "e. social" in rotated
    assert "a. sense" in rotated and "d. sense" in base
    # The class set is unchanged; only which letter names it moves.
    base_classes = {line.split(". ", 1)[1] for line in base.splitlines() if line[:3] in {f"{l}. " for l in OPTION_LETTERS}}
    rotated_classes = {line.split(". ", 1)[1] for line in rotated.splitlines() if line[:3] in {f"{l}. " for l in OPTION_LETTERS}}
    assert base_classes == rotated_classes


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


def test_probe_heads_share_one_forward_pass():
    from hermitlm.tools.decision_model import ProbeDecisionModel

    model = ProbeDecisionModel(
        "checkpoints/best_model.pt", TOKENIZER_PATH, "checkpoints/decision_probe.pt", device="cpu"
    )
    model.predict("hello crab")
    model.predict_fine("hello crab")
    assert len(model._hidden_cache) == 1
    model.predict("a different query")
    assert len(model._hidden_cache) == 2


def test_probe_predict_fine_known_type():
    from hermitlm.tools.decision_model import ProbeDecisionModel

    model = ProbeDecisionModel(
        "checkpoints/best_model.pt", TOKENIZER_PATH, "checkpoints/decision_probe.pt", device="cpu"
    )
    result = model.predict_fine("hello crab")
    assert result is not None
    fine_label, by_fine = result
    assert fine_label == "greeting"
    assert abs(sum(by_fine.values()) - 1.0) < 1e-5
