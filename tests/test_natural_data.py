from hermitlm.tools.decision_model import FINE21_ORDER
from hermitlm.training.generate_data import build_natural_samples
from hermitlm.training.natural_data import NATURAL_INPUTS, perturb


def test_natural_pool_covers_all_fine_types():
    assert set(NATURAL_INPUTS) == set(FINE21_ORDER)
    assert all(len(phrases) >= 10 for phrases in NATURAL_INPUTS.values())


def test_build_natural_samples_returns_balanced_rows():
    rows = build_natural_samples(per_type=3)
    assert len(rows) == len(FINE21_ORDER) * 3
    assert {r["type"] for r in rows} == set(FINE21_ORDER)
    assert all(r["input"] and r["output"] for r in rows)


def test_perturb_is_bounded_and_seeded():
    import random

    a = perturb("hello there, crab", random.Random(1))
    b = perturb("hello there, crab", random.Random(1))
    assert a == b
    assert "crab" in a
