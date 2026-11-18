import hermitlm.tools.memory as memory_mod
from hermitlm.tools.memory import extract_fact, get_memory, init_memory_db, save_memory


def test_memory(tmp_path, monkeypatch):
    monkeypatch.setattr(memory_mod, "DB_PATH", str(tmp_path / "hermit.db"))
    init_memory_db()

    save_memory(
        "test_user",
        "name",
        "darshi"
    )
    result = get_memory(
        "test_user",
        "name"
    )
    assert result == "darshi"


def test_extract_fact_is_bounded_and_preserves_original_case():
    assert extract_fact("My name is Darshi, and I like tea") == ("name", "Darshi")
    assert extract_fact("I LIKE kelp. really") == ("likes", "kelp")
    assert extract_fact("I am from Éire!") == ("location", "Éire")
    assert extract_fact("hello crab") is None
