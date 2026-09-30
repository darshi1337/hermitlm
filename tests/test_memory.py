import hermitlm.tools.memory as memory_mod
from hermitlm.tools.memory import get_memory, init_memory_db, save_memory


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
