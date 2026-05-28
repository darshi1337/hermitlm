from hermitlm.tools.memory import (
    save_memory,
    get_memory,
)

def test_memory():
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