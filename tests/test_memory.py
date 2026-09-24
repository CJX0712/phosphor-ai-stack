import tempfile
from pathlib import Path

from phosphor.core.types import Message
from phosphor.memory import InProcMemory, build_memory


def test_append_and_recent():
    mem = InProcMemory()
    mem.append("s1", Message(role="user", content="第一条"))
    mem.append("s1", Message(role="assistant", content="第二条"))
    assert [m.content for m in mem.recent("s1")] == ["第一条", "第二条"]


def test_window_is_bounded():
    mem = InProcMemory(max_per_session=3)
    for i in range(10):
        mem.append("s", Message(role="user", content=f"m{i}"))
    assert len(mem.recent("s", limit=100)) == 3


def test_clear_returns_removed_count():
    mem = InProcMemory()
    mem.append("s", Message(role="user", content="x"))
    assert mem.clear("s") == 1
    assert mem.recent("s") == []


def test_persistence_round_trip():
    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "memory.json")
        first = build_memory(path=path)
        first.append("s", Message(role="user", content="持久化内容"))
        second = build_memory(path=path)
        assert second.recent("s")[0].content == "持久化内容"


def test_stats():
    mem = InProcMemory()
    mem.append("s", Message(role="user", content="x"))
    assert mem.stats()["sessions"] == 1
    assert mem.stats()["messages"] == 1
