import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from agentmemoryos.core.working_memory import WorkingMemory


def test_basic_add_and_get():
    wm = WorkingMemory(capacity=5)
    wm.add("k1", "hello world")
    item = wm.get("k1")
    assert item is not None
    assert item.content == "hello world"


def test_lru_eviction():
    wm = WorkingMemory(capacity=3)
    wm.add("a", 1)
    wm.add("b", 2)
    wm.add("c", 3)
    # access 'a' to make it recently used
    wm.get("a")
    # adding 'd' should evict 'b' (least recently used)
    wm.add("d", 4)
    assert "b" not in wm
    assert "a" in wm
    assert "d" in wm


def test_capacity():
    wm = WorkingMemory(capacity=5)
    for i in range(10):
        wm.add(f"key_{i}", i)
    assert len(wm) == 5


def test_access_count():
    wm = WorkingMemory(capacity=10)
    wm.add("x", "test")
    wm.get("x")
    wm.get("x")
    item = wm.get("x")
    assert item.access_count == 3


def test_update_existing():
    wm = WorkingMemory(capacity=5)
    wm.add("key", "old value")
    wm.add("key", "new value")
    assert wm.get("key").content == "new value"
    assert len(wm) == 1


def test_remove():
    wm = WorkingMemory(capacity=5)
    wm.add("to_remove", "gone")
    assert wm.remove("to_remove") is True
    assert wm.get("to_remove") is None
    assert wm.remove("not_there") is False


def test_search_by_tag():
    wm = WorkingMemory(capacity=10)
    wm.add("a", "first", tags=["important"])
    wm.add("b", "second", tags=["temp"])
    wm.add("c", "third", tags=["important"])
    results = wm.search_by_tag("important")
    assert len(results) == 2


def test_clear():
    wm = WorkingMemory(capacity=5)
    wm.add("x", 1)
    wm.add("y", 2)
    wm.clear()
    assert len(wm) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
