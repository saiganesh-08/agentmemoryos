import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class MemoryItem:
    key: str
    content: Any
    timestamp: float = field(default_factory=time.time)
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)
    importance: float = 0.5
    tags: list = field(default_factory=list)

    def touch(self):
        self.access_count += 1
        self.last_accessed = time.time()


class WorkingMemory:
    """
    Short-term memory buffer with limited capacity.
    Uses LRU eviction when full — least recently used gets pushed out.

    Inspired loosely by cognitive science models of working memory
    (Miller's 7 +/- 2, but we default to 20 for agents since they
    don't have the same bottleneck humans do)
    """

    def __init__(self, capacity=20):
        self.capacity = capacity
        self._store: OrderedDict[str, MemoryItem] = OrderedDict()

    def add(self, key: str, content: Any, importance=0.5, tags=None) -> MemoryItem:
        if key in self._store:
            self._store.move_to_end(key)
            item = self._store[key]
            item.content = content
            item.touch()
            return item

        if len(self._store) >= self.capacity:
            evicted_key, _ = self._store.popitem(last=False)
            # TODO: maybe emit an event here so consolidator can catch it

        item = MemoryItem(key=key, content=content, importance=importance, tags=tags or [])
        self._store[key] = item
        return item

    def get(self, key: str) -> Optional[MemoryItem]:
        if key not in self._store:
            return None
        self._store.move_to_end(key)
        item = self._store[key]
        item.touch()
        return item

    def peek(self, key: str) -> Optional[MemoryItem]:
        """get without updating access stats - useful for consolidator"""
        return self._store.get(key)

    def remove(self, key: str) -> bool:
        if key in self._store:
            del self._store[key]
            return True
        return False

    def all_items(self) -> list[MemoryItem]:
        return list(self._store.values())

    def search_by_tag(self, tag: str) -> list[MemoryItem]:
        return [item for item in self._store.values() if tag in item.tags]

    def clear(self):
        self._store.clear()

    def __len__(self):
        return len(self._store)

    def __contains__(self, key):
        return key in self._store

    def __repr__(self):
        return f"WorkingMemory(capacity={self.capacity}, used={len(self)})"
