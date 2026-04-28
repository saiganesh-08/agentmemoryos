import time
import json
from typing import Any, Optional
from dataclasses import dataclass, field, asdict


@dataclass
class LongTermItem:
    key: str
    content: Any
    importance: float
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0
    memory_strength: float = 1.0   # decays over time via forgetting engine
    tags: list = field(default_factory=list)
    session_id: Optional[str] = None

    def to_dict(self):
        d = asdict(self)
        # content might not be json serializable if someone shoves weird stuff in
        try:
            json.dumps(d["content"])
        except (TypeError, ValueError):
            d["content"] = str(d["content"])
        return d

    @classmethod
    def from_dict(cls, d: dict):
        return cls(**d)


class LongTermMemory:
    """
    Persistent memory store backed by SQLite (via the storage layer).
    Handles importance scoring, retrieval, and memory strength tracking.
    """

    def __init__(self, store, importance_scorer):
        self.store = store
        self.importance_scorer = importance_scorer
        self._cache = {}  # in-memory cache to avoid hitting sqlite on every read

    def store_item(self, key: str, content: Any, tags=None, session_id=None, importance=None) -> LongTermItem:
        if importance is None:
            importance = self.importance_scorer.score(content, tags or [])

        item = LongTermItem(
            key=key,
            content=content,
            importance=importance,
            tags=tags or [],
            session_id=session_id
        )
        self.store.save(key, item.to_dict())
        self._cache[key] = item
        return item

    def retrieve(self, key: str) -> Optional[LongTermItem]:
        if key in self._cache:
            item = self._cache[key]
            item.access_count += 1
            item.last_accessed = time.time()
            self.store.save(key, item.to_dict())
            return item

        raw = self.store.load(key)
        if raw is None:
            return None

        item = LongTermItem.from_dict(raw)
        item.access_count += 1
        item.last_accessed = time.time()
        self._cache[key] = item
        self.store.save(key, item.to_dict())
        return item

    def update_strength(self, key: str, new_strength: float):
        item = self.retrieve(key)
        if item:
            item.memory_strength = max(0.0, min(1.0, new_strength))
            self.store.save(key, item.to_dict())

    def forget(self, key: str):
        self._cache.pop(key, None)
        self.store.delete(key)

    def all_keys(self) -> list[str]:
        return self.store.list_keys()

    def all_items(self) -> list[LongTermItem]:
        items = []
        for key in self.all_keys():
            item = self.retrieve(key)
            if item:
                items.append(item)
        return items

    def search_by_tag(self, tag: str) -> list[LongTermItem]:
        # this is slow for large stores, should use indexed search eventually
        # fine for now
        return [i for i in self.all_items() if tag in i.tags]

    def top_by_importance(self, n=10) -> list[LongTermItem]:
        items = self.all_items()
        return sorted(items, key=lambda x: x.importance * x.memory_strength, reverse=True)[:n]

    def __len__(self):
        return len(self.all_keys())
