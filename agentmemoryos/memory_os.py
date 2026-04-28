import time
from typing import Any, Optional, Callable

from .core.working_memory import WorkingMemory
from .core.long_term_memory import LongTermMemory
from .core.episodic_memory import EpisodicMemory
from .engine.importance import ImportanceScorer
from .engine.consolidator import Consolidator
from .engine.forgetting import compute_retention, time_until_forgotten
from .retrieval.semantic import SemanticIndex
from .storage.sqlite_store import SQLiteStore


class MemoryOS:
    """
    The main interface. Orchestrates working memory, long-term memory,
    episodic memory, forgetting, consolidation, and semantic retrieval.

    Think of it as the memory manager for your AI agent.

    Quick start:
        mem = MemoryOS()
        sid = mem.start_session()

        mem.remember("user_name", "Alice", tags=["user_pref"])
        mem.remember("user_goal", "build a personal finance app", tags=["goal"])

        results = mem.recall("what does the user want to build")
        print(results)

        mem.end_session()
    """

    def __init__(
        self,
        db_path: str = "agentmemory.db",
        working_capacity: int = 20,
        consolidation_threshold: float = 0.4,
        auto_consolidate: bool = True,
        consolidate_every: int = 50,   # interactions
        embed_fn: Optional[Callable] = None,
    ):
        self.auto_consolidate = auto_consolidate
        self.consolidate_every = consolidate_every
        self._interaction_count = 0

        # init subsystems
        self._store = SQLiteStore(db_path)
        self._importance = ImportanceScorer()
        self.working = WorkingMemory(capacity=working_capacity)
        self.long_term = LongTermMemory(self._store, self._importance)
        self.episodic = EpisodicMemory()
        self.consolidator = Consolidator(
            self.working,
            self.long_term,
            importance_threshold=consolidation_threshold,
        )
        self._index = SemanticIndex(embed_fn=embed_fn)

        # rebuild semantic index from existing long-term memory
        self._rebuild_index()

    def _rebuild_index(self):
        for item in self.long_term.all_items():
            text = item.content if isinstance(item.content, str) else str(item.content)
            self._index.add(item.key, text)

    def _maybe_consolidate(self):
        self._interaction_count += 1
        if self.auto_consolidate and self._interaction_count % self.consolidate_every == 0:
            self.consolidate()

    # ── Core API ──────────────────────────────────────────────────────────────

    def remember(self, key: str, content: Any, tags=None, importance=None, session_id=None):
        """
        Store something in working memory.
        Important items get consolidated to long-term automatically.
        """
        tags = tags or []
        if importance is None:
            importance = self._importance.score(content, tags)

        item = self.working.add(key, content, importance=importance, tags=tags)

        # log to episodic
        sid = session_id or self.episodic.active_session_id
        self.episodic.log("remember", {"key": key, "importance": importance}, session_id=sid)

        # if importance is really high, push straight to long-term
        if importance >= 0.75:
            self.long_term.store_item(key, content, tags=tags, session_id=sid, importance=importance)
            text = content if isinstance(content, str) else str(content)
            self._index.add(key, text)

        self._maybe_consolidate()
        return item

    def recall(self, query: str, top_k=5, search_long_term=True) -> list[dict]:
        """
        Retrieve memories relevant to a query.
        Searches working memory first, then long-term via semantic search.
        """
        results = []

        # check working memory by key match first (fast path)
        wm_item = self.working.get(query)
        if wm_item:
            results.append({
                "key": wm_item.key,
                "content": wm_item.content,
                "source": "working",
                "score": 1.0,
                "importance": wm_item.importance,
            })

        # semantic search in long-term
        if search_long_term:
            hits = self._index.search(query, top_k=top_k)
            for key, score in hits:
                if key == query and results:
                    continue  # already got it from working memory
                lt_item = self.long_term.retrieve(key)
                if lt_item:
                    results.append({
                        "key": lt_item.key,
                        "content": lt_item.content,
                        "source": "long_term",
                        "score": round(score, 4),
                        "importance": lt_item.importance,
                        "memory_strength": lt_item.memory_strength,
                    })

        # sort by score desc
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def forget(self, key: str, from_all=True):
        """manually forget a memory"""
        self.working.remove(key)
        if from_all:
            self.long_term.forget(key)
            self._index.remove(key)

    def consolidate(self, session_id=None) -> dict:
        """
        Run a consolidation pass — promotes working memory to long-term
        and applies forgetting to stale long-term memories.
        """
        sid = session_id or self.episodic.active_session_id
        result = self.consolidator.consolidate(session_id=sid)

        # update semantic index with newly promoted items
        for key in result["promoted"]:
            lt_item = self.long_term.retrieve(key)
            if lt_item:
                text = lt_item.content if isinstance(lt_item.content, str) else str(lt_item.content)
                self._index.add(key, text)

        # remove forgotten items from index
        for key in result["forgotten"]:
            self._index.remove(key)

        return result

    # ── Session management ────────────────────────────────────────────────────

    def start_session(self, session_id=None, metadata=None) -> str:
        return self.episodic.start_session(session_id=session_id, metadata=metadata)

    def end_session(self, session_id=None, summary=None, consolidate=True):
        sid = session_id or self.episodic.active_session_id
        self.episodic.end_session(sid, summary=summary)
        if consolidate:
            self.consolidate(session_id=sid)

    # ── Introspection ─────────────────────────────────────────────────────────

    def memory_health(self) -> dict:
        """snapshot of the current memory state"""
        lt_items = self.long_term.all_items()
        avg_strength = (
            sum(i.memory_strength for i in lt_items) / len(lt_items)
            if lt_items else 0.0
        )
        at_risk = [
            i.key for i in lt_items if i.memory_strength < 0.3
        ]
        return {
            "working_memory": {
                "used": len(self.working),
                "capacity": self.working.capacity,
            },
            "long_term_memory": {
                "total_items": len(lt_items),
                "avg_strength": round(avg_strength, 3),
                "at_risk": at_risk,
            },
            "episodic": {
                "total_sessions": len(self.episodic),
                "active_session": self.episodic.active_session_id,
            },
            "semantic_index": {
                "indexed_items": len(self._index),
            },
            "consolidation_runs": self.consolidator.stats["runs"],
        }

    def retention_report(self) -> list[dict]:
        """how long until each long-term memory fades"""
        report = []
        now = time.time()
        for item in self.long_term.all_items():
            retention = compute_retention(
                item.importance, item.created_at, item.last_accessed, item.access_count, now
            )
            hours_left = time_until_forgotten(
                item.importance, item.created_at, item.last_accessed, item.access_count
            )
            report.append({
                "key": item.key,
                "retention": retention,
                "hours_until_forgotten": hours_left,
                "importance": item.importance,
            })
        return sorted(report, key=lambda x: x["retention"])

    def stats(self) -> dict:
        return {
            "consolidation": self.consolidator.stats,
            "interactions": self._interaction_count,
        }

    def close(self):
        self._store.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __repr__(self):
        return (
            f"MemoryOS("
            f"working={len(self.working)}/{self.working.capacity}, "
            f"long_term={len(self.long_term)}, "
            f"sessions={len(self.episodic)})"
        )
