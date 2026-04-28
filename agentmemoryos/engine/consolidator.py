import time
from ..core.working_memory import WorkingMemory
from ..core.long_term_memory import LongTermMemory
from .forgetting import compute_retention, should_forget


class Consolidator:
    """
    Handles memory consolidation — moving things from working memory
    to long-term storage, and running forgetting passes on long-term memory.

    In neuroscience this roughly corresponds to what happens during sleep
    (memory replay and consolidation). Here we just run it periodically
    or on demand.

    Call consolidate() whenever you want to trigger a consolidation pass.
    Typically: end of session, or every N interactions.
    """

    def __init__(
        self,
        working: WorkingMemory,
        long_term: LongTermMemory,
        importance_threshold=0.4,
        forget_threshold=0.1,
    ):
        self.working = working
        self.long_term = long_term
        self.importance_threshold = importance_threshold
        self.forget_threshold = forget_threshold
        self._last_run = None
        self.stats = {
            "consolidated": 0,
            "forgotten": 0,
            "runs": 0,
        }

    def consolidate(self, session_id=None) -> dict:
        now = time.time()
        promoted = []
        forgotten = []

        # step 1: promote important working memory items to long-term
        for item in self.working.all_items():
            if item.importance >= self.importance_threshold:
                self.long_term.store_item(
                    key=item.key,
                    content=item.content,
                    tags=item.tags,
                    session_id=session_id,
                    importance=item.importance,
                )
                promoted.append(item.key)

        # step 2: run forgetting pass on long-term memory
        for lt_item in self.long_term.all_items():
            retention = compute_retention(
                importance=lt_item.importance,
                created_at=lt_item.created_at,
                last_accessed=lt_item.last_accessed,
                access_count=lt_item.access_count,
                now=now,
            )
            self.long_term.update_strength(lt_item.key, retention)

            if should_forget(retention, threshold=self.forget_threshold):
                self.long_term.forget(lt_item.key)
                forgotten.append(lt_item.key)

        self._last_run = now
        self.stats["consolidated"] += len(promoted)
        self.stats["forgotten"] += len(forgotten)
        self.stats["runs"] += 1

        return {
            "promoted": promoted,
            "forgotten": forgotten,
            "run_at": now,
        }

    def force_promote(self, key: str, session_id=None):
        """manually push a working memory item to long-term regardless of importance"""
        item = self.working.peek(key)
        if item is None:
            raise KeyError(f"Key '{key}' not in working memory")
        self.long_term.store_item(
            key=item.key,
            content=item.content,
            tags=item.tags,
            session_id=session_id,
            importance=item.importance,
        )
        return item
