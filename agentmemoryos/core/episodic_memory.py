import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class Episode:
    session_id: str
    events: list = field(default_factory=list)
    started_at: float = field(default_factory=time.time)
    ended_at: Optional[float] = None
    summary: Optional[str] = None
    metadata: dict = field(default_factory=dict)

    def add_event(self, event_type: str, content: Any, ts=None):
        self.events.append({
            "type": event_type,
            "content": content,
            "ts": ts or time.time()
        })

    def close(self, summary=None):
        self.ended_at = time.time()
        self.summary = summary

    @property
    def duration(self):
        end = self.ended_at or time.time()
        return end - self.started_at

    @property
    def is_active(self):
        return self.ended_at is None


class EpisodicMemory:
    """
    Tracks sequences of events within sessions.
    Useful for understanding context across a conversation or task.

    Each "episode" is one session/conversation/task run.
    Events within an episode are ordered and timestamped.
    """

    def __init__(self, max_episodes=500):
        self.max_episodes = max_episodes
        self._episodes: dict[str, Episode] = {}
        self._active_session: Optional[str] = None

    def start_session(self, session_id=None, metadata=None) -> str:
        sid = session_id or str(uuid.uuid4())[:8]
        ep = Episode(session_id=sid, metadata=metadata or {})
        self._episodes[sid] = ep
        self._active_session = sid

        # prune old episodes if we're over the limit
        if len(self._episodes) > self.max_episodes:
            oldest = sorted(self._episodes.values(), key=lambda e: e.started_at)
            for old in oldest[:len(self._episodes) - self.max_episodes]:
                del self._episodes[old.session_id]

        return sid

    def end_session(self, session_id=None, summary=None):
        sid = session_id or self._active_session
        if sid and sid in self._episodes:
            self._episodes[sid].close(summary=summary)
        if self._active_session == sid:
            self._active_session = None

    def log(self, event_type: str, content: Any, session_id=None):
        sid = session_id or self._active_session
        if sid is None:
            # auto-start a session if none active
            sid = self.start_session()
        if sid not in self._episodes:
            raise ValueError(f"Session {sid} not found")
        self._episodes[sid].add_event(event_type, content)

    def get_session(self, session_id: str) -> Optional[Episode]:
        return self._episodes.get(session_id)

    def get_active(self) -> Optional[Episode]:
        if self._active_session:
            return self._episodes.get(self._active_session)
        return None

    def recent_sessions(self, n=10) -> list[Episode]:
        sorted_eps = sorted(self._episodes.values(), key=lambda e: e.started_at, reverse=True)
        return sorted_eps[:n]

    def all_events_for_type(self, event_type: str) -> list[dict]:
        results = []
        for ep in self._episodes.values():
            for ev in ep.events:
                if ev["type"] == event_type:
                    results.append({**ev, "session_id": ep.session_id})
        return sorted(results, key=lambda x: x["ts"])

    @property
    def active_session_id(self):
        return self._active_session

    def __len__(self):
        return len(self._episodes)
