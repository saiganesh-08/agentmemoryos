import sqlite3
import json
import os
from typing import Optional


class SQLiteStore:
    """
    Simple key-value store backed by SQLite.
    Values are JSON serialized.

    Not the fastest thing in the world but it's reliable and portable.
    If you need speed, swap this out for Redis or similar.
    """

    def __init__(self, db_path: str = "agentmemory.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._init_db()

    def _init_db(self):
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS memory (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at REAL DEFAULT (unixepoch('now'))
            )
        """)
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_updated ON memory(updated_at)")
        self._conn.commit()

    def save(self, key: str, value: dict):
        serialized = json.dumps(value)
        self._conn.execute(
            "INSERT OR REPLACE INTO memory (key, value, updated_at) VALUES (?, ?, unixepoch('now'))",
            (key, serialized)
        )
        self._conn.commit()

    def load(self, key: str) -> Optional[dict]:
        cur = self._conn.execute("SELECT value FROM memory WHERE key = ?", (key,))
        row = cur.fetchone()
        if row is None:
            return None
        return json.loads(row[0])

    def delete(self, key: str):
        self._conn.execute("DELETE FROM memory WHERE key = ?", (key,))
        self._conn.commit()

    def list_keys(self) -> list[str]:
        cur = self._conn.execute("SELECT key FROM memory")
        return [row[0] for row in cur.fetchall()]

    def count(self) -> int:
        cur = self._conn.execute("SELECT COUNT(*) FROM memory")
        return cur.fetchone()[0]

    def close(self):
        self._conn.close()

    def __del__(self):
        try:
            self._conn.close()
        except Exception:
            pass
