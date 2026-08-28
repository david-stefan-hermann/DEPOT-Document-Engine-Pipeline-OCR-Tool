from __future__ import annotations

import sqlite3
from pathlib import Path

MAX_PERMANENT_FAILURES = 3

_SCHEMA = """
CREATE TABLE IF NOT EXISTS failures (
    filename TEXT PRIMARY KEY,
    count INTEGER NOT NULL DEFAULT 0
);
"""


class StateStore:
    """Tracks permanent (per-file) failure counts across restarts, so a
    consistently broken scan gets quarantined after a few attempts instead of
    being retried forever on every startup sweep. Transient infrastructure
    failures (Ollama/WebDAV unreachable) should NOT go through this store."""

    def __init__(self, db_path: str):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute(_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def increment_failure(self, filename: str) -> int:
        with self._conn:
            self._conn.execute(
                "INSERT INTO failures (filename, count) VALUES (?, 1) "
                "ON CONFLICT(filename) DO UPDATE SET count = count + 1",
                (filename,),
            )
            row = self._conn.execute(
                "SELECT count FROM failures WHERE filename = ?", (filename,)
            ).fetchone()
        return row[0]

    def reset(self, filename: str) -> None:
        with self._conn:
            self._conn.execute("DELETE FROM failures WHERE filename = ?", (filename,))

    def should_quarantine(self, filename: str) -> bool:
        row = self._conn.execute(
            "SELECT count FROM failures WHERE filename = ?", (filename,)
        ).fetchone()
        return bool(row) and row[0] >= MAX_PERMANENT_FAILURES
