from __future__ import annotations

import sqlite3
from pathlib import Path

MAX_PERMANENT_FAILURES = 3

_SCHEMA = """
CREATE TABLE IF NOT EXISTS failures (
    filename TEXT PRIMARY KEY,
    count INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS processed (
    sha256 TEXT PRIMARY KEY,
    dest_path TEXT NOT NULL,
    source_deleted INTEGER NOT NULL DEFAULT 1
);
"""

# Added after the table first shipped; CREATE TABLE IF NOT EXISTS leaves an
# existing table alone, so the column is added separately.
_MIGRATIONS = [
    "ALTER TABLE processed ADD COLUMN source_deleted INTEGER NOT NULL DEFAULT 1",
]


class StateStore:
    """Tracks permanent (per-file) failure counts across restarts, so a
    consistently broken scan gets quarantined after a few attempts instead of
    being retried forever on every startup sweep. Transient infrastructure
    failures (Ollama/WebDAV unreachable) should NOT go through this store.

    Also remembers the content hash of every successfully filed scan and
    where it went, so the same file dropped into the inbox again is
    recognized as a duplicate."""

    def __init__(self, db_path: str):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.executescript(_SCHEMA)
        for statement in _MIGRATIONS:
            try:
                self._conn.execute(statement)
            except sqlite3.OperationalError:
                pass  # already applied
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

    def find_processed(self, sha256: str) -> str | None:
        """Destination path a scan with this content hash was filed to, if any."""
        row = self._conn.execute(
            "SELECT dest_path FROM processed WHERE sha256 = ?", (sha256,)
        ).fetchone()
        return row[0] if row else None

    def record_processed(self, sha256: str, dest_path: str, source_deleted: bool = True) -> None:
        """Remember where a scan was filed. Recorded with source_deleted=False
        right after the upload and before the scan is removed from the
        inbox, so that a connection lost in between is recognized on the
        next attempt (see source_pending_delete) instead of filing a second
        copy."""
        with self._conn:
            self._conn.execute(
                "INSERT INTO processed (sha256, dest_path, source_deleted) VALUES (?, ?, ?) "
                "ON CONFLICT(sha256) DO UPDATE SET "
                "dest_path = excluded.dest_path, source_deleted = excluded.source_deleted",
                (sha256, dest_path, int(source_deleted)),
            )

    def mark_source_deleted(self, sha256: str) -> None:
        with self._conn:
            self._conn.execute("UPDATE processed SET source_deleted = 1 WHERE sha256 = ?", (sha256,))

    def source_pending_delete(self, sha256: str) -> bool:
        """True if this content was filed but its scan is still known to be
        in the inbox."""
        row = self._conn.execute(
            "SELECT source_deleted FROM processed WHERE sha256 = ?", (sha256,)
        ).fetchone()
        return bool(row) and not row[0]
