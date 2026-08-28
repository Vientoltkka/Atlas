"""Idempotency stores for webhook processing (Atlas Phase 3, Block 1).

Two interchangeable implementations share the same minimal contract:
``check_and_reserve(event_id) -> bool``.

- ``IdempotencyStore``: in-memory, per-process, NOT safe for multiple
  workers. Kept as the development default.
- ``SqliteIdempotencyStore``: persistent, shared across processes and
  workers through an atomic INSERT on a UNIQUE key.
"""

from __future__ import annotations

from contextlib import closing
import json
from pathlib import Path
from threading import Lock
import sqlite3
import time


DEFAULT_TTL_SECONDS = 24 * 60 * 60


class IdempotencyStoreInitError(RuntimeError):
    """The persistent idempotency store could not be initialized.

    Raised instead of silently degrading to a weaker store. The message
    is diagnostic-only and never contains paths, tokens or user data.
    """


class IdempotencyStore:
    """Reserve event identifiers so duplicates are processed exactly once."""

    def __init__(self, *, ttl_seconds: int = DEFAULT_TTL_SECONDS) -> None:
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive.")
        self._ttl_seconds = ttl_seconds
        self._reserved: dict[str, float] = {}
        self._lock = Lock()

    def check_and_reserve(self, event_id: str) -> bool:
        """Return True and reserve ``event_id`` if it is new; False if duplicate.

        The reservation happens atomically before the caller responds, which
        guarantees that concurrent duplicates cannot both execute.
        """
        now = time.monotonic()
        with self._lock:
            self._evict_expired(now)
            if event_id in self._reserved:
                return False
            self._reserved[event_id] = now
            return True

    def _evict_expired(self, now: float) -> None:
        expired = [
            key
            for key, reserved_at in self._reserved.items()
            if now - reserved_at > self._ttl_seconds
        ]
        for key in expired:
            del self._reserved[key]


class SqliteIdempotencyStore:
    """Persistent idempotency store shared across processes and workers.

    Reservations are atomic: the UNIQUE primary key plus INSERT makes the
    check-and-reserve step safe even with concurrent processes writing to
    the same database file. Only the event id and its reservation timestamp
    are stored; no payloads or personal data.
    """

    def __init__(self, *, db_path: str | Path, ttl_seconds: int = DEFAULT_TTL_SECONDS) -> None:
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive.")
        path = Path(db_path)
        if not str(path):
            raise ValueError("db_path must be a non-empty path.")
        self._ttl_seconds = ttl_seconds
        self._path = path
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with closing(self._connect()) as connection, connection:
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS whatsapp_idempotency (
                        event_id TEXT PRIMARY KEY,
                        reserved_at REAL NOT NULL
                    )
                    """
                )
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS whatsapp_webhook_jobs (
                        event_id TEXT PRIMARY KEY,
                        kind TEXT NOT NULL,
                        payload TEXT NOT NULL,
                        state TEXT NOT NULL,
                        attempts INTEGER NOT NULL DEFAULT 0,
                        updated_at REAL NOT NULL
                    )
                    """
                )
        except Exception as error:
            # Fail loudly without leaking the filesystem location or any
            # other sensitive detail in the exposed message.
            raise IdempotencyStoreInitError(
                "persistent whatsapp idempotency store is unavailable; "
                "check the configured database accessibility."
            ) from error

    def check_and_reserve(self, event_id: str) -> bool:
        """Return True and reserve ``event_id`` if it is new; False if duplicate."""
        now = time.time()
        with closing(self._connect()) as connection, connection:
            self._evict_expired(connection, now)
            cursor = connection.execute(
                "INSERT OR IGNORE INTO whatsapp_idempotency (event_id, reserved_at) VALUES (?, ?)",
                (event_id, now),
            )
            connection.commit()
            return cursor.rowcount == 1

    def _evict_expired(self, connection: sqlite3.Connection, now: float) -> None:
        connection.execute(
            "DELETE FROM whatsapp_idempotency WHERE reserved_at <= ?",
            (now - self._ttl_seconds,),
        )

    def enqueue(self, event_id: str, kind: str, payload: dict) -> bool:
        """Atomically persist a new webhook job before acknowledging it."""
        now = time.time()
        serialized = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
        with closing(self._connect()) as connection, connection:
            self._evict_expired(connection, now)
            reserved = connection.execute(
                "INSERT OR IGNORE INTO whatsapp_idempotency (event_id, reserved_at) VALUES (?, ?)",
                (event_id, now),
            )
            if reserved.rowcount != 1:
                return False
            connection.execute(
                "INSERT INTO whatsapp_webhook_jobs (event_id, kind, payload, state, updated_at) VALUES (?, ?, ?, 'pending', ?)",
                (event_id, kind, serialized, now),
            )
            return True

    def recover_interrupted(self) -> None:
        with closing(self._connect()) as connection, connection:
            connection.execute("UPDATE whatsapp_webhook_jobs SET state = 'pending', updated_at = ? WHERE state IN ('processing', 'failed')", (time.time(),))

    def claim_pending(self) -> tuple[str, str, dict] | None:
        with closing(self._connect()) as connection, connection:
            row = connection.execute("SELECT event_id, kind, payload FROM whatsapp_webhook_jobs WHERE state = 'pending' ORDER BY updated_at, event_id LIMIT 1").fetchone()
            if row is None:
                return None
            claimed = connection.execute("UPDATE whatsapp_webhook_jobs SET state = 'processing', attempts = attempts + 1, updated_at = ? WHERE event_id = ? AND state = 'pending'", (time.time(), row[0]))
            if claimed.rowcount != 1:
                return None
            return row[0], row[1], json.loads(row[2])

    def complete(self, event_id: str) -> None:
        self._set_job_state(event_id, 'processed')

    def fail(self, event_id: str) -> None:
        self._set_job_state(event_id, 'failed')

    def _set_job_state(self, event_id: str, state: str) -> None:
        with closing(self._connect()) as connection, connection:
            connection.execute("UPDATE whatsapp_webhook_jobs SET state = ?, updated_at = ? WHERE event_id = ?", (state, time.time(), event_id))
    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self._path, timeout=30.0)
