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

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self._path, timeout=30.0)
