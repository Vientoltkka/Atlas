"""In-memory idempotency store for webhook processing (Atlas Phase 2).

LIMITATION: this implementation is per-process and NOT safe for multiple
uvicorn workers or replicas. Phase 2 must run with workers=1. In Phase 3 the
implementation can be swapped for a shared store (SQLite/Redis) without
changing this contract.
"""

from __future__ import annotations

from threading import Lock
import time


DEFAULT_TTL_SECONDS = 24 * 60 * 60


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
