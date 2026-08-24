"""In-memory per-sender rate limiting for the WhatsApp webhook (V4.0-F2).

Sliding-window counter keyed by the pseudonymized sender (never the raw
phone number). Disabled when ``limit_per_minute`` is 0 or negative.
Fail-open policy lives at the call site: a limiter error must never
block message processing. No PII in logs, keys or representations.
"""

from __future__ import annotations

from collections import deque
from threading import Lock
from typing import Callable
import time


MAX_TRACKED_SENDERS = 4096


class WhatsAppRateLimiter:
    """Thread-safe sliding-window rate limiter (messages per minute)."""

    def __init__(
        self,
        *,
        limit_per_minute: int,
        window_seconds: float = 60.0,
        clock: Callable[[], float] | None = None,
    ) -> None:
        if not isinstance(limit_per_minute, int) or isinstance(limit_per_minute, bool):
            raise ValueError("limit_per_minute must be an integer.")
        self._limit = max(limit_per_minute, 0)
        if window_seconds <= 0:
            raise ValueError("window_seconds must be positive.")
        self._window_seconds = window_seconds
        self._clock = clock or time.monotonic
        self._lock = Lock()
        self._hits: dict[str, deque] = {}

    @property
    def enabled(self) -> bool:
        return self._limit > 0

    def allow(self, sender_key: str) -> bool:
        """Return True when the pseudonymous sender may proceed."""
        if not self.enabled or not isinstance(sender_key, str) or not sender_key:
            return True
        now = self._clock()
        with self._lock:
            self._evict_stale_senders(now)
            hits = self._hits.get(sender_key)
            if hits is None:
                hits = deque()
                self._hits[sender_key] = hits
            self._prune(hits, now)
            if len(hits) >= self._limit:
                return False
            hits.append(now)
            return True

    def _prune(self, hits: deque, now: float) -> None:
        while hits and now - hits[0] >= self._window_seconds:
            hits.popleft()

    def _evict_stale_senders(self, now: float) -> None:
        """Bound memory usage when many distinct senders appear."""
        if len(self._hits) <= MAX_TRACKED_SENDERS:
            return
        stale = [
            key
            for key, hits in self._hits.items()
            if not hits or now - hits[-1] >= self._window_seconds
        ]
        for key in stale:
            del self._hits[key]

    def __repr__(self) -> str:
        return (
            f"WhatsAppRateLimiter(enabled={self.enabled}, "
            f"limit_per_minute={self._limit}, "
            f"window_seconds={self._window_seconds})"
        )
