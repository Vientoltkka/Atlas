"""Lightweight in-memory observability for the WhatsApp channel (F5.1).

Counters only: no message content, tokens, URLs, media ids or PII are
stored. Thread-safe for concurrent webhook usage. Specific to the
WhatsApp channel; unrelated to core/execution_metrics.py.
"""

from __future__ import annotations

import threading
from typing import Any


MESSAGES_RECEIVED = "messages_received"
MESSAGES_DUPLICATED = "messages_duplicated"
MESSAGES_FAILED = "messages_failed"
AUDIO_RECEIVED = "audio_received"
VOICE_REPLIES = "voice_replies"
CHANNEL_ERRORS = "channel_errors"

STATUS_SENT = "status_sent"
STATUS_DELIVERED = "status_delivered"
STATUS_READ = "status_read"
STATUS_FAILED = "status_failed"
RATE_LIMITED = "rate_limited"

_EVENTS = (
    MESSAGES_RECEIVED,
    MESSAGES_DUPLICATED,
    MESSAGES_FAILED,
    AUDIO_RECEIVED,
    VOICE_REPLIES,
    CHANNEL_ERRORS,
    STATUS_SENT,
    STATUS_DELIVERED,
    STATUS_READ,
    STATUS_FAILED,
    RATE_LIMITED,
)

_STATUS_EVENT_BY_CODE = {
    "sent": STATUS_SENT,
    "delivered": STATUS_DELIVERED,
    "read": STATUS_READ,
    "failed": STATUS_FAILED,
}


def status_event(status: str) -> str | None:
    """Map a WhatsApp status code to its counter event; None if unknown."""
    return _STATUS_EVENT_BY_CODE.get(status.strip().casefold())


class WhatsAppMetricsRecorder:
    """Thread-safe in-memory counters for WhatsApp channel events."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: dict[str, int] = {event: 0 for event in _EVENTS}

    def record(self, event: str) -> None:
        with self._lock:
            self._counters[event] = self._counters.get(event, 0) + 1

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return dict(self._counters)

    def value(self, event: str) -> int:
        with self._lock:
            return self._counters.get(event, 0)

    def restore(self, counters: Any) -> None:
        """Merge previously persisted counters into this recorder.

        Only known events with valid non-negative integers are applied;
        anything else is ignored. Never raises for malformed input.
        """
        if not isinstance(counters, dict):
            return
        with self._lock:
            for event in _EVENTS:
                value = counters.get(event)
                if isinstance(value, int) and not isinstance(value, bool) and value > 0:
                    self._counters[event] += value

    def __repr__(self) -> str:
        return f"WhatsAppMetricsRecorder(counters={self.snapshot()})"


def safe_record(recorder: Any, event: str) -> None:
    """Record an event, swallowing any recorder failure so that the
    message pipeline is never affected by observability."""
    if recorder is None:
        return
    try:
        recorder.record(event)
    except Exception:
        pass
