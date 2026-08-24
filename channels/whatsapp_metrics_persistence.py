"""Optional local persistence for WhatsApp channel metrics (V4.0-F3).

Stores only aggregated event counters as JSON using atomic writes
(tmp file + replace). No PII, message ids, timestamps or content are
ever written. All failures are fail-safe: persistence problems must
never affect webhook processing. Disabled entirely when no path is
configured.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from threading import Lock

from channels.whatsapp_metrics import WhatsAppMetricsRecorder


logger = logging.getLogger(__name__)

FLUSH_THRESHOLD_EVENTS = 50


class WhatsAppMetricsPersistence:
    """Atomic JSON persistence for a WhatsAppMetricsRecorder."""

    def __init__(self, *, recorder: WhatsAppMetricsRecorder, path: str | Path) -> None:
        self._recorder = recorder
        self._path = Path(path)
        self._pending = 0
        self._lock = Lock()

    @property
    def path(self) -> Path:
        return self._path

    def load_existing(self) -> None:
        """Restore previously persisted counters; fail-safe on any error."""
        try:
            if not self._path.exists():
                return
            raw = json.loads(self._path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                self._recorder.restore(raw)
        except Exception as error:
            logger.warning(
                "whatsapp metrics load failed | starting from zero | type=%s",
                type(error).__name__,
            )

    def record_and_maybe_flush(self, event: str) -> None:
        """Record one event and flush when the pending threshold is reached.

        Never raises: observability failures are isolated from the
        pipeline by contract.
        """
        try:
            self._recorder.record(event)
            with self._lock:
                self._pending += 1
                due = self._pending >= FLUSH_THRESHOLD_EVENTS
                if due:
                    self._pending = 0
            if due:
                self.flush()
        except Exception:
            logger.warning("whatsapp metrics record/flush failed | type=%s", type(event).__name__)

    def flush(self) -> bool:
        """Write the current snapshot atomically. Returns True on success."""
        snapshot = self._recorder.snapshot()
        tmp_path = None
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            payload = json.dumps(snapshot, ensure_ascii=False, sort_keys=True)
            tmp_path = self._path.with_suffix(self._path.suffix + ".tmp")
            tmp_path.write_text(payload, encoding="utf-8")
            tmp_path.replace(self._path)
            with self._lock:
                self._pending = 0
            return True
        except Exception as error:
            logger.warning(
                "whatsapp metrics flush failed | counters remain in memory | type=%s",
                type(error).__name__,
            )
            if tmp_path is not None:
                try:
                    tmp_path.unlink(missing_ok=True)
                except OSError:
                    pass
            return False

    def record(self, event: str) -> None:
        """Recorder-compatible alias for :meth:`record_and_maybe_flush`."""
        self.record_and_maybe_flush(event)

    def snapshot(self) -> dict[str, int]:
        return self._recorder.snapshot()

    def value(self, event: str) -> int:
        return self._recorder.value(event)

    def __repr__(self) -> str:
        return (
            f"WhatsAppMetricsPersistence(path={self._path.name}, "
            f"counters={self._recorder.snapshot()})"
        )
