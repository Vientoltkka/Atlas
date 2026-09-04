"""Cooperative background pump that progresses READY goals without user turns.

Reuses the existing AsyncTaskScheduler: instead of re-running run_ready()
(which holds the scheduler lock for the whole loop), the pump calls
run_next_ready() repeatedly, processing exactly one unit per wake-up and
releasing the lock in between. No worker pool, no external queue.
"""

from __future__ import annotations

import logging
import threading

from core.async_task_scheduler import AsyncTaskScheduler

logger = logging.getLogger(__name__)

_DEFAULT_INTERVAL_SECONDS = 2.0
_PROGRESS_INTERVAL_SECONDS = 0.05
_THREAD_NAME = "atlas-goal-pump"


class BackgroundGoalPump:
    """Single cooperative daemon thread that drives autonomous goal progress."""

    def __init__(
        self,
        scheduler: AsyncTaskScheduler,
        *,
        interval_seconds: float = _DEFAULT_INTERVAL_SECONDS,
        progress_interval_seconds: float = _PROGRESS_INTERVAL_SECONDS,
    ) -> None:
        self._scheduler = scheduler
        self._interval_seconds = max(0.05, float(interval_seconds))
        self._progress_interval_seconds = max(0.01, float(progress_interval_seconds))
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._start_lock = threading.Lock()
        self._consecutive_errors = 0

    # -- lifecycle ---------------------------------------------------------------

    def start(self) -> None:
        """Start the pump once; recovers persisted goals before looping."""
        with self._start_lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop_event.clear()
            self._recover_persisted_goals()
            self._thread = threading.Thread(
                target=self._run_loop,
                name=_THREAD_NAME,
                daemon=True,
            )
            self._thread.start()

    def stop(self, timeout: float = 8.0) -> None:
        """Signal the pump to stop and wait for a clean shutdown."""
        self._stop_event.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=timeout)
        self._thread = None

    @property
    def running(self) -> bool:
        thread = self._thread
        return thread is not None and thread.is_alive()

    # -- pumping -----------------------------------------------------------------

    def run_once(self) -> bool:
        """Process one READY/RESUMABLE unit; True when work was executed."""
        processed = self._scheduler.run_next_ready()
        return processed is not None

    def _run_loop(self) -> None:
        """Wake periodically; never busy-loop and never die on one task error."""
        while not self._stop_event.is_set():
            progressed = False
            try:
                progressed = self.run_once()
                self._consecutive_errors = 0
            except Exception:  # noqa: BLE001 - the pump must survive any task
                self._consecutive_errors += 1
                logger.exception(
                    "background goal pump iteration failed "
                    "(consecutive=%d)",
                    self._consecutive_errors,
                )
            wait = (
                self._progress_interval_seconds
                if progressed
                else self._interval_seconds
            )
            self._stop_event.wait(wait)

    def _recover_persisted_goals(self) -> None:
        """Reload persisted goals after a restart; DONE work is never repeated."""
        try:
            goal_ids = self._scheduler.persisted_goal_ids()
        except Exception:  # noqa: BLE001 - recovery must never block startup
            logger.exception("background goal pump recovery scan failed")
            return
        for goal_id in goal_ids:
            try:
                self._scheduler.load_goal(goal_id)
            except Exception:  # noqa: BLE001 - one bad goal must not stop startup
                logger.warning(
                    "background goal pump could not restore goal %s", goal_id
                )
