"""DEBUG/TEST-ONLY hook to provoke exactly one retryable scheduler failure.

This module is NOT part of the business logic. It exists exclusively to
reproduce a real E2E retry (scheduler + RetryableErrorClassifier + backoff +
GoalBudget + BackgroundGoalPump) without models and without network.

Activation (default OFF, per session only, never persisted to .env):

    ATLAS_DEBUG_RETRY_ONCE=1

Behaviour when ON: the FIRST execution of a read_file task (never approvals,
never sensitive writes) fails with the existing declarative retryable code
``TRANSIENT_ERROR``; the second execution delegates to the real executor.
At most one injection per (task_id, goal_id), enforced both in-memory and
via ``retry_count`` so a restart cannot double-inject.
"""

from __future__ import annotations

import os
from typing import Any, Callable

from core.async_task_scheduler import Task, TaskOutcome

DEBUG_FAIL_ONCE_ENV = "ATLAS_DEBUG_RETRY_ONCE"
_DEBUG_TARGET_TOOL = "read_file"
_INJECTED_KEYS: set[tuple[str, str]] = set()


def debug_fail_once_enabled() -> bool:
    """True only when the session flag is exactly ``1``; default OFF."""
    return os.environ.get(DEBUG_FAIL_ONCE_ENV, "").strip() == "1"


def reset_debug_fail_once_state() -> None:
    """DEBUG/TEST helper: forget which tasks were already injected."""
    _INJECTED_KEYS.clear()


class DebugFailOnceExecutor:
    """DEBUG/TEST wrapper around the real task executor.

    Every non-injected call is delegated untouched; unknown attributes
    (e.g. ``bind_result_lookup``) are forwarded to the base executor too.
    """

    def __init__(self, base_executor: Callable[[Task, dict[str, Any] | None], TaskOutcome]) -> None:
        self._base_executor = base_executor

    def __getattr__(self, name: str) -> Any:
        base = self.__dict__.get("_base_executor")
        if base is None:
            raise AttributeError(name)
        return getattr(base, name)

    def __call__(self, task: Task, resumable_payload: dict[str, Any] | None) -> TaskOutcome:
        if self._should_inject(task, resumable_payload):
            _INJECTED_KEYS.add((task.task_id, task.goal_id))
            return TaskOutcome.fail(
                "TRANSIENT_ERROR: fallo transitorio inyectado por el hook "
                "DEBUG/TEST fail-once (no es un error real)",
                metadata={
                    "error_code": "TRANSIENT_ERROR",
                    "debug_fail_once": True,
                },
            )
        return self._base_executor(task, resumable_payload)

    def _should_inject(self, task: Task, resumable_payload: dict[str, Any] | None) -> bool:
        if not debug_fail_once_enabled():
            return False
        if (task.task_id, task.goal_id) in _INJECTED_KEYS:
            return False
        if task.requires_approval:
            return False
        if resumable_payload:  # approval/wait resume path: never inject
            return False
        if task.retry_count > 0:  # one injection per task even across restarts
            return False
        payload = task.initial_payload or {}
        return payload.get("tool") == _DEBUG_TARGET_TOOL


def wrap_debug_fail_once(
    base_executor: Callable[[Task, dict[str, Any] | None], TaskOutcome],
) -> Callable[[Task, dict[str, Any] | None], TaskOutcome]:
    """Return the DEBUG wrapper only when the session flag is ON.

    When OFF the base executor is returned unchanged, so the wiring is
    behaviourally identical to the previous code path.
    """
    if not debug_fail_once_enabled():
        return base_executor
    return DebugFailOnceExecutor(base_executor)
