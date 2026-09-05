"""DEBUG/TEST-only fail-once hook: provokes exactly one retryable failure.

Red/green contract for the E2E offline retry drill:
hook ON -> first read_file execution fails retryable (TRANSIENT_ERROR),
scheduler retries through its own backoff, the real executor then completes
the task; approvals and sensitive writes are never touched.
"""

from __future__ import annotations

import os

import pytest

from core.async_task_scheduler import (
    AsyncTaskScheduler,
    GoalBudget,
    TaskOutcome,
    TaskStatus,
)
from core.background_goal_pump import BackgroundGoalPump
from core.debug_fail_once_hook import (
    DEBUG_FAIL_ONCE_ENV,
    DebugFailOnceExecutor,
    reset_debug_fail_once_state,
    wrap_debug_fail_once,
)


class FakeClock:
    def __init__(self, start: float = 1_000_000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class BaseExecutor:
    """Stands in for the real ToolTaskExecutor; records every invocation."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, bool]] = []

    def __call__(self, task, resumable_payload):
        self.calls.append((task.task_id, resumable_payload is not None))
        if task.initial_payload.get("tool") == "read_file":
            return TaskOutcome.succeed(f"contenido de {task.task_id}")
        if task.initial_payload.get("tool") == "write_file":
            if resumable_payload is None:
                return TaskOutcome.pause_for_approval(
                    f"¿Autorizas la escritura {task.task_id}?",
                    resumable_payload={"tool": "write_file", "arguments": {}},
                )
            return TaskOutcome.succeed("escrito")
        return TaskOutcome.succeed("ok")


@pytest.fixture(autouse=True)
def _clean_hook_state(monkeypatch):
    monkeypatch.delenv(DEBUG_FAIL_ONCE_ENV, raising=False)
    reset_debug_fail_once_state()
    yield
    reset_debug_fail_once_state()


def _scheduler(
    base: BaseExecutor, clock: FakeClock, backoff: tuple[float, float]
) -> tuple[AsyncTaskScheduler, str]:
    scheduler = AsyncTaskScheduler(
        wrap_debug_fail_once(base),
        clock=clock,
        retry_backoff_seconds=backoff,  # existing backoff, no extra sleeps
    )
    goal_id = scheduler.submit_goal(
        "lee un archivo y escribe un resumen",
        [
            {
                "task_id": "read",
                "description": "leer archivo",
                "payload": {"tool": "read_file", "arguments": {"path": "x.txt"}},
            },
            {
                "task_id": "write",
                "description": "escribir resumen",
                "dependencies": ["read"],
                "requires_approval": True,
                "payload": {"tool": "write_file", "arguments": {"path": "y.txt"}},
            },
            {
                "task_id": "aux",
                "description": "tarea independiente",
                "payload": {"tool": "list_dir", "arguments": {}},
            },
        ],
        budget=GoalBudget(max_task_executions=20, max_duration_seconds=600.0),
    )
    return scheduler, goal_id


def test_hook_on_first_read_file_fails_retryable_then_real_executor_completes():
    os.environ[DEBUG_FAIL_ONCE_ENV] = "1"
    base = BaseExecutor()
    clock = FakeClock()
    scheduler, goal_id = _scheduler(base, clock, backoff=(5.0, 10.0))

    # First execution: the hook injects exactly one retryable failure.
    assert scheduler.run_next_ready() == "read"
    read = scheduler.task("read")
    assert read.status is TaskStatus.RESUMABLE
    assert read.retry_count == 1
    assert read.max_retries >= 1
    assert read.metadata["wait_reason"] == "retry_backoff"
    assert read.metadata["last_error"].startswith("TRANSIENT_ERROR")
    assert scheduler.goal_status(goal_id) is not TaskStatus.FAILED

    # The independent task keeps progressing during the backoff window.
    assert scheduler.run_next_ready() == "aux"
    assert scheduler.task("aux").status is TaskStatus.DONE
    assert scheduler.run_next_ready() is None

    # Retry once the declarative backoff elapses: the real executor runs.
    clock.advance(6.0)
    assert scheduler.run_next_ready() == "read"
    read = scheduler.task("read")
    assert read.status is TaskStatus.DONE
    assert read.result == "contenido de read"
    assert read.retry_count == 1

    # The dependent write pauses for approval; the hook never touched it.
    assert scheduler.run_next_ready() == "write"
    write = scheduler.task("write")
    assert write.status is TaskStatus.WAITING_APPROVAL
    assert write.pending_confirmation_id is not None

    # No loop: further rounds execute nothing new.
    executed = scheduler.goal(goal_id).executed_count
    assert scheduler.run_next_ready() is None
    assert scheduler.goal(goal_id).executed_count == executed

    # The hook injected the first failure without reaching the base executor:
    # the base saw only aux, the real read retry and the write approval ask.
    assert sorted(base.calls) == [
        ("aux", False),
        ("read", False),
        ("write", False),
    ]


def test_hook_on_through_background_pump_completes_without_loop():
    os.environ[DEBUG_FAIL_ONCE_ENV] = "1"
    base = BaseExecutor()
    clock = FakeClock()
    scheduler, goal_id = _scheduler(base, clock, backoff=(0.0, 0.0))
    pump = BackgroundGoalPump(scheduler, interval_seconds=60.0)

    processed = 0
    while pump.run_once():
        processed += 1
        assert processed < 10  # no retry loop

    assert processed >= 4  # inject + retry + write approval + aux
    read = scheduler.task("read")
    assert read.status is TaskStatus.DONE
    assert read.result == "contenido de read"
    assert read.retry_count == 1
    assert scheduler.task("aux").status is TaskStatus.DONE
    assert scheduler.task("write").status is TaskStatus.WAITING_APPROVAL
    assert scheduler.goal_status(goal_id) is TaskStatus.WAITING_APPROVAL


def test_hook_off_is_executor_pass_through():
    os.environ[DEBUG_FAIL_ONCE_ENV] = "0"
    base = BaseExecutor()
    clock = FakeClock()
    scheduler, goal_id = _scheduler(base, clock, backoff=(5.0, 10.0))

    scheduler.run_ready()

    read = scheduler.task("read")
    assert read.status is TaskStatus.DONE
    assert read.result == "contenido de read"
    assert read.retry_count == 0
    assert sorted(base.calls) == [
        ("aux", False),
        ("read", False),
        ("write", False),
    ]
    assert scheduler.goal_status(goal_id) is TaskStatus.WAITING_APPROVAL


def test_hook_never_injects_approvals_or_sensitive_writes_and_only_once_per_task():
    os.environ[DEBUG_FAIL_ONCE_ENV] = "1"
    base = BaseExecutor()
    clock = FakeClock()
    scheduler, goal_id = _scheduler(base, clock, backoff=(0.0, 0.0))

    scheduler.run_ready()

    write = scheduler.task("write")
    assert write.status is TaskStatus.WAITING_APPROVAL
    assert write.retry_count == 0
    assert write.pending_confirmation_id is not None
    read = scheduler.task("read")
    assert read.retry_count == 1
    assert read.status is TaskStatus.DONE
    assert scheduler.goal_status(goal_id) is TaskStatus.WAITING_APPROVAL


def test_wrap_returns_same_executor_when_disabled():
    base = BaseExecutor()
    os.environ[DEBUG_FAIL_ONCE_ENV] = "0"
    assert wrap_debug_fail_once(base) is base
    os.environ[DEBUG_FAIL_ONCE_ENV] = "1"
    wrapped = wrap_debug_fail_once(base)
    assert isinstance(wrapped, DebugFailOnceExecutor)
