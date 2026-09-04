"""Canonical tests: bounded retry, deterministic verification, generalized wait."""

from __future__ import annotations

from pathlib import Path
import time

from core.async_task_scheduler import (
    AsyncTaskScheduler,
    GoalBudget,
    JsonGoalTaskStore,
    TaskOutcome,
    TaskStatus,
)
from core.background_goal_pump import BackgroundGoalPump


class FakeClock:
    """Deterministic wall clock for backoff and deadline tests."""

    def __init__(self, start: float = 1_000_000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class RecordingExecutor:
    """Deterministic executor that records every invocation."""

    def __init__(self, behavior):
        self._behavior = behavior
        self.calls: list[str] = []

    def __call__(self, task, resumable_payload):
        self.calls.append(task.task_id)
        return self._behavior(task, resumable_payload, len(self.calls))


def _goal(clock: FakeClock, executor, **kwargs) -> tuple[AsyncTaskScheduler, str]:
    scheduler = AsyncTaskScheduler(
        executor,
        clock=clock,
        retry_backoff_seconds=(5.0, 10.0),
        **kwargs,
    )
    goal_id = scheduler.submit_goal(
        "objetivo con reintentos",
        [
            {"task_id": "A", "description": "lectura segura"},
            {
                "task_id": "B",
                "description": "falla transitoriamente",
                "dependencies": ["A"],
            },
            {"task_id": "C", "description": "tarea independiente"},
        ],
    )
    return scheduler, goal_id


def test_transient_failure_retries_with_backoff_and_others_progress():
    clock = FakeClock()

    def behavior(task, payload, call_number):
        if task.task_id == "B":
            if task.retry_count == 0:
                return TaskOutcome.fail("TRANSIENT_ERROR: backend timeout")
            return TaskOutcome.succeed("B-done")
        return TaskOutcome.succeed(f"{task.task_id}-result")

    scheduler, goal_id = _goal(clock, RecordingExecutor(behavior))

    scheduler.run_ready()

    assert scheduler.task("A").status is TaskStatus.DONE
    assert scheduler.task("C").status is TaskStatus.DONE
    assert scheduler.task("B").status is TaskStatus.RESUMABLE
    assert scheduler.task("B").retry_count == 1
    assert scheduler.task("B").metadata["wait_reason"] == "retry_backoff"
    assert scheduler.task("B").metadata["last_error"] == "TRANSIENT_ERROR: backend timeout"
    assert scheduler.goal_status(goal_id) is not TaskStatus.FAILED

    clock.advance(6.0)
    scheduler.run_ready()

    assert scheduler.task("B").status is TaskStatus.DONE
    assert scheduler.task("B").result == "B-done"
    assert scheduler.goal_finished(goal_id)
    # canonical counters: A=1, B=2, C=1 executions
    assert scheduler.goal(goal_id).executed_count == 4


def test_retry_backoff_delays_resume_until_due():
    clock = FakeClock()

    def behavior(task, payload, call_number):
        if task.task_id == "B" and task.retry_count == 0:
            return TaskOutcome.fail("timeout: backend no respondió")
        return TaskOutcome.succeed(f"{task.task_id}-done")

    executor = RecordingExecutor(behavior)
    scheduler, _ = _goal(clock, executor)

    scheduler.run_ready()

    assert executor.calls == ["A", "C", "B"]
    clock.advance(1.0)
    scheduler.run_ready()
    assert executor.calls == ["A", "C", "B"]
    clock.advance(1.0)
    scheduler.run_ready()
    assert executor.calls == ["A", "C", "B"]
    clock.advance(4.0)
    scheduler.run_ready()
    assert executor.calls == ["A", "C", "B", "B"]
    assert scheduler.task("B").status is TaskStatus.DONE


def test_permanent_error_never_retries_and_blocks_dependents():
    clock = FakeClock()

    def behavior(task, payload, call_number):
        if task.task_id == "B":
            return TaskOutcome.fail("invalid_argument: ruta fuera de scope")
        return TaskOutcome.succeed(f"{task.task_id}-result")

    scheduler = AsyncTaskScheduler(RecordingExecutor(behavior), clock=clock)
    goal_id = scheduler.submit_goal(
        "error permanente",
        [
            {"task_id": "B", "description": "falla permanente"},
            {"task_id": "D", "description": "depende de B", "dependencies": ["B"]},
        ],
    )

    scheduler.run_ready()

    assert scheduler.task("B").status is TaskStatus.FAILED
    assert scheduler.task("D").status is TaskStatus.BLOCKED
    assert scheduler.task("B").retry_count == 0


def test_explicit_metadata_overrides_error_classification():
    clock = FakeClock()

    def behavior(task, payload, call_number):
        if task.task_id == "B":
            return TaskOutcome.fail(
                "provider unavailable",
                metadata={"retryable": False},
            )
        return TaskOutcome.succeed(f"{task.task_id}-result")

    scheduler = AsyncTaskScheduler(RecordingExecutor(behavior), clock=clock)
    goal_id = scheduler.submit_goal(
        "sin reintento declarado",
        [{"task_id": "B", "description": "fallo"}],
    )

    scheduler.run_ready()

    assert scheduler.task("B").status is TaskStatus.FAILED
    assert scheduler.goal(goal_id).tasks["B"].retry_count == 0


def test_invalid_result_is_not_done_and_retries_bounded():
    clock = FakeClock()

    def behavior(task, payload, call_number):
        if task.task_id == "T":
            return TaskOutcome.succeed("   ")
        return TaskOutcome.succeed(f"{task.task_id}-result")

    scheduler = AsyncTaskScheduler(RecordingExecutor(behavior), clock=clock)
    goal_id = scheduler.submit_goal(
        "resultado invÃ¡lido",
        [
            {
                "task_id": "T",
                "description": "transform con salida vacÃ­a",
                "payload": {"kind": "transform", "instruction": "resume: {input}"},
                "max_retries": 1,
            },
        ],
    )

    scheduler.run_ready()

    task = scheduler.task("T")
    assert task.status is TaskStatus.RESUMABLE
    assert task.retry_count == 1

    clock.advance(6.0)
    scheduler.run_ready()

    task = scheduler.task("T")
    assert task.status is TaskStatus.FAILED
    assert task.error == "verifier rejected the task result"
    assert scheduler.goal(goal_id).executed_count == 2


def test_wait_resource_pauses_only_that_task_and_resumes_when_due():
    clock = FakeClock()
    waited: list[bool] = []

    def behavior(task, payload, call_number):
        if task.task_id == "B":
            if not waited:
                waited.append(True)
                return TaskOutcome.wait(
                    "resource_unavailable", resume_at=clock() + 30.0
                )
            return TaskOutcome.succeed("B-after-resource")
        return TaskOutcome.succeed(f"{task.task_id}-result")

    executor = RecordingExecutor(behavior)
    scheduler, goal_id = _goal(clock, executor)

    scheduler.run_ready()

    assert scheduler.task("B").status is TaskStatus.RESUMABLE
    assert scheduler.task("B").metadata["wait_reason"] == "resource_unavailable"
    assert scheduler.task("B").retry_count == 0
    assert scheduler.task("A").status is TaskStatus.DONE
    assert scheduler.task("C").status is TaskStatus.DONE

    clock.advance(31.0)
    scheduler.run_ready()

    assert scheduler.task("B").status is TaskStatus.DONE
    assert scheduler.goal_finished(goal_id)
    assert executor.calls.count("B") == 2


def test_restart_preserves_retry_wait_without_repeating_done_work(tmp_path: Path):
    clock = FakeClock()

    def behavior(task, payload, call_number):
        if task.task_id == "B":
            if task.retry_count == 0:
                return TaskOutcome.fail("TEMPORARY_UNAVAILABLE: servicio saturado")
            return TaskOutcome.succeed("B-recovered")
        return TaskOutcome.succeed(f"{task.task_id}-result")

    store = JsonGoalTaskStore(tmp_path / "task_scheduler")
    first_executor = RecordingExecutor(behavior)
    first_scheduler = AsyncTaskScheduler(
        first_executor,
        store=store,
        clock=clock,
        retry_backoff_seconds=(50.0, 100.0),
    )
    goal_id = first_scheduler.submit_goal(
        "objetivo persistente con reintento",
        [
            {"task_id": "A", "description": "lectura segura"},
            {
                "task_id": "B",
                "description": "falla transitoriamente",
                "dependencies": ["A"],
            },
            {"task_id": "C", "description": "tarea independiente"},
        ],
    )
    first_scheduler.run_ready()
    resumed_at = first_scheduler.task("B").resume_at
    assert resumed_at is not None

    restart_clock = FakeClock(start=clock.now + 10.0)
    second_executor = RecordingExecutor(behavior)
    second_scheduler = AsyncTaskScheduler(
        second_executor,
        store=store,
        clock=restart_clock,
        retry_backoff_seconds=(50.0, 100.0),
    )
    second_scheduler.load_goal(goal_id)

    assert second_executor.calls == []
    assert second_scheduler.task("A").status is TaskStatus.DONE
    assert second_scheduler.task("C").status is TaskStatus.DONE
    assert second_scheduler.task("B").status is TaskStatus.RESUMABLE

    second_scheduler.run_next_ready()
    assert second_executor.calls == []

    restart_clock.advance(45.0)
    processed = second_scheduler.run_next_ready()
    assert processed == "B"
    assert second_executor.calls == ["B"]
    assert second_scheduler.task("B").status is TaskStatus.DONE
    assert second_scheduler.goal_finished(goal_id)


def test_retry_respects_task_execution_budget():
    clock = FakeClock()

    def behavior(task, payload, call_number):
        if task.task_id == "B":
            return TaskOutcome.fail("TRANSIENT_ERROR: backend timeout")
        return TaskOutcome.succeed(f"{task.task_id}-result")

    scheduler = AsyncTaskScheduler(RecordingExecutor(behavior), clock=clock)
    goal_id = scheduler.submit_goal(
        "presupuesto agotado antes del reintento",
        [
            {"task_id": "A", "description": "lectura segura"},
            {
                "task_id": "B",
                "description": "falla transitoriamente",
                "dependencies": ["A"],
            },
        ],
        budget=GoalBudget(max_task_executions=2),
    )

    scheduler.run_ready()

    assert scheduler.task("A").status is TaskStatus.DONE
    assert scheduler.task("B").status is TaskStatus.BLOCKED
    assert scheduler.task("B").error == "presupuesto agotado"
    assert scheduler.goal_status(goal_id) is TaskStatus.BLOCKED
    assert scheduler.goal(goal_id).executed_count == 2


def test_cancel_during_retry_wait_prevents_resume():
    clock = FakeClock()

    def behavior(task, payload, call_number):
        if task.task_id == "B":
            return TaskOutcome.fail("TRANSIENT_ERROR: backend timeout")
        return TaskOutcome.succeed(f"{task.task_id}-result")

    scheduler = AsyncTaskScheduler(RecordingExecutor(behavior), clock=clock)
    goal_id = scheduler.submit_goal(
        "cancelado en espera",
        [
            {"task_id": "A", "description": "lectura segura"},
            {
                "task_id": "B",
                "description": "falla transitoriamente",
                "dependencies": ["A"],
            },
        ],
    )

    scheduler.run_ready()
    assert scheduler.task("B").status is TaskStatus.RESUMABLE

    scheduler.cancel_goal(goal_id)
    clock.advance(60.0)
    processed = scheduler.run_next_ready()

    assert processed is None
    assert scheduler.task("B").status is TaskStatus.CANCELLED
    assert scheduler.goal_status(goal_id) is TaskStatus.CANCELLED


def test_pump_progresses_independent_work_while_one_task_waits_retry(tmp_path: Path):
    def behavior(task, payload, call_number):
        if task.task_id == "B":
            if task.retry_count == 0:
                return TaskOutcome.fail("TRANSIENT_ERROR: backend timeout")
            return TaskOutcome.succeed("B-done")
        return TaskOutcome.succeed(f"{task.task_id}-result")

    store = JsonGoalTaskStore(tmp_path / "task_scheduler")
    scheduler = AsyncTaskScheduler(
        RecordingExecutor(behavior),
        store=store,
        retry_backoff_seconds=(0.1, 0.2),
    )
    goal_id = scheduler.submit_goal(
        "objetivo con reintento de fondo",
        [
            {"task_id": "A", "description": "lectura segura"},
            {
                "task_id": "B",
                "description": "falla transitoriamente",
                "dependencies": ["A"],
            },
            {"task_id": "C", "description": "tarea independiente"},
        ],
    )
    pump = BackgroundGoalPump(scheduler, interval_seconds=0.02)
    pump.start()
    try:
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if scheduler.goal_finished(goal_id):
                break
            time.sleep(0.02)
    finally:
        pump.stop()

    assert scheduler.goal_finished(goal_id)
    assert scheduler.task("B").status is TaskStatus.DONE
    assert scheduler.goal(goal_id).executed_count == 4


def _wait_until(predicate, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()
