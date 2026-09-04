"""Canonical background-pump tests: progress without new user turns."""

from __future__ import annotations

from pathlib import Path
import time

from core.async_task_scheduler import (
    AsyncTaskScheduler,
    InvalidApprovalError,
    JsonGoalTaskStore,
    GoalBudget,
    TaskOutcome,
    TaskStatus,
)
from core.background_goal_pump import BackgroundGoalPump


class RecordingExecutor:
    """Deterministic executor that also records every invocation."""

    def __init__(self, behavior):
        self._behavior = behavior
        self.calls: list[str] = []

    def __call__(self, task, resumable_payload):
        self.calls.append(task.task_id)
        return self._behavior(task, resumable_payload)


def _canonical_behavior(task, payload):
    if task.task_id == "C" and payload is None:
        return TaskOutcome.pause_for_approval(
            "¿Autorizas la operación C?",
            resumable_payload={"step": "c_pending"},
        )
    return TaskOutcome.succeed(f"{task.task_id}-result")


def _wait_until(predicate, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


def _canonical_tasks():
    return [
        {"task_id": "A", "description": "lectura segura"},
        {"task_id": "B", "description": "segunda lectura segura"},
        {
            "task_id": "C",
            "description": "requiere autorización",
            "requires_approval": True,
        },
        {"task_id": "D", "description": "tarea segura independiente"},
    ]


def _pump(scheduler, interval=0.05) -> BackgroundGoalPump:
    return BackgroundGoalPump(scheduler, interval_seconds=interval)


def test_canonical_goal_progresses_without_new_user_turns(tmp_path: Path):
    store = JsonGoalTaskStore(tmp_path / "task_scheduler")
    executor = RecordingExecutor(_canonical_behavior)
    scheduler = AsyncTaskScheduler(executor, store=store)
    goal_id = scheduler.submit_goal("objetivo canónico", _canonical_tasks())
    pump = _pump(scheduler)
    pump.start()
    try:
        reached = _wait_until(
            lambda: scheduler.task("A").status is TaskStatus.DONE
            and scheduler.task("B").status is TaskStatus.DONE
            and scheduler.task("D").status is TaskStatus.DONE
            and scheduler.task("C").status is TaskStatus.WAITING_APPROVAL
        )
    finally:
        pump.stop()

    assert reached, [
        (task_id, scheduler.task(task_id).status) for task_id in "ABCD"
    ]
    assert scheduler.goal_status(goal_id) is TaskStatus.WAITING_APPROVAL
    assert set(executor.calls) == {"A", "B", "C", "D"}
    assert executor.calls.count("C") == 1

    approval = scheduler.pending_approvals(goal_id)[0]
    resumed = scheduler.approve(approval.confirmation_id)

    assert resumed == "C"
    assert scheduler.task("C").status is TaskStatus.DONE
    assert scheduler.task("C").result == "C-result"
    assert scheduler.goal_status(goal_id) is TaskStatus.DONE
    assert executor.calls.count("A") == 1
    assert executor.calls.count("B") == 1
    assert executor.calls.count("D") == 1
    assert executor.calls.count("C") == 2


def test_pump_executes_submitted_goal_without_manual_run_ready(tmp_path: Path):
    store = JsonGoalTaskStore(tmp_path / "task_scheduler")
    executor = RecordingExecutor(_canonical_behavior)
    scheduler = AsyncTaskScheduler(executor, store=store)
    scheduler.submit_goal(
        "objetivo de fondo",
        [{"task_id": "A", "description": "tarea segura"}],
    )
    pump = _pump(scheduler)

    pump.start()
    try:
        progressed = _wait_until(lambda: executor.calls == ["A"])
    finally:
        pump.stop()

    assert progressed
    assert scheduler.task("A").status is TaskStatus.DONE


def test_budget_limits_iterations_and_stops_the_goal(tmp_path: Path):
    store = JsonGoalTaskStore(tmp_path / "task_scheduler")
    executor = RecordingExecutor(_canonical_behavior)
    scheduler = AsyncTaskScheduler(executor, store=store)
    goal_id = scheduler.submit_goal(
        "objetivo con presupuesto",
        [
            {"task_id": "A", "description": "tarea uno"},
            {"task_id": "B", "description": "tarea dos"},
        ],
        budget=GoalBudget(max_task_executions=1),
    )
    pump = _pump(scheduler)

    pump.start()
    try:
        reached = _wait_until(
            lambda: scheduler.goal_status(goal_id) is TaskStatus.BLOCKED
        )
    finally:
        pump.stop()

    assert reached
    assert executor.calls == ["A"]
    assert scheduler.task("A").status is TaskStatus.DONE
    assert scheduler.task("B").status is TaskStatus.BLOCKED
    assert scheduler.task("B").error == "presupuesto agotado"
    summary = scheduler.goal_summary(goal_id)
    assert summary["budget_exhausted"] is True
    assert pump.run_once() is False


def test_duration_deadline_blocks_pending_work(tmp_path: Path):
    store = JsonGoalTaskStore(tmp_path / "task_scheduler")
    executor = RecordingExecutor(_canonical_behavior)
    scheduler = AsyncTaskScheduler(executor, store=store)
    goal_id = scheduler.submit_goal(
        "objetivo con plazo",
        [
            {"task_id": "A", "description": "tarea uno"},
            {"task_id": "B", "description": "tarea dos"},
        ],
        budget=GoalBudget(max_duration_seconds=0.05),
    )

    assert scheduler.run_next_ready() == "A"
    time.sleep(0.2)
    scheduler.run_next_ready()

    assert scheduler.task("A").status is TaskStatus.DONE
    assert scheduler.task("B").status is TaskStatus.BLOCKED
    assert scheduler.task("B").error == "presupuesto agotado"
    assert scheduler.goal_status(goal_id) is TaskStatus.BLOCKED


def test_restart_recovers_progress_without_repeating_done_work(tmp_path: Path):
    store = JsonGoalTaskStore(tmp_path / "task_scheduler")
    first_executor = RecordingExecutor(_canonical_behavior)
    first_scheduler = AsyncTaskScheduler(first_executor, store=store)
    goal_id = first_scheduler.submit_goal(
        "objetivo persistente",
        _canonical_tasks(),
    )
    pump = _pump(first_scheduler)
    pump.start()
    reached = _wait_until(
        lambda: first_scheduler.task("A").status is TaskStatus.DONE
        and first_scheduler.task("B").status is TaskStatus.DONE
        and first_scheduler.task("D").status is TaskStatus.DONE
        and first_scheduler.task("C").status is TaskStatus.WAITING_APPROVAL
    )
    pump.stop()
    assert reached
    approval = first_scheduler.pending_approvals(goal_id)[0]
    confirmation_id = approval.confirmation_id

    second_executor = RecordingExecutor(_canonical_behavior)
    second_scheduler = AsyncTaskScheduler(second_executor, store=store)
    second_pump = _pump(second_scheduler)
    second_pump.start()
    try:
        restored = _wait_until(
            lambda: second_scheduler.task("A").status is TaskStatus.DONE
            and second_scheduler.task("B").status is TaskStatus.DONE
            and second_scheduler.task("D").status is TaskStatus.DONE
            and second_scheduler.task("C").status is TaskStatus.WAITING_APPROVAL
        )
        assert restored
        assert second_executor.calls == [], second_executor.calls
        resumed = second_scheduler.approve(confirmation_id)
    finally:
        second_pump.stop()

    assert resumed == "C"
    assert second_scheduler.task("C").status is TaskStatus.DONE
    assert second_scheduler.goal_status(goal_id) is TaskStatus.DONE
    assert second_executor.calls == ["C"]


def test_cancellation_stops_new_work_and_keeps_approvals_pending(tmp_path: Path):
    store = JsonGoalTaskStore(tmp_path / "task_scheduler")
    executor = RecordingExecutor(_canonical_behavior)
    scheduler = AsyncTaskScheduler(executor, store=store)
    goal_id = scheduler.submit_goal(
        "objetivo cancelable",
        [
            {"task_id": "A", "description": "tarea uno"},
            {"task_id": "C", "description": "requiere autorización"},
        ],
    )
    pump = _pump(scheduler)
    pump.start()
    reached = _wait_until(
        lambda: scheduler.task("C").status is TaskStatus.WAITING_APPROVAL
    )
    assert reached

    scheduler.cancel_goal(goal_id)
    pump.stop()

    assert scheduler.goal_status(goal_id) is TaskStatus.CANCELLED
    assert scheduler.task("A").status is TaskStatus.DONE
    assert scheduler.task("C").status is TaskStatus.CANCELLED
    assert pump.run_once() is False
    pending = scheduler.pending_approvals(goal_id)
    assert len(pending) == 1
    try:
        scheduler.approve(pending[0].confirmation_id)
        raise AssertionError("expected InvalidApprovalError after cancellation")
    except InvalidApprovalError:
        pass


def test_pump_start_is_idempotent_and_stop_is_clean(tmp_path: Path):
    store = JsonGoalTaskStore(tmp_path / "task_scheduler")
    executor = RecordingExecutor(_canonical_behavior)
    scheduler = AsyncTaskScheduler(executor, store=store)
    scheduler.submit_goal(
        "objetivo breve",
        [{"task_id": "A", "description": "tarea segura"}],
    )
    pump = _pump(scheduler)

    pump.start()
    pump.start()
    progressed = _wait_until(lambda: executor.calls == ["A"])
    assert progressed
    threads = [t for t in __import__("threading").enumerate() if t.name == "atlas-goal-pump"]
    assert len(threads) == 1

    pump.stop()
    assert not pump.running
