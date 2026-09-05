"""Canonical tests for the non-blocking async task scheduler."""

from __future__ import annotations

from pathlib import Path

from core.async_task_scheduler import (
    AsyncTaskScheduler,
    InvalidApprovalError,
    JsonGoalTaskStore,
    TaskOutcome,
    TaskStatus,
)


class RecordingExecutor:
    """Deterministic executor that also records every invocation."""

    def __init__(self, behavior):
        self._behavior = behavior
        self.calls: list[str] = []

    def __call__(self, task, resumable_payload):
        self.calls.append(task.task_id)
        return self._behavior(task, resumable_payload)


def _behavior(task, payload):
    if task.task_id == "B":
        if payload is None:
            return TaskOutcome.pause_for_approval(
                "Â¿Autorizas la operaciÃ³n B?",
                resumable_payload={"step": "send_pending"},
            )
        return TaskOutcome.succeed("B-done-with-payload")
    return TaskOutcome.succeed(f"{task.task_id}-result")


def _scheduler_with_canonical_tasks(**kwargs) -> tuple[AsyncTaskScheduler, str]:
    scheduler = AsyncTaskScheduler(_behavior, **kwargs)
    goal_id = scheduler.submit_goal(
        "objetivo canÃ³nico",
        [
            {"task_id": "A", "description": "lectura segura"},
            {"task_id": "B", "description": "requiere autorizaciÃ³n"},
            {"task_id": "C", "description": "otra operaciÃ³n segura"},
        ],
    )
    return scheduler, goal_id


def test_waiting_approval_does_not_block_other_tasks():
    scheduler, goal_id = _scheduler_with_canonical_tasks()

    scheduler.run_ready()

    assert scheduler.task("A").status is TaskStatus.DONE
    assert scheduler.task("B").status is TaskStatus.WAITING_APPROVAL
    assert scheduler.task("C").status is TaskStatus.DONE
    assert scheduler.goal_status(goal_id) is TaskStatus.WAITING_APPROVAL
    assert not scheduler.goal_finished(goal_id)


def test_approval_resumes_only_the_paused_task():
    scheduler, goal_id = _scheduler_with_canonical_tasks()
    scheduler.run_ready()
    approval = scheduler.pending_approvals(goal_id)[0]

    resumed = scheduler.approve(approval.confirmation_id)

    assert resumed == "B"
    assert scheduler.task("B").status is TaskStatus.DONE
    assert scheduler.task("B").result == "B-done-with-payload"
    assert scheduler.task("B").resumable_payload == {"step": "send_pending"}
    assert scheduler.task("A").status is TaskStatus.DONE
    assert scheduler.task("C").status is TaskStatus.DONE
    assert scheduler.goal_status(goal_id) is TaskStatus.DONE
    assert scheduler.goal_finished(goal_id)


def test_denial_blocks_only_that_task():
    scheduler, goal_id = _scheduler_with_canonical_tasks()
    scheduler.run_ready()
    approval = scheduler.pending_approvals(goal_id)[0]

    denied = scheduler.deny(approval.confirmation_id)

    assert denied == "B"
    assert scheduler.task("B").status is TaskStatus.BLOCKED
    assert scheduler.goal_status(goal_id) is TaskStatus.DONE
    assert scheduler.goal_finished(goal_id)


def test_approval_token_is_single_use_and_bound_to_one_task():
    scheduler, _ = _scheduler_with_canonical_tasks()
    scheduler.run_ready()
    approval = scheduler.pending_approvals()[0]

    scheduler.approve(approval.confirmation_id)

    try:
        scheduler.approve(approval.confirmation_id)
        raise AssertionError("expected InvalidApprovalError")
    except InvalidApprovalError:
        pass


def test_dependencies_are_respected():
    scheduler = AsyncTaskScheduler(_behavior)
    scheduler.submit_goal(
        "con dependencias",
        [
            {"task_id": "A", "description": "primero"},
            {"task_id": "B", "description": "requiere autorizaciÃ³n"},
            {"task_id": "C", "description": "depende de A y B", "dependencies": ["A", "B"]},
        ],
    )

    scheduler.run_ready()

    assert scheduler.task("A").status is TaskStatus.DONE
    assert scheduler.task("B").status is TaskStatus.WAITING_APPROVAL
    assert scheduler.task("C").status is TaskStatus.PENDING

    approval = scheduler.pending_approvals()[0]
    scheduler.approve(approval.confirmation_id)

    assert scheduler.task("C").status is TaskStatus.DONE


def test_denied_dependency_blocks_dependent_task_without_retry():
    scheduler = AsyncTaskScheduler(_behavior)
    scheduler.submit_goal(
        "dependencia denegada",
        [
            {"task_id": "B", "description": "requiere autorizaciÃ³n"},
            {"task_id": "D", "description": "depende de B", "dependencies": ["B"]},
        ],
    )

    scheduler.run_ready()
    approval = scheduler.pending_approvals()[0]
    scheduler.deny(approval.confirmation_id)

    scheduler.run_ready()

    assert scheduler.task("B").status is TaskStatus.BLOCKED
    assert scheduler.task("D").status is TaskStatus.BLOCKED


def test_verifier_rejects_invalid_results():
    executor = RecordingExecutor(lambda task, payload: TaskOutcome.succeed("bad"))
    scheduler = AsyncTaskScheduler(executor, verifier=lambda task, result: result == "ok")
    scheduler.submit_goal(
        "verificado",
        [{"task_id": "A", "description": "tarea", "max_retries": 0}],
    )

    scheduler.run_ready()

    assert scheduler.task("A").status is TaskStatus.FAILED
    assert scheduler.task("A").error == "verifier rejected the task result"


def test_persistence_rebuilds_pending_state_and_resumes_without_repetition(tmp_path: Path):
    store = JsonGoalTaskStore(tmp_path / "task_scheduler")
    executor = RecordingExecutor(_behavior)
    scheduler = AsyncTaskScheduler(executor, store=store)
    goal_id = scheduler.submit_goal(
        "objetivo persistente",
        [
            {"task_id": "A", "description": "lectura segura"},
            {"task_id": "B", "description": "requiere autorizaciÃ³n"},
            {"task_id": "C", "description": "otra operaciÃ³n segura"},
        ],
    )

    scheduler.run_ready()
    approval = scheduler.pending_approvals(goal_id)[0]

    restored_scheduler = AsyncTaskScheduler(_behavior, store=store)
    restored_scheduler.load_goal(goal_id)

    assert restored_scheduler.task("A").status is TaskStatus.DONE
    assert restored_scheduler.task("C").status is TaskStatus.DONE
    assert restored_scheduler.task("B").status is TaskStatus.WAITING_APPROVAL

    restored = restored_scheduler.approve(approval.confirmation_id)

    assert restored == "B"
    assert restored_scheduler.task("B").status is TaskStatus.DONE
    assert restored_scheduler.goal_status(goal_id) is TaskStatus.DONE
    assert executor.calls == ["A", "B", "C"], executor.calls



def test_approval_is_goal_scoped_when_goals_reuse_task_ids():
    """An approval token must resolve its task inside its own goal.

    Goals reuse canonical task ids, so a global task-id lookup would bind
    one goal's token to another goal's task and reject the approval.
    """

    class PauseThenSucceedExecutor:
        def __init__(self):
            self.resumed_goals: list[str] = []

        def __call__(self, task, resumable_payload):
            if resumable_payload is None:
                return TaskOutcome.pause_for_approval(
                    "¿Autorizas la operación?",
                    resumable_payload={"resume": task.goal_id},
                )
            self.resumed_goals.append(task.goal_id)
            return TaskOutcome.succeed("ok")

    executor = PauseThenSucceedExecutor()
    scheduler = AsyncTaskScheduler(executor)
    first = scheduler.submit_goal(
        "objetivo uno",
        [{"task_id": "write_target", "description": "guardar uno"}],
    )
    second = scheduler.submit_goal(
        "objetivo dos",
        [{"task_id": "write_target", "description": "guardar dos"}],
    )
    scheduler.run_ready()
    second_token = scheduler.pending_approvals(second)[0].confirmation_id

    resumed = scheduler.approve(second_token)

    assert resumed == "write_target"
    assert executor.resumed_goals == [second]
    assert scheduler.goal(second).tasks["write_target"].status is TaskStatus.DONE
    assert (
        scheduler.goal(first).tasks["write_target"].status
        is TaskStatus.WAITING_APPROVAL
    )
