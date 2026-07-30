from __future__ import annotations

import json
from datetime import datetime, timezone
from threading import Barrier, Thread

import pytest

from core.concurrent_step_executor import ConcurrentStepResult, ExecutionBatchResult
from core.execution_session_persistence import (
    ExecutionRecoveryPolicy,
    ExecutionRecoveryService,
    ExecutionSessionSnapshot,
    ExecutionSnapshotCorruptedError,
    FileExecutionSessionRepository,
    RecoveryDecisionType,
    UnsupportedExecutionSnapshotVersion,
    snapshot_from_dict,
    snapshot_to_dict,
)
from core.execution_plan_executor import PlanExecutionResult, PlanExecutionStatus
from core.execution_plan_validator import PlanValidationResult, plan_signature
from core.execution_supervisor import (
    ExecutionSessionAlreadyExistsError,
    ExecutionState,
    ExecutionSupervisor,
    StepExecutionState,
)
from core.planner import ExecutionPlan, ExecutionStep
from core.structured_execution import StructuredExecutionCoordinator
from core.structured_plan_replanner import ReplanReason, ReplanRecord


def _plan(*, recovery_safe: bool = False, requires_confirmation: bool = False) -> ExecutionPlan:
    second_tool = "write_file" if requires_confirmation else "read_file"
    return ExecutionPlan(
        goal="persisted execution",
        ordered_steps=(
            ExecutionStep(
                "step_1",
                "read",
                "read_file",
                idempotent=recovery_safe,
                recovery_safe=recovery_safe,
                side_effect_free=recovery_safe,
            ),
            ExecutionStep(
                "step_2",
                "write" if requires_confirmation else "read again",
                second_tool,
                ("step_1",),
                idempotent=recovery_safe,
                recovery_safe=recovery_safe,
                side_effect_free=recovery_safe,
            ),
        ),
        estimated_steps=2,
        required_tools=("read_file", second_tool),
        detected_risks=("write",) if requires_confirmation else (),
        requires_confirmation=requires_confirmation,
    )


def _running_snapshot(*, recovery_safe: bool = False) -> ExecutionSessionSnapshot:
    supervisor = ExecutionSupervisor()
    session = supervisor.start(_plan(recovery_safe=recovery_safe))
    supervisor.mark_running(session.session_id, current_step="step_1")
    supervisor.mark_step_started(session.session_id, "step_1")
    return ExecutionSessionSnapshot.from_session(
        supervisor.get_session(session.session_id),
        recovery_metadata={"token": "secret-value", "note": "safe"},
    )


def test_snapshot_serialization_round_trip_rebuilds_types() -> None:
    snapshot = _running_snapshot(recovery_safe=True)

    loaded = snapshot_from_dict(snapshot_to_dict(snapshot))

    assert loaded.session_id == snapshot.session_id
    assert loaded.state is ExecutionState.RUNNING
    assert loaded.step_states["step_1"].state is StepExecutionState.RUNNING
    assert loaded.step_states["step_1"].started_at is not None
    assert loaded.step_states["step_1"].attempt_count == 1
    assert loaded.step_states["step_1"].max_attempts == 1
    assert loaded.active_plan.ordered_steps[0].recovery_safe is True
    assert loaded.recovery_metadata == {"note": "safe"}


def test_snapshot_reader_defaults_new_supervision_fields_for_legacy_payload() -> None:
    payload = snapshot_to_dict(_running_snapshot())
    step_payload = payload["step_states"]["step_1"]
    for key in (
        "started_at",
        "finished_at",
        "attempt_count",
        "max_attempts",
        "is_critical",
    ):
        step_payload.pop(key)

    loaded = snapshot_from_dict(payload)
    step = loaded.step_states["step_1"]

    assert step.started_at is None
    assert step.finished_at is None
    assert step.attempt_count == 0
    assert step.max_attempts == 1
    assert step.is_critical is False


def test_replanning_record_round_trip_and_legacy_defaults() -> None:
    original = _plan()
    replacement = ExecutionPlan(
        goal="replacement",
        ordered_steps=(ExecutionStep("step_1", "alternative", "read_file"),),
        estimated_steps=1,
        required_tools=("read_file",),
        detected_risks=(),
        requires_confirmation=False,
    )
    supervisor = ExecutionSupervisor()
    session = supervisor.start(original)
    supervisor.mark_running(session.session_id)
    supervisor.mark_failed(session.session_id, "failed", current_step="step_1")
    supervisor.mark_replanning(
        session.session_id,
        attempt_number=1,
        current_step="step_1",
    )
    supervisor.record_replan(
        session.session_id,
        ReplanRecord(
            attempt_number=1,
            previous_plan=original,
            revised_plan=replacement,
            reason=ReplanReason.RECOVERABLE_FAILURE,
            failed_step="step_1",
            error="failed",
            created_at=datetime.now(timezone.utc),
            replacement_step_ids=("step_1",),
            validation_status="valid",
            recovery_result="pending",
        ),
    )
    payload = snapshot_to_dict(
        ExecutionSessionSnapshot.from_session(
            supervisor.get_session(session.session_id)
        )
    )

    loaded = snapshot_from_dict(payload)
    record = loaded.replan_history[0]

    assert record.replacement_step_ids == ("step_1",)
    assert record.validation_status == "valid"
    assert record.recovery_result == "pending"

    legacy_record = payload["replan_history"][0]
    legacy_record.pop("replacement_step_ids")
    legacy_record.pop("validation_status")
    legacy_record.pop("recovery_result")
    legacy = snapshot_from_dict(payload).replan_history[0]

    assert legacy.replacement_step_ids == ()
    assert legacy.validation_status == "valid"
    assert legacy.recovery_result == "pending"


def test_repository_save_load_exists_delete_and_path_traversal(tmp_path) -> None:
    repository = FileExecutionSessionRepository(tmp_path)
    snapshot = _running_snapshot()

    repository.save(snapshot)

    assert repository.exists(snapshot.session_id)
    assert repository.list() == (snapshot.session_id,)
    assert repository.load(snapshot.session_id).session_id == snapshot.session_id
    with pytest.raises(ExecutionSnapshotCorruptedError, match="Invalid"):
        repository.load("../outside")
    repository.delete(snapshot.session_id)
    assert repository.load(snapshot.session_id) is None


def test_new_supervisor_continues_persisted_session_ids_without_overwrite(
    tmp_path,
) -> None:
    repository = FileExecutionSessionRepository(tmp_path)
    first_supervisor = ExecutionSupervisor(session_repository=repository)
    first = first_supervisor.start(_plan())
    second_plan = ExecutionPlan(
        goal="second persisted execution",
        ordered_steps=_plan().ordered_steps,
        estimated_steps=2,
        required_tools=("read_file", "read_file"),
        detected_risks=(),
        requires_confirmation=False,
    )

    second_supervisor = ExecutionSupervisor(session_repository=repository)
    second = second_supervisor.start(second_plan)

    assert first.session_id == "execution.session.000001"
    assert second.session_id == "execution.session.000002"
    assert repository.list() == (
        "execution.session.000001",
        "execution.session.000002",
    )
    restored_first = repository.load(first.session_id)
    restored_second = repository.load(second.session_id)
    assert restored_first is not None
    assert restored_second is not None
    assert restored_first.active_plan.goal == "persisted execution"
    assert restored_second.active_plan.goal == "second persisted execution"


def test_repository_rejects_corrupt_and_unsupported_snapshots(tmp_path) -> None:
    repository = FileExecutionSessionRepository(tmp_path)
    (tmp_path / "bad.json").write_text("{bad", encoding="utf-8")
    with pytest.raises(ExecutionSnapshotCorruptedError):
        repository.load("bad")

    payload = snapshot_to_dict(_running_snapshot())
    payload["schema_version"] = 999
    (tmp_path / "execution.session.000001.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )
    with pytest.raises(UnsupportedExecutionSnapshotVersion):
        repository.load("execution.session.000001")


def test_atomic_save_failure_preserves_previous_valid_file(tmp_path, monkeypatch) -> None:
    repository = FileExecutionSessionRepository(tmp_path)
    snapshot = _running_snapshot()
    repository.save(snapshot)
    before = (tmp_path / f"{snapshot.session_id}.json").read_text(encoding="utf-8")

    def fail_replace(_src, _dst):
        raise OSError("simulated crash")

    monkeypatch.setattr("core.execution_session_persistence.os.replace", fail_replace)
    with pytest.raises(Exception, match="Could not save"):
        repository.save(snapshot)

    assert (tmp_path / f"{snapshot.session_id}.json").read_text(encoding="utf-8") == before


def test_running_session_restores_as_interrupted_and_requires_review(tmp_path) -> None:
    repository = FileExecutionSessionRepository(tmp_path)
    snapshot = _running_snapshot(recovery_safe=True)
    repository.save(snapshot)
    supervisor = ExecutionSupervisor()
    service = ExecutionRecoveryService(repository, supervisor)

    report = service.recover()
    restored = supervisor.get_session(snapshot.session_id)

    assert restored.state is ExecutionState.INTERRUPTED
    assert restored.step_states["step_1"].state is StepExecutionState.INTERRUPTED
    assert report.interrupted_session_ids == (snapshot.session_id,)
    assert report.decisions[snapshot.session_id].decision is RecoveryDecisionType.REQUIRE_MANUAL_REVIEW


def test_pending_recovery_requires_explicit_step_safety() -> None:
    unsafe_supervisor = ExecutionSupervisor()
    unsafe = unsafe_supervisor.start(_plan(recovery_safe=False))
    safe_supervisor = ExecutionSupervisor()
    safe = safe_supervisor.start(_plan(recovery_safe=True))
    policy = ExecutionRecoveryPolicy()

    assert policy.evaluate(ExecutionSessionSnapshot.from_session(unsafe)).decision is RecoveryDecisionType.REQUIRE_MANUAL_REVIEW
    assert policy.evaluate(ExecutionSessionSnapshot.from_session(safe)).decision is RecoveryDecisionType.RESUME_AUTOMATICALLY


def test_waiting_confirmation_is_restored_without_execution(tmp_path) -> None:
    supervisor = ExecutionSupervisor()
    session = supervisor.start(_plan(requires_confirmation=True))
    supervisor.mark_running(session.session_id)
    supervisor.mark_waiting_confirmation(session.session_id, current_step="step_1")
    repository = FileExecutionSessionRepository(tmp_path)
    repository.save(ExecutionSessionSnapshot.from_session(supervisor.get_session(session.session_id)))
    restored_supervisor = ExecutionSupervisor()

    report = ExecutionRecoveryService(repository, restored_supervisor).recover()

    restored = restored_supervisor.get_session(session.session_id)
    assert restored.state is ExecutionState.WAITING_CONFIRMATION
    assert report.decisions[session.session_id].decision is RecoveryDecisionType.REQUIRE_CONFIRMATION


def test_batch_history_and_partial_results_survive_round_trip() -> None:
    supervisor = ExecutionSupervisor()
    session = supervisor.start(_plan(recovery_safe=True))
    supervisor.mark_running(session.session_id)
    batch_result = ExecutionBatchResult(
        batch_id="batch.1",
        step_results=(
            ConcurrentStepResult("step_1", "completed", result={"ok": True}),
        ),
    )
    supervisor.record_execution_batch_result(session.session_id, batch_result)
    supervisor.mark_completed(
        session.session_id,
        results={"step_1": {"ok": True}, "step_2": object()},
    )

    loaded = snapshot_from_dict(
        snapshot_to_dict(
            ExecutionSessionSnapshot.from_session(
                supervisor.get_session(session.session_id)
            )
        )
    )

    assert loaded.batch_history[0].completed_step_ids == ("step_1",)
    assert loaded.results["step_1"].serializable_value == {"ok": True}
    assert loaded.results["step_2"].fully_restorable is False


def test_restore_session_rejects_duplicate_live_session() -> None:
    supervisor = ExecutionSupervisor()
    session = supervisor.start(_plan())
    snapshot = ExecutionSessionSnapshot.from_session(session)

    with pytest.raises(ExecutionSessionAlreadyExistsError):
        supervisor.restore_session(snapshot.to_session())


def test_repository_writes_same_session_concurrently_without_corruption(tmp_path) -> None:
    repository = FileExecutionSessionRepository(tmp_path)
    snapshot = _running_snapshot(recovery_safe=True)
    barrier = Barrier(4)

    def save_snapshot() -> None:
        barrier.wait()
        repository.save(snapshot)

    threads = [Thread(target=save_snapshot) for _ in range(3)]
    for thread in threads:
        thread.start()
    barrier.wait()
    for thread in threads:
        thread.join()

    assert repository.load(snapshot.session_id).session_id == snapshot.session_id


def test_resume_recovered_session_does_not_repeat_completed_steps(tmp_path) -> None:
    supervisor = ExecutionSupervisor()
    session = supervisor.start(_plan(recovery_safe=True))
    supervisor.mark_step_completed(session.session_id, "step_1")
    repository = FileExecutionSessionRepository(tmp_path)
    repository.save(ExecutionSessionSnapshot.from_session(supervisor.get_session(session.session_id)))
    restored_supervisor = ExecutionSupervisor()
    service = ExecutionRecoveryService(repository, restored_supervisor)
    service.recover()
    executor = _RecordingExecutor()
    coordinator = StructuredExecutionCoordinator(
        planner=object(),  # type: ignore[arg-type]
        validator=_PassthroughValidator(),  # type: ignore[arg-type]
        executor=executor,  # type: ignore[arg-type]
        execution_supervisor=restored_supervisor,
        recovery_service=service,
    )

    response = coordinator.resume_recovered_session(session.session_id)

    assert response.status == "completed"
    assert executor.called_step_ids == ("step_2",)


class _PassthroughValidator:
    def validate(self, plan: ExecutionPlan) -> PlanValidationResult:
        return PlanValidationResult(
            is_valid=True,
            status="valid",
            requires_confirmation=plan.requires_confirmation,
            plan_signature=plan_signature(plan),
        )


class _RecordingExecutor:
    def __init__(self) -> None:
        self.called_step_ids: tuple[str, ...] = ()

    def execute(self, plan: ExecutionPlan, *_args, **_kwargs) -> PlanExecutionResult:
        self.called_step_ids = tuple(step.id for step in plan.ordered_steps)
        return PlanExecutionResult(
            plan_status=PlanExecutionStatus.COMPLETED.value,
            success=True,
            completed=True,
            completed_steps=list(self.called_step_ids),
        )
