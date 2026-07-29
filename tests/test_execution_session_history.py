from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json

import pytest

from bootstrap.bootstrap import _execution_history_path
from core.execution_history import ExecutionSessionHistory
from core.execution_report import OperationalExecutionStatus
from core.execution_session_persistence import (
    ExecutionSessionSnapshot,
    FileExecutionSessionRepository,
    snapshot_to_dict,
)
from core.execution_supervisor import (
    ExecutionSession,
    ExecutionState,
    ExecutionSupervisor,
    ExecutionSupervisorEvent,
    StepExecutionSnapshot,
    StepExecutionState,
)
from core.planner import ExecutionPlan, ExecutionStep
from core.structured_plan_replanner import ReplanReason, ReplanRecord


_START = datetime(2026, 7, 29, 9, 0, tzinfo=timezone.utc)


def _plan(goal: str = "Preparar informe") -> ExecutionPlan:
    return ExecutionPlan(
        goal=goal,
        ordered_steps=(
            ExecutionStep("read", "Leer datos", "read_file"),
            ExecutionStep("write", "Escribir informe", "write_file"),
        ),
        estimated_steps=2,
        required_tools=("read_file", "write_file"),
        detected_risks=(),
        requires_confirmation=False,
    )


def _step(
    step_id: str,
    state: StepExecutionState,
    *,
    attempts: int = 1,
    error: str | None = None,
) -> StepExecutionSnapshot:
    started_at = _START + timedelta(seconds=1) if attempts else None
    finished_at = (
        _START + timedelta(seconds=4)
        if state
        in {
            StepExecutionState.COMPLETED,
            StepExecutionState.FAILED,
            StepExecutionState.BLOCKED,
            StepExecutionState.INTERRUPTED,
            StepExecutionState.SKIPPED,
            StepExecutionState.CANCELLED,
        }
        else None
    )
    if finished_at is not None and started_at is None:
        started_at = finished_at
    return StepExecutionSnapshot(
        step_id=step_id,
        state=state,
        error=error,
        started_at=started_at,
        finished_at=finished_at,
        attempt_count=attempts,
        max_attempts=max(1, attempts),
    )


def _session(
    session_id: str,
    *,
    state: ExecutionState,
    goal: str = "Preparar informe",
    offset_minutes: int = 0,
    retry_count: int = 0,
    skipped: bool = False,
    replanned: bool = False,
) -> ExecutionSession:
    original = _plan(goal)
    active = original
    replan_history: tuple[ReplanRecord, ...] = ()
    if replanned:
        active = ExecutionPlan(
            goal=goal,
            ordered_steps=(
                ExecutionStep("read", "Leer datos alternativos", "read_file"),
                ExecutionStep("publish", "Publicar informe", "write_file"),
            ),
            estimated_steps=2,
            required_tools=("read_file", "write_file"),
            detected_risks=(),
            requires_confirmation=False,
        )
        replan_history = (
            ReplanRecord(
                attempt_number=1,
                previous_plan=original,
                revised_plan=active,
                reason=ReplanReason.RECOVERABLE_FAILURE,
                failed_step="write",
                error="write failed",
                created_at=_START,
                replacement_step_ids=("publish",),
                recovery_result="succeeded",
            ),
        )
    started = _START + timedelta(minutes=offset_minutes)
    failed = state is ExecutionState.FAILED
    step_states = {
        "read": _step(
            "read",
            StepExecutionState.COMPLETED,
            attempts=1 + retry_count,
        ),
        "write": _step(
            "write",
            (
                StepExecutionState.FAILED
                if failed
                else (
                    StepExecutionState.SKIPPED
                    if skipped
                    else StepExecutionState.COMPLETED
                )
            ),
            attempts=0 if skipped else 1,
            error="write failed" if failed else None,
        ),
    }
    if replanned:
        step_states["publish"] = _step(
            "publish",
            StepExecutionState.COMPLETED,
        )
    return ExecutionSession(
        session_id=session_id,
        plan=active,
        original_plan=original,
        active_plan=active,
        state=state,
        current_step=None,
        started_at=started,
        finished_at=started + timedelta(seconds=10),
        last_error="write failed" if failed else None,
        events=(
            (
                ExecutionSupervisorEvent(
                    session_id=session_id,
                    event_type="replan_recovery_succeeded",
                    state=state,
                    timestamp=started + timedelta(seconds=9),
                ),
            )
            if replanned
            else ()
        ),
        step_states=step_states,
        replan_count=len(replan_history),
        replan_history=replan_history,
    )


def test_supervisor_registration_is_queryable_without_duplicate_history_storage(
    tmp_path,
) -> None:
    tick = {"value": 0}

    def clock() -> datetime:
        tick["value"] += 1
        return _START + timedelta(seconds=tick["value"])

    repository = FileExecutionSessionRepository(tmp_path)
    supervisor = ExecutionSupervisor(
        clock=clock,
        session_repository=repository,
    )
    session = supervisor.start(_plan())
    supervisor.mark_running(session.session_id)
    supervisor.mark_step_started(session.session_id, "read")
    supervisor.mark_step_retrying(
        session.session_id,
        "read",
        attempt_number=2,
        max_attempts=2,
        error="temporary",
    )
    supervisor.mark_step_completed(session.session_id, "read")
    supervisor.mark_step_skipped(session.session_id, "write")
    supervisor.mark_completed(session.session_id)

    history = ExecutionSessionHistory(
        session_source=supervisor,
        session_repository=repository,
    )
    record = history.latest_execution()

    assert repository.list() == (session.session_id,)
    assert record is not None
    assert record.id == session.session_id
    assert record.objective == "Preparar informe"
    assert record.final_result is OperationalExecutionStatus.COMPLETED
    assert record.executed_step_ids == ("read",)
    assert record.omitted_step_ids == ("write",)
    assert record.retry_count == 1
    assert record.recovery_types == ("retry",)
    assert record.operational_report.session_id == session.session_id


def test_queries_filter_sort_and_deduplicate_live_and_persisted_sessions(
    tmp_path,
) -> None:
    repository = FileExecutionSessionRepository(tmp_path)
    completed = _session(
        "execution.1",
        state=ExecutionState.COMPLETED,
        offset_minutes=1,
    )
    failed = _session(
        "execution.2",
        state=ExecutionState.FAILED,
        goal="Publicar informe",
        offset_minutes=2,
    )
    recovered = _session(
        "execution.3",
        state=ExecutionState.COMPLETED,
        goal=" preparar   INFORME ",
        offset_minutes=3,
        replanned=True,
    )
    active = ExecutionSession(
        session_id="execution.active",
        plan=_plan(),
        state=ExecutionState.RUNNING,
        current_step="read",
        started_at=_START + timedelta(minutes=4),
        step_states={
            "read": _step("read", StepExecutionState.RUNNING),
            "write": _step("write", StepExecutionState.PENDING, attempts=0),
        },
    )
    for session in (completed, failed, recovered, active):
        repository.save(ExecutionSessionSnapshot.from_session(session))

    supervisor = ExecutionSupervisor()
    supervisor.restore_session(recovered)
    history = ExecutionSessionHistory(
        session_source=supervisor,
        session_repository=repository,
    )

    assert tuple(
        record.id for record in history.latest_executions(2)
    ) == ("execution.3", "execution.2")
    assert history.latest_execution().id == "execution.3"
    assert tuple(
        record.id
        for record in history.executions_by_objective("PREPARAR informe")
    ) == ("execution.3", "execution.1")
    assert tuple(
        record.id for record in history.failed_executions()
    ) == ("execution.2",)
    assert tuple(
        record.id for record in history.successful_executions()
    ) == ("execution.3", "execution.1")
    assert tuple(
        record.id for record in history.executions_with_recovery()
    ) == ("execution.3",)
    assert (
        history.latest_execution().final_result
        is OperationalExecutionStatus.COMPLETED_WITH_RECOVERY
    )
    assert history.latest_executions(0) == ()


def test_statistics_are_rebuilt_from_terminal_sessions(tmp_path) -> None:
    repository = FileExecutionSessionRepository(tmp_path)
    sessions = (
        _session(
            "execution.1",
            state=ExecutionState.COMPLETED,
            retry_count=1,
            skipped=True,
        ),
        _session(
            "execution.2",
            state=ExecutionState.FAILED,
            offset_minutes=1,
        ),
        _session(
            "execution.3",
            state=ExecutionState.COMPLETED,
            offset_minutes=2,
            replanned=True,
        ),
    )
    for session in sessions:
        repository.save(ExecutionSessionSnapshot.from_session(session))

    stats = ExecutionSessionHistory(session_repository=repository).statistics()

    assert stats.total_executions == 3
    assert stats.successful_executions == 2
    assert stats.failed_executions == 1
    assert stats.cancelled_executions == 0
    assert stats.success_frequency == pytest.approx(2 / 3, abs=0.0001)
    assert stats.failure_frequency == pytest.approx(1 / 3, abs=0.0001)
    assert stats.average_duration_seconds == 10.0
    assert stats.average_retry_count == pytest.approx(1 / 3, abs=0.001)
    assert stats.frequently_failed_steps == {"write": 1}
    assert stats.normally_omitted_steps == {"write": 1}
    assert stats.recovery_types == {
        "replan:recoverable_failure": 1,
        "retry": 1,
    }


def test_legacy_persisted_execution_rebuilds_history_when_fields_are_absent(
    tmp_path,
) -> None:
    repository = FileExecutionSessionRepository(tmp_path)
    session = _session(
        "execution.legacy",
        state=ExecutionState.COMPLETED,
        retry_count=1,
        replanned=True,
    )
    payload = snapshot_to_dict(ExecutionSessionSnapshot.from_session(session))
    for step_payload in payload["step_states"].values():
        for key in (
            "started_at",
            "finished_at",
            "attempt_count",
            "max_attempts",
            "is_critical",
        ):
            step_payload.pop(key)
    legacy_replan = payload["replan_history"][0]
    legacy_replan.pop("replacement_step_ids")
    legacy_replan.pop("validation_status")
    legacy_replan.pop("recovery_result")
    (tmp_path / "execution.legacy.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )

    record = ExecutionSessionHistory(
        session_repository=repository
    ).latest_execution()

    assert record is not None
    assert record.id == "execution.legacy"
    assert record.retry_count == 0
    assert record.replanned_step_ids == ("write", "publish")
    assert record.recovery_types == ("replan:recoverable_failure",)
    assert record.operational_report.metadata["source"] == "execution_session"


@pytest.mark.parametrize("limit", [-1, True, 1.5])
def test_latest_execution_limit_is_validated(tmp_path, limit) -> None:
    history = ExecutionSessionHistory(
        session_repository=FileExecutionSessionRepository(tmp_path)
    )

    expected = ValueError if limit == -1 else TypeError
    with pytest.raises(expected):
        history.latest_executions(limit)


def test_history_requires_a_source_and_valid_objective(tmp_path) -> None:
    with pytest.raises(ValueError, match="session_source or session_repository"):
        ExecutionSessionHistory()

    history = ExecutionSessionHistory(
        session_repository=FileExecutionSessionRepository(tmp_path)
    )
    with pytest.raises(ValueError, match="objective"):
        history.executions_by_objective(" ")


def test_history_path_is_compatible_with_execution_persistence_configuration(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.delenv("ATLAS_EXECUTION_HISTORY_PATH", raising=False)
    monkeypatch.setenv(
        "ATLAS_EXECUTION_STATE_PATH",
        str(tmp_path / "state.json"),
    )
    assert _execution_history_path() == tmp_path / "execution_sessions"

    configured = tmp_path / "history"
    monkeypatch.setenv("ATLAS_EXECUTION_HISTORY_PATH", str(configured))
    assert _execution_history_path() == configured


def test_corrupt_persisted_session_is_skipped_when_other_history_is_usable(
    tmp_path,
) -> None:
    repository = FileExecutionSessionRepository(tmp_path)
    valid = _session(
        "execution.valid",
        state=ExecutionState.COMPLETED,
    )
    repository.save(ExecutionSessionSnapshot.from_session(valid))
    (tmp_path / "execution.corrupt.json").write_text("{bad", encoding="utf-8")

    records = ExecutionSessionHistory(
        session_repository=repository
    ).latest_executions(10)

    assert tuple(record.id for record in records) == ("execution.valid",)
