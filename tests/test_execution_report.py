from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json

import pytest

from core.execution_report import (
    ExecutionReportGenerator,
    OperationalExecutionStatus,
)
from core.execution_session_persistence import (
    ExecutionSessionSnapshot,
    snapshot_from_dict,
    snapshot_to_dict,
)
from core.execution_supervisor import (
    ExecutionSession,
    ExecutionState,
    ExecutionSummary,
    ExecutionSupervisorEvent,
    ReplanRecoveryStatus,
    StepExecutionSnapshot,
    StepExecutionState,
)
from core.planner import ExecutionPlan, ExecutionStep
from core.structured_plan_replanner import ReplanReason, ReplanRecord


_START = datetime(2026, 7, 28, 10, 0, tzinfo=timezone.utc)


def _plan(
    *steps: ExecutionStep,
    goal: str = "Preparar el informe",
) -> ExecutionPlan:
    return ExecutionPlan(
        goal=goal,
        ordered_steps=steps
        or (
            ExecutionStep("step_1", "Leer los datos", "read_file"),
            ExecutionStep("step_2", "Preparar el resultado", "write_file"),
        ),
        estimated_steps=len(steps) if steps else 2,
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
    critical: bool = False,
) -> StepExecutionSnapshot:
    started_at = None if attempts == 0 else _START + timedelta(seconds=1)
    finished_at = (
        _START + timedelta(seconds=3)
        if state
        in {
            StepExecutionState.COMPLETED,
            StepExecutionState.FAILED,
            StepExecutionState.SKIPPED,
            StepExecutionState.CANCELLED,
        }
        and started_at is not None
        else None
    )
    return StepExecutionSnapshot(
        step_id=step_id,
        state=state,
        error=error,
        started_at=started_at,
        finished_at=finished_at,
        attempt_count=attempts,
        max_attempts=max(1, attempts),
        is_critical=critical,
    )


def _event(event_type: str) -> ExecutionSupervisorEvent:
    return ExecutionSupervisorEvent(
        session_id="session.1",
        event_type=event_type,
        state=ExecutionState.FAILED,
        timestamp=_START + timedelta(seconds=4),
    )


def _session(
    step_states: dict[str, StepExecutionSnapshot],
    *,
    state: ExecutionState,
    plan: ExecutionPlan | None = None,
    results: dict[str, object] | None = None,
    events: tuple[ExecutionSupervisorEvent, ...] = (),
    last_error: str | None = None,
    replan_history: tuple[ReplanRecord, ...] = (),
    active_plan: ExecutionPlan | None = None,
) -> ExecutionSession:
    original = plan or _plan()
    return ExecutionSession(
        session_id="session.1",
        plan=active_plan or original,
        original_plan=original,
        active_plan=active_plan or original,
        state=state,
        current_step=None,
        started_at=_START,
        finished_at=(
            _START + timedelta(seconds=5)
            if state
            in {
                ExecutionState.COMPLETED,
                ExecutionState.FAILED,
                ExecutionState.CANCELLED,
            }
            else None
        ),
        last_error=last_error,
        results=results or {},
        events=events,
        step_states=step_states,
        replan_count=len(replan_history),
        replan_history=replan_history,
    )


def _summary(
    session: ExecutionSession,
    *,
    replan_status: ReplanRecoveryStatus = ReplanRecoveryStatus.NOT_NEEDED,
) -> ExecutionSummary:
    states = tuple(session.step_states.values())
    return ExecutionSummary(
        session_id=session.session_id,
        state=session.state,
        total_steps=len(states),
        pending_steps=sum(step.state is StepExecutionState.PENDING for step in states),
        running_steps=sum(step.state is StepExecutionState.RUNNING for step in states),
        successful_steps=sum(
            step.state is StepExecutionState.COMPLETED for step in states
        ),
        failed_steps=sum(step.state is StepExecutionState.FAILED for step in states),
        retrying_steps=sum(
            step.state is StepExecutionState.RETRYING for step in states
        ),
        cancelled_steps=sum(
            step.state is StepExecutionState.CANCELLED for step in states
        ),
        skipped_steps=sum(step.state is StepExecutionState.SKIPPED for step in states),
        progress=session.progress,
        retry_count=sum(max(0, step.attempt_count - 1) for step in states),
        started_at=session.started_at,
        finished_at=session.finished_at,
        duration_seconds=5.0,
        errors={
            step.step_id: step.error for step in states if step.error is not None
        },
        critical_failure_step=next(
            (
                step.step_id
                for step in states
                if step.is_critical and step.state is StepExecutionState.FAILED
            ),
            None,
        ),
        replan_status=replan_status,
        replan_count=session.replan_count,
    )


def _report(
    session: ExecutionSession,
    *,
    replan_status: ReplanRecoveryStatus = ReplanRecoveryStatus.NOT_NEEDED,
):
    return ExecutionReportGenerator().generate(
        session,
        _summary(session, replan_status=replan_status),
    )


def test_completed_report_has_real_counts_duration_and_no_actions() -> None:
    session = _session(
        {
            "step_1": _step("step_1", StepExecutionState.COMPLETED),
            "step_2": _step("step_2", StepExecutionState.COMPLETED),
        },
        state=ExecutionState.COMPLETED,
    )

    report = _report(session)

    assert report.status is OperationalExecutionStatus.COMPLETED
    assert report.completed_steps == 2
    assert report.progress_percent == 100.0
    assert report.duration_seconds == 5.0
    assert report.pending_user_actions == ()
    assert report.replan_status == "not_needed"


def test_completed_report_summarizes_retries() -> None:
    session = _session(
        {
            "step_1": _step(
                "step_1",
                StepExecutionState.COMPLETED,
                attempts=3,
            ),
            "step_2": _step("step_2", StepExecutionState.COMPLETED),
        },
        state=ExecutionState.COMPLETED,
    )

    report = _report(session)

    assert report.retry_count == 2
    assert report.retried_step_ids == ("step_1",)
    assert "Reintentos: 2 en step_1." in report.to_text()


def test_successful_replan_marks_recovery_and_replaced_steps() -> None:
    original = _plan(ExecutionStep("step_1", "Leer origen", "read_file"))
    replacement = _plan(
        ExecutionStep("step_1_alt", "Leer copia segura", "read_file")
    )
    record = ReplanRecord(
        attempt_number=1,
        previous_plan=original,
        revised_plan=replacement,
        reason=ReplanReason.RECOVERABLE_FAILURE,
        failed_step="step_1",
        error="origen no disponible",
        created_at=_START + timedelta(seconds=3),
        replacement_step_ids=("step_1_alt",),
        recovery_result="succeeded",
    )
    session = _session(
        {
            "step_1": _step(
                "step_1",
                StepExecutionState.FAILED,
                error="origen no disponible",
            ),
            "step_1_alt": _step("step_1_alt", StepExecutionState.COMPLETED),
        },
        state=ExecutionState.COMPLETED,
        plan=original,
        active_plan=replacement,
        replan_history=(record,),
        events=(_event("replan_recovery_succeeded"),),
    )

    report = _report(session, replan_status=ReplanRecoveryStatus.SUCCEEDED)

    assert report.status is OperationalExecutionStatus.COMPLETED_WITH_RECOVERY
    assert report.replan_count == 1
    assert report.steps[0].replaced is True
    assert report.steps[1].replaced is False
    assert "Replanificación: aplicada con éxito." in report.to_text()


def test_partial_failed_cancelled_and_optional_outcomes() -> None:
    partial = _session(
        {
            "step_1": _step("step_1", StepExecutionState.COMPLETED),
            "step_2": _step(
                "step_2",
                StepExecutionState.FAILED,
                error="fallo de escritura",
            ),
        },
        state=ExecutionState.FAILED,
    )
    cancelled = _session(
        {
            "step_1": _step(
                "step_1",
                StepExecutionState.FAILED,
                error="fallo crítico",
                critical=True,
            ),
            "step_2": _step(
                "step_2",
                StepExecutionState.CANCELLED,
                attempts=0,
            ),
        },
        state=ExecutionState.CANCELLED,
    )
    omitted = _session(
        {
            "step_1": _step("step_1", StepExecutionState.COMPLETED),
            "step_2": _step(
                "step_2",
                StepExecutionState.SKIPPED,
                attempts=0,
            ),
        },
        state=ExecutionState.COMPLETED,
    )

    assert _report(partial).status is OperationalExecutionStatus.PARTIALLY_COMPLETED
    cancelled_report = _report(cancelled)
    assert cancelled_report.status is OperationalExecutionStatus.CANCELLED
    assert cancelled_report.cancelled_steps == 1
    assert "paso crítico" in " ".join(cancelled_report.warnings)
    omitted_report = _report(omitted)
    assert omitted_report.skipped_steps == 1
    assert omitted_report.steps[1].omitted is True


@pytest.mark.parametrize(
    ("event_type", "replan_status", "expected_action"),
    [
        (
            "replan_validation_rejected",
            ReplanRecoveryStatus.VALIDATION_REJECTED,
            "alternativa rechazada",
        ),
        (
            "replan_limit_reached",
            ReplanRecoveryStatus.LIMIT_REACHED,
            "Revisa el fallo",
        ),
        (
            "replan_no_safe_alternative",
            ReplanRecoveryStatus.NO_SAFE_ALTERNATIVE,
            "alternativa segura",
        ),
    ],
)
def test_recovery_stop_reports_concrete_user_action(
    event_type: str,
    replan_status: ReplanRecoveryStatus,
    expected_action: str,
) -> None:
    session = _session(
        {
            "step_1": _step(
                "step_1",
                StepExecutionState.FAILED,
                error="recuperación detenida",
            )
        },
        state=ExecutionState.FAILED,
        events=(_event(event_type),),
    )

    report = _report(session, replan_status=replan_status)

    assert report.status is OperationalExecutionStatus.USER_ACTION_REQUIRED
    assert expected_action in " ".join(report.pending_user_actions)


@pytest.mark.parametrize(
    ("results", "expected"),
    [
        ({"error_code": "CONFIRMATION_REQUIRED"}, "Confirma o cancela"),
        ({"error_code": "ACCESS_DENIED"}, "Concede el permiso"),
        ({"error_code": "MISSING_REQUIRED_DATA"}, "datos obligatorios"),
        ({"error_code": "NO_COMPATIBLE_RESOURCE"}, "recurso compatible"),
    ],
)
def test_report_only_requests_supported_concrete_actions(
    results: dict[str, object],
    expected: str,
) -> None:
    session = _session(
        {"step_1": _step("step_1", StepExecutionState.PENDING, attempts=0)},
        state=ExecutionState.WAITING_CONFIRMATION,
        results=results,
    )

    report = _report(session)

    assert report.status is OperationalExecutionStatus.USER_ACTION_REQUIRED
    assert expected in " ".join(report.pending_user_actions)


def test_report_sanitizes_secrets_and_does_not_expose_results() -> None:
    session = _session(
        {
            "step_1": _step(
                "step_1",
                StepExecutionState.FAILED,
                error="Authorization: Bearer super-secret-value",
            )
        },
        state=ExecutionState.FAILED,
        results={
            "output": "private raw output",
            "api_key": "sk-1234567890abcdef",
        },
    )

    payload = json.dumps(_report(session).to_dict(), ensure_ascii=False)

    assert "super-secret-value" not in payload
    assert "private raw output" not in payload
    assert "sk-1234567890abcdef" not in payload
    assert "[redacted]" in payload


def test_structured_and_text_rendering_are_serializable_and_deterministic() -> None:
    session = _session(
        {
            "step_1": _step("step_1", StepExecutionState.COMPLETED),
            "step_2": _step("step_2", StepExecutionState.SKIPPED, attempts=0),
        },
        state=ExecutionState.COMPLETED,
    )

    first = _report(session)
    second = _report(session)

    json.dumps(first.to_dict())
    assert first.to_dict() == second.to_dict()
    assert first.to_text() == second.to_text()
    assert "step_1: Leer los datos" in first.to_text()
    assert "step_2: Preparar el resultado" in first.to_text()


def test_report_can_be_reconstructed_from_persisted_session() -> None:
    session = _session(
        {
            "step_1": _step("step_1", StepExecutionState.COMPLETED),
            "step_2": _step("step_2", StepExecutionState.COMPLETED),
        },
        state=ExecutionState.COMPLETED,
    )
    snapshot = ExecutionSessionSnapshot.from_session(
        session,
        updated_at=_START + timedelta(seconds=6),
    )

    restored = snapshot_from_dict(snapshot_to_dict(snapshot)).to_session()
    restored_report = _report(restored)

    assert restored_report.to_dict() == _report(session).to_dict()
    assert restored_report.metadata["source"] == "execution_session"


def test_report_accepts_legacy_snapshot_without_recent_step_fields() -> None:
    session = _session(
        {
            "step_1": _step("step_1", StepExecutionState.COMPLETED),
            "step_2": _step("step_2", StepExecutionState.COMPLETED),
        },
        state=ExecutionState.COMPLETED,
    )
    payload = snapshot_to_dict(ExecutionSessionSnapshot.from_session(session))
    for step_payload in payload["step_states"].values():
        for field_name in (
            "started_at",
            "finished_at",
            "attempt_count",
            "max_attempts",
            "is_critical",
        ):
            step_payload.pop(field_name)

    restored = snapshot_from_dict(payload).to_session()
    report = _report(restored)

    assert report.status is OperationalExecutionStatus.COMPLETED
    assert all(step.attempts == 0 for step in report.steps)
    assert all(step.duration_seconds is None for step in report.steps)
