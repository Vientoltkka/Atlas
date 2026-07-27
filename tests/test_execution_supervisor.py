from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from core.execution_plan_executor import PlanExecutionResult, PlanExecutionStatus
from core.execution_plan_validator import PlanValidationResult, plan_signature
from core.execution_supervisor import (
    ExecutionSessionNotFoundError,
    ExecutionState,
    ExecutionSupervisor,
    InvalidExecutionTransitionError,
)
from core.planner import ExecutionPlan, ExecutionStep, PlanGenerationResult
from core.structured_execution import StructuredExecutionCoordinator


class _Clock:
    def __init__(self) -> None:
        self.current = datetime(2026, 1, 1, tzinfo=timezone.utc)

    def __call__(self) -> datetime:
        value = self.current
        self.current = self.current + timedelta(seconds=1)
        return value


class _FixedPlanner:
    def __init__(self, plan: ExecutionPlan) -> None:
        self.plan = plan
        self.calls = 0

    def generate_execution_plan(self, _objective: str, **_kwargs) -> PlanGenerationResult:
        self.calls += 1
        return PlanGenerationResult(
            success=True,
            plan=self.plan,
            generation_attempted=True,
        )


class _FixedValidator:
    def validate(self, plan: ExecutionPlan) -> PlanValidationResult:
        return PlanValidationResult(
            is_valid=True,
            requires_confirmation=plan.requires_confirmation,
            status="valid",
            plan_signature=plan_signature(plan),
        )


class _FixedExecutor:
    def __init__(self, result: PlanExecutionResult) -> None:
        self.result = result
        self.calls = 0

    def execute(self, *_args, **_kwargs) -> PlanExecutionResult:
        self.calls += 1
        return self.result


def _plan(*, requires_confirmation: bool = False) -> ExecutionPlan:
    return ExecutionPlan(
        goal="test plan",
        ordered_steps=(ExecutionStep("step_1", "read", "read_file"),),
        estimated_steps=1,
        required_tools=("read_file",),
        detected_risks=(),
        requires_confirmation=requires_confirmation,
    )


def test_start_creates_pending_session_with_initial_event() -> None:
    supervisor = ExecutionSupervisor(clock=_Clock())
    plan = _plan()

    session = supervisor.start(plan)

    assert session.session_id == "execution.session.000001"
    assert session.plan is plan
    assert session.state is ExecutionState.PENDING
    assert session.current_step is None
    assert session.finished_at is None
    assert session.last_error is None
    assert session.results == {}
    assert [event.event_type for event in session.events] == ["execution_started"]


def test_get_session_returns_current_snapshot() -> None:
    supervisor = ExecutionSupervisor(clock=_Clock())
    session = supervisor.start(_plan())

    assert supervisor.get_session(session.session_id) is session


def test_get_session_unknown_id_raises_clear_error() -> None:
    supervisor = ExecutionSupervisor(clock=_Clock())

    with pytest.raises(ExecutionSessionNotFoundError, match="was not found"):
        supervisor.get_session("missing")


def test_mark_running_and_completed_updates_state_and_results() -> None:
    supervisor = ExecutionSupervisor(clock=_Clock())
    session = supervisor.start(_plan())

    running = supervisor.mark_running(session.session_id, current_step="step_1")
    completed = supervisor.mark_completed(
        session.session_id,
        results={"completed_steps": ("step_1",)},
    )

    assert running.state is ExecutionState.RUNNING
    assert running.current_step == "step_1"
    assert completed.state is ExecutionState.COMPLETED
    assert completed.current_step == "step_1"
    assert completed.finished_at is not None
    assert completed.results["completed_steps"] == ("step_1",)
    assert [event.event_type for event in completed.events] == [
        "execution_started",
        "execution_running",
        "execution_completed",
    ]


def test_mark_failed_records_error_and_current_step() -> None:
    supervisor = ExecutionSupervisor(clock=_Clock())
    session = supervisor.start(_plan())

    supervisor.mark_running(session.session_id)
    failed = supervisor.mark_failed(
        session.session_id,
        RuntimeError("tool failed"),
        current_step="step_1",
    )

    assert failed.state is ExecutionState.FAILED
    assert failed.current_step == "step_1"
    assert failed.last_error == "tool failed"
    assert failed.is_terminal is True


def test_mark_cancelled_from_running() -> None:
    supervisor = ExecutionSupervisor(clock=_Clock())
    session = supervisor.start(_plan())

    supervisor.mark_running(session.session_id)
    cancelled = supervisor.mark_cancelled(session.session_id)

    assert cancelled.state is ExecutionState.CANCELLED
    assert cancelled.finished_at is not None


def test_waiting_confirmation_can_resume_running() -> None:
    supervisor = ExecutionSupervisor(clock=_Clock())
    session = supervisor.start(_plan(requires_confirmation=True))

    supervisor.mark_running(session.session_id)
    waiting = supervisor.mark_waiting_confirmation(session.session_id)
    running = supervisor.mark_running(session.session_id)

    assert waiting.state is ExecutionState.WAITING_CONFIRMATION
    assert running.state is ExecutionState.RUNNING


def test_invalid_transitions_raise_clear_exception() -> None:
    supervisor = ExecutionSupervisor(clock=_Clock())
    session = supervisor.start(_plan())

    with pytest.raises(InvalidExecutionTransitionError, match="pending -> completed"):
        supervisor.mark_completed(session.session_id)

    supervisor.mark_running(session.session_id)
    supervisor.mark_completed(session.session_id)
    with pytest.raises(InvalidExecutionTransitionError, match="completed -> running"):
        supervisor.mark_running(session.session_id)


def test_structured_execution_success_is_supervised() -> None:
    plan = _plan()
    supervisor = ExecutionSupervisor(clock=_Clock())
    executor = _FixedExecutor(
        PlanExecutionResult(
            plan_status=PlanExecutionStatus.COMPLETED.value,
            success=True,
            completed=True,
            completed_steps=["step_1"],
        )
    )
    coordinator = StructuredExecutionCoordinator(
        planner=_FixedPlanner(plan),  # type: ignore[arg-type]
        validator=_FixedValidator(),  # type: ignore[arg-type]
        executor=executor,  # type: ignore[arg-type]
        execution_supervisor=supervisor,
    )

    response = coordinator.handle("run")
    session = supervisor.get_session("execution.session.000001")

    assert response.status == "completed"
    assert executor.calls == 1
    assert session.state is ExecutionState.COMPLETED
    assert session.results["plan_status"] == PlanExecutionStatus.COMPLETED.value


def test_structured_execution_failure_is_supervised_with_current_step() -> None:
    plan = _plan()
    supervisor = ExecutionSupervisor(clock=_Clock())
    executor = _FixedExecutor(
        PlanExecutionResult(
            plan_status=PlanExecutionStatus.FAILED.value,
            success=False,
            failed=True,
            failed_step="step_1",
            current_step="step_1",
            failed_steps=["step_1"],
            error="tool failed",
            error_code="TOOL_EXECUTION_FAILED",
        )
    )
    coordinator = StructuredExecutionCoordinator(
        planner=_FixedPlanner(plan),  # type: ignore[arg-type]
        validator=_FixedValidator(),  # type: ignore[arg-type]
        executor=executor,  # type: ignore[arg-type]
        execution_supervisor=supervisor,
    )

    response = coordinator.handle("run")
    session = supervisor.get_session("execution.session.000001")

    assert response.status == "failed"
    assert session.state is ExecutionState.FAILED
    assert session.current_step == "step_1"
    assert session.last_error == "tool failed"


def test_confirmation_pending_session_resumes_and_completes_on_confirm() -> None:
    plan = _plan(requires_confirmation=True)
    supervisor = ExecutionSupervisor(clock=_Clock())
    executor = _FixedExecutor(
        PlanExecutionResult(
            plan_status=PlanExecutionStatus.COMPLETED.value,
            success=True,
            completed=True,
            completed_steps=["step_1"],
        )
    )
    coordinator = StructuredExecutionCoordinator(
        planner=_FixedPlanner(plan),  # type: ignore[arg-type]
        validator=_FixedValidator(),  # type: ignore[arg-type]
        executor=executor,  # type: ignore[arg-type]
        execution_supervisor=supervisor,
    )

    pending = coordinator.handle("run")
    waiting = supervisor.get_session("execution.session.000001")
    confirmed = coordinator.confirm_pending()
    completed = supervisor.get_session("execution.session.000001")

    assert pending.status == "confirmation_required"
    assert waiting.state is ExecutionState.WAITING_CONFIRMATION
    assert confirmed.status == "completed"
    assert completed.state is ExecutionState.COMPLETED
    assert executor.calls == 1
