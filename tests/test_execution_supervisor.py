from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import datetime, timedelta, timezone
from threading import Lock, Thread

import pytest

from core.concurrent_step_executor import (
    ConcurrentStepExecutor,
    ExecutionConcurrencyPolicy,
)
from core.execution_plan_executor import PlanExecutionResult, PlanExecutionStatus
from core.execution_plan_executor import StepExecutionResult
from core.execution_report import OperationalExecutionStatus
from core.execution_plan_validator import PlanValidationResult, plan_signature
from core.execution_retry import RetryPolicy
from core.execution_supervisor import (
    ExecutionSummary,
    ExecutionOverview,
    ExecutionSessionNotFoundError,
    ExecutionState,
    ExecutionSupervisor,
    InvalidExecutionTransitionError,
    ReplanRecoveryStatus,
    StepExecutionState,
)
from core.planner import ExecutionPlan, ExecutionStep, PlanGenerationResult
from core.structured_execution import StructuredExecutionCoordinator
from core.structured_plan_replanner import (
    ReplanPolicy,
    ReplanReason,
    ReplanRequest,
    ReplanResult,
    ReplanResultStatus,
)


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


class _RejectGoalValidator(_FixedValidator):
    def __init__(self, rejected_goal: str) -> None:
        self.rejected_goal = rejected_goal

    def validate(self, plan: ExecutionPlan) -> PlanValidationResult:
        if plan.goal == self.rejected_goal:
            return PlanValidationResult(
                is_valid=False,
                errors=["unsafe replacement fragment"],
                status="invalid",
            )
        return super().validate(plan)


class _FixedExecutor:
    def __init__(self, result: PlanExecutionResult) -> None:
        self.result = result
        self.calls = 0

    def execute(self, *_args, **_kwargs) -> PlanExecutionResult:
        self.calls += 1
        return self.result


class _SequenceExecutor:
    def __init__(self, *results: PlanExecutionResult) -> None:
        self.results = list(results)
        self.calls: list[ExecutionPlan] = []

    def execute(self, plan: ExecutionPlan, *_args, **_kwargs) -> PlanExecutionResult:
        self.calls.append(plan)
        if not self.results:
            raise AssertionError("unexpected executor call")
        return self.results.pop(0)


class _FakeReplanner:
    def __init__(self, *results: ReplanResult) -> None:
        self.results = list(results)
        self.calls: list[ReplanRequest] = []

    def replan(self, request: ReplanRequest) -> ReplanResult:
        self.calls.append(request)
        if not self.results:
            raise AssertionError("unexpected replanner call")
        return self.results.pop(0)


def _plan(
    *,
    requires_confirmation: bool = False,
    goal: str = "test plan",
    tool: str = "read_file",
) -> ExecutionPlan:
    return ExecutionPlan(
        goal=goal,
        ordered_steps=(ExecutionStep("step_1", "read", tool),),
        estimated_steps=1,
        required_tools=(tool,),
        detected_risks=(),
        requires_confirmation=requires_confirmation,
    )


def _completed_result(*, completed_steps: list[str] | None = None) -> PlanExecutionResult:
    steps = completed_steps or ["step_1"]
    return PlanExecutionResult(
        plan_status=PlanExecutionStatus.COMPLETED.value,
        success=True,
        completed=True,
        completed_steps=steps,
    )


def _failed_result(
    *,
    error_code: str = "TOOL_EXECUTION_FAILED",
    error: str = "tool failed",
    completed_step_result: bool = False,
) -> PlanExecutionResult:
    step_results = []
    completed_steps = []
    if completed_step_result:
        completed_steps.append("step_0")
        step_results.append(
            StepExecutionResult(
                step_id="step_0",
                status="completed",
                success=True,
                tool_name="read_file",
                output={"value": "partial"},
            )
        )
    return PlanExecutionResult(
        plan_status=PlanExecutionStatus.FAILED.value,
        success=False,
        failed=True,
        failed_step="step_1",
        current_step="step_1",
        completed_steps=completed_steps,
        failed_steps=["step_1"],
        step_results=step_results,
        error=error,
        error_code=error_code,
    )


def _blocked_result() -> PlanExecutionResult:
    return PlanExecutionResult(
        plan_status=PlanExecutionStatus.BLOCKED.value,
        success=False,
        blocked=True,
        completed_steps=["step_1"],
        blocked_steps=["step_2"],
        pending_steps=["step_3"],
        current_step="step_2",
        error="blocked",
        error_code="DEPENDENCY_NOT_COMPLETED",
        step_results=[
            StepExecutionResult(
                step_id="step_1",
                status="completed",
                success=True,
                tool_name="read_file",
                output="ok",
            ),
            StepExecutionResult(
                step_id="step_2",
                status="blocked",
                success=False,
                tool_name="write_file",
                error="blocked",
            ),
        ],
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
    assert session.step_states["step_1"].state is StepExecutionState.PENDING
    assert [event.event_type for event in session.events] == ["execution_started"]


def test_supervisor_tracks_timing_retry_progress_and_final_summary() -> None:
    clock = _Clock()
    plan = ExecutionPlan(
        goal="retry one step",
        ordered_steps=(
            ExecutionStep(
                "step_1",
                "read",
                "read_file",
                retry_policy=RetryPolicy(max_attempts=3),
            ),
            ExecutionStep("step_2", "optional", "read_file"),
        ),
        estimated_steps=2,
        required_tools=("read_file",),
        detected_risks=(),
        requires_confirmation=False,
    )
    supervisor = ExecutionSupervisor(clock=clock)
    session = supervisor.start(plan)
    supervisor.mark_running(session.session_id)

    supervisor.mark_step_started(session.session_id, "step_1")
    supervisor.mark_step_retrying(
        session.session_id,
        "step_1",
        attempt_number=2,
        max_attempts=3,
        error="temporary failure",
    )
    supervisor.mark_step_completed(session.session_id, "step_1")
    supervisor.mark_step_skipped(session.session_id, "step_2")
    supervisor.mark_completed(session.session_id)

    finished = supervisor.get_session(session.session_id)
    snapshot = finished.step_states["step_1"]
    summary = supervisor.generate_summary(session.session_id)

    assert StepExecutionState.SUCCESS is StepExecutionState.COMPLETED
    assert snapshot.started_at is not None
    assert snapshot.finished_at is not None
    assert snapshot.finished_at >= snapshot.started_at
    assert snapshot.attempt_count == 2
    assert snapshot.max_attempts == 3
    assert finished.progress == 1.0
    assert isinstance(summary, ExecutionSummary)
    assert summary.successful_steps == 1
    assert summary.skipped_steps == 1
    assert summary.retry_count == 1
    assert summary.progress == 1.0
    assert summary.finished_at is not None
    assert summary.duration_seconds >= 0
    assert "step_retrying" in [event.event_type for event in finished.events]


def test_critical_step_failure_cancels_session_and_pending_steps() -> None:
    plan = ExecutionPlan(
        goal="critical plan",
        ordered_steps=(
            ExecutionStep("critical", "critical", "tool", criticality=1),
            ExecutionStep("remaining", "remaining", "tool"),
        ),
        estimated_steps=2,
        required_tools=("tool",),
        detected_risks=(),
        requires_confirmation=False,
    )
    supervisor = ExecutionSupervisor(clock=_Clock())
    session = supervisor.start(plan)
    supervisor.mark_running(session.session_id)
    supervisor.mark_step_started(session.session_id, "critical")

    cancelled = supervisor.mark_step_failed(
        session.session_id,
        "critical",
        "irrecoverable",
    )
    summary = supervisor.get_summary(session.session_id)

    assert cancelled.state is ExecutionState.CANCELLED
    assert cancelled.finished_at is not None
    assert cancelled.step_states["critical"].state is StepExecutionState.FAILED
    assert cancelled.step_states["remaining"].state is StepExecutionState.CANCELLED
    assert summary.critical_failure_step == "critical"
    assert summary.failed_steps == 1
    assert summary.cancelled_steps == 1
    assert summary.errors["critical"] == "irrecoverable"
    assert cancelled.events[-1].event_type == "execution_cancelled_critical_step"


def test_coordinator_records_executor_retry_metadata_in_supervisor() -> None:
    plan = _plan()
    result = PlanExecutionResult(
        plan_status=PlanExecutionStatus.COMPLETED.value,
        success=True,
        completed=True,
        completed_steps=["step_1"],
        step_results=[
            StepExecutionResult(
                step_id="step_1",
                status="completed",
                success=True,
                tool_name="read_file",
                output="ok",
                metadata={
                    "attempt_number": 2,
                    "max_attempts": 2,
                    "retry_history": [
                        {"error_code": "TEMPORARY_UNAVAILABLE", "error": "retry me"}
                    ],
                },
            )
        ],
    )
    supervisor = ExecutionSupervisor(clock=_Clock())
    coordinator = StructuredExecutionCoordinator(
        planner=_FixedPlanner(plan),  # type: ignore[arg-type]
        validator=_FixedValidator(),  # type: ignore[arg-type]
        executor=_FixedExecutor(result),  # type: ignore[arg-type]
        execution_supervisor=supervisor,
    )

    response = coordinator.handle("run")
    session = supervisor.get_session("execution.session.000001")

    assert response.status == "completed"
    assert session.step_states["step_1"].attempt_count == 2
    assert session.step_states["step_1"].state is StepExecutionState.COMPLETED
    assert "step_retrying" in [event.event_type for event in session.events]
    assert response.operational_report is not None
    assert response.operational_report.status is OperationalExecutionStatus.COMPLETED
    assert response.operational_report.retry_count == 1
    assert (
        coordinator.get_execution_report(session.session_id).to_dict()
        == response.operational_report.to_dict()
    )


def test_empty_overview_has_zero_counts_and_no_latest_session() -> None:
    supervisor = ExecutionSupervisor(clock=_Clock())

    overview = supervisor.get_overview()

    assert overview == ExecutionOverview(
        total_sessions=0,
        pending_sessions=0,
        running_sessions=0,
        waiting_confirmation_sessions=0,
        replanning_sessions=0,
        interrupted_sessions=0,
        completed_sessions=0,
        failed_sessions=0,
        cancelled_sessions=0,
        active_sessions=0,
        terminal_sessions=0,
        latest_session_id=None,
        generated_at=overview.generated_at,
    )


def test_overview_counts_new_pending_session_as_active() -> None:
    supervisor = ExecutionSupervisor(clock=_Clock())

    session = supervisor.start(_plan())
    overview = supervisor.get_overview()

    assert overview.total_sessions == 1
    assert overview.pending_sessions == 1
    assert overview.active_sessions == 1
    assert overview.terminal_sessions == 0
    assert overview.latest_session_id == session.session_id


def test_overview_updates_pending_to_running_transition() -> None:
    supervisor = ExecutionSupervisor(clock=_Clock())
    session = supervisor.start(_plan())

    supervisor.mark_running(session.session_id)
    overview = supervisor.get_overview()

    assert overview.total_sessions == 1
    assert overview.pending_sessions == 0
    assert overview.running_sessions == 1
    assert overview.active_sessions == 1


def test_overview_updates_running_to_completed_transition() -> None:
    supervisor = ExecutionSupervisor(clock=_Clock())
    session = supervisor.start(_plan())

    supervisor.mark_running(session.session_id)
    supervisor.mark_completed(session.session_id)
    overview = supervisor.get_overview()

    assert overview.running_sessions == 0
    assert overview.completed_sessions == 1
    assert overview.active_sessions == 0
    assert overview.terminal_sessions == 1


def test_overview_updates_running_to_failed_transition() -> None:
    supervisor = ExecutionSupervisor(clock=_Clock())
    session = supervisor.start(_plan())

    supervisor.mark_running(session.session_id)
    supervisor.mark_failed(session.session_id, "tool failed")
    overview = supervisor.get_overview()

    assert overview.running_sessions == 0
    assert overview.failed_sessions == 1
    assert overview.terminal_sessions == 1


def test_overview_updates_running_to_waiting_confirmation_transition() -> None:
    supervisor = ExecutionSupervisor(clock=_Clock())
    session = supervisor.start(_plan(requires_confirmation=True))

    supervisor.mark_running(session.session_id)
    supervisor.mark_waiting_confirmation(session.session_id)
    overview = supervisor.get_overview()

    assert overview.running_sessions == 0
    assert overview.waiting_confirmation_sessions == 1
    assert overview.active_sessions == 1


def test_overview_updates_cancelled_transition() -> None:
    supervisor = ExecutionSupervisor(clock=_Clock())
    session = supervisor.start(_plan())

    supervisor.mark_running(session.session_id)
    supervisor.mark_cancelled(session.session_id)
    overview = supervisor.get_overview()

    assert overview.cancelled_sessions == 1
    assert overview.active_sessions == 0
    assert overview.terminal_sessions == 1


def test_overview_counts_multiple_states_and_preserves_invariants() -> None:
    supervisor = ExecutionSupervisor(clock=_Clock())
    pending = supervisor.start(_plan())
    running = supervisor.start(_plan())
    waiting = supervisor.start(_plan(requires_confirmation=True))
    completed = supervisor.start(_plan())
    failed = supervisor.start(_plan())
    cancelled = supervisor.start(_plan())

    supervisor.mark_running(running.session_id)
    supervisor.mark_running(waiting.session_id)
    supervisor.mark_waiting_confirmation(waiting.session_id)
    supervisor.mark_running(completed.session_id)
    supervisor.mark_completed(completed.session_id)
    supervisor.mark_running(failed.session_id)
    supervisor.mark_failed(failed.session_id, "tool failed")
    supervisor.mark_running(cancelled.session_id)
    supervisor.mark_cancelled(cancelled.session_id)

    overview = supervisor.get_overview()

    assert pending.state is ExecutionState.PENDING
    assert overview.total_sessions == 6
    assert overview.pending_sessions == 1
    assert overview.running_sessions == 1
    assert overview.waiting_confirmation_sessions == 1
    assert overview.replanning_sessions == 0
    assert overview.completed_sessions == 1
    assert overview.failed_sessions == 1
    assert overview.cancelled_sessions == 1
    assert (
        overview.total_sessions
        == overview.pending_sessions
        + overview.running_sessions
        + overview.waiting_confirmation_sessions
        + overview.replanning_sessions
        + overview.interrupted_sessions
        + overview.completed_sessions
        + overview.failed_sessions
        + overview.cancelled_sessions
    )
    assert (
        overview.active_sessions
        == overview.pending_sessions
        + overview.running_sessions
        + overview.waiting_confirmation_sessions
        + overview.replanning_sessions
        + overview.interrupted_sessions
    )
    assert (
        overview.terminal_sessions
        == overview.completed_sessions
        + overview.failed_sessions
        + overview.cancelled_sessions
    )


def test_list_sessions_without_filter_returns_all_newest_first() -> None:
    supervisor = ExecutionSupervisor(clock=_Clock())
    first = supervisor.start(_plan())
    second = supervisor.start(_plan())
    third = supervisor.start(_plan())

    sessions = supervisor.list_sessions()

    assert tuple(session.session_id for session in sessions) == (
        third.session_id,
        second.session_id,
        first.session_id,
    )


def test_list_sessions_can_return_oldest_first_deterministically() -> None:
    supervisor = ExecutionSupervisor(clock=_Clock())
    first = supervisor.start(_plan())
    second = supervisor.start(_plan())

    sessions = supervisor.list_sessions(newest_first=False)

    assert tuple(session.session_id for session in sessions) == (
        first.session_id,
        second.session_id,
    )


def test_list_sessions_filters_by_execution_state() -> None:
    supervisor = ExecutionSupervisor(clock=_Clock())
    pending = supervisor.start(_plan())
    running = supervisor.start(_plan())
    failed = supervisor.start(_plan())
    supervisor.mark_running(running.session_id)
    supervisor.mark_running(failed.session_id)
    supervisor.mark_failed(failed.session_id, "tool failed")

    assert supervisor.list_sessions(state=ExecutionState.PENDING) == (pending,)
    assert supervisor.list_sessions(state=ExecutionState.RUNNING) == (
        supervisor.get_session(running.session_id),
    )
    assert supervisor.list_sessions(state=ExecutionState.FAILED) == (
        supervisor.get_session(failed.session_id),
    )


def test_list_sessions_limit_is_explicit_and_validated() -> None:
    supervisor = ExecutionSupervisor(clock=_Clock())
    supervisor.start(_plan())
    newest = supervisor.start(_plan())
    supervisor.start(_plan())

    assert supervisor.list_sessions(limit=0) == ()
    assert supervisor.list_sessions(limit=1) == (
        supervisor.get_session("execution.session.000003"),
    )
    assert supervisor.list_sessions(limit=2, newest_first=False) == (
        supervisor.get_session("execution.session.000001"),
        newest,
    )
    with pytest.raises(ValueError, match="limit cannot be negative"):
        supervisor.list_sessions(limit=-1)
    with pytest.raises(TypeError, match="limit must be an integer"):
        supervisor.list_sessions(limit=True)  # type: ignore[arg-type]


def test_list_sessions_rejects_free_string_state_filters() -> None:
    supervisor = ExecutionSupervisor(clock=_Clock())

    with pytest.raises(TypeError, match="state must be an ExecutionState"):
        supervisor.list_sessions(state="running")  # type: ignore[arg-type]


def test_list_sessions_returns_immutable_snapshots_not_internal_storage() -> None:
    supervisor = ExecutionSupervisor(clock=_Clock())
    created = supervisor.start(_plan())
    listed = supervisor.list_sessions()
    snapshot = listed[0]

    supervisor.mark_running(created.session_id)

    assert isinstance(listed, tuple)
    assert snapshot.state is ExecutionState.PENDING
    assert supervisor.get_session(created.session_id).state is ExecutionState.RUNNING
    with pytest.raises(FrozenInstanceError):
        snapshot.state = ExecutionState.FAILED  # type: ignore[misc]


def test_latest_session_id_uses_latest_created_session_after_transitions() -> None:
    supervisor = ExecutionSupervisor(clock=_Clock())
    first = supervisor.start(_plan())
    second = supervisor.start(_plan())

    supervisor.mark_running(first.session_id)
    supervisor.mark_completed(first.session_id)
    overview = supervisor.get_overview()

    assert overview.latest_session_id == second.session_id


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


def test_replanning_is_active_and_can_resume_running() -> None:
    supervisor = ExecutionSupervisor(clock=_Clock())
    session = supervisor.start(_plan())

    supervisor.mark_running(session.session_id)
    supervisor.mark_failed(session.session_id, "tool failed", current_step="step_1")
    replanning = supervisor.mark_replanning(
        session.session_id,
        attempt_number=1,
        current_step="step_1",
    )
    overview = supervisor.get_overview()
    replanning_sessions = supervisor.list_sessions(state=ExecutionState.REPLANNING)
    running = supervisor.mark_running(session.session_id)

    assert replanning.state is ExecutionState.REPLANNING
    assert overview.replanning_sessions == 1
    assert overview.active_sessions == 1
    assert replanning_sessions == (replanning,)
    assert running.state is ExecutionState.RUNNING


def test_supervisor_records_step_state_snapshots_and_events() -> None:
    supervisor = ExecutionSupervisor(clock=_Clock())
    session = supervisor.start(_plan())

    supervisor.mark_step_ready(session.session_id, "step_1")
    supervisor.mark_step_started(session.session_id, "step_1")
    supervisor.mark_step_completed(session.session_id, "step_1")
    updated = supervisor.get_session(session.session_id)

    assert updated.step_states["step_1"].state is StepExecutionState.COMPLETED
    assert updated.step_states["step_1"].dependency_ids == ()
    assert [event.event_type for event in updated.events][-3:] == [
        "step_ready",
        "step_started",
        "step_completed",
    ]


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


def test_structured_execution_coordinator_exposes_supervisor_read_models() -> None:
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

    coordinator.handle("run")
    overview = coordinator.get_execution_overview()
    sessions = coordinator.list_execution_sessions(
        state=ExecutionState.COMPLETED,
        limit=1,
    )

    assert overview.completed_sessions == 1
    assert len(sessions) == 1
    assert sessions[0].state is ExecutionState.COMPLETED


def test_non_recoverable_failure_does_not_invoke_replanner() -> None:
    plan = _plan()
    supervisor = ExecutionSupervisor(clock=_Clock())
    replanner = _FakeReplanner()
    executor = _SequenceExecutor(_failed_result(error_code="INVALID_PLAN"))
    coordinator = StructuredExecutionCoordinator(
        planner=_FixedPlanner(plan),  # type: ignore[arg-type]
        validator=_FixedValidator(),  # type: ignore[arg-type]
        executor=executor,  # type: ignore[arg-type]
        execution_supervisor=supervisor,
        execution_replanner=replanner,  # type: ignore[arg-type]
        replan_policy=ReplanPolicy(max_replans_per_session=1),
    )

    response = coordinator.handle("run")
    session = supervisor.get_session("execution.session.000001")

    assert response.status == "failed"
    assert replanner.calls == []
    assert session.state is ExecutionState.FAILED


def test_recoverable_failure_creates_replan_request_with_context() -> None:
    original = _plan(goal="original")
    revised = _plan(goal="revised")
    supervisor = ExecutionSupervisor(clock=_Clock())
    replanner = _FakeReplanner(
        ReplanResult(
            status=ReplanResultStatus.ACCEPTED,
            revised_plan=revised,
            reason=ReplanReason.RECOVERABLE_FAILURE,
        )
    )
    executor = _SequenceExecutor(
        _failed_result(completed_step_result=True),
        _completed_result(),
    )
    coordinator = StructuredExecutionCoordinator(
        planner=_FixedPlanner(original),  # type: ignore[arg-type]
        validator=_FixedValidator(),  # type: ignore[arg-type]
        executor=executor,  # type: ignore[arg-type]
        execution_supervisor=supervisor,
        execution_replanner=replanner,  # type: ignore[arg-type]
        replan_policy=ReplanPolicy(max_replans_per_session=1),
    )

    coordinator.handle("run")
    request = replanner.calls[0]

    assert request.session_id == "execution.session.000001"
    assert request.original_plan is original
    assert request.active_plan is original
    assert request.failed_step == "step_1"
    assert request.error == "tool failed"
    assert request.error_code == "TOOL_EXECUTION_FAILED"
    assert request.partial_results["step_0"] == {"value": "partial"}
    assert request.completed_step_ids == ("step_0",)
    assert request.pending_step_ids == ()
    assert request.blocked_step_ids == ()
    assert request.dependency_graph == {"step_1": ()}
    assert request.attempt_number == 1
    assert request.max_attempts == 1
    assert request.plan_id == plan_signature(original)
    assert request.attempts_performed == 1
    assert request.retries_exhausted is True


def test_partial_replan_executes_fragment_and_pending_without_completed_steps() -> None:
    original = ExecutionPlan(
        goal="partial recovery",
        ordered_steps=(
            ExecutionStep("step_1", "completed", "read_file"),
            ExecutionStep("step_2", "failed", "read_file", ("step_1",)),
            ExecutionStep("step_3", "pending", "read_file", ("step_2",)),
        ),
        estimated_steps=3,
        required_tools=("read_file",),
        detected_risks=(),
        requires_confirmation=False,
    )
    fragment = ExecutionPlan(
        goal="replacement fragment",
        ordered_steps=(ExecutionStep("step_2", "alternative", "read_file"),),
        estimated_steps=1,
        required_tools=("read_file",),
        detected_risks=(),
        requires_confirmation=False,
    )
    first = PlanExecutionResult(
        plan_status=PlanExecutionStatus.FAILED.value,
        success=False,
        failed=True,
        failed_step="step_2",
        current_step="step_2",
        completed_steps=["step_1"],
        failed_steps=["step_2"],
        pending_steps=["step_3"],
        error="temporary failure",
        error_code="TOOL_EXECUTION_FAILED",
        step_results=[
            StepExecutionResult(
                step_id="step_1",
                status="completed",
                success=True,
                tool_name="read_file",
                output={"value": "kept"},
            ),
            StepExecutionResult(
                step_id="step_2",
                status="failed",
                success=False,
                tool_name="read_file",
                error="temporary failure",
                error_code="TOOL_EXECUTION_FAILED",
                metadata={
                    "attempt_number": 3,
                    "max_attempts": 3,
                    "retry_exhausted": True,
                    "retry_scheduled": False,
                },
            ),
        ],
    )
    replanner = _FakeReplanner(
        ReplanResult(
            status=ReplanResultStatus.ACCEPTED,
            revised_plan=fragment,
            reason=ReplanReason.RECOVERABLE_FAILURE,
        )
    )
    executor = _SequenceExecutor(first, _completed_result(completed_steps=["step_2", "step_3"]))
    supervisor = ExecutionSupervisor(clock=_Clock())
    coordinator = StructuredExecutionCoordinator(
        planner=_FixedPlanner(original),  # type: ignore[arg-type]
        validator=_FixedValidator(),  # type: ignore[arg-type]
        executor=executor,  # type: ignore[arg-type]
        execution_supervisor=supervisor,
        execution_replanner=replanner,  # type: ignore[arg-type]
        replan_policy=ReplanPolicy(max_replans_per_session=1),
    )

    response = coordinator.handle("run")
    recovery_plan = executor.calls[1]
    request = replanner.calls[0]
    summary = supervisor.get_summary("execution.session.000001")

    assert response.status == "completed"
    assert [step.id for step in recovery_plan.ordered_steps] == ["step_2", "step_3"]
    assert recovery_plan.ordered_steps[1].depends_on == ("step_2",)
    assert "step_1" not in [step.id for step in recovery_plan.ordered_steps]
    assert request.attempts_performed == 3
    assert request.retries_exhausted is True
    assert summary.replan_status is ReplanRecoveryStatus.SUCCEEDED
    assert summary.replan_count == 1
    assert supervisor.get_session("execution.session.000001").replan_history[
        0
    ].replacement_step_ids == ("step_2",)


def test_replanning_waits_until_normal_retries_are_exhausted() -> None:
    plan = _plan()
    replanner = _FakeReplanner()
    failure = _failed_result()
    failure.step_results.append(
        StepExecutionResult(
            step_id="step_1",
            status="failed",
            success=False,
            tool_name="read_file",
            error="temporary",
            error_code="TOOL_EXECUTION_FAILED",
            metadata={
                "attempt_number": 1,
                "max_attempts": 3,
                "retry_scheduled": True,
                "retry_exhausted": False,
            },
        )
    )
    supervisor = ExecutionSupervisor(clock=_Clock())
    coordinator = StructuredExecutionCoordinator(
        planner=_FixedPlanner(plan),  # type: ignore[arg-type]
        validator=_FixedValidator(),  # type: ignore[arg-type]
        executor=_SequenceExecutor(failure),  # type: ignore[arg-type]
        execution_supervisor=supervisor,
        execution_replanner=replanner,  # type: ignore[arg-type]
        replan_policy=ReplanPolicy(max_replans_per_session=1),
    )

    response = coordinator.handle("run")

    assert response.status == "failed"
    assert replanner.calls == []
    assert (
        "replan_rejected"
        in [
            event.event_type
            for event in supervisor.get_session("execution.session.000001").events
        ]
    )


def test_critical_step_failure_never_requests_replanning() -> None:
    plan = ExecutionPlan(
        goal="critical failure",
        ordered_steps=(
            ExecutionStep("step_1", "critical", "read_file", criticality=1),
            ExecutionStep("step_2", "pending", "read_file", ("step_1",)),
        ),
        estimated_steps=2,
        required_tools=("read_file",),
        detected_risks=(),
        requires_confirmation=False,
    )
    replanner = _FakeReplanner()
    supervisor = ExecutionSupervisor(clock=_Clock())
    coordinator = StructuredExecutionCoordinator(
        planner=_FixedPlanner(plan),  # type: ignore[arg-type]
        validator=_FixedValidator(),  # type: ignore[arg-type]
        executor=_SequenceExecutor(_failed_result()),  # type: ignore[arg-type]
        execution_supervisor=supervisor,
        execution_replanner=replanner,  # type: ignore[arg-type]
        replan_policy=ReplanPolicy(max_replans_per_session=1),
    )

    response = coordinator.handle("run")

    assert response.status == "failed"
    assert replanner.calls == []
    assert supervisor.get_session("execution.session.000001").state is ExecutionState.CANCELLED


def test_optional_failure_is_omitted_without_calling_replanner() -> None:
    plan = ExecutionPlan(
        goal="optional continuation",
        ordered_steps=(
            ExecutionStep("step_1", "optional", "read_file", optional=True),
            ExecutionStep("step_2", "continue", "read_file", ("step_1",)),
        ),
        estimated_steps=2,
        required_tools=("read_file",),
        detected_risks=(),
        requires_confirmation=False,
    )
    replanner = _FakeReplanner()
    executor = _SequenceExecutor(
        PlanExecutionResult(
            plan_status=PlanExecutionStatus.FAILED.value,
            success=False,
            failed=True,
            failed_step="step_1",
            current_step="step_1",
            failed_steps=["step_1"],
            pending_steps=["step_2"],
            error="optional failed",
            error_code="TOOL_EXECUTION_FAILED",
        ),
        _completed_result(completed_steps=["step_2"]),
    )
    supervisor = ExecutionSupervisor(clock=_Clock())
    coordinator = StructuredExecutionCoordinator(
        planner=_FixedPlanner(plan),  # type: ignore[arg-type]
        validator=_FixedValidator(),  # type: ignore[arg-type]
        executor=executor,  # type: ignore[arg-type]
        execution_supervisor=supervisor,
        execution_replanner=replanner,  # type: ignore[arg-type]
        replan_policy=ReplanPolicy(max_replans_per_session=1),
    )

    response = coordinator.handle("run")

    assert response.status == "completed"
    assert replanner.calls == []
    assert [step.id for step in executor.calls[1].ordered_steps] == ["step_2"]
    assert executor.calls[1].ordered_steps[0].depends_on == ()
    assert (
        supervisor.get_session("execution.session.000001")
        .step_states["step_1"]
        .state
        is StepExecutionState.SKIPPED
    )


def test_invalid_replacement_fragment_is_rejected_before_execution() -> None:
    original = _plan(goal="original")
    invalid_fragment = ExecutionPlan(
        goal="invalid fragment",
        ordered_steps=(ExecutionStep("step_1", "alternative", "read_file"),),
        estimated_steps=1,
        required_tools=("read_file",),
        detected_risks=(),
        requires_confirmation=False,
    )
    replanner = _FakeReplanner(
        ReplanResult(
            status=ReplanResultStatus.ACCEPTED,
            revised_plan=invalid_fragment,
            reason=ReplanReason.RECOVERABLE_FAILURE,
        )
    )
    executor = _SequenceExecutor(
        PlanExecutionResult(
            plan_status=PlanExecutionStatus.FAILED.value,
            success=False,
            failed=True,
            failed_step="step_1",
            current_step="step_1",
            failed_steps=["step_1"],
            error="failed",
            error_code="TOOL_EXECUTION_FAILED",
        )
    )
    supervisor = ExecutionSupervisor(clock=_Clock())
    coordinator = StructuredExecutionCoordinator(
        planner=_FixedPlanner(original),  # type: ignore[arg-type]
        validator=_RejectGoalValidator("invalid fragment"),  # type: ignore[arg-type]
        executor=executor,  # type: ignore[arg-type]
        execution_supervisor=supervisor,
        execution_replanner=replanner,  # type: ignore[arg-type]
        replan_policy=ReplanPolicy(max_replans_per_session=1),
    )

    response = coordinator.handle("run")
    summary = supervisor.get_summary("execution.session.000001")

    assert response.status == "failed"
    assert len(executor.calls) == 1
    assert summary.replan_status is ReplanRecoveryStatus.VALIDATION_REJECTED


def test_identical_replacement_is_rejected_to_prevent_replanning_loop() -> None:
    plan = _plan(goal="same plan")
    replanner = _FakeReplanner(
        ReplanResult(
            status=ReplanResultStatus.ACCEPTED,
            revised_plan=plan,
            reason=ReplanReason.RECOVERABLE_FAILURE,
        )
    )
    supervisor = ExecutionSupervisor(clock=_Clock())
    coordinator = StructuredExecutionCoordinator(
        planner=_FixedPlanner(plan),  # type: ignore[arg-type]
        validator=_FixedValidator(),  # type: ignore[arg-type]
        executor=_SequenceExecutor(_failed_result()),  # type: ignore[arg-type]
        execution_supervisor=supervisor,
        execution_replanner=replanner,  # type: ignore[arg-type]
        replan_policy=ReplanPolicy(max_replans_per_session=3),
    )

    response = coordinator.handle("run")
    session = supervisor.get_session("execution.session.000001")
    summary = supervisor.get_summary(session.session_id)

    assert response.status == "failed"
    assert len(replanner.calls) == 1
    assert session.replan_count == 0
    assert summary.replan_status is ReplanRecoveryStatus.NO_SAFE_ALTERNATIVE


def test_successful_replan_updates_active_plan_history_and_completes() -> None:
    original = _plan(goal="original")
    revised = _plan(goal="revised", tool="write_file")
    supervisor = ExecutionSupervisor(clock=_Clock())
    replanner = _FakeReplanner(
        ReplanResult(
            status=ReplanResultStatus.ACCEPTED,
            revised_plan=revised,
            reason=ReplanReason.RECOVERABLE_FAILURE,
        )
    )
    executor = _SequenceExecutor(_failed_result(), _completed_result())
    coordinator = StructuredExecutionCoordinator(
        planner=_FixedPlanner(original),  # type: ignore[arg-type]
        validator=_FixedValidator(),  # type: ignore[arg-type]
        executor=executor,  # type: ignore[arg-type]
        execution_supervisor=supervisor,
        execution_replanner=replanner,  # type: ignore[arg-type]
        replan_policy=ReplanPolicy(max_replans_per_session=1),
    )

    response = coordinator.handle("run")
    session = supervisor.get_session("execution.session.000001")

    assert response.status == "completed"
    assert executor.calls == [original, revised]
    assert session.state is ExecutionState.COMPLETED
    assert session.original_plan is original
    assert session.active_plan is revised
    assert session.plan is revised
    assert session.replan_count == 1
    assert len(session.replan_history) == 1
    assert session.replan_history[0].previous_plan is original
    assert session.replan_history[0].revised_plan is revised
    assert session.last_error == "tool failed"


def test_second_failure_with_limit_one_does_not_replan_again() -> None:
    original = _plan(goal="original")
    revised = _plan(goal="revised")
    supervisor = ExecutionSupervisor(clock=_Clock())
    replanner = _FakeReplanner(
        ReplanResult(
            status=ReplanResultStatus.ACCEPTED,
            revised_plan=revised,
            reason=ReplanReason.RECOVERABLE_FAILURE,
        )
    )
    executor = _SequenceExecutor(
        _failed_result(error="first failure"),
        _failed_result(error="second failure"),
    )
    coordinator = StructuredExecutionCoordinator(
        planner=_FixedPlanner(original),  # type: ignore[arg-type]
        validator=_FixedValidator(),  # type: ignore[arg-type]
        executor=executor,  # type: ignore[arg-type]
        execution_supervisor=supervisor,
        execution_replanner=replanner,  # type: ignore[arg-type]
        replan_policy=ReplanPolicy(max_replans_per_session=1),
    )

    response = coordinator.handle("run")
    session = supervisor.get_session("execution.session.000001")

    assert response.status == "failed"
    assert len(replanner.calls) == 1
    assert session.state is ExecutionState.FAILED
    assert session.replan_count == 1
    assert session.last_error == "second failure"
    assert "replan_limit_reached" in [event.event_type for event in session.events]
    assert (
        supervisor.get_summary(session.session_id).replan_status
        is ReplanRecoveryStatus.LIMIT_REACHED
    )


def test_replan_rejected_keeps_session_failed_and_traced() -> None:
    plan = _plan()
    supervisor = ExecutionSupervisor(clock=_Clock())
    replanner = _FakeReplanner(
        ReplanResult(
            status=ReplanResultStatus.REJECTED,
            reason=ReplanReason.REJECTED,
            error="no revised plan",
        )
    )
    executor = _SequenceExecutor(_failed_result())
    coordinator = StructuredExecutionCoordinator(
        planner=_FixedPlanner(plan),  # type: ignore[arg-type]
        validator=_FixedValidator(),  # type: ignore[arg-type]
        executor=executor,  # type: ignore[arg-type]
        execution_supervisor=supervisor,
        execution_replanner=replanner,  # type: ignore[arg-type]
        replan_policy=ReplanPolicy(max_replans_per_session=1),
    )

    coordinator.handle("run")
    session = supervisor.get_session("execution.session.000001")

    assert session.state is ExecutionState.FAILED
    assert session.replan_count == 0
    assert session.last_error == "no revised plan"
    assert "replan_rejected" in [event.event_type for event in session.events]


def test_replan_planner_error_preserves_original_error_and_records_failure() -> None:
    plan = _plan()
    supervisor = ExecutionSupervisor(clock=_Clock())
    replanner = _FakeReplanner(
        ReplanResult(
            status=ReplanResultStatus.PLANNER_ERROR,
            reason=ReplanReason.PLANNER_ERROR,
            error="planner crashed",
        )
    )
    executor = _SequenceExecutor(_failed_result(error="tool failed first"))
    coordinator = StructuredExecutionCoordinator(
        planner=_FixedPlanner(plan),  # type: ignore[arg-type]
        validator=_FixedValidator(),  # type: ignore[arg-type]
        executor=executor,  # type: ignore[arg-type]
        execution_supervisor=supervisor,
        execution_replanner=replanner,  # type: ignore[arg-type]
        replan_policy=ReplanPolicy(max_replans_per_session=1),
    )

    coordinator.handle("run")
    session = supervisor.get_session("execution.session.000001")

    assert session.state is ExecutionState.FAILED
    assert session.replan_count == 0
    assert session.last_error == "planner crashed"
    assert [event.event_type for event in session.events].count("execution_failed") == 2
    first_failure = next(
        event for event in session.events if event.event_type == "execution_failed"
    )
    assert first_failure.details["error"] == "tool failed first"
    assert "replan_failed" in [event.event_type for event in session.events]


def test_cancelled_execution_never_replans() -> None:
    plan = _plan()
    supervisor = ExecutionSupervisor(clock=_Clock())
    replanner = _FakeReplanner()
    executor = _SequenceExecutor(
        PlanExecutionResult(
            plan_status=PlanExecutionStatus.CANCELLED.value,
            success=False,
            cancelled=True,
            current_step="step_1",
            interruption_reason="cancelled",
            error_code="EXECUTION_CANCELLED",
        )
    )
    coordinator = StructuredExecutionCoordinator(
        planner=_FixedPlanner(plan),  # type: ignore[arg-type]
        validator=_FixedValidator(),  # type: ignore[arg-type]
        executor=executor,  # type: ignore[arg-type]
        execution_supervisor=supervisor,
        execution_replanner=replanner,  # type: ignore[arg-type]
        replan_policy=ReplanPolicy(max_replans_per_session=1),
    )

    response = coordinator.handle("run")
    session = supervisor.get_session("execution.session.000001")

    assert response.status == "cancelled"
    assert replanner.calls == []
    assert session.state is ExecutionState.CANCELLED


def test_cancelled_pending_confirmation_never_replans() -> None:
    plan = _plan(requires_confirmation=True)
    supervisor = ExecutionSupervisor(clock=_Clock())
    replanner = _FakeReplanner()
    coordinator = StructuredExecutionCoordinator(
        planner=_FixedPlanner(plan),  # type: ignore[arg-type]
        validator=_FixedValidator(),  # type: ignore[arg-type]
        executor=_SequenceExecutor(),  # type: ignore[arg-type]
        execution_supervisor=supervisor,
        execution_replanner=replanner,  # type: ignore[arg-type]
        replan_policy=ReplanPolicy(max_replans_per_session=1),
    )

    pending = coordinator.handle("run")
    cancelled = coordinator.cancel_pending()
    session = supervisor.get_session("execution.session.000001")

    assert pending.status == "confirmation_required"
    assert cancelled.status == "pending_execution_cancelled"
    assert replanner.calls == []
    assert session.state is ExecutionState.CANCELLED


def test_max_replans_zero_disables_replanning_explicitly() -> None:
    plan = _plan()
    supervisor = ExecutionSupervisor(clock=_Clock())
    replanner = _FakeReplanner()
    executor = _SequenceExecutor(_failed_result())
    coordinator = StructuredExecutionCoordinator(
        planner=_FixedPlanner(plan),  # type: ignore[arg-type]
        validator=_FixedValidator(),  # type: ignore[arg-type]
        executor=executor,  # type: ignore[arg-type]
        execution_supervisor=supervisor,
        execution_replanner=replanner,  # type: ignore[arg-type]
        replan_policy=ReplanPolicy(max_replans_per_session=0),
    )

    coordinator.handle("run")
    session = supervisor.get_session("execution.session.000001")

    assert replanner.calls == []
    assert session.state is ExecutionState.FAILED
    assert "replan_limit_reached" in [event.event_type for event in session.events]


def test_replan_history_is_immutable_from_outside() -> None:
    original = _plan(goal="original")
    revised = _plan(goal="revised")
    supervisor = ExecutionSupervisor(clock=_Clock())
    replanner = _FakeReplanner(
        ReplanResult(
            status=ReplanResultStatus.ACCEPTED,
            revised_plan=revised,
            reason=ReplanReason.RECOVERABLE_FAILURE,
        )
    )
    coordinator = StructuredExecutionCoordinator(
        planner=_FixedPlanner(original),  # type: ignore[arg-type]
        validator=_FixedValidator(),  # type: ignore[arg-type]
        executor=_SequenceExecutor(_failed_result(), _completed_result()),  # type: ignore[arg-type]
        execution_supervisor=supervisor,
        execution_replanner=replanner,  # type: ignore[arg-type]
        replan_policy=ReplanPolicy(max_replans_per_session=1),
    )

    coordinator.handle("run")
    session = supervisor.get_session("execution.session.000001")

    assert isinstance(session.replan_history, tuple)
    with pytest.raises(FrozenInstanceError):
        session.replan_count = 10  # type: ignore[misc]
    with pytest.raises(AttributeError):
        session.replan_history.append("bad")  # type: ignore[attr-defined]


def test_blocked_execution_records_blocked_step_state() -> None:
    plan = ExecutionPlan(
        goal="blocked",
        ordered_steps=(
            ExecutionStep("step_1", "read", "read_file"),
            ExecutionStep("step_2", "write", "write_file", ("step_1",)),
            ExecutionStep("step_3", "independent", "read_file"),
        ),
        estimated_steps=3,
        required_tools=("read_file", "write_file"),
        detected_risks=(),
        requires_confirmation=False,
    )
    supervisor = ExecutionSupervisor(clock=_Clock())
    coordinator = StructuredExecutionCoordinator(
        planner=_FixedPlanner(plan),  # type: ignore[arg-type]
        validator=_FixedValidator(),  # type: ignore[arg-type]
        executor=_SequenceExecutor(_blocked_result()),  # type: ignore[arg-type]
        execution_supervisor=supervisor,
    )

    response = coordinator.handle("run")
    session = supervisor.get_session("execution.session.000001")

    assert response.status == "blocked"
    assert session.state is ExecutionState.FAILED
    assert session.step_states["step_1"].state is StepExecutionState.COMPLETED
    assert session.step_states["step_2"].state is StepExecutionState.BLOCKED
    assert session.step_states["step_3"].state is StepExecutionState.READY
    assert "execution_blocked" in [event.event_type for event in session.events]


def test_supervisor_records_batch_state_and_history() -> None:
    from core.concurrent_step_executor import ExecutionBatch, ExecutionBatchResult

    supervisor = ExecutionSupervisor(clock=_Clock())
    session = supervisor.start(_plan())
    supervisor.mark_running(session.session_id)
    batch = ExecutionBatch(
        batch_id="batch.1",
        step_ids=("step_1",),
        created_at="2026-01-01T00:00:00+00:00",
        concurrency_limit=1,
        execution_order=("step_1",),
    )
    result = ExecutionBatchResult(batch_id="batch.1", step_results=())

    supervisor.record_execution_batch_created(session.session_id, batch)
    supervisor.mark_execution_batch_started(session.session_id, batch)
    supervisor.record_execution_batch_result(session.session_id, result)

    current = supervisor.get_session(session.session_id)
    assert current.active_batch_id is None
    assert current.active_step_ids == ()
    assert current.last_batch_result is result
    assert current.batch_history == (result,)
    assert "execution_batch_result_recorded" in [
        event.event_type for event in current.events
    ]


def test_supervisor_step_updates_are_thread_safe() -> None:
    plan = ExecutionPlan(
        goal="parallel states",
        ordered_steps=tuple(
            ExecutionStep(f"step_{index}", "run", "tool")
            for index in range(1, 6)
        ),
        estimated_steps=5,
        required_tools=("tool",),
        detected_risks=(),
        requires_confirmation=False,
    )
    supervisor = ExecutionSupervisor(clock=_Clock())
    session = supervisor.start(plan)
    supervisor.mark_running(session.session_id)

    def mark(step_id: str) -> None:
        supervisor.mark_step_started(session.session_id, step_id)
        supervisor.mark_step_completed(session.session_id, step_id)

    threads = [
        Thread(target=mark, args=(step.id,))
        for step in plan.ordered_steps
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    current = supervisor.get_session(session.session_id)
    assert {
        snapshot.state
        for snapshot in current.step_states.values()
    } == {StepExecutionState.COMPLETED}


def test_coordinator_executes_ready_independent_steps_in_parallel_when_opted_in() -> None:
    plan = ExecutionPlan(
        goal="parallel plan",
        ordered_steps=(
            ExecutionStep("a", "root", "tool", parallel_safe=True),
            ExecutionStep("b", "left", "tool", ("a",), parallel_safe=True),
            ExecutionStep("c", "right", "tool", ("a",), parallel_safe=True),
            ExecutionStep("d", "join", "tool", ("b", "c"), parallel_safe=True),
        ),
        estimated_steps=4,
        required_tools=("tool",),
        detected_risks=(),
        requires_confirmation=False,
    )
    active = 0
    max_seen = 0
    lock = Lock()

    def runner(step: ExecutionStep) -> str:
        nonlocal active, max_seen
        with lock:
            active += 1
            max_seen = max(max_seen, active)
        if step.id in {"b", "c"}:
            import time

            time.sleep(0.02)
        with lock:
            active -= 1
        return f"done:{step.id}"

    supervisor = ExecutionSupervisor(clock=_Clock())
    coordinator = StructuredExecutionCoordinator(
        planner=_FixedPlanner(plan),  # type: ignore[arg-type]
        validator=_FixedValidator(),  # type: ignore[arg-type]
        executor=_SequenceExecutor(),  # type: ignore[arg-type]
        execution_supervisor=supervisor,
        concurrency_policy=ExecutionConcurrencyPolicy(
            enabled=True,
            max_concurrency=2,
        ),
        concurrent_step_executor=ConcurrentStepExecutor(runner),
    )

    response = coordinator.handle("run")
    session = supervisor.get_session("execution.session.000001")

    assert response.status == "completed"
    assert response.execution_result is not None
    assert response.execution_result.completed_steps == ["a", "b", "c", "d"]
    assert max_seen == 2
    assert len(session.batch_history) == 3
    assert "execution_batch_started" in [event.event_type for event in session.events]


def test_concurrent_failure_replan_request_includes_batch_context() -> None:
    failed_plan = ExecutionPlan(
        goal="parallel failure",
        ordered_steps=(
            ExecutionStep("a", "root", "tool", parallel_safe=True),
            ExecutionStep("b", "left", "tool", parallel_safe=True),
        ),
        estimated_steps=2,
        required_tools=("tool",),
        detected_risks=(),
        requires_confirmation=False,
    )
    revised_plan = _plan(goal="revised")

    def runner(step: ExecutionStep) -> str:
        if step.id == "b":
            raise RuntimeError("tool broke")
        return step.id

    replanner = _FakeReplanner(
        ReplanResult(
            status=ReplanResultStatus.ACCEPTED,
            revised_plan=revised_plan,
            reason=ReplanReason.RECOVERABLE_FAILURE,
        )
    )
    coordinator = StructuredExecutionCoordinator(
        planner=_FixedPlanner(failed_plan),  # type: ignore[arg-type]
        validator=_FixedValidator(),  # type: ignore[arg-type]
        executor=_SequenceExecutor(_completed_result()),  # type: ignore[arg-type]
        execution_replanner=replanner,  # type: ignore[arg-type]
        replan_policy=ReplanPolicy(max_replans_per_session=1),
        concurrency_policy=ExecutionConcurrencyPolicy(
            enabled=True,
            max_concurrency=2,
            fail_fast=False,
        ),
        concurrent_step_executor=ConcurrentStepExecutor(runner),
    )

    response = coordinator.handle("run")

    assert response.status == "completed"
    assert replanner.calls
    request = replanner.calls[0]
    assert request.batch_id == "execution.session.000001.batch.000001"
    assert request.completed_step_ids == ("a",)
    assert request.failed_step_ids == ("b",)
    assert request.errors_by_step["b"] == "tool broke"
