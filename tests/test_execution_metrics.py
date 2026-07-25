from __future__ import annotations

from datetime import datetime, timezone

from core.execution_metrics import ExecutionMetricsCalculator
from core.execution_trace import ExecutionTrace, TraceEventStatus, TraceStatus


def _trace(
    *,
    status: str = TraceStatus.SUCCESS.value,
) -> ExecutionTrace:
    started = datetime(2026, 7, 20, 10, 0, 0, tzinfo=timezone.utc)
    finished = datetime(2026, 7, 20, 10, 0, 3, tzinfo=timezone.utc)
    trace = ExecutionTrace(execution_id="exec-1", started_at=started)
    trace.finish(status, finished_at=finished)
    return trace


def test_calculates_empty_trace_without_division_by_zero() -> None:
    trace = _trace()

    metrics = ExecutionMetricsCalculator().calculate(trace)

    assert metrics.execution_id == "exec-1"
    assert metrics.execution_status == TraceStatus.SUCCESS.value
    assert metrics.total_duration_ms == 3000
    assert metrics.total_events == 0
    assert metrics.started_steps == 0
    assert metrics.successful_steps == 0
    assert metrics.failed_steps == 0
    assert metrics.skipped_steps == 0
    assert metrics.success_rate == 0.0
    assert metrics.total_step_duration_ms == 0
    assert metrics.average_step_duration_ms == 0.0
    assert metrics.minimum_step_duration_ms is None
    assert metrics.maximum_step_duration_ms is None
    assert metrics.components == ()
    assert metrics.actions == ()
    assert metrics.events_by_component == ()
    assert metrics.events_by_action == ()


def test_calculates_successful_execution_metrics() -> None:
    trace = _trace()
    trace.add_event(
        component="ExecutionPlanExecutor",
        action="STEP_STARTED",
        status=TraceEventStatus.STARTED.value,
    )
    trace.add_event(
        component="ExecutionPlanExecutor",
        action="STEP_FINISHED",
        status=TraceEventStatus.FINISHED.value,
        duration_ms=40,
    )

    metrics = ExecutionMetricsCalculator().calculate(trace)

    assert metrics.started_steps == 1
    assert metrics.successful_steps == 1
    assert metrics.failed_steps == 0
    assert metrics.skipped_steps == 0
    assert metrics.success_rate == 1.0
    assert metrics.total_step_duration_ms == 40
    assert metrics.average_step_duration_ms == 40.0
    assert metrics.minimum_step_duration_ms == 40
    assert metrics.maximum_step_duration_ms == 40
    assert metrics.components == ("ExecutionPlanExecutor",)
    assert metrics.actions == ("STEP_FINISHED", "STEP_STARTED")


def test_calculates_failed_and_mixed_step_metrics() -> None:
    trace = _trace(status=TraceStatus.FAILED.value)
    trace.add_event(
        component="ExecutionPlanExecutor",
        action="STEP_STARTED",
        status=TraceEventStatus.STARTED.value,
    )
    trace.add_event(
        component="ExecutionPlanExecutor",
        action="STEP_FINISHED",
        status=TraceEventStatus.FINISHED.value,
        duration_ms=20,
    )
    trace.add_event(
        component="ToolExecutor",
        action="STEP_STARTED",
        status=TraceEventStatus.STARTED.value,
    )
    trace.add_event(
        component="ToolExecutor",
        action="STEP_FAILED",
        status=TraceEventStatus.FAILED.value,
        duration_ms=60,
    )

    metrics = ExecutionMetricsCalculator().calculate(trace)

    assert metrics.execution_status == TraceStatus.FAILED.value
    assert metrics.total_events == 4
    assert metrics.started_steps == 2
    assert metrics.successful_steps == 1
    assert metrics.failed_steps == 1
    assert metrics.skipped_steps == 0
    assert metrics.success_rate == 0.5
    assert metrics.total_step_duration_ms == 80
    assert metrics.average_step_duration_ms == 40.0
    assert metrics.minimum_step_duration_ms == 20
    assert metrics.maximum_step_duration_ms == 60
    assert metrics.events_by_component == (
        ("ExecutionPlanExecutor", 2),
        ("ToolExecutor", 2),
    )
    assert metrics.events_by_action == (
        ("STEP_FAILED", 1),
        ("STEP_FINISHED", 1),
        ("STEP_STARTED", 2),
    )


def test_ignores_missing_step_durations() -> None:
    trace = _trace(status=TraceStatus.FAILED.value)
    trace.add_event(
        component="ExecutionPlanExecutor",
        action="STEP_STARTED",
        status=TraceEventStatus.STARTED.value,
    )
    trace.add_event(
        component="ExecutionPlanExecutor",
        action="STEP_FAILED",
        status=TraceEventStatus.FAILED.value,
    )

    metrics = ExecutionMetricsCalculator().calculate(trace)

    assert metrics.failed_steps == 1
    assert metrics.skipped_steps == 0
    assert metrics.total_step_duration_ms == 0
    assert metrics.average_step_duration_ms == 0.0
    assert metrics.minimum_step_duration_ms is None
    assert metrics.maximum_step_duration_ms is None


def test_calculator_does_not_modify_trace_or_event_order() -> None:
    trace = _trace()
    trace.add_event(
        component="B",
        action="STEP_STARTED",
        status=TraceEventStatus.STARTED.value,
    )
    trace.add_event(
        component="A",
        action="STEP_FINISHED",
        status=TraceEventStatus.FINISHED.value,
        duration_ms=1,
    )
    before = tuple(trace.events)

    ExecutionMetricsCalculator().calculate(trace)

    assert tuple(trace.events) == before
    assert [event.component for event in trace.events] == ["B", "A"]


def test_goal_verification_metrics_are_counted() -> None:
    trace = ExecutionTrace("exec-goal")
    trace.add_event(
        component="GoalVerifier",
        action="goal_verification_started",
        status="STARTED",
    )
    trace.add_event(
        component="GoalVerifier",
        action="goal_verification_failed",
        status="FAILED",
    )
    trace.add_event(
        component="GoalVerifier",
        action="goal_missing_output",
        status="FAILED",
        details={"output_name": "entries"},
    )
    trace.add_event(
        component="GoalVerifier",
        action="goal_output_invalid",
        status="FAILED",
        details={"output_name": "entries"},
    )
    trace.finish("FAILED")

    metrics = ExecutionMetricsCalculator().calculate(trace)

    assert metrics.goals_verified == 1
    assert metrics.goals_failed == 1
    assert metrics.goals_satisfied == 0
    assert metrics.missing_required_outputs == 1
    assert metrics.output_validation_failures == 1


def test_skipped_steps_are_counted_but_excluded_from_success_rate() -> None:
    trace = _trace()
    trace.add_event(
        component="ExecutionPlanExecutor",
        action="execution_step_skipped",
        status=TraceEventStatus.FINISHED.value,
    )

    metrics = ExecutionMetricsCalculator().calculate(trace)

    assert metrics.skipped_steps == 1
    assert metrics.successful_steps == 0
    assert metrics.failed_steps == 0
    assert metrics.success_rate == 0.0
