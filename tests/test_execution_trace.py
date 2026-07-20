from __future__ import annotations

from datetime import datetime, timezone

import pytest

from core.execution_trace import ExecutionTrace, TraceEvent, TraceEventStatus, TraceStatus


def test_execution_trace_starts_running() -> None:
    started = datetime(2026, 7, 20, 10, 0, 0, tzinfo=timezone.utc)

    trace = ExecutionTrace(execution_id="exec-1", started_at=started)

    assert trace.execution_id == "exec-1"
    assert trace.started_at == started
    assert trace.finished_at is None
    assert trace.status == TraceStatus.RUNNING.value
    assert trace.events == []


def test_trace_event_requires_known_status() -> None:
    timestamp = datetime(2026, 7, 20, 10, 0, 0, tzinfo=timezone.utc)

    event = TraceEvent(
        timestamp=timestamp,
        component="ExecutionPlanExecutor",
        action="STEP_STARTED",
        status=TraceEventStatus.STARTED.value,
        details={"step_id": "step_1"},
    )

    assert event.timestamp == timestamp
    assert event.component == "ExecutionPlanExecutor"
    assert event.action == "STEP_STARTED"
    assert event.status == TraceEventStatus.STARTED.value
    with pytest.raises(ValueError):
        TraceEvent(timestamp, "component", "ACTION", "UNKNOWN")


def test_add_event_preserves_chronological_order() -> None:
    first = datetime(2026, 7, 20, 10, 0, 0, tzinfo=timezone.utc)
    second = datetime(2026, 7, 20, 10, 0, 1, tzinfo=timezone.utc)
    trace = ExecutionTrace(execution_id="exec-1", started_at=first)

    trace.add_event(
        timestamp=first,
        component="ExecutionPlanExecutor",
        action="STEP_STARTED",
        status=TraceEventStatus.STARTED.value,
    )
    trace.add_event(
        timestamp=second,
        component="ExecutionPlanExecutor",
        action="STEP_FINISHED",
        status=TraceEventStatus.FINISHED.value,
        duration_ms=1000,
    )

    assert [event.action for event in trace.events] == [
        "STEP_STARTED",
        "STEP_FINISHED",
    ]
    assert trace.events[0].timestamp <= trace.events[1].timestamp


def test_finish_changes_status_and_duration() -> None:
    started = datetime(2026, 7, 20, 10, 0, 0, tzinfo=timezone.utc)
    finished = datetime(2026, 7, 20, 10, 0, 2, 500000, tzinfo=timezone.utc)
    trace = ExecutionTrace(execution_id="exec-1", started_at=started)

    trace.finish(TraceStatus.SUCCESS.value, finished_at=finished)

    assert trace.status == TraceStatus.SUCCESS.value
    assert trace.finished_at == finished
    assert trace.duration() == 2500


def test_finish_rejects_running_as_terminal_status() -> None:
    trace = ExecutionTrace(execution_id="exec-1")

    with pytest.raises(ValueError):
        trace.finish(TraceStatus.RUNNING.value)


def test_failed_event_records_safe_error_details() -> None:
    timestamp = datetime(2026, 7, 20, 10, 0, 0, tzinfo=timezone.utc)
    trace = ExecutionTrace(execution_id="exec-1", started_at=timestamp)

    event = trace.add_event(
        timestamp=timestamp,
        component="ExecutionPlanExecutor",
        action="STEP_FAILED",
        status=TraceEventStatus.FAILED.value,
        duration_ms=12,
        details={"step_id": "step_1", "error_code": "TOOL_EXCEPTION"},
    )

    assert event.status == TraceEventStatus.FAILED.value
    assert event.duration_ms == 12
    assert event.details == {"step_id": "step_1", "error_code": "TOOL_EXCEPTION"}
