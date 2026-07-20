from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from core.execution_history import ExecutionHistory, ExecutionHistoryEntry
from core.execution_metrics import ExecutionMetricsCalculator
from core.execution_plan_executor import ExecutionPlanExecutor, PlanExecutionResult
from core.execution_plan_validator import ExecutionPlanValidator
from core.execution_trace import ExecutionTrace, TraceEventStatus, TraceStatus
from core.planner import ExecutionPlan, ExecutionStep
from tools.base_tool import BaseTool
from tools.registry import ToolRegistry
from tools.tool_context import ToolContext


class SpyTool(BaseTool):
    def __init__(self, calls: list[str]) -> None:
        self._calls = calls

    @property
    def name(self) -> str:
        return "safe_tool"

    @property
    def description(self) -> str:
        return "Safe fake tool."

    def execute(self, context: ToolContext) -> object:
        self._calls.append(context.step_id or "")
        return "ok"


def _timestamp(seconds: int) -> datetime:
    return datetime(2026, 7, 20, 10, 0, seconds, tzinfo=timezone.utc)


def _result(
    execution_id: str,
    *,
    status: str = TraceStatus.SUCCESS.value,
    duration_ms: int = 100,
    with_metrics: bool = True,
) -> PlanExecutionResult:
    trace = ExecutionTrace(
        execution_id=execution_id,
        started_at=_timestamp(0),
    )
    trace.add_event(
        timestamp=_timestamp(0),
        component="ExecutionPlanExecutor",
        action="STEP_STARTED",
        status=TraceEventStatus.STARTED.value,
    )
    terminal_action = (
        "STEP_FINISHED"
        if status == TraceStatus.SUCCESS.value
        else "STEP_FAILED"
    )
    terminal_status = (
        TraceEventStatus.FINISHED.value
        if status == TraceStatus.SUCCESS.value
        else TraceEventStatus.FAILED.value
    )
    trace.add_event(
        timestamp=_timestamp(1),
        component="ExecutionPlanExecutor",
        action=terminal_action,
        status=terminal_status,
        duration_ms=duration_ms,
    )
    trace.finish(status, finished_at=_timestamp(0) + timedelta(milliseconds=duration_ms))
    metrics = ExecutionMetricsCalculator().calculate(trace) if with_metrics else None
    return PlanExecutionResult(
        plan_status="completed" if status == TraceStatus.SUCCESS.value else "failed",
        success=status == TraceStatus.SUCCESS.value,
        trace=trace,
        metrics=metrics,
    )


def _plan() -> ExecutionPlan:
    return ExecutionPlan(
        goal="Execute safely.",
        ordered_steps=(
            ExecutionStep(
                id="step_1",
                description="Run safe tool.",
                tool="safe_tool",
            ),
        ),
        estimated_steps=1,
        required_tools=("safe_tool",),
        detected_risks=(),
        requires_confirmation=False,
    )


def test_history_requires_positive_capacity() -> None:
    with pytest.raises(ValueError):
        ExecutionHistory(max_entries=0)
    with pytest.raises(ValueError):
        ExecutionHistory(max_entries=-1)


def test_add_rejects_result_without_trace() -> None:
    history = ExecutionHistory()

    with pytest.raises(ValueError):
        history.add(PlanExecutionResult(plan_status="completed", success=True))


def test_add_stores_entry_and_calculates_missing_metrics() -> None:
    history = ExecutionHistory()
    result = _result("exec-1", with_metrics=False)

    entry = history.add(result)

    assert isinstance(entry, ExecutionHistoryEntry)
    assert entry.execution_id == "exec-1"
    assert entry.status == TraceStatus.SUCCESS.value
    assert entry.started_at == result.trace.started_at  # type: ignore[union-attr]
    assert entry.finished_at == result.trace.finished_at  # type: ignore[union-attr]
    assert entry.trace is result.trace
    assert entry.metrics.execution_id == "exec-1"
    assert entry.result is result
    assert history.count() == 1


def test_capacity_discards_oldest_entry() -> None:
    history = ExecutionHistory(max_entries=2)

    history.add(_result("exec-1"))
    history.add(_result("exec-2"))
    history.add(_result("exec-3"))

    assert history.get("exec-1") is None
    assert tuple(entry.execution_id for entry in history) == ("exec-2", "exec-3")
    assert history.count() == 2


def test_duplicate_execution_id_replaces_and_moves_to_latest() -> None:
    history = ExecutionHistory(max_entries=3)

    history.add(_result("exec-1", duration_ms=100))
    history.add(_result("exec-2", duration_ms=200))
    replacement = history.add(_result("exec-1", duration_ms=300))

    assert history.count() == 2
    assert history.latest() is replacement
    assert tuple(entry.execution_id for entry in history) == ("exec-2", "exec-1")
    assert history.get("exec-1").metrics.total_step_duration_ms == 300  # type: ignore[union-attr]


def test_latest_recent_get_and_clear() -> None:
    history = ExecutionHistory(max_entries=3)
    first = history.add(_result("exec-1"))
    second = history.add(_result("exec-2"))

    assert history.latest() is second
    assert history.recent(0) == ()
    assert history.recent(10) == (second, first)
    assert history.get("exec-1") is first
    with pytest.raises(ValueError):
        history.recent(-1)

    recent = history.recent(2)
    assert isinstance(recent, tuple)
    history.clear()
    assert history.latest() is None
    assert history.count() == 0


def test_counts_successful_failed_and_ignores_cancelled_for_failed_count() -> None:
    history = ExecutionHistory()

    history.add(_result("success", status=TraceStatus.SUCCESS.value))
    history.add(_result("failed", status=TraceStatus.FAILED.value))
    history.add(_result("cancelled", status=TraceStatus.CANCELLED.value))

    assert history.count() == 3
    assert history.count_successful() == 1
    assert history.count_failed() == 1


def test_slowest_returns_newest_entry_on_duration_tie() -> None:
    history = ExecutionHistory()
    history.add(_result("exec-1", duration_ms=100))
    first_slowest = history.add(_result("exec-2", duration_ms=300))
    newest_tie = history.add(_result("exec-3", duration_ms=300))

    assert history.slowest() is newest_tie
    assert history.slowest() is not first_slowest


def test_executor_optionally_records_final_result_once() -> None:
    calls: list[str] = []
    history = ExecutionHistory()
    registry = ToolRegistry()
    registry.register(SpyTool(calls))
    plan = _plan()

    result = ExecutionPlanExecutor(
        registry,
        execution_history=history,
    ).execute(plan, ExecutionPlanValidator().validate(plan))

    assert calls == ["step_1"]
    assert history.count() == 1
    assert history.latest() is not None
    assert history.latest().result is result  # type: ignore[union-attr]
    assert history.latest().trace is result.trace  # type: ignore[union-attr]
    assert history.latest().metrics is result.metrics  # type: ignore[union-attr]


def test_history_does_not_modify_result_trace_or_metrics() -> None:
    history = ExecutionHistory()
    result = _result("exec-1")
    trace_events = tuple(result.trace.events)  # type: ignore[union-attr]
    metrics = result.metrics

    entry = history.add(result)
    entry_snapshot = history.recent(1)

    assert tuple(result.trace.events) == trace_events  # type: ignore[union-attr]
    assert result.metrics is metrics
    assert history.count() == 1
    assert history.latest() is entry
    assert entry_snapshot == (entry,)


def test_result_replacement_object_is_not_modified() -> None:
    history = ExecutionHistory()
    result = _result("exec-1")
    modified = replace(result, metrics=None)

    history.add(modified)

    assert modified.metrics is None
    assert history.latest().metrics is not None  # type: ignore[union-attr]
