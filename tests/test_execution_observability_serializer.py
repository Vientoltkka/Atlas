from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import pytest

from core.execution_metrics import ExecutionMetrics, ExecutionMetricsCalculator
from core.execution_observability_serializer import (
    OBSERVABILITY_SCHEMA_VERSION,
    ExecutionObservabilitySerializer,
)
from core.execution_plan_executor import (
    ExecutionPlanExecutor,
    PlanExecutionResult,
    PlanExecutionStatus,
)
from core.execution_trace import ExecutionTrace, TraceEvent, TraceEventStatus, TraceStatus
from core.execution_plan_validator import ExecutionPlanValidator
from core.planner import ExecutionPlan, ExecutionStep
from tools.base_tool import BaseTool
from tools.registry import ToolRegistry
from tools.tool_context import ToolContext


class SpyTool(BaseTool):
    def __init__(self, calls: list[str], output: object = "ok") -> None:
        self._calls = calls
        self._output = output

    @property
    def name(self) -> str:
        return "safe_tool"

    @property
    def description(self) -> str:
        return "Safe fake tool."

    def execute(self, context: ToolContext) -> object:
        self._calls.append(context.step_id or "")
        return self._output


def _timestamp(seconds: int = 0) -> datetime:
    return datetime(2026, 7, 20, 10, 0, seconds, tzinfo=timezone.utc)


def _trace() -> ExecutionTrace:
    trace = ExecutionTrace(execution_id="exec-1", started_at=_timestamp(0))
    trace.add_event(
        timestamp=_timestamp(0),
        component="ExecutionPlanExecutor",
        action="STEP_STARTED",
        status=TraceEventStatus.STARTED,
        details={
            "step_id": "step_1",
            "unicode": "acción",
            "nested": {"items": (1, True, None)},
            "tags": {"b", "a"},
        },
    )
    trace.add_event(
        timestamp=_timestamp(1),
        component="ExecutionPlanExecutor",
        action="STEP_FINISHED",
        status=TraceEventStatus.FINISHED.value,
        duration_ms=100,
        details={"step_id": "step_1"},
    )
    trace.finish(TraceStatus.SUCCESS.value, finished_at=_timestamp(2))
    return trace


def _metrics() -> ExecutionMetrics:
    return ExecutionMetricsCalculator().calculate(_trace())


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


def test_serializes_trace_event_with_iso_timestamp_enum_value_and_details() -> None:
    details = {"duration": None, "values": (1, 2), "labels": {"z", "a"}}
    event = TraceEvent(
        timestamp=_timestamp(),
        component="ExecutionPlanExecutor",
        action="STEP_STARTED",
        status=TraceEventStatus.STARTED,
        duration_ms=None,
        details=details,
    )

    payload = ExecutionObservabilitySerializer().trace_event_to_dict(event)

    assert payload == {
        "timestamp": "2026-07-20T10:00:00+00:00",
        "component": "ExecutionPlanExecutor",
        "action": "STEP_STARTED",
        "status": "STARTED",
        "duration_ms": None,
        "details": {
            "duration": None,
            "values": [1, 2],
            "labels": ["a", "z"],
        },
    }
    assert details == {"duration": None, "values": (1, 2), "labels": {"z", "a"}}


def test_serializes_empty_trace_with_schema_and_duration() -> None:
    trace = ExecutionTrace(execution_id="exec-1", started_at=_timestamp(0))
    trace.finish(TraceStatus.SUCCESS.value, finished_at=_timestamp(1))

    payload = ExecutionObservabilitySerializer().trace_to_dict(trace)

    assert payload["schema_version"] == OBSERVABILITY_SCHEMA_VERSION
    assert payload["execution_id"] == "exec-1"
    assert payload["started_at"] == "2026-07-20T10:00:00+00:00"
    assert payload["finished_at"] == "2026-07-20T10:00:01+00:00"
    assert payload["status"] == "SUCCESS"
    assert payload["duration_ms"] == 1000
    assert payload["events"] == []


def test_serializes_trace_with_events_preserving_order() -> None:
    payload = ExecutionObservabilitySerializer().trace_to_dict(_trace())

    events = payload["events"]
    assert isinstance(events, list)
    assert [event["action"] for event in events] == [
        "STEP_STARTED",
        "STEP_FINISHED",
    ]
    assert events[0]["details"]["unicode"] == "acción"


def test_serializes_metrics_with_json_compatible_collections_and_counts() -> None:
    payload = ExecutionObservabilitySerializer().metrics_to_dict(_metrics())

    assert payload["schema_version"] == OBSERVABILITY_SCHEMA_VERSION
    assert payload["execution_id"] == "exec-1"
    assert payload["execution_status"] == "SUCCESS"
    assert payload["total_events"] == 2
    assert payload["started_steps"] == 1
    assert payload["successful_steps"] == 1
    assert payload["failed_steps"] == 0
    assert payload["success_rate"] == 1.0
    assert payload["components"] == ["ExecutionPlanExecutor"]
    assert payload["actions"] == ["STEP_FINISHED", "STEP_STARTED"]
    assert payload["events_by_component"] == [
        {"component": "ExecutionPlanExecutor", "count": 2}
    ]
    assert payload["events_by_action"] == [
        {"action": "STEP_FINISHED", "count": 1},
        {"action": "STEP_STARTED", "count": 1},
    ]


def test_serializes_plan_execution_result_with_trace_and_metrics() -> None:
    calls: list[str] = []
    registry = ToolRegistry()
    registry.register(SpyTool(calls, output={"message": "hola"}))
    plan = _plan()

    result = ExecutionPlanExecutor(registry).execute(
        plan,
        ExecutionPlanValidator().validate(plan),
    )
    payload = ExecutionObservabilitySerializer().result_to_dict(result)

    assert payload["schema_version"] == OBSERVABILITY_SCHEMA_VERSION
    assert payload["plan_status"] == PlanExecutionStatus.COMPLETED.value
    assert payload["trace"] is not None
    assert payload["metrics"] is not None
    assert payload["trace"]["execution_id"] == payload["metrics"]["execution_id"]
    assert payload["step_results"][0]["output"] == {"message": "hola"}
    assert calls == ["step_1"]


def test_serializes_plan_execution_result_without_trace_or_metrics() -> None:
    result = PlanExecutionResult(
        plan_status=PlanExecutionStatus.FAILED.value,
        success=False,
        error="failed",
    )

    payload = ExecutionObservabilitySerializer().result_to_dict(result)

    assert payload["schema_version"] == OBSERVABILITY_SCHEMA_VERSION
    assert payload["trace"] is None
    assert payload["metrics"] is None
    assert payload["error"] == "failed"


def test_to_json_generates_valid_unicode_json_with_optional_indent() -> None:
    serializer = ExecutionObservabilitySerializer()
    payload = {"texto": "acción", "items": (1, 2)}

    compact = serializer.to_json(payload)
    pretty = serializer.to_json(payload, indent=2)

    assert json.loads(compact) == {"items": [1, 2], "texto": "acción"}
    assert json.loads(pretty) == json.loads(compact)
    assert "acción" in compact
    assert "\n  " in pretty


def test_rejects_nan_and_infinity() -> None:
    serializer = ExecutionObservabilitySerializer()

    with pytest.raises(TypeError):
        serializer.to_json({"value": float("nan")})
    with pytest.raises(TypeError):
        serializer.to_json({"value": float("inf")})


def test_rejects_unsupported_types_and_non_string_dict_keys() -> None:
    serializer = ExecutionObservabilitySerializer()

    with pytest.raises(TypeError):
        serializer.to_json({"path": Path("x")})
    with pytest.raises(TypeError):
        serializer.to_json({1: "bad"})


def test_serialized_structures_are_independent_and_deterministic() -> None:
    trace = _trace()
    serializer = ExecutionObservabilitySerializer()

    first = serializer.trace_to_dict(trace)
    second = serializer.trace_to_dict(trace)
    first["events"][0]["details"]["unicode"] = "changed"  # type: ignore[index]

    assert serializer.trace_to_dict(trace) == second
    assert trace.events[0].details["unicode"] == "acción"
    assert serializer.to_json(second) == serializer.to_json(serializer.trace_to_dict(trace))


def test_serializer_does_not_write_files_or_modify_inputs(tmp_path: Path) -> None:
    trace = _trace()
    metrics = _metrics()
    trace_events_before = tuple(trace.events)

    serializer = ExecutionObservabilitySerializer()
    serializer.trace_to_dict(trace)
    serializer.metrics_to_dict(metrics)
    serializer.to_json(serializer.trace_to_dict(trace))

    assert tuple(trace.events) == trace_events_before
    assert list(tmp_path.iterdir()) == []
