from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path

import pytest

from core.execution_metrics import ExecutionMetrics
from core.execution_observability_deserializer import (
    ExecutionObservabilityDeserializer,
    ObservabilityDeserializationError,
    ObservabilityJsonError,
    ObservabilitySchemaError,
    ObservabilityValidationError,
)
from core.execution_observability_serializer import (
    OBSERVABILITY_SCHEMA_VERSION,
    ExecutionObservabilitySerializer,
)
from core.execution_plan_executor import ExecutionPlanExecutor
from core.execution_plan_validator import ExecutionPlanValidator
from core.execution_trace import ExecutionTrace, TraceEvent, TraceEventStatus, TraceStatus
from core.planner import ExecutionPlan, ExecutionStep
from tools.base_tool import BaseTool
from tools.registry import ToolRegistry
from tools.tool_context import ToolContext


class SpyTool(BaseTool):
    @property
    def name(self) -> str:
        return "safe_tool"

    @property
    def description(self) -> str:
        return "Safe fake tool."

    def execute(self, context: ToolContext) -> object:
        return {"message": "acción"}


def _ts(seconds: int = 0) -> datetime:
    return datetime(2026, 7, 20, 10, 0, seconds, tzinfo=timezone.utc)


def _event_payload() -> dict[str, object]:
    return {
        "timestamp": "2026-07-20T10:00:00+00:00",
        "component": "ExecutionPlanExecutor",
        "action": "STEP_STARTED",
        "status": "STARTED",
        "duration_ms": None,
        "details": {"unicode": "acción", "items": [1, True, None]},
    }


def _trace_payload() -> dict[str, object]:
    return {
        "schema_version": OBSERVABILITY_SCHEMA_VERSION,
        "execution_id": "exec-1",
        "started_at": "2026-07-20T10:00:00+00:00",
        "finished_at": "2026-07-20T10:00:02+00:00",
        "status": "SUCCESS",
        "duration_ms": 2000,
        "events": [
            _event_payload(),
            {
                "timestamp": "2026-07-20T10:00:01+00:00",
                "component": "ExecutionPlanExecutor",
                "action": "STEP_FINISHED",
                "status": "FINISHED",
                "duration_ms": 25,
                "details": {"step_id": "step_1"},
            },
        ],
    }


def _metrics_payload() -> dict[str, object]:
    return {
        "schema_version": OBSERVABILITY_SCHEMA_VERSION,
        "execution_id": "exec-1",
        "execution_status": "SUCCESS",
        "total_duration_ms": 2000,
        "total_events": 2,
        "started_steps": 1,
        "successful_steps": 1,
        "failed_steps": 0,
        "skipped_steps": 0,
        "success_rate": 1.0,
        "total_step_duration_ms": 25,
        "average_step_duration_ms": 25.0,
        "minimum_step_duration_ms": 25,
        "maximum_step_duration_ms": 25,
        "components": ["ExecutionPlanExecutor"],
        "actions": ["STEP_FINISHED", "STEP_STARTED"],
        "events_by_component": [
            {"component": "ExecutionPlanExecutor", "count": 2}
        ],
        "events_by_action": [
            {"action": "STEP_FINISHED", "count": 1},
            {"action": "STEP_STARTED", "count": 1},
        ],
    }


def _plan() -> ExecutionPlan:
    return ExecutionPlan(
        goal="Execute safely.",
        ordered_steps=(
            ExecutionStep("step_1", "Run safe tool.", "safe_tool"),
        ),
        estimated_steps=1,
        required_tools=("safe_tool",),
        detected_risks=(),
        requires_confirmation=False,
    )


def test_trace_event_from_dict_and_json_data_are_valid() -> None:
    payload = _event_payload()

    event = ExecutionObservabilityDeserializer().trace_event_from_dict(payload)
    event_from_json_data = ExecutionObservabilityDeserializer().trace_event_from_dict(
        json.loads(json.dumps(payload))
    )

    assert isinstance(event, TraceEvent)
    assert event.timestamp == _ts()
    assert event.timestamp.tzinfo is not None
    assert event.component == "ExecutionPlanExecutor"
    assert event.action == "STEP_STARTED"
    assert event.status == TraceEventStatus.STARTED.value
    assert event.duration_ms is None
    assert event.details == {"unicode": "acción", "items": [1, True, None]}
    assert event_from_json_data == event


@pytest.mark.parametrize(
    ("field", "value", "error_type"),
    [
        ("timestamp", "not-a-date", ObservabilityValidationError),
        ("component", None, ObservabilitySchemaError),
        ("component", "", ObservabilityValidationError),
        ("action", None, ObservabilitySchemaError),
        ("action", "", ObservabilityValidationError),
        ("status", "UNKNOWN", ObservabilityValidationError),
        ("duration_ms", -1, ObservabilityValidationError),
        ("duration_ms", True, ObservabilityValidationError),
        ("duration_ms", float("nan"), ObservabilityValidationError),
        ("duration_ms", float("inf"), ObservabilityValidationError),
        ("details", Path("x"), ObservabilityValidationError),
    ],
)
def test_trace_event_rejects_invalid_fields(
    field: str,
    value: object,
    error_type: type[Exception],
) -> None:
    payload = _event_payload()
    if value is None:
        payload.pop(field)
    else:
        payload[field] = value

    with pytest.raises(error_type):
        ExecutionObservabilityDeserializer().trace_event_from_dict(payload)


def test_trace_event_details_are_independent_from_original_payload() -> None:
    payload = _event_payload()

    event = ExecutionObservabilityDeserializer().trace_event_from_dict(payload)
    payload["details"]["unicode"] = "changed"  # type: ignore[index]

    assert event.details["unicode"] == "acción"


def test_trace_from_dict_valid_empty_running_and_with_events() -> None:
    deserializer = ExecutionObservabilityDeserializer()
    empty = {
        "schema_version": OBSERVABILITY_SCHEMA_VERSION,
        "execution_id": "exec-running",
        "started_at": "2026-07-20T10:00:00+00:00",
        "finished_at": None,
        "status": "RUNNING",
        "duration_ms": 0,
        "events": [],
    }

    running = deserializer.trace_from_dict(empty)
    trace = deserializer.trace_from_dict(_trace_payload())

    assert running.status == TraceStatus.RUNNING.value
    assert running.finished_at is None
    assert trace.execution_id == "exec-1"
    assert trace.status == TraceStatus.SUCCESS.value
    assert trace.duration() == 2000
    assert [event.action for event in trace.events] == ["STEP_STARTED", "STEP_FINISHED"]


@pytest.mark.parametrize(
    ("field", "value", "error_type"),
    [
        ("schema_version", None, ObservabilitySchemaError),
        ("schema_version", "2.0", ObservabilitySchemaError),
        ("execution_id", None, ObservabilitySchemaError),
        ("execution_id", "", ObservabilityValidationError),
        ("started_at", "bad", ObservabilityValidationError),
        ("finished_at", "bad", ObservabilityValidationError),
        ("status", "UNKNOWN", ObservabilityValidationError),
        ("events", "bad", ObservabilityValidationError),
    ],
)
def test_trace_rejects_invalid_schema_and_fields(
    field: str,
    value: object,
    error_type: type[Exception],
) -> None:
    payload = _trace_payload()
    if value is None:
        payload.pop(field)
    else:
        payload[field] = value

    with pytest.raises(error_type):
        ExecutionObservabilityDeserializer().trace_from_dict(payload)


def test_trace_rejects_temporal_and_event_consistency_errors() -> None:
    deserializer = ExecutionObservabilityDeserializer()
    finished_before_started = _trace_payload()
    finished_before_started["finished_at"] = "2026-07-20T09:59:59+00:00"
    final_without_finished = _trace_payload()
    final_without_finished["finished_at"] = None
    bad_event = _trace_payload()
    bad_event["events"] = ["bad"]
    inconsistent_duration = _trace_payload()
    inconsistent_duration["duration_ms"] = 1

    for payload in (
        finished_before_started,
        final_without_finished,
        bad_event,
        inconsistent_duration,
    ):
        with pytest.raises(ObservabilityValidationError):
            deserializer.trace_from_dict(payload)


def test_trace_from_json_rejects_invalid_json_and_non_object_root() -> None:
    deserializer = ExecutionObservabilityDeserializer()

    assert deserializer.trace_from_json(json.dumps(_trace_payload())).execution_id == "exec-1"
    with pytest.raises(ObservabilityJsonError) as invalid:
        deserializer.trace_from_json("{")
    assert invalid.value.__cause__ is not None
    for payload in ("[]", "1", '"x"', "null", "true"):
        with pytest.raises(ObservabilitySchemaError):
            deserializer.trace_from_json(payload)


def test_metrics_from_dict_valid_payload() -> None:
    metrics = ExecutionObservabilityDeserializer().metrics_from_dict(_metrics_payload())

    assert isinstance(metrics, ExecutionMetrics)
    assert metrics.execution_id == "exec-1"
    assert metrics.execution_status == TraceStatus.SUCCESS.value
    assert metrics.total_duration_ms == 2000
    assert metrics.total_events == 2
    assert metrics.started_steps == 1
    assert metrics.successful_steps == 1
    assert metrics.failed_steps == 0
    assert metrics.skipped_steps == 0
    assert metrics.success_rate == 1.0
    assert metrics.components == ("ExecutionPlanExecutor",)
    assert metrics.actions == ("STEP_FINISHED", "STEP_STARTED")
    assert metrics.events_by_component == (("ExecutionPlanExecutor", 2),)
    assert metrics.events_by_action == (("STEP_FINISHED", 1), ("STEP_STARTED", 1))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("execution_status", "UNKNOWN"),
        ("total_events", -1),
        ("total_events", True),
        ("success_rate", -0.1),
        ("success_rate", 1.1),
        ("success_rate", float("nan")),
        ("total_duration_ms", -1),
        ("average_step_duration_ms", float("inf")),
        ("components", "bad"),
        ("actions", "bad"),
    ],
)
def test_metrics_rejects_invalid_fields(field: str, value: object) -> None:
    payload = _metrics_payload()
    payload[field] = value

    with pytest.raises(ObservabilityValidationError):
        ExecutionObservabilityDeserializer().metrics_from_dict(payload)


def test_metrics_rejects_inconsistent_counts_and_ranges() -> None:
    deserializer = ExecutionObservabilityDeserializer()
    successful_gt_started = _metrics_payload()
    successful_gt_started["successful_steps"] = 2
    failed_gt_started = _metrics_payload()
    failed_gt_started["failed_steps"] = 2
    minimum_gt_maximum = _metrics_payload()
    minimum_gt_maximum["minimum_step_duration_ms"] = 30
    minimum_gt_maximum["maximum_step_duration_ms"] = 20
    bad_component_count = _metrics_payload()
    bad_component_count["events_by_component"] = [
        {"component": "ExecutionPlanExecutor", "count": -1}
    ]
    bad_action_count = _metrics_payload()
    bad_action_count["events_by_action"] = [
        {"action": "STEP_STARTED", "count": -1}
    ]

    for payload in (
        successful_gt_started,
        failed_gt_started,
        minimum_gt_maximum,
        bad_component_count,
        bad_action_count,
    ):
        with pytest.raises(ObservabilityValidationError):
            deserializer.metrics_from_dict(payload)


def test_metrics_accepts_dict_count_maps_and_json() -> None:
    payload = _metrics_payload()
    payload["events_by_component"] = {"ExecutionPlanExecutor": 2}
    payload["events_by_action"] = {"STEP_FINISHED": 1, "STEP_STARTED": 1}

    metrics = ExecutionObservabilityDeserializer().metrics_from_dict(payload)
    from_json = ExecutionObservabilityDeserializer().metrics_from_json(json.dumps(_metrics_payload()))

    assert metrics.events_by_component == (("ExecutionPlanExecutor", 2),)
    assert from_json.execution_id == "exec-1"


def test_round_trip_trace_event_trace_and_metrics_preserve_unicode_and_timezone() -> None:
    serializer = ExecutionObservabilitySerializer()
    deserializer = ExecutionObservabilityDeserializer()
    trace = ExecutionTrace(
        execution_id="exec-1",
        started_at=_ts(0),
    )
    event = trace.add_event(
        timestamp=_ts(1),
        component="ExecutionPlanExecutor",
        action="STEP_FINISHED",
        status=TraceEventStatus.FINISHED.value,
        duration_ms=15,
        details={"unicode": "acción"},
    )
    trace.finish(TraceStatus.SUCCESS.value, finished_at=_ts(2))
    metrics_payload = serializer.metrics_to_dict(
        ExecutionMetrics(
            execution_id="exec-1",
            execution_status=TraceStatus.SUCCESS.value,
            total_duration_ms=2000,
            total_events=1,
            started_steps=1,
            successful_steps=1,
            failed_steps=0,
            skipped_steps=0,
            success_rate=1.0,
            total_step_duration_ms=15,
            average_step_duration_ms=15.0,
            minimum_step_duration_ms=15,
            maximum_step_duration_ms=15,
            components=("ExecutionPlanExecutor",),
            actions=("STEP_FINISHED",),
            events_by_component=(("ExecutionPlanExecutor", 1),),
            events_by_action=(("STEP_FINISHED", 1),),
        )
    )

    event_round_trip = deserializer.trace_event_from_dict(serializer.trace_event_to_dict(event))
    trace_round_trip = deserializer.trace_from_dict(serializer.trace_to_dict(trace))
    metrics_round_trip = deserializer.metrics_from_dict(metrics_payload)

    assert event_round_trip == event
    assert trace_round_trip.execution_id == trace.execution_id
    assert trace_round_trip.started_at == trace.started_at
    assert trace_round_trip.events[0].details["unicode"] == "acción"
    assert trace_round_trip.events[0].timestamp.tzinfo is not None
    assert metrics_round_trip.execution_id == "exec-1"


def test_deserialized_objects_are_independent_and_payload_is_not_modified() -> None:
    payload = _trace_payload()
    before = deepcopy(payload)

    trace = ExecutionObservabilityDeserializer().trace_from_dict(payload)
    payload["events"][0]["details"]["unicode"] = "changed"  # type: ignore[index]

    assert trace.events[0].details["unicode"] == "acción"
    assert before["events"][0]["details"]["unicode"] == "acción"  # type: ignore[index]
    assert before != payload


def test_errors_have_useful_messages_and_share_base_type() -> None:
    payload = _event_payload()
    payload["duration_ms"] = True

    with pytest.raises(ObservabilityDeserializationError) as error:
        ExecutionObservabilityDeserializer().trace_event_from_dict(payload)

    assert "duration_ms" in str(error.value)


def test_no_pickle_or_file_io_in_deserializer_source() -> None:
    source = Path("core/execution_observability_deserializer.py").read_text(
        encoding="utf-8"
    )

    assert "pickle" not in source
    assert "open(" not in source
    assert "read_text" not in source
    assert "write_text" not in source
    assert "eval(" not in source
    assert "literal_eval" not in source


def test_compatible_with_execution_plan_executor_generated_trace() -> None:
    registry = ToolRegistry()
    registry.register(SpyTool())
    plan = _plan()
    result = ExecutionPlanExecutor(registry).execute(
        plan,
        ExecutionPlanValidator().validate(plan),
    )
    serializer = ExecutionObservabilitySerializer()
    deserializer = ExecutionObservabilityDeserializer()

    trace = deserializer.trace_from_dict(serializer.trace_to_dict(result.trace))  # type: ignore[arg-type]
    metrics = deserializer.metrics_from_dict(serializer.metrics_to_dict(result.metrics))  # type: ignore[arg-type]

    assert trace.execution_id == result.trace.execution_id  # type: ignore[union-attr]
    assert metrics.execution_id == result.metrics.execution_id  # type: ignore[union-attr]
