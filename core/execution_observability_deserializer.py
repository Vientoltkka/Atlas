"""Safe deserialization for execution observability payloads."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
import json
import math
from typing import Any

from core.execution_metrics import ExecutionMetrics
from core.execution_observability_serializer import OBSERVABILITY_SCHEMA_VERSION
from core.execution_trace import ExecutionTrace, TraceEvent, TraceEventStatus, TraceStatus


class ObservabilityDeserializationError(Exception):
    """Base error for observability deserialization failures."""


class ObservabilityJsonError(ObservabilityDeserializationError):
    """Raised when JSON cannot be parsed safely."""


class ObservabilitySchemaError(ObservabilityDeserializationError):
    """Raised when the payload schema is unsupported or incomplete."""


class ObservabilityValidationError(ObservabilityDeserializationError):
    """Raised when a payload value is invalid for the target object."""


class ExecutionObservabilityDeserializer:
    """Deserialize validated observability payloads into domain objects."""

    def parse_json(
        self,
        payload: str,
    ) -> object:
        """Parse a JSON string and return its root value."""
        if not isinstance(payload, str):
            raise ObservabilityJsonError("JSON payload must be a string.")
        try:
            return json.loads(payload)
        except json.JSONDecodeError as error:
            raise ObservabilityJsonError("Invalid JSON payload.") from error

    def trace_event_from_dict(
        self,
        data: dict[str, object],
    ) -> TraceEvent:
        """Deserialize one TraceEvent from a validated mapping."""
        payload = _require_dict_root(data, "TraceEvent")
        return TraceEvent(
            timestamp=_timestamp(payload, "timestamp"),
            component=_non_empty_str(payload, "component"),
            action=_non_empty_str(payload, "action"),
            status=_event_status(payload, "status"),
            duration_ms=_optional_non_negative_number(payload, "duration_ms"),
            details=_safe_details(_required_field(payload, "details"), "details"),
        )

    def trace_from_dict(
        self,
        data: dict[str, object],
    ) -> ExecutionTrace:
        """Deserialize one ExecutionTrace from a validated mapping."""
        payload = _require_dict_root(data, "ExecutionTrace")
        _validate_schema_version(payload)
        execution_id = _non_empty_str(payload, "execution_id")
        started_at = _timestamp(payload, "started_at")
        finished_at = _optional_timestamp(payload, "finished_at")
        status = _trace_status(payload, "status")
        serialized_duration = _optional_non_negative_number(payload, "duration_ms")
        events_payload = _list(payload, "events")
        events = [self.trace_event_from_dict(_event_dict(item, index)) for index, item in enumerate(events_payload)]

        if finished_at is not None and finished_at < started_at:
            raise ObservabilityValidationError(
                "finished_at cannot be earlier than started_at."
            )
        if status != TraceStatus.RUNNING.value and finished_at is None:
            raise ObservabilityValidationError(
                f"finished_at is required when status is {status}."
            )

        trace = ExecutionTrace(
            execution_id=execution_id,
            started_at=started_at,
            finished_at=finished_at,
            status=status,
            events=events,
        )
        if serialized_duration is not None and finished_at is not None:
            calculated = trace.duration()
            if int(serialized_duration) != calculated:
                raise ObservabilityValidationError(
                    "duration_ms is inconsistent with started_at and finished_at."
                )
        return trace

    def metrics_from_dict(
        self,
        data: dict[str, object],
    ) -> ExecutionMetrics:
        """Deserialize one ExecutionMetrics object from a validated mapping."""
        payload = _require_dict_root(data, "ExecutionMetrics")
        _validate_schema_version(payload)
        execution_status = _trace_status(payload, "execution_status")
        started_steps = _non_negative_int(payload, "started_steps")
        successful_steps = _non_negative_int(payload, "successful_steps")
        failed_steps = _non_negative_int(payload, "failed_steps")
        skipped_steps = _optional_non_negative_int(payload, "skipped_steps", default=0)
        minimum = _optional_non_negative_number(payload, "minimum_step_duration_ms")
        maximum = _optional_non_negative_number(payload, "maximum_step_duration_ms")
        if minimum is not None and maximum is not None and minimum > maximum:
            raise ObservabilityValidationError(
                "minimum_step_duration_ms cannot be greater than maximum_step_duration_ms."
            )
        if successful_steps > started_steps:
            raise ObservabilityValidationError(
                "successful_steps cannot be greater than started_steps."
            )
        if failed_steps > started_steps:
            raise ObservabilityValidationError(
                "failed_steps cannot be greater than started_steps."
            )

        success_rate = _bounded_rate(payload, "success_rate")
        return ExecutionMetrics(
            execution_id=_non_empty_str(payload, "execution_id"),
            execution_status=execution_status,
            total_duration_ms=_non_negative_int(payload, "total_duration_ms"),
            total_events=_non_negative_int(payload, "total_events"),
            started_steps=started_steps,
            successful_steps=successful_steps,
            failed_steps=failed_steps,
            skipped_steps=skipped_steps,
            success_rate=success_rate,
            total_step_duration_ms=_non_negative_int(payload, "total_step_duration_ms"),
            average_step_duration_ms=_non_negative_float(payload, "average_step_duration_ms"),
            minimum_step_duration_ms=minimum,
            maximum_step_duration_ms=maximum,
            components=_str_tuple(payload, "components"),
            actions=_str_tuple(payload, "actions"),
            events_by_component=_count_pairs(payload, "events_by_component", "component"),
            events_by_action=_count_pairs(payload, "events_by_action", "action"),
        )

    def trace_from_json(
        self,
        payload: str,
    ) -> ExecutionTrace:
        """Deserialize an ExecutionTrace from a JSON object string."""
        return self.trace_from_dict(_json_object(self.parse_json(payload), "ExecutionTrace"))

    def metrics_from_json(
        self,
        payload: str,
    ) -> ExecutionMetrics:
        """Deserialize ExecutionMetrics from a JSON object string."""
        return self.metrics_from_dict(_json_object(self.parse_json(payload), "ExecutionMetrics"))


def _require_dict_root(
    data: object,
    label: str,
) -> dict[str, object]:
    if not isinstance(data, dict):
        raise ObservabilitySchemaError(f"{label} payload must be an object.")
    return data


def _json_object(
    data: object,
    label: str,
) -> dict[str, object]:
    if not isinstance(data, dict):
        raise ObservabilitySchemaError(f"{label} JSON root must be an object.")
    return data


def _validate_schema_version(
    payload: dict[str, object],
) -> None:
    version = _required_field(payload, "schema_version")
    if not isinstance(version, str):
        raise ObservabilitySchemaError("schema_version must be a string.")
    if version != OBSERVABILITY_SCHEMA_VERSION:
        raise ObservabilitySchemaError(
            f"Unsupported schema_version: {version}."
        )


def _required_field(
    payload: dict[str, object],
    field: str,
) -> object:
    if field not in payload:
        raise ObservabilitySchemaError(f"Missing required field: {field}.")
    return payload[field]


def _non_empty_str(
    payload: dict[str, object],
    field: str,
) -> str:
    value = _required_field(payload, field)
    if not isinstance(value, str):
        raise ObservabilityValidationError(f"{field} must be a string.")
    if not value:
        raise ObservabilityValidationError(f"{field} cannot be empty.")
    return value


def _timestamp(
    payload: dict[str, object],
    field: str,
) -> datetime:
    value = _required_field(payload, field)
    if not isinstance(value, str):
        raise ObservabilityValidationError(f"{field} must be an ISO 8601 string.")
    try:
        return datetime.fromisoformat(value)
    except ValueError as error:
        raise ObservabilityValidationError(
            f"{field} is not a valid ISO 8601 timestamp: {value}."
        ) from error


def _optional_timestamp(
    payload: dict[str, object],
    field: str,
) -> datetime | None:
    value = _required_field(payload, field)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ObservabilityValidationError(f"{field} must be an ISO 8601 string or null.")
    try:
        return datetime.fromisoformat(value)
    except ValueError as error:
        raise ObservabilityValidationError(
            f"{field} is not a valid ISO 8601 timestamp: {value}."
        ) from error


def _event_status(
    payload: dict[str, object],
    field: str,
) -> str:
    value = _required_field(payload, field)
    try:
        return TraceEventStatus(value).value
    except ValueError as error:
        raise ObservabilityValidationError(
            f"{field} has unknown TraceEventStatus: {value}."
        ) from error


def _trace_status(
    payload: dict[str, object],
    field: str,
) -> str:
    value = _required_field(payload, field)
    try:
        return TraceStatus(value).value
    except ValueError as error:
        raise ObservabilityValidationError(
            f"{field} has unknown TraceStatus: {value}."
        ) from error


def _optional_non_negative_number(
    payload: dict[str, object],
    field: str,
) -> int | float | None:
    value = _required_field(payload, field)
    if value is None:
        return None
    return _non_negative_number_value(value, field)


def _non_negative_int(
    payload: dict[str, object],
    field: str,
) -> int:
    value = _required_field(payload, field)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ObservabilityValidationError(f"{field} must be an integer.")
    if value < 0:
        raise ObservabilityValidationError(f"{field} cannot be negative.")
    return value


def _optional_non_negative_int(
    payload: dict[str, object],
    field: str,
    *,
    default: int,
) -> int:
    if field not in payload:
        return default
    return _non_negative_int(payload, field)


def _non_negative_float(
    payload: dict[str, object],
    field: str,
) -> float:
    value = _non_negative_number_value(_required_field(payload, field), field)
    return float(value)


def _bounded_rate(
    payload: dict[str, object],
    field: str,
) -> float:
    value = _non_negative_float(payload, field)
    if value > 1.0:
        raise ObservabilityValidationError(f"{field} cannot be greater than 1.0.")
    return value


def _non_negative_number_value(
    value: object,
    field: str,
) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ObservabilityValidationError(f"{field} must be a finite number.")
    if not math.isfinite(float(value)):
        raise ObservabilityValidationError(f"{field} must be finite.")
    if value < 0:
        raise ObservabilityValidationError(f"{field} cannot be negative.")
    return value


def _list(
    payload: dict[str, object],
    field: str,
) -> list[object]:
    value = _required_field(payload, field)
    if not isinstance(value, list):
        raise ObservabilityValidationError(f"{field} must be a list.")
    return value


def _event_dict(
    value: object,
    index: int,
) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ObservabilityValidationError(f"events[{index}] must be an object.")
    return value


def _str_tuple(
    payload: dict[str, object],
    field: str,
) -> tuple[str, ...]:
    values = _list(payload, field)
    result: list[str] = []
    for index, value in enumerate(values):
        if not isinstance(value, str):
            raise ObservabilityValidationError(f"{field}[{index}] must be a string.")
        result.append(value)
    return tuple(result)


def _count_pairs(
    payload: dict[str, object],
    field: str,
    key_name: str,
) -> tuple[tuple[str, int], ...]:
    value = _required_field(payload, field)
    if isinstance(value, dict):
        result: list[tuple[str, int]] = []
        for key, count in value.items():
            if not isinstance(key, str):
                raise ObservabilityValidationError(f"{field} contains a non-string key.")
            result.append((key, _count_value(count, f"{field}.{key}")))
        return tuple(result)
    if not isinstance(value, list):
        raise ObservabilityValidationError(f"{field} must be a list or object.")

    result: list[tuple[str, int]] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise ObservabilityValidationError(f"{field}[{index}] must be an object.")
        key = _non_empty_str(item, key_name)
        count = _count_value(_required_field(item, "count"), f"{field}[{index}].count")
        result.append((key, count))
    return tuple(result)


def _count_value(
    value: object,
    field: str,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ObservabilityValidationError(f"{field} must be an integer.")
    if value < 0:
        raise ObservabilityValidationError(f"{field} cannot be negative.")
    return value


def _safe_details(
    value: object,
    field: str,
) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ObservabilityValidationError(f"{field} must be an object.")
    return _safe_dict(value, field)


def _safe_json_value(
    value: object,
    field: str,
) -> object:
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ObservabilityValidationError(f"{field} must be finite.")
        return value
    if isinstance(value, list):
        return [
            _safe_json_value(item, f"{field}[{index}]")
            for index, item in enumerate(value)
        ]
    if isinstance(value, dict):
        return _safe_dict(value, field)
    if isinstance(value, Enum):
        raise ObservabilityValidationError(f"{field} contains unsupported enum value.")
    raise ObservabilityValidationError(
        f"{field} contains unsupported type: {type(value).__name__}."
    )


def _safe_dict(
    value: dict[object, object],
    field: str,
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise ObservabilityValidationError(f"{field} contains a non-string key.")
        result[key] = _safe_json_value(item, f"{field}.{key}")
    return result
