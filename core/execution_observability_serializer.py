"""Safe JSON-compatible serialization for execution observability data."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
import json
import math
from typing import Any

from core.execution_metrics import ExecutionMetrics
from core.execution_plan_executor import PlanExecutionResult, StepExecutionResult
from core.execution_trace import ExecutionTrace, TraceEvent


OBSERVABILITY_SCHEMA_VERSION = "1.0"


class ExecutionObservabilitySerializer:
    """Serialize traces, metrics and execution results without side effects."""

    def trace_event_to_dict(
        self,
        event: TraceEvent,
    ) -> dict[str, object]:
        """Return a JSON-compatible representation of one trace event."""
        return {
            "timestamp": event.timestamp.isoformat(),
            "component": event.component,
            "action": event.action,
            "status": _to_json_compatible(event.status),
            "duration_ms": event.duration_ms,
            "details": _to_json_compatible(event.details),
        }

    def trace_to_dict(
        self,
        trace: ExecutionTrace,
    ) -> dict[str, object]:
        """Return a JSON-compatible representation of an execution trace."""
        return {
            "schema_version": OBSERVABILITY_SCHEMA_VERSION,
            "execution_id": trace.execution_id,
            "started_at": trace.started_at.isoformat(),
            "finished_at": (
                trace.finished_at.isoformat()
                if trace.finished_at is not None
                else None
            ),
            "status": _to_json_compatible(trace.status),
            "duration_ms": trace.duration(),
            "events": [self.trace_event_to_dict(event) for event in trace.events],
        }

    def metrics_to_dict(
        self,
        metrics: ExecutionMetrics,
    ) -> dict[str, object]:
        """Return a JSON-compatible representation of execution metrics."""
        return {
            "schema_version": OBSERVABILITY_SCHEMA_VERSION,
            "execution_id": metrics.execution_id,
            "execution_status": metrics.execution_status,
            "total_duration_ms": metrics.total_duration_ms,
            "total_events": metrics.total_events,
            "started_steps": metrics.started_steps,
            "successful_steps": metrics.successful_steps,
            "failed_steps": metrics.failed_steps,
            "skipped_steps": metrics.skipped_steps,
            "blocked_steps": metrics.blocked_steps,
            "branches_evaluated": metrics.branches_evaluated,
            "then_branches_selected": metrics.then_branches_selected,
            "else_branches_selected": metrics.else_branches_selected,
            "branches_skipped": metrics.branches_skipped,
            "branches_failed": metrics.branches_failed,
            "success_rate": metrics.success_rate,
            "total_step_duration_ms": metrics.total_step_duration_ms,
            "average_step_duration_ms": metrics.average_step_duration_ms,
            "minimum_step_duration_ms": metrics.minimum_step_duration_ms,
            "maximum_step_duration_ms": metrics.maximum_step_duration_ms,
            "components": list(metrics.components),
            "actions": list(metrics.actions),
            "events_by_component": [
                {"component": component, "count": count}
                for component, count in metrics.events_by_component
            ],
            "events_by_action": [
                {"action": action, "count": count}
                for action, count in metrics.events_by_action
            ],
        }

    def result_to_dict(
        self,
        result: PlanExecutionResult,
    ) -> dict[str, object]:
        """Return a JSON-compatible representation of a plan execution result."""
        return {
            "schema_version": OBSERVABILITY_SCHEMA_VERSION,
            "plan_status": result.plan_status,
            "success": result.success,
            "completed_steps": list(result.completed_steps),
            "failed_step": result.failed_step,
            "skipped_steps": list(result.skipped_steps),
            "step_results": [
                self._step_result_to_dict(step_result)
                for step_result in result.step_results
            ],
            "error": result.error,
            "requires_confirmation": result.requires_confirmation,
            "interrupted": result.interrupted,
            "completed": result.completed,
            "cancelled": result.cancelled,
            "failed": result.failed,
            "blocked": result.blocked,
            "resumable": result.resumable,
            "failed_steps": list(result.failed_steps),
            "blocked_steps": list(result.blocked_steps),
            "pending_steps": list(result.pending_steps),
            "current_step": result.current_step,
            "interruption_reason": result.interruption_reason,
            "failure_reason": result.failure_reason,
            "error_code": result.error_code,
            "started_at": result.started_at,
            "finished_at": result.finished_at,
            "metadata": _to_json_compatible(result.metadata),
            "trace": (
                self.trace_to_dict(result.trace)
                if result.trace is not None
                else None
            ),
            "metrics": (
                self.metrics_to_dict(result.metrics)
                if result.metrics is not None
                else None
            ),
        }

    def to_json(
        self,
        data: object,
        *,
        indent: int | None = None,
    ) -> str:
        """Return valid deterministic JSON for compatible data."""
        return json.dumps(
            _to_json_compatible(data),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            indent=indent,
        )

    def _step_result_to_dict(
        self,
        result: StepExecutionResult,
    ) -> dict[str, object]:
        return {
            "step_id": result.step_id,
            "status": result.status,
            "success": result.success,
            "tool_name": result.tool_name,
            "output": _to_json_compatible(result.output),
            "error": result.error,
            "error_code": result.error_code,
            "interruption_reason": result.interruption_reason,
            "started_at": result.started_at,
            "finished_at": result.finished_at,
            "metadata": _to_json_compatible(result.metadata),
        }


def _to_json_compatible(
    value: object,
) -> object:
    if value is None or isinstance(value, (bool, int, str)):
        return value

    if isinstance(value, float):
        if not math.isfinite(value):
            raise TypeError("Non-finite float values cannot be serialized.")
        return value

    if isinstance(value, datetime):
        return value.isoformat()

    if isinstance(value, Enum):
        return _to_json_compatible(value.value)

    if isinstance(value, dict):
        result: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("Dictionary keys must be strings.")
            result[key] = _to_json_compatible(item)
        return result

    if isinstance(value, (list, tuple)):
        return [_to_json_compatible(item) for item in value]

    if isinstance(value, set):
        converted = [_to_json_compatible(item) for item in value]
        return sorted(converted, key=_stable_sort_key)

    raise TypeError(f"Unsupported observability serialization type: {type(value).__name__}")


def _stable_sort_key(
    value: object,
) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
