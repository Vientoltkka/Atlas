"""Declarative output definitions for Atlas execution plans."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import math
from types import MappingProxyType, ModuleType
from typing import Any, Mapping

from core.execution_variable_reference import ExecutionVariableReference
from core.parameter_resolver import (
    ExecutionValueProvider,
    ParameterResolutionError,
    ParameterResolver,
)
from core.step_output_reference import StepOutputReference


MAX_EXECUTION_OUTPUT_DEPTH = 16
MAX_EXECUTION_OUTPUT_NODES = 256


class ExecutionPlanOutputError(ValueError):
    """Base error for declarative execution plan outputs."""

    error_code = "EXECUTION_PLAN_OUTPUT_ERROR"


class InvalidExecutionPlanOutputError(ExecutionPlanOutputError):
    """Raised when an output definition contains unsupported structure."""

    error_code = "INVALID_EXECUTION_PLAN_OUTPUT"


class ExecutionPlanOutputTooDeepError(InvalidExecutionPlanOutputError):
    """Raised when an output definition exceeds the depth limit."""

    error_code = "EXECUTION_PLAN_OUTPUT_TOO_DEEP"


class ExecutionPlanOutputTooLargeError(InvalidExecutionPlanOutputError):
    """Raised when an output definition exceeds the node limit."""

    error_code = "EXECUTION_PLAN_OUTPUT_TOO_LARGE"


class ExecutionPlanOutputResolutionError(ExecutionPlanOutputError):
    """Raised when a validated output cannot be resolved at runtime."""

    error_code = "EXECUTION_PLAN_OUTPUT_RESOLUTION_FAILED"


class ExecutionPlanOutputStepNotFoundError(ExecutionPlanOutputResolutionError):
    """Raised when an output references an unavailable step result."""

    error_code = "EXECUTION_PLAN_OUTPUT_STEP_NOT_FOUND"


class ExecutionPlanOutputStepUnavailableError(ExecutionPlanOutputResolutionError):
    """Raised when an output references a step without usable output."""

    error_code = "EXECUTION_PLAN_OUTPUT_STEP_UNAVAILABLE"


class ExecutionPlanOutputVariableNotFoundError(ExecutionPlanOutputResolutionError):
    """Raised when an output references an unavailable execution variable."""

    error_code = "EXECUTION_PLAN_OUTPUT_VARIABLE_NOT_FOUND"


class ExecutionPlanOutputPathError(ExecutionPlanOutputResolutionError):
    """Raised when an output reference path cannot be navigated."""

    error_code = "EXECUTION_PLAN_OUTPUT_PATH_ERROR"


class ExecutionPlanOutputUnsafeValueError(ExecutionPlanOutputResolutionError):
    """Raised when a resolved output value is not safe to expose."""

    error_code = "EXECUTION_PLAN_OUTPUT_UNSAFE_VALUE"


@dataclass(frozen=True, slots=True)
class ExecutionPlanOutputStats:
    """Safe statistics for one output definition."""

    node_count: int
    reference_count: int
    step_reference_count: int
    variable_reference_count: int
    output_kind: str


@dataclass(frozen=True, slots=True)
class _Counter:
    nodes: int = 0


class ExecutionPlanOutput:
    """Immutable declarative output definition for an ExecutionPlan."""

    def __init__(self, value: object) -> None:
        counter = {"nodes": 0}
        normalized = _normalize_output_value(
            value,
            path="$",
            depth=0,
            counter=counter,
        )
        object.__setattr__(self, "_definition", normalized)
        object.__setattr__(self, "_node_count", counter["nodes"])

    def resolve(
        self,
        context: ExecutionValueProvider,
    ) -> object:
        """Resolve this definition against the final execution context."""
        try:
            resolved = ParameterResolver().resolve_value(
                self._definition,
                context,
                path="execution_plan_output",
            )
            return _safe_runtime_value(resolved, path="$", depth=0, counter={"nodes": 0})
        except ParameterResolutionError as error:
            raise _resolution_error_from_message(str(error)) from error
        except InvalidExecutionPlanOutputError as error:
            raise ExecutionPlanOutputUnsafeValueError(str(error)) from error

    def as_definition(self) -> object:
        """Return a defensive copy of the normalized definition."""
        return _copy_output_value(self._definition)

    def stats(self) -> ExecutionPlanOutputStats:
        """Return safe structural statistics without exposing values."""
        refs = _reference_counts(self._definition)
        return ExecutionPlanOutputStats(
            node_count=self._node_count,
            reference_count=refs["step"] + refs["variable"],
            step_reference_count=refs["step"],
            variable_reference_count=refs["variable"],
            output_kind=_output_kind(self._definition),
        )


def copy_execution_plan_output(
    output: ExecutionPlanOutput | object | None,
) -> ExecutionPlanOutput | None:
    """Normalize an optional plan output at model boundaries."""
    if output is None:
        return None
    if isinstance(output, ExecutionPlanOutput):
        return ExecutionPlanOutput(output.as_definition())
    return ExecutionPlanOutput(output)


def _normalize_output_value(
    value: object,
    *,
    path: str,
    depth: int,
    counter: dict[str, int],
) -> object:
    if depth > MAX_EXECUTION_OUTPUT_DEPTH:
        raise ExecutionPlanOutputTooDeepError(
            f"path={path} reason=execution plan output exceeds depth {MAX_EXECUTION_OUTPUT_DEPTH}"
        )
    counter["nodes"] += 1
    if counter["nodes"] > MAX_EXECUTION_OUTPUT_NODES:
        raise ExecutionPlanOutputTooLargeError(
            f"path={path} reason=execution plan output exceeds {MAX_EXECUTION_OUTPUT_NODES} nodes"
        )

    if isinstance(value, (StepOutputReference, ExecutionVariableReference)):
        return value
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise InvalidExecutionPlanOutputError(
                f"path={path} reason=non-finite float values are not supported"
            )
        return value
    if isinstance(value, Mapping):
        normalized: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise InvalidExecutionPlanOutputError(
                    f"path={path} reason=output mapping keys must be strings"
                )
            if not key.strip():
                raise InvalidExecutionPlanOutputError(
                    f"path={path} reason=output mapping keys cannot be empty"
                )
            normalized[key] = _normalize_output_value(
                item,
                path=f"{path}.{key}",
                depth=depth + 1,
                counter=counter,
            )
        return MappingProxyType(normalized)
    if isinstance(value, list):
        return tuple(
            _normalize_output_value(
                item,
                path=f"{path}[{index}]",
                depth=depth + 1,
                counter=counter,
            )
            for index, item in enumerate(value)
        )
    if isinstance(value, tuple):
        return tuple(
            _normalize_output_value(
                item,
                path=f"{path}[{index}]",
                depth=depth + 1,
                counter=counter,
            )
            for index, item in enumerate(value)
        )

    if isinstance(value, ModuleType):
        type_name = "module"
    elif callable(value):
        type_name = "callable"
    elif isinstance(value, type):
        type_name = "class"
    else:
        type_name = type(value).__name__
    raise InvalidExecutionPlanOutputError(
        f"path={path} reason=unsupported output value type {type_name}"
    )


def _safe_runtime_value(
    value: object,
    *,
    path: str,
    depth: int,
    counter: dict[str, int],
) -> object:
    normalized = _normalize_output_value(
        value,
        path=path,
        depth=depth,
        counter=counter,
    )
    return _copy_output_value(normalized)


def _copy_output_value(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _copy_output_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_copy_output_value(item) for item in value]
    if isinstance(value, list):
        return [_copy_output_value(item) for item in value]
    return deepcopy(value)


def _reference_counts(value: object) -> dict[str, int]:
    counts = {"step": 0, "variable": 0}

    def visit(item: object) -> None:
        if isinstance(item, StepOutputReference):
            counts["step"] += 1
            return
        if isinstance(item, ExecutionVariableReference):
            counts["variable"] += 1
            return
        if isinstance(item, Mapping):
            for nested in item.values():
                visit(nested)
            return
        if isinstance(item, (list, tuple)):
            for nested in item:
                visit(nested)

    visit(value)
    return counts


def _output_kind(value: object) -> str:
    if isinstance(value, StepOutputReference):
        return "step_reference"
    if isinstance(value, ExecutionVariableReference):
        return "variable_reference"
    if isinstance(value, Mapping):
        return "mapping"
    if isinstance(value, (list, tuple)):
        return "sequence"
    return type(value).__name__


def _resolution_error_from_message(message: str) -> ExecutionPlanOutputResolutionError:
    lowered = message.lower()
    if "variable" in lowered and "not found" in lowered:
        return ExecutionPlanOutputVariableNotFoundError(message)
    if "step" in lowered and ("not found" in lowered or "not produced" in lowered):
        return ExecutionPlanOutputStepNotFoundError(message)
    if "skipped" in lowered or "failed" in lowered or "not completed" in lowered:
        return ExecutionPlanOutputStepUnavailableError(message)
    if "path" in lowered or "index" in lowered or "field" in lowered:
        return ExecutionPlanOutputPathError(message)
    return ExecutionPlanOutputResolutionError(message)
