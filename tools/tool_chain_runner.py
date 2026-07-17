"""Run deterministic linear chains of Atlas tool intents."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping

from tools.intent_selector import ToolIntent
from tools.single_tool_runner import SingleToolRunner, ToolRunResult


_REFERENCE_PATTERN = re.compile(
    r"\$\{steps\.([A-Za-z0-9_-]+)\.(output|result)(?:\.([A-Za-z0-9_.-]+))?\}"
)


@dataclass(frozen=True, slots=True)
class ToolChainStep:
    """One ordered step in a deterministic tool chain."""

    step_id: str
    tool_name: str
    arguments: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.step_id:
            raise ValueError("Tool chain step id cannot be empty.")

        if not self.tool_name:
            raise ValueError("Tool chain tool name cannot be empty.")

        object.__setattr__(
            self,
            "arguments",
            MappingProxyType(dict(self.arguments)),
        )


@dataclass(frozen=True, slots=True)
class ToolChainStepResult:
    """Result for one chain step."""

    step_id: str
    tool_name: str
    arguments: Mapping[str, Any]
    resolved_arguments: Mapping[str, Any] | None
    result: ToolRunResult

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "arguments",
            MappingProxyType(dict(self.arguments)),
        )

        if self.resolved_arguments is not None:
            object.__setattr__(
                self,
                "resolved_arguments",
                MappingProxyType(dict(self.resolved_arguments)),
            )


@dataclass(frozen=True, slots=True)
class ToolChainResult:
    """Uniform result for a deterministic linear tool chain."""

    success: bool
    status: str
    steps: tuple[ToolChainStepResult, ...]
    failed_step_id: str | None = None
    execution_count: int = 0
    confirmation_id: str | None = None
    error_code: str | None = None
    error_message: str | None = None
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.metadata is not None:
            object.__setattr__(
                self,
                "metadata",
                MappingProxyType(dict(self.metadata)),
            )


@dataclass(frozen=True, slots=True)
class PendingToolChain:
    """Chain paused on a pending tool confirmation."""

    confirmation_id: str
    steps: tuple[ToolChainStep, ...]
    next_index: int
    completed_results: tuple[ToolChainStepResult, ...]


class ToolChainRunner:
    """Run linear tool chains using the existing single-tool runner."""

    def __init__(
        self,
        single_tool_runner: SingleToolRunner,
    ) -> None:
        self._single_tool_runner = single_tool_runner
        self._pending_chains: dict[str, PendingToolChain] = {}

    @property
    def pending_chains(self) -> tuple[PendingToolChain, ...]:
        """Return pending chains without exposing internal storage."""
        return tuple(self._pending_chains.values())

    def run(
        self,
        steps: tuple[ToolChainStep, ...],
    ) -> ToolChainResult:
        """Run a linear chain from the first step."""
        return self._run_from(
            steps=steps,
            start_index=0,
            completed_results=(),
        )

    def confirm(
        self,
        confirmation_id: str,
        response: str,
    ) -> ToolChainResult:
        """Apply one confirmation response to a paused chain."""
        pending = self._pending_chains.get(confirmation_id)

        if pending is None:
            return ToolChainResult(
                success=False,
                status="confirmation_not_found",
                steps=(),
                failed_step_id=None,
                execution_count=0,
                confirmation_id=confirmation_id,
                error_code="confirmation_not_found",
                error_message="confirmation id is not pending for this chain",
            )

        step = pending.steps[pending.next_index]
        outcome = self._single_tool_runner.confirm(confirmation_id, response)
        step_result = ToolChainStepResult(
            step_id=step.step_id,
            tool_name=step.tool_name,
            arguments=step.arguments,
            resolved_arguments=outcome.validated_arguments,
            result=outcome,
        )
        results = pending.completed_results + (step_result,)

        if outcome.status == "invalid_confirmation":
            return ToolChainResult(
                success=False,
                status="invalid_confirmation",
                steps=results,
                failed_step_id=step.step_id,
                execution_count=_execution_count(results),
                confirmation_id=confirmation_id,
                error_code="invalid_confirmation",
                error_message=outcome.error_message,
                metadata=outcome.metadata,
            )

        del self._pending_chains[confirmation_id]

        if outcome.status == "cancelled":
            return ToolChainResult(
                success=False,
                status="cancelled",
                steps=results,
                failed_step_id=step.step_id,
                execution_count=_execution_count(results),
                confirmation_id=confirmation_id,
                error_code="cancelled",
                error_message=outcome.error_message,
            )

        if not outcome.success:
            return ToolChainResult(
                success=False,
                status=outcome.status,
                steps=results,
                failed_step_id=step.step_id,
                execution_count=_execution_count(results),
                confirmation_id=confirmation_id,
                error_code=outcome.error_code,
                error_message=outcome.error_message,
            )

        return self._run_from(
            steps=pending.steps,
            start_index=pending.next_index + 1,
            completed_results=results,
        )

    def _run_from(
        self,
        *,
        steps: tuple[ToolChainStep, ...],
        start_index: int,
        completed_results: tuple[ToolChainStepResult, ...],
    ) -> ToolChainResult:
        results = completed_results

        for index in range(start_index, len(steps)):
            step = steps[index]
            resolved = _resolve_arguments(step.arguments, results)

            if isinstance(resolved, _ReferenceError):
                outcome = ToolRunResult(
                    success=False,
                    status=resolved.status,
                    intent=ToolIntent(step.tool_name, step.arguments),
                    tool_name=step.tool_name,
                    original_arguments=step.arguments,
                    executed=False,
                    execution_count=0,
                    result=None,
                    error_code=resolved.status,
                    error_message=resolved.message,
                    error_field=resolved.field,
                )
                results = results + (
                    ToolChainStepResult(
                        step_id=step.step_id,
                        tool_name=step.tool_name,
                        arguments=step.arguments,
                        resolved_arguments=None,
                        result=outcome,
                    ),
                )
                return _chain_error(results, step.step_id, outcome)

            outcome = self._single_tool_runner.run(
                ToolIntent(
                    step.tool_name,
                    resolved,
                )
            )
            step_result = ToolChainStepResult(
                step_id=step.step_id,
                tool_name=step.tool_name,
                arguments=step.arguments,
                resolved_arguments=resolved,
                result=outcome,
            )
            results = results + (step_result,)

            if outcome.status == "confirmation_required" and outcome.confirmation_id:
                self._pending_chains[outcome.confirmation_id] = PendingToolChain(
                    confirmation_id=outcome.confirmation_id,
                    steps=steps,
                    next_index=index,
                    completed_results=results[:-1],
                )
                return ToolChainResult(
                    success=False,
                    status="confirmation_required",
                    steps=results,
                    failed_step_id=step.step_id,
                    execution_count=_execution_count(results),
                    confirmation_id=outcome.confirmation_id,
                    error_code="confirmation_required",
                    error_message=outcome.error_message,
                    metadata=outcome.metadata,
                )

            if not outcome.success:
                return _chain_error(results, step.step_id, outcome)

        return ToolChainResult(
            success=True,
            status="success",
            steps=results,
            failed_step_id=None,
            execution_count=_execution_count(results),
        )


@dataclass(frozen=True, slots=True)
class _ReferenceError:
    status: str
    message: str
    field: str


def _resolve_arguments(
    arguments: Mapping[str, Any],
    previous_results: tuple[ToolChainStepResult, ...],
) -> dict[str, Any] | _ReferenceError:
    resolved: dict[str, Any] = {}

    for key, value in arguments.items():
        item = _resolve_value(value, previous_results, key)
        if isinstance(item, _ReferenceError):
            return item
        resolved[key] = item

    return resolved


def _resolve_value(
    value: Any,
    previous_results: tuple[ToolChainStepResult, ...],
    field: str,
) -> Any:
    if isinstance(value, str):
        return _resolve_string(value, previous_results, field)

    if isinstance(value, list):
        resolved_list = []
        for item in value:
            resolved = _resolve_value(item, previous_results, field)
            if isinstance(resolved, _ReferenceError):
                return resolved
            resolved_list.append(resolved)
        return resolved_list

    if isinstance(value, dict):
        resolved_dict = {}
        for key, item in value.items():
            resolved = _resolve_value(item, previous_results, field)
            if isinstance(resolved, _ReferenceError):
                return resolved
            resolved_dict[key] = resolved
        return resolved_dict

    return value


def _resolve_string(
    value: str,
    previous_results: tuple[ToolChainStepResult, ...],
    field: str,
) -> Any:
    matches = list(_REFERENCE_PATTERN.finditer(value))

    if not matches:
        return value

    if len(matches) == 1 and matches[0].span() == (0, len(value)):
        return _resolve_reference(matches[0], previous_results, field)

    resolved = value
    for match in matches:
        item = _resolve_reference(match, previous_results, field)
        if isinstance(item, _ReferenceError):
            return item
        resolved = resolved.replace(match.group(0), str(item))

    return resolved


def _resolve_reference(
    match: re.Match[str],
    previous_results: tuple[ToolChainStepResult, ...],
    field: str,
) -> Any:
    step_id = match.group(1)
    path = match.group(3)
    result = _result_for_step(previous_results, step_id)

    if result is None:
        return _ReferenceError(
            status="reference_not_found",
            message=f"step reference does not exist: {step_id}",
            field=field,
        )

    value = result.result.result

    if path is None:
        return value

    if path == "content" and isinstance(value, str):
        return value

    for part in path.split("."):
        if isinstance(value, dict) and part in value:
            value = value[part]
            continue

        if isinstance(value, list) and part.isdigit() and int(part) < len(value):
            value = value[int(part)]
            continue

        if hasattr(value, part):
            value = getattr(value, part)
            continue

        return _ReferenceError(
            status="reference_field_not_found",
            message=f"reference field does not exist: {step_id}.{path}",
            field=field,
        )

    return value


def _result_for_step(
    previous_results: tuple[ToolChainStepResult, ...],
    step_id: str,
) -> ToolChainStepResult | None:
    for result in previous_results:
        if result.step_id == step_id:
            return result

    return None


def _execution_count(results: tuple[ToolChainStepResult, ...]) -> int:
    return sum(result.result.execution_count for result in results)


def _chain_error(
    results: tuple[ToolChainStepResult, ...],
    failed_step_id: str,
    outcome: ToolRunResult,
) -> ToolChainResult:
    return ToolChainResult(
        success=False,
        status=outcome.status,
        steps=results,
        failed_step_id=failed_step_id,
        execution_count=_execution_count(results),
        confirmation_id=outcome.confirmation_id,
        error_code=outcome.error_code,
        error_message=outcome.error_message,
        metadata=outcome.metadata,
    )
