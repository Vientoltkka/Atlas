"""Run one selected and validated Atlas tool intent."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

from tools.argument_schema import ArgumentValidationResult, ArgumentValidator
from tools.executor import ToolExecutor
from tools.intent_selector import ToolIntent, ToolSelection, ToolSelector
from tools.registry import ToolDescriptor
from tools.tool_context import ToolContext


@dataclass(frozen=True, slots=True)
class ValidatedToolRequest:
    """Immutable request ready for exactly one tool execution."""

    intent: ToolIntent
    tool_name: str
    descriptor: ToolDescriptor
    original_arguments: Mapping[str, Any]
    validated_arguments: Mapping[str, Any]
    validated: bool
    executed: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "original_arguments",
            MappingProxyType(dict(self.original_arguments)),
        )
        object.__setattr__(
            self,
            "validated_arguments",
            MappingProxyType(dict(self.validated_arguments)),
        )


class SingleToolRunner:
    """Coordinate selection, validation, and one execution of one tool."""

    def __init__(
        self,
        selector: ToolSelector,
        validator: ArgumentValidator,
        executor: ToolExecutor,
    ) -> None:
        self._selector = selector
        self._validator = validator
        self._executor = executor
        self._last_request: ValidatedToolRequest | None = None
        self._execution_count = 0

    @property
    def last_request(self) -> ValidatedToolRequest | None:
        """Return the latest validated request produced by this runner."""
        return self._last_request

    @property
    def execution_count(self) -> int:
        """Return executions completed by this runner instance."""
        return self._execution_count

    def build_request(
        self,
        intent: ToolIntent,
    ) -> ValidatedToolRequest:
        """Select and validate one intent without executing it."""
        selection = self._selector.select(intent)
        validation = self._validator.validate(selection)

        return self._to_request(selection, validation)

    def run(
        self,
        intent: ToolIntent,
    ) -> Any:
        """Run exactly one validated tool and return the raw tool result."""
        request = self.build_request(intent)
        self._last_request = request

        if not request.validated:
            raise RuntimeError("Cannot execute a tool request before validation.")

        result = self._executor.execute(
            request.tool_name,
            ToolContext(parameters=dict(request.validated_arguments)),
        )
        self._execution_count += 1

        return result

    def _to_request(
        self,
        selection: ToolSelection,
        validation: ArgumentValidationResult,
    ) -> ValidatedToolRequest:
        return ValidatedToolRequest(
            intent=selection.intent,
            tool_name=selection.tool_name,
            descriptor=selection.descriptor,
            original_arguments=validation.original_arguments,
            validated_arguments=validation.validated_arguments,
            validated=validation.valid,
            executed=False,
        )
