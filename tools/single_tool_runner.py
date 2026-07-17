"""Run one selected and validated Atlas tool intent."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

from tools.argument_schema import (
    ArgumentSchemaNotRegisteredError,
    ArgumentValidationError,
    ArgumentValidationResult,
    ArgumentValidator,
)
from tools.executor import ToolExecutor
from tools.intent_selector import (
    ToolIntent,
    ToolIntentNotSupportedError,
    ToolSelection,
    ToolSelector,
)
from tools.registry import ToolDescriptor, ToolNotRegisteredError
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


@dataclass(frozen=True, slots=True)
class ToolRunResult:
    """Uniform outcome for a single Atlas tool run."""

    success: bool
    status: str
    intent: ToolIntent
    tool_name: str | None = None
    original_arguments: Mapping[str, Any] | None = None
    validated_arguments: Mapping[str, Any] | None = None
    executed: bool = False
    execution_count: int = 0
    result: Any = None
    error_code: str | None = None
    error_message: str | None = None
    error_field: str | None = None
    exception_type: str | None = None
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.original_arguments is not None:
            object.__setattr__(
                self,
                "original_arguments",
                MappingProxyType(dict(self.original_arguments)),
            )

        if self.validated_arguments is not None:
            object.__setattr__(
                self,
                "validated_arguments",
                MappingProxyType(dict(self.validated_arguments)),
            )

        if self.metadata is not None:
            object.__setattr__(
                self,
                "metadata",
                MappingProxyType(dict(self.metadata)),
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
    ) -> ToolRunResult:
        """Run exactly one validated tool and return a uniform result."""
        selection: ToolSelection | None = None

        try:
            selection = self._selector.select(intent)
            validation = self._validator.validate(selection)
            request = self._to_request(selection, validation)
        except ToolIntentNotSupportedError as error:
            return self._error_result(
                intent,
                "unknown_intent",
                error,
                original_arguments=intent.arguments,
            )
        except ToolNotRegisteredError as error:
            return self._error_result(
                intent,
                "tool_not_registered",
                error,
                original_arguments=intent.arguments,
            )
        except ArgumentSchemaNotRegisteredError as error:
            return self._error_result(
                intent,
                "schema_not_registered",
                error,
                tool_name=selection.tool_name if selection is not None else None,
                original_arguments=intent.arguments,
            )
        except ArgumentValidationError as error:
            return self._validation_error_result(
                intent,
                error,
                tool_name=selection.tool_name if selection is not None else None,
            )
        except Exception as error:
            return self._error_result(
                intent,
                "internal_error",
                error,
                original_arguments=intent.arguments,
            )

        self._last_request = request

        if not request.validated:
            error = RuntimeError("Cannot execute a tool request before validation.")
            return self._error_result(
                intent,
                "internal_error",
                error,
                tool_name=request.tool_name,
                original_arguments=request.original_arguments,
                validated_arguments=request.validated_arguments,
            )

        try:
            result = self._executor.execute(
                request.tool_name,
                ToolContext(parameters=dict(request.validated_arguments)),
            )
        except Exception as error:
            self._execution_count += 1
            return self._error_result(
                intent,
                "tool_execution_error",
                error,
                tool_name=request.tool_name,
                original_arguments=request.original_arguments,
                validated_arguments=request.validated_arguments,
                executed=True,
                execution_count=1,
            )

        self._execution_count += 1

        return ToolRunResult(
            success=True,
            status="success",
            intent=intent,
            tool_name=request.tool_name,
            original_arguments=request.original_arguments,
            validated_arguments=request.validated_arguments,
            executed=True,
            execution_count=1,
            result=result,
        )

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

    def _validation_error_result(
        self,
        intent: ToolIntent,
        error: ArgumentValidationError,
        tool_name: str | None,
    ) -> ToolRunResult:
        status = _validation_status(error.reason)

        return self._error_result(
            intent,
            status,
            error,
            tool_name=tool_name,
            original_arguments=intent.arguments,
            error_field=error.field,
            error_message=error.reason,
        )

    def _error_result(
        self,
        intent: ToolIntent,
        status: str,
        error: Exception,
        *,
        tool_name: str | None = None,
        original_arguments: Mapping[str, Any] | None = None,
        validated_arguments: Mapping[str, Any] | None = None,
        executed: bool = False,
        execution_count: int = 0,
        error_field: str | None = None,
        error_message: str | None = None,
    ) -> ToolRunResult:
        return ToolRunResult(
            success=False,
            status=status,
            intent=intent,
            tool_name=tool_name,
            original_arguments=original_arguments,
            validated_arguments=validated_arguments,
            executed=executed,
            execution_count=execution_count,
            result=None,
            error_code=status,
            error_message=error_message or str(error),
            error_field=error_field,
            exception_type=type(error).__name__,
        )


def _validation_status(reason: str) -> str:
    if reason == "required argument is missing":
        return "missing_argument"
    if reason.startswith("expected "):
        return "invalid_argument_type"
    if reason == "unexpected argument":
        return "unexpected_argument"
    if reason == "None is not allowed":
        return "none_not_allowed"

    return "invalid_argument"
