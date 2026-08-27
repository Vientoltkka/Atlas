"""Run one selected and validated Atlas tool intent."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping
from uuid import uuid4

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
    confirmation_id: str | None = None
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


@dataclass(frozen=True, slots=True)
class PendingToolConfirmation:
    """A selected and validated tool request waiting for explicit confirmation."""

    confirmation_id: str
    request: ValidatedToolRequest
    prompt: str


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
        self._pending_confirmations: dict[str, PendingToolConfirmation] = {}

    @property
    def last_request(self) -> ValidatedToolRequest | None:
        """Return the latest validated request produced by this runner."""
        return self._last_request

    @property
    def execution_count(self) -> int:
        """Return executions completed by this runner instance."""
        return self._execution_count

    @property
    def pending_confirmations(self) -> tuple[PendingToolConfirmation, ...]:
        """Return pending confirmations without exposing internal storage."""
        return tuple(self._pending_confirmations.values())

    def pending_confirmation(
        self,
        confirmation_id: str,
    ) -> PendingToolConfirmation | None:
        """Return one pending confirmation by id without exposing storage."""
        return self._pending_confirmations.get(confirmation_id)

    def run_registered_tool(
        self,
        tool_name: str,
        arguments: Mapping[str, Any],
    ) -> ToolRunResult | None:
        """Run one exact registered tool through an existing intent mapping.

        Returning ``None`` means this runner has no current intent mapping for
        that tool; callers must not select a different tool as a fallback.
        """
        for action in self._selector.supported_intents():
            try:
                selection = self._selector.select(ToolIntent(action))
            except (ToolIntentNotSupportedError, ToolNotRegisteredError):
                continue
            if selection.tool_name == tool_name:
                return self.run(ToolIntent(action, dict(arguments)))
        return None

    def cancel_pending_confirmation(
        self,
        confirmation_id: str,
    ) -> PendingToolConfirmation | None:
        """Remove one pending confirmation without executing it."""
        return self._pending_confirmations.pop(confirmation_id, None)

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

        if request.descriptor.requires_confirmation or self._requires_authorization(
            request.tool_name
        ):
            return self._confirmation_required_result(request)

        return self._execute_request(request)

    def confirm(
        self,
        confirmation_id: str,
        response: str,
    ) -> ToolRunResult:
        """Apply a user response to one pending confirmation."""
        pending = self._pending_confirmations.get(confirmation_id)

        if pending is None:
            return ToolRunResult(
                success=False,
                status="confirmation_not_found",
                intent=ToolIntent("confirmation.unknown"),
                executed=False,
                execution_count=0,
                result=None,
                error_code="confirmation_not_found",
                error_message="confirmation id is not pending",
                confirmation_id=confirmation_id,
            )

        decision = _confirmation_decision(response)

        if decision is None:
            return ToolRunResult(
                success=False,
                status="invalid_confirmation",
                intent=pending.request.intent,
                tool_name=pending.request.tool_name,
                original_arguments=pending.request.original_arguments,
                validated_arguments=pending.request.validated_arguments,
                executed=False,
                execution_count=0,
                result=None,
                error_code="invalid_confirmation",
                error_message="confirmation response must be yes or no",
                confirmation_id=confirmation_id,
                metadata={"prompt": pending.prompt},
            )

        del self._pending_confirmations[confirmation_id]

        if decision is False:
            return ToolRunResult(
                success=False,
                status="cancelled",
                intent=pending.request.intent,
                tool_name=pending.request.tool_name,
                original_arguments=pending.request.original_arguments,
                validated_arguments=pending.request.validated_arguments,
                executed=False,
                execution_count=0,
                result=None,
                error_code="cancelled",
                error_message="operation cancelled by user",
                confirmation_id=confirmation_id,
            )

        return self._execute_request(pending.request, confirmation_id=confirmation_id)

    def reissue_pending_confirmation(
        self,
        confirmation_id: str,
        arguments: Mapping[str, Any],
    ) -> ToolRunResult:
        """Replace a pending request with revalidated arguments and a new id."""
        pending = self._pending_confirmations.get(confirmation_id)
        if pending is None:
            return ToolRunResult(
                success=False,
                status="confirmation_not_found",
                intent=ToolIntent("confirmation.unknown"),
                executed=False,
                execution_count=0,
                result=None,
                error_code="confirmation_not_found",
                error_message="confirmation id is not pending",
                confirmation_id=confirmation_id,
            )

        try:
            request = self.build_request(
                ToolIntent(pending.request.intent.action, dict(arguments))
            )
        except ToolIntentNotSupportedError as error:
            return self._error_result(
                pending.request.intent,
                "unknown_intent",
                error,
                original_arguments=arguments,
            )
        except ToolNotRegisteredError as error:
            return self._error_result(
                pending.request.intent,
                "tool_not_registered",
                error,
                original_arguments=arguments,
            )
        except ArgumentSchemaNotRegisteredError as error:
            return self._error_result(
                pending.request.intent,
                "schema_not_registered",
                error,
                tool_name=pending.request.tool_name,
                original_arguments=arguments,
            )
        except ArgumentValidationError as error:
            return self._validation_error_result(
                ToolIntent(pending.request.intent.action, dict(arguments)),
                error,
                tool_name=pending.request.tool_name,
            )

        del self._pending_confirmations[confirmation_id]
        self._last_request = request

        if not (
            request.descriptor.requires_confirmation
            or self._requires_authorization(request.tool_name)
        ):
            return ToolRunResult(
                success=False,
                status="confirmation_not_required",
                intent=request.intent,
                tool_name=request.tool_name,
                original_arguments=request.original_arguments,
                validated_arguments=request.validated_arguments,
                executed=False,
                execution_count=0,
                result=None,
                error_code="confirmation_not_required",
                error_message="modified pending request no longer requires confirmation",
            )

        return self._confirmation_required_result(request)

    def _execute_request(
        self,
        request: ValidatedToolRequest,
        confirmation_id: str | None = None,
    ) -> ToolRunResult:
        try:
            context = ToolContext(parameters=dict(request.validated_arguments))
            if confirmation_id is not None and self._requires_authorization(request.tool_name):
                result = self._executor.execute(
                    request.tool_name,
                    context,
                    authorization=self._executor.authorize(request.tool_name),
                )
            else:
                result = self._executor.execute(request.tool_name, context)
        except Exception as error:
            self._execution_count += 1
            return self._error_result(
                request.intent,
                "tool_execution_error",
                error,
                tool_name=request.tool_name,
                original_arguments=request.original_arguments,
                validated_arguments=request.validated_arguments,
                executed=True,
                execution_count=1,
                confirmation_id=confirmation_id,
            )

        self._execution_count += 1

        return ToolRunResult(
            success=True,
            status="success",
            intent=request.intent,
            tool_name=request.tool_name,
            original_arguments=request.original_arguments,
            validated_arguments=request.validated_arguments,
            executed=True,
            execution_count=1,
            result=result,
            confirmation_id=confirmation_id,
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

    def _requires_authorization(self, tool_name: str) -> bool:
        check = getattr(self._executor, "requires_explicit_authorization", None)
        return bool(check(tool_name)) if callable(check) else False

    def _confirmation_required_result(
        self,
        request: ValidatedToolRequest,
    ) -> ToolRunResult:
        confirmation_id = uuid4().hex
        prompt = (
            f"Confirmar ejecucion de {request.tool_name} "
            f"con id {confirmation_id}? [s/N]: "
        )
        self._pending_confirmations[confirmation_id] = PendingToolConfirmation(
            confirmation_id=confirmation_id,
            request=request,
            prompt=prompt,
        )

        return ToolRunResult(
            success=False,
            status="confirmation_required",
            intent=request.intent,
            tool_name=request.tool_name,
            original_arguments=request.original_arguments,
            validated_arguments=request.validated_arguments,
            executed=False,
            execution_count=0,
            result=None,
            error_code="confirmation_required",
            error_message="explicit confirmation is required",
            confirmation_id=confirmation_id,
            metadata={"prompt": prompt},
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
        confirmation_id: str | None = None,
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
            confirmation_id=confirmation_id,
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


def _confirmation_decision(response: str) -> bool | None:
    normalized = response.strip().lower()

    if normalized in {"s", "si", "sí", "yes", "y"}:
        return True

    if normalized in {"n", "no"}:
        return False

    return None
