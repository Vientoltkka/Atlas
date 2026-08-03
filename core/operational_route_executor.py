"""Operational execution of already-classified Atlas route decisions."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field, fields, is_dataclass, replace
from datetime import datetime, timezone
from enum import Enum
import inspect
import re
from types import MappingProxyType
from typing import Any, Protocol

from agents.registry import AgentRegistry
from core.autonomous_execution import (
    AutonomousExecutionOptions,
    AutonomousExecutionOrchestrator,
    AutonomousExecutionOutcome,
    AutonomousExecutionResult,
)
from core.operational_request_router import (
    MemoryOperation,
    RequestRoute,
    RouteDecision,
    SystemCommand,
)
from core.operational_context import (
    OperationalContext,
    OperationalContextBuilder,
    OperationalContextError,
)
from core.execution_memory_recorder import ExecutionMemoryRecorder
from core.request_gateway import AtlasRequest, RequestSource
from memory.operational import (
    AmbiguousMemoryMatchError,
    InvalidMemoryEntryError,
    MemoryCategory,
    MemoryEntry,
    MemoryEntryNotFoundError,
    SensitiveMemoryRejectedError,
)
from tools.executor import ToolExecutor
from tools.registry import ToolRegistry
from tools.single_tool_runner import SingleToolRunner, ToolRunResult


MAX_TRACE_ENTRIES = 100
_SECRET_KEYS = ("token", "secret", "password", "api_key", "apikey", "credential")
def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class RouteExecutionStatus(str, Enum):
    COMPLETED = "completed"
    FAILED = "failed"
    WAITING_CONFIRMATION = "waiting_confirmation"
    CLARIFICATION_REQUIRED = "clarification_required"
    UNSUPPORTED = "unsupported"
    CANCELLED = "cancelled"
    INTERRUPTED = "interrupted"
    REJECTED = "rejected"


@dataclass(frozen=True, slots=True)
class RouteExecutionErrorInfo:
    request_id: str
    route: RequestRoute
    target: str | None
    code: str
    recoverable: bool
    summary: str
    safe_cause: str | None = None


class OperationalRouteExecutionError(RuntimeError):
    code = "OPERATIONAL_ROUTE_EXECUTION_ERROR"

    def __init__(
        self,
        *,
        request_id: str,
        route: RequestRoute,
        target: str | None,
        summary: str,
        recoverable: bool = False,
        safe_cause: str | None = None,
    ) -> None:
        super().__init__(summary)
        self.request_id = request_id
        self.route = route
        self.target = target
        self.summary = summary
        self.recoverable = recoverable
        self.safe_cause = safe_cause

    def to_info(self) -> RouteExecutionErrorInfo:
        return RouteExecutionErrorInfo(
            request_id=self.request_id,
            route=self.route,
            target=self.target,
            code=self.code,
            recoverable=self.recoverable,
            summary=self.summary,
            safe_cause=self.safe_cause,
        )


class InvalidRouteDecisionExecutionError(OperationalRouteExecutionError):
    code = "INVALID_ROUTE_DECISION_EXECUTION"


class RouteTargetUnavailableError(OperationalRouteExecutionError):
    code = "ROUTE_TARGET_UNAVAILABLE"


class RouteExecutionTimeoutError(OperationalRouteExecutionError):
    code = "ROUTE_EXECUTION_TIMEOUT"


class RouteExecutionRejectedError(OperationalRouteExecutionError):
    code = "ROUTE_EXECUTION_REJECTED"


class DuplicateRouteExecutionError(OperationalRouteExecutionError):
    code = "DUPLICATE_ROUTE_EXECUTION"


class MissingRouteArgumentsError(OperationalRouteExecutionError):
    code = "MISSING_ROUTE_ARGUMENTS"


class RouteHandlerNotConfiguredError(OperationalRouteExecutionError):
    code = "ROUTE_HANDLER_NOT_CONFIGURED"


@dataclass(frozen=True, slots=True)
class RouteExecutionTraceEntry:
    sequence: int
    timestamp: datetime
    request_id: str
    route: RequestRoute
    action: str
    target: str | None
    status_before: RouteExecutionStatus | None
    status_after: RouteExecutionStatus | None
    error_code: str | None = None
    summary: str = ""

    def __post_init__(self) -> None:
        _require_aware(self.timestamp, "trace timestamp")
        object.__setattr__(self, "summary", _sanitize_summary(self.summary))


@dataclass(frozen=True, slots=True)
class RouteExecutionResult:
    request_id: str
    route: RequestRoute
    status: RouteExecutionStatus
    output: Any
    error: RouteExecutionErrorInfo | None
    started_at: datetime
    finished_at: datetime
    duration: float
    target_tool_name: str | None = None
    target_agent_name: str | None = None
    session_id: str | None = None
    requires_confirmation: bool = False
    requires_clarification: bool = False
    clarification_question: str | None = None
    side_effects_performed: bool = False
    execution_reference: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    trace: tuple[RouteExecutionTraceEntry, ...] = ()

    def __post_init__(self) -> None:
        _require_aware(self.started_at, "started_at")
        _require_aware(self.finished_at, "finished_at")
        if self.finished_at < self.started_at:
            raise ValueError("finished_at cannot precede started_at.")
        if self.duration < 0:
            raise ValueError("duration cannot be negative.")
        object.__setattr__(self, "output", _freeze_safe(self.output))
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))
        object.__setattr__(self, "trace", tuple(self.trace)[-MAX_TRACE_ENTRIES:])


@dataclass(frozen=True, slots=True)
class SingleToolInvocation:
    tool_name: str
    arguments: Mapping[str, Any]
    source_request_id: str
    confirmation_required: bool
    risk_level: str
    prepared_at: datetime

    def __post_init__(self) -> None:
        _require_aware(self.prepared_at, "prepared_at")
        object.__setattr__(self, "arguments", _freeze_mapping(self.arguments))


@dataclass(frozen=True, slots=True)
class AgentDelegationRequest:
    request_id: str
    agent_name: str
    objective: str
    context: Mapping[str, Any]
    attachments: tuple[Mapping[str, Any], ...]
    locale: str
    safety_context: Mapping[str, Any]
    operational_context: OperationalContext | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "context", _freeze_mapping(self.context))
        object.__setattr__(
            self,
            "attachments",
            tuple(_freeze_mapping(item) for item in self.attachments),
        )
        object.__setattr__(self, "safety_context", _freeze_mapping(self.safety_context))


@dataclass(frozen=True, slots=True)
class SystemCommandResult:
    command: SystemCommand
    success: bool
    signal: str | None = None
    output: Any = None
    session_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "output", _freeze_safe(self.output))


@dataclass(frozen=True, slots=True)
class RouteExecutionEvent:
    event_type: str
    request_id: str
    route: RequestRoute
    handler: str | None
    target: str | None
    status: RouteExecutionStatus | None
    duration: float | None
    error_code: str | None
    timestamp: datetime

    def __post_init__(self) -> None:
        _require_aware(self.timestamp, "event timestamp")


class RouteHandler(Protocol):
    def execute(
        self,
        request: AtlasRequest,
        decision: RouteDecision,
    ) -> RouteExecutionResult:
        ...


class RequestExecutionLedger:
    """Minimal in-memory protection against repeated route side effects."""

    def __init__(self) -> None:
        self._results: dict[str, RouteExecutionResult] = {}

    def get(self, request_id: str) -> RouteExecutionResult | None:
        return self._results.get(request_id)

    def record(self, result: RouteExecutionResult) -> None:
        self._results[result.request_id] = result


class _BaseRouteHandler:
    def __init__(self, *, clock: Callable[[], datetime] | None = None) -> None:
        self._clock = clock or _utc_now

    def _result(
        self,
        request: AtlasRequest,
        decision: RouteDecision,
        *,
        started_at: datetime,
        status: RouteExecutionStatus,
        output: Any = None,
        error: OperationalRouteExecutionError | RouteExecutionErrorInfo | None = None,
        side_effects_performed: bool = False,
        requires_confirmation: bool = False,
        requires_clarification: bool = False,
        clarification_question: str | None = None,
        session_id: str | None = None,
        execution_reference: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        action: str = "route_executed",
        target: str | None = None,
    ) -> RouteExecutionResult:
        finished_at = self._clock()
        error_info = error.to_info() if isinstance(error, OperationalRouteExecutionError) else error
        trace = (
            RouteExecutionTraceEntry(
                sequence=1,
                timestamp=started_at,
                request_id=request.request_id,
                route=decision.route,
                action="route_execution_started",
                target=target,
                status_before=None,
                status_after=None,
                summary="route execution started",
            ),
            RouteExecutionTraceEntry(
                sequence=2,
                timestamp=finished_at,
                request_id=request.request_id,
                route=decision.route,
                action=action,
                target=target,
                status_before=None,
                status_after=status,
                error_code=error_info.code if error_info else None,
                summary=(
                    error_info.summary
                    if error_info is not None
                    else f"route finished with status {status.value}"
                ),
            ),
        )
        return RouteExecutionResult(
            request_id=request.request_id,
            route=decision.route,
            status=status,
            output=output,
            error=error_info,
            started_at=started_at,
            finished_at=finished_at,
            duration=max(0.0, (finished_at - started_at).total_seconds()),
            target_tool_name=decision.target_tool_name,
            target_agent_name=decision.target_agent_name,
            session_id=session_id,
            requires_confirmation=requires_confirmation,
            requires_clarification=requires_clarification,
            clarification_question=clarification_question,
            side_effects_performed=side_effects_performed,
            execution_reference=execution_reference,
            metadata=metadata or {},
            trace=trace,
        )


class DirectResponseRouteHandler(_BaseRouteHandler):
    def __init__(
        self,
        responder: Callable[[AtlasRequest], Any] | None,
        streaming_responder: Callable[[AtlasRequest], Any] | None = None,
        *,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        super().__init__(clock=clock)
        self._responder = responder
        self._streaming_responder = streaming_responder

    def execute(self, request: AtlasRequest, decision: RouteDecision) -> RouteExecutionResult:
        return self.execute_with_context(request, decision, None)

    def execute_with_context(
        self,
        request: AtlasRequest,
        decision: RouteDecision,
        context: OperationalContext | None,
    ) -> RouteExecutionResult:
        started = self._clock()
        if self._responder is None:
            error = RouteHandlerNotConfiguredError(
                request_id=request.request_id,
                route=decision.route,
                target="direct_response",
                summary="Direct response service is not configured.",
                recoverable=True,
            )
            return self._result(
                request,
                decision,
                started_at=started,
                status=RouteExecutionStatus.FAILED,
                error=error,
                target="direct_response",
                action="route_execution_failed",
            )
        try:
            output = _invoke_contextual_callable(self._responder, request, context)
        except Exception as cause:
            return self._failure(request, decision, started, "direct_response", cause)
        return self._result(
            request,
            decision,
            started_at=started,
            status=RouteExecutionStatus.COMPLETED,
            output=output,
            target="direct_response",
            action="route_execution_completed",
        )

    def execute_streaming_with_context(
        self,
        request: AtlasRequest,
        decision: RouteDecision,
        context: OperationalContext | None,
        fragment_sink: Callable[[str], bool | None],
    ) -> RouteExecutionResult:
        """Stream one direct response while preserving the normal route result."""
        if self._streaming_responder is None:
            result = self.execute_with_context(request, decision, context)
            if result.status is RouteExecutionStatus.COMPLETED:
                rendered = _present_output(result.output)
                if rendered:
                    fragment_sink(rendered)
            return result

        started = self._clock()
        stream = None
        try:
            stream = _invoke_contextual_callable(
                self._streaming_responder,
                request,
                context,
            )
            fragments: list[str] = []
            for fragment in stream:
                if not isinstance(fragment, str):
                    raise TypeError("Direct response stream yielded a non-text fragment.")
                if not fragment:
                    continue
                fragments.append(fragment)
                if fragment_sink(fragment) is False:
                    break
        except Exception as cause:
            return self._failure(request, decision, started, "direct_response", cause)
        finally:
            close = getattr(stream, "close", None)
            if callable(close):
                close()

        return self._result(
            request,
            decision,
            started_at=started,
            status=RouteExecutionStatus.COMPLETED,
            output="".join(fragments),
            target="direct_response",
            action="route_execution_completed",
        )

    def _failure(
        self,
        request: AtlasRequest,
        decision: RouteDecision,
        started: datetime,
        target: str,
        cause: Exception,
    ) -> RouteExecutionResult:
        error = OperationalRouteExecutionError(
            request_id=request.request_id,
            route=decision.route,
            target=target,
            summary="Direct response failed.",
            recoverable=True,
            safe_cause=type(cause).__name__,
        )
        return self._result(
            request,
            decision,
            started_at=started,
            status=RouteExecutionStatus.FAILED,
            error=error,
            target=target,
            action="route_execution_failed",
        )


class MemoryRouteHandler(_BaseRouteHandler):
    def __init__(
        self,
        memory: object | None,
        *,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        super().__init__(clock=clock)
        self._memory = memory

    def execute(self, request: AtlasRequest, decision: RouteDecision) -> RouteExecutionResult:
        return self.execute_with_context(request, decision, None)

    def execute_with_context(
        self,
        request: AtlasRequest,
        decision: RouteDecision,
        context: OperationalContext | None,
    ) -> RouteExecutionResult:
        del context
        started = self._clock()
        operation = decision.memory_operation
        if operation is None:
            return _clarification_result(
                self,
                request,
                decision,
                started,
                "Que operacion de memoria quieres realizar?",
                ("memory_operation",),
            )
        if self._memory is None:
            return self._unsupported(request, decision, started, operation)
        if operation is MemoryOperation.STORE:
            store_entry = getattr(self._memory, "store_entry", None)
            content = _memory_content(request)
            if not callable(store_entry):
                add_user = getattr(self._memory, "add_user", None)
                if not callable(add_user):
                    return self._unsupported(request, decision, started, operation)
                if not content:
                    return _clarification_result(
                        self,
                        request,
                        decision,
                        started,
                        "Que informacion quieres que recuerde?",
                        ("memory_content",),
                    )
                add_user(content)
                return self._result(
                    request,
                    decision,
                    started_at=started,
                    status=RouteExecutionStatus.COMPLETED,
                    output={"operation": operation.value, "stored": True},
                    side_effects_performed=True,
                    target=operation.value,
                    action="route_execution_completed",
                )
            if not content:
                return _clarification_result(
                    self,
                    request,
                    decision,
                    started,
                    "Que informacion quieres que recuerde?",
                    ("memory_content",),
                )
            try:
                entry = store_entry(
                    content,
                    category=_memory_category(request),
                    source_request_id=request.request_id,
                    user_id=request.user_id,
                    conversation_id=request.conversation_id,
                    importance=_memory_importance(request),
                    tags=_memory_tags(request),
                    sensitive=bool(request.metadata.get("sensitive", False)),
                    expires_at=_memory_expires_at(request),
                    metadata=_memory_safe_metadata(request),
                )
            except (InvalidMemoryEntryError, SensitiveMemoryRejectedError) as cause:
                return self._memory_failure(
                    request,
                    decision,
                    started,
                    operation,
                    cause,
                    recoverable=True,
                )
            return self._result(
                request,
                decision,
                started_at=started,
                status=RouteExecutionStatus.COMPLETED,
                output={
                    "operation": operation.value,
                    "stored": True,
                    "memory_id": entry.memory_id,
                    "entry": _memory_entry_view(entry),
                },
                side_effects_performed=True,
                target=operation.value,
                action="route_execution_completed",
            )
        if operation is MemoryOperation.RETRIEVE:
            retrieve_entries = getattr(self._memory, "retrieve_entries", None)
            if not callable(retrieve_entries):
                history = getattr(self._memory, "history", None)
                if not callable(history):
                    return self._unsupported(request, decision, started, operation)
                items = history()
                return self._result(
                    request,
                    decision,
                    started_at=started,
                    status=RouteExecutionStatus.COMPLETED,
                    output={
                        "operation": operation.value,
                        "items": items,
                        "count": len(items),
                    },
                    target=operation.value,
                    action="route_execution_completed",
                )
            entries = retrieve_entries(
                _memory_query(request),
                user_id=request.user_id,
                conversation_id=request.conversation_id,
                tags=_memory_tags(request),
                categories=_memory_categories_filter(request),
                include_sensitive=False,
                limit=_memory_limit(request),
            )
            return self._result(
                request,
                decision,
                started_at=started,
                status=RouteExecutionStatus.COMPLETED,
                output={
                    "operation": operation.value,
                    "items": tuple(_memory_entry_view(entry) for entry in entries),
                    "count": len(entries),
                },
                target=operation.value,
                action="route_execution_completed",
            )
        if operation is MemoryOperation.LIST:
            list_entries = getattr(self._memory, "list_entries", None)
            if not callable(list_entries):
                history = getattr(self._memory, "history", None)
                if not callable(history):
                    return self._unsupported(request, decision, started, operation)
                items = history()
                return self._result(
                    request,
                    decision,
                    started_at=started,
                    status=RouteExecutionStatus.COMPLETED,
                    output={
                        "operation": operation.value,
                        "items": items,
                        "count": len(items),
                    },
                    target=operation.value,
                    action="route_execution_completed",
                )
            entries = list_entries(
                active_only=True,
                include_expired=False,
                include_sensitive=False,
                user_id=request.user_id,
                conversation_id=request.conversation_id,
                tags=_memory_tags(request),
                categories=_memory_categories_filter(request),
                limit=_memory_limit(request),
            )
            return self._result(
                request,
                decision,
                started_at=started,
                status=RouteExecutionStatus.COMPLETED,
                output={
                    "operation": operation.value,
                    "items": tuple(_memory_entry_view(entry) for entry in entries),
                    "count": len(entries),
                },
                target=operation.value,
                action="route_execution_completed",
            )
        if operation is MemoryOperation.FORGET:
            return self._forget(request, decision, started)
        if operation is MemoryOperation.UPDATE:
            return self._update(request, decision, started)
        return self._unsupported(request, decision, started, operation)

    def _forget(
        self,
        request: AtlasRequest,
        decision: RouteDecision,
        started: datetime,
    ) -> RouteExecutionResult:
        forget_entry = getattr(self._memory, "forget_entry", None)
        if not callable(forget_entry):
            return self._unsupported(request, decision, started, MemoryOperation.FORGET)
        memory_id, ambiguity = _resolve_memory_id(self._memory, request)
        if ambiguity:
            return _clarification_result(
                self,
                request,
                decision,
                started,
                "Hay varias entradas coincidentes. Indica el memory_id exacto.",
                ("memory_id",),
                target=MemoryOperation.FORGET.value,
            )
        if not memory_id:
            return _clarification_result(
                self,
                request,
                decision,
                started,
                "Que entrada quieres olvidar? Indica el memory_id.",
                ("memory_id",),
                target=MemoryOperation.FORGET.value,
            )
        try:
            entry = forget_entry(memory_id, source_request_id=request.request_id)
        except MemoryEntryNotFoundError as cause:
            return self._memory_failure(
                request,
                decision,
                started,
                MemoryOperation.FORGET,
                cause,
                recoverable=True,
            )
        return self._result(
            request,
            decision,
            started_at=started,
            status=RouteExecutionStatus.COMPLETED,
            output={
                "operation": MemoryOperation.FORGET.value,
                "memory_id": entry.memory_id,
                "forgotten": not entry.active,
            },
            side_effects_performed=True,
            target=MemoryOperation.FORGET.value,
            action="route_execution_completed",
        )

    def _update(
        self,
        request: AtlasRequest,
        decision: RouteDecision,
        started: datetime,
    ) -> RouteExecutionResult:
        update_entry = getattr(self._memory, "update_entry", None)
        if not callable(update_entry):
            return self._unsupported(request, decision, started, MemoryOperation.UPDATE)
        memory_id, ambiguity = _resolve_memory_id(self._memory, request)
        if ambiguity or not memory_id:
            return _clarification_result(
                self,
                request,
                decision,
                started,
                (
                    "Hay varias entradas coincidentes. Indica el memory_id exacto."
                    if ambiguity
                    else "Que entrada quieres actualizar? Indica el memory_id."
                ),
                ("memory_id",),
                target=MemoryOperation.UPDATE.value,
            )
        content = _memory_update_content(request, memory_id)
        if not content:
            return _clarification_result(
                self,
                request,
                decision,
                started,
                "Cual es el nuevo contenido de la entrada?",
                ("memory_content",),
                target=MemoryOperation.UPDATE.value,
            )
        try:
            entry = update_entry(
                memory_id,
                content=content,
                category=_memory_optional_category(request),
                importance=_memory_optional_importance(request),
                tags=_memory_optional_tags(request),
                sensitive=_memory_optional_sensitive(request),
                source_request_id=request.request_id,
            )
        except (
            MemoryEntryNotFoundError,
            InvalidMemoryEntryError,
            SensitiveMemoryRejectedError,
        ) as cause:
            return self._memory_failure(
                request,
                decision,
                started,
                MemoryOperation.UPDATE,
                cause,
                recoverable=True,
            )
        return self._result(
            request,
            decision,
            started_at=started,
            status=RouteExecutionStatus.COMPLETED,
            output={
                "operation": MemoryOperation.UPDATE.value,
                "memory_id": entry.memory_id,
                "updated": True,
                "entry": _memory_entry_view(entry),
            },
            side_effects_performed=True,
            target=MemoryOperation.UPDATE.value,
            action="route_execution_completed",
        )

    def _memory_failure(
        self,
        request: AtlasRequest,
        decision: RouteDecision,
        started: datetime,
        operation: MemoryOperation,
        cause: Exception,
        *,
        recoverable: bool,
    ) -> RouteExecutionResult:
        error = OperationalRouteExecutionError(
            request_id=request.request_id,
            route=decision.route,
            target=operation.value,
            summary=_safe_memory_error_summary(cause, operation),
            recoverable=recoverable,
            safe_cause=getattr(cause, "code", type(cause).__name__),
        )
        return self._result(
            request,
            decision,
            started_at=started,
            status=(
                RouteExecutionStatus.REJECTED
                if isinstance(cause, SensitiveMemoryRejectedError)
                else RouteExecutionStatus.FAILED
            ),
            error=error,
            target=operation.value,
            action="route_execution_failed",
        )

    def _unsupported(
        self,
        request: AtlasRequest,
        decision: RouteDecision,
        started: datetime,
        operation: MemoryOperation,
    ) -> RouteExecutionResult:
        return self._result(
            request,
            decision,
            started_at=started,
            status=RouteExecutionStatus.UNSUPPORTED,
            output={"reason": f"Memory operation '{operation.value}' is not available."},
            target=operation.value,
            action="route_execution_unsupported",
        )


class SingleToolRouteHandler(_BaseRouteHandler):
    def __init__(
        self,
        tool_registry: ToolRegistry | None,
        tool_executor: ToolExecutor | None,
        single_tool_runner: SingleToolRunner | None = None,
        *,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        super().__init__(clock=clock)
        self._tool_registry = tool_registry
        self._tool_executor = tool_executor
        self._single_tool_runner = single_tool_runner

    def execute(self, request: AtlasRequest, decision: RouteDecision) -> RouteExecutionResult:
        started = self._clock()
        tool_name = decision.target_tool_name
        if not tool_name or self._tool_registry is None or not self._tool_registry.exists(tool_name):
            error = RouteTargetUnavailableError(
                request_id=request.request_id,
                route=decision.route,
                target=tool_name,
                summary="The selected tool is not registered.",
                recoverable=True,
            )
            return self._result(
                request,
                decision,
                started_at=started,
                status=RouteExecutionStatus.FAILED,
                error=error,
                target=tool_name,
                action="route_execution_failed",
            )
        descriptor = self._tool_registry.descriptor(tool_name)
        arguments, missing = _prepare_tool_arguments(
            request,
            descriptor.arguments_schema,
            tool_name,
        )
        if missing:
            return _clarification_result(
                self,
                request,
                decision,
                started,
                "Faltan argumentos para ejecutar la herramienta: " + ", ".join(missing) + ".",
                missing,
                target=tool_name,
            )
        invocation = SingleToolInvocation(
            tool_name=tool_name,
            arguments=arguments,
            source_request_id=request.request_id,
            confirmation_required=bool(
                descriptor.requires_confirmation or decision.requires_confirmation
            ),
            risk_level="high" if descriptor.dangerous else "low",
            prepared_at=started,
        )
        if not request.safety_context.allow_side_effects and _tool_has_side_effects(descriptor):
            error = RouteExecutionRejectedError(
                request_id=request.request_id,
                route=decision.route,
                target=tool_name,
                summary="Tool side effects are disabled by RequestSafetyContext.",
                recoverable=True,
            )
            return self._result(
                request,
                decision,
                started_at=started,
                status=RouteExecutionStatus.REJECTED,
                error=error,
                output={"tool_name": tool_name, "risk_level": invocation.risk_level},
                target=tool_name,
                action="route_execution_failed",
            )
        runner_result = (
            self._single_tool_runner.run_registered_tool(
                invocation.tool_name,
                invocation.arguments,
            )
            if self._single_tool_runner is not None
            else None
        )
        if runner_result is not None:
            return _tool_run_result(
                self,
                request,
                decision,
                started,
                invocation,
                runner_result,
            )
        if invocation.confirmation_required:
            error = RouteHandlerNotConfiguredError(
                request_id=request.request_id,
                route=decision.route,
                target=tool_name,
                summary="The existing confirmation runner cannot execute this registered tool.",
                recoverable=True,
            )
            return self._result(
                request,
                decision,
                started_at=started,
                status=RouteExecutionStatus.FAILED,
                error=error,
                target=tool_name,
                action="route_execution_failed",
            )
        if self._tool_executor is None:
            error = RouteHandlerNotConfiguredError(
                request_id=request.request_id,
                route=decision.route,
                target=tool_name,
                summary="ToolExecutor is not configured.",
                recoverable=True,
            )
            return self._result(
                request,
                decision,
                started_at=started,
                status=RouteExecutionStatus.FAILED,
                error=error,
                target=tool_name,
                action="route_execution_failed",
            )
        try:
            output = self._tool_executor.execute(tool_name, arguments=dict(arguments))
        except Exception as cause:
            error = OperationalRouteExecutionError(
                request_id=request.request_id,
                route=decision.route,
                target=tool_name,
                summary="Tool execution failed.",
                recoverable=True,
                safe_cause=type(cause).__name__,
            )
            return self._result(
                request,
                decision,
                started_at=started,
                status=RouteExecutionStatus.FAILED,
                error=error,
                target=tool_name,
                action="route_execution_failed",
            )
        return self._result(
            request,
            decision,
            started_at=started,
            status=RouteExecutionStatus.COMPLETED,
            output=output,
            side_effects_performed=_tool_has_side_effects(descriptor),
            target=tool_name,
            metadata={"arguments": arguments, "risk_level": invocation.risk_level},
            action="route_execution_completed",
        )

class AgentDelegationRouteHandler(_BaseRouteHandler):
    def __init__(
        self,
        agent_registry: AgentRegistry | None,
        *,
        model_selector: Callable[[str], str] | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        super().__init__(clock=clock)
        self._agent_registry = agent_registry
        self._model_selector = model_selector

    def execute(self, request: AtlasRequest, decision: RouteDecision) -> RouteExecutionResult:
        return self.execute_with_context(request, decision, None)

    def execute_with_context(
        self,
        request: AtlasRequest,
        decision: RouteDecision,
        context: OperationalContext | None,
    ) -> RouteExecutionResult:
        started = self._clock()
        agent_name = decision.target_agent_name
        agent = self._agent_registry.get(agent_name) if self._agent_registry and agent_name else None
        if agent is None or agent_name == "chat":
            error = RouteTargetUnavailableError(
                request_id=request.request_id,
                route=decision.route,
                target=agent_name,
                summary="The selected specialized agent is not registered.",
                recoverable=True,
            )
            return self._result(
                request,
                decision,
                started_at=started,
                status=RouteExecutionStatus.FAILED,
                error=error,
                target=agent_name,
                action="route_execution_failed",
            )
        delegation = AgentDelegationRequest(
            request_id=request.request_id,
            agent_name=agent_name,
            objective=request.content,
            context={
                "conversation_id": request.conversation_id,
                "correlation_id": request.correlation_id,
            },
            attachments=tuple(asdict(item) for item in request.attachments),
            locale=request.locale,
            safety_context=asdict(request.safety_context),
            operational_context=context,
        )
        messages = _context_messages(context)
        messages.append({"role": "user", "content": delegation.objective})
        try:
            model = self._model_selector(agent_name) if self._model_selector else ""
            output = agent.run(model=model, messages=messages)
        except Exception as cause:
            error = OperationalRouteExecutionError(
                request_id=request.request_id,
                route=decision.route,
                target=agent_name,
                summary="Agent delegation failed.",
                recoverable=True,
                safe_cause=type(cause).__name__,
            )
            return self._result(
                request,
                decision,
                started_at=started,
                status=RouteExecutionStatus.FAILED,
                error=error,
                target=agent_name,
                action="route_execution_failed",
            )
        return self._result(
            request,
            decision,
            started_at=started,
            status=RouteExecutionStatus.COMPLETED,
            output=output,
            target=agent_name,
            metadata={
                "delegation": {
                    "request_id": delegation.request_id,
                    "agent_name": delegation.agent_name,
                    "locale": delegation.locale,
                },
                "operational_context": (
                    context.safe_summary() if context is not None else {}
                ),
            },
            action="route_execution_completed",
        )


class AutonomousExecutionRouteHandler(_BaseRouteHandler):
    def __init__(
        self,
        orchestrator: AutonomousExecutionOrchestrator | None,
        *,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        super().__init__(clock=clock)
        self._orchestrator = orchestrator

    def execute(self, request: AtlasRequest, decision: RouteDecision) -> RouteExecutionResult:
        return self.execute_with_context(request, decision, None)

    def execute_with_context(
        self,
        request: AtlasRequest,
        decision: RouteDecision,
        context: OperationalContext | None,
    ) -> RouteExecutionResult:
        started = self._clock()
        if self._orchestrator is None:
            error = RouteHandlerNotConfiguredError(
                request_id=request.request_id,
                route=decision.route,
                target="autonomous_execution",
                summary="AutonomousExecutionOrchestrator is not configured.",
                recoverable=True,
            )
            return self._result(
                request,
                decision,
                started_at=started,
                status=RouteExecutionStatus.FAILED,
                error=error,
                target="autonomous_execution",
                action="route_execution_failed",
            )
        operational_context = context
        request_context = request.execution_context
        options = AutonomousExecutionOptions(
            max_wall_time_seconds=(
                request_context.requested_timeout
                if request_context and request_context.requested_timeout is not None
                else 300.0
            ),
            max_total_cost=(
                request_context.requested_budget
                if request_context and request_context.requested_budget is not None
                else None
            ),
            dry_run=bool(request_context and request_context.dry_run),
        )
        try:
            autonomous = self._orchestrator.execute_objective(
                request.content,
                planning_context={
                    "request_id": request.request_id,
                    "locale": request.locale,
                    "source": request.source.value,
                    "relevant_context": (
                        operational_context.prompt_context()
                        if operational_context is not None
                        else ""
                    ),
                    "selected_memory_ids": (
                        operational_context.selected_memory_ids
                        if operational_context is not None
                        else ()
                    ),
                    "execution_context": (
                        operational_context.execution_context
                        if operational_context is not None
                        else {}
                    ),
                },
                execution_options=options,
            )
        except Exception as cause:
            error = OperationalRouteExecutionError(
                request_id=request.request_id,
                route=decision.route,
                target="autonomous_execution",
                summary="Autonomous execution failed.",
                recoverable=True,
                safe_cause=type(cause).__name__,
            )
            return self._result(
                request,
                decision,
                started_at=started,
                status=RouteExecutionStatus.FAILED,
                error=error,
                target="autonomous_execution",
                action="route_execution_failed",
            )
        return _autonomous_result(self, request, decision, started, autonomous)


class ResumeExecutionRouteHandler(_BaseRouteHandler):
    def __init__(
        self,
        orchestrator: AutonomousExecutionOrchestrator | None,
        single_tool_runner: SingleToolRunner | None = None,
        *,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        super().__init__(clock=clock)
        self._orchestrator = orchestrator
        self._single_tool_runner = single_tool_runner

    def execute(self, request: AtlasRequest, decision: RouteDecision) -> RouteExecutionResult:
        started = self._clock()
        context = request.execution_context
        target = decision.target_session_id or (
            context.session_id if context is not None else None
        )
        if not target:
            return _clarification_result(
                self,
                request,
                decision,
                started,
                "Que ejecucion quieres continuar? Indica el session_id.",
                ("session_id",),
            )
        confirmation = context.confirmation_response if context else None
        if self._single_tool_runner is not None:
            pending = self._single_tool_runner.pending_confirmation(target)
            if pending is not None:
                if confirmation is None:
                    return _clarification_result(
                        self,
                        request,
                        decision,
                        started,
                        "Apruebas o rechazas la accion pendiente?",
                        ("confirmation_response",),
                        target=target,
                    )
                run_result = self._single_tool_runner.confirm(
                    target,
                    "si" if confirmation else "no",
                )
                invocation = SingleToolInvocation(
                    tool_name=pending.request.tool_name,
                    arguments=pending.request.validated_arguments,
                    source_request_id=request.request_id,
                    confirmation_required=True,
                    risk_level="high",
                    prepared_at=started,
                )
                return _tool_run_result(
                    self,
                    request,
                    decision,
                    started,
                    invocation,
                    run_result,
                    session_id=target,
                )
        if self._orchestrator is None:
            error = RouteHandlerNotConfiguredError(
                request_id=request.request_id,
                route=decision.route,
                target=target,
                summary="AutonomousExecutionOrchestrator is not configured.",
                recoverable=True,
            )
            return self._result(
                request,
                decision,
                started_at=started,
                status=RouteExecutionStatus.FAILED,
                error=error,
                session_id=target,
                target=target,
                action="route_execution_failed",
            )
        try:
            if _is_cancel_request(request):
                autonomous = self._orchestrator.cancel_execution(target)
            else:
                autonomous = self._orchestrator.resume_execution(
                    target,
                    confirmation=confirmation,
                    recovery_authorization=(
                        context.recovery_authorization if context else None
                    ),
                )
        except Exception as cause:
            error = OperationalRouteExecutionError(
                request_id=request.request_id,
                route=decision.route,
                target=target,
                summary="Execution could not be resumed.",
                recoverable=True,
                safe_cause=type(cause).__name__,
            )
            return self._result(
                request,
                decision,
                started_at=started,
                status=RouteExecutionStatus.FAILED,
                error=error,
                session_id=target,
                target=target,
                action="route_execution_failed",
            )
        return _autonomous_result(self, request, decision, started, autonomous)


class SystemCommandRouteHandler(_BaseRouteHandler):
    def __init__(
        self,
        *,
        supervisor: object | None = None,
        autonomous_orchestrator: AutonomousExecutionOrchestrator | None = None,
        diagnostics: Callable[[], Any] | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        super().__init__(clock=clock)
        self._supervisor = supervisor
        self._autonomous = autonomous_orchestrator
        self._diagnostics = diagnostics

    def execute(self, request: AtlasRequest, decision: RouteDecision) -> RouteExecutionResult:
        started = self._clock()
        command = decision.system_command
        if command is None:
            return _clarification_result(
                self,
                request,
                decision,
                started,
                "Que comando interno quieres ejecutar?",
                ("system_command",),
            )
        if command is SystemCommand.CANCEL_EXECUTION:
            session_id = decision.target_session_id or (
                request.execution_context.session_id
                if request.execution_context is not None
                else None
            )
            if not session_id:
                return _clarification_result(
                    self,
                    request,
                    decision,
                    started,
                    "Que ejecucion quieres cancelar? Indica el session_id.",
                    ("session_id",),
                    target=command.value,
                )
        try:
            command_result = self._execute_command(request, decision, command)
        except OperationalRouteExecutionError as error:
            return self._result(
                request,
                decision,
                started_at=started,
                status=RouteExecutionStatus.FAILED,
                error=error,
                target=command.value,
                action="route_execution_failed",
            )
        return self._result(
            request,
            decision,
            started_at=started,
            status=(
                RouteExecutionStatus.COMPLETED
                if command_result.success
                else RouteExecutionStatus.FAILED
            ),
            output=command_result,
            side_effects_performed=command in {
                SystemCommand.CANCEL_EXECUTION,
                SystemCommand.VOICE_MODE,
                SystemCommand.STOP_LISTENING,
            },
            session_id=command_result.session_id,
            target=command.value,
            action="route_execution_completed",
        )

    def _execute_command(
        self,
        request: AtlasRequest,
        decision: RouteDecision,
        command: SystemCommand,
    ) -> SystemCommandResult:
        if command is SystemCommand.EXIT:
            return SystemCommandResult(command, True, signal="exit_requested")
        if command is SystemCommand.HELP:
            return SystemCommandResult(
                command,
                True,
                output={"commands": tuple(item.value for item in SystemCommand)},
            )
        if command is SystemCommand.STATUS:
            overview = _call_optional(self._supervisor, "get_overview")
            return SystemCommandResult(command, True, output=overview or {"status": "idle"})
        if command is SystemCommand.LIST_EXECUTIONS:
            sessions = _call_optional(self._supervisor, "list_sessions")
            if sessions is None:
                raise self._unavailable(request, decision, command)
            return SystemCommandResult(command, True, output=sessions)
        if command is SystemCommand.CANCEL_EXECUTION:
            session_id = decision.target_session_id or (
                request.execution_context.session_id
                if request.execution_context is not None
                else None
            )
            if not session_id or self._autonomous is None:
                raise self._unavailable(request, decision, command)
            result = self._autonomous.cancel_execution(session_id)
            return SystemCommandResult(
                command,
                True,
                output=result.summary,
                session_id=session_id,
            )
        if command is SystemCommand.DIAGNOSTICS:
            return SystemCommandResult(
                command,
                True,
                output=self._diagnostics() if self._diagnostics else {"status": "available"},
            )
        if command is SystemCommand.VOICE_MODE:
            return SystemCommandResult(command, True, signal="voice_mode_requested")
        if command is SystemCommand.STOP_LISTENING:
            return SystemCommandResult(command, True, signal="stop_listening_requested")
        raise self._unavailable(request, decision, command)

    @staticmethod
    def _unavailable(
        request: AtlasRequest,
        decision: RouteDecision,
        command: SystemCommand,
    ) -> RouteTargetUnavailableError:
        return RouteTargetUnavailableError(
            request_id=request.request_id,
            route=decision.route,
            target=command.value,
            summary="The selected system command is not available in the current runtime.",
            recoverable=True,
        )


class ClarificationRouteHandler(_BaseRouteHandler):
    def execute(self, request: AtlasRequest, decision: RouteDecision) -> RouteExecutionResult:
        started = self._clock()
        question = decision.clarification_question or "Que informacion falta para continuar?"
        return _clarification_result(
            self,
            request,
            decision,
            started,
            question,
            _decision_missing_information(decision),
        )


class UnsupportedRouteHandler(_BaseRouteHandler):
    def execute(self, request: AtlasRequest, decision: RouteDecision) -> RouteExecutionResult:
        started = self._clock()
        return self._result(
            request,
            decision,
            started_at=started,
            status=RouteExecutionStatus.UNSUPPORTED,
            output={"reason": decision.reason},
            action="route_execution_unsupported",
        )


class OperationalRouteExecutor:
    """Validate and execute a RouteDecision without classifying again."""

    def __init__(
        self,
        handlers: Mapping[RequestRoute, RouteHandler],
        *,
        enabled_routes: frozenset[RequestRoute] | None = None,
        ledger: RequestExecutionLedger | None = None,
        context_builder: OperationalContextBuilder | None = None,
        execution_memory_recorder: ExecutionMemoryRecorder | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._clock = clock or _utc_now
        self._enabled_routes = enabled_routes or frozenset(RequestRoute)
        self._ledger = ledger or RequestExecutionLedger()
        self._context_builder = context_builder
        self._execution_memory_recorder = execution_memory_recorder
        self._handlers: dict[RequestRoute, RouteHandler] = {}
        self._events: list[RouteExecutionEvent] = []
        for route, handler in handlers.items():
            self.register_handler(route, handler)

    @property
    def events(self) -> tuple[RouteExecutionEvent, ...]:
        return tuple(self._events)

    @property
    def handlers(self) -> Mapping[RequestRoute, RouteHandler]:
        return MappingProxyType(self._handlers)

    def register_handler(self, route: RequestRoute, handler: RouteHandler) -> None:
        normalized = route if isinstance(route, RequestRoute) else RequestRoute(route)
        if normalized in self._handlers:
            raise ValueError(f"Handler already registered for route '{normalized.value}'.")
        if not callable(getattr(handler, "execute", None)):
            raise TypeError("handler must implement execute(request, decision).")
        self._handlers[normalized] = handler

    def execute(
        self,
        request: AtlasRequest,
        decision: RouteDecision,
        *,
        output_fragment_sink: Callable[[str], bool | None] | None = None,
    ) -> RouteExecutionResult:
        started = self._clock()
        handler_name: str | None = None
        target = _decision_target(decision)
        self._record(
            "route_execution_started",
            request,
            decision,
            handler=None,
            target=target,
            status=None,
            duration=None,
            error_code=None,
        )
        try:
            self._validate(request, decision)
            previous = self._ledger.get(request.request_id)
            if previous is not None and _requires_idempotency(decision):
                self._record(
                    "duplicate_request_detected",
                    request,
                    decision,
                    handler=None,
                    target=target,
                    status=previous.status,
                    duration=0.0,
                    error_code=DuplicateRouteExecutionError.code,
                )
                return previous
            context = (
                self._context_builder.build(request, decision)
                if self._context_builder is not None
                else _empty_operational_context(request, self._clock())
            )
            handler = self._handlers.get(decision.route)
            if handler is None:
                raise RouteHandlerNotConfiguredError(
                    request_id=request.request_id,
                    route=decision.route,
                    target=target,
                    summary="No handler is configured for the selected route.",
                    recoverable=True,
                )
            handler_name = type(handler).__name__
            self._record(
                "route_handler_selected",
                request,
                decision,
                handler=handler_name,
                target=target,
                status=None,
                duration=None,
                error_code=None,
            )
            timeout = (
                request.execution_context.requested_timeout
                if request.execution_context is not None
                else None
            )
            if timeout == 0:
                raise RouteExecutionTimeoutError(
                    request_id=request.request_id,
                    route=decision.route,
                    target=target,
                    summary="Route execution timed out before it started.",
                    recoverable=True,
                )
            execute_with_context = getattr(handler, "execute_with_context", None)
            execute_streaming_with_context = getattr(
                handler,
                "execute_streaming_with_context",
                None,
            )
            if output_fragment_sink is not None and callable(execute_streaming_with_context):
                result = execute_streaming_with_context(
                    request,
                    decision,
                    context,
                    output_fragment_sink,
                )
            else:
                result = (
                    execute_with_context(request, decision, context)
                    if callable(execute_with_context)
                    else handler.execute(request, decision)
                )
            result = replace(
                result,
                metadata={
                    **dict(result.metadata),
                    "operational_context": context.safe_summary(),
                },
            )
            elapsed = max(0.0, (self._clock() - started).total_seconds())
            if timeout is not None and elapsed > timeout:
                error = RouteExecutionTimeoutError(
                    request_id=request.request_id,
                    route=decision.route,
                    target=target,
                    summary="Route execution exceeded the requested timeout.",
                    recoverable=False,
                )
                result = _failure_result(request, decision, started, self._clock(), error)
            if _requires_idempotency(decision):
                self._ledger.record(result)
            if self._execution_memory_recorder is not None:
                self._execution_memory_recorder.record(result, request=request)
            self._record_result(result, handler_name, target)
            return result
        except OperationalContextError as cause:
            error = OperationalRouteExecutionError(
                request_id=request.request_id,
                route=decision.route,
                target=target,
                summary="Operational context could not be built.",
                recoverable=True,
                safe_cause=cause.code,
            )
            finished = self._clock()
            result = _failure_result(request, decision, started, finished, error)
            self._record_result(result, handler_name, target)
            return result
        except OperationalRouteExecutionError as error:
            finished = self._clock()
            result = _failure_result(request, decision, started, finished, error)
            self._record_result(result, handler_name, target)
            return result

    def _validate(self, request: AtlasRequest, decision: RouteDecision) -> None:
        if not isinstance(request, AtlasRequest) or not isinstance(decision, RouteDecision):
            raise TypeError("request and decision must use Atlas operational models.")
        if request.request_id != decision.request_id:
            raise InvalidRouteDecisionExecutionError(
                request_id=request.request_id,
                route=decision.route,
                target=_decision_target(decision),
                summary="RouteDecision does not belong to this AtlasRequest.",
            )
        if decision.route not in self._enabled_routes:
            raise InvalidRouteDecisionExecutionError(
                request_id=request.request_id,
                route=decision.route,
                target=_decision_target(decision),
                summary="The selected route is disabled.",
                recoverable=True,
            )
        if decision.route is RequestRoute.SINGLE_TOOL and not decision.target_tool_name:
            raise InvalidRouteDecisionExecutionError(
                request_id=request.request_id,
                route=decision.route,
                target=None,
                summary="SINGLE_TOOL requires target_tool_name.",
                recoverable=True,
            )
        if decision.route is RequestRoute.AGENT_DELEGATION and not decision.target_agent_name:
            raise InvalidRouteDecisionExecutionError(
                request_id=request.request_id,
                route=decision.route,
                target=None,
                summary="AGENT_DELEGATION requires target_agent_name.",
                recoverable=True,
            )
        if (
            decision.route is RequestRoute.CLARIFICATION_REQUIRED
            and not decision.requires_clarification
        ):
            raise InvalidRouteDecisionExecutionError(
                request_id=request.request_id,
                route=decision.route,
                target=None,
                summary="Clarification route has an incoherent decision flag.",
                recoverable=True,
            )
        if (
            decision.requires_clarification
            and decision.route is not RequestRoute.CLARIFICATION_REQUIRED
        ):
            raise InvalidRouteDecisionExecutionError(
                request_id=request.request_id,
                route=decision.route,
                target=_decision_target(decision),
                summary="RouteDecision clarification flags are incoherent.",
                recoverable=True,
            )
        if (
            request.source is RequestSource.RESUME
            and decision.route
            not in {
                RequestRoute.RESUME_EXECUTION,
                RequestRoute.CLARIFICATION_REQUIRED,
            }
        ):
            raise InvalidRouteDecisionExecutionError(
                request_id=request.request_id,
                route=decision.route,
                target=_decision_target(decision),
                summary="A RESUME request is incompatible with the selected route.",
                recoverable=True,
            )
        context_session_id = (
            request.execution_context.session_id
            if request.execution_context is not None
            else None
        )
        if (
            decision.route is RequestRoute.RESUME_EXECUTION
            and decision.target_session_id
            and context_session_id
            and decision.target_session_id != context_session_id
        ):
            raise InvalidRouteDecisionExecutionError(
                request_id=request.request_id,
                route=decision.route,
                target=decision.target_session_id,
                summary="RouteDecision targets a different execution session.",
                recoverable=True,
            )
        if (
            request.source is RequestSource.SYSTEM
            and request.safety_context.trusted_source
            and "trusted_system_source" in decision.safety_flags
        ):
            raise InvalidRouteDecisionExecutionError(
                request_id=request.request_id,
                route=decision.route,
                target=_decision_target(decision),
                summary="SYSTEM source cannot elevate privileges.",
            )
        external_call = bool(
            {"external_call", "requires_external_calls"} & set(decision.safety_flags)
        )
        if external_call and not request.safety_context.allow_external_calls:
            raise RouteExecutionRejectedError(
                request_id=request.request_id,
                route=decision.route,
                target=_decision_target(decision),
                summary="External calls are disabled by RequestSafetyContext.",
                recoverable=True,
            )
        if external_call and request.safety_context.contains_sensitive_data:
            raise RouteExecutionRejectedError(
                request_id=request.request_id,
                route=decision.route,
                target=_decision_target(decision),
                summary="Sensitive request data cannot be sent to this external route.",
                recoverable=True,
            )
        memory_write = (
            decision.route is RequestRoute.MEMORY_QUERY
            and decision.memory_operation
            in {MemoryOperation.STORE, MemoryOperation.FORGET, MemoryOperation.UPDATE}
        )
        autonomous_write = (
            decision.route is RequestRoute.AUTONOMOUS_EXECUTION
            and not bool(
                request.execution_context
                and request.execution_context.dry_run
            )
        )
        system_write = (
            decision.route is RequestRoute.SYSTEM_COMMAND
            and decision.system_command is SystemCommand.CANCEL_EXECUTION
        )
        if (
            (memory_write or autonomous_write or system_write)
            and not request.safety_context.allow_side_effects
        ):
            raise RouteExecutionRejectedError(
                request_id=request.request_id,
                route=decision.route,
                target=_decision_target(decision),
                summary="Route side effects are disabled by RequestSafetyContext.",
                recoverable=True,
            )
        self._record(
            "route_target_validated",
            request,
            decision,
            handler=None,
            target=_decision_target(decision),
            status=None,
            duration=None,
            error_code=None,
        )

    def _record_result(
        self,
        result: RouteExecutionResult,
        handler: str | None,
        target: str | None,
    ) -> None:
        event_type = {
            RouteExecutionStatus.COMPLETED: "route_execution_completed",
            RouteExecutionStatus.FAILED: "route_execution_failed",
            RouteExecutionStatus.WAITING_CONFIRMATION: "route_execution_waiting_confirmation",
            RouteExecutionStatus.CLARIFICATION_REQUIRED: "route_execution_clarification_required",
            RouteExecutionStatus.UNSUPPORTED: "route_execution_unsupported",
            RouteExecutionStatus.REJECTED: "route_execution_failed",
            RouteExecutionStatus.CANCELLED: "route_execution_completed",
            RouteExecutionStatus.INTERRUPTED: "route_execution_completed",
        }[result.status]
        self._events.append(
            RouteExecutionEvent(
                event_type=event_type,
                request_id=result.request_id,
                route=result.route,
                handler=handler,
                target=target,
                status=result.status,
                duration=result.duration,
                error_code=result.error.code if result.error else None,
                timestamp=result.finished_at,
            )
        )

    def _record(
        self,
        event_type: str,
        request: AtlasRequest,
        decision: RouteDecision,
        *,
        handler: str | None,
        target: str | None,
        status: RouteExecutionStatus | None,
        duration: float | None,
        error_code: str | None,
    ) -> None:
        self._events.append(
            RouteExecutionEvent(
                event_type=event_type,
                request_id=request.request_id,
                route=decision.route,
                handler=handler,
                target=target,
                status=status,
                duration=duration,
                error_code=error_code,
                timestamp=self._clock(),
            )
        )


class RouteExecutionPresenter:
    """Temporary safe text presentation until the unified response phase."""

    def present(self, result: RouteExecutionResult) -> str:
        if result.status is RouteExecutionStatus.COMPLETED:
            return _present_output(result.output) or "Operacion completada."
        if result.status is RouteExecutionStatus.WAITING_CONFIRMATION:
            tool = result.target_tool_name or "la accion"
            return f"Necesito confirmacion antes de ejecutar {tool}."
        if result.status is RouteExecutionStatus.CLARIFICATION_REQUIRED:
            return result.clarification_question or "Necesito mas informacion para continuar."
        if result.status is RouteExecutionStatus.UNSUPPORTED:
            reason = _mapping_value(result.output, "reason")
            return f"No puedo ejecutar esta solicitud. {reason or ''}".strip()
        if result.status is RouteExecutionStatus.CANCELLED:
            return "La ejecucion fue cancelada."
        if result.status is RouteExecutionStatus.INTERRUPTED:
            return "La ejecucion fue interrumpida."
        if result.status is RouteExecutionStatus.REJECTED:
            return result.error.summary if result.error else "La ejecucion fue rechazada."
        return result.error.summary if result.error else "La ejecucion fallo."


def build_default_route_handlers(
    *,
    direct_responder: Callable[[AtlasRequest], Any] | None = None,
    direct_streaming_responder: Callable[[AtlasRequest], Any] | None = None,
    memory: object | None = None,
    tool_registry: ToolRegistry | None = None,
    tool_executor: ToolExecutor | None = None,
    single_tool_runner: SingleToolRunner | None = None,
    agent_registry: AgentRegistry | None = None,
    model_selector: Callable[[str], str] | None = None,
    autonomous_orchestrator: AutonomousExecutionOrchestrator | None = None,
    execution_supervisor: object | None = None,
    diagnostics: Callable[[], Any] | None = None,
    clock: Callable[[], datetime] | None = None,
) -> Mapping[RequestRoute, RouteHandler]:
    """Build exactly one explicit handler for every RequestRoute."""

    handlers: dict[RequestRoute, RouteHandler] = {
        RequestRoute.DIRECT_RESPONSE: DirectResponseRouteHandler(
            direct_responder,
            direct_streaming_responder,
            clock=clock,
        ),
        RequestRoute.MEMORY_QUERY: MemoryRouteHandler(memory, clock=clock),
        RequestRoute.SINGLE_TOOL: SingleToolRouteHandler(
            tool_registry,
            tool_executor,
            single_tool_runner,
            clock=clock,
        ),
        RequestRoute.AGENT_DELEGATION: AgentDelegationRouteHandler(
            agent_registry,
            model_selector=model_selector,
            clock=clock,
        ),
        RequestRoute.AUTONOMOUS_EXECUTION: AutonomousExecutionRouteHandler(
            autonomous_orchestrator,
            clock=clock,
        ),
        RequestRoute.RESUME_EXECUTION: ResumeExecutionRouteHandler(
            autonomous_orchestrator,
            single_tool_runner,
            clock=clock,
        ),
        RequestRoute.SYSTEM_COMMAND: SystemCommandRouteHandler(
            supervisor=execution_supervisor,
            autonomous_orchestrator=autonomous_orchestrator,
            diagnostics=diagnostics,
            clock=clock,
        ),
        RequestRoute.CLARIFICATION_REQUIRED: ClarificationRouteHandler(clock=clock),
        RequestRoute.UNSUPPORTED: UnsupportedRouteHandler(clock=clock),
    }
    return MappingProxyType(handlers)


def _clarification_result(
    handler: _BaseRouteHandler,
    request: AtlasRequest,
    decision: RouteDecision,
    started: datetime,
    question: str,
    missing: tuple[str, ...],
    *,
    target: str | None = None,
) -> RouteExecutionResult:
    return handler._result(
        request,
        decision,
        started_at=started,
        status=RouteExecutionStatus.CLARIFICATION_REQUIRED,
        output={
            "missing_information": tuple(missing),
            "candidate_route": decision.fallback_route.value
            if decision.fallback_route
            else None,
        },
        requires_clarification=True,
        clarification_question=question,
        target=target,
        action="route_execution_clarification_required",
    )


def _tool_run_result(
    handler: _BaseRouteHandler,
    request: AtlasRequest,
    decision: RouteDecision,
    started: datetime,
    invocation: SingleToolInvocation,
    run_result: ToolRunResult,
    *,
    session_id: str | None = None,
) -> RouteExecutionResult:
    if run_result.status == "missing_argument":
        field_name = run_result.error_field or "arguments"
        return _clarification_result(
            handler,
            request,
            decision,
            started,
            f"Falta el argumento '{field_name}' para ejecutar la herramienta.",
            (field_name,),
            target=invocation.tool_name,
        )
    if run_result.status == "confirmation_required":
        return handler._result(
            request,
            decision,
            started_at=started,
            status=RouteExecutionStatus.WAITING_CONFIRMATION,
            output={
                "tool_name": invocation.tool_name,
                "action": run_result.intent.action,
                "risk_level": invocation.risk_level,
                "missing_information": (),
            },
            requires_confirmation=True,
            execution_reference=run_result.confirmation_id,
            session_id=run_result.confirmation_id,
            target=invocation.tool_name,
            metadata={"arguments": invocation.arguments},
            action="route_execution_waiting_confirmation",
        )
    if run_result.status == "cancelled":
        return handler._result(
            request,
            decision,
            started_at=started,
            status=RouteExecutionStatus.CANCELLED,
            output={"tool_name": invocation.tool_name},
            session_id=session_id,
            target=invocation.tool_name,
            action="route_execution_completed",
        )
    if run_result.success:
        return handler._result(
            request,
            decision,
            started_at=started,
            status=RouteExecutionStatus.COMPLETED,
            output=run_result.result,
            side_effects_performed=run_result.executed,
            execution_reference=run_result.confirmation_id,
            session_id=session_id,
            target=invocation.tool_name,
            metadata={"arguments": invocation.arguments},
            action="route_execution_completed",
        )
    error = OperationalRouteExecutionError(
        request_id=request.request_id,
        route=decision.route,
        target=invocation.tool_name,
        summary=run_result.error_message or "Tool execution failed.",
        recoverable=not run_result.executed,
        safe_cause=run_result.exception_type,
    )
    return handler._result(
        request,
        decision,
        started_at=started,
        status=RouteExecutionStatus.FAILED,
        error=error,
        target=invocation.tool_name,
        action="route_execution_failed",
    )


def _autonomous_result(
    handler: _BaseRouteHandler,
    request: AtlasRequest,
    decision: RouteDecision,
    started: datetime,
    autonomous: AutonomousExecutionResult,
) -> RouteExecutionResult:
    status = {
        AutonomousExecutionOutcome.COMPLETED: RouteExecutionStatus.COMPLETED,
        AutonomousExecutionOutcome.CANCELLED: RouteExecutionStatus.CANCELLED,
        AutonomousExecutionOutcome.WAITING_CONFIRMATION: RouteExecutionStatus.WAITING_CONFIRMATION,
        AutonomousExecutionOutcome.INTERRUPTED: RouteExecutionStatus.INTERRUPTED,
    }.get(autonomous.outcome, RouteExecutionStatus.FAILED)
    error: OperationalRouteExecutionError | None = None
    if status is RouteExecutionStatus.FAILED:
        error = OperationalRouteExecutionError(
            request_id=request.request_id,
            route=decision.route,
            target=autonomous.session_id,
            summary=autonomous.summary or "Autonomous execution failed.",
            recoverable=autonomous.requires_manual_review,
            safe_cause=autonomous.outcome.value,
        )
    output = {
        "summary": autonomous.summary,
        "outcome": autonomous.outcome.value,
        "completed_step_ids": autonomous.completed_step_ids,
        "failed_step_ids": autonomous.failed_step_ids,
        "blocked_step_ids": autonomous.blocked_step_ids,
        "errors": autonomous.errors,
        "budget_usage": autonomous.budget_usage,
    }
    return handler._result(
        request,
        decision,
        started_at=started,
        status=status,
        output=output,
        error=error,
        session_id=autonomous.session_id,
        execution_reference=autonomous.session_id,
        requires_confirmation=autonomous.requires_confirmation,
        side_effects_performed=bool(autonomous.completed_step_ids),
        target=autonomous.session_id,
        action=(
            "route_execution_waiting_confirmation"
            if status is RouteExecutionStatus.WAITING_CONFIRMATION
            else "route_execution_completed"
            if status in {
                RouteExecutionStatus.COMPLETED,
                RouteExecutionStatus.CANCELLED,
                RouteExecutionStatus.INTERRUPTED,
            }
            else "route_execution_failed"
        ),
    )


def _failure_result(
    request: AtlasRequest,
    decision: RouteDecision,
    started: datetime,
    finished: datetime,
    error: OperationalRouteExecutionError,
) -> RouteExecutionResult:
    status = (
        RouteExecutionStatus.REJECTED
        if isinstance(error, RouteExecutionRejectedError)
        else RouteExecutionStatus.FAILED
    )
    trace = (
        RouteExecutionTraceEntry(
            sequence=1,
            timestamp=started,
            request_id=request.request_id,
            route=decision.route,
            action="route_execution_started",
            target=error.target,
            status_before=None,
            status_after=None,
            summary="route execution started",
        ),
        RouteExecutionTraceEntry(
            sequence=2,
            timestamp=finished,
            request_id=request.request_id,
            route=decision.route,
            action=(
                "route_execution_timeout"
                if isinstance(error, RouteExecutionTimeoutError)
                else "route_execution_failed"
            ),
            target=error.target,
            status_before=None,
            status_after=status,
            error_code=error.code,
            summary=error.summary,
        ),
    )
    return RouteExecutionResult(
        request_id=request.request_id,
        route=decision.route,
        status=status,
        output=None,
        error=error.to_info(),
        started_at=started,
        finished_at=finished,
        duration=max(0.0, (finished - started).total_seconds()),
        target_tool_name=decision.target_tool_name,
        target_agent_name=decision.target_agent_name,
        session_id=decision.target_session_id,
        trace=trace,
    )


def _prepare_tool_arguments(
    request: AtlasRequest,
    schema: object | None,
    tool_name: str,
) -> tuple[Mapping[str, Any], tuple[str, ...]]:
    arguments: dict[str, Any] = {}
    for key in ("tool_arguments", "arguments"):
        candidate = request.metadata.get(key)
        if isinstance(candidate, Mapping):
            arguments.update(candidate)
    parameters = tuple(getattr(schema, "parameters", ())) if schema is not None else ()
    known_names = _known_argument_names(tool_name)
    for name in known_names:
        if name in arguments:
            continue
        extracted = _extract_known_argument(name, request.content)
        if extracted is not None:
            arguments[name] = extracted
    for parameter in parameters:
        if parameter.name in arguments:
            continue
        extracted = _extract_known_argument(parameter.name, request.content)
        if extracted is not None:
            arguments[parameter.name] = extracted
    missing = tuple(
        parameter.name
        for parameter in parameters
        if parameter.required
        and not parameter.has_default
        and parameter.name not in arguments
    )
    if schema is not None and not missing:
        validation = schema.validate("<route>", arguments)
        if not validation.is_valid:
            missing = tuple(
                error.parameter_name or "arguments"
                for error in validation.errors
            )
        else:
            arguments = dict(validation.normalized_arguments)
    return MappingProxyType(dict(sorted(arguments.items()))), tuple(dict.fromkeys(missing))


def _known_argument_names(tool_name: str) -> tuple[str, ...]:
    exact: dict[str, tuple[str, ...]] = {
        "desktop.open_application": ("application",),
        "desktop.open_file": ("path", "application"),
        "desktop.open_folder": ("path",),
        "desktop.type_text": ("text", "window_title"),
        "desktop.copy_clipboard_text": ("text",),
        "desktop.get_process": ("pid",),
        "desktop.is_process_running": ("application",),
        "desktop.terminate_process": ("pid",),
        "read_file": ("path",),
        "write_file": ("path", "content"),
        "list_directory": ("path",),
        "project_tree": ("path",),
    }
    return exact.get(tool_name, ())


def _extract_known_argument(name: str, content: str) -> Any:
    quoted = re.findall(r"""["']([^"']+)["']""", content)
    if name in {"path", "application", "text", "title", "window_title", "query", "content"}:
        if quoted:
            return quoted[0]
    if name == "path":
        match = re.search(
            r"(?:[A-Za-z]:[\\/])?[\w .()_-]+(?:[\\/][\w .()_-]+)*\.[A-Za-z0-9]{1,8}",
            content,
        )
        return match.group(0).strip() if match else None
    if name == "application":
        match = re.search(
            r"\b(?:abre|abrir|inicia|lanza|open)\s+(?:la\s+|el\s+)?(.+)$",
            content,
            re.IGNORECASE,
        )
        return match.group(1).strip() if match else None
    if name == "pid":
        match = re.search(r"\b(?:pid\s*)?(\d{2,})\b", content, re.IGNORECASE)
        return int(match.group(1)) if match else None
    if name == "keys":
        match = re.search(r"\b((?:ctrl|control|alt|shift|win)(?:\s*\+\s*\w+)+)\b", content, re.I)
        return [part.strip().lower() for part in match.group(1).split("+")] if match else None
    return None


def _tool_has_side_effects(descriptor: object) -> bool:
    name = str(getattr(descriptor, "name", ""))
    read_only_markers = (
        "read",
        "list",
        "get",
        "status",
        "has_text",
        "screen_size",
        "cursor_position",
        "screenshot",
        "tree",
    )
    return bool(getattr(descriptor, "dangerous", False)) or not any(
        marker in name for marker in read_only_markers
    )


def _memory_content(request: AtlasRequest) -> str:
    explicit = request.metadata.get("memory_content")
    if isinstance(explicit, str):
        return explicit.strip()
    normalized = re.sub(
        r"^\s*(?:recuerda(?:\s+que)?|remember(?:\s+that)?|guarda(?:\s+que)?)\s*",
        "",
        request.content,
        flags=re.IGNORECASE,
    )
    return normalized.strip()


def _memory_query(request: AtlasRequest) -> str:
    explicit = request.metadata.get("memory_query")
    if isinstance(explicit, str):
        return explicit.strip()
    return re.sub(
        r"^\s*(?:que\s+recuerdas(?:\s+de)?|recupera|busca\s+en\s+memoria|memoria)\s*",
        "",
        request.content,
        flags=re.IGNORECASE,
    ).strip()


def _memory_category(request: AtlasRequest) -> MemoryCategory:
    explicit = request.metadata.get("memory_category")
    if isinstance(explicit, str):
        return MemoryCategory(explicit)
    normalized = request.content.casefold()
    if "prefiero" in normalized or "preferencia" in normalized:
        return MemoryCategory.USER_PREFERENCE
    if "proyecto" in normalized:
        return MemoryCategory.PROJECT_FACT
    if "decid" in normalized:
        return MemoryCategory.DECISION
    return MemoryCategory.CONVERSATION_NOTE


def _memory_optional_category(request: AtlasRequest) -> MemoryCategory | None:
    explicit = request.metadata.get("memory_category")
    return MemoryCategory(explicit) if isinstance(explicit, str) else None


def _memory_importance(request: AtlasRequest) -> float:
    value = request.metadata.get("importance", 0.5)
    return float(value)


def _memory_optional_importance(request: AtlasRequest) -> float | None:
    value = request.metadata.get("importance")
    return float(value) if value is not None else None


def _memory_tags(request: AtlasRequest) -> tuple[str, ...]:
    value = request.metadata.get("tags", ())
    if isinstance(value, str):
        return (value,)
    if isinstance(value, (tuple, list)):
        return tuple(str(item) for item in value)
    return ()


def _memory_optional_tags(request: AtlasRequest) -> tuple[str, ...] | None:
    return _memory_tags(request) if "tags" in request.metadata else None


def _memory_optional_sensitive(request: AtlasRequest) -> bool | None:
    value = request.metadata.get("sensitive")
    return bool(value) if value is not None else None


def _memory_expires_at(request: AtlasRequest) -> datetime | None:
    value = request.metadata.get("expires_at")
    if not isinstance(value, str) or not value.strip():
        return None
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise InvalidMemoryEntryError("expires_at must be timezone-aware.")
    return parsed


def _memory_safe_metadata(request: AtlasRequest) -> Mapping[str, Any]:
    value = request.metadata.get("memory_metadata")
    return value if isinstance(value, Mapping) else {}


def _memory_categories_filter(request: AtlasRequest) -> tuple[MemoryCategory, ...]:
    value = request.metadata.get("memory_categories", ())
    if isinstance(value, str):
        values = (value,)
    elif isinstance(value, (tuple, list)):
        values = tuple(str(item) for item in value)
    else:
        values = ()
    return tuple(MemoryCategory(item) for item in values)


def _memory_limit(request: AtlasRequest) -> int | None:
    value = request.metadata.get("limit")
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise InvalidMemoryEntryError("Memory query limit must be a positive integer.")
    return value


def _resolve_memory_id(
    memory: object,
    request: AtlasRequest,
) -> tuple[str | None, bool]:
    explicit = request.metadata.get("memory_id")
    if isinstance(explicit, str) and explicit.strip():
        return explicit.strip(), False
    match = re.search(r"\bmemory-\d{6}\b", request.content, re.IGNORECASE)
    if match:
        return match.group(0).lower(), False
    retrieve_entries = getattr(memory, "retrieve_entries", None)
    if not callable(retrieve_entries):
        return None, False
    query = re.sub(
        r"^\s*(?:olvida|forget|actualiza|update)\s*",
        "",
        request.content,
        flags=re.IGNORECASE,
    ).strip()
    if not query:
        return None, False
    matches = retrieve_entries(query, include_sensitive=False)
    if len(matches) == 1:
        return matches[0].memory_id, False
    return None, len(matches) > 1


def _memory_update_content(request: AtlasRequest, memory_id: str) -> str:
    explicit = request.metadata.get("memory_content")
    if isinstance(explicit, str):
        return explicit.strip()
    without_prefix = re.sub(
        r"^\s*(?:actualiza|update)\s*",
        "",
        request.content,
        flags=re.IGNORECASE,
    )
    without_id = re.sub(re.escape(memory_id), "", without_prefix, flags=re.IGNORECASE)
    return re.sub(r"^\s*(?:a|con|to)\s*", "", without_id, flags=re.IGNORECASE).strip()


def _memory_entry_view(entry: MemoryEntry) -> Mapping[str, Any]:
    return {
        "memory_id": entry.memory_id,
        "content": entry.content,
        "category": entry.category.value,
        "created_at": entry.created_at.isoformat(),
        "updated_at": entry.updated_at.isoformat(),
        "importance": entry.importance,
        "tags": entry.tags,
        "active": entry.active,
        "expires_at": entry.expires_at.isoformat() if entry.expires_at else None,
    }


def _safe_memory_error_summary(
    cause: Exception,
    operation: MemoryOperation,
) -> str:
    if isinstance(cause, SensitiveMemoryRejectedError):
        return "Sensitive memory content was rejected by policy."
    if isinstance(cause, MemoryEntryNotFoundError):
        return "The requested memory entry was not found."
    if isinstance(cause, InvalidMemoryEntryError):
        return f"Memory {operation.value} request is invalid."
    return f"Memory {operation.value} failed."


def _invoke_contextual_callable(
    function: Callable[..., Any],
    request: AtlasRequest,
    context: OperationalContext | None,
) -> Any:
    try:
        inspect.signature(function).bind(request, context)
    except (TypeError, ValueError):
        return function(request)
    return function(request, context)


def _context_messages(
    context: OperationalContext | None,
) -> list[dict[str, str]]:
    if context is None:
        return []
    messages = [
        {
            "role": str(message["role"]),
            "content": str(message["content"]),
        }
        for message in context.recent_messages
    ]
    prompt_context = context.prompt_context()
    if prompt_context:
        messages.append(
            {
                "role": "system",
                "content": "Contexto operativo limitado:\n" + prompt_context,
            }
        )
    return messages


def _empty_operational_context(
    request: AtlasRequest,
    generated_at: datetime,
) -> OperationalContext:
    return OperationalContext(
        request_id=request.request_id,
        conversation_id=request.conversation_id,
        recent_messages=(),
        relevant_memories=(),
        user_preferences=(),
        project_context=(),
        execution_context=(
            asdict(request.execution_context)
            if request.execution_context is not None
            else {}
        ),
        selected_memory_ids=(),
        total_characters=0,
        truncated=False,
        generated_at=generated_at,
    )


def _decision_missing_information(decision: RouteDecision) -> tuple[str, ...]:
    for rule in decision.matched_rules:
        if "missing_session" in rule:
            return ("session_id",)
        if "missing_target" in rule or "missing_reference" in rule:
            return ("target",)
        if "ambiguous" in rule:
            return ("selection",)
    return ()


def _decision_target(decision: RouteDecision) -> str | None:
    return (
        decision.target_tool_name
        or decision.target_agent_name
        or decision.target_session_id
        or (decision.system_command.value if decision.system_command else None)
        or (decision.memory_operation.value if decision.memory_operation else None)
    )


def _requires_idempotency(decision: RouteDecision) -> bool:
    if decision.route in {
        RequestRoute.SINGLE_TOOL,
        RequestRoute.AUTONOMOUS_EXECUTION,
        RequestRoute.RESUME_EXECUTION,
    }:
        return True
    if decision.route is RequestRoute.MEMORY_QUERY:
        return decision.memory_operation in {
            MemoryOperation.STORE,
            MemoryOperation.FORGET,
            MemoryOperation.UPDATE,
        }
    if decision.route is RequestRoute.SYSTEM_COMMAND:
        return decision.system_command in {
            SystemCommand.CANCEL_EXECUTION,
            SystemCommand.VOICE_MODE,
            SystemCommand.STOP_LISTENING,
        }
    return False


def _is_cancel_request(request: AtlasRequest) -> bool:
    normalized = request.content.strip().lower()
    return normalized.startswith(("cancela", "cancelar", "cancel "))


def _call_optional(target: object | None, method: str) -> Any:
    function = getattr(target, method, None)
    return function() if callable(function) else None


def _freeze_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType(
        {
            str(key): _freeze_safe(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            if not any(secret in str(key).lower() for secret in _SECRET_KEYS)
        }
    )


def _freeze_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, datetime):
        return value.isoformat()
    if is_dataclass(value) and not isinstance(value, type):
        return _freeze_mapping(
            {
                item.name: getattr(value, item.name)
                for item in fields(value)
            }
        )
    if isinstance(value, Mapping):
        return _freeze_mapping(value)
    if isinstance(value, (list, tuple, set, frozenset)):
        return tuple(_freeze_safe(item) for item in value)
    return f"<{type(value).__name__}>"


def _sanitize_summary(value: str) -> str:
    sanitized = " ".join(str(value).split())
    return sanitized[:300]


def _require_aware(value: datetime, name: str) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware.")


def _present_output(output: Any) -> str:
    if isinstance(output, str):
        return output
    if isinstance(output, Mapping):
        summary = output.get("summary")
        if isinstance(summary, str) and summary:
            return summary
        operation = output.get("operation")
        if operation == MemoryOperation.STORE.value:
            return f"Memoria guardada con id {output.get('memory_id', 'desconocido')}."
        if operation == MemoryOperation.RETRIEVE.value:
            count = int(output.get("count", 0))
            return (
                f"Se recuperaron {count} entradas de memoria."
                if count
                else "No se encontraron recuerdos coincidentes."
            )
        if operation == MemoryOperation.LIST.value:
            return f"Hay {int(output.get('count', 0))} entradas de memoria activas."
        if operation == MemoryOperation.FORGET.value:
            return f"Entrada {output.get('memory_id', '')} olvidada.".strip()
        if operation == MemoryOperation.UPDATE.value:
            return f"Entrada {output.get('memory_id', '')} actualizada.".strip()
        if "signal" in output:
            return str(output["signal"])
    return "Operacion completada." if output is not None else ""


def _mapping_value(value: Any, key: str) -> Any:
    return value.get(key) if isinstance(value, Mapping) else None
