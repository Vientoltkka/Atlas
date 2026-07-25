"""Controlled deterministic execution for specialized Atlas agents."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import inspect
import json
import math
import re
import types
from types import MappingProxyType
from typing import Protocol, runtime_checkable

from core.agent_context import (
    AgentContext,
    AgentContextBuilder,
    AgentContextRequest,
    AgentContextStatus,
)
from core.agent_registry import AgentDefinition, validate_agent_id
from core.agent_resolver import (
    AgentResolutionRequest,
    AgentResolutionResult,
    AgentResolutionStatus,
    AgentResolver,
    agent_resolution_request_signature,
)


MAX_AGENT_EXECUTION_IDS = 32
MAX_AGENT_EXECUTION_METADATA_ITEMS = 16
MAX_AGENT_EXECUTION_RESULT_DEPTH = 4
MAX_AGENT_EXECUTION_RESULT_STRING_LENGTH = 1_000
MAX_AGENT_EXECUTION_RESULT_SEQUENCE_ITEMS = 32
MAX_AGENT_EXECUTION_RESULT_MAPPING_ITEMS = 32
MAX_AGENT_EXECUTION_RESULT_TOTAL_ITEMS = 256
_PERMISSION_IDS = frozenset(
    {
        "can_read_project",
        "can_write_files",
        "can_execute_tools",
        "can_modify_memory",
        "can_use_network",
        "requires_confirmation",
    }
)
_SENSITIVE_KEY_PARTS = (
    "token",
    "access_token",
    "refresh_token",
    "secret",
    "password",
    "api_key",
    "apikey",
    "authorization",
    "cookie",
    "private_key",
    "credential",
)


class AgentExecutionError(RuntimeError):
    """Base error for controlled specialized-agent execution."""


class InvalidAgentExecutionRequestError(AgentExecutionError):
    """Raised when an execution request is structurally invalid."""


class AgentHandlerRegistryError(AgentExecutionError):
    """Base error for handler registry operations."""


class InvalidAgentHandlerError(AgentHandlerRegistryError):
    """Raised when a handler object is malformed."""


class AgentHandlerAlreadyRegisteredError(AgentHandlerRegistryError):
    """Raised when registering a duplicate handler id."""


class AgentHandlerNotFoundError(AgentHandlerRegistryError):
    """Raised when a handler id is not registered."""


class AgentExecutionStatus(str, Enum):
    """Structured terminal statuses for controlled agent execution."""

    COMPLETED = "COMPLETED"
    INVALID_REQUEST = "INVALID_REQUEST"
    NO_AGENT_CANDIDATES = "NO_AGENT_CANDIDATES"
    AGENT_AMBIGUOUS = "AGENT_AMBIGUOUS"
    AGENT_DISABLED = "AGENT_DISABLED"
    CONTEXT_BUILD_FAILED = "CONTEXT_BUILD_FAILED"
    HANDLER_UNAVAILABLE = "HANDLER_UNAVAILABLE"
    PERMISSION_DENIED = "PERMISSION_DENIED"
    CAPABILITY_NOT_ALLOWED = "CAPABILITY_NOT_ALLOWED"
    EXECUTION_FAILED = "EXECUTION_FAILED"
    CANCELLED = "CANCELLED"
    INTERNAL_ERROR = "INTERNAL_ERROR"


@runtime_checkable
class AgentHandler(Protocol):
    """Protocol for a deterministic specialized-agent handler."""

    @property
    def agent_id(self) -> str:
        """Return the exact agent id this handler serves."""

    def handle(
        self,
        context: AgentContext,
    ) -> Mapping[str, object]:
        """Return a structured safe result from an authorized AgentContext."""


@dataclass(frozen=True, slots=True)
class AgentExecutionRequest:
    """Structured request for resolving, contextualizing, and handling one agent."""

    resolution_request: AgentResolutionRequest
    task_id: str | None = None
    execution_id: str | None = None
    correlation_id: str | None = None
    session_id: str | None = None
    user_input: str | None = None
    structured_input: Mapping[str, object] | None = None
    shared_context: Mapping[str, object] | None = None
    conversation_context: Sequence[object] | None = None
    memory_context: Mapping[str, object] | None = None
    tool_results: Mapping[str, object] | None = None
    workflow_results: Mapping[str, object] | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)
    required_capability_ids: tuple[str, ...] = ()
    required_permission_ids: tuple[str, ...] = ()
    cancel_requested: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.resolution_request, AgentResolutionRequest):
            raise InvalidAgentExecutionRequestError("resolution_request must be AgentResolutionRequest.")
        for field_name in ("task_id", "execution_id", "correlation_id", "session_id"):
            value = getattr(self, field_name)
            if value is not None:
                object.__setattr__(self, field_name, _safe_identifier(value, field_name))
        if self.user_input is not None and not isinstance(self.user_input, str):
            raise InvalidAgentExecutionRequestError("user_input must be a string or None.")
        if not isinstance(self.cancel_requested, bool):
            raise InvalidAgentExecutionRequestError("cancel_requested must be a bool.")
        object.__setattr__(
            self,
            "required_capability_ids",
            _identifier_tuple(self.required_capability_ids, "required_capability_ids"),
        )
        object.__setattr__(
            self,
            "required_permission_ids",
            _permission_tuple(self.required_permission_ids, "required_permission_ids"),
        )
        object.__setattr__(self, "metadata", MappingProxyType(_sanitize_metadata(self.metadata)))


@dataclass(frozen=True, slots=True)
class AgentExecutionEvent:
    """Safe event emitted during controlled execution."""

    name: str
    status: AgentExecutionStatus
    details: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _safe_identifier(self.name, "event name"))
        object.__setattr__(self, "status", _execution_status(self.status))
        object.__setattr__(self, "details", MappingProxyType(_sanitize_metadata(self.details)))


@dataclass(frozen=True, slots=True)
class AgentExecutionResult:
    """Immutable result of a controlled specialized-agent execution."""

    status: AgentExecutionStatus
    request_signature: str
    execution_id: str | None = None
    correlation_id: str | None = None
    agent_id: str | None = None
    resolution_result: AgentResolutionResult | None = None
    context: AgentContext | None = None
    output: Mapping[str, object] | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)
    events: tuple[AgentExecutionEvent, ...] = ()
    error_code: str | None = None
    safe_message: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", _execution_status(self.status))
        if self.execution_id is not None:
            object.__setattr__(self, "execution_id", _safe_identifier(self.execution_id, "execution_id"))
        if self.correlation_id is not None:
            object.__setattr__(self, "correlation_id", _safe_identifier(self.correlation_id, "correlation_id"))
        if self.agent_id is not None:
            object.__setattr__(self, "agent_id", validate_agent_id(self.agent_id))
        if self.context is not None and not isinstance(self.context, AgentContext):
            raise InvalidAgentExecutionRequestError("context must be AgentContext or None.")
        if self.output is not None:
            if not isinstance(self.output, Mapping):
                raise InvalidAgentExecutionRequestError("output must be a mapping or None.")
            object.__setattr__(self, "output", MappingProxyType(dict(self.output)))
        object.__setattr__(self, "metadata", MappingProxyType(_sanitize_metadata(self.metadata)))
        object.__setattr__(self, "events", tuple(self.events))
        if not all(isinstance(event, AgentExecutionEvent) for event in self.events):
            raise InvalidAgentExecutionRequestError("events must be AgentExecutionEvent values.")
        if self.safe_message is not None:
            object.__setattr__(self, "safe_message", _safe_message(self.safe_message))

    @property
    def completed(self) -> bool:
        """Return whether the handler completed successfully."""

        return self.status is AgentExecutionStatus.COMPLETED


class AgentHandlerRegistry:
    """Explicit in-memory registry for deterministic agent handlers."""

    def __init__(
        self,
        handlers: Iterable[AgentHandler] = (),
    ) -> None:
        self._handlers: OrderedDict[str, AgentHandler] = OrderedDict()
        for handler in handlers:
            self.register(handler)

    def register(
        self,
        handler: AgentHandler,
        *,
        replace: bool = False,
    ) -> AgentHandler:
        """Register a handler by its declared agent id."""

        agent_id = _handler_agent_id(handler)
        if agent_id in self._handlers and not replace:
            raise AgentHandlerAlreadyRegisteredError(f"handler already registered: {agent_id}")
        self._handlers[agent_id] = handler
        return handler

    def unregister(
        self,
        agent_id: str,
    ) -> bool:
        """Remove a handler if present and return whether one was removed."""

        normalized = validate_agent_id(agent_id)
        return self._handlers.pop(normalized, None) is not None

    def get(
        self,
        agent_id: str,
    ) -> AgentHandler:
        """Return a handler by agent id."""

        normalized = validate_agent_id(agent_id)
        try:
            return self._handlers[normalized]
        except KeyError as error:
            raise AgentHandlerNotFoundError(f"handler not found: {normalized}") from error

    def contains(
        self,
        agent_id: str,
    ) -> bool:
        """Return whether a handler id is registered."""

        return validate_agent_id(agent_id) in self._handlers

    def list_handlers(self) -> tuple[AgentHandler, ...]:
        """Return handlers in deterministic registration order."""

        return tuple(self._handlers.values())

    def clear(self) -> None:
        """Remove all registered handlers."""

        self._handlers.clear()

    def __len__(self) -> int:
        return len(self._handlers)


class AgentExecutor:
    """Resolve one agent, build its context, and invoke one registered handler."""

    def __init__(
        self,
        agent_resolver: AgentResolver,
        agent_context_builder: AgentContextBuilder,
        agent_handler_registry: AgentHandlerRegistry,
    ) -> None:
        if not isinstance(agent_resolver, AgentResolver):
            raise AgentExecutionError("AgentExecutor requires AgentResolver.")
        if not isinstance(agent_context_builder, AgentContextBuilder):
            raise AgentExecutionError("AgentExecutor requires AgentContextBuilder.")
        if not isinstance(agent_handler_registry, AgentHandlerRegistry):
            raise AgentExecutionError("AgentExecutor requires AgentHandlerRegistry.")
        self._agent_resolver = agent_resolver
        self._agent_context_builder = agent_context_builder
        self._agent_handler_registry = agent_handler_registry

    def execute(
        self,
        request: AgentExecutionRequest,
    ) -> AgentExecutionResult:
        """Execute one specialized agent through a registered handler."""

        if not isinstance(request, AgentExecutionRequest):
            return _result(
                AgentExecutionStatus.INVALID_REQUEST,
                "",
                error_code="INVALID_REQUEST",
                safe_message="request must be AgentExecutionRequest.",
            )
        try:
            signature = agent_execution_request_signature(request)
        except AgentExecutionError as error:
            return _result(
                AgentExecutionStatus.INVALID_REQUEST,
                "",
                execution_id=request.execution_id,
                correlation_id=request.correlation_id,
                error_code="INVALID_REQUEST",
                safe_message=str(error),
            )

        events: list[AgentExecutionEvent] = [
            _event("agent_execution_started", AgentExecutionStatus.COMPLETED, request=request),
        ]
        if request.cancel_requested:
            events.append(_event("agent_execution_failed", AgentExecutionStatus.CANCELLED, request=request))
            return _result(
                AgentExecutionStatus.CANCELLED,
                signature,
                execution_id=request.execution_id,
                correlation_id=request.correlation_id,
                events=events,
                error_code="CANCELLED",
                safe_message="execution was cancelled before resolution.",
            )

        resolution_result = self._agent_resolver.resolve(request.resolution_request)
        resolution_status = _map_resolution_status(resolution_result.status)
        if resolution_status is not AgentExecutionStatus.COMPLETED:
            events.append(_event("agent_execution_failed", resolution_status, request=request))
            return _result(
                resolution_status,
                signature,
                execution_id=request.execution_id,
                correlation_id=request.correlation_id,
                resolution_result=resolution_result,
                events=events,
                error_code=resolution_result.error_code,
                safe_message=resolution_result.error_message,
            )

        agent = resolution_result.selected_agent
        if agent is None:
            events.append(_event("agent_execution_failed", AgentExecutionStatus.NO_AGENT_CANDIDATES, request=request))
            return _result(
                AgentExecutionStatus.NO_AGENT_CANDIDATES,
                signature,
                execution_id=request.execution_id,
                correlation_id=request.correlation_id,
                resolution_result=resolution_result,
                events=events,
                error_code="NO_AGENT_CANDIDATES",
                safe_message="resolution did not select an agent.",
            )
        events.append(_event("agent_resolution_succeeded", AgentExecutionStatus.COMPLETED, request=request, agent=agent))

        if not agent.enabled:
            events.append(_event("agent_execution_failed", AgentExecutionStatus.AGENT_DISABLED, request=request, agent=agent))
            return _result(
                AgentExecutionStatus.AGENT_DISABLED,
                signature,
                execution_id=request.execution_id,
                correlation_id=request.correlation_id,
                agent_id=agent.agent_id,
                resolution_result=resolution_result,
                events=events,
                error_code="AGENT_DISABLED",
                safe_message="selected agent is disabled.",
            )
        missing_permissions = tuple(
            permission for permission in request.required_permission_ids if not bool(getattr(agent.permissions, permission))
        )
        if missing_permissions:
            events.append(_event("agent_execution_failed", AgentExecutionStatus.PERMISSION_DENIED, request=request, agent=agent))
            return _result(
                AgentExecutionStatus.PERMISSION_DENIED,
                signature,
                execution_id=request.execution_id,
                correlation_id=request.correlation_id,
                agent_id=agent.agent_id,
                resolution_result=resolution_result,
                events=events,
                error_code="PERMISSION_DENIED",
                safe_message="selected agent does not grant a required permission.",
            )
        missing_capabilities = tuple(
            capability for capability in request.required_capability_ids if capability not in agent.capabilities.capabilities
        )
        if missing_capabilities:
            events.append(
                _event("agent_execution_failed", AgentExecutionStatus.CAPABILITY_NOT_ALLOWED, request=request, agent=agent)
            )
            return _result(
                AgentExecutionStatus.CAPABILITY_NOT_ALLOWED,
                signature,
                execution_id=request.execution_id,
                correlation_id=request.correlation_id,
                agent_id=agent.agent_id,
                resolution_result=resolution_result,
                events=events,
                error_code="CAPABILITY_NOT_ALLOWED",
                safe_message="selected agent does not declare a required capability.",
            )

        context_request = AgentContextRequest(
            agent=agent,
            task_id=request.task_id,
            execution_id=request.execution_id,
            session_id=request.session_id,
            user_input=request.user_input,
            structured_input=request.structured_input,
            shared_context=request.shared_context,
            conversation_context=request.conversation_context,
            memory_context=request.memory_context,
            tool_results=request.tool_results,
            workflow_results=request.workflow_results,
            metadata=request.metadata,
        )
        context_result = self._agent_context_builder.build(context_request)
        if context_result.status is not AgentContextStatus.BUILT or context_result.context is None:
            events.append(_event("agent_execution_failed", AgentExecutionStatus.CONTEXT_BUILD_FAILED, request=request, agent=agent))
            return _result(
                AgentExecutionStatus.CONTEXT_BUILD_FAILED,
                signature,
                execution_id=request.execution_id,
                correlation_id=request.correlation_id,
                agent_id=agent.agent_id,
                resolution_result=resolution_result,
                events=events,
                error_code=context_result.error_code,
                safe_message=context_result.safe_message,
            )
        context = context_result.context
        events.append(_event("agent_context_built", AgentExecutionStatus.COMPLETED, request=request, agent=agent))

        try:
            handler = self._agent_handler_registry.get(agent.agent_id)
        except AgentHandlerNotFoundError:
            events.append(_event("agent_execution_failed", AgentExecutionStatus.HANDLER_UNAVAILABLE, request=request, agent=agent))
            return _result(
                AgentExecutionStatus.HANDLER_UNAVAILABLE,
                signature,
                execution_id=request.execution_id,
                correlation_id=request.correlation_id,
                agent_id=agent.agent_id,
                resolution_result=resolution_result,
                context=context,
                events=events,
                error_code="HANDLER_UNAVAILABLE",
                safe_message="no handler is registered for selected agent.",
            )
        if _handler_agent_id(handler) != agent.agent_id:
            events.append(_event("agent_execution_failed", AgentExecutionStatus.HANDLER_UNAVAILABLE, request=request, agent=agent))
            return _result(
                AgentExecutionStatus.HANDLER_UNAVAILABLE,
                signature,
                execution_id=request.execution_id,
                correlation_id=request.correlation_id,
                agent_id=agent.agent_id,
                resolution_result=resolution_result,
                context=context,
                events=events,
                error_code="HANDLER_UNAVAILABLE",
                safe_message="handler id is incompatible with selected agent.",
            )

        events.append(_event("agent_handler_started", AgentExecutionStatus.COMPLETED, request=request, agent=agent))
        try:
            raw_output = handler.handle(context)
            output, sanitized_count = _sanitize_output(raw_output)
        except Exception as error:
            events.append(_event("agent_handler_failed", AgentExecutionStatus.EXECUTION_FAILED, request=request, agent=agent))
            events.append(_event("agent_execution_failed", AgentExecutionStatus.EXECUTION_FAILED, request=request, agent=agent))
            return _result(
                AgentExecutionStatus.EXECUTION_FAILED,
                signature,
                execution_id=request.execution_id,
                correlation_id=request.correlation_id,
                agent_id=agent.agent_id,
                resolution_result=resolution_result,
                context=context,
                events=events,
                error_code=type(error).__name__,
                safe_message=str(error),
            )

        events.append(_event("agent_handler_succeeded", AgentExecutionStatus.COMPLETED, request=request, agent=agent))
        events.append(_event("agent_execution_completed", AgentExecutionStatus.COMPLETED, request=request, agent=agent))
        return _result(
            AgentExecutionStatus.COMPLETED,
            signature,
            execution_id=request.execution_id,
            correlation_id=request.correlation_id,
            agent_id=agent.agent_id,
            resolution_result=resolution_result,
            context=context,
            output=output,
            metadata={"sanitized_output_fields": sanitized_count},
            events=events,
        )


def agent_execution_request_signature(
    request: AgentExecutionRequest,
) -> str:
    """Return a stable SHA-256 signature for an execution request."""

    if not isinstance(request, AgentExecutionRequest):
        raise InvalidAgentExecutionRequestError("request must be AgentExecutionRequest.")
    payload = {
        "resolution_request_signature": agent_resolution_request_signature(request.resolution_request),
        "task_id": request.task_id,
        "execution_id": request.execution_id,
        "correlation_id": request.correlation_id,
        "session_id": request.session_id,
        "user_input": request.user_input,
        "structured_input": _jsonable(request.structured_input),
        "shared_context": _jsonable(request.shared_context),
        "conversation_context": _jsonable(request.conversation_context),
        "memory_context": _jsonable(request.memory_context),
        "tool_results": _jsonable(request.tool_results),
        "workflow_results": _jsonable(request.workflow_results),
        "metadata": _jsonable(request.metadata),
        "required_capability_ids": sorted(request.required_capability_ids),
        "required_permission_ids": sorted(request.required_permission_ids),
        "cancel_requested": request.cancel_requested,
    }
    return _signature(payload)


def _map_resolution_status(
    status: AgentResolutionStatus,
) -> AgentExecutionStatus:
    if status is AgentResolutionStatus.RESOLVED:
        return AgentExecutionStatus.COMPLETED
    if status in (
        AgentResolutionStatus.NO_AGENTS,
        AgentResolutionStatus.NO_MATCHING_AGENTS,
        AgentResolutionStatus.BELOW_MINIMUM_SCORE,
    ):
        return AgentExecutionStatus.NO_AGENT_CANDIDATES
    if status is AgentResolutionStatus.AMBIGUOUS:
        return AgentExecutionStatus.AGENT_AMBIGUOUS
    if status is AgentResolutionStatus.INVALID_REQUEST:
        return AgentExecutionStatus.INVALID_REQUEST
    if status is AgentResolutionStatus.REGISTRY_UNAVAILABLE:
        return AgentExecutionStatus.INTERNAL_ERROR
    return AgentExecutionStatus.INTERNAL_ERROR


def _handler_agent_id(
    handler: AgentHandler,
) -> str:
    if not isinstance(handler, AgentHandler):
        raise InvalidAgentHandlerError("handler must implement AgentHandler.")
    try:
        raw_agent_id = handler.agent_id
    except (AttributeError, TypeError, ValueError) as error:
        raise InvalidAgentHandlerError("handler must expose agent_id.") from error
    if not callable(getattr(handler, "handle", None)):
        raise InvalidAgentHandlerError("handler must expose handle(context).")
    return validate_agent_id(raw_agent_id)


def _sanitize_output(
    output: Mapping[str, object],
) -> tuple[Mapping[str, object], int]:
    if not isinstance(output, Mapping):
        raise AgentExecutionError("handler result must be a mapping.")
    counter = {"nodes": 0, "chars": 0}
    sanitized, count = _sanitize_mapping(output, depth=0, counter=counter)
    return MappingProxyType(sanitized), count


def _sanitize_metadata(
    metadata: Mapping[str, object],
) -> dict[str, object]:
    if not isinstance(metadata, Mapping):
        raise InvalidAgentExecutionRequestError("metadata must be a mapping.")
    if len(metadata) > MAX_AGENT_EXECUTION_METADATA_ITEMS:
        raise InvalidAgentExecutionRequestError("metadata has too many items.")
    counter = {"nodes": 0, "chars": 0}
    sanitized, _ = _sanitize_mapping(metadata, depth=0, counter=counter)
    return sanitized


def _sanitize_mapping(
    value: Mapping[str, object],
    *,
    depth: int,
    counter: dict[str, int],
) -> tuple[dict[str, object], int]:
    _check_depth(depth)
    if len(value) > MAX_AGENT_EXECUTION_RESULT_MAPPING_ITEMS:
        raise AgentExecutionError("mapping item limit exceeded.")
    result: dict[str, object] = {}
    sanitized_count = 0
    for raw_key in sorted(value):
        key = _safe_key(raw_key)
        if _is_sensitive_key(key):
            sanitized_count += 1
            continue
        sanitized_value, child_count = _sanitize_value(value[raw_key], depth=depth + 1, counter=counter)
        result[key] = sanitized_value
        sanitized_count += child_count
    return result, sanitized_count


def _sanitize_sequence(
    value: Sequence[object],
    *,
    depth: int,
    counter: dict[str, int],
) -> tuple[tuple[object, ...], int]:
    _check_depth(depth)
    if len(value) > MAX_AGENT_EXECUTION_RESULT_SEQUENCE_ITEMS:
        raise AgentExecutionError("sequence item limit exceeded.")
    items: list[object] = []
    sanitized_count = 0
    for item in value:
        sanitized, count = _sanitize_value(item, depth=depth + 1, counter=counter)
        items.append(sanitized)
        sanitized_count += count
    return tuple(items), sanitized_count


def _sanitize_value(
    value: object,
    *,
    depth: int,
    counter: dict[str, int],
) -> tuple[object, int]:
    counter["nodes"] += 1
    if counter["nodes"] > MAX_AGENT_EXECUTION_RESULT_TOTAL_ITEMS:
        raise AgentExecutionError("result total item limit exceeded.")
    if value is None or isinstance(value, (bool, int)):
        return value, 0
    if isinstance(value, float):
        if not math.isfinite(value):
            raise AgentExecutionError("non-finite floats are not allowed.")
        return value, 0
    if isinstance(value, str):
        if len(value) > MAX_AGENT_EXECUTION_RESULT_STRING_LENGTH:
            raise AgentExecutionError("string length limit exceeded.")
        counter["chars"] += len(value)
        if counter["chars"] > MAX_AGENT_EXECUTION_RESULT_STRING_LENGTH * MAX_AGENT_EXECUTION_RESULT_MAPPING_ITEMS:
            raise AgentExecutionError("string budget exceeded.")
        return value, 0
    if isinstance(value, Mapping):
        sanitized, count = _sanitize_mapping(value, depth=depth, counter=counter)
        return MappingProxyType(sanitized), count
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        sanitized, count = _sanitize_sequence(value, depth=depth, counter=counter)
        return sanitized, count
    if inspect.isfunction(value) or inspect.ismethod(value) or inspect.isclass(value) or isinstance(value, types.ModuleType):
        raise AgentExecutionError("executable objects are not allowed.")
    raise AgentExecutionError("unsupported result value type.")


def _identifier_tuple(
    values: Iterable[str],
    field_name: str,
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Iterable):
        raise InvalidAgentExecutionRequestError(f"{field_name} must be an iterable of strings.")
    normalized = tuple(dict.fromkeys(_safe_identifier(value, field_name) for value in values))
    if len(normalized) > MAX_AGENT_EXECUTION_IDS:
        raise InvalidAgentExecutionRequestError(f"{field_name} exceeds the item limit.")
    return normalized


def _permission_tuple(
    values: Iterable[str],
    field_name: str,
) -> tuple[str, ...]:
    normalized = _identifier_tuple(values, field_name)
    unknown = tuple(value for value in normalized if value not in _PERMISSION_IDS)
    if unknown:
        raise InvalidAgentExecutionRequestError(f"{field_name} contains an unknown permission id.")
    return normalized


def _safe_identifier(
    value: str,
    field_name: str,
) -> str:
    if not isinstance(value, str) or not value.strip() or value.strip() != value:
        raise InvalidAgentExecutionRequestError(f"{field_name} must be a non-empty identifier.")
    if "/" in value or "\\" in value or ".." in value or any(ord(character) < 32 for character in value):
        raise InvalidAgentExecutionRequestError(f"{field_name} is unsafe.")
    if not all(character.isalnum() or character in "_.-" for character in value):
        raise InvalidAgentExecutionRequestError(f"{field_name} contains unsupported characters.")
    if len(value) > 128:
        raise InvalidAgentExecutionRequestError(f"{field_name} exceeds the length limit.")
    return value


def _safe_key(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AgentExecutionError("mapping keys must be non-empty strings.")
    key = value.strip()
    if key != value or any(ord(character) < 32 for character in key):
        raise AgentExecutionError("mapping keys are unsafe.")
    return key


def _is_sensitive_key(key: str) -> bool:
    normalized = key.replace("-", "_").lower()
    return any(part in normalized for part in _SENSITIVE_KEY_PARTS)


def _check_depth(depth: int) -> None:
    if depth > MAX_AGENT_EXECUTION_RESULT_DEPTH:
        raise AgentExecutionError("result depth limit exceeded.")


def _jsonable(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(value[key]) for key in sorted(value)}
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, float) and not math.isfinite(value):
        raise InvalidAgentExecutionRequestError("non-finite floats are not allowed.")
    if inspect.isfunction(value) or inspect.ismethod(value) or inspect.isclass(value) or isinstance(value, types.ModuleType):
        raise InvalidAgentExecutionRequestError("executable objects are not allowed.")
    if value is not None and not isinstance(value, (str, int, float, bool)):
        raise InvalidAgentExecutionRequestError("unsupported request value type.")
    return value


def _signature(value: object) -> str:
    encoded = json.dumps(_jsonable(value), ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _event(
    name: str,
    status: AgentExecutionStatus,
    *,
    request: AgentExecutionRequest,
    agent: AgentDefinition | None = None,
) -> AgentExecutionEvent:
    details: dict[str, object] = {}
    if request.execution_id is not None:
        details["execution_id"] = request.execution_id
    if request.correlation_id is not None:
        details["correlation_id"] = request.correlation_id
    if agent is not None:
        details["agent_id"] = agent.agent_id
    return AgentExecutionEvent(name=name, status=status, details=details)


def _result(
    status: AgentExecutionStatus,
    request_signature: str,
    *,
    execution_id: str | None = None,
    correlation_id: str | None = None,
    agent_id: str | None = None,
    resolution_result: AgentResolutionResult | None = None,
    context: AgentContext | None = None,
    output: Mapping[str, object] | None = None,
    metadata: Mapping[str, object] | None = None,
    events: Iterable[AgentExecutionEvent] = (),
    error_code: str | None = None,
    safe_message: str | None = None,
) -> AgentExecutionResult:
    return AgentExecutionResult(
        status=status,
        request_signature=request_signature,
        execution_id=execution_id,
        correlation_id=correlation_id,
        agent_id=agent_id,
        resolution_result=resolution_result,
        context=context,
        output=output,
        metadata=metadata or {},
        events=tuple(events),
        error_code=error_code,
        safe_message=_safe_message(safe_message) if safe_message is not None else None,
    )


def _safe_message(value: str) -> str:
    text = " ".join(str(value).split())
    for part in _SENSITIVE_KEY_PARTS:
        text = re.sub(re.escape(part), "[redacted]", text, flags=re.IGNORECASE)
    return text[:240]


def _execution_status(value: AgentExecutionStatus | str) -> AgentExecutionStatus:
    if isinstance(value, AgentExecutionStatus):
        return value
    if isinstance(value, str):
        return AgentExecutionStatus(value)
    raise InvalidAgentExecutionRequestError("status must be AgentExecutionStatus.")
