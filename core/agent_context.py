"""Safe deterministic context construction for specialized agents."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import inspect
import json
import math
import types
from types import MappingProxyType
from typing import Any

from core.agent_registry import AgentDefinition


SENSITIVE_KEY_PARTS = (
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
MAX_AGENT_CONTEXT_METADATA_ITEMS = 16


class AgentContextError(RuntimeError):
    """Base error for agent-context construction."""


class InvalidAgentContextRequestError(AgentContextError):
    """Raised when a context request is structurally invalid."""


class AgentContextPolicyViolationError(AgentContextError):
    """Raised when requested context violates agent policy."""


class AgentContextLimitError(AgentContextError):
    """Raised when context exceeds explicit limits."""


class AgentContextStatus(str, Enum):
    """Structured statuses for context construction."""

    BUILT = "BUILT"
    INVALID_REQUEST = "INVALID_REQUEST"
    POLICY_VIOLATION = "POLICY_VIOLATION"
    LIMIT_EXCEEDED = "LIMIT_EXCEEDED"
    INTERNAL_ERROR = "INTERNAL_ERROR"


@dataclass(frozen=True, slots=True)
class AgentContextRequest:
    """Structured input for building a safe agent context."""

    agent: AgentDefinition
    task_id: str | None = None
    execution_id: str | None = None
    session_id: str | None = None
    user_input: str | None = None
    structured_input: Mapping[str, object] | None = None
    shared_context: Mapping[str, object] | None = None
    conversation_context: Sequence[object] | None = None
    memory_context: Mapping[str, object] | None = None
    tool_results: Mapping[str, object] | None = None
    workflow_results: Mapping[str, object] | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.agent, AgentDefinition):
            raise InvalidAgentContextRequestError("agent must be AgentDefinition.")
        for field_name in ("task_id", "execution_id", "session_id"):
            value = getattr(self, field_name)
            if value is not None:
                object.__setattr__(self, field_name, _safe_identifier(value, field_name))
        if self.user_input is not None and not isinstance(self.user_input, str):
            raise InvalidAgentContextRequestError("user_input must be a string or None.")
        object.__setattr__(self, "metadata", MappingProxyType(_sanitize_metadata(self.metadata)))


@dataclass(frozen=True, slots=True)
class _MetadataPolicy:
    max_context_depth: int = 3
    max_mapping_items: int = MAX_AGENT_CONTEXT_METADATA_ITEMS
    max_sequence_items: int = MAX_AGENT_CONTEXT_METADATA_ITEMS
    max_string_length: int = 1_000
    max_total_items: int = 64
    max_context_items: int = 16
    allowed_context_keys: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class AgentContext:
    """Immutable filtered context for one selected agent."""

    agent_id: str
    task_id: str | None
    execution_id: str | None
    session_id: str | None
    structured_input: Mapping[str, object]
    user_input: str | None
    shared_context: Mapping[str, object]
    conversation_context: tuple[object, ...]
    memory_context: Mapping[str, object]
    tool_results: Mapping[str, object]
    workflow_results: Mapping[str, object]
    metadata: Mapping[str, object]
    omitted_sections: tuple[str, ...]
    context_signature: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "agent_id", _safe_identifier(self.agent_id, "agent_id"))
        for name in ("structured_input", "shared_context", "memory_context", "tool_results", "workflow_results", "metadata"):
            value = getattr(self, name)
            if not isinstance(value, Mapping):
                raise InvalidAgentContextRequestError(f"{name} must be a mapping.")
            object.__setattr__(self, name, MappingProxyType(dict(value)))
        object.__setattr__(self, "conversation_context", tuple(self.conversation_context))
        object.__setattr__(self, "omitted_sections", tuple(self.omitted_sections))


@dataclass(frozen=True, slots=True)
class AgentContextResult:
    """Structured result for safe context construction."""

    status: AgentContextStatus
    context: AgentContext | None = None
    error_code: str | None = None
    safe_message: str | None = None
    omitted_sections: tuple[str, ...] = ()
    sanitized_fields_count: int = 0
    request_signature: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", _status(self.status))
        object.__setattr__(self, "omitted_sections", tuple(self.omitted_sections))
        if isinstance(self.sanitized_fields_count, bool) or not isinstance(self.sanitized_fields_count, int):
            raise InvalidAgentContextRequestError("sanitized_fields_count must be an integer.")


class AgentContextBuilder:
    """Build a safe context using only the selected agent definition and request."""

    def build(
        self,
        request: AgentContextRequest,
    ) -> AgentContextResult:
        """Return a filtered context result without executing anything."""

        try:
            if not isinstance(request, AgentContextRequest):
                raise InvalidAgentContextRequestError("request must be AgentContextRequest.")
            request_signature = agent_context_request_signature(request)
            omitted: list[str] = []
            sanitized_count = 0
            policy = request.agent.context_policy
            memory_policy = request.agent.memory_policy
            security_policy = request.agent.security_policy
            counter = {"nodes": 0, "chars": 0}

            user_input = None
            if request.user_input is not None:
                if not policy.allow_user_input:
                    omitted.append("user_input:policy_denied")
                else:
                    user_input = _sanitize_string(request.user_input, policy, counter)

            structured_input, count = _section_mapping(
                request.structured_input,
                "structured_input",
                request.agent,
                counter,
            )
            sanitized_count += count

            shared_context, count = _allowed_mapping_section(
                request.shared_context,
                "shared_context",
                allowed=policy.allow_shared_context,
                agent=request.agent,
                omitted=omitted,
                counter=counter,
            )
            sanitized_count += count

            conversation_context: tuple[object, ...] = ()
            if request.conversation_context is not None:
                if not policy.include_conversation_context:
                    omitted.append("conversation_context:policy_denied")
                else:
                    conversation_context, count = _section_sequence(
                        request.conversation_context,
                        "conversation_context",
                        request.agent,
                        counter,
                    )
                    sanitized_count += count

            memory_context: Mapping[str, object] = MappingProxyType({})
            if request.memory_context is not None:
                if not memory_policy.can_read_memory:
                    omitted.append("memory_context:policy_denied")
                else:
                    memory_context, count = _section_mapping(
                        request.memory_context,
                        "memory_context",
                        request.agent,
                        counter,
                    )
                    sanitized_count += count

            tool_results, count = _allowed_mapping_section(
                request.tool_results,
                "tool_results",
                allowed=policy.allow_tool_results and request.agent.permissions.can_execute_tools,
                agent=request.agent,
                omitted=omitted,
                counter=counter,
            )
            sanitized_count += count
            if request.tool_results is not None and security_policy.allowed_tools:
                tool_results = MappingProxyType(
                    {
                        key: value
                        for key, value in tool_results.items()
                        if key in security_policy.allowed_tools
                    }
                )
                omitted.append("tool_results:unauthorized_tools_omitted")

            workflow_results, count = _allowed_mapping_section(
                request.workflow_results,
                "workflow_results",
                allowed=policy.allow_workflow_results,
                agent=request.agent,
                omitted=omitted,
                counter=counter,
            )
            sanitized_count += count

            metadata, count = _section_mapping(request.metadata, "metadata", request.agent, counter)
            sanitized_count += count
            if counter["nodes"] > policy.max_total_items:
                raise AgentContextLimitError("context total item limit exceeded.")

            raw_context = {
                "agent_id": request.agent.agent_id,
                "task_id": request.task_id,
                "execution_id": request.execution_id,
                "session_id": request.session_id,
                "structured_input": structured_input,
                "user_input": user_input,
                "shared_context": shared_context,
                "conversation_context": conversation_context,
                "memory_context": memory_context,
                "tool_results": tool_results,
                "workflow_results": workflow_results,
                "metadata": metadata,
                "omitted_sections": tuple(omitted),
            }
            signature = _signature(raw_context)
            context = AgentContext(
                agent_id=request.agent.agent_id,
                task_id=request.task_id,
                execution_id=request.execution_id,
                session_id=request.session_id,
                structured_input=structured_input,
                user_input=user_input,
                shared_context=shared_context,
                conversation_context=conversation_context,
                memory_context=memory_context,
                tool_results=tool_results,
                workflow_results=workflow_results,
                metadata=metadata,
                omitted_sections=tuple(omitted),
                context_signature=signature,
            )
            return AgentContextResult(
                status=AgentContextStatus.BUILT,
                context=context,
                omitted_sections=tuple(omitted),
                sanitized_fields_count=sanitized_count,
                request_signature=request_signature,
            )
        except InvalidAgentContextRequestError as error:
            return _error(AgentContextStatus.INVALID_REQUEST, "INVALID_REQUEST", error)
        except AgentContextPolicyViolationError as error:
            return _error(AgentContextStatus.POLICY_VIOLATION, "POLICY_VIOLATION", error)
        except AgentContextLimitError as error:
            return _error(AgentContextStatus.LIMIT_EXCEEDED, "LIMIT_EXCEEDED", error)
        except (TypeError, ValueError, RuntimeError) as error:
            return _error(AgentContextStatus.INTERNAL_ERROR, type(error).__name__, error)


def agent_context_request_signature(
    request: AgentContextRequest,
) -> str:
    """Return a deterministic request signature over safe canonical data."""

    if not isinstance(request, AgentContextRequest):
        raise InvalidAgentContextRequestError("request must be AgentContextRequest.")
    payload = {
        "agent_id": request.agent.agent_id,
        "task_id": request.task_id,
        "execution_id": request.execution_id,
        "session_id": request.session_id,
        "user_input": request.user_input,
        "structured_input": _jsonable(request.structured_input),
        "shared_context": _jsonable(request.shared_context),
        "conversation_context": _jsonable(request.conversation_context),
        "memory_context": _jsonable(request.memory_context),
        "tool_results": _jsonable(request.tool_results),
        "workflow_results": _jsonable(request.workflow_results),
        "metadata": _jsonable(request.metadata),
        "context_policy": _jsonable(_policy_payload(request.agent.context_policy)),
        "memory_policy": _jsonable(_policy_payload(request.agent.memory_policy)),
        "security_policy": _jsonable(_policy_payload(request.agent.security_policy)),
    }
    return _signature(payload)


def _allowed_mapping_section(
    value: Mapping[str, object] | None,
    section: str,
    *,
    allowed: bool,
    agent: AgentDefinition,
    omitted: list[str],
    counter: dict[str, int],
) -> tuple[Mapping[str, object], int]:
    if value is None:
        return MappingProxyType({}), 0
    if not allowed:
        omitted.append(f"{section}:policy_denied")
        return MappingProxyType({}), 0
    return _section_mapping(value, section, agent, counter)


def _section_mapping(
    value: Mapping[str, object] | None,
    section: str,
    agent: AgentDefinition,
    counter: dict[str, int],
) -> tuple[Mapping[str, object], int]:
    if value is None:
        return MappingProxyType({}), 0
    if not isinstance(value, Mapping):
        raise InvalidAgentContextRequestError(f"{section} must be a mapping.")
    owner = value.get("agent_id")
    if owner is not None and owner != agent.agent_id:
        return MappingProxyType({}), 0
    sanitized, count = _sanitize_mapping(value, agent.context_policy, counter, depth=0)
    return MappingProxyType(sanitized), count


def _section_sequence(
    value: Sequence[object],
    section: str,
    agent: AgentDefinition,
    counter: dict[str, int],
) -> tuple[tuple[object, ...], int]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise InvalidAgentContextRequestError(f"{section} must be a sequence.")
    sanitized, count = _sanitize_sequence(value, agent.context_policy, counter, depth=0)
    return sanitized, count


def _sanitize_mapping(
    value: Mapping[str, object],
    policy,
    counter: dict[str, int],
    *,
    depth: int,
) -> tuple[dict[str, object], int]:
    _check_depth(depth, policy)
    if len(value) > policy.max_mapping_items:
        raise AgentContextLimitError("mapping item limit exceeded.")
    result: dict[str, object] = {}
    sanitized_count = 0
    allowed = set(policy.allowed_context_keys)
    for raw_key in sorted(value):
        key = _safe_key(raw_key)
        if allowed and key not in allowed and key != "agent_id":
            continue
        if _is_sensitive_key(key):
            sanitized_count += 1
            continue
        sanitized_value, child_count = _sanitize_value(value[raw_key], policy, counter, depth=depth + 1)
        result[key] = sanitized_value
        sanitized_count += child_count
    return result, sanitized_count


def _sanitize_sequence(
    value: Sequence[object],
    policy,
    counter: dict[str, int],
    *,
    depth: int,
) -> tuple[tuple[object, ...], int]:
    _check_depth(depth, policy)
    if len(value) > policy.max_sequence_items:
        raise AgentContextLimitError("sequence item limit exceeded.")
    items: list[object] = []
    sanitized_count = 0
    for item in value:
        sanitized, count = _sanitize_value(item, policy, counter, depth=depth + 1)
        items.append(sanitized)
        sanitized_count += count
    return tuple(items), sanitized_count


def _sanitize_value(
    value: object,
    policy,
    counter: dict[str, int],
    *,
    depth: int,
) -> tuple[object, int]:
    counter["nodes"] += 1
    if counter["nodes"] > policy.max_total_items:
        raise AgentContextLimitError("context total item limit exceeded.")
    if value is None or isinstance(value, (bool, int)):
        return value, 0
    if isinstance(value, float):
        if not math.isfinite(value):
            raise InvalidAgentContextRequestError("non-finite floats are not allowed.")
        return value, 0
    if isinstance(value, str):
        return _sanitize_string(value, policy, counter), 0
    if isinstance(value, Mapping):
        sanitized, count = _sanitize_mapping(value, policy, counter, depth=depth)
        return MappingProxyType(sanitized), count
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        sanitized, count = _sanitize_sequence(value, policy, counter, depth=depth)
        return sanitized, count
    if inspect.isfunction(value) or inspect.ismethod(value) or inspect.isclass(value) or isinstance(value, types.ModuleType):
        raise InvalidAgentContextRequestError("executable objects are not allowed.")
    raise InvalidAgentContextRequestError("unsupported context value type.")


def _sanitize_string(
    value: str,
    policy,
    counter: dict[str, int],
) -> str:
    if len(value) > policy.max_string_length:
        raise AgentContextLimitError("string length limit exceeded.")
    counter["chars"] += len(value)
    if counter["chars"] > policy.max_string_length * max(1, policy.max_context_items):
        raise AgentContextLimitError("context string budget exceeded.")
    return value


def _sanitize_metadata(metadata: Mapping[str, object]) -> dict[str, object]:
    if not isinstance(metadata, Mapping):
        raise InvalidAgentContextRequestError("metadata must be a mapping.")
    if len(metadata) > MAX_AGENT_CONTEXT_METADATA_ITEMS:
        raise InvalidAgentContextRequestError("metadata has too many items.")
    safe: dict[str, object] = {}
    counter = {"nodes": 0, "chars": 0}
    policy = _MetadataPolicy()
    sanitized, _ = _sanitize_mapping(metadata, policy, counter, depth=0)
    safe.update(sanitized)
    return safe


def _check_depth(depth: int, policy) -> None:
    if depth > policy.max_context_depth:
        raise AgentContextLimitError("context depth limit exceeded.")


def _safe_identifier(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip() or value.strip() != value:
        raise InvalidAgentContextRequestError(f"{field_name} must be a non-empty identifier.")
    if "/" in value or "\\" in value or ".." in value or any(ord(character) < 32 for character in value):
        raise InvalidAgentContextRequestError(f"{field_name} is unsafe.")
    return value


def _safe_key(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise InvalidAgentContextRequestError("context keys must be non-empty strings.")
    key = value.strip()
    if key != value or any(ord(character) < 32 for character in key):
        raise InvalidAgentContextRequestError("context keys are unsafe.")
    return key


def _is_sensitive_key(key: str) -> bool:
    normalized = key.replace("-", "_").lower()
    return any(part in normalized for part in SENSITIVE_KEY_PARTS)


def _policy_payload(policy: object) -> dict[str, object]:
    fields = getattr(policy, "__dataclass_fields__", {})
    return {name: getattr(policy, name) for name in fields}


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
        raise InvalidAgentContextRequestError("non-finite floats are not allowed.")
    if inspect.isfunction(value) or inspect.ismethod(value) or inspect.isclass(value) or isinstance(value, types.ModuleType):
        raise InvalidAgentContextRequestError("executable objects are not allowed.")
    if value is not None and not isinstance(value, (str, int, float, bool)):
        raise InvalidAgentContextRequestError("unsupported context value type.")
    return value


def _signature(value: object) -> str:
    encoded = json.dumps(_jsonable(value), ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _error(status: AgentContextStatus, code: str, error: Exception) -> AgentContextResult:
    return AgentContextResult(
        status=status,
        error_code=code,
        safe_message=_safe_message(str(error)),
        request_signature="",
    )


def _safe_message(value: str) -> str:
    text = " ".join(str(value).split())
    for part in SENSITIVE_KEY_PARTS:
        text = text.replace(part, "[redacted]")
    return text[:240]


def _status(value: AgentContextStatus | str) -> AgentContextStatus:
    if isinstance(value, AgentContextStatus):
        return value
    if isinstance(value, str):
        return AgentContextStatus(value)
    raise InvalidAgentContextRequestError("status must be AgentContextStatus.")
