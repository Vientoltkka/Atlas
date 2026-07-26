"""Controlled execution for resolved Atlas skills."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
import math
from types import MappingProxyType

from core.agent_executor import AgentExecutionRequest, AgentExecutor
from core.agent_registry import AgentDefinition
from core.agent_resolver import AgentResolutionRequest
from core.capability_execution_service import CapabilityExecutionRequest, CapabilityExecutionService
from core.skill_registry import SkillDefinition, SkillExecutionTargetType, validate_skill_id
from tools.executor import ToolExecutor


class SkillExecutionStatus(str, Enum):
    """Terminal skill execution status."""

    COMPLETED = "COMPLETED"
    INVALID_REQUEST = "INVALID_REQUEST"
    SKILL_DISABLED = "SKILL_DISABLED"
    SKILL_NOT_AUTHORIZED = "SKILL_NOT_AUTHORIZED"
    TARGET_UNAVAILABLE = "TARGET_UNAVAILABLE"
    TARGET_TYPE_INVALID = "TARGET_TYPE_INVALID"
    EXECUTION_FAILED = "EXECUTION_FAILED"
    BLOCKED = "BLOCKED"


class SkillExecutionError(RuntimeError):
    """Base skill execution error."""


class InvalidSkillExecutionRequestError(SkillExecutionError):
    """Raised for malformed execution requests."""


@dataclass(frozen=True, slots=True)
class SkillExecutionPolicy:
    """Bounded skill execution policy."""

    timeout_seconds: int = 30
    allow_disabled: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.timeout_seconds, bool) or not isinstance(self.timeout_seconds, int) or self.timeout_seconds <= 0:
            raise InvalidSkillExecutionRequestError("timeout_seconds must be positive.")
        if not isinstance(self.allow_disabled, bool):
            raise InvalidSkillExecutionRequestError("allow_disabled must be a bool.")


@dataclass(frozen=True, slots=True)
class SkillExecutionRequest:
    """Request to execute one already resolved skill."""

    skill: SkillDefinition
    inputs: Mapping[str, object] = field(default_factory=dict)
    agent: AgentDefinition | None = None
    policy: SkillExecutionPolicy = field(default_factory=SkillExecutionPolicy)
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.skill, SkillDefinition):
            raise InvalidSkillExecutionRequestError("skill must be SkillDefinition.")
        object.__setattr__(self, "inputs", MappingProxyType(_safe_mapping(self.inputs)))
        if self.agent is not None and not isinstance(self.agent, AgentDefinition):
            raise InvalidSkillExecutionRequestError("agent must be AgentDefinition or None.")
        if not isinstance(self.policy, SkillExecutionPolicy):
            raise InvalidSkillExecutionRequestError("policy must be SkillExecutionPolicy.")
        object.__setattr__(self, "metadata", MappingProxyType(_safe_mapping(self.metadata)))


@dataclass(frozen=True, slots=True)
class SkillExecutionEvent:
    """Safe skill execution event."""

    name: str
    status: str
    details: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SkillExecutionResult:
    """Structured skill execution result."""

    status: SkillExecutionStatus
    request_signature: str
    skill_id: str | None = None
    output: Mapping[str, object] | None = None
    events: tuple[SkillExecutionEvent, ...] = ()
    metrics: Mapping[str, int] = field(default_factory=dict)
    error_code: str | None = None
    safe_message: str | None = None

    @property
    def completed(self) -> bool:
        return self.status is SkillExecutionStatus.COMPLETED


class SkillHandlerRegistry:
    """Minimal explicit registry for safe skill handlers."""

    def __init__(self) -> None:
        self._handlers: dict[str, Callable[[Mapping[str, object]], Mapping[str, object]]] = {}

    def register(self, handler_id: str, handler: Callable[[Mapping[str, object]], Mapping[str, object]], *, replace: bool = False) -> None:
        normalized = validate_skill_id(handler_id)
        if not callable(handler):
            raise InvalidSkillExecutionRequestError("handler must be callable.")
        if normalized in self._handlers and not replace:
            raise InvalidSkillExecutionRequestError("handler already registered.")
        self._handlers[normalized] = handler

    def get(self, handler_id: str) -> Callable[[Mapping[str, object]], Mapping[str, object]]:
        normalized = validate_skill_id(handler_id)
        try:
            return self._handlers[normalized]
        except KeyError as error:
            raise InvalidSkillExecutionRequestError("handler is not registered.") from error


class SkillExecutor:
    """Execute one skill by delegating to existing Atlas components."""

    def __init__(
        self,
        *,
        tool_executor: ToolExecutor | None = None,
        capability_execution_service: CapabilityExecutionService | None = None,
        agent_executor: AgentExecutor | None = None,
        handler_registry: SkillHandlerRegistry | None = None,
    ) -> None:
        self._tool_executor = tool_executor
        self._capability_execution_service = capability_execution_service
        self._agent_executor = agent_executor
        self._handler_registry = handler_registry or SkillHandlerRegistry()

    def execute(self, request: SkillExecutionRequest) -> SkillExecutionResult:
        if not isinstance(request, SkillExecutionRequest):
            return _result(SkillExecutionStatus.INVALID_REQUEST, "", error_code="INVALID_REQUEST")
        signature = skill_execution_request_signature(request)
        events = [SkillExecutionEvent("skill_execution_started", "started", {"skill_id": request.skill.skill_id})]
        if not request.skill.enabled and not request.policy.allow_disabled:
            events.append(SkillExecutionEvent("skill_execution_blocked", "blocked", {"reason": "disabled"}))
            return _result(SkillExecutionStatus.SKILL_DISABLED, signature, request.skill.skill_id, events=events, error_code="SKILL_DISABLED")
        if request.agent is not None and not _agent_authorized(request.agent, request.skill):
            events.append(SkillExecutionEvent("skill_execution_blocked", "blocked", {"reason": "unauthorized"}))
            return _result(SkillExecutionStatus.SKILL_NOT_AUTHORIZED, signature, request.skill.skill_id, events=events, error_code="SKILL_NOT_AUTHORIZED")
        try:
            output = self._execute_target(request)
        except (RuntimeError, ValueError, TypeError, OSError) as error:
            events.append(SkillExecutionEvent("skill_execution_failed", "failed", {"skill_id": request.skill.skill_id}))
            return _result(
                SkillExecutionStatus.EXECUTION_FAILED,
                signature,
                request.skill.skill_id,
                events=events,
                error_code=type(error).__name__,
                safe_message=str(error),
            )
        events.append(SkillExecutionEvent("skill_execution_succeeded", "finished", {"skill_id": request.skill.skill_id}))
        return _result(SkillExecutionStatus.COMPLETED, signature, request.skill.skill_id, output=_safe_output(output), events=events)

    def _execute_target(self, request: SkillExecutionRequest) -> Mapping[str, object]:
        skill = request.skill
        if skill.execution_target_type is SkillExecutionTargetType.TOOL:
            if self._tool_executor is None:
                raise RuntimeError("tool executor is unavailable")
            return {"result": self._tool_executor.execute(skill.execution_target, arguments=request.inputs)}
        if skill.execution_target_type is SkillExecutionTargetType.CAPABILITY:
            if self._capability_execution_service is None:
                raise RuntimeError("capability execution service is unavailable")
            result = self._capability_execution_service.execute(
                CapabilityExecutionRequest(capability_id=skill.execution_target, inputs=request.inputs)
            )
            return {"status": result.status.value, "output": result.output or {}}
        if skill.execution_target_type is SkillExecutionTargetType.AGENT:
            if self._agent_executor is None:
                raise RuntimeError("agent executor is unavailable")
            result = self._agent_executor.execute(
                AgentExecutionRequest(
                    resolution_request=AgentResolutionRequest(required_agent_ids=(skill.execution_target,), require_unique_top_score=False),
                    structured_input=request.inputs,
                    required_capability_ids=skill.required_capability_ids,
                    required_permission_ids=skill.required_permission_ids,
                )
            )
            return {"status": result.status.value, "output": result.output or {}}
        if skill.execution_target_type is SkillExecutionTargetType.HANDLER:
            handler = self._handler_registry.get(skill.handler_id or skill.execution_target)
            return handler(request.inputs)
        raise RuntimeError("target type is invalid")


def skill_execution_request_signature(request: SkillExecutionRequest) -> str:
    payload = {
        "skill_id": request.skill.skill_id,
        "inputs": _jsonable(request.inputs),
        "agent_id": None if request.agent is None else request.agent.agent_id,
        "timeout_seconds": request.policy.timeout_seconds,
        "allow_disabled": request.policy.allow_disabled,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _agent_authorized(agent: AgentDefinition, skill: SkillDefinition) -> bool:
    denied = _metadata_ids(agent.metadata.get("denied_skill_ids"))
    required = _metadata_ids(agent.metadata.get("required_skill_ids"))
    allowed = _metadata_ids(agent.metadata.get("allowed_skill_ids"))
    if skill.skill_id in denied:
        return False
    if required and skill.skill_id not in required:
        return False
    if allowed and skill.skill_id not in allowed:
        return False
    if skill.allowed_agent_types and agent.agent_type not in skill.allowed_agent_types:
        return False
    return True


def _metadata_ids(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, str):
        return ()
    return tuple(validate_skill_id(part.strip()) for part in value.split(",") if part.strip())


def _result(
    status: SkillExecutionStatus,
    signature: str,
    skill_id: str | None = None,
    *,
    output: Mapping[str, object] | None = None,
    events: list[SkillExecutionEvent] | None = None,
    error_code: str | None = None,
    safe_message: str | None = None,
) -> SkillExecutionResult:
    return SkillExecutionResult(
        status,
        signature,
        skill_id=skill_id,
        output=output,
        events=tuple(events or ()),
        metrics={
            "skill_executions_requested": 1 if signature else 0,
            "skill_executions_succeeded": 1 if status is SkillExecutionStatus.COMPLETED else 0,
            "skill_executions_failed": 1 if status is SkillExecutionStatus.EXECUTION_FAILED else 0,
            "skill_executions_blocked": 1 if status in (SkillExecutionStatus.SKILL_DISABLED, SkillExecutionStatus.SKILL_NOT_AUTHORIZED) else 0,
        },
        error_code=error_code,
        safe_message=_safe_message(safe_message) if safe_message else None,
    )


def _safe_mapping(mapping: Mapping[str, object]) -> dict[str, object]:
    if not isinstance(mapping, Mapping):
        raise InvalidSkillExecutionRequestError("mapping expected.")
    return {str(key): _safe_value(str(key), value) for key, value in sorted(mapping.items(), key=lambda item: str(item[0]))}


def _safe_value(key: str, value: object) -> object:
    if _is_sensitive_key(key):
        raise InvalidSkillExecutionRequestError("sensitive keys are not allowed.")
    if value is None or type(value) in (bool, int, str):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise InvalidSkillExecutionRequestError("floats must be finite.")
        return value
    if isinstance(value, Mapping):
        return MappingProxyType(_safe_mapping(value))
    if isinstance(value, (tuple, list)):
        return tuple(_safe_value(key, item) for item in value)
    raise InvalidSkillExecutionRequestError("unsupported value.")


def _safe_output(output: Mapping[str, object]) -> Mapping[str, object]:
    return MappingProxyType(_safe_mapping(output))


def _jsonable(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(value[key]) for key in sorted(value)}
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    return value


def _safe_message(message: str) -> str:
    text = " ".join(str(message).split())[:300]
    for part in ("token", "secret", "password", "api_key", "authorization", "cookie", "credential"):
        text = text.replace(part, "[redacted]")
    return text


def _is_sensitive_key(key: str) -> bool:
    normalized = key.replace("-", "_").lower()
    return any(part in normalized for part in ("token", "secret", "password", "api_key", "authorization", "cookie", "credential"))
