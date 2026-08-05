from __future__ import annotations

from dataclasses import dataclass, replace
from types import MappingProxyType

import pytest

from bootstrap.agent_executor import build_core_agent_executor
from core.agent_context import AgentContext, AgentContextBuilder
from core.agent_executor import (
    AgentExecutionRequest,
    AgentExecutionStatus,
    AgentExecutor,
    AgentHandlerAlreadyRegisteredError,
    AgentHandlerRegistry,
    AgentHandlerRegistryError,
    agent_execution_request_signature,
)
from core.agent_registry import (
    AgentCapabilities,
    AgentContextPolicy,
    AgentDefinition,
    AgentLimits,
    AgentPermissions,
    AgentRegistry,
    AgentType,
)
from core.agent_resolver import AgentResolutionRequest, AgentResolver
from core.skill_execution_context import SkillExecutionContext


@dataclass(frozen=True)
class EchoHandler:
    agent_id: str = "atlas.agent.echo"

    def handle(self, context: AgentContext):
        return {
            "agent_id": context.agent_id,
            "user_input": context.user_input,
            "structured": context.structured_input,
        }


@dataclass(frozen=True)
class FailingHandler:
    agent_id: str = "atlas.agent.echo"

    def handle(self, context: AgentContext):
        raise RuntimeError("handler failed with token raw-secret")


@dataclass(frozen=True)
class StaticHandler:
    result: object
    agent_id: str = "atlas.agent.echo"

    def handle(self, context: AgentContext):
        return self.result


def _definition(
    agent_id: str = "atlas.agent.echo",
    *,
    capability_ids: tuple[str, ...] = ("agent.echo",),
    enabled: bool = True,
    context_policy: AgentContextPolicy | None = None,
    permissions: AgentPermissions | None = None,
) -> AgentDefinition:
    return AgentDefinition(
        agent_id=agent_id,
        agent_type=AgentType.GENERAL,
        name=f"Agent {agent_id}",
        description="Deterministic test agent.",
        permissions=permissions or AgentPermissions(requires_confirmation=False),
        limits=AgentLimits(max_steps=1, max_tool_calls=0, max_context_items=8),
        capabilities=AgentCapabilities(capabilities=capability_ids),
        context_policy=context_policy or AgentContextPolicy(allow_user_input=True),
        enabled=enabled,
    )


def _executor(
    *definitions: AgentDefinition,
    handlers: tuple[object, ...] = (EchoHandler(),),
) -> AgentExecutor:
    registry = AgentRegistry(definitions)
    resolver = AgentResolver(registry)
    context_builder = AgentContextBuilder()
    handler_registry = AgentHandlerRegistry(handlers)
    return AgentExecutor(resolver, context_builder, handler_registry)


def _request(
    *,
    required_resolution_capabilities: tuple[str, ...] = ("agent.echo",),
    preferred_agent_ids: tuple[str, ...] = (),
    enabled_only: bool = True,
    unique: bool = True,
    required_capabilities: tuple[str, ...] = ("agent.echo",),
    required_permissions: tuple[str, ...] = (),
    user_input: str | None = "hello",
    structured_input: dict[str, object] | None = None,
) -> AgentExecutionRequest:
    return AgentExecutionRequest(
        resolution_request=AgentResolutionRequest(
            required_capability_ids=required_resolution_capabilities,
            preferred_agent_ids=preferred_agent_ids,
            enabled_only=enabled_only,
            require_unique_top_score=unique,
        ),
        execution_id="exec-1",
        correlation_id="corr-1",
        user_input=user_input,
        structured_input=structured_input or {"value": 1},
        required_capability_ids=required_capabilities,
        required_permission_ids=required_permissions,
        metadata={"source": "test"},
    )


def test_agent_executor_runs_deterministic_handler_end_to_end():
    executor = _executor(_definition())

    result = executor.execute(_request())

    assert result.status is AgentExecutionStatus.COMPLETED
    assert result.completed is True
    assert result.agent_id == "atlas.agent.echo"
    assert result.context is not None
    assert result.context.user_input == "hello"
    assert result.output == {
        "agent_id": "atlas.agent.echo",
        "structured": {"value": 1},
        "user_input": "hello",
    }
    assert result.metadata["sanitized_output_fields"] == 0
    assert tuple(event.name for event in result.events) == (
        "agent_execution_started",
        "agent_resolution_succeeded",
        "agent_context_built",
        "agent_handler_started",
        "agent_handler_succeeded",
        "agent_execution_completed",
    )


def test_executor_returns_no_agent_candidates_for_missing_agent():
    executor = _executor(_definition(capability_ids=("other.capability",)))

    result = executor.execute(_request())

    assert result.status is AgentExecutionStatus.NO_AGENT_CANDIDATES
    assert result.output is None


def test_executor_returns_ambiguous_for_non_unique_top_score():
    executor = _executor(
        _definition("atlas.agent.a"),
        _definition("atlas.agent.b"),
        handlers=(EchoHandler("atlas.agent.a"), EchoHandler("atlas.agent.b")),
    )

    result = executor.execute(_request(unique=True))

    assert result.status is AgentExecutionStatus.AGENT_AMBIGUOUS


def test_executor_rejects_disabled_selected_agent():
    executor = _executor(_definition(enabled=False))

    result = executor.execute(_request(enabled_only=False, preferred_agent_ids=("atlas.agent.echo",)))

    assert result.status is AgentExecutionStatus.AGENT_DISABLED
    assert result.agent_id == "atlas.agent.echo"


def test_executor_rejects_missing_handler():
    executor = _executor(_definition(), handlers=())

    result = executor.execute(_request())

    assert result.status is AgentExecutionStatus.HANDLER_UNAVAILABLE


def test_executor_surfaces_context_build_failure_without_handler_call():
    policy = AgentContextPolicy(allow_user_input=True, max_string_length=3)
    executor = _executor(_definition(context_policy=policy))

    result = executor.execute(_request(user_input="too long"))

    assert result.status is AgentExecutionStatus.CONTEXT_BUILD_FAILED
    assert result.error_code == "LIMIT_EXCEEDED"


def test_executor_rejects_missing_execution_permission():
    executor = _executor(_definition())

    result = executor.execute(_request(required_permissions=("can_write_files",)))

    assert result.status is AgentExecutionStatus.PERMISSION_DENIED
    assert "permission" in result.safe_message


def test_executor_rejects_execution_capability_not_allowed():
    executor = _executor(_definition())

    result = executor.execute(
        _request(
            preferred_agent_ids=("atlas.agent.echo",),
            required_capabilities=("agent.codegen",),
        )
    )

    assert result.status is AgentExecutionStatus.CAPABILITY_NOT_ALLOWED


def test_executor_converts_handler_exception_to_safe_failure():
    executor = _executor(_definition(), handlers=(FailingHandler(),))

    result = executor.execute(_request())

    assert result.status is AgentExecutionStatus.EXECUTION_FAILED
    assert result.error_code == "RuntimeError"
    assert "raw-secret" not in result.safe_message
    assert "[redacted]" in result.safe_message
    assert "Traceback" not in result.safe_message
    assert "agent_handler_failed" in tuple(event.name for event in result.events)


def test_executor_sanitizes_nested_sensitive_output_keys():
    executor = _executor(
        _definition(),
        handlers=(StaticHandler({"token": "secret", "nested": {"password": "secret", "ok": 1}}),),
    )

    result = executor.execute(_request())

    assert result.status is AgentExecutionStatus.COMPLETED
    assert result.output == {"nested": {"ok": 1}}
    assert result.metadata["sanitized_output_fields"] == 2


def test_executor_rejects_invalid_handler_result():
    executor = _executor(_definition(), handlers=(StaticHandler(["not", "a", "mapping"]),))

    result = executor.execute(_request())

    assert result.status is AgentExecutionStatus.EXECUTION_FAILED
    assert result.error_code == "AgentExecutionError"


def test_executor_rejects_result_limits_exceeded():
    executor = _executor(_definition(), handlers=(StaticHandler({"value": "x" * 1001}),))

    result = executor.execute(_request())

    assert result.status is AgentExecutionStatus.EXECUTION_FAILED
    assert result.safe_message == "string length limit exceeded."


def test_execution_request_signature_is_deterministic():
    request_a = _request(structured_input={"b": 2, "a": 1})
    request_b = _request(structured_input={"a": 1, "b": 2})

    assert agent_execution_request_signature(request_a) == agent_execution_request_signature(request_b)


def test_handler_registry_rejects_duplicates_and_allows_replace():
    registry = AgentHandlerRegistry()
    first = EchoHandler()
    second = StaticHandler({"ok": True})

    registry.register(first)
    with pytest.raises(AgentHandlerAlreadyRegisteredError):
        registry.register(second)

    registry.register(second, replace=True)
    assert registry.get("atlas.agent.echo") is second
    assert registry.contains("atlas.agent.echo") is True
    assert registry.list_handlers() == (second,)
    assert registry.unregister("atlas.agent.echo") is True
    assert registry.unregister("atlas.agent.echo") is False


def test_handler_registry_rejects_invalid_handler():
    registry = AgentHandlerRegistry()

    with pytest.raises(AgentHandlerRegistryError):
        registry.register(object())


def test_handler_registry_clear_preserves_empty_deterministic_state():
    registry = AgentHandlerRegistry((EchoHandler(),))

    registry.clear()

    assert len(registry) == 0
    assert registry.list_handlers() == ()


def test_executor_rejects_non_agent_execution_request():
    executor = _executor(_definition())

    result = executor.execute(object())

    assert result.status is AgentExecutionStatus.INVALID_REQUEST


def test_bootstrap_builds_core_agent_executor():
    registry = AgentRegistry((_definition(),))
    resolver = AgentResolver(registry)
    context_builder = AgentContextBuilder()
    handler_registry = AgentHandlerRegistry((EchoHandler(),))

    executor = build_core_agent_executor(resolver, context_builder, handler_registry)
    result = executor.execute(_request())

    assert isinstance(result.output, MappingProxyType)
    assert result.status is AgentExecutionStatus.COMPLETED


def test_agent_execution_context_is_propagated_to_handler_context() -> None:
    execution_context = SkillExecutionContext()

    result = _executor(_definition()).execute(
        replace(_request(), execution_context=execution_context)
    )

    assert result.status is AgentExecutionStatus.COMPLETED
    assert result.context is not None
    assert result.context.execution_context is execution_context


def test_cancelled_skill_execution_context_stops_agent_before_handler() -> None:
    execution_context = SkillExecutionContext(cancelled=True)

    result = _executor(_definition()).execute(
        replace(_request(), execution_context=execution_context)
    )

    assert result.status is AgentExecutionStatus.CANCELLED
    assert result.error_code == "CANCELLED"
    assert result.context is None
