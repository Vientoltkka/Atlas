from __future__ import annotations

from dataclasses import FrozenInstanceError, dataclass
from types import MappingProxyType

import pytest

from bootstrap.agent_handler_registration import build_core_agent_handler_registration_service
from core.agent_context import AgentContext, AgentContextBuilder
from core.agent_executor import (
    AgentExecutionRequest,
    AgentExecutionStatus,
    AgentHandlerRegistry,
    AgentHandlerRegistryError,
    AgentExecutor,
)
from core.agent_handler_registration import (
    AgentHandlerDuplicatePolicy,
    AgentHandlerRegistrationItem,
    AgentHandlerRegistrationPolicy,
    AgentHandlerRegistrationRequest,
    AgentHandlerRegistrationService,
    AgentHandlerRegistrationStatus,
    InvalidAgentHandlerRegistrationRequestError,
    agent_handler_registration_request_signature,
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


@dataclass(frozen=True)
class EchoHandler:
    agent_id: str = "atlas.agent.echo"

    def handle(self, context: AgentContext):
        return {"agent_id": context.agent_id, "ok": True}


@dataclass(frozen=True)
class OtherHandler:
    agent_id: str = "atlas.agent.other"

    def handle(self, context: AgentContext):
        return {"agent_id": context.agent_id, "other": True}


class FailingHandlerRegistry(AgentHandlerRegistry):
    def __init__(self, *, fail_on_call: int) -> None:
        super().__init__()
        self._fail_on_call = fail_on_call
        self._calls = 0

    def register(self, handler, *, replace: bool = False):
        self._calls += 1
        if self._calls == self._fail_on_call:
            raise AgentHandlerRegistryError("controlled token failure")
        return super().register(handler, replace=replace)


def _definition(agent_id: str = "atlas.agent.echo", handler_id: str | None = None) -> AgentDefinition:
    return AgentDefinition(
        agent_id=agent_id,
        agent_type=AgentType.GENERAL,
        name=f"Agent {agent_id}",
        description="Deterministic handler registration test agent.",
        permissions=AgentPermissions(requires_confirmation=False),
        limits=AgentLimits(max_steps=1, max_tool_calls=0, max_context_items=8),
        capabilities=AgentCapabilities(capabilities=("agent.echo",)),
        context_policy=AgentContextPolicy(allow_user_input=True),
        metadata={"handler_id": handler_id or f"{agent_id}.handler"},
    )


def _service(
    agent_registry: AgentRegistry | None = None,
    handler_registry: AgentHandlerRegistry | None = None,
) -> AgentHandlerRegistrationService:
    return AgentHandlerRegistrationService(
        agent_registry=agent_registry if agent_registry is not None else AgentRegistry((_definition(),)),
        agent_handler_registry=handler_registry if handler_registry is not None else AgentHandlerRegistry(),
    )


def _item(
    agent_id: str = "atlas.agent.echo",
    handler_id: str | None = None,
    handler=None,
    metadata: dict[str, object] | None = None,
) -> AgentHandlerRegistrationItem:
    return AgentHandlerRegistrationItem(
        agent_id=agent_id,
        handler_id=handler_id or f"{agent_id}.handler",
        handler=handler or EchoHandler(agent_id),
        metadata=metadata or {},
    )


def test_registers_one_valid_handler() -> None:
    handler_registry = AgentHandlerRegistry()
    result = _service(handler_registry=handler_registry).register(
        AgentHandlerRegistrationRequest(handlers=(_item(),))
    )

    assert result.status is AgentHandlerRegistrationStatus.COMPLETED
    assert result.registered_agent_ids == ("atlas.agent.echo",)
    assert handler_registry.contains("atlas.agent.echo") is True


def test_registers_multiple_handlers_deterministically() -> None:
    agent_registry = AgentRegistry(
        (
            _definition("atlas.agent.b"),
            _definition("atlas.agent.a"),
        )
    )
    handler_registry = AgentHandlerRegistry()

    result = _service(agent_registry, handler_registry).register(
        AgentHandlerRegistrationRequest(
            handlers=(
                _item("atlas.agent.b", handler=EchoHandler("atlas.agent.b")),
                _item("atlas.agent.a", handler=EchoHandler("atlas.agent.a")),
            )
        )
    )

    assert result.status is AgentHandlerRegistrationStatus.COMPLETED
    assert result.registered_agent_ids == ("atlas.agent.a", "atlas.agent.b")
    assert [handler.agent_id for handler in handler_registry.list_handlers()] == ["atlas.agent.a", "atlas.agent.b"]


def test_dry_run_does_not_mutate_handler_registry() -> None:
    handler_registry = AgentHandlerRegistry()

    result = _service(handler_registry=handler_registry).register(
        AgentHandlerRegistrationRequest(
            handlers=(_item(),),
            policy=AgentHandlerRegistrationPolicy(dry_run=True),
        )
    )

    assert result.status is AgentHandlerRegistrationStatus.DRY_RUN_COMPLETED
    assert result.entries[0].action == "would_register"
    assert len(handler_registry) == 0


def test_invalid_ids_are_rejected() -> None:
    with pytest.raises(InvalidAgentHandlerRegistrationRequestError):
        AgentHandlerRegistrationItem(handler_id="bad handler", agent_id="atlas.agent.echo", handler=EchoHandler())


def test_missing_agent_is_rejected() -> None:
    result = _service(agent_registry=AgentRegistry()).register(
        AgentHandlerRegistrationRequest(handlers=(_item(),))
    )

    assert result.status is AgentHandlerRegistrationStatus.AGENT_NOT_FOUND
    assert result.rejected_agent_ids == ("atlas.agent.echo",)


def test_handler_id_incompatible_with_agent_definition_is_rejected() -> None:
    result = _service().register(
        AgentHandlerRegistrationRequest(handlers=(_item(handler_id="atlas.handler.other"),))
    )

    assert result.status is AgentHandlerRegistrationStatus.HANDLER_INCOMPATIBLE


def test_duplicate_rejected_by_default() -> None:
    handler_registry = AgentHandlerRegistry((EchoHandler(),))

    result = _service(handler_registry=handler_registry).register(
        AgentHandlerRegistrationRequest(handlers=(_item(),))
    )

    assert result.status is AgentHandlerRegistrationStatus.DUPLICATE_HANDLER
    assert result.rejected_agent_ids == ("atlas.agent.echo",)


def test_keep_existing_policy_skips_registered_handler() -> None:
    existing = EchoHandler()
    handler_registry = AgentHandlerRegistry((existing,))
    new_handler = EchoHandler()

    result = _service(handler_registry=handler_registry).register(
        AgentHandlerRegistrationRequest(
            handlers=(_item(handler=new_handler),),
            policy=AgentHandlerRegistrationPolicy(duplicate_handler_policy=AgentHandlerDuplicatePolicy.KEEP_EXISTING),
        )
    )

    assert result.status is AgentHandlerRegistrationStatus.COMPLETED
    assert result.skipped_agent_ids == ("atlas.agent.echo",)
    assert handler_registry.get("atlas.agent.echo") is existing


def test_replace_policy_replaces_registered_handler() -> None:
    existing = EchoHandler()
    replacement = EchoHandler()
    handler_registry = AgentHandlerRegistry((existing,))

    result = _service(handler_registry=handler_registry).register(
        AgentHandlerRegistrationRequest(
            handlers=(_item(handler=replacement),),
            policy=AgentHandlerRegistrationPolicy(duplicate_handler_policy="REPLACE"),
        )
    )

    assert result.status is AgentHandlerRegistrationStatus.COMPLETED
    assert result.replaced_agent_ids == ("atlas.agent.echo",)
    assert handler_registry.get("atlas.agent.echo") is replacement


def test_atomic_batch_with_invalid_item_registers_none() -> None:
    agent_registry = AgentRegistry((_definition("atlas.agent.a"), _definition("atlas.agent.b")))
    handler_registry = AgentHandlerRegistry()

    result = _service(agent_registry, handler_registry).register(
        AgentHandlerRegistrationRequest(
            handlers=(
                _item("atlas.agent.a", handler=EchoHandler("atlas.agent.a")),
                _item("atlas.agent.b", handler_id="wrong.handler", handler=EchoHandler("atlas.agent.b")),
            )
        )
    )

    assert result.status is AgentHandlerRegistrationStatus.HANDLER_INCOMPATIBLE
    assert len(handler_registry) == 0


def test_controlled_registry_exception_rolls_back_partial_registration() -> None:
    agent_registry = AgentRegistry((_definition("atlas.agent.a"), _definition("atlas.agent.b")))
    handler_registry = FailingHandlerRegistry(fail_on_call=2)

    result = _service(agent_registry, handler_registry).register(
        AgentHandlerRegistrationRequest(
            handlers=(
                _item("atlas.agent.a", handler=EchoHandler("atlas.agent.a")),
                _item("atlas.agent.b", handler=EchoHandler("atlas.agent.b")),
            )
        )
    )

    assert result.status is AgentHandlerRegistrationStatus.REGISTRATION_FAILED
    assert len(handler_registry) == 0
    assert "token" not in " ".join(result.errors).lower()


def test_signature_is_stable() -> None:
    first = AgentHandlerRegistrationRequest(handlers=(_item(metadata={"trace": "a"}),), metadata={"batch": "x"})
    same = AgentHandlerRegistrationRequest(handlers=(_item(metadata={"trace": "a"}),), metadata={"batch": "x"})
    different = AgentHandlerRegistrationRequest(handlers=(_item(metadata={"trace": "b"}),), metadata={"batch": "x"})

    assert agent_handler_registration_request_signature(first) == agent_handler_registration_request_signature(same)
    assert agent_handler_registration_request_signature(first) != agent_handler_registration_request_signature(different)


def test_sensitive_metadata_is_rejected_and_result_errors_are_sanitized() -> None:
    with pytest.raises(InvalidAgentHandlerRegistrationRequestError):
        AgentHandlerRegistrationRequest(handlers=(_item(),), metadata={"api_key": "hidden-token"})

    handler_registry = FailingHandlerRegistry(fail_on_call=1)
    result = _service(handler_registry=handler_registry).register(
        AgentHandlerRegistrationRequest(handlers=(_item(),))
    )

    text = " ".join(result.errors).lower()
    assert "token" not in text
    assert "[redacted]" in text


def test_request_metadata_is_immutable() -> None:
    request = AgentHandlerRegistrationRequest(handlers=(_item(),), metadata={"trace": "safe"})

    assert isinstance(request.metadata, MappingProxyType)
    with pytest.raises(TypeError):
        request.metadata["trace"] = "other"  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        request.metadata = {}  # type: ignore[misc]


def test_callable_handler_is_adapted_without_execution() -> None:
    calls: list[str] = []

    def callback(context: AgentContext):
        calls.append(context.agent_id)
        return {"ok": True}

    handler_registry = AgentHandlerRegistry()
    result = _service(handler_registry=handler_registry).register(
        AgentHandlerRegistrationRequest(handlers=(_item(handler=callback),))
    )

    assert result.status is AgentHandlerRegistrationStatus.COMPLETED
    assert calls == []
    assert handler_registry.contains("atlas.agent.echo") is True


def test_compatibility_with_agent_executor_existing_flow() -> None:
    agent_registry = AgentRegistry((_definition(),))
    handler_registry = AgentHandlerRegistry()
    service = build_core_agent_handler_registration_service(agent_registry, handler_registry)

    registration = service.register(AgentHandlerRegistrationRequest(handlers=(_item(),)))
    executor = AgentExecutor(AgentResolver(agent_registry), AgentContextBuilder(), handler_registry)
    execution = executor.execute(
        AgentExecutionRequest(
            resolution_request=AgentResolutionRequest(
                required_agent_ids=("atlas.agent.echo",),
                require_unique_top_score=False,
            ),
            user_input="hello",
        )
    )

    assert isinstance(service, AgentHandlerRegistrationService)
    assert registration.status is AgentHandlerRegistrationStatus.COMPLETED
    assert execution.status is AgentExecutionStatus.COMPLETED
    assert execution.output == {"agent_id": "atlas.agent.echo", "ok": True}
