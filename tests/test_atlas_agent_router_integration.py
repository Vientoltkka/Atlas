from __future__ import annotations

from collections.abc import Mapping

import pytest

from bootstrap.agent_system import build_core_agent_system
from bootstrap.atlas_router import build_core_atlas_router
from core.agent_context import AgentContext
from core.agent_executor import AgentExecutionStatus
from core.agent_registry import (
    AgentCapabilities,
    AgentContextPolicy,
    AgentDefinition,
    AgentPermissions,
    AgentType,
)
from core.agent_system import AgentSystem
from core.atlas_router import (
    AtlasAgentRoutingRequest,
    AtlasRouteType,
    AtlasRouter,
    AtlasRoutingRequest,
    AtlasRoutingStatus,
    InvalidAtlasRoutingRequestError,
    atlas_routing_request_signature,
)
from core.capability_execution_service import (
    CapabilityExecutionRequest,
    CapabilityExecutionResult,
    CapabilityExecutionService,
    CapabilityExecutionStatus,
)


class EchoHandler:
    def __init__(self, agent_id: str = "atlas.agent.echo", *, fail: bool = False) -> None:
        self._agent_id = agent_id
        self.fail = fail
        self.calls = 0

    @property
    def agent_id(self) -> str:
        return self._agent_id

    def handle(self, context: AgentContext) -> Mapping[str, object]:
        self.calls += 1
        if self.fail:
            raise RuntimeError("authorization token leaked")
        return {
            "agent_id": context.agent_id,
            "input": context.structured_input,
            "token": "hidden",
        }


class CountingCapabilityService(CapabilityExecutionService):
    def __init__(self) -> None:
        self.calls = 0

    def execute(self, request: CapabilityExecutionRequest) -> CapabilityExecutionResult:
        del request
        self.calls += 1
        return CapabilityExecutionResult(CapabilityExecutionStatus.COMPLETED, output={"ok": True})


def _definition(
    agent_id: str = "atlas.agent.echo",
    *,
    enabled: bool = True,
    capabilities: tuple[str, ...] = ("agent.echo",),
    permissions: AgentPermissions | None = None,
    context_policy: AgentContextPolicy | None = None,
) -> AgentDefinition:
    return AgentDefinition(
        agent_id=agent_id,
        agent_type=AgentType.GENERAL,
        name="Echo agent",
        description="Deterministic test agent.",
        enabled=enabled,
        capabilities=AgentCapabilities(capabilities=capabilities),
        permissions=permissions or AgentPermissions(requires_confirmation=False),
        context_policy=context_policy or AgentContextPolicy(allow_user_input=True, allow_shared_context=True),
        metadata={"handler_id": f"{agent_id}.handler"},
    )


def _system(*definitions: AgentDefinition, handler: EchoHandler | None = None) -> AgentSystem:
    result = build_core_agent_system()
    assert result.system is not None
    system = result.system
    for definition in definitions:
        system.agent_registry.register(definition)
    if handler is not None:
        system.agent_handler_registry.register(handler)
    return system


def test_agent_route_executes_explicit_agent_through_existing_executor() -> None:
    handler = EchoHandler()
    router = build_core_atlas_router(agent_system=_system(_definition(), handler=handler))

    result = router.route(
        AtlasRoutingRequest(
            AtlasRouteType.AGENT,
            AtlasAgentRoutingRequest(
                agent_id="atlas.agent.echo",
                payload={"value": 1},
                shared_context={"trace": "safe"},
                required_capabilities=("agent.echo",),
                user_input="run",
            ),
            request_id="agent-request",
        )
    )

    assert result.status is AtlasRoutingStatus.COMPLETED
    assert result.route_type is AtlasRouteType.AGENT
    assert result.agent_result is not None
    assert result.agent_result.status is AgentExecutionStatus.COMPLETED
    assert dict(result.output) == {"agent_id": "atlas.agent.echo", "input": {"value": 1}}
    assert result.metrics["agent_route_completed"] == 1
    assert handler.calls == 1


def test_agent_route_accepts_mapping_payload_and_convenience_api() -> None:
    handler = EchoHandler()
    router = AtlasRouter(agent_system=_system(_definition(), handler=handler))

    mapped = router.route(
        AtlasRoutingRequest(
            "agent",
            payload={
                "agent_id": "atlas.agent.echo",
                "payload": {"value": 2},
                "required_capabilities": ("agent.echo",),
            },
        )
    )
    direct = router.route_agent_request(AtlasAgentRoutingRequest("atlas.agent.echo", payload={"value": 3}))

    assert mapped.status is AtlasRoutingStatus.COMPLETED
    assert direct.status is AtlasRoutingStatus.COMPLETED
    assert handler.calls == 2


def test_agent_route_without_system_is_structured_service_unavailable() -> None:
    result = AtlasRouter().route(
        AtlasRoutingRequest(AtlasRouteType.AGENT, AtlasAgentRoutingRequest("atlas.agent.echo"))
    )

    assert result.status is AtlasRoutingStatus.SERVICE_UNAVAILABLE
    assert result.error_code == "AGENT_SYSTEM_UNAVAILABLE"
    assert result.agent_result is None


def test_agent_route_rejects_invalid_or_missing_agent_id_without_execution() -> None:
    router = AtlasRouter(agent_system=_system(_definition(), handler=EchoHandler()))

    missing = router.route(AtlasRoutingRequest(AtlasRouteType.AGENT, payload={"payload": {"x": 1}}))

    assert missing.status is AtlasRoutingStatus.INVALID_REQUEST
    assert missing.error_code == "INSUFFICIENT_SELECTION_CRITERIA"
    with pytest.raises(InvalidAtlasRoutingRequestError):
        AtlasAgentRoutingRequest("__class__")


def test_agent_route_nonexistent_agent_returns_executor_error() -> None:
    handler = EchoHandler()
    result = AtlasRouter(agent_system=_system(_definition(), handler=handler)).route(
        AtlasRoutingRequest(AtlasRouteType.AGENT, AtlasAgentRoutingRequest("atlas.agent.missing"))
    )

    assert result.status is AtlasRoutingStatus.EXECUTION_FAILED
    assert result.agent_result is not None
    assert result.agent_result.status is AgentExecutionStatus.NO_AGENT_CANDIDATES
    assert handler.calls == 0


def test_agent_route_disabled_agent_missing_handler_context_permissions_and_capabilities() -> None:
    disabled = AtlasRouter(agent_system=_system(_definition(enabled=False), handler=EchoHandler())).route(
        AtlasRoutingRequest(AtlasRouteType.AGENT, AtlasAgentRoutingRequest("atlas.agent.echo"))
    )
    missing_handler = AtlasRouter(agent_system=_system(_definition())).route(
        AtlasRoutingRequest(AtlasRouteType.AGENT, AtlasAgentRoutingRequest("atlas.agent.echo"))
    )
    context_rejected = AtlasRouter(
        agent_system=_system(
            _definition(context_policy=AgentContextPolicy(allow_user_input=True, max_string_length=3)),
            handler=EchoHandler(),
        )
    ).route(AtlasRoutingRequest(AtlasRouteType.AGENT, AtlasAgentRoutingRequest("atlas.agent.echo", user_input="long")))
    permission_denied = AtlasRouter(agent_system=_system(_definition(), handler=EchoHandler())).route(
        AtlasRoutingRequest(
            AtlasRouteType.AGENT,
            AtlasAgentRoutingRequest("atlas.agent.echo", required_permissions=("can_write_files",)),
        )
    )
    capability_denied = AtlasRouter(agent_system=_system(_definition(), handler=EchoHandler())).route(
        AtlasRoutingRequest(
            AtlasRouteType.AGENT,
            AtlasAgentRoutingRequest("atlas.agent.echo", required_capabilities=("agent.missing",)),
        )
    )

    assert disabled.agent_result is not None
    assert disabled.agent_result.status is AgentExecutionStatus.AGENT_DISABLED
    assert missing_handler.agent_result is not None
    assert missing_handler.agent_result.status is AgentExecutionStatus.HANDLER_UNAVAILABLE
    assert context_rejected.agent_result is not None
    assert context_rejected.agent_result.status is AgentExecutionStatus.CONTEXT_BUILD_FAILED
    assert permission_denied.agent_result is not None
    assert permission_denied.agent_result.status is AgentExecutionStatus.PERMISSION_DENIED
    assert capability_denied.agent_result is not None
    assert capability_denied.agent_result.status is AgentExecutionStatus.CAPABILITY_NOT_ALLOWED


def test_agent_handler_exception_is_sanitized_and_structured() -> None:
    result = AtlasRouter(agent_system=_system(_definition(), handler=EchoHandler(fail=True))).route(
        AtlasRoutingRequest(AtlasRouteType.AGENT, AtlasAgentRoutingRequest("atlas.agent.echo"))
    )

    assert result.status is AtlasRoutingStatus.EXECUTION_FAILED
    assert result.agent_result is not None
    assert result.agent_result.status is AgentExecutionStatus.EXECUTION_FAILED
    assert "authorization" not in repr(result)
    assert "token" not in repr(result)


def test_agent_route_without_explicit_agent_id_uses_fail_closed_automatic_selection() -> None:
    handler = EchoHandler()
    router = AtlasRouter(
        agent_system=_system(
            _definition(),
            _definition("atlas.agent.other", capabilities=("agent.echo",)),
            handler=handler,
        )
    )

    result = router.route(
        AtlasRoutingRequest(AtlasRouteType.AGENT, payload={"required_capabilities": ("agent.echo",)})
    )

    assert result.status is AtlasRoutingStatus.EXECUTION_FAILED
    assert result.error_code == "AMBIGUOUS_SELECTION"
    assert handler.calls == 0


def test_agent_route_signature_is_stable() -> None:
    first = AtlasRoutingRequest(
        AtlasRouteType.AGENT,
        AtlasAgentRoutingRequest(
            "atlas.agent.echo",
            payload={"b": 2, "a": 1},
            required_capabilities=("agent.echo",),
        ),
        request_id="r1",
    )
    second = AtlasRoutingRequest(
        "agent",
        AtlasAgentRoutingRequest(
            "atlas.agent.echo",
            payload={"a": 1, "b": 2},
            required_capabilities=("agent.echo",),
        ),
        request_id="r1",
    )

    assert atlas_routing_request_signature(first) == atlas_routing_request_signature(second)


def test_previous_routes_remain_compatible() -> None:
    service = CountingCapabilityService()
    router = build_core_atlas_router(capability_execution_service=service, agent_system=_system())

    capability = router.route(AtlasRoutingRequest(AtlasRouteType.CAPABILITY, CapabilityExecutionRequest()))
    tool = router.route(AtlasRoutingRequest(AtlasRouteType.TOOL, payload={"safe": "value"}))
    conversation = router.route(AtlasRoutingRequest(AtlasRouteType.CONVERSATION, payload={"safe": "value"}))

    assert capability.status is AtlasRoutingStatus.COMPLETED
    assert service.calls == 1
    assert tool.status is AtlasRoutingStatus.ROUTE_UNAVAILABLE
    assert conversation.status is AtlasRoutingStatus.ROUTE_UNAVAILABLE


def test_bootstrap_router_uses_shared_agent_system_without_build_time_execution() -> None:
    handler = EchoHandler()
    system = _system(_definition(), handler=handler)
    router = build_core_atlas_router(agent_system=system)

    assert handler.calls == 0
    result = router.route(AtlasRoutingRequest(AtlasRouteType.AGENT, AtlasAgentRoutingRequest("atlas.agent.echo")))

    assert result.status is AtlasRoutingStatus.COMPLETED
    assert result.agent_result is not None
    assert result.agent_result.context is not None
    assert result.agent_result.context.agent_id == system.agent_registry.get("atlas.agent.echo").agent_id
    assert handler.calls == 1
