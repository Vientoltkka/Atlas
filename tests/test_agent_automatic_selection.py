from __future__ import annotations

from collections.abc import Mapping
from types import SimpleNamespace

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
from core.atlas_request_adapter import AtlasRequestAdapter
from core.atlas_request_classifier import AtlasRequestClassifier, StructuredInput
from core.atlas_request_normalizer import AtlasRequestNormalizer
from core.atlas_router import (
    AgentSelectionStatus,
    AtlasAgentRoutingRequest,
    AtlasRouteType,
    AtlasRouter,
    AtlasRoutingRequest,
    AtlasRoutingStatus,
    InvalidAtlasRoutingRequestError,
    atlas_routing_request_signature,
)
from core.orchestrator import AtlasOrchestrator
from core.router import Router
from memory.conversation import ConversationMemory


class RecordingHandler:
    def __init__(self, agent_id: str, *, fail: bool = False) -> None:
        self._agent_id = agent_id
        self.fail = fail
        self.calls = 0

    @property
    def agent_id(self) -> str:
        return self._agent_id

    def handle(self, context: AgentContext) -> Mapping[str, object]:
        self.calls += 1
        if self.fail:
            raise RuntimeError("api_token leaked")
        return {"agent_id": context.agent_id, "value": context.structured_input.get("value", "ok")}


class ChatAgent:
    name = "chat"
    generated_path = None

    def run(self, *, model: str, messages):
        del model, messages
        return "chat"


def _definition(
    agent_id: str,
    *,
    agent_type: AgentType = AgentType.GENERAL,
    enabled: bool = True,
    capabilities: tuple[str, ...] = ("agent.inspect",),
    permissions: AgentPermissions | None = None,
) -> AgentDefinition:
    return AgentDefinition(
        agent_id=agent_id,
        agent_type=agent_type,
        name=agent_id,
        description="Automatic selection test agent.",
        enabled=enabled,
        capabilities=AgentCapabilities(capabilities=capabilities),
        permissions=permissions or AgentPermissions(requires_confirmation=False),
        context_policy=AgentContextPolicy(allow_user_input=True, allow_shared_context=True),
        metadata={"handler_id": f"{agent_id}.handler"},
    )


def _system(
    definitions: tuple[AgentDefinition, ...],
    handlers: tuple[RecordingHandler, ...] = (),
) -> AgentSystem:
    result = build_core_agent_system()
    assert result.system is not None
    system = result.system
    for definition in definitions:
        system.agent_registry.register(definition)
    for handler in handlers:
        system.agent_handler_registry.register(handler)
    return system


def _route(
    system: AgentSystem,
    request: AtlasAgentRoutingRequest,
) -> object:
    return AtlasRouter(agent_system=system).route(AtlasRoutingRequest(AtlasRouteType.AGENT, request))


def _orchestrator(router: AtlasRouter) -> AtlasOrchestrator:
    chat = ChatAgent()
    registry = SimpleNamespace(get=lambda name: chat if name == "chat" else None)
    return AtlasOrchestrator(
        planner=SimpleNamespace(create_plan=lambda prompt: SimpleNamespace(task=prompt, objective=prompt)),
        router=Router(),
        model_manager=SimpleNamespace(choose_model=lambda agent_name: f"model:{agent_name}"),
        memory=ConversationMemory(),
        registry=registry,
        write_file=SimpleNamespace(execute=lambda *_args: "written"),
        atlas_router=router,
        atlas_request_adapter=AtlasRequestAdapter(),
        atlas_request_classifier=AtlasRequestClassifier(),
        atlas_request_normalizer=AtlasRequestNormalizer(),
    )


def test_selects_single_compatible_agent_automatically() -> None:
    selected = RecordingHandler("agent.coding")
    other = RecordingHandler("agent.other")
    result = _route(
        _system(
            (
                _definition("agent.other", capabilities=("agent.inspect",)),
                _definition("agent.coding", capabilities=("code.edit",)),
            ),
            (selected, other),
        ),
        AtlasAgentRoutingRequest(payload={"value": "done"}, required_capabilities=("code.edit",)),
    )

    assert result.status is AtlasRoutingStatus.COMPLETED
    assert result.agent_resolution_result is not None
    assert result.agent_resolution_result.selected_agent_id == "agent.coding"
    assert dict(result.output) == {"agent_id": "agent.coding", "value": "done"}
    assert selected.calls == 1
    assert other.calls == 0


def test_explicit_agent_id_does_not_activate_automatic_selection_or_fallback() -> None:
    requested = RecordingHandler("agent.requested")
    fallback = RecordingHandler("agent.fallback")
    result = _route(
        _system(
            (
                _definition("agent.requested", capabilities=("read.only",)),
                _definition("agent.fallback", capabilities=("code.edit",)),
            ),
            (requested, fallback),
        ),
        AtlasAgentRoutingRequest(agent_id="agent.requested", required_capabilities=("code.edit",)),
    )

    assert result.status is AtlasRoutingStatus.EXECUTION_FAILED
    assert result.agent_result is not None
    assert result.agent_result.status is AgentExecutionStatus.CAPABILITY_NOT_ALLOWED
    assert requested.calls == 0
    assert fallback.calls == 0
    assert result.metrics["agent_auto_selections_requested"] == 0


def test_explicit_agent_id_ignores_preferred_agent_ids() -> None:
    requested = RecordingHandler("agent.requested")
    preferred = RecordingHandler("agent.preferred")
    result = _route(
        _system(
            (_definition("agent.requested"), _definition("agent.preferred")),
            (requested, preferred),
        ),
        AtlasAgentRoutingRequest(
            agent_id="agent.requested",
            preferred_agent_ids=("agent.preferred",),
            required_capabilities=("agent.inspect",),
        ),
    )

    assert result.status is AtlasRoutingStatus.COMPLETED
    assert requested.calls == 1
    assert preferred.calls == 0


def test_filters_disabled_and_excluded_agents() -> None:
    handler = RecordingHandler("agent.enabled")
    disabled = _route(
        _system(
            (
                _definition("agent.disabled", enabled=False, capabilities=("code.edit",)),
                _definition("agent.enabled", capabilities=("code.edit",)),
            ),
            (handler,),
        ),
        AtlasAgentRoutingRequest(required_capabilities=("code.edit",)),
    )
    excluded = _route(
        _system(
            (
                _definition("agent.excluded", capabilities=("code.edit", "extra.capability")),
                _definition("agent.enabled", capabilities=("code.edit",)),
            ),
            (handler,),
        ),
        AtlasAgentRoutingRequest(required_capabilities=("code.edit",), excluded_agent_ids=("agent.excluded",)),
    )

    assert disabled.status is AtlasRoutingStatus.COMPLETED
    assert disabled.agent_resolution_result is not None
    assert {item.agent_id for item in disabled.agent_resolution_result.rejections} == {"agent.disabled"}
    assert excluded.status is AtlasRoutingStatus.COMPLETED
    assert excluded.agent_resolution_result is not None
    assert {item.agent_id for item in excluded.agent_resolution_result.rejections} == {"agent.excluded"}


def test_rejects_capability_and_permission_mismatches() -> None:
    no_capability = _route(
        _system((_definition("agent.reader", capabilities=("read.project",)),)),
        AtlasAgentRoutingRequest(required_capabilities=("code.edit",)),
    )
    no_permission = _route(
        _system((_definition("agent.reader", permissions=AgentPermissions(can_execute_tools=False, requires_confirmation=False)),)),
        AtlasAgentRoutingRequest(required_permissions=("can_execute_tools",)),
    )

    assert no_capability.status is AtlasRoutingStatus.EXECUTION_FAILED
    assert no_capability.error_code == AgentSelectionStatus.NO_MATCHING_AGENT.value
    assert no_permission.status is AtlasRoutingStatus.EXECUTION_FAILED
    assert no_permission.error_code == AgentSelectionStatus.NO_MATCHING_AGENT.value


def test_preferred_agent_ids_and_agent_types_break_ties_declaratively() -> None:
    preferred_id_handler = RecordingHandler("agent.preferred")
    preferred_id = _route(
        _system(
            (_definition("agent.base"), _definition("agent.preferred")),
            (preferred_id_handler,),
        ),
        AtlasAgentRoutingRequest(
            required_capabilities=("agent.inspect",),
            preferred_agent_ids=("agent.preferred",),
        ),
    )
    preferred_type_handler = RecordingHandler("agent.memory")
    preferred_type = _route(
        _system(
            (
                _definition("agent.coding", agent_type=AgentType.CODING),
                _definition("agent.memory", agent_type=AgentType.MEMORY),
            ),
            (preferred_type_handler,),
        ),
        AtlasAgentRoutingRequest(
            required_capabilities=("agent.inspect",),
            preferred_agent_types=(AgentType.MEMORY,),
        ),
    )

    assert preferred_id.status is AtlasRoutingStatus.COMPLETED
    assert preferred_id.agent_resolution_result is not None
    assert preferred_id.agent_resolution_result.selected_agent_id == "agent.preferred"
    assert preferred_type.status is AtlasRoutingStatus.COMPLETED
    assert preferred_type.agent_resolution_result is not None
    assert preferred_type.agent_resolution_result.selected_agent_id == "agent.memory"


def test_no_match_and_ambiguous_selection_do_not_execute_agents() -> None:
    first = RecordingHandler("agent.a")
    second = RecordingHandler("agent.b")
    no_match = _route(
        _system((_definition("agent.a"),), (first,)),
        AtlasAgentRoutingRequest(required_capabilities=("missing.capability",)),
    )
    ambiguous = _route(
        _system((_definition("agent.a"), _definition("agent.b")), (first, second)),
        AtlasAgentRoutingRequest(required_capabilities=("agent.inspect",)),
    )

    assert no_match.status is AtlasRoutingStatus.EXECUTION_FAILED
    assert no_match.error_code == AgentSelectionStatus.NO_MATCHING_AGENT.value
    assert ambiguous.status is AtlasRoutingStatus.EXECUTION_FAILED
    assert ambiguous.error_code == AgentSelectionStatus.AMBIGUOUS_SELECTION.value
    assert first.calls == 0
    assert second.calls == 0


def test_selection_is_independent_of_registration_order_and_signature_normalizes_required_lists() -> None:
    first_handler = RecordingHandler("agent.preferred")
    first = _route(
        _system(
            (
                _definition("agent.other", capabilities=("a.capability", "b.capability")),
                _definition("agent.preferred", capabilities=("a.capability", "b.capability")),
            ),
            (first_handler,),
        ),
        AtlasAgentRoutingRequest(
            required_capabilities=("b.capability", "a.capability"),
            preferred_agent_ids=("agent.preferred",),
            excluded_agent_ids=("agent.z", "agent.y"),
        ),
    )
    second_handler = RecordingHandler("agent.preferred")
    second = _route(
        _system(
            (
                _definition("agent.preferred", capabilities=("a.capability", "b.capability")),
                _definition("agent.other", capabilities=("a.capability", "b.capability")),
            ),
            (second_handler,),
        ),
        AtlasAgentRoutingRequest(
            required_capabilities=("a.capability", "b.capability"),
            preferred_agent_ids=("agent.preferred",),
            excluded_agent_ids=("agent.y", "agent.z"),
        ),
    )
    first_signature = atlas_routing_request_signature(
        AtlasRoutingRequest(
            AtlasRouteType.AGENT,
            AtlasAgentRoutingRequest(
                required_capabilities=("b.capability", "a.capability"),
                required_permissions=("can_read_project", "requires_confirmation"),
                excluded_agent_ids=("agent.z", "agent.y"),
                preferred_agent_ids=("agent.preferred",),
            ),
        )
    )
    second_signature = atlas_routing_request_signature(
        AtlasRoutingRequest(
            AtlasRouteType.AGENT,
            AtlasAgentRoutingRequest(
                required_capabilities=("a.capability", "b.capability"),
                required_permissions=("requires_confirmation", "can_read_project"),
                excluded_agent_ids=("agent.y", "agent.z"),
                preferred_agent_ids=("agent.preferred",),
            ),
        )
    )

    assert first.status is AtlasRoutingStatus.COMPLETED
    assert second.status is AtlasRoutingStatus.COMPLETED
    assert first.agent_resolution_result is not None
    assert second.agent_resolution_result is not None
    assert first.agent_resolution_result.selected_agent_id == second.agent_resolution_result.selected_agent_id
    assert first_signature == second_signature


def test_rejects_insufficient_criteria_invalid_identifiers_sensitive_keys_and_limits() -> None:
    handler = RecordingHandler("agent.a")
    insufficient = _route(
        _system((_definition("agent.a"),), (handler,)),
        AtlasAgentRoutingRequest(payload={"safe": "value"}),
    )

    assert insufficient.status is AtlasRoutingStatus.INVALID_REQUEST
    assert insufficient.error_code == AgentSelectionStatus.INSUFFICIENT_SELECTION_CRITERIA.value
    assert handler.calls == 0
    with pytest.raises(InvalidAtlasRoutingRequestError):
        AtlasAgentRoutingRequest(preferred_agent_ids=("__class__",))
    with pytest.raises(InvalidAtlasRoutingRequestError):
        AtlasAgentRoutingRequest(required_capabilities=("agent.inspect",), payload={"api_key": "hidden"})
    with pytest.raises(InvalidAtlasRoutingRequestError):
        AtlasAgentRoutingRequest(required_capabilities=("agent.inspect",), metadata={"token": "hidden"})
    with pytest.raises(InvalidAtlasRoutingRequestError):
        AtlasAgentRoutingRequest(required_capabilities=tuple(f"cap.{index}" for index in range(33)))


def test_valid_selection_executes_one_handler_and_missing_handler_is_structured() -> None:
    selected = RecordingHandler("agent.selected")
    completed = _route(
        _system((_definition("agent.selected"), _definition("agent.other", capabilities=("other.capability",))), (selected,)),
        AtlasAgentRoutingRequest(required_capabilities=("agent.inspect",)),
    )
    missing_handler = _route(
        _system((_definition("agent.selected"),)),
        AtlasAgentRoutingRequest(required_capabilities=("agent.inspect",)),
    )

    assert completed.status is AtlasRoutingStatus.COMPLETED
    assert selected.calls == 1
    assert completed.metrics["agent_auto_selections_succeeded"] == 1
    assert missing_handler.status is AtlasRoutingStatus.EXECUTION_FAILED
    assert missing_handler.agent_result is not None
    assert missing_handler.agent_result.status is AgentExecutionStatus.HANDLER_UNAVAILABLE


def test_agent_system_absent_keeps_service_unavailable() -> None:
    result = AtlasRouter().route(
        AtlasRoutingRequest(
            AtlasRouteType.AGENT,
            AtlasAgentRoutingRequest(required_capabilities=("agent.inspect",)),
        )
    )

    assert result.status is AtlasRoutingStatus.SERVICE_UNAVAILABLE
    assert result.error_code == "AGENT_SYSTEM_UNAVAILABLE"


def test_no_fallback_after_selected_agent_handler_failure() -> None:
    failing = RecordingHandler("agent.preferred", fail=True)
    fallback = RecordingHandler("agent.fallback")
    result = _route(
        _system(
            (_definition("agent.preferred"), _definition("agent.fallback")),
            (failing, fallback),
        ),
        AtlasAgentRoutingRequest(
            required_capabilities=("agent.inspect",),
            preferred_agent_ids=("agent.preferred",),
        ),
    )

    assert result.status is AtlasRoutingStatus.EXECUTION_FAILED
    assert result.agent_result is not None
    assert result.agent_result.status is AgentExecutionStatus.EXECUTION_FAILED
    assert failing.calls == 1
    assert fallback.calls == 0


def test_e2e_structured_input_to_agent_executor() -> None:
    handler = RecordingHandler("agent.coding")
    router = build_core_atlas_router(
        agent_system=_system(
            (_definition("agent.coding", capabilities=("code.edit",)),),
            (handler,),
        )
    )

    result = _orchestrator(router).route_structured_input(
        StructuredInput(route="agent", payload={"required_capabilities": ("code.edit",), "payload": {"value": "e2e"}})
    )

    assert result.status is AtlasRoutingStatus.COMPLETED
    assert dict(result.output) == {"agent_id": "agent.coding", "value": "e2e"}
    assert handler.calls == 1


def test_events_and_metrics_are_safe_and_explain_selection() -> None:
    result = _route(
        _system((_definition("agent.a"),), (RecordingHandler("agent.a"),)),
        AtlasAgentRoutingRequest(required_capabilities=("agent.inspect",)),
    )
    rendered = repr(result.events) + repr(result.metrics)

    assert result.status is AtlasRoutingStatus.COMPLETED
    assert "agent_auto_selection_requested" in {event.name for event in result.events}
    assert "agent_candidate_evaluated" in {event.name for event in result.events}
    assert result.metrics["agent_candidates_evaluated"] == 1
    assert "token" not in rendered
    assert "api_key" not in rendered
