from __future__ import annotations

from collections.abc import Mapping

import pytest

from bootstrap.agent_delegation import build_core_agent_delegation_service
from bootstrap.agent_system import build_core_agent_system
from core.agent_context import AgentContext, AgentContextBuilder
from core.agent_delegation import (
    AgentDelegationPolicy,
    AgentDelegationRequest,
    AgentDelegationService,
    AgentDelegationStatus,
    InvalidAgentDelegationRequestError,
    agent_delegation_request_signature,
)
from core.agent_executor import AgentExecutionRequest, AgentExecutionResult, AgentExecutionStatus, AgentExecutor
from core.agent_registry import AgentCapabilities, AgentContextPolicy, AgentDefinition, AgentPermissions, AgentRegistry, AgentType
from core.agent_resolver import AgentResolutionRequest, AgentResolver


class RecordingHandler:
    calls: list[tuple[str, Mapping[str, object], Mapping[str, object], Mapping[str, object]]] = []

    def __init__(self, agent_id: str, *, fail: bool = False) -> None:
        self._agent_id = agent_id
        self.fail = fail

    @property
    def agent_id(self) -> str:
        return self._agent_id

    def handle(self, context: AgentContext) -> Mapping[str, object]:
        RecordingHandler.calls.append(
            (self._agent_id, context.structured_input, context.shared_context, context.metadata)
        )
        if self.fail:
            raise RuntimeError("authorization token raw-secret")
        return {
            "agent_id": self._agent_id,
            "structured": context.structured_input,
            "shared": context.shared_context,
            "token": "hidden",
        }


class FailingResolver(AgentResolver):
    def resolve(self, request: AgentResolutionRequest):  # type: ignore[override]
        raise RuntimeError("resolver failed with api_key raw-secret")


class FailingExecutor(AgentExecutor):
    def execute(self, request: AgentExecutionRequest) -> AgentExecutionResult:  # type: ignore[override]
        return AgentExecutionResult(
            status=AgentExecutionStatus.EXECUTION_FAILED,
            request_signature="executor-signature",
            execution_id=request.execution_id,
            agent_id="agent.target",
            error_code="CONTROLLED_FAILURE",
            safe_message="authorization token raw-secret",
        )


def _definition(
    agent_id: str,
    *,
    agent_type: AgentType = AgentType.GENERAL,
    capabilities: tuple[str, ...] = ("agent.inspect",),
    permissions: AgentPermissions | None = None,
    enabled: bool = True,
    context_policy: AgentContextPolicy | None = None,
) -> AgentDefinition:
    return AgentDefinition(
        agent_id=agent_id,
        agent_type=agent_type,
        name=agent_id,
        description="Delegation test agent.",
        capabilities=AgentCapabilities(capabilities=capabilities),
        permissions=permissions or AgentPermissions(requires_confirmation=False),
        context_policy=context_policy or AgentContextPolicy(allow_shared_context=True),
        enabled=enabled,
    )


def _system(
    definitions: tuple[AgentDefinition, ...] | None = None,
    handlers: tuple[RecordingHandler, ...] | None = None,
):
    result = build_core_agent_system()
    assert result.system is not None
    system = result.system
    for definition in definitions or (_definition("agent.origin"), _definition("agent.target")):
        system.agent_registry.register(definition)
    for handler in handlers or (RecordingHandler("agent.target"),):
        system.agent_handler_registry.register(handler)
    return system


def _policy(**overrides: object) -> AgentDelegationPolicy:
    values = {"enabled": True}
    values.update(overrides)
    return AgentDelegationPolicy(**values)


def _request(**overrides: object) -> AgentDelegationRequest:
    values = {
        "origin_agent_id": "agent.origin",
        "target_agent_id": "agent.target",
        "required_capability_ids": ("agent.inspect",),
        "structured_input": {"value": 1},
        "shared_context": {"trace": "ok"},
        "metadata": {"source": "test"},
        "execution_id": "exec-1",
        "parent_execution_id": "parent-1",
        "reason_code": "handoff",
        "policy": _policy(propagate_shared_context=True, propagate_metadata=True),
    }
    values.update(overrides)
    return AgentDelegationRequest(**values)


def setup_function() -> None:
    RecordingHandler.calls = []


def test_explicit_target_delegation_executes_one_target_through_executor() -> None:
    result = _system().agent_delegation_service.delegate(_request())

    assert result.status is AgentDelegationStatus.SUCCESS
    assert result.success is True
    assert result.target_agent_id == "agent.target"
    assert RecordingHandler.calls == [
        (
            "agent.target",
            {"value": 1},
            {"trace": "ok"},
            {
                "delegation_depth": 1,
                "origin_agent_id": "agent.origin",
                "parent_execution_id": "parent-1",
                "reason_code": "handoff",
                "source": "test",
            },
        )
    ]
    assert result.safe_output == {
        "agent_id": "agent.target",
        "shared": {"trace": "ok"},
        "structured": {"value": 1},
    }


def test_automatic_resolution_uses_agent_resolver() -> None:
    system = _system()

    result = system.agent_delegation_service.delegate(_request(target_agent_id=None))

    assert result.status is AgentDelegationStatus.SUCCESS
    assert result.resolution_result is not None
    assert result.resolution_result.target_agent_id == "agent.target"
    assert result.resolution_result.resolution_result is not None


def test_origin_and_target_existence_are_validated() -> None:
    missing_origin = _system(definitions=(_definition("agent.target"),)).agent_delegation_service.delegate(_request())
    missing_target = _system(definitions=(_definition("agent.origin"),), handlers=()).agent_delegation_service.delegate(
        _request()
    )

    assert missing_origin.status is AgentDelegationStatus.ORIGIN_AGENT_NOT_FOUND
    assert missing_target.status is AgentDelegationStatus.TARGET_AGENT_NOT_FOUND


def test_disabled_origin_and_target_are_rejected_when_policy_requires_enabled() -> None:
    disabled_origin = _system(
        definitions=(_definition("agent.origin", enabled=False), _definition("agent.target"))
    ).agent_delegation_service.delegate(_request())
    disabled_target = _system(
        definitions=(_definition("agent.origin"), _definition("agent.target", enabled=False))
    ).agent_delegation_service.delegate(_request())

    assert disabled_origin.status is AgentDelegationStatus.ORIGIN_AGENT_DISABLED
    assert disabled_target.status is AgentDelegationStatus.TARGET_AGENT_DISABLED


def test_self_delegation_rejected_and_allowed_by_policy() -> None:
    rejected = _system(
        definitions=(_definition("agent.origin"),),
        handlers=(RecordingHandler("agent.origin"),),
    ).agent_delegation_service.delegate(_request(target_agent_id="agent.origin"))
    allowed = _system(
        definitions=(_definition("agent.origin"),),
        handlers=(RecordingHandler("agent.origin"),),
    ).agent_delegation_service.delegate(
        _request(target_agent_id="agent.origin", policy=_policy(allow_self_delegation=True))
    )

    assert rejected.status is AgentDelegationStatus.SELF_DELEGATION_DENIED
    assert allowed.status is AgentDelegationStatus.SUCCESS
    assert RecordingHandler.calls[-1][0] == "agent.origin"


def test_capabilities_permissions_and_type_are_enforced() -> None:
    missing_capability = _system(
        definitions=(_definition("agent.origin"), _definition("agent.target", capabilities=("other.capability",)))
    ).agent_delegation_service.delegate(_request())
    missing_permission = _system().agent_delegation_service.delegate(
        _request(required_permission_ids=("can_execute_tools",))
    )
    wrong_type = _system().agent_delegation_service.delegate(
        _request(required_agent_types=(AgentType.CODING,))
    )

    assert missing_capability.status is AgentDelegationStatus.MISSING_CAPABILITIES
    assert missing_permission.status is AgentDelegationStatus.MISSING_PERMISSIONS
    assert wrong_type.status is AgentDelegationStatus.TYPE_INCOMPATIBLE


def test_policy_allowed_and_denied_targets_are_enforced() -> None:
    not_allowed = _system().agent_delegation_service.delegate(
        _request(policy=_policy(allowed_target_agent_ids=("agent.other",)))
    )
    denied = _system().agent_delegation_service.delegate(
        _request(policy=_policy(denied_target_agent_ids=("agent.target",)))
    )

    assert not_allowed.status is AgentDelegationStatus.TARGET_NOT_ALLOWED
    assert denied.status is AgentDelegationStatus.TARGET_DENIED


def test_depth_total_delegations_and_invalid_path_are_rejected() -> None:
    max_depth = _system().agent_delegation_service.delegate(
        _request(
            delegation_path=("agent.a", "agent.origin"),
            delegation_depth=1,
            policy=_policy(max_delegation_depth=1),
        )
    )
    max_total = _system().agent_delegation_service.delegate(
        _request(
            delegation_path=("agent.a", "agent.origin"),
            delegation_depth=1,
            policy=_policy(max_total_delegations=1),
        )
    )

    assert max_depth.status is AgentDelegationStatus.MAX_DEPTH_REACHED
    assert max_total.status is AgentDelegationStatus.MAX_DELEGATIONS_REACHED
    with pytest.raises(InvalidAgentDelegationRequestError):
        AgentDelegationRequest(
            origin_agent_id="agent.origin",
            delegation_path=("agent.origin", "agent.other"),
            delegation_depth=1,
            policy=_policy(),
        )


def test_shared_context_and_metadata_propagation_are_explicit() -> None:
    blocked = _system().agent_delegation_service.delegate(
        _request(policy=_policy(propagate_shared_context=False, propagate_metadata=False))
    )
    allowed = _system().agent_delegation_service.delegate(
        _request(policy=_policy(propagate_shared_context=True, propagate_metadata=True))
    )

    assert blocked.status is AgentDelegationStatus.SUCCESS
    assert RecordingHandler.calls[0][2] == {}
    assert "source" not in RecordingHandler.calls[0][3]
    assert allowed.status is AgentDelegationStatus.SUCCESS
    assert RecordingHandler.calls[1][2] == {"trace": "ok"}
    assert RecordingHandler.calls[1][3]["source"] == "test"


def test_sensitive_metadata_and_unsafe_objects_are_rejected() -> None:
    with pytest.raises(InvalidAgentDelegationRequestError) as sensitive:
        _request(metadata={"api_key": "raw-secret"})
    with pytest.raises(InvalidAgentDelegationRequestError):
        _request(structured_input={"callback": lambda: None})
    with pytest.raises(InvalidAgentDelegationRequestError):
        _request(structured_input={"bad": float("nan")})

    assert "raw-secret" not in str(sensitive.value)


def test_resolver_failure_and_ambiguous_selection_are_structured() -> None:
    registry = AgentRegistry((_definition("agent.origin"), _definition("agent.target")))
    resolver = FailingResolver(registry)
    context_builder = AgentContextBuilder()
    executor = AgentExecutor(resolver, context_builder, _system().agent_handler_registry)
    service = build_core_agent_delegation_service(
        agent_registry=registry,
        agent_resolver=resolver,
        agent_context_builder=context_builder,
        agent_executor=executor,
    )
    failed = service.delegate(_request(target_agent_id=None))

    ambiguous_system = _system(
        definitions=(
            _definition("agent.origin"),
            _definition("agent.a"),
            _definition("agent.b"),
        ),
        handlers=(RecordingHandler("agent.a"), RecordingHandler("agent.b")),
    )
    ambiguous = ambiguous_system.agent_delegation_service.delegate(
        _request(target_agent_id=None, required_capability_ids=("agent.inspect",))
    )

    assert failed.status is AgentDelegationStatus.INTERNAL_ERROR
    assert failed.error_message == "[redacted]"
    assert ambiguous.status is AgentDelegationStatus.AMBIGUOUS_AGENT_SELECTION


def test_context_builder_failure_maps_to_context_rejected() -> None:
    system = _system(
        definitions=(
            _definition("agent.origin"),
            _definition("agent.target", context_policy=AgentContextPolicy(max_mapping_items=1)),
        )
    )

    result = system.agent_delegation_service.delegate(
        _request(structured_input={"a": 1, "b": 2})
    )

    assert result.status is AgentDelegationStatus.CONTEXT_REJECTED
    assert result.agent_execution_result is not None
    assert result.agent_execution_result.status is AgentExecutionStatus.CONTEXT_BUILD_FAILED


def test_executor_failure_is_sanitized_and_structured() -> None:
    system = _system()
    service = AgentDelegationService(
        agent_registry=system.agent_registry,
        agent_resolver=system.agent_resolver,
        agent_context_builder=system.agent_context_builder,
        agent_executor=FailingExecutor(
            system.agent_resolver,
            system.agent_context_builder,
            system.agent_handler_registry,
        ),
    )

    result = service.delegate(_request())

    assert result.status is AgentDelegationStatus.EXECUTION_FAILED
    assert result.error_code == "CONTROLLED_FAILURE"
    assert result.error_message == "[redacted]"


def test_success_result_events_metrics_and_signature_are_stable() -> None:
    request = _request()
    result = _system().agent_delegation_service.delegate(request)
    repeat = agent_delegation_request_signature(_request())

    assert result.request_signature == repeat
    assert len(result.request_signature) == 64
    assert tuple(event.name for event in result.events) == (
        "agent_delegation_requested",
        "agent_delegation_validation_started",
        "agent_delegation_origin_validated",
        "agent_delegation_target_resolution_started",
        "agent_delegation_target_resolved",
        "agent_delegation_execution_started",
        "agent_delegation_execution_succeeded",
        "agent_delegation_completed",
    )
    assert result.metrics["agent_delegations_requested"] == 1
    assert result.metrics["agent_delegations_succeeded"] == 1
    assert result.metrics["agent_delegations_failed"] == 0


def test_policy_disabled_by_default_and_agent_system_integration() -> None:
    system = _system()
    disabled = system.agent_delegation_service.delegate(
        AgentDelegationRequest(origin_agent_id="agent.origin", target_agent_id="agent.target")
    )

    assert isinstance(system.agent_delegation_service, AgentDelegationService)
    assert disabled.status is AgentDelegationStatus.DISABLED
    assert RecordingHandler.calls == []


def test_existing_agent_executor_remains_compatible() -> None:
    system = _system()
    execution = system.agent_executor.execute(
        AgentExecutionRequest(
            resolution_request=AgentResolutionRequest(required_agent_ids=("agent.target",)),
            structured_input={"direct": True},
        )
    )

    assert execution.status is AgentExecutionStatus.COMPLETED
    assert execution.output == {"agent_id": "agent.target", "shared": {}, "structured": {"direct": True}}
