from __future__ import annotations

from collections.abc import Mapping

from bootstrap.agent_system import build_core_agent_system
from core.agent_context import AgentContext
from core.agent_executor import AgentExecutionStatus
from core.agent_registry import AgentCapabilities, AgentContextPolicy, AgentDefinition, AgentPermissions, AgentType
from core.multi_agent import (
    MultiAgentExecutionPolicy,
    MultiAgentExecutionRequest,
    MultiAgentExecutionStatus,
    MultiAgentFailurePolicy,
)


class RecordingHandler:
    calls: list[str] = []

    def __init__(self, agent_id: str, *, fail: bool = False) -> None:
        self._agent_id = agent_id
        self.fail = fail

    @property
    def agent_id(self) -> str:
        return self._agent_id

    def handle(self, context: AgentContext) -> Mapping[str, object]:
        RecordingHandler.calls.append(self._agent_id)
        if self.fail:
            raise RuntimeError("authorization token leaked")
        return {
            "agent_id": self._agent_id,
            "input": context.structured_input,
            "shared": context.shared_context,
            "token": "hidden",
        }


def _definition(
    agent_id: str,
    *,
    capabilities: tuple[str, ...] = ("agent.inspect",),
    context_policy: AgentContextPolicy | None = None,
) -> AgentDefinition:
    return AgentDefinition(
        agent_id=agent_id,
        agent_type=AgentType.GENERAL,
        name=agent_id,
        description="Coordinator test agent.",
        capabilities=AgentCapabilities(capabilities=capabilities),
        permissions=AgentPermissions(requires_confirmation=False),
        context_policy=context_policy or AgentContextPolicy(allow_shared_context=True),
    )


def _system(definitions: tuple[AgentDefinition, ...], handlers: tuple[RecordingHandler, ...]):
    result = build_core_agent_system()
    assert result.system is not None
    system = result.system
    for definition in definitions:
        system.agent_registry.register(definition)
    for handler in handlers:
        system.agent_handler_registry.register(handler)
    return system


def setup_function() -> None:
    RecordingHandler.calls = []


def test_executes_agents_sequentially_once_through_existing_executor() -> None:
    system = _system(
        (_definition("agent.a"), _definition("agent.b")),
        (RecordingHandler("agent.a"), RecordingHandler("agent.b")),
    )

    result = system.multi_agent_coordinator.execute(
        MultiAgentExecutionRequest(
            required_agent_ids=("agent.a", "agent.b"),
            payload={"value": "ok"},
            shared_context={"trace": "safe"},
            policy=MultiAgentExecutionPolicy(min_agents=2, max_agents=2),
        )
    )

    assert result.status is MultiAgentExecutionStatus.SUCCESS
    assert RecordingHandler.calls == ["agent.a", "agent.b"]
    assert [step.status for step in result.step_results] == [AgentExecutionStatus.COMPLETED, AgentExecutionStatus.COMPLETED]
    assert result.plan is not None
    assert [step.agent_id for step in result.plan.steps] == ["agent.a", "agent.b"]


def test_stop_on_first_failure_and_continue_on_failure() -> None:
    stop_system = _system(
        (_definition("agent.a"), _definition("agent.b")),
        (RecordingHandler("agent.a", fail=True), RecordingHandler("agent.b")),
    )
    stop = stop_system.multi_agent_coordinator.execute(
        MultiAgentExecutionRequest(
            required_agent_ids=("agent.a", "agent.b"),
            policy=MultiAgentExecutionPolicy(min_agents=2, max_agents=2),
        )
    )
    assert stop.status is MultiAgentExecutionStatus.FAILED
    assert RecordingHandler.calls == ["agent.a"]

    RecordingHandler.calls = []
    continue_system = _system(
        (_definition("agent.a"), _definition("agent.b")),
        (RecordingHandler("agent.a", fail=True), RecordingHandler("agent.b")),
    )
    continued = continue_system.multi_agent_coordinator.execute(
        MultiAgentExecutionRequest(
            required_agent_ids=("agent.a", "agent.b"),
            policy=MultiAgentExecutionPolicy(
                min_agents=2,
                max_agents=2,
                failure_policy=MultiAgentFailurePolicy.CONTINUE_ON_FAILURE,
            ),
        )
    )

    assert continued.status is MultiAgentExecutionStatus.PARTIAL_SUCCESS
    assert RecordingHandler.calls == ["agent.a", "agent.b"]


def test_aggregation_is_structured_and_sanitizes_outputs() -> None:
    system = _system(
        (_definition("agent.a"), _definition("agent.b")),
        (RecordingHandler("agent.a"), RecordingHandler("agent.b")),
    )

    result = system.multi_agent_coordinator.execute(
        MultiAgentExecutionRequest(
            required_agent_ids=("agent.a", "agent.b"),
            policy=MultiAgentExecutionPolicy(min_agents=2, max_agents=2),
        )
    )

    assert result.output is not None
    assert result.output["team"] == ("agent.a", "agent.b")
    assert result.output["summary"]["completed_count"] == 2  # type: ignore[index]
    assert "token" not in repr(result.output)
    assert result.aggregation_result is not None


def test_shared_context_authorized_and_blocked_by_agent_policy() -> None:
    authorized = _system(
        (_definition("agent.a"), _definition("agent.b")),
        (RecordingHandler("agent.a"), RecordingHandler("agent.b")),
    ).multi_agent_coordinator.execute(
        MultiAgentExecutionRequest(
            required_agent_ids=("agent.a", "agent.b"),
            shared_context={"shared": "yes"},
            policy=MultiAgentExecutionPolicy(min_agents=2, max_agents=2),
        )
    )
    blocked = _system(
        (
            _definition("agent.a", context_policy=AgentContextPolicy(allow_shared_context=False)),
            _definition("agent.b"),
        ),
        (RecordingHandler("agent.a"), RecordingHandler("agent.b")),
    ).multi_agent_coordinator.execute(
        MultiAgentExecutionRequest(
            required_agent_ids=("agent.a", "agent.b"),
            shared_context={"shared": "yes"},
            policy=MultiAgentExecutionPolicy(min_agents=2, max_agents=2),
        )
    )

    assert authorized.status is MultiAgentExecutionStatus.SUCCESS
    assert blocked.status is MultiAgentExecutionStatus.SUCCESS
    first_output = blocked.step_results[0].output
    assert first_output is not None
    assert first_output["shared"] == {}


def test_missing_handler_and_resolution_failures_are_structured() -> None:
    missing_handler = _system((_definition("agent.a"), _definition("agent.b")), (RecordingHandler("agent.a"),)).multi_agent_coordinator.execute(
        MultiAgentExecutionRequest(
            required_agent_ids=("agent.a", "agent.b"),
            policy=MultiAgentExecutionPolicy(min_agents=2, max_agents=2),
        )
    )
    no_team = _system((_definition("agent.a"),), (RecordingHandler("agent.a"),)).multi_agent_coordinator.execute(
        MultiAgentExecutionRequest(
            required_capability_ids=("missing.capability",),
            policy=MultiAgentExecutionPolicy(min_agents=2, max_agents=2),
        )
    )

    assert missing_handler.status is MultiAgentExecutionStatus.AGENT_EXECUTION_FAILED
    assert missing_handler.step_results[-1].status is AgentExecutionStatus.HANDLER_UNAVAILABLE
    assert no_team.status is MultiAgentExecutionStatus.NO_MATCHING_TEAM


def test_events_metrics_and_secret_sanitization() -> None:
    result = _system(
        (_definition("agent.a"), _definition("agent.b")),
        (RecordingHandler("agent.a"), RecordingHandler("agent.b")),
    ).multi_agent_coordinator.execute(
        MultiAgentExecutionRequest(
            required_agent_ids=("agent.a", "agent.b"),
            policy=MultiAgentExecutionPolicy(min_agents=2, max_agents=2),
        )
    )
    event_names = {event.name for event in result.events}

    assert "multi_agent_execution_requested" in event_names
    assert "multi_agent_team_resolved" in event_names
    assert "multi_agent_step_started" in event_names
    assert "multi_agent_aggregation_succeeded" in event_names
    assert result.metrics["multi_agent_steps_started"] == 2
    assert result.metrics["multi_agent_steps_succeeded"] == 2
    assert "authorization" not in repr(result)
    assert "token" not in repr(result)
