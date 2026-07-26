from __future__ import annotations

from collections.abc import Mapping
from dataclasses import FrozenInstanceError

import pytest

from bootstrap.agent_cooperation_automatic_execution import (
    build_core_agent_cooperation_automatic_execution_service,
)
from bootstrap.agent_system import build_core_agent_system
from core.agent_context import AgentContext
from core.agent_cooperation_automatic_execution import (
    AgentCooperationAutomaticExecutionDecisionType,
    AgentCooperationAutomaticExecutionPolicy,
    AgentCooperationAutomaticExecutionRequest,
    AgentCooperationAutomaticExecutionService,
    AgentCooperationAutomaticExecutionStatus,
    InvalidAgentCooperationAutomaticExecutionRequestError,
    agent_cooperation_automatic_execution_request_signature,
)
from core.agent_cooperation_automatic_planner import (
    AgentCooperationObjectiveType,
    AgentCooperationPlanningPolicy,
    AgentCooperationPlanningRequest,
    AgentCooperationPlanningStatus,
    AgentCooperationPlanningTaskRequirement,
)
from core.agent_cooperation_plan import (
    AgentCooperationFailureMode,
    AgentCooperationPlan,
    AgentCooperationPlanPolicy,
    AgentCooperationPlanStatus,
)
from core.agent_registry import (
    AgentCapabilities,
    AgentContextPolicy,
    AgentDefinition,
    AgentPermissions,
    AgentType,
)


class LocalHandler:
    calls: list[tuple[str, Mapping[str, object]]] = []
    failing_agent_ids: set[str] = set()
    include_secret = False

    def __init__(self, agent_id: str) -> None:
        self._agent_id = agent_id

    @property
    def agent_id(self) -> str:
        return self._agent_id

    def handle(self, context: AgentContext) -> Mapping[str, object]:
        LocalHandler.calls.append((context.agent_id, context.structured_input))
        if context.agent_id in LocalHandler.failing_agent_ids:
            raise RuntimeError("controlled handler failure")
        output = {
            "agent_id": context.agent_id,
            "received": dict(context.structured_input),
        }
        if LocalHandler.include_secret:
            output["api_key"] = "must-not-escape"
        return output


def _agent(
    agent_id: str,
    *,
    capabilities: tuple[str, ...] = ("cap.test",),
) -> AgentDefinition:
    return AgentDefinition(
        agent_id=agent_id,
        agent_type=AgentType.GENERAL,
        name=agent_id,
        description="Controlled automatic execution test agent.",
        capabilities=AgentCapabilities(capabilities=capabilities),
        permissions=AgentPermissions(requires_confirmation=False),
        context_policy=AgentContextPolicy(allow_shared_context=True),
    )


def _system(
    agents: tuple[AgentDefinition, ...] = (_agent("agent.a"),),
):
    result = build_core_agent_system()
    assert result.system is not None
    system = result.system
    for agent in agents:
        system.agent_registry.register(agent)
        system.agent_handler_registry.register(LocalHandler(agent.agent_id))
    return system


def _planning_policy(**overrides: object) -> AgentCooperationPlanningPolicy:
    values = {
        "enabled": True,
        "max_agents": 8,
        "max_tasks": 16,
        "max_dependencies": 48,
        "max_plan_depth": 8,
    }
    values.update(overrides)
    return AgentCooperationPlanningPolicy(**values)


def _execution_policy(**overrides: object) -> AgentCooperationPlanPolicy:
    values = {
        "enabled": True,
        "max_tasks": 16,
        "max_dependencies": 48,
        "max_depth": 8,
        "max_total_executions": 16,
        "max_output_items": 256,
    }
    values.update(overrides)
    return AgentCooperationPlanPolicy(**values)


def _policy(**overrides: object) -> AgentCooperationAutomaticExecutionPolicy:
    values = {
        "enabled": True,
        "planning_policy": _planning_policy(),
        "execution_policy": _execution_policy(),
        "execute_generated_plan": True,
    }
    values.update(overrides)
    return AgentCooperationAutomaticExecutionPolicy(**values)


def _planning_request(**overrides: object) -> AgentCooperationPlanningRequest:
    values = {
        "objective_id": "objective.test",
        "objective_type": AgentCooperationObjectiveType.ANALYSIS,
        "required_agent_ids": ("agent.a",),
        "structured_input": {"value": 7},
        "shared_context": {"scope": "test"},
        "execution_id": "planning.exec",
    }
    values.update(overrides)
    return AgentCooperationPlanningRequest(**values)


def _request(**overrides: object) -> AgentCooperationAutomaticExecutionRequest:
    values = {
        "planning_request": _planning_request(),
        "policy": _policy(),
        "execution_id": "automatic.exec",
        "correlation_id": "correlation.test",
        "metadata": {"source": "tests"},
    }
    values.update(overrides)
    return AgentCooperationAutomaticExecutionRequest(**values)


def setup_function() -> None:
    LocalHandler.calls = []
    LocalHandler.failing_agent_ids = set()
    LocalHandler.include_secret = False


def test_policy_is_disabled_by_default_without_planning(monkeypatch) -> None:
    system = _system()
    calls = 0

    def plan(_request):
        nonlocal calls
        calls += 1
        raise AssertionError("planner must not run")

    monkeypatch.setattr(system.agent_cooperation_automatic_planner, "plan", plan)
    result = system.agent_cooperation_automatic_execution_service.execute(
        AgentCooperationAutomaticExecutionRequest(_planning_request())
    )

    assert result.status is AgentCooperationAutomaticExecutionStatus.DISABLED
    assert result.decision.decision is AgentCooperationAutomaticExecutionDecisionType.PLANNING_BLOCKED
    assert calls == 0
    assert LocalHandler.calls == []


def test_invalid_request_is_structured_and_models_reject_invalid_values() -> None:
    service = _system().agent_cooperation_automatic_execution_service

    result = service.execute(object())

    assert result.status is AgentCooperationAutomaticExecutionStatus.INVALID_REQUEST
    assert result.error_code == "INVALID_REQUEST"
    with pytest.raises(InvalidAgentCooperationAutomaticExecutionRequestError):
        AgentCooperationAutomaticExecutionPolicy(max_tasks=0)
    with pytest.raises(InvalidAgentCooperationAutomaticExecutionRequestError):
        AgentCooperationAutomaticExecutionRequest(
            _planning_request(),
            metadata={"api_key": "not-accepted"},
        )


@pytest.mark.parametrize(
    ("policy", "error_code"),
    (
        (
            AgentCooperationAutomaticExecutionPolicy(
                enabled=True,
                execution_policy=_execution_policy(),
            ),
            "MISSING_PLANNING_POLICY",
        ),
        (
            AgentCooperationAutomaticExecutionPolicy(
                enabled=True,
                planning_policy=_planning_policy(),
            ),
            "MISSING_EXECUTION_POLICY",
        ),
        (
            _policy(execute_generated_plan=False),
            "EXECUTION_NOT_AUTHORIZED",
        ),
        (
            _policy(execution_policy=_execution_policy(enabled=False)),
            "EXECUTION_POLICY_DISABLED",
        ),
    ),
)
def test_required_policy_gates_block_before_planning(
    policy: AgentCooperationAutomaticExecutionPolicy,
    error_code: str,
) -> None:
    result = _system().agent_cooperation_automatic_execution_service.execute(
        _request(policy=policy)
    )

    assert result.status is AgentCooperationAutomaticExecutionStatus.INVALID_REQUEST
    assert result.error_code == error_code
    assert LocalHandler.calls == []


def test_dry_run_plans_and_validates_without_execution(monkeypatch) -> None:
    system = _system()
    execute_calls = 0
    original = system.agent_cooperation_planner.execute

    def execute(request):
        nonlocal execute_calls
        execute_calls += 1
        return original(request)

    monkeypatch.setattr(system.agent_cooperation_planner, "execute", execute)
    result = system.agent_cooperation_automatic_execution_service.execute(
        _request(
            policy=_policy(
                dry_run=True,
                execute_generated_plan=False,
                execution_policy=_execution_policy(enabled=False),
            )
        )
    )

    assert result.status is AgentCooperationAutomaticExecutionStatus.DRY_RUN_COMPLETED
    assert result.generated_plan is not None
    assert result.cooperation_result is None
    assert result.metrics["dry_runs"] == 1
    assert execute_calls == 0
    assert LocalHandler.calls == []


def test_successful_planning_and_execution_propagates_safe_input() -> None:
    result = _system().agent_cooperation_automatic_execution_service.execute(_request())

    assert result.status is AgentCooperationAutomaticExecutionStatus.EXECUTION_SUCCEEDED
    assert result.planning_result is not None
    assert result.planning_result.status is AgentCooperationPlanningStatus.SUCCESS
    assert result.cooperation_result is not None
    assert result.cooperation_result.status is AgentCooperationPlanStatus.SUCCESS
    assert result.plan_signature == result.cooperation_result.plan_signature
    assert LocalHandler.calls == [("agent.a", {"value": 7})]
    assert result.metrics["tasks_executed"] == 1
    assert result.metrics["agents_executed"] == 1
    assert tuple(item.decision for item in result.decisions) == (
        AgentCooperationAutomaticExecutionDecisionType.PLANNING_ALLOWED,
        AgentCooperationAutomaticExecutionDecisionType.PLAN_ACCEPTED,
        AgentCooperationAutomaticExecutionDecisionType.EXECUTION_ALLOWED,
        AgentCooperationAutomaticExecutionDecisionType.EXECUTION_COMPLETED,
    )


def test_planning_ambiguous_never_executes() -> None:
    system = _system((_agent("agent.a"), _agent("agent.b")))
    result = system.agent_cooperation_automatic_execution_service.execute(
        _request(
            planning_request=_planning_request(
                required_agent_ids=(),
                required_agent_types=(AgentType.GENERAL,),
            )
        )
    )

    assert result.status is AgentCooperationAutomaticExecutionStatus.PLANNING_AMBIGUOUS
    assert result.cooperation_result is None
    assert LocalHandler.calls == []


def test_missing_coverage_never_executes() -> None:
    result = _system().agent_cooperation_automatic_execution_service.execute(
        _request(
            planning_request=_planning_request(
                required_agent_ids=(),
                required_capability_ids=("cap.missing",),
            )
        )
    )

    assert result.status is AgentCooperationAutomaticExecutionStatus.PLANNING_FAILED
    assert result.planning_result is not None
    assert result.planning_result.status is AgentCooperationPlanningStatus.MISSING_CAPABILITY
    assert LocalHandler.calls == []


def test_generated_plan_validation_failure_stops_execution(monkeypatch) -> None:
    system = _system()
    monkeypatch.setattr(
        system.agent_cooperation_planner,
        "_validate",
        lambda plan, policy: (
            AgentCooperationPlanStatus.INVALID_PLAN,
            "FORCED_INVALID",
            "forced validation failure",
        ),
    )

    result = system.agent_cooperation_automatic_execution_service.execute(_request())

    assert result.status is AgentCooperationAutomaticExecutionStatus.PLAN_VALIDATION_FAILED
    assert result.error_code == "FORCED_INVALID"
    assert LocalHandler.calls == []


def test_empty_plan_from_planner_is_rejected(monkeypatch) -> None:
    system = _system()
    original = system.agent_cooperation_automatic_planner.plan

    def plan(request):
        result = original(request)
        empty_plan = object.__new__(AgentCooperationPlan)
        object.__setattr__(empty_plan, "tasks", ())
        object.__setattr__(result, "plan", empty_plan)
        return result

    monkeypatch.setattr(system.agent_cooperation_automatic_planner, "plan", plan)
    result = system.agent_cooperation_automatic_execution_service.execute(_request())

    assert result.status is AgentCooperationAutomaticExecutionStatus.NO_VALID_PLAN
    assert result.error_code == "NO_VALID_PLAN"
    assert LocalHandler.calls == []


def test_manipulated_plan_signature_is_rejected(monkeypatch) -> None:
    system = _system()
    original = system.agent_cooperation_automatic_planner.plan

    def plan(request):
        result = original(request)
        object.__setattr__(result, "plan_signature", "0" * 64)
        return result

    monkeypatch.setattr(system.agent_cooperation_automatic_planner, "plan", plan)
    result = system.agent_cooperation_automatic_execution_service.execute(_request())

    assert result.status is AgentCooperationAutomaticExecutionStatus.NO_VALID_PLAN
    assert result.error_code == "PLAN_SIGNATURE_MISMATCH"
    assert LocalHandler.calls == []


def test_missing_plan_signature_is_rejected(monkeypatch) -> None:
    system = _system()
    original = system.agent_cooperation_automatic_planner.plan

    def plan(request):
        result = original(request)
        object.__setattr__(result, "plan_signature", None)
        return result

    monkeypatch.setattr(system.agent_cooperation_automatic_planner, "plan", plan)
    result = system.agent_cooperation_automatic_execution_service.execute(_request())

    assert result.status is AgentCooperationAutomaticExecutionStatus.NO_VALID_PLAN
    assert result.error_code == "MISSING_PLAN_SIGNATURE"
    assert LocalHandler.calls == []


def test_execution_partial_is_not_converted_to_success() -> None:
    LocalHandler.failing_agent_ids = {"agent.b"}
    system = _system((_agent("agent.a"), _agent("agent.b")))
    planning_request = _planning_request(
        required_agent_ids=(),
        task_requirements=(
            AgentCooperationPlanningTaskRequirement(
                "task.a",
                required_agent_ids=("agent.a",),
                inherit_objective_requirements=False,
            ),
            AgentCooperationPlanningTaskRequirement(
                "task.b",
                required_agent_ids=("agent.b",),
                inherit_objective_requirements=False,
                order=1,
            ),
        ),
    )
    execution_policy = _execution_policy(
        failure_mode=AgentCooperationFailureMode.CONTINUE_INDEPENDENT_TASKS,
        allow_partial_success=True,
    )

    result = system.agent_cooperation_automatic_execution_service.execute(
        _request(
            planning_request=planning_request,
            policy=_policy(execution_policy=execution_policy),
        )
    )

    assert result.status is AgentCooperationAutomaticExecutionStatus.EXECUTION_PARTIAL
    assert result.metrics["executions_partial"] == 1
    assert result.metrics["tasks_failed"] == 1


def test_execution_failure_is_not_converted_to_success() -> None:
    LocalHandler.failing_agent_ids = {"agent.a"}

    result = _system().agent_cooperation_automatic_execution_service.execute(_request())

    assert result.status is AgentCooperationAutomaticExecutionStatus.EXECUTION_FAILED
    assert result.metrics["executions_failed"] == 1
    assert result.error_code is not None


def test_outer_limit_is_enforced_before_execution() -> None:
    system = _system((_agent("agent.a"), _agent("agent.b")))
    planning_request = _planning_request(
        required_agent_ids=(),
        task_requirements=(
            AgentCooperationPlanningTaskRequirement(
                "task.a",
                required_agent_ids=("agent.a",),
                inherit_objective_requirements=False,
            ),
            AgentCooperationPlanningTaskRequirement(
                "task.b",
                required_agent_ids=("agent.b",),
                inherit_objective_requirements=False,
            ),
        ),
    )
    result = system.agent_cooperation_automatic_execution_service.execute(
        _request(
            planning_request=planning_request,
            policy=_policy(
                max_agents=1,
                planning_policy=_planning_policy(max_agents=1),
            ),
        )
    )

    assert result.status is AgentCooperationAutomaticExecutionStatus.LIMIT_REACHED
    assert result.error_code == "MAX_AGENTS"
    assert result.metrics["limits_reached"] == 1
    assert LocalHandler.calls == []


def test_planner_exception_is_sanitized() -> None:
    system = _system()

    def plan(_request):
        raise RuntimeError("api_key=super-sensitive")

    system.agent_cooperation_automatic_planner.plan = plan
    result = system.agent_cooperation_automatic_execution_service.execute(_request())

    assert result.status is AgentCooperationAutomaticExecutionStatus.INTERNAL_ERROR
    assert result.reason == "automatic cooperation execution failed."
    assert "super-sensitive" not in result.reason


def test_planner_and_executor_are_called_exactly_once(monkeypatch) -> None:
    system = _system()
    planning_calls = 0
    execution_calls = 0
    original_plan = system.agent_cooperation_automatic_planner.plan
    original_execute = system.agent_cooperation_planner.execute

    def plan(request):
        nonlocal planning_calls
        planning_calls += 1
        return original_plan(request)

    def execute(request):
        nonlocal execution_calls
        execution_calls += 1
        return original_execute(request)

    monkeypatch.setattr(system.agent_cooperation_automatic_planner, "plan", plan)
    monkeypatch.setattr(system.agent_cooperation_planner, "execute", execute)

    result = system.agent_cooperation_automatic_execution_service.execute(_request())

    assert result.status is AgentCooperationAutomaticExecutionStatus.EXECUTION_SUCCEEDED
    assert planning_calls == 1
    assert execution_calls == 1


def test_events_metrics_and_signature_are_deterministic() -> None:
    system = _system()
    first_request = _request()
    same_request = _request()
    changed_request = _request(execution_id="automatic.changed")

    first = system.agent_cooperation_automatic_execution_service.execute(first_request)

    assert first.signature == agent_cooperation_automatic_execution_request_signature(
        same_request
    )
    assert first.signature != agent_cooperation_automatic_execution_request_signature(
        changed_request
    )
    assert len(first.signature) == 64
    names = {event.name for event in first.events}
    assert "cooperation_auto_execution_requested" in names
    assert "cooperation_auto_planning_succeeded" in names
    assert "cooperation_generated_plan_validation_succeeded" in names
    assert "cooperation_auto_execution_succeeded" in names
    assert "cooperation_auto_execution_completed" in names
    assert all(type(value) is int and value >= 0 for value in first.metrics.values())


def test_models_are_frozen_and_safe_summary_contains_no_payload() -> None:
    result = _system().agent_cooperation_automatic_execution_service.execute(_request())

    with pytest.raises(FrozenInstanceError):
        result.status = AgentCooperationAutomaticExecutionStatus.DISABLED
    assert "received" not in result.safe_summary
    assert "value" not in result.safe_summary


def test_nested_execution_results_and_sensitive_outputs_are_removed() -> None:
    LocalHandler.include_secret = True

    result = _system().agent_cooperation_automatic_execution_service.execute(_request())

    assert result.cooperation_result is not None
    task_result = result.cooperation_result.task_results[0]
    assert task_result.execution_result is None
    assert task_result.output is not None
    assert "api_key" not in task_result.output
    assert "api_key" not in result.cooperation_result.outputs["objective.test"]


def test_agent_system_reuses_all_shared_execution_dependencies() -> None:
    system = _system()
    service = system.agent_cooperation_automatic_execution_service

    assert service._agent_registry is system.agent_registry
    assert service._agent_resolver is system.agent_resolver
    assert service._agent_context_builder is system.agent_context_builder
    assert service._agent_executor is system.agent_executor
    assert service._skill_system is system.skill_system
    assert service._agent_delegation_service is system.agent_delegation_service
    assert service._agent_delegation_chain_service is system.agent_delegation_chain_service
    assert service._agent_delegation_coordinator is system.agent_delegation_coordinator
    assert service._multi_agent_resolver is system.multi_agent_resolver
    assert service._multi_agent_coordinator is system.multi_agent_coordinator
    assert service._agent_cooperation_planner is system.agent_cooperation_planner
    assert (
        service._agent_cooperation_automatic_planner
        is system.agent_cooperation_automatic_planner
    )


def test_bootstrap_factory_uses_injected_instances_without_execution() -> None:
    system = _system()

    service = build_core_agent_cooperation_automatic_execution_service(
        agent_registry=system.agent_registry,
        agent_resolver=system.agent_resolver,
        agent_context_builder=system.agent_context_builder,
        agent_executor=system.agent_executor,
        skill_system=system.skill_system,
        agent_delegation_service=system.agent_delegation_service,
        agent_delegation_chain_service=system.agent_delegation_chain_service,
        agent_delegation_coordinator=system.agent_delegation_coordinator,
        multi_agent_resolver=system.multi_agent_resolver,
        multi_agent_coordinator=system.multi_agent_coordinator,
        agent_cooperation_planner=system.agent_cooperation_planner,
        agent_cooperation_automatic_planner=system.agent_cooperation_automatic_planner,
    )

    assert isinstance(service, AgentCooperationAutomaticExecutionService)
    assert LocalHandler.calls == []
