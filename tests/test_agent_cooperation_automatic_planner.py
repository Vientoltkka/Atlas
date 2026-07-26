from __future__ import annotations

from collections.abc import Mapping
from dataclasses import FrozenInstanceError
import math

import pytest

from bootstrap.agent_cooperation_automatic_planner import (
    build_core_agent_cooperation_automatic_planner,
)
from bootstrap.agent_system import build_core_agent_system
from core.agent_context import AgentContext
from core.agent_cooperation_automatic_planner import (
    AgentCooperationAutomaticPlanner,
    AgentCooperationObjectiveType,
    AgentCooperationPlanningDecisionStatus,
    AgentCooperationPlanningPolicy,
    AgentCooperationPlanningRequest,
    AgentCooperationPlanningStatus,
    AgentCooperationPlanningTaskRequirement,
    InvalidAgentCooperationPlanningRequestError,
    agent_cooperation_planning_request_signature,
)
from core.agent_cooperation_plan import (
    AgentCooperationExecutionType,
    AgentCooperationOutputBinding,
    AgentCooperationPlan,
)
from core.agent_registry import (
    AgentCapabilities,
    AgentContextPolicy,
    AgentDefinition,
    AgentPermissions,
    AgentType,
)
from core.skill_executor import SkillExecutionRequest, SkillExecutionResult, SkillExecutor
from core.skill_registry import SkillDefinition, SkillExecutionTargetType
from core.skill_system import build_skill_system


class NeverRunHandler:
    calls: list[str] = []

    def __init__(self, agent_id: str) -> None:
        self._agent_id = agent_id

    @property
    def agent_id(self) -> str:
        return self._agent_id

    def handle(self, context: AgentContext) -> Mapping[str, object]:
        NeverRunHandler.calls.append(context.agent_id)
        return {"agent_id": context.agent_id}


class CountingSkillExecutor(SkillExecutor):
    calls = 0

    def execute(self, request: SkillExecutionRequest) -> SkillExecutionResult:
        CountingSkillExecutor.calls += 1
        return super().execute(request)


def _agent(
    agent_id: str,
    *,
    capabilities: tuple[str, ...] = ("cap.default",),
    agent_type: AgentType = AgentType.GENERAL,
    permissions: AgentPermissions | None = None,
    enabled: bool = True,
    metadata: Mapping[str, object] | None = None,
) -> AgentDefinition:
    return AgentDefinition(
        agent_id=agent_id,
        agent_type=agent_type,
        name=agent_id,
        description="Automatic planning test agent.",
        capabilities=AgentCapabilities(capabilities=capabilities),
        permissions=permissions or AgentPermissions(requires_confirmation=False),
        context_policy=AgentContextPolicy(allow_shared_context=True),
        enabled=enabled,
        metadata={} if metadata is None else metadata,
    )


def _skill(skill_id: str = "skill.inspect") -> SkillDefinition:
    return SkillDefinition(
        skill_id=skill_id,
        name=skill_id,
        version="1.0",
        description="Controlled planning skill.",
        required_capability_ids=("cap.inspect",),
        allowed_agent_types=(AgentType.GENERAL,),
        execution_target="handler.inspect",
        execution_target_type=SkillExecutionTargetType.HANDLER,
    )


def _system(
    agents: tuple[AgentDefinition, ...] | None = None,
    *,
    skills: tuple[SkillDefinition, ...] = (),
):
    skill_system = build_skill_system(skill_executor=CountingSkillExecutor())
    built = build_core_agent_system(skill_system=skill_system)
    assert built.system is not None
    system = built.system
    for agent in agents or (_agent("agent.a"),):
        system.agent_registry.register(agent)
        system.agent_handler_registry.register(NeverRunHandler(agent.agent_id))
    for skill in skills:
        system.skill_system.skill_registry.register(skill)
    return system


def _policy(**overrides: object) -> AgentCooperationPlanningPolicy:
    values = {"enabled": True}
    values.update(overrides)
    return AgentCooperationPlanningPolicy(**values)


def _request(**overrides: object) -> AgentCooperationPlanningRequest:
    values = {
        "objective_id": "objective.test",
        "objective_type": AgentCooperationObjectiveType.ANALYSIS,
        "required_agent_ids": ("agent.a",),
        "structured_input": {"value": 1},
        "shared_context": {"trace": "test"},
        "metadata": {"source": "tests"},
        "execution_id": "exec.test",
        "correlation_id": "corr.test",
        "policy": _policy(),
    }
    values.update(overrides)
    return AgentCooperationPlanningRequest(**values)


def setup_function() -> None:
    NeverRunHandler.calls = []
    CountingSkillExecutor.calls = 0


def test_policy_is_disabled_by_default() -> None:
    request = AgentCooperationPlanningRequest(
        "objective.disabled",
        AgentCooperationObjectiveType.CUSTOM,
        required_agent_ids=("agent.a",),
    )

    result = _system().agent_cooperation_automatic_planner.plan(request)

    assert result.status is AgentCooperationPlanningStatus.DISABLED
    assert result.plan is None


def test_invalid_request_is_structured_and_invalid_models_are_rejected() -> None:
    planner = _system().agent_cooperation_automatic_planner

    assert planner.plan(object()).status is AgentCooperationPlanningStatus.INVALID_REQUEST
    with pytest.raises(InvalidAgentCooperationPlanningRequestError):
        AgentCooperationPlanningRequest("objective.bad", "FREE_TEXT")
    with pytest.raises(InvalidAgentCooperationPlanningRequestError):
        AgentCooperationPlanningPolicy(deterministic_ordering=False)


def test_simple_plan_is_deterministic_and_compatible_with_phase_12_4() -> None:
    system = _system()

    first = system.agent_cooperation_automatic_planner.plan(_request())
    second = system.agent_cooperation_automatic_planner.plan(_request())

    assert first.status is AgentCooperationPlanningStatus.SUCCESS
    assert isinstance(first.plan, AgentCooperationPlan)
    assert first.plan_signature == second.plan_signature
    assert first.plan == second.plan
    assert first.plan.policy.enabled is False


def test_selection_by_required_agent_id() -> None:
    system = _system((_agent("agent.a"), _agent("agent.b")))

    result = system.agent_cooperation_automatic_planner.plan(
        _request(required_agent_ids=("agent.b",))
    )

    assert result.selected_agent_ids == ("agent.b",)
    assert result.plan.tasks[0].agent_id == "agent.b"


def test_selection_by_agent_type() -> None:
    system = _system(
        (
            _agent("agent.general"),
            _agent("agent.coding", agent_type=AgentType.CODING),
        )
    )

    result = system.agent_cooperation_automatic_planner.plan(
        _request(required_agent_ids=(), required_agent_types=(AgentType.CODING,))
    )

    assert result.selected_agent_ids == ("agent.coding",)


def test_selection_by_capability_and_permission() -> None:
    system = _system(
        (
            _agent("agent.read", capabilities=("cap.read",)),
            _agent(
                "agent.execute",
                capabilities=("cap.execute",),
                permissions=AgentPermissions(
                    can_execute_tools=True,
                    requires_confirmation=False,
                ),
            ),
        )
    )

    by_capability = system.agent_cooperation_automatic_planner.plan(
        _request(required_agent_ids=(), required_capability_ids=("cap.read",))
    )
    by_permission = system.agent_cooperation_automatic_planner.plan(
        _request(
            required_agent_ids=(),
            required_agent_types=(AgentType.GENERAL,),
            required_permission_ids=("can_execute_tools",),
        )
    )

    assert by_capability.selected_agent_ids == ("agent.read",)
    assert by_permission.selected_agent_ids == ("agent.execute",)


def test_required_skill_must_exist_and_be_authorized() -> None:
    system = _system(
        (
            _agent(
                "agent.a",
                capabilities=("cap.inspect",),
                metadata={"allowed_skill_ids": "skill.inspect"},
            ),
        ),
        skills=(_skill(),),
    )

    result = system.agent_cooperation_automatic_planner.plan(
        _request(required_skill_ids=("skill.inspect",))
    )

    assert result.status is AgentCooperationPlanningStatus.SUCCESS
    assert result.available_skill_ids == ("skill.inspect",)
    assert result.plan.tasks[0].required_skill_ids == ("skill.inspect",)


def test_missing_or_unauthorized_required_skill_fails() -> None:
    missing = _system().agent_cooperation_automatic_planner.plan(
        _request(required_skill_ids=("skill.missing",))
    )
    unauthorized_system = _system(
        (
            _agent(
                "agent.a",
                capabilities=("cap.inspect",),
                metadata={"denied_skill_ids": "skill.inspect"},
            ),
        ),
        skills=(_skill(),),
    )
    unauthorized = unauthorized_system.agent_cooperation_automatic_planner.plan(
        _request(required_skill_ids=("skill.inspect",))
    )

    assert missing.status is AgentCooperationPlanningStatus.MISSING_SKILL
    assert unauthorized.status is AgentCooperationPlanningStatus.MISSING_SKILL


def test_disabled_and_excluded_agents_are_rejected() -> None:
    system = _system(
        (
            _agent("agent.disabled", capabilities=("cap.target",), enabled=False),
            _agent("agent.excluded", capabilities=("cap.target",)),
        )
    )

    result = system.agent_cooperation_automatic_planner.plan(
        _request(
            required_agent_ids=(),
            required_capability_ids=("cap.target",),
            excluded_agent_ids=("agent.excluded",),
        )
    )

    assert result.status is AgentCooperationPlanningStatus.NO_MATCHING_AGENTS
    assert result.rejected_agent_ids == ("agent.disabled", "agent.excluded")


def test_missing_capability_and_permission_are_structured() -> None:
    system = _system((_agent("agent.a", capabilities=("cap.other",)),))

    missing_capability = system.agent_cooperation_automatic_planner.plan(
        _request(required_agent_ids=(), required_capability_ids=("cap.missing",))
    )
    missing_permission = system.agent_cooperation_automatic_planner.plan(
        _request(required_agent_ids=(), required_permission_ids=("can_execute_tools",))
    )

    assert missing_capability.status is AgentCooperationPlanningStatus.MISSING_CAPABILITY
    assert missing_capability.missing_capability_ids == ("cap.missing",)
    assert missing_permission.status is AgentCooperationPlanningStatus.MISSING_PERMISSION


def test_minimal_agent_set_is_selected() -> None:
    system = _system(
        (
            _agent("agent.a", capabilities=("cap.a",)),
            _agent("agent.b", capabilities=("cap.b",)),
            _agent("agent.both", capabilities=("cap.a", "cap.b")),
        )
    )

    result = system.agent_cooperation_automatic_planner.plan(
        _request(
            required_agent_ids=(),
            required_capability_ids=("cap.a", "cap.b"),
        )
    )

    assert result.status is AgentCooperationPlanningStatus.SUCCESS
    assert result.selected_agent_ids == ("agent.both",)


def test_real_tie_is_ambiguous_when_required() -> None:
    system = _system(
        (
            _agent("agent.a", capabilities=("cap.same",)),
            _agent("agent.b", capabilities=("cap.same",)),
        )
    )

    result = system.agent_cooperation_automatic_planner.plan(
        _request(required_agent_ids=(), required_capability_ids=("cap.same",))
    )

    assert result.status is AgentCooperationPlanningStatus.AMBIGUOUS
    assert result.metrics["cooperation_planning_ambiguous"] == 1


def test_registration_order_does_not_change_permitted_tiebreak() -> None:
    agents = (
        _agent("agent.a", capabilities=("cap.same",)),
        _agent("agent.b", capabilities=("cap.same",)),
    )
    request = _request(
        required_agent_ids=(),
        required_capability_ids=("cap.same",),
        policy=_policy(fail_on_ambiguous_agent=False),
    )

    first = _system(agents).agent_cooperation_automatic_planner.plan(request)
    reversed_result = _system(tuple(reversed(agents))).agent_cooperation_automatic_planner.plan(request)

    assert first.selected_agent_ids == ("agent.a",)
    assert reversed_result.selected_agent_ids == first.selected_agent_ids
    assert reversed_result.plan_signature == first.plan_signature


def test_explicit_preference_resolves_an_otherwise_equal_set() -> None:
    system = _system(
        (
            _agent("agent.a", capabilities=("cap.same",)),
            _agent("agent.b", capabilities=("cap.same",)),
        )
    )

    result = system.agent_cooperation_automatic_planner.plan(
        _request(
            required_agent_ids=(),
            required_capability_ids=("cap.same",),
            preferred_agent_ids=("agent.b",),
        )
    )

    assert result.status is AgentCooperationPlanningStatus.SUCCESS
    assert result.selected_agent_ids == ("agent.b",)


def test_optional_skill_coverage_is_a_deterministic_preference() -> None:
    system = _system(
        (
            _agent("agent.a", capabilities=("cap.same",)),
            _agent("agent.b", capabilities=("cap.same", "cap.inspect")),
        ),
        skills=(_skill(),),
    )

    result = system.agent_cooperation_automatic_planner.plan(
        _request(
            required_agent_ids=(),
            required_capability_ids=("cap.same",),
            optional_skill_ids=("skill.inspect",),
        )
    )

    assert result.status is AgentCooperationPlanningStatus.SUCCESS
    assert result.selected_agent_ids == ("agent.b",)


def test_max_agent_limit_is_enforced_after_minimum_set_calculation() -> None:
    system = _system(
        (
            _agent("agent.a", capabilities=("cap.a",)),
            _agent("agent.b", capabilities=("cap.b",)),
        )
    )

    result = system.agent_cooperation_automatic_planner.plan(
        _request(
            required_agent_ids=(),
            required_capability_ids=("cap.a", "cap.b"),
            policy=_policy(max_agents=1),
        )
    )

    assert result.status is AgentCooperationPlanningStatus.LIMIT_REACHED


def test_task_dependency_and_depth_limits_are_enforced() -> None:
    requirements = (
        AgentCooperationPlanningTaskRequirement("task.a"),
        AgentCooperationPlanningTaskRequirement("task.b", depends_on=("task.a",)),
    )
    max_tasks = _system().agent_cooperation_automatic_planner.plan(
        _request(task_requirements=requirements, policy=_policy(max_tasks=1))
    )
    max_depth = _system().agent_cooperation_automatic_planner.plan(
        _request(task_requirements=requirements, policy=_policy(max_plan_depth=1))
    )

    assert max_tasks.status is AgentCooperationPlanningStatus.LIMIT_REACHED
    assert max_depth.status is AgentCooperationPlanningStatus.LIMIT_REACHED


def test_dependency_count_limit_is_enforced() -> None:
    requirements = (
        AgentCooperationPlanningTaskRequirement("task.a"),
        AgentCooperationPlanningTaskRequirement("task.b"),
        AgentCooperationPlanningTaskRequirement(
            "task.c",
            depends_on=("task.a", "task.b"),
        ),
    )

    result = _system().agent_cooperation_automatic_planner.plan(
        _request(
            task_requirements=requirements,
            policy=_policy(max_dependencies=1),
        )
    )

    assert result.status is AgentCooperationPlanningStatus.LIMIT_REACHED
    assert result.error_code == "MAX_DEPENDENCIES"


def test_invalid_and_circular_dependencies_fail_without_plan() -> None:
    invalid = _system().agent_cooperation_automatic_planner.plan(
        _request(
            task_requirements=(
                AgentCooperationPlanningTaskRequirement("task.a", depends_on=("missing",)),
            )
        )
    )
    circular = _system().agent_cooperation_automatic_planner.plan(
        _request(
            task_requirements=(
                AgentCooperationPlanningTaskRequirement("task.a", depends_on=("task.b",)),
                AgentCooperationPlanningTaskRequirement("task.b", depends_on=("task.a",)),
            )
        )
    )

    assert invalid.status is AgentCooperationPlanningStatus.INVALID_REQUEST
    assert circular.status is AgentCooperationPlanningStatus.INVALID_REQUEST
    assert invalid.plan is None and circular.plan is None


def test_safe_binding_is_carried_into_valid_plan() -> None:
    binding = AgentCooperationOutputBinding(
        "task.a",
        source_path=("result", "value"),
        target_path=("input", "previous"),
    )
    requirements = (
        AgentCooperationPlanningTaskRequirement("task.a"),
        AgentCooperationPlanningTaskRequirement(
            "task.b",
            depends_on=("task.a",),
            output_bindings=(binding,),
        ),
    )

    result = _system().agent_cooperation_automatic_planner.plan(
        _request(task_requirements=requirements)
    )

    assert result.status is AgentCooperationPlanningStatus.SUCCESS
    assert result.created_dependencies[0].prerequisite_task_id == "task.a"
    assert result.plan.tasks[1].dependency_output_bindings == (binding,)


def test_sensitive_keys_non_finite_and_arbitrary_objects_are_rejected() -> None:
    with pytest.raises(InvalidAgentCooperationPlanningRequestError):
        _request(metadata={"api_key": "hidden"})
    with pytest.raises(InvalidAgentCooperationPlanningRequestError):
        _request(structured_input={"value": math.nan})
    with pytest.raises(InvalidAgentCooperationPlanningRequestError):
        _request(structured_input={"value": object()})
    with pytest.raises(InvalidAgentCooperationPlanningRequestError):
        _request(metadata={"python_path": "package.module"})


def test_request_result_and_plan_are_immutable_and_request_is_not_modified() -> None:
    source = {"value": {"nested": 1}}
    request = _request(structured_input=source)
    source["value"]["nested"] = 2
    result = _system().agent_cooperation_automatic_planner.plan(request)

    assert request.structured_input["value"]["nested"] == 1
    with pytest.raises(TypeError):
        request.structured_input["new"] = 1
    with pytest.raises(FrozenInstanceError):
        result.status = AgentCooperationPlanningStatus.INTERNAL_ERROR
    with pytest.raises(FrozenInstanceError):
        result.plan.plan_id = "changed"


def test_request_and_plan_signatures_are_deterministic() -> None:
    first = _request(structured_input={"b": 2, "a": 1})
    same = _request(structured_input={"a": 1, "b": 2})
    different = _request(structured_input={"a": 2, "b": 2})

    assert agent_cooperation_planning_request_signature(first) == agent_cooperation_planning_request_signature(same)
    assert agent_cooperation_planning_request_signature(first) != agent_cooperation_planning_request_signature(different)
    first_result = _system().agent_cooperation_automatic_planner.plan(first)
    same_result = _system().agent_cooperation_automatic_planner.plan(same)
    assert first_result.plan_signature == same_result.plan_signature


def test_events_metrics_and_decisions_are_explainable() -> None:
    result = _system().agent_cooperation_automatic_planner.plan(_request())
    names = {event.name for event in result.events}

    assert {
        "agent_cooperation_planning_requested",
        "agent_cooperation_planning_validation_started",
        "agent_cooperation_candidate_evaluated",
        "agent_cooperation_candidate_accepted",
        "agent_cooperation_requirements_covered",
        "agent_cooperation_plan_build_started",
        "agent_cooperation_plan_built",
        "agent_cooperation_planning_completed",
    }.issubset(names)
    assert result.metrics["cooperation_planning_succeeded"] == 1
    assert result.decisions[0].status is AgentCooperationPlanningDecisionStatus.SELECTED


def test_agent_system_and_factory_share_existing_dependencies() -> None:
    system = _system()
    planner = system.agent_cooperation_automatic_planner
    standalone = build_core_agent_cooperation_automatic_planner(
        agent_registry=system.agent_registry,
        agent_resolver=system.agent_resolver,
        agent_context_builder=system.agent_context_builder,
        agent_executor=system.agent_executor,
        skill_system=system.skill_system,
        agent_cooperation_planner=system.agent_cooperation_planner,
        agent_delegation_service=system.agent_delegation_service,
        agent_delegation_chain_service=system.agent_delegation_chain_service,
        agent_delegation_coordinator=system.agent_delegation_coordinator,
        multi_agent_coordinator=system.multi_agent_coordinator,
    )

    assert isinstance(planner, AgentCooperationAutomaticPlanner)
    assert isinstance(standalone, AgentCooperationAutomaticPlanner)
    assert planner._agent_registry is system.agent_registry
    assert planner._agent_resolver is system.agent_resolver
    assert planner._agent_context_builder is system.agent_context_builder
    assert planner._agent_executor is system.agent_executor
    assert planner._skill_system is system.skill_system
    assert planner._agent_cooperation_planner is system.agent_cooperation_planner


def test_planning_does_not_execute_agents_skills_or_modify_registries() -> None:
    system = _system(
        (_agent("agent.a", capabilities=("cap.inspect",)),),
        skills=(_skill(),),
    )
    agents_before = system.agent_registry.list_agents()
    skills_before = system.skill_system.skill_registry.list_skills()
    handlers_before = system.agent_handler_registry.list_handlers()

    result = system.agent_cooperation_automatic_planner.plan(
        _request(required_skill_ids=("skill.inspect",))
    )

    assert result.status is AgentCooperationPlanningStatus.SUCCESS
    assert NeverRunHandler.calls == []
    assert CountingSkillExecutor.calls == 0
    assert system.agent_registry.list_agents() == agents_before
    assert system.skill_system.skill_registry.list_skills() == skills_before
    assert system.agent_handler_registry.list_handlers() == handlers_before


def test_multi_agent_and_delegation_task_types_are_built_declaratively() -> None:
    multi_system = _system(
        (
            _agent("agent.a", capabilities=("cap.a",)),
            _agent("agent.b", capabilities=("cap.b",)),
        )
    )
    multi = multi_system.agent_cooperation_automatic_planner.plan(
        _request(
            required_agent_ids=(),
            required_capability_ids=("cap.a", "cap.b"),
            execution_type=AgentCooperationExecutionType.MULTI_AGENT,
        )
    )
    delegation_system = _system(
        (
            _agent("agent.source", capabilities=("cap.source",)),
            _agent("agent.target", capabilities=("cap.target",)),
        )
    )
    delegation = delegation_system.agent_cooperation_automatic_planner.plan(
        _request(
            required_agent_ids=(),
            required_capability_ids=("cap.target",),
            execution_type=AgentCooperationExecutionType.DELEGATION,
            source_agent_id="agent.source",
        )
    )

    assert multi.status is AgentCooperationPlanningStatus.SUCCESS
    assert multi.plan.tasks[0].execution_type is AgentCooperationExecutionType.MULTI_AGENT
    assert delegation.status is AgentCooperationPlanningStatus.SUCCESS
    assert delegation.plan.tasks[0].execution_type is AgentCooperationExecutionType.DELEGATION
    assert NeverRunHandler.calls == []


def test_missing_delegation_source_is_rejected_without_execution() -> None:
    system = _system((_agent("agent.target", capabilities=("cap.target",)),))

    result = system.agent_cooperation_automatic_planner.plan(
        _request(
            required_agent_ids=(),
            required_capability_ids=("cap.target",),
            execution_type=AgentCooperationExecutionType.DELEGATION,
            source_agent_id="agent.missing",
        )
    )

    assert result.status is AgentCooperationPlanningStatus.NO_MATCHING_AGENTS
    assert result.error_code == "SOURCE_AGENT_MISSING"
    assert NeverRunHandler.calls == []


def test_e2e_local_builds_valid_plan_with_agents_handlers_and_skill_without_execution() -> None:
    system = _system(
        (
            _agent(
                "agent.analysis",
                capabilities=("cap.inspect",),
                metadata={"allowed_skill_ids": "skill.inspect"},
            ),
        ),
        skills=(_skill(),),
    )

    result = system.agent_cooperation_automatic_planner.plan(
        AgentCooperationPlanningRequest(
            objective_id="objective.e2e",
            objective_type=AgentCooperationObjectiveType.ANALYSIS,
            required_capability_ids=("cap.inspect",),
            required_skill_ids=("skill.inspect",),
            structured_input={"project": "Atlas"},
            policy=_policy(),
        )
    )

    assert result.status is AgentCooperationPlanningStatus.SUCCESS
    assert isinstance(result.plan, AgentCooperationPlan)
    assert result.plan.tasks[0].agent_id == "agent.analysis"
    assert result.plan.tasks[0].required_skill_ids == ("skill.inspect",)
    assert result.plan_signature
    assert NeverRunHandler.calls == []
    assert CountingSkillExecutor.calls == 0
