from __future__ import annotations

from collections.abc import Mapping
import math

import pytest

from bootstrap.agent_cooperation_plan import build_core_agent_cooperation_planner
from bootstrap.agent_system import build_core_agent_system
from core.agent_context import AgentContext
from core.agent_cooperation_plan import (
    AgentCooperationCycleError,
    AgentCooperationDependency,
    AgentCooperationDependencyError,
    AgentCooperationExecutionType,
    AgentCooperationFailureMode,
    AgentCooperationOutputBinding,
    AgentCooperationPlan,
    AgentCooperationPlanPolicy,
    AgentCooperationPlanRequest,
    AgentCooperationPlanStatus,
    AgentCooperationPlanner,
    AgentCooperationTask,
    AgentCooperationTaskStatus,
    InvalidAgentCooperationPlanError,
    agent_cooperation_plan_signature,
)
from core.agent_delegation import AgentDelegationPolicy, AgentDelegationRequest
from core.agent_delegation_chain import (
    AgentDelegationChainPolicy,
    AgentDelegationChainRequest,
    AgentDelegationChainStep,
)
from core.agent_delegation_coordinator import (
    AgentDelegationCoordinationChain,
    AgentDelegationCoordinationPlan,
    AgentDelegationCoordinationPolicy,
    AgentDelegationCoordinationRequest,
)
from core.agent_registry import (
    AgentCapabilities,
    AgentContextPolicy,
    AgentDefinition,
    AgentPermissions,
    AgentType,
)
from core.multi_agent import MultiAgentExecutionPolicy, MultiAgentExecutionRequest
from core.skill_registry import SkillDefinition, SkillExecutionTargetType


class CooperationHandler:
    calls: list[tuple[str, Mapping[str, object], Mapping[str, object]]] = []

    def __init__(self, agent_id: str, *, fail: bool = False) -> None:
        self._agent_id = agent_id
        self._fail = fail

    @property
    def agent_id(self) -> str:
        return self._agent_id

    def handle(self, context: AgentContext) -> Mapping[str, object]:
        CooperationHandler.calls.append(
            (self._agent_id, context.structured_input, context.shared_context)
        )
        if self._fail:
            raise RuntimeError("authorization token hidden-value")
        return {
            "agent_id": self._agent_id,
            "value": context.structured_input.get("value", self._agent_id),
            "received": context.structured_input,
            "token": "must-not-propagate",
        }


def _definition(
    agent_id: str,
    *,
    capabilities: tuple[str, ...] = ("cooperate.inspect",),
    agent_type: AgentType = AgentType.GENERAL,
    metadata: Mapping[str, object] | None = None,
) -> AgentDefinition:
    return AgentDefinition(
        agent_id=agent_id,
        agent_type=agent_type,
        name=agent_id,
        description="Cooperation plan test agent.",
        capabilities=AgentCapabilities(capabilities=capabilities),
        permissions=AgentPermissions(requires_confirmation=False),
        context_policy=AgentContextPolicy(allow_shared_context=True),
        metadata={} if metadata is None else metadata,
    )


def _system(*, fail: tuple[str, ...] = (), agents: tuple[AgentDefinition, ...] | None = None):
    built = build_core_agent_system()
    assert built.system is not None
    system = built.system
    definitions = agents or (
        _definition("agent.a", capabilities=("cooperate.inspect", "unique.a")),
        _definition("agent.b"),
        _definition("agent.c"),
        _definition("agent.d"),
    )
    for definition in definitions:
        system.agent_registry.register(definition)
        system.agent_handler_registry.register(
            CooperationHandler(definition.agent_id, fail=definition.agent_id in fail)
        )
    return system


def _policy(**overrides: object) -> AgentCooperationPlanPolicy:
    values = {
        "enabled": True,
        "max_tasks": 20,
        "max_dependencies": 40,
        "max_depth": 10,
        "max_total_executions": 20,
        "max_output_items": 256,
        "max_propagated_items": 128,
    }
    values.update(overrides)
    return AgentCooperationPlanPolicy(**values)


def _single(task_id: str, agent_id: str = "agent.a", **overrides: object) -> AgentCooperationTask:
    values = {
        "task_id": task_id,
        "objective_id": f"objective.{task_id}",
        "agent_id": agent_id,
        "structured_input": {"value": task_id},
    }
    values.update(overrides)
    return AgentCooperationTask(**values)


def _plan(
    *tasks: AgentCooperationTask,
    dependencies: tuple[AgentCooperationDependency, ...] = (),
    policy: AgentCooperationPlanPolicy | None = None,
) -> AgentCooperationPlan:
    return AgentCooperationPlan(
        plan_id="plan.test",
        tasks=tasks or (_single("task.a"),),
        dependencies=dependencies,
        policy=policy or _policy(),
        metadata={"owner": "tests"},
    )


def _execute(system, plan: AgentCooperationPlan, policy: AgentCooperationPlanPolicy | None = None):
    return system.agent_cooperation_planner.execute(
        AgentCooperationPlanRequest(plan=plan, policy=policy)
    )


def _delegation_request() -> AgentDelegationRequest:
    return AgentDelegationRequest(
        origin_agent_id="agent.a",
        target_agent_id="agent.b",
        required_capability_ids=("cooperate.inspect",),
        policy=AgentDelegationPolicy(enabled=True, propagate_structured_input=True),
    )


def _chain_request(source: str = "agent.a", target: str = "agent.b") -> AgentDelegationChainRequest:
    return AgentDelegationChainRequest(
        steps=(
            AgentDelegationChainStep(
                source_agent_id=source,
                target_agent_id=target,
                execution_required_capability_ids=("cooperate.inspect",),
            ),
        ),
        policy=AgentDelegationChainPolicy(
            enabled=True,
            max_steps=3,
            max_depth=3,
            max_total_delegations=3,
        ),
    )


def _coordination_request() -> AgentDelegationCoordinationRequest:
    return AgentDelegationCoordinationRequest(
        source_agent_id="agent.a",
        plan=AgentDelegationCoordinationPlan(
            plan_id="coord.plan",
            chains=(
                AgentDelegationCoordinationChain(
                    chain_id="chain.a",
                    chain_request=_chain_request(),
                ),
            ),
        ),
        policy=AgentDelegationCoordinationPolicy(
            enabled=True,
            max_chains=2,
            max_total_steps=4,
        ),
    )


def setup_function() -> None:
    CooperationHandler.calls = []


def test_linear_plan_a_b_c_runs_in_dependency_order() -> None:
    plan = _plan(
        _single("task.a"),
        _single("task.b"),
        _single("task.c"),
        dependencies=(
            AgentCooperationDependency("task.a", "task.b"),
            AgentCooperationDependency("task.b", "task.c"),
        ),
    )

    result = _execute(_system(), plan)

    assert result.status is AgentCooperationPlanStatus.SUCCESS
    assert result.execution_order == ("task.a", "task.b", "task.c")


def test_independent_tasks_use_order_priority_then_task_id() -> None:
    plan = _plan(
        _single("task.c", order=1),
        _single("task.b", order=0, priority=1),
        _single("task.a", order=0, priority=1),
        _single("task.z", order=0, priority=2),
    )

    result = _execute(_system(), plan)

    assert result.execution_order == ("task.z", "task.a", "task.b", "task.c")


def test_multiple_dependencies_a_b_to_c() -> None:
    plan = _plan(
        _single("task.a"),
        _single("task.b"),
        _single("task.c"),
        dependencies=(
            AgentCooperationDependency("task.a", "task.c"),
            AgentCooperationDependency("task.b", "task.c"),
        ),
    )

    result = _execute(_system(), plan)

    assert result.status is AgentCooperationPlanStatus.SUCCESS
    assert result.execution_order[-1] == "task.c"
    assert result.metrics["cooperation_dependencies_satisfied"] == 2


def test_empty_plan_and_duplicate_task_ids_are_rejected() -> None:
    with pytest.raises(InvalidAgentCooperationPlanError):
        AgentCooperationPlan("plan.empty", ())
    with pytest.raises(InvalidAgentCooperationPlanError):
        _plan(_single("task.a"), _single("task.a"))


def test_unknown_self_and_duplicate_dependencies_are_rejected() -> None:
    with pytest.raises(AgentCooperationDependencyError):
        _plan(_single("task.a"), dependencies=(AgentCooperationDependency("missing", "task.a"),))
    with pytest.raises(AgentCooperationDependencyError):
        AgentCooperationDependency("task.a", "task.a")
    duplicate = AgentCooperationDependency("task.a", "task.b")
    with pytest.raises(AgentCooperationDependencyError):
        _plan(_single("task.a"), _single("task.b"), dependencies=(duplicate, duplicate))


@pytest.mark.parametrize(
    "dependencies",
    [
        (
            AgentCooperationDependency("task.a", "task.b"),
            AgentCooperationDependency("task.b", "task.a"),
        ),
        (
            AgentCooperationDependency("task.a", "task.b"),
            AgentCooperationDependency("task.b", "task.c"),
            AgentCooperationDependency("task.c", "task.a"),
        ),
    ],
)
def test_cycles_are_detected_deterministically(dependencies) -> None:
    with pytest.raises(AgentCooperationCycleError):
        _plan(_single("task.a"), _single("task.b"), _single("task.c"), dependencies=dependencies)


def test_explicit_and_automatic_single_agent_resolution() -> None:
    system = _system()
    explicit = _execute(system, _plan(_single("task.explicit", "agent.b")))
    automatic = _execute(
        system,
        _plan(
            AgentCooperationTask(
                task_id="task.auto",
                objective_id="objective.auto",
                required_capability_ids=("unique.a",),
            )
        ),
    )

    assert explicit.task_results[0].agent_ids == ("agent.b",)
    assert automatic.task_results[0].agent_ids == ("agent.a",)


def test_missing_and_ambiguous_agent_resolution_are_structured() -> None:
    missing = _execute(
        _system(),
        _plan(
            AgentCooperationTask(
                "task.missing",
                "objective.missing",
                required_capability_ids=("capability.missing",),
            )
        ),
    )
    ambiguous = _execute(
        _system(),
        _plan(
            AgentCooperationTask(
                "task.ambiguous",
                "objective.ambiguous",
                required_capability_ids=("cooperate.inspect",),
            )
        ),
    )

    assert missing.task_results[0].status is AgentCooperationTaskStatus.AGENT_RESOLUTION_FAILED
    assert ambiguous.task_results[0].status is AgentCooperationTaskStatus.AGENT_RESOLUTION_FAILED


def test_single_agent_execution_uses_existing_executor() -> None:
    system = _system()
    result = _execute(system, _plan(_single("task.single", "agent.b")))

    assert result.status is AgentCooperationPlanStatus.SUCCESS
    assert result.task_results[0].execution_result is not None
    assert CooperationHandler.calls[0][0] == "agent.b"


def test_delegation_execution_uses_existing_service() -> None:
    task = AgentCooperationTask(
        "task.delegation",
        "objective.delegation",
        execution_type=AgentCooperationExecutionType.DELEGATION,
        delegation_request=_delegation_request(),
        structured_input={"value": "delegated"},
    )

    result = _execute(_system(), _plan(task))

    assert result.status is AgentCooperationPlanStatus.SUCCESS
    assert result.task_results[0].agent_ids == ("agent.b",)


def test_delegation_chain_execution_uses_existing_service() -> None:
    task = AgentCooperationTask(
        "task.chain",
        "objective.chain",
        execution_type=AgentCooperationExecutionType.DELEGATION_CHAIN,
        delegation_chain_request=_chain_request(),
        structured_input={"value": "chain"},
    )

    result = _execute(_system(), _plan(task))

    assert result.status is AgentCooperationPlanStatus.SUCCESS
    assert result.task_results[0].agent_ids == ("agent.b",)


def test_multi_agent_execution_uses_existing_coordinator() -> None:
    task = AgentCooperationTask(
        "task.multi",
        "objective.multi",
        execution_type=AgentCooperationExecutionType.MULTI_AGENT,
        multi_agent_request=MultiAgentExecutionRequest(
            required_agent_ids=("agent.a", "agent.b"),
            policy=MultiAgentExecutionPolicy(min_agents=2, max_agents=2),
        ),
        structured_input={"value": "multi"},
    )

    result = _execute(_system(), _plan(task))

    assert result.status is AgentCooperationPlanStatus.SUCCESS
    assert result.task_results[0].agent_ids == ("agent.a", "agent.b")


def test_coordinated_chains_execution_uses_existing_coordinator() -> None:
    task = AgentCooperationTask(
        "task.coordinated",
        "objective.coordinated",
        execution_type=AgentCooperationExecutionType.COORDINATED_CHAINS,
        coordinated_chains_request=_coordination_request(),
        structured_input={"value": "coordinated"},
    )

    result = _execute(_system(), _plan(task))

    assert result.status is AgentCooperationPlanStatus.SUCCESS
    assert "chain.a" in result.task_results[0].output


def test_execution_type_xor_is_enforced() -> None:
    with pytest.raises(InvalidAgentCooperationPlanError):
        AgentCooperationTask(
            "task.bad",
            "objective.bad",
            execution_type=AgentCooperationExecutionType.DELEGATION,
            delegation_request=_delegation_request(),
            delegation_chain_request=_chain_request(),
        )
    with pytest.raises(InvalidAgentCooperationPlanError):
        AgentCooperationTask(
            "task.bad",
            "objective.bad",
            execution_type=AgentCooperationExecutionType.MULTI_AGENT,
        )


def test_dependency_output_binding_propagates_safe_value() -> None:
    source = _single("task.a", structured_input={"value": "from-a"})
    target = _single(
        "task.b",
        "agent.b",
        dependency_output_bindings=(
            AgentCooperationOutputBinding(
                "task.a",
                source_path=("result", "value"),
                target_path=("input", "previous"),
            ),
        ),
    )
    plan = _plan(
        source,
        target,
        dependencies=(AgentCooperationDependency("task.a", "task.b"),),
        policy=_policy(propagate_dependency_outputs=True),
    )

    result = _execute(_system(), plan)

    assert result.status is AgentCooperationPlanStatus.SUCCESS
    assert CooperationHandler.calls[1][1]["previous"] == "from-a"
    assert result.metrics["cooperation_outputs_propagated"] == 1


def test_binding_collision_and_missing_output_are_structured_failures() -> None:
    source = _single("task.a")
    collision = _single(
        "task.b",
        structured_input={"previous": "exists"},
        dependency_output_bindings=(
            AgentCooperationOutputBinding(
                "task.a",
                source_path=("result", "value"),
                target_path=("input", "previous"),
            ),
        ),
    )
    missing = _single(
        "task.c",
        dependency_output_bindings=(
            AgentCooperationOutputBinding(
                "task.a",
                source_path=("result", "missing"),
                target_path=("input", "previous"),
            ),
        ),
    )
    dependencies = (
        AgentCooperationDependency("task.a", "task.b"),
        AgentCooperationDependency("task.a", "task.c"),
    )

    collision_result = _execute(
        _system(),
        _plan(source, collision, dependencies=(dependencies[0],), policy=_policy(propagate_dependency_outputs=True)),
    )
    missing_result = _execute(
        _system(),
        _plan(source, missing, dependencies=(dependencies[1],), policy=_policy(propagate_dependency_outputs=True)),
    )

    assert collision_result.task_results[1].status is AgentCooperationTaskStatus.OUTPUT_BINDING_FAILED
    assert missing_result.task_results[1].status is AgentCooperationTaskStatus.OUTPUT_BINDING_FAILED


def test_binding_source_must_be_explicit_dependency_and_paths_are_safe() -> None:
    bound = _single(
        "task.b",
        dependency_output_bindings=(AgentCooperationOutputBinding("task.a"),),
    )
    with pytest.raises(AgentCooperationDependencyError):
        _plan(_single("task.a"), bound)
    with pytest.raises(InvalidAgentCooperationPlanError):
        AgentCooperationOutputBinding(
            "task.a",
            source_path=("result", "__class__"),
            target_path=("input", "value"),
        )
    with pytest.raises(InvalidAgentCooperationPlanError):
        AgentCooperationOutputBinding(
            "task.a",
            target_path=("input", "api_key"),
        )


def test_sensitive_output_is_removed_and_sensitive_input_is_rejected() -> None:
    result = _execute(_system(), _plan(_single("task.a")))

    assert "token" not in result.task_results[0].output
    with pytest.raises(InvalidAgentCooperationPlanError):
        _single("task.bad", metadata={"api_key": "hidden"})
    with pytest.raises(InvalidAgentCooperationPlanError):
        _single("task.bad", structured_input={"password": "hidden"})


@pytest.mark.parametrize("value", [math.nan, math.inf, object(), lambda: None, int])
def test_non_finite_and_arbitrary_objects_are_rejected(value: object) -> None:
    with pytest.raises(InvalidAgentCooperationPlanError):
        _single("task.bad", structured_input={"value": value})


def test_stop_on_first_failure_skips_remaining_tasks() -> None:
    plan = _plan(
        _single("task.a", "agent.a"),
        _single("task.b", "agent.b"),
        policy=_policy(
            failure_mode=AgentCooperationFailureMode.STOP_ON_FIRST_FAILURE,
            allow_skipped_tasks=True,
        ),
    )

    result = _execute(_system(fail=("agent.a",)), plan)

    assert result.task_results[0].status is AgentCooperationTaskStatus.AGENT_EXECUTION_FAILED
    assert result.task_results[1].status is AgentCooperationTaskStatus.SKIPPED


def test_continue_independent_tasks_and_partial_success() -> None:
    plan = _plan(
        _single("task.a", "agent.a", continue_on_failure=True),
        _single("task.b", "agent.b"),
        policy=_policy(
            failure_mode=AgentCooperationFailureMode.CONTINUE_INDEPENDENT_TASKS,
            allow_partial_success=True,
        ),
    )

    result = _execute(_system(fail=("agent.a",)), plan)

    assert result.status is AgentCooperationPlanStatus.PARTIAL_SUCCESS
    assert result.task_results[1].status is AgentCooperationTaskStatus.SUCCESS


def test_require_all_success_and_minimum_success() -> None:
    require_all = _plan(
        _single("task.a", "agent.a"),
        _single("task.b", "agent.b"),
        policy=_policy(
            failure_mode=AgentCooperationFailureMode.REQUIRE_ALL_SUCCESS,
            allow_skipped_tasks=True,
        ),
    )
    minimum = _plan(
        _single("task.a", "agent.a"),
        _single("task.b", "agent.b"),
        policy=_policy(
            failure_mode=AgentCooperationFailureMode.REQUIRE_MINIMUM_SUCCESS,
            minimum_successful_tasks=1,
            allow_partial_success=True,
        ),
    )

    all_result = _execute(_system(fail=("agent.a",)), require_all)
    minimum_result = _execute(_system(fail=("agent.a",)), minimum)

    assert all_result.status is AgentCooperationPlanStatus.FAILED
    assert minimum_result.status is AgentCooperationPlanStatus.PARTIAL_SUCCESS


def test_minimum_success_not_reached() -> None:
    plan = _plan(
        _single("task.a", "agent.a"),
        _single("task.b", "agent.b"),
        policy=_policy(
            failure_mode=AgentCooperationFailureMode.REQUIRE_MINIMUM_SUCCESS,
            minimum_successful_tasks=2,
            allow_partial_success=True,
        ),
    )

    result = _execute(_system(fail=("agent.a",)), plan)

    assert result.status is AgentCooperationPlanStatus.MINIMUM_SUCCESS_NOT_REACHED


def test_failed_dependency_blocks_dependent_task() -> None:
    plan = _plan(
        _single("task.a", "agent.a", continue_on_failure=True),
        _single("task.b", "agent.b"),
        _single("task.c", "agent.c"),
        dependencies=(AgentCooperationDependency("task.a", "task.b"),),
        policy=_policy(
            failure_mode=AgentCooperationFailureMode.CONTINUE_INDEPENDENT_TASKS,
            allow_partial_success=True,
            stop_on_blocked_task=False,
        ),
    )

    result = _execute(_system(fail=("agent.a",)), plan)

    statuses = {item.task_id: item.status for item in result.task_results}
    assert statuses["task.b"] is AgentCooperationTaskStatus.DEPENDENCY_FAILED
    assert statuses["task.c"] is AgentCooperationTaskStatus.SUCCESS


def test_task_dependency_depth_and_execution_limits() -> None:
    tasks = (_single("task.a"), _single("task.b"), _single("task.c"))
    dependencies = (
        AgentCooperationDependency("task.a", "task.b"),
        AgentCooperationDependency("task.b", "task.c"),
    )
    max_tasks = _execute(_system(), _plan(*tasks), _policy(max_tasks=2))
    max_dependencies = _execute(
        _system(),
        _plan(*tasks, dependencies=dependencies),
        _policy(max_dependencies=1),
    )
    max_depth = _execute(
        _system(),
        _plan(*tasks, dependencies=dependencies),
        _policy(max_depth=2),
    )
    max_executions = _execute(
        _system(),
        _plan(*tasks),
        _policy(max_total_executions=1, allow_skipped_tasks=True),
    )

    assert max_tasks.status is AgentCooperationPlanStatus.LIMIT_REACHED
    assert max_dependencies.status is AgentCooperationPlanStatus.LIMIT_REACHED
    assert max_depth.status is AgentCooperationPlanStatus.LIMIT_REACHED
    assert any(item.status is AgentCooperationTaskStatus.LIMIT_REACHED for item in max_executions.task_results)


def test_plan_signature_is_stable_and_sensitive_to_content() -> None:
    first = _plan(_single("task.a", structured_input={"b": 2, "a": 1}))
    same = _plan(_single("task.a", structured_input={"a": 1, "b": 2}))
    different = _plan(_single("task.a", structured_input={"a": 2, "b": 2}))

    assert agent_cooperation_plan_signature(first) == agent_cooperation_plan_signature(same)
    assert agent_cooperation_plan_signature(first) != agent_cooperation_plan_signature(different)


def test_skill_existence_and_authorization_are_validated() -> None:
    system = _system(
        agents=(
            _definition(
                "agent.a",
                metadata={"allowed_skill_ids": "skill.allowed"},
            ),
        )
    )
    system.skill_system.skill_registry.register(
        SkillDefinition(
            skill_id="skill.allowed",
            name="Allowed",
            version="1.0",
            description="Allowed skill.",
            required_capability_ids=("cooperate.inspect",),
            allowed_agent_types=(AgentType.GENERAL,),
            execution_target="handler.allowed",
            execution_target_type=SkillExecutionTargetType.HANDLER,
        )
    )
    allowed = _execute(
        system,
        _plan(_single("task.allowed", required_skill_ids=("skill.allowed",))),
    )
    missing = _execute(
        system,
        _plan(_single("task.missing", required_skill_ids=("skill.missing",))),
    )

    assert allowed.status is AgentCooperationPlanStatus.SUCCESS
    assert missing.status is AgentCooperationPlanStatus.INVALID_PLAN


def test_disabled_by_default_and_bootstrap_executes_nothing() -> None:
    system = _system()
    calls_before = list(CooperationHandler.calls)
    default_plan = AgentCooperationPlan(
        "plan.disabled",
        (_single("task.a"),),
    )

    result = _execute(system, default_plan)
    standalone = build_core_agent_cooperation_planner(
        agent_registry=system.agent_registry,
        agent_resolver=system.agent_resolver,
        agent_context_builder=system.agent_context_builder,
        agent_executor=system.agent_executor,
        agent_delegation_service=system.agent_delegation_service,
        agent_delegation_chain_service=system.agent_delegation_chain_service,
        agent_delegation_coordinator=system.agent_delegation_coordinator,
        multi_agent_coordinator=system.multi_agent_coordinator,
        skill_system=system.skill_system,
    )

    assert result.status is AgentCooperationPlanStatus.BLOCKED
    assert CooperationHandler.calls == calls_before
    assert isinstance(standalone, AgentCooperationPlanner)
    assert CooperationHandler.calls == calls_before


def test_agent_system_shares_all_existing_dependencies() -> None:
    system = _system()
    planner = system.agent_cooperation_planner

    assert planner._agent_registry is system.agent_registry
    assert planner._agent_resolver is system.agent_resolver
    assert planner._agent_context_builder is system.agent_context_builder
    assert planner._agent_executor is system.agent_executor
    assert planner._agent_delegation_service is system.agent_delegation_service
    assert planner._agent_delegation_chain_service is system.agent_delegation_chain_service
    assert planner._agent_delegation_coordinator is system.agent_delegation_coordinator
    assert planner._multi_agent_coordinator is system.multi_agent_coordinator
    assert planner._skill_system is system.skill_system


def test_observability_is_structured_flat_and_sensitive_free() -> None:
    result = _execute(_system(), _plan(_single("task.a")))
    names = {event.name for event in result.events}

    assert {
        "agent_cooperation_plan_requested",
        "agent_cooperation_plan_validation_started",
        "agent_cooperation_plan_validation_succeeded",
        "agent_cooperation_plan_started",
        "agent_cooperation_task_ready",
        "agent_cooperation_task_started",
        "agent_cooperation_task_succeeded",
        "agent_cooperation_plan_completed",
    }.issubset(names)
    assert all(type(value) is int for value in result.metrics.values())
    assert "must-not-propagate" not in repr(result)


def test_e2e_declared_plan_with_real_handlers_and_binding() -> None:
    source = _single("task.analyze", "agent.a", structured_input={"value": "analysis"})
    target = _single(
        "task.apply",
        "agent.b",
        dependency_output_bindings=(
            AgentCooperationOutputBinding(
                "task.analyze",
                source_path=("result", "value"),
                target_path=("input", "previous_analysis"),
            ),
        ),
    )
    plan = _plan(
        source,
        target,
        dependencies=(AgentCooperationDependency("task.analyze", "task.apply"),),
        policy=_policy(propagate_dependency_outputs=True),
    )

    result = _execute(_system(), plan)

    assert result.status is AgentCooperationPlanStatus.SUCCESS
    assert result.execution_order == ("task.analyze", "task.apply")
    assert CooperationHandler.calls[1][1]["previous_analysis"] == "analysis"
    assert result.metrics["cooperation_tasks_succeeded"] == 2
