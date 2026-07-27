from __future__ import annotations

from collections.abc import Mapping
from dataclasses import FrozenInstanceError
import math
import types

import pytest

from bootstrap.agent_plan_supervisor import build_core_agent_plan_supervisor
from bootstrap.agent_system import build_core_agent_system
from core.agent_context import AgentContext
from core.agent_cooperation_automatic_execution import (
    AgentCooperationAutomaticExecutionPolicy,
    AgentCooperationAutomaticExecutionRequest,
    AgentCooperationAutomaticExecutionStatus,
)
from core.agent_cooperation_automatic_planner import (
    AgentCooperationObjectiveType,
    AgentCooperationPlanningPolicy,
    AgentCooperationPlanningRequest,
)
from core.agent_cooperation_plan import (
    AgentCooperationDependency,
    AgentCooperationExecutionType,
    AgentCooperationFailureMode,
    AgentCooperationPlan,
    AgentCooperationPlanEvent,
    AgentCooperationPlanPolicy,
    AgentCooperationPlanResult,
    AgentCooperationPlanStatus,
    AgentCooperationTask,
    AgentCooperationTaskResult,
    AgentCooperationTaskStatus,
    agent_cooperation_plan_signature,
)
from core.agent_plan_supervisor import (
    AgentPlanSupervisor,
    AgentPlanSupervisorDecisionType,
    AgentPlanSupervisorPolicy,
    AgentPlanSupervisorRequest,
    AgentPlanSupervisorStatus,
    InvalidAgentPlanSupervisorRequestError,
    agent_plan_supervisor_request_signature,
)
from core.agent_registry import (
    AgentCapabilities,
    AgentContextPolicy,
    AgentDefinition,
    AgentPermissions,
    AgentType,
)
from core.agent_system import AgentSystemBuildStatus


class SupervisorHandler:
    calls: list[str] = []

    def __init__(self, agent_id: str) -> None:
        self._agent_id = agent_id

    @property
    def agent_id(self) -> str:
        return self._agent_id

    def handle(self, context: AgentContext) -> Mapping[str, object]:
        SupervisorHandler.calls.append(context.agent_id)
        return {"agent_id": context.agent_id, "ok": True}


def _agent(agent_id: str = "agent.a") -> AgentDefinition:
    return AgentDefinition(
        agent_id=agent_id,
        agent_type=AgentType.GENERAL,
        name=agent_id,
        description="Agent plan supervisor test agent.",
        capabilities=AgentCapabilities(capabilities=("cap.supervise",)),
        permissions=AgentPermissions(requires_confirmation=False),
        context_policy=AgentContextPolicy(allow_shared_context=True),
    )


def _task(
    task_id: str,
    *,
    agent_id: str = "agent.a",
    continue_on_failure: bool = False,
    logical_timeout_limit: int = 1,
) -> AgentCooperationTask:
    return AgentCooperationTask(
        task_id=task_id,
        objective_id=f"objective.{task_id}",
        execution_type=AgentCooperationExecutionType.SINGLE_AGENT,
        agent_id=agent_id,
        continue_on_failure=continue_on_failure,
        logical_timeout_limit=logical_timeout_limit,
    )


def _plan(
    task_ids: tuple[str, ...] = ("task.a",),
    *,
    dependencies: tuple[AgentCooperationDependency, ...] = (),
    policy: AgentCooperationPlanPolicy | None = None,
) -> AgentCooperationPlan:
    return AgentCooperationPlan(
        plan_id="plan.supervisor",
        tasks=tuple(_task(task_id) for task_id in task_ids),
        dependencies=dependencies,
        policy=policy or AgentCooperationPlanPolicy(enabled=True),
    )


def _task_result(
    task_id: str,
    status: AgentCooperationTaskStatus = AgentCooperationTaskStatus.SUCCESS,
    *,
    output: Mapping[str, object] | None = None,
) -> AgentCooperationTaskResult:
    return AgentCooperationTaskResult(
        task_id=task_id,
        status=status,
        execution_type=AgentCooperationExecutionType.SINGLE_AGENT,
        agent_ids=("agent.a",),
        output={"task_id": task_id, "ok": True} if output is None and status is AgentCooperationTaskStatus.SUCCESS else output,
    )


def _result(
    plan: AgentCooperationPlan,
    *,
    status: AgentCooperationPlanStatus = AgentCooperationPlanStatus.SUCCESS,
    task_results: tuple[AgentCooperationTaskResult, ...] | None = None,
    execution_order: tuple[str, ...] | None = None,
    outputs: Mapping[str, object] | None = None,
    plan_signature: str | None = None,
    request_signature: str = "a" * 64,
    metrics: Mapping[str, int] | None = None,
    events: tuple[AgentCooperationPlanEvent, ...] = (),
) -> AgentCooperationPlanResult:
    items = task_results if task_results is not None else tuple(_task_result(task.task_id) for task in plan.tasks)
    return AgentCooperationPlanResult(
        status=status,
        plan_id=plan.plan_id,
        plan_signature=plan_signature or agent_cooperation_plan_signature(plan),
        request_signature=request_signature,
        task_results=items,
        execution_order=execution_order if execution_order is not None else tuple(item.task_id for item in items),
        outputs={} if outputs is None else outputs,
        events=events,
        metrics=dict(metrics or {}),
    )


def _policy(**overrides: object) -> AgentPlanSupervisorPolicy:
    values = {"enabled": True}
    values.update(overrides)
    return AgentPlanSupervisorPolicy(**values)


def _supervise(
    plan: AgentCooperationPlan,
    result: AgentCooperationPlanResult,
    *,
    policy: AgentPlanSupervisorPolicy | None = None,
):
    return AgentPlanSupervisor().supervise(
        AgentPlanSupervisorRequest(plan=plan, execution_result=result, policy=policy or _policy())
    )


def _unsafe_task_result(
    task_id: str,
    output: object,
    status: AgentCooperationTaskStatus = AgentCooperationTaskStatus.SUCCESS,
) -> AgentCooperationTaskResult:
    item = object.__new__(AgentCooperationTaskResult)
    object.__setattr__(item, "task_id", task_id)
    object.__setattr__(item, "status", status)
    object.__setattr__(item, "execution_type", AgentCooperationExecutionType.SINGLE_AGENT)
    object.__setattr__(item, "agent_ids", ("agent.a",))
    object.__setattr__(item, "output", output)
    object.__setattr__(item, "execution_result", None)
    object.__setattr__(item, "error_code", None)
    object.__setattr__(item, "safe_message", None)
    object.__setattr__(item, "position", 0)
    return item


def test_policy_disabled_skips_supervision() -> None:
    plan = _plan()
    result = _result(plan)

    supervised = AgentPlanSupervisor().supervise(
        AgentPlanSupervisorRequest(plan=plan, execution_result=result)
    )

    assert supervised.status is AgentPlanSupervisorStatus.SUPERVISION_DISABLED
    assert supervised.decision.decision is AgentPlanSupervisorDecisionType.SKIPPED


def test_complete_success_is_coherent() -> None:
    plan = _plan(("task.a", "task.b"))
    supervised = _supervise(plan, _result(plan))

    assert supervised.status is AgentPlanSupervisorStatus.SUCCESS
    assert supervised.succeeded_tasks == ("task.a", "task.b")
    assert supervised.success_ratio == 1.0


def test_partial_success_is_coherent() -> None:
    plan = _plan(("task.a", "task.b"))
    result = _result(
        plan,
        status=AgentCooperationPlanStatus.PARTIAL_SUCCESS,
        task_results=(
            _task_result("task.a"),
            _task_result("task.b", AgentCooperationTaskStatus.FAILED),
        ),
    )

    supervised = _supervise(plan, result, policy=_policy(minimum_success_ratio=0.5))

    assert supervised.status is AgentPlanSupervisorStatus.PARTIAL_SUCCESS
    assert supervised.failed_tasks == ("task.b",)


def test_failed_result_is_coherent() -> None:
    plan = _plan()
    result = _result(
        plan,
        status=AgentCooperationPlanStatus.FAILED,
        task_results=(_task_result("task.a", AgentCooperationTaskStatus.FAILED),),
    )

    supervised = _supervise(plan, result, policy=_policy(minimum_success_ratio=0.0))

    assert supervised.status is AgentPlanSupervisorStatus.FAILED
    assert supervised.failed_tasks == ("task.a",)


def test_invalid_plan_signature_is_rejected() -> None:
    plan = _plan()
    result = _result(plan, plan_signature="0" * 64)

    supervised = _supervise(plan, result)

    assert supervised.status is AgentPlanSupervisorStatus.INVALID_SIGNATURE
    assert "PLAN_SIGNATURE_MISMATCH" in supervised.inconsistencies


def test_invalid_result_signature_is_rejected() -> None:
    plan = _plan()
    result = _result(plan, request_signature="not-a-signature")

    supervised = _supervise(plan, result)

    assert supervised.status is AgentPlanSupervisorStatus.INVALID_SIGNATURE
    assert "RESULT_SIGNATURE_MISSING" in supervised.inconsistencies


def test_result_from_another_plan_is_rejected() -> None:
    plan = _plan()
    result = _result(plan)
    object.__setattr__(result, "plan_id", "other.plan")

    supervised = _supervise(plan, result)

    assert supervised.status is AgentPlanSupervisorStatus.INCONSISTENT_RESULT
    assert "PLAN_ID_MISMATCH" in supervised.inconsistencies


def test_missing_required_task_is_detected() -> None:
    plan = _plan(("task.a", "task.b"))
    result = _result(plan, task_results=(_task_result("task.a"),), execution_order=("task.a",))

    supervised = _supervise(plan, result, policy=_policy(minimum_success_ratio=0.5))

    assert supervised.missing_tasks == ("task.b",)
    assert "MISSING_TASK_RESULTS" in supervised.inconsistencies


def test_unknown_task_result_is_detected() -> None:
    plan = _plan()
    result = _result(plan, task_results=(_task_result("task.a"), _task_result("task.unknown")))

    supervised = _supervise(plan, result)

    assert supervised.unknown_tasks == ("task.unknown",)
    assert "UNKNOWN_TASK_RESULTS" in supervised.inconsistencies


def test_duplicate_task_result_is_detected() -> None:
    plan = _plan()
    result = _result(plan, task_results=(_task_result("task.a"), _task_result("task.a")))

    supervised = _supervise(plan, result)

    assert supervised.duplicate_tasks == ("task.a",)
    assert "DUPLICATE_TASK_RESULTS" in supervised.inconsistencies


def test_unsatisfied_dependency_is_detected() -> None:
    dependency = AgentCooperationDependency("task.a", "task.b")
    plan = _plan(("task.a", "task.b"), dependencies=(dependency,))
    result = _result(
        plan,
        status=AgentCooperationPlanStatus.PARTIAL_SUCCESS,
        task_results=(
            _task_result("task.a", AgentCooperationTaskStatus.FAILED),
            _task_result("task.b"),
        ),
    )

    supervised = _supervise(plan, result, policy=_policy(minimum_success_ratio=0.5))

    assert "DEPENDENCIES_UNSATISFIED" in supervised.inconsistencies


def test_inconsistent_aggregate_status_is_detected() -> None:
    plan = _plan()
    result = _result(
        plan,
        status=AgentCooperationPlanStatus.SUCCESS,
        task_results=(_task_result("task.a", AgentCooperationTaskStatus.FAILED),),
    )

    supervised = _supervise(plan, result, policy=_policy(minimum_success_ratio=0.0))

    assert "AGGREGATE_STATUS_INCONSISTENT" in supervised.inconsistencies


def test_required_empty_output_is_detected() -> None:
    plan = _plan()
    result = _result(plan, task_results=(_task_result("task.a", output={}),)
)

    supervised = _supervise(plan, result)

    assert supervised.invalid_outputs == ("task.a",)
    assert "REQUIRED_OUTPUT_EMPTY" in supervised.inconsistencies


def test_invalid_output_structure_is_detected() -> None:
    plan = _plan()
    result = _result(plan, task_results=(_unsafe_task_result("task.a", {"bad": object()}),))

    supervised = _supervise(plan, result)

    assert "task.a" in supervised.invalid_outputs
    assert "INVALID_OUTPUT_STRUCTURE" in supervised.inconsistencies


def test_nested_sensitive_key_is_detected() -> None:
    plan = _plan()
    result = _result(plan, task_results=(_unsafe_task_result("task.a", {"nested": {"api_key": "x"}}),))

    supervised = _supervise(plan, result)

    assert "INVALID_OUTPUT_STRUCTURE" in supervised.inconsistencies


def test_nan_and_infinity_are_detected() -> None:
    plan = _plan(("task.a", "task.b"))
    result = _result(
        plan,
        task_results=(
            _unsafe_task_result("task.a", {"value": math.nan}),
            _unsafe_task_result("task.b", {"value": math.inf}),
        ),
    )

    supervised = _supervise(plan, result)

    assert supervised.invalid_outputs == ("task.a", "task.b")


def test_function_class_module_and_arbitrary_objects_are_detected() -> None:
    plan = _plan(("task.a", "task.b", "task.c", "task.d"))
    result = _result(
        plan,
        task_results=(
            _unsafe_task_result("task.a", {"value": test_function_class_module_and_arbitrary_objects_are_detected}),
            _unsafe_task_result("task.b", {"value": SupervisorHandler}),
            _unsafe_task_result("task.c", {"value": types}),
            _unsafe_task_result("task.d", {"value": object()}),
        ),
    )

    supervised = _supervise(plan, result)

    assert supervised.invalid_outputs == ("task.a", "task.b", "task.c", "task.d")


def test_task_limit_is_enforced() -> None:
    plan = _plan(("task.a", "task.b"))

    supervised = _supervise(plan, _result(plan), policy=_policy(max_tasks=1))

    assert supervised.status is AgentPlanSupervisorStatus.LIMIT_REACHED
    assert supervised.limits_reached == ("MAX_TASKS",)


def test_outputs_limit_is_enforced() -> None:
    plan = _plan()
    result = _result(plan, outputs={"a": 1, "b": 2})

    supervised = _supervise(plan, result, policy=_policy(max_outputs=1))

    assert supervised.status is AgentPlanSupervisorStatus.LIMIT_REACHED
    assert "MAX_OUTPUTS" in supervised.limits_reached


def test_depth_limit_is_enforced() -> None:
    plan = _plan()
    result = _result(plan, task_results=(_unsafe_task_result("task.a", {"a": {"b": {"c": 1}}}),))

    supervised = _supervise(plan, result, policy=_policy(max_depth=1))

    assert supervised.status is AgentPlanSupervisorStatus.LIMIT_REACHED
    assert "MAX_DEPTH" in supervised.limits_reached


def test_logical_time_limit_is_enforced() -> None:
    plan = AgentCooperationPlan(
        plan_id="plan.supervisor",
        tasks=(_task("task.a", logical_timeout_limit=2),),
        policy=AgentCooperationPlanPolicy(enabled=True),
    )

    supervised = _supervise(plan, _result(plan), policy=_policy(max_logical_time=1))

    assert supervised.status is AgentPlanSupervisorStatus.LIMIT_REACHED
    assert "MAX_LOGICAL_TIME" in supervised.limits_reached


def test_success_ratio_is_calculated() -> None:
    plan = _plan(("task.a", "task.b", "task.c", "task.d"))
    result = _result(
        plan,
        status=AgentCooperationPlanStatus.PARTIAL_SUCCESS,
        task_results=(
            _task_result("task.a"),
            _task_result("task.b"),
            _task_result("task.c", AgentCooperationTaskStatus.FAILED),
            _task_result("task.d", AgentCooperationTaskStatus.FAILED),
        ),
    )

    supervised = _supervise(plan, result, policy=_policy(minimum_success_ratio=0.5))

    assert supervised.success_ratio == 0.5


def test_metrics_are_reported() -> None:
    plan = _plan()

    supervised = _supervise(plan, _result(plan))

    assert supervised.metrics["agent_plan_supervisions_requested"] == 1
    assert supervised.metrics["agent_plan_supervision_tasks_expected"] == 1
    assert supervised.metrics["agent_plan_supervision_tasks_succeeded"] == 1


def test_events_are_reported() -> None:
    plan = _plan()

    supervised = _supervise(plan, _result(plan))

    names = tuple(event.name for event in supervised.events)
    assert "agent_plan_supervision_requested" in names
    assert "agent_plan_supervision_signature_checked" in names
    assert "agent_plan_supervision_completed" in names


def test_signature_is_deterministic() -> None:
    plan = _plan()
    result = _result(plan)
    first = AgentPlanSupervisorRequest(plan=plan, execution_result=result, policy=_policy())
    same = AgentPlanSupervisorRequest(plan=plan, execution_result=result, policy=_policy())
    changed = AgentPlanSupervisorRequest(
        plan=plan,
        execution_result=result,
        policy=_policy(minimum_success_ratio=0.5),
    )

    assert agent_plan_supervisor_request_signature(first) == agent_plan_supervisor_request_signature(same)
    assert agent_plan_supervisor_request_signature(first) != agent_plan_supervisor_request_signature(changed)


def test_public_results_are_immutable() -> None:
    plan = _plan()
    supervised = _supervise(plan, _result(plan))

    with pytest.raises(FrozenInstanceError):
        supervised.status = AgentPlanSupervisorStatus.FAILED
    with pytest.raises(TypeError):
        supervised.metrics["x"] = 1


def test_supervision_does_not_mutate_plan() -> None:
    plan = _plan()
    before = agent_cooperation_plan_signature(plan)

    _supervise(plan, _result(plan))

    assert agent_cooperation_plan_signature(plan) == before


def test_supervision_does_not_mutate_result() -> None:
    plan = _plan()
    result = _result(plan)
    before = result.task_results

    _supervise(plan, result)

    assert result.task_results is before


def test_agent_system_exposes_supervisor() -> None:
    built = build_core_agent_system()

    assert built.status is AgentSystemBuildStatus.COMPLETED
    assert isinstance(built.system.agent_plan_supervisor, AgentPlanSupervisor)


def test_building_agent_system_does_not_run_supervision_or_agents() -> None:
    SupervisorHandler.calls = []

    built = build_core_agent_system()

    assert built.status is AgentSystemBuildStatus.COMPLETED
    assert SupervisorHandler.calls == []


def test_compatibility_with_12_4_12_5_and_12_6_services() -> None:
    built = build_core_agent_system()
    system = built.system

    assert system.agent_cooperation_planner is not None
    assert system.agent_cooperation_automatic_planner is not None
    assert system.agent_cooperation_automatic_execution_service is not None
    assert system.agent_plan_supervisor is not None


def test_e2e_with_real_automatic_execution_result_without_external_services() -> None:
    SupervisorHandler.calls = []
    built = build_core_agent_system()
    system = built.system
    system.agent_registry.register(_agent())
    system.agent_handler_registry.register(SupervisorHandler("agent.a"))
    planning_policy = AgentCooperationPlanningPolicy(enabled=True)
    execution_policy = AgentCooperationPlanPolicy(
        enabled=True,
        max_tasks=16,
        max_dependencies=48,
        max_depth=8,
        max_total_executions=16,
        max_output_items=256,
        failure_mode=AgentCooperationFailureMode.REQUIRE_ALL_SUCCESS,
        require_all_success=True,
    )
    automatic = system.agent_cooperation_automatic_execution_service.execute(
        AgentCooperationAutomaticExecutionRequest(
            planning_request=AgentCooperationPlanningRequest(
                objective_id="objective.supervisor",
                objective_type=AgentCooperationObjectiveType.ANALYSIS,
                required_agent_ids=("agent.a",),
            ),
            policy=AgentCooperationAutomaticExecutionPolicy(
                enabled=True,
                planning_policy=planning_policy,
                execution_policy=execution_policy,
                execute_generated_plan=True,
            ),
        )
    )
    assert automatic.status is AgentCooperationAutomaticExecutionStatus.EXECUTION_SUCCEEDED
    assert automatic.generated_plan is not None

    supervised = system.agent_plan_supervisor.supervise(
        AgentPlanSupervisorRequest(
            plan=automatic.generated_plan,
            execution_result=automatic,
            policy=_policy(),
        )
    )

    assert supervised.status is AgentPlanSupervisorStatus.SUCCESS
    assert SupervisorHandler.calls == ("agent.a",) or SupervisorHandler.calls == ["agent.a"]


def test_invalid_policy_values_are_rejected() -> None:
    with pytest.raises(InvalidAgentPlanSupervisorRequestError):
        AgentPlanSupervisorPolicy(enabled=True, max_tasks=0)
