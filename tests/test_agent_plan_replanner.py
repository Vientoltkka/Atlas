from __future__ import annotations

from collections.abc import Mapping
from dataclasses import FrozenInstanceError
import math
import types

import pytest

from bootstrap.agent_plan_replanner import build_core_agent_plan_replanner
from bootstrap.agent_system import build_core_agent_system
from core.agent_cooperation_plan import (
    AgentCooperationDependency,
    AgentCooperationExecutionType,
    AgentCooperationPlan,
    AgentCooperationPlanPolicy,
    AgentCooperationPlanResult,
    AgentCooperationPlanStatus,
    AgentCooperationTask,
    AgentCooperationTaskResult,
    AgentCooperationTaskStatus,
    agent_cooperation_plan_signature,
)
from core.agent_plan_replanner import (
    AgentPlanReplanner,
    AgentPlanReplanningActionType,
    AgentPlanReplanningPolicy,
    AgentPlanReplanningRequest,
    AgentPlanReplanningStatus,
    InvalidAgentPlanReplanningRequestError,
    agent_plan_replanning_request_signature,
)
from core.agent_plan_supervisor import (
    AgentPlanSupervisor,
    AgentPlanSupervisorPolicy,
    AgentPlanSupervisorRequest,
)
from core.agent_registry import (
    AgentCapabilities,
    AgentContextPolicy,
    AgentDefinition,
    AgentPermissions,
    AgentRegistry,
    AgentType,
)
from core.agent_resolver import AgentResolver
from core.agent_system import AgentSystemBuildStatus


class CountingPlanner:
    calls = 0

    def run(self) -> None:
        CountingPlanner.calls += 1


def _definition(
    agent_id: str,
    *,
    capabilities: tuple[str, ...] = ("cap.replan",),
    permissions: AgentPermissions | None = None,
    enabled: bool = True,
    metadata: Mapping[str, object] | None = None,
) -> AgentDefinition:
    return AgentDefinition(
        agent_id=agent_id,
        agent_type=AgentType.GENERAL,
        name=agent_id,
        description="Deterministic replanner test agent.",
        capabilities=AgentCapabilities(capabilities=capabilities),
        permissions=permissions or AgentPermissions(requires_confirmation=False),
        context_policy=AgentContextPolicy(allow_shared_context=True),
        enabled=enabled,
        metadata={} if metadata is None else metadata,
    )


def _task(
    task_id: str,
    *,
    agent_id: str = "agent.a",
    capability: str = "cap.replan",
    permission_ids: tuple[str, ...] = (),
    excluded: tuple[str, ...] = (),
    required_ids: tuple[str, ...] = (),
    required_marker_ids: tuple[str, ...] = (),
    logical_timeout_limit: int = 1,
) -> AgentCooperationTask:
    return AgentCooperationTask(
        task_id=task_id,
        objective_id=f"objective.{task_id}",
        execution_type=AgentCooperationExecutionType.SINGLE_AGENT,
        agent_id=agent_id,
        required_capability_ids=(capability,),
        required_permission_ids=permission_ids,
        excluded_agent_ids=excluded,
        required_skill_ids=required_marker_ids,
        preferred_agent_ids=required_ids,
        logical_timeout_limit=logical_timeout_limit,
    )


def _plan(
    task_ids: tuple[str, ...] = ("task.a",),
    *,
    dependencies: tuple[AgentCooperationDependency, ...] = (),
    tasks: tuple[AgentCooperationTask, ...] | None = None,
) -> AgentCooperationPlan:
    return AgentCooperationPlan(
        plan_id="plan.replanner",
        tasks=tasks or tuple(_task(task_id) for task_id in task_ids),
        dependencies=dependencies,
        policy=AgentCooperationPlanPolicy(enabled=True),
    )


def _task_result(
    task_id: str,
    status: AgentCooperationTaskStatus = AgentCooperationTaskStatus.SUCCESS,
) -> AgentCooperationTaskResult:
    return AgentCooperationTaskResult(
        task_id=task_id,
        status=status,
        execution_type=AgentCooperationExecutionType.SINGLE_AGENT,
        agent_ids=("agent.a",),
        output={"task_id": task_id, "ok": status is AgentCooperationTaskStatus.SUCCESS}
        if status is AgentCooperationTaskStatus.SUCCESS
        else None,
    )


def _plan_result(
    plan: AgentCooperationPlan,
    statuses: Mapping[str, AgentCooperationTaskStatus],
    *,
    aggregate: AgentCooperationPlanStatus | None = None,
) -> AgentCooperationPlanResult:
    items = tuple(_task_result(task.task_id, statuses.get(task.task_id, AgentCooperationTaskStatus.SUCCESS)) for task in plan.tasks)
    status = aggregate
    if status is None:
        if all(item.status is AgentCooperationTaskStatus.SUCCESS for item in items):
            status = AgentCooperationPlanStatus.SUCCESS
        elif any(item.status is AgentCooperationTaskStatus.SUCCESS for item in items):
            status = AgentCooperationPlanStatus.PARTIAL_SUCCESS
        else:
            status = AgentCooperationPlanStatus.FAILED
    return AgentCooperationPlanResult(
        status=status,
        plan_id=plan.plan_id,
        plan_signature=agent_cooperation_plan_signature(plan),
        request_signature="a" * 64,
        task_results=items,
        execution_order=tuple(item.task_id for item in items),
        metrics={},
    )


def _supervision(
    plan: AgentCooperationPlan,
    statuses: Mapping[str, AgentCooperationTaskStatus],
):
    return AgentPlanSupervisor().supervise(
        AgentPlanSupervisorRequest(
            plan=plan,
            execution_result=_plan_result(plan, statuses),
            policy=AgentPlanSupervisorPolicy(
                enabled=True,
                minimum_success_ratio=0.0,
                require_consistent_aggregate_status=False,
            ),
        )
    )


def _replanner(
    definitions: tuple[AgentDefinition, ...] = (_definition("agent.a"),),
) -> AgentPlanReplanner:
    registry = AgentRegistry(definitions)
    return build_core_agent_plan_replanner(
        agent_registry=registry,
        agent_resolver=AgentResolver(registry),
        agent_cooperation_planner=CountingPlanner(),
        agent_plan_supervisor=AgentPlanSupervisor(),
    )


def _request(
    plan: AgentCooperationPlan,
    *,
    statuses: Mapping[str, AgentCooperationTaskStatus],
    policy: AgentPlanReplanningPolicy | None = None,
) -> AgentPlanReplanningRequest:
    return AgentPlanReplanningRequest(
        original_plan=plan,
        supervision_result=_supervision(plan, statuses),
        policy=policy or AgentPlanReplanningPolicy(enabled=True, fail_closed=False),
    )


def test_policy_disabled_by_default() -> None:
    plan = _plan()
    result = _replanner().replan(AgentPlanReplanningRequest(plan, _supervision(plan, {})))

    assert result.status is AgentPlanReplanningStatus.DISABLED
    assert result.proposed_plan is None


def test_valid_request_has_signature() -> None:
    plan = _plan()
    request = _request(plan, statuses={})

    assert len(agent_plan_replanning_request_signature(request)) == 64


def test_invalid_request_is_structured() -> None:
    result = _replanner().replan("bad")

    assert result.status is AgentPlanReplanningStatus.INVALID_REQUEST


def test_plan_signature_correct_is_accepted_for_analysis() -> None:
    plan = _plan()
    result = _replanner().replan(_request(plan, statuses={}))

    assert result.original_plan_signature == agent_cooperation_plan_signature(plan)


def test_plan_signature_mismatch_is_rejected() -> None:
    plan = _plan()
    supervision = _supervision(plan, {})
    object.__setattr__(supervision, "plan_signature", "0" * 64)
    request = AgentPlanReplanningRequest(plan, supervision, AgentPlanReplanningPolicy(enabled=True))

    result = _replanner().replan(request)

    assert result.status is AgentPlanReplanningStatus.PLAN_SIGNATURE_MISMATCH


def test_incompatible_supervision_is_rejected_when_fail_closed() -> None:
    plan = _plan()
    request = _request(
        plan,
        statuses={"task.a": AgentCooperationTaskStatus.FAILED},
        policy=AgentPlanReplanningPolicy(enabled=True, fail_closed=True),
    )

    result = _replanner().replan(request)

    assert result.status is AgentPlanReplanningStatus.SUPERVISION_REJECTED


def test_keep_original_when_all_tasks_succeeded() -> None:
    plan = _plan(("task.a", "task.b"))
    result = _replanner().replan(_request(plan, statuses={}))

    assert result.status is AgentPlanReplanningStatus.KEEP_ORIGINAL_PLAN
    assert result.proposed_plan is plan


def test_retry_failed_task_when_authorized() -> None:
    plan = _plan()
    request = _request(
        plan,
        statuses={"task.a": AgentCooperationTaskStatus.FAILED},
        policy=AgentPlanReplanningPolicy(enabled=True, fail_closed=False, allow_retry_failed_tasks=True),
    )

    result = _replanner().replan(request)

    assert result.status is AgentPlanReplanningStatus.REPLAN_PROPOSED
    assert result.retried_tasks == ("task.a",)
    assert result.proposed_plan_signature != result.original_plan_signature


def test_retry_blocked_by_policy() -> None:
    plan = _plan()
    result = _replanner().replan(_request(plan, statuses={"task.a": AgentCooperationTaskStatus.FAILED}))

    assert result.status is AgentPlanReplanningStatus.NO_RECOVERABLE_ACTION
    assert result.proposed_plan is None


def test_authorized_skip_of_unrecoverable_task() -> None:
    plan = _plan(("task.a", "task.b"))
    result = _replanner().replan(
        _request(
            plan,
            statuses={"task.b": AgentCooperationTaskStatus.FAILED},
            policy=AgentPlanReplanningPolicy(
                enabled=True,
                fail_closed=False,
                allow_skip_unrecoverable_tasks=True,
                allow_dependency_rebuild=True,
            ),
        )
    )

    assert result.status is AgentPlanReplanningStatus.REPLAN_PROPOSED
    assert result.skipped_tasks == ("task.b",)
    assert tuple(task.task_id for task in result.proposed_plan.tasks) == ("task.a",)


def test_skip_blocked_by_policy() -> None:
    plan = _plan()
    result = _replanner().replan(_request(plan, statuses={"task.a": AgentCooperationTaskStatus.SKIPPED}))

    assert result.status is AgentPlanReplanningStatus.NO_RECOVERABLE_ACTION


def test_authorized_removal_of_blocked_task() -> None:
    plan = _plan(("task.a", "task.b"))
    result = _replanner().replan(
        _request(
            plan,
            statuses={"task.b": AgentCooperationTaskStatus.BLOCKED},
            policy=AgentPlanReplanningPolicy(
                enabled=True,
                fail_closed=False,
                allow_remove_blocked_tasks=True,
                allow_dependency_rebuild=True,
            ),
        )
    )

    assert result.status is AgentPlanReplanningStatus.REPLAN_PROPOSED
    assert result.removed_tasks == ("task.b",)


def test_dependency_rebuild_removes_affected_edges() -> None:
    dependency = AgentCooperationDependency("task.a", "task.b")
    plan = _plan(("task.a", "task.b"), dependencies=(dependency,))
    result = _replanner().replan(
        _request(
            plan,
            statuses={"task.b": AgentCooperationTaskStatus.FAILED},
            policy=AgentPlanReplanningPolicy(
                enabled=True,
                fail_closed=False,
                allow_skip_unrecoverable_tasks=True,
                allow_dependency_rebuild=True,
            ),
        )
    )

    assert result.status is AgentPlanReplanningStatus.REPLAN_PROPOSED
    assert result.rebuilt_dependencies == ("plan",)
    assert result.proposed_plan.dependencies == ()


def test_cycle_detection_rejects_invalid_proposal() -> None:
    dependency = AgentCooperationDependency("task.a", "task.b")
    plan = _plan(("task.a", "task.b"), dependencies=(dependency,))
    result = _replanner().replan(
        _request(
            plan,
            statuses={"task.b": AgentCooperationTaskStatus.FAILED},
            policy=AgentPlanReplanningPolicy(enabled=True, fail_closed=False, allow_skip_unrecoverable_tasks=True),
        )
    )

    assert result.status is AgentPlanReplanningStatus.REPLAN_REJECTED


def test_missing_dependency_is_rejected_by_existing_model() -> None:
    with pytest.raises(Exception):
        _plan(("task.a",), dependencies=(AgentCooperationDependency("missing", "task.a"),))


def test_duplicate_task_is_rejected_by_existing_model() -> None:
    with pytest.raises(Exception):
        _plan(tasks=(_task("task.a"), _task("task.a")))


def test_unknown_task_in_supervision_blocks_replan() -> None:
    plan = _plan()
    supervision = _supervision(plan, {})
    object.__setattr__(supervision, "unknown_tasks", ("task.unknown",))
    request = AgentPlanReplanningRequest(plan, supervision, AgentPlanReplanningPolicy(enabled=True, fail_closed=False))

    result = _replanner().replan(request)

    assert result.status is AgentPlanReplanningStatus.NO_RECOVERABLE_ACTION


def test_compatible_agent_replacement() -> None:
    plan = _plan()
    result = _replanner((_definition("agent.a"), _definition("agent.b"))).replan(
        _request(
            plan,
            statuses={"task.a": AgentCooperationTaskStatus.FAILED},
            policy=AgentPlanReplanningPolicy(enabled=True, fail_closed=False, allow_agent_reselection=True),
        )
    )

    assert result.status is AgentPlanReplanningStatus.REPLAN_PROPOSED
    assert result.replaced_agents == ("task.a",)
    assert result.proposed_plan.tasks[0].agent_id == "agent.b"


def test_replacement_without_candidate() -> None:
    plan = _plan(tasks=(_task("task.a", capability="cap.missing"),))
    result = _replanner((_definition("agent.a"),)).replan(
        _request(
            plan,
            statuses={"task.a": AgentCooperationTaskStatus.FAILED},
            policy=AgentPlanReplanningPolicy(enabled=True, fail_closed=False, allow_agent_reselection=True),
        )
    )

    assert result.status is AgentPlanReplanningStatus.NO_RECOVERABLE_ACTION


def test_replacement_ambiguous_candidate() -> None:
    plan = _plan()
    result = _replanner((_definition("agent.a"), _definition("agent.b"), _definition("agent.c"))).replan(
        _request(
            plan,
            statuses={"task.a": AgentCooperationTaskStatus.FAILED},
            policy=AgentPlanReplanningPolicy(enabled=True, fail_closed=False, allow_agent_reselection=True),
        )
    )

    assert result.status is AgentPlanReplanningStatus.NO_RECOVERABLE_ACTION


def test_disabled_agent_is_not_selected() -> None:
    plan = _plan()
    result = _replanner((_definition("agent.a"), _definition("agent.b", enabled=False))).replan(
        _request(
            plan,
            statuses={"task.a": AgentCooperationTaskStatus.FAILED},
            policy=AgentPlanReplanningPolicy(enabled=True, fail_closed=False, allow_agent_reselection=True),
        )
    )

    assert result.status is AgentPlanReplanningStatus.NO_RECOVERABLE_ACTION


def test_missing_capability_prevents_replacement() -> None:
    plan = _plan(tasks=(_task("task.a", capability="cap.special"),))
    result = _replanner((_definition("agent.a"), _definition("agent.b", capabilities=("cap.other",)))).replan(
        _request(
            plan,
            statuses={"task.a": AgentCooperationTaskStatus.FAILED},
            policy=AgentPlanReplanningPolicy(enabled=True, fail_closed=False, allow_agent_reselection=True),
        )
    )

    assert result.status is AgentPlanReplanningStatus.NO_RECOVERABLE_ACTION


def test_missing_permission_prevents_replacement() -> None:
    plan = _plan(tasks=(_task("task.a", permission_ids=("can_use_network",)),))
    result = _replanner((_definition("agent.a"), _definition("agent.b"))).replan(
        _request(
            plan,
            statuses={"task.a": AgentCooperationTaskStatus.FAILED},
            policy=AgentPlanReplanningPolicy(enabled=True, fail_closed=False, allow_agent_reselection=True),
        )
    )

    assert result.status is AgentPlanReplanningStatus.NO_RECOVERABLE_ACTION


def test_required_marker_not_authorized_prevents_replacement() -> None:
    plan = _plan(tasks=(_task("task.a", required_marker_ids=("marker.safe",)),))
    result = _replanner(
        (
            _definition("agent.a"),
            _definition("agent.b", metadata={"allowed_skill_ids": "marker.other"}),
        )
    ).replan(
        _request(
            plan,
            statuses={"task.a": AgentCooperationTaskStatus.FAILED},
            policy=AgentPlanReplanningPolicy(enabled=True, fail_closed=False, allow_agent_reselection=True),
        )
    )

    assert result.status is AgentPlanReplanningStatus.NO_RECOVERABLE_ACTION


def test_limits_are_respected() -> None:
    plan = _plan(("task.a", "task.b"))
    result = _replanner().replan(
        _request(plan, statuses={}, policy=AgentPlanReplanningPolicy(enabled=True, max_tasks=1))
    )

    assert result.status is AgentPlanReplanningStatus.LIMIT_REACHED


def test_no_structural_progress_is_reported() -> None:
    plan = _plan()
    result = _replanner().replan(
        _request(
            plan,
            statuses={"task.a": AgentCooperationTaskStatus.FAILED},
            policy=AgentPlanReplanningPolicy(enabled=True, fail_closed=False, require_progress=True),
        )
    )

    assert result.status is AgentPlanReplanningStatus.NO_RECOVERABLE_ACTION


def test_result_is_immutable() -> None:
    plan = _plan()
    result = _replanner().replan(_request(plan, statuses={}))

    with pytest.raises(FrozenInstanceError):
        result.status = AgentPlanReplanningStatus.INTERNAL_ERROR
    with pytest.raises(TypeError):
        result.metrics["x"] = 1


def test_sensitive_keys_are_rejected() -> None:
    plan = _plan()

    with pytest.raises(InvalidAgentPlanReplanningRequestError):
        AgentPlanReplanningRequest(plan, _supervision(plan, {}), metadata={"api_key": "x"})


def test_functions_classes_and_modules_are_rejected() -> None:
    plan = _plan()

    with pytest.raises(InvalidAgentPlanReplanningRequestError):
        AgentPlanReplanningRequest(plan, _supervision(plan, {}), metadata={"fn": test_functions_classes_and_modules_are_rejected})
    with pytest.raises(InvalidAgentPlanReplanningRequestError):
        AgentPlanReplanningRequest(plan, _supervision(plan, {}), metadata={"cls": CountingPlanner})
    with pytest.raises(InvalidAgentPlanReplanningRequestError):
        AgentPlanReplanningRequest(plan, _supervision(plan, {}), metadata={"mod": types})


def test_nan_and_infinity_are_rejected() -> None:
    plan = _plan()

    with pytest.raises(InvalidAgentPlanReplanningRequestError):
        AgentPlanReplanningRequest(plan, _supervision(plan, {}), metadata={"value": math.nan})
    with pytest.raises(InvalidAgentPlanReplanningRequestError):
        AgentPlanReplanningRequest(plan, _supervision(plan, {}), metadata={"value": math.inf})


def test_signature_is_stable() -> None:
    plan = _plan()
    first = _request(plan, statuses={})
    same = _request(plan, statuses={})
    changed = _request(plan, statuses={}, policy=AgentPlanReplanningPolicy(enabled=True, max_actions=2))

    assert agent_plan_replanning_request_signature(first) == agent_plan_replanning_request_signature(same)
    assert agent_plan_replanning_request_signature(first) != agent_plan_replanning_request_signature(changed)


def test_events_are_safe_and_structured() -> None:
    plan = _plan()
    result = _replanner().replan(_request(plan, statuses={}))

    names = tuple(event.name for event in result.events)
    assert "agent_plan_replanning_requested" in names
    assert "agent_plan_replanning_completed" in names
    assert all("secret" not in str(event.details).lower() for event in result.events)


def test_metrics_are_correct() -> None:
    plan = _plan()
    result = _replanner().replan(
        _request(
            plan,
            statuses={"task.a": AgentCooperationTaskStatus.FAILED},
            policy=AgentPlanReplanningPolicy(enabled=True, fail_closed=False, allow_retry_failed_tasks=True),
        )
    )

    assert result.metrics["agent_plan_replanning_requests"] == 1
    assert result.metrics["agent_plan_replanning_tasks_retried"] == 1
    assert result.metrics["agent_plan_replanning_actions_created"] >= 1


def test_agent_system_integration() -> None:
    built = build_core_agent_system()

    assert built.status is AgentSystemBuildStatus.COMPLETED
    assert isinstance(built.system.agent_plan_replanner, AgentPlanReplanner)


def test_shared_dependency_identity() -> None:
    built = build_core_agent_system()
    system = built.system

    assert system.agent_plan_replanner.agent_registry is system.agent_registry
    assert system.agent_plan_replanner.agent_resolver is system.agent_resolver
    assert system.agent_plan_replanner.agent_cooperation_planner is system.agent_cooperation_planner
    assert system.agent_plan_replanner.agent_plan_supervisor is system.agent_plan_supervisor


def test_replanning_does_not_run_agent() -> None:
    CountingPlanner.calls = 0
    plan = _plan()
    replanner = _replanner()

    replanner.replan(
        _request(
            plan,
            statuses={"task.a": AgentCooperationTaskStatus.FAILED},
            policy=AgentPlanReplanningPolicy(enabled=True, fail_closed=False, allow_retry_failed_tasks=True),
        )
    )

    assert CountingPlanner.calls == 0


def test_no_runtime_services_are_called() -> None:
    plan = _plan()
    result = _replanner().replan(
        _request(
            plan,
            statuses={"task.a": AgentCooperationTaskStatus.FAILED},
            policy=AgentPlanReplanningPolicy(enabled=True, fail_closed=False, allow_retry_failed_tasks=True),
        )
    )

    assert result.proposed_plan is not None
    assert result.status is AgentPlanReplanningStatus.REPLAN_PROPOSED


def test_compatibility_with_phase_services() -> None:
    built = build_core_agent_system()
    system = built.system

    assert system.agent_cooperation_planner is not None
    assert system.agent_cooperation_automatic_planner is not None
    assert system.agent_cooperation_automatic_execution_service is not None
    assert system.agent_plan_supervisor is not None
    assert system.agent_plan_replanner is not None


def test_structured_e2e_from_supervision_to_replanning_proposal() -> None:
    dependency = AgentCooperationDependency("task.a", "task.b")
    plan = _plan(("task.a", "task.b"), dependencies=(dependency,))
    supervision = _supervision(plan, {"task.b": AgentCooperationTaskStatus.FAILED})
    replanner = _replanner()

    result = replanner.replan(
        AgentPlanReplanningRequest(
            original_plan=plan,
            supervision_result=supervision,
            policy=AgentPlanReplanningPolicy(
                enabled=True,
                fail_closed=False,
                allow_retry_failed_tasks=True,
            ),
            metadata={"case": "e2e"},
        )
    )

    assert result.status is AgentPlanReplanningStatus.REPLAN_PROPOSED
    assert result.proposed_plan is not None
    assert result.retried_tasks == ("task.b",)
    assert len(result.proposed_plan_signature) == 64
    assert result.metrics["agent_plan_replanning_tasks_retried"] == 1
    assert "agent_plan_replanning_plan_created" in tuple(event.name for event in result.events)


def test_build_factory_returns_replanner() -> None:
    registry = AgentRegistry((_definition("agent.a"),))
    resolver = AgentResolver(registry)

    replanner = build_core_agent_plan_replanner(
        agent_registry=registry,
        agent_resolver=resolver,
        agent_cooperation_planner=CountingPlanner(),
        agent_plan_supervisor=AgentPlanSupervisor(),
    )

    assert isinstance(replanner, AgentPlanReplanner)

