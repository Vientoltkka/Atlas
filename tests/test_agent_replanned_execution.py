from __future__ import annotations

from collections.abc import Mapping
from dataclasses import FrozenInstanceError
import math
import types

import pytest

from bootstrap.agent_system import build_core_agent_system
from core.agent_context import AgentContext
from core.agent_cooperation_plan import (
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
    AgentPlanReplanningPolicy,
    AgentPlanReplanningRequest,
    AgentPlanReplanningStatus,
)
from core.agent_plan_supervisor import (
    AgentPlanSupervisorPolicy,
    AgentPlanSupervisorRequest,
)
from core.agent_replanned_execution import (
    AgentReplannedExecutionPolicy,
    AgentReplannedExecutionRequest,
    AgentReplannedExecutionService,
    AgentReplannedExecutionStatus,
    InvalidAgentReplannedExecutionRequestError,
    agent_replanned_execution_request_signature,
    build_core_agent_replanned_execution_service,
)
from core.agent_registry import (
    AgentCapabilities,
    AgentContextPolicy,
    AgentDefinition,
    AgentPermissions,
    AgentType,
)
from core.agent_system import AgentSystemBuildStatus


class ReplannedHandler:
    calls: list[str] = []
    fail = False
    output_count = 1

    def __init__(self, agent_id: str) -> None:
        self._agent_id = agent_id

    @property
    def agent_id(self) -> str:
        return self._agent_id

    def handle(self, context: AgentContext) -> Mapping[str, object]:
        ReplannedHandler.calls.append(context.agent_id)
        if ReplannedHandler.fail:
            raise RuntimeError("controlled failure")
        return {f"value_{index}": index for index in range(ReplannedHandler.output_count)}


def _definition(agent_id: str = "agent.a") -> AgentDefinition:
    return AgentDefinition(
        agent_id=agent_id,
        agent_type=AgentType.GENERAL,
        name=agent_id,
        description="Replanned execution test agent.",
        capabilities=AgentCapabilities(capabilities=("cap.execute",)),
        permissions=AgentPermissions(requires_confirmation=False),
        context_policy=AgentContextPolicy(allow_shared_context=True),
    )


def _task(task_id: str, *, timeout: int = 1) -> AgentCooperationTask:
    return AgentCooperationTask(
        task_id=task_id,
        objective_id=f"objective.{task_id}",
        execution_type=AgentCooperationExecutionType.SINGLE_AGENT,
        agent_id="agent.a",
        required_capability_ids=("cap.execute",),
        logical_timeout_limit=timeout,
    )


def _plan(task_ids: tuple[str, ...] = ("task.a",), *, timeout: int = 1) -> AgentCooperationPlan:
    return AgentCooperationPlan(
        plan_id="plan.replanned",
        tasks=tuple(_task(task_id, timeout=timeout) for task_id in task_ids),
        policy=AgentCooperationPlanPolicy(enabled=True, allow_partial_success=True),
    )


def _original_result(
    plan: AgentCooperationPlan,
    statuses: Mapping[str, AgentCooperationTaskStatus],
) -> AgentCooperationPlanResult:
    task_results = tuple(
        AgentCooperationTaskResult(
            task_id=task.task_id,
            status=statuses.get(task.task_id, AgentCooperationTaskStatus.SUCCESS),
            execution_type=AgentCooperationExecutionType.SINGLE_AGENT,
            agent_ids=("agent.a",),
            output={"ok": True} if statuses.get(task.task_id, AgentCooperationTaskStatus.SUCCESS) is AgentCooperationTaskStatus.SUCCESS else None,
        )
        for task in plan.tasks
    )
    return AgentCooperationPlanResult(
        status=AgentCooperationPlanStatus.PARTIAL_SUCCESS,
        plan_id=plan.plan_id,
        plan_signature=agent_cooperation_plan_signature(plan),
        request_signature="a" * 64,
        task_results=task_results,
        execution_order=tuple(item.task_id for item in task_results),
        metrics={},
    )


def _system():
    ReplannedHandler.calls = []
    ReplannedHandler.fail = False
    ReplannedHandler.output_count = 1
    built = build_core_agent_system()
    assert built.system is not None
    system = built.system
    system.agent_registry.register(_definition())
    system.agent_handler_registry.register(ReplannedHandler("agent.a"))
    return system


def _replanning_result(system, plan: AgentCooperationPlan, statuses: Mapping[str, AgentCooperationTaskStatus]):
    supervision = system.agent_plan_supervisor.supervise(
        AgentPlanSupervisorRequest(
            plan=plan,
            execution_result=_original_result(plan, statuses),
            policy=AgentPlanSupervisorPolicy(
                enabled=True,
                minimum_success_ratio=0.0,
                require_consistent_aggregate_status=False,
            ),
        )
    )
    return system.agent_plan_replanner.replan(
        AgentPlanReplanningRequest(
            original_plan=plan,
            supervision_result=supervision,
            policy=AgentPlanReplanningPolicy(
                enabled=True,
                fail_closed=False,
                allow_retry_failed_tasks=True,
            ),
        )
    )


def _request(replanning_result, **policy_overrides: object) -> AgentReplannedExecutionRequest:
    values = {
        "enabled": True,
        "execute_replanned_plan": True,
    }
    values.update(policy_overrides)
    return AgentReplannedExecutionRequest(
        replanning_result=replanning_result,
        policy=AgentReplannedExecutionPolicy(**values),
        execution_id="replanned.execution",
    )


def test_policy_disabled_by_default() -> None:
    system = _system()
    replanning = _replanning_result(system, _plan(), {"task.a": AgentCooperationTaskStatus.FAILED})

    result = system.agent_replanned_execution_service.execute(
        AgentReplannedExecutionRequest(replanning)
    )

    assert result.status is AgentReplannedExecutionStatus.DISABLED
    assert ReplannedHandler.calls == []


def test_valid_request_signature() -> None:
    system = _system()
    request = _request(_replanning_result(system, _plan(), {"task.a": AgentCooperationTaskStatus.FAILED}))

    assert len(agent_replanned_execution_request_signature(request)) == 64


def test_invalid_request_is_structured() -> None:
    result = _system().agent_replanned_execution_service.execute("bad")

    assert result.status is AgentReplannedExecutionStatus.INVALID_REQUEST


def test_valid_signature_executes() -> None:
    system = _system()
    replanning = _replanning_result(system, _plan(), {"task.a": AgentCooperationTaskStatus.FAILED})

    result = system.agent_replanned_execution_service.execute(_request(replanning))

    assert result.status is AgentReplannedExecutionStatus.SUCCESS
    assert result.proposed_plan_signature == replanning.proposed_plan_signature


def test_invalid_signature_is_rejected() -> None:
    system = _system()
    replanning = _replanning_result(system, _plan(), {"task.a": AgentCooperationTaskStatus.FAILED})
    object.__setattr__(replanning, "proposed_plan_signature", "0" * 64)

    result = system.agent_replanned_execution_service.execute(_request(replanning))

    assert result.status is AgentReplannedExecutionStatus.SIGNATURE_ERROR
    assert ReplannedHandler.calls == []


def test_repeated_plan_returns_duplicate_without_second_run() -> None:
    system = _system()
    request = _request(_replanning_result(system, _plan(), {"task.a": AgentCooperationTaskStatus.FAILED}))

    first = system.agent_replanned_execution_service.execute(request)
    second = system.agent_replanned_execution_service.execute(request)

    assert first.status is AgentReplannedExecutionStatus.SUCCESS
    assert second.status is AgentReplannedExecutionStatus.DUPLICATE
    assert ReplannedHandler.calls == ["agent.a"]


def test_idempotence_returns_same_signature() -> None:
    system = _system()
    request = _request(_replanning_result(system, _plan(), {"task.a": AgentCooperationTaskStatus.FAILED}))

    first = system.agent_replanned_execution_service.execute(request)
    second = system.agent_replanned_execution_service.execute(request)

    assert first.request_signature == second.request_signature
    assert second.metrics["agent_replanned_execution_duplicates"] == 1


def test_progress_required() -> None:
    system = _system()
    replanning = _replanning_result(system, _plan(), {"task.a": AgentCooperationTaskStatus.FAILED})

    result = system.agent_replanned_execution_service.execute(_request(replanning, require_progress=True))

    assert result.status is AgentReplannedExecutionStatus.SUCCESS


def test_without_progress_is_rejected() -> None:
    system = _system()
    replanning = _replanning_result(system, _plan(), {"task.a": AgentCooperationTaskStatus.FAILED})
    object.__setattr__(replanning, "progress_reasons", ())

    result = system.agent_replanned_execution_service.execute(_request(replanning, require_progress=True))

    assert result.status is AgentReplannedExecutionStatus.REJECTED
    assert ReplannedHandler.calls == []


def test_task_limit_is_rejected_before_run() -> None:
    system = _system()
    plan = _plan(("task.a", "task.b"))
    replanning = _replanning_result(system, plan, {"task.b": AgentCooperationTaskStatus.FAILED})

    result = system.agent_replanned_execution_service.execute(_request(replanning, max_tasks=1))

    assert result.status is AgentReplannedExecutionStatus.LIMIT_REACHED


def test_output_limit_is_reported_after_single_run() -> None:
    system = _system()
    replanning = _replanning_result(system, _plan(("task.a", "task.b")), {"task.b": AgentCooperationTaskStatus.FAILED})

    result = system.agent_replanned_execution_service.execute(_request(replanning, max_outputs=1))

    assert result.status is AgentReplannedExecutionStatus.LIMIT_REACHED
    assert ReplannedHandler.calls == ["agent.a", "agent.a"]


def test_logical_time_limit_is_rejected_before_run() -> None:
    system = _system()
    replanning = _replanning_result(system, _plan(timeout=2), {"task.a": AgentCooperationTaskStatus.FAILED})

    result = system.agent_replanned_execution_service.execute(_request(replanning, max_logical_time=1))

    assert result.status is AgentReplannedExecutionStatus.LIMIT_REACHED
    assert ReplannedHandler.calls == []


def test_successful_execution() -> None:
    system = _system()
    result = system.agent_replanned_execution_service.execute(
        _request(_replanning_result(system, _plan(), {"task.a": AgentCooperationTaskStatus.FAILED}))
    )

    assert result.status is AgentReplannedExecutionStatus.SUCCESS
    assert result.execution_result.status is AgentCooperationPlanStatus.SUCCESS


def test_structured_failure() -> None:
    system = _system()
    ReplannedHandler.fail = True
    result = system.agent_replanned_execution_service.execute(
        _request(_replanning_result(system, _plan(), {"task.a": AgentCooperationTaskStatus.FAILED}))
    )

    assert result.status is AgentReplannedExecutionStatus.FAILED


def test_partial_success() -> None:
    system = _system()
    plan = _plan(("task.a", "task.b"))
    ReplannedHandler.fail = True
    result = system.agent_replanned_execution_service.execute(
        _request(
            _replanning_result(system, plan, {"task.b": AgentCooperationTaskStatus.FAILED}),
            allow_partial_success=True,
        )
    )

    assert result.status in (AgentReplannedExecutionStatus.PARTIAL_SUCCESS, AgentReplannedExecutionStatus.FAILED)


def test_execution_result_is_sanitized() -> None:
    system = _system()
    result = system.agent_replanned_execution_service.execute(
        _request(_replanning_result(system, _plan(), {"task.a": AgentCooperationTaskStatus.FAILED}))
    )

    assert all(item.execution_result is None for item in result.execution_result.task_results)


def test_nan_rejected() -> None:
    system = _system()
    replanning = _replanning_result(system, _plan(), {"task.a": AgentCooperationTaskStatus.FAILED})

    with pytest.raises(InvalidAgentReplannedExecutionRequestError):
        AgentReplannedExecutionRequest(replanning, metadata={"value": math.nan})


def test_infinity_rejected() -> None:
    system = _system()
    replanning = _replanning_result(system, _plan(), {"task.a": AgentCooperationTaskStatus.FAILED})

    with pytest.raises(InvalidAgentReplannedExecutionRequestError):
        AgentReplannedExecutionRequest(replanning, metadata={"value": math.inf})


def test_arbitrary_objects_rejected() -> None:
    system = _system()
    replanning = _replanning_result(system, _plan(), {"task.a": AgentCooperationTaskStatus.FAILED})

    with pytest.raises(InvalidAgentReplannedExecutionRequestError):
        AgentReplannedExecutionRequest(replanning, metadata={"obj": object()})
    with pytest.raises(InvalidAgentReplannedExecutionRequestError):
        AgentReplannedExecutionRequest(replanning, metadata={"cls": ReplannedHandler})
    with pytest.raises(InvalidAgentReplannedExecutionRequestError):
        AgentReplannedExecutionRequest(replanning, metadata={"mod": types})


def test_secret_keys_rejected() -> None:
    system = _system()
    replanning = _replanning_result(system, _plan(), {"task.a": AgentCooperationTaskStatus.FAILED})

    with pytest.raises(InvalidAgentReplannedExecutionRequestError):
        AgentReplannedExecutionRequest(replanning, metadata={"api_key": "x"})


def test_compatibility_with_12_4() -> None:
    system = _system()
    assert system.agent_cooperation_planner is not None


def test_compatibility_with_12_5() -> None:
    system = _system()
    assert system.agent_cooperation_automatic_planner is not None


def test_compatibility_with_12_6() -> None:
    system = _system()
    assert system.agent_cooperation_automatic_execution_service is not None


def test_compatibility_with_12_7() -> None:
    system = _system()
    assert system.agent_plan_supervisor is not None


def test_compatibility_with_12_8() -> None:
    system = _system()
    assert system.agent_plan_replanner is not None


def test_agent_system_exposes_service_and_shared_identities() -> None:
    built = build_core_agent_system()
    assert built.status is AgentSystemBuildStatus.COMPLETED
    system = built.system

    assert isinstance(system.agent_replanned_execution_service, AgentReplannedExecutionService)
    assert system.agent_replanned_execution_service.agent_registry is system.agent_registry
    assert system.agent_replanned_execution_service.agent_resolver is system.agent_resolver
    assert system.agent_replanned_execution_service.agent_cooperation_planner is system.agent_cooperation_planner
    assert system.agent_replanned_execution_service.agent_plan_supervisor is system.agent_plan_supervisor
    assert system.agent_replanned_execution_service.agent_plan_replanner is system.agent_plan_replanner


def test_factory_builds_service() -> None:
    system = _system()
    service = build_core_agent_replanned_execution_service(
        agent_registry=system.agent_registry,
        agent_resolver=system.agent_resolver,
        agent_cooperation_planner=system.agent_cooperation_planner,
        agent_plan_supervisor=system.agent_plan_supervisor,
        agent_plan_replanner=system.agent_plan_replanner,
    )

    assert isinstance(service, AgentReplannedExecutionService)


def test_result_is_immutable() -> None:
    system = _system()
    result = system.agent_replanned_execution_service.execute(
        _request(_replanning_result(system, _plan(), {"task.a": AgentCooperationTaskStatus.FAILED}))
    )

    with pytest.raises(FrozenInstanceError):
        result.status = AgentReplannedExecutionStatus.FAILED
    with pytest.raises(TypeError):
        result.metrics["x"] = 1


def test_e2e_has_one_execution_and_no_second_pipeline_calls() -> None:
    system = _system()
    plan = _plan()
    replanning = _replanning_result(system, plan, {"task.a": AgentCooperationTaskStatus.FAILED})
    request = _request(replanning)

    result = system.agent_replanned_execution_service.execute(request)

    assert result.status is AgentReplannedExecutionStatus.SUCCESS
    assert result.replanning_result is replanning
    assert result.execution_result is not None
    assert ReplannedHandler.calls == ["agent.a"]
    assert tuple(event.name for event in result.events).count("agent_replanned_execution_started") == 1
