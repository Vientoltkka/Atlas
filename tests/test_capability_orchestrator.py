from __future__ import annotations

from typing import Any

import pytest

from bootstrap.capability_orchestrator import build_core_capability_orchestrator
from core.capability_orchestrator import (
    CapabilityOrchestrationError,
    CapabilityOrchestrationPolicy,
    CapabilityOrchestrationRequest,
    CapabilityOrchestrationStatus,
    InvalidCapabilityOrchestrationRequestError,
)
from core.capability_planner import (
    CapabilityPlanner,
    CapabilityPlanningDecision,
    CapabilityPlanningError,
    CapabilityPlanningRequest,
    CapabilityPlanningStatus,
    capability_planning_request_signature,
)
from core.capability_resolver import CapabilityResolver
from core.execution_plan_executor import ExecutionControl, ExecutionPlanExecutor
from core.execution_plan_library import WorkflowDefinition
from core.execution_plan_registry import ExecutionPlanReference
from core.execution_plan_validator import ExecutionPlanValidator
from core.planner import ExecutionPlan, ExecutionStep
from core.workflow_selector import WorkflowSelector
from tools.base_tool import BaseTool
from tools.registry import ToolRegistry
from tools.tool_context import ToolContext


class SpyTool(BaseTool):
    def __init__(self, name: str = "demo.tool", output: Any = "ok", *, fail: bool = False) -> None:
        self._name = name
        self._output = output
        self._fail = fail
        self.calls = 0

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return "Safe spy tool."

    def execute(self, context: ToolContext) -> Any:
        del context
        self.calls += 1
        if self._fail:
            raise RuntimeError("tool failed")
        return self._output


class StaticPlanner(CapabilityPlanner):
    def __init__(
        self,
        decision: CapabilityPlanningDecision | None = None,
        *,
        fail: bool = False,
        invalid_return: bool = False,
    ) -> None:
        super().__init__(CapabilityResolver(()), WorkflowSelector())
        self.decision = decision
        self.fail = fail
        self.invalid_return = invalid_return
        self.calls = 0

    def plan(self, request: CapabilityPlanningRequest):  # type: ignore[override]
        self.calls += 1
        if self.fail:
            raise CapabilityPlanningError("planning failed")
        if self.invalid_return:
            return object()
        assert self.decision is not None
        return self.decision


def _plan(
    tool_name: str = "demo.tool",
    *,
    required_tools: tuple[str, ...] | None = None,
    requires_confirmation: bool = False,
) -> ExecutionPlan:
    return ExecutionPlan(
        goal="Execute selected workflow.",
        ordered_steps=(ExecutionStep("step_1", "Run selected workflow.", tool_name),),
        estimated_steps=1,
        required_tools=(tool_name,) if required_tools is None else required_tools,
        detected_risks=("confirmation-gated operation",) if requires_confirmation else (),
        requires_confirmation=requires_confirmation,
    )


def _workflow(plan: ExecutionPlan) -> WorkflowDefinition:
    return WorkflowDefinition(
        reference=ExecutionPlanReference("workflow.demo", "1.0"),
        plan=plan,
        title="Demo workflow",
        description="Safe workflow used by orchestration tests.",
        category="demo",
        tags=("demo",),
    )


def _decision(
    status: CapabilityPlanningStatus,
    *,
    plan: ExecutionPlan | None = None,
) -> CapabilityPlanningDecision:
    selected_workflow = _workflow(plan) if plan is not None else None
    return CapabilityPlanningDecision(
        status=status,
        request_signature=capability_planning_request_signature(CapabilityPlanningRequest("Execute demo workflow")),
        capability_resolution_result=None,
        workflow_selection_result=None,
        selected_capability=None,
        selected_workflow=selected_workflow,
        selected_workflow_reference=selected_workflow.reference if selected_workflow is not None else None,
        plan=plan,
        reasons=("test decision",),
        library_id="atlas.test" if plan is not None else None,
        plan_id="workflow.demo" if plan is not None else None,
        version="1.0" if plan is not None else None,
        plan_signature="test-signature" if plan is not None else None,
    )


def _registry(*tools: BaseTool) -> ToolRegistry:
    registry = ToolRegistry()
    for tool in tools:
        registry.register(tool)
    return registry


def _orchestrator(planner: CapabilityPlanner, registry: ToolRegistry):
    return build_core_capability_orchestrator(
        planner,
        ExecutionPlanValidator(registry),
        ExecutionPlanExecutor(registry),
    )


def _request(*, control: ExecutionControl | None = None) -> CapabilityOrchestrationRequest:
    return CapabilityOrchestrationRequest(
        CapabilityPlanningRequest("Execute demo workflow"),
        policy=CapabilityOrchestrationPolicy(control=control),
        metadata={"trace": "safe"},
    )


def test_selected_valid_plan_is_validated_and_executed() -> None:
    tool = SpyTool(output={"result": "done"})
    plan = _plan()
    result = _orchestrator(StaticPlanner(_decision(CapabilityPlanningStatus.SELECTED, plan=plan)), _registry(tool)).orchestrate(
        _request()
    )

    assert result.status is CapabilityOrchestrationStatus.COMPLETED
    assert result.completed is True
    assert result.planning_decision is not None
    assert result.selected_plan is plan
    assert result.validation_result is not None
    assert result.validation_result.is_valid is True
    assert result.execution_result is not None
    assert result.execution_result.success is True
    assert result.execution_result.completed_steps == ["step_1"]
    assert tool.calls == 1
    assert [event.name for event in result.events] == [
        "capability_orchestration_started",
        "capability_planning_started",
        "capability_planning_succeeded",
        "capability_plan_validation_started",
        "capability_plan_validation_succeeded",
        "capability_execution_started",
        "capability_execution_succeeded",
        "capability_orchestration_completed",
    ]


@pytest.mark.parametrize(
    ("planning_status", "orchestration_status"),
    (
        (CapabilityPlanningStatus.NO_CAPABILITY_CANDIDATES, CapabilityOrchestrationStatus.NO_CAPABILITY_CANDIDATES),
        (CapabilityPlanningStatus.CAPABILITY_AMBIGUOUS, CapabilityOrchestrationStatus.CAPABILITY_AMBIGUOUS),
        (CapabilityPlanningStatus.NO_WORKFLOW_CANDIDATES, CapabilityOrchestrationStatus.NO_WORKFLOW_CANDIDATES),
        (
            CapabilityPlanningStatus.WORKFLOW_BELOW_MINIMUM_SCORE,
            CapabilityOrchestrationStatus.WORKFLOW_BELOW_MINIMUM_SCORE,
        ),
        (CapabilityPlanningStatus.WORKFLOW_AMBIGUOUS, CapabilityOrchestrationStatus.WORKFLOW_AMBIGUOUS),
        (CapabilityPlanningStatus.INVALID_REQUEST, CapabilityOrchestrationStatus.INVALID_REQUEST),
    ),
)
def test_unselected_planning_states_do_not_validate_or_execute(
    planning_status: CapabilityPlanningStatus,
    orchestration_status: CapabilityOrchestrationStatus,
) -> None:
    tool = SpyTool()
    result = _orchestrator(StaticPlanner(_decision(planning_status)), _registry(tool)).orchestrate(_request())

    assert result.status is orchestration_status
    assert result.validation_result is None
    assert result.execution_result is None
    assert tool.calls == 0
    assert "capability_plan_validation_started" not in {event.name for event in result.events}
    assert "capability_execution_started" not in {event.name for event in result.events}


def test_unmapped_planning_failure_state_does_not_validate_or_execute() -> None:
    tool = SpyTool()
    result = _orchestrator(StaticPlanner(_decision(CapabilityPlanningStatus.SELECTION_FAILED)), _registry(tool)).orchestrate(
        _request()
    )

    assert result.status is CapabilityOrchestrationStatus.PLANNING_FAILED
    assert result.validation_result is None
    assert result.execution_result is None
    assert tool.calls == 0


def test_invalid_request_and_planner_failures_are_structured_results() -> None:
    registry = _registry(SpyTool())
    orchestrator = _orchestrator(StaticPlanner(fail=True), registry)

    invalid = orchestrator.orchestrate(object())  # type: ignore[arg-type]
    failed = orchestrator.orchestrate(_request())

    assert invalid.status is CapabilityOrchestrationStatus.INVALID_REQUEST
    assert invalid.error_code == "INVALID_REQUEST"
    assert failed.status is CapabilityOrchestrationStatus.PLANNING_FAILED
    assert failed.error_code == "CapabilityPlanningError"


def test_invalid_selected_plan_is_not_executed() -> None:
    tool = SpyTool()
    plan = _plan(required_tools=())
    result = _orchestrator(StaticPlanner(_decision(CapabilityPlanningStatus.SELECTED, plan=plan)), _registry(tool)).orchestrate(
        _request()
    )

    assert result.status is CapabilityOrchestrationStatus.PLAN_VALIDATION_FAILED
    assert result.validation_result is not None
    assert result.validation_result.is_valid is False
    assert result.execution_result is None
    assert tool.calls == 0


def test_execution_failure_and_cancellation_are_mapped() -> None:
    failing_tool = SpyTool(fail=True)
    failing_plan = _plan()
    failed = _orchestrator(
        StaticPlanner(_decision(CapabilityPlanningStatus.SELECTED, plan=failing_plan)),
        _registry(failing_tool),
    ).orchestrate(_request())

    cancelling_tool = SpyTool()
    cancelled = _orchestrator(
        StaticPlanner(_decision(CapabilityPlanningStatus.SELECTED, plan=_plan())),
        _registry(cancelling_tool),
    ).orchestrate(_request(control=ExecutionControl(should_cancel=lambda: True)))

    assert failed.status is CapabilityOrchestrationStatus.EXECUTION_FAILED
    assert failed.execution_result is not None
    assert failed.execution_result.success is False
    assert failing_tool.calls == 1
    assert cancelled.status is CapabilityOrchestrationStatus.CANCELLED
    assert cancelled.execution_result is not None
    assert cancelled.execution_result.cancelled is True
    assert cancelling_tool.calls == 0


def test_public_contracts_are_immutable_and_observability_is_safe() -> None:
    observed = []
    tool = SpyTool()
    registry = _registry(tool)
    orchestrator = build_core_capability_orchestrator(
        StaticPlanner(_decision(CapabilityPlanningStatus.SELECTED, plan=_plan())),
        ExecutionPlanValidator(registry),
        ExecutionPlanExecutor(registry),
        observer=observed.append,
    )
    result = orchestrator.orchestrate(_request())

    assert observed == list(result.events)
    assert result.metadata == {}
    with pytest.raises(TypeError):
        result.events[0].details["secret"] = "value"  # type: ignore[index]
    assert all("arguments" not in event.details for event in result.events)
    assert all("output" not in event.details for event in result.events)
    with pytest.raises(InvalidCapabilityOrchestrationRequestError):
        CapabilityOrchestrationPolicy(control=object())  # type: ignore[arg-type]
    with pytest.raises(CapabilityOrchestrationError):
        build_core_capability_orchestrator(object(), ExecutionPlanValidator(), ExecutionPlanExecutor(registry))  # type: ignore[arg-type]
