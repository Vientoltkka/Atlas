from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from bootstrap.bootstrap import Bootstrap
from bootstrap.capability_execution_service import build_capability_execution_service
from bootstrap.capability_orchestrator import build_core_capability_orchestrator
from bootstrap.capability_planner import build_core_capability_planner
from bootstrap.capability_resolver import build_core_capability_resolver
from bootstrap.workflow_selector import build_core_workflow_selector
from core.capability_execution_service import (
    CapabilityExecutionRequest,
    CapabilityExecutionResult,
    CapabilityExecutionSelectedCapability,
    CapabilityExecutionService,
    CapabilityExecutionStatus,
    InvalidCapabilityExecutionRequestError,
    unavailable_capability_execution_result,
)
from core.capability_orchestrator import CapabilityOrchestrationRequest
from core.capability_planner import (
    CapabilityPlanner,
    CapabilityPlanningDecision,
    CapabilityPlanningRequest,
    CapabilityPlanningStatus,
    capability_planning_request_signature,
)
from core.capability_resolver import CapabilityResolver, CapabilityType
from core.execution_plan_executor import ExecutionControl, ExecutionPlanExecutor
from core.execution_replanner import ReplanningPolicy, ReplanningStrategy
from core.goal_verifier import GoalVerificationReason, OutputValidatorKind
from core.goal_driven_execution import GoalDrivenExecutionPolicy
from core.execution_plan_library import ExecutionPlanLibrary, WorkflowDefinition
from core.execution_plan_registry import ExecutionPlanReference
from core.execution_plan_validator import ExecutionPlanValidator
from core.orchestrator import AtlasOrchestrator
from core.planner import ExecutionPlan, ExecutionStep
from core.router import Router
from core.step_output_reference import StepOutputReference
from core.workflow_selector import WorkflowSelector
from memory.conversation import ConversationMemory
from tools.base_tool import BaseTool
from tools.filesystem.list_directory_tool import ListDirectoryTool
from tools.registry import ToolRegistry
from tools.tool_context import ToolContext


class SpyTool(BaseTool):
    def __init__(self, name: str = "demo.tool", output: Any = "ok", *, fail: bool = False) -> None:
        self._name = name
        self._output = output
        self._fail = fail
        self.calls = 0
        self.contexts: list[ToolContext] = []

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return "Safe capability execution test tool."

    def execute(self, context: ToolContext) -> Any:
        self.calls += 1
        self.contexts.append(context)
        if self._fail:
            raise RuntimeError("tool failed")
        return self._output


class StaticPlanner(CapabilityPlanner):
    def __init__(self, decision: CapabilityPlanningDecision | None = None) -> None:
        super().__init__(CapabilityResolver(()), WorkflowSelector())
        self.decision = decision
        self.calls = 0

    def plan(self, request: CapabilityPlanningRequest) -> CapabilityPlanningDecision:
        self.calls += 1
        assert self.decision is not None
        return self.decision


class CountingService:
    def __init__(self, result: CapabilityExecutionResult) -> None:
        self.result = result
        self.calls = 0
        self.requests: list[CapabilityExecutionRequest] = []

    def execute(self, request: CapabilityExecutionRequest) -> CapabilityExecutionResult:
        self.calls += 1
        self.requests.append(request)
        return self.result


class ChatAgent:
    name = "chat"
    generated_path = None

    def __init__(self) -> None:
        self.calls = 0

    def run(self, *, model: str, messages):
        del model, messages
        self.calls += 1
        return "respuesta anterior"


def _step(tool_name: str = "demo.tool") -> ExecutionStep:
    return ExecutionStep("step_1", "Execute demo workflow.", tool_name)


def _plan(
    tool_name: str = "demo.tool",
    *,
    required_tools: tuple[str, ...] | None = None,
) -> ExecutionPlan:
    return ExecutionPlan(
        goal="Execute demo workflow.",
        ordered_steps=(_step(tool_name),),
        estimated_steps=1,
        required_tools=(tool_name,) if required_tools is None else required_tools,
        detected_risks=(),
        requires_confirmation=False,
    )


def _workflow(plan: ExecutionPlan, plan_id: str = "workflow.demo") -> WorkflowDefinition:
    return WorkflowDefinition(
        reference=ExecutionPlanReference(plan_id, "1.0"),
        plan=plan,
        title="Demo workflow",
        description="Safe capability execution workflow.",
        category="demo",
        tags=("demo",),
    )


def _registry(*tools: BaseTool) -> ToolRegistry:
    registry = ToolRegistry()
    for tool in tools:
        registry.register(tool)
    return registry


def _service_for_library(
    registry: ToolRegistry,
    library: ExecutionPlanLibrary,
) -> CapabilityExecutionService:
    resolver = build_core_capability_resolver(
        tool_registry=registry,
        execution_plan_libraries=(library,),
    )
    selector, _policy = build_core_workflow_selector()
    planner = build_core_capability_planner(
        capability_resolver=resolver,
        workflow_selector=selector,
        execution_plan_libraries=(library,),
    )
    validator = ExecutionPlanValidator(registry)
    executor = ExecutionPlanExecutor(registry)
    orchestrator = build_core_capability_orchestrator(planner, validator, executor)
    return build_capability_execution_service(orchestrator)


def _decision(status: CapabilityPlanningStatus, *, plan: ExecutionPlan | None = None) -> CapabilityPlanningDecision:
    workflow = _workflow(plan) if plan is not None else None
    return CapabilityPlanningDecision(
        status=status,
        request_signature=capability_planning_request_signature(CapabilityPlanningRequest("Execute demo workflow")),
        capability_resolution_result=None,
        workflow_selection_result=None,
        selected_capability=None,
        selected_workflow=workflow,
        selected_workflow_reference=workflow.reference if workflow is not None else None,
        plan=plan,
    )


def _service_with_static_planner(
    planner: CapabilityPlanner,
    registry: ToolRegistry,
) -> CapabilityExecutionService:
    return CapabilityExecutionService(
        build_core_capability_orchestrator(
            planner,
            ExecutionPlanValidator(registry),
            ExecutionPlanExecutor(registry),
        )
    )


def _atlas_orchestrator(capability_execution_service=None):
    agent = ChatAgent()
    registry = SimpleNamespace(get=lambda name: agent if name == "chat" else None)
    return AtlasOrchestrator(
        planner=SimpleNamespace(create_plan=lambda prompt: SimpleNamespace(task=prompt, objective=prompt)),
        router=Router(),
        model_manager=SimpleNamespace(choose_model=lambda agent_name: f"model:{agent_name}"),
        memory=ConversationMemory(),
        registry=registry,
        write_file=SimpleNamespace(execute=lambda *_args: "written"),
        capability_execution_service=capability_execution_service,
    ), agent


def test_service_executes_valid_request_with_real_chain_and_safe_output() -> None:
    tool = SpyTool(output={"public": "done", "api_token": "hidden"})
    registry = _registry(tool)
    library = ExecutionPlanLibrary("atlas.test", (_workflow(_plan()),), version="1.0")
    service = _service_for_library(registry, library)

    result = service.execute(CapabilityExecutionRequest(required_tags=("demo",), metadata={"source": "test"}))

    assert result.status is CapabilityExecutionStatus.COMPLETED
    assert result.completed is True
    assert isinstance(result.selected_capability, CapabilityExecutionSelectedCapability)
    assert result.selected_capability.capability_type == "workflow"
    assert result.selected_workflow_reference == ExecutionPlanReference("workflow.demo", "1.0")
    assert result.plan_signature
    assert result.execution_id
    assert result.execution_status == "completed"
    assert result.goal_verification_result is not None
    assert result.goal_verification_result.satisfied is True
    assert result.output == {"public": "done", "api_token": "[redacted]"}
    assert tool.calls == 1
    assert tool.contexts[0].parameters == {}


def test_service_replans_from_failed_goal_to_alternative_workflow_with_real_read_only_tool(tmp_path) -> None:
    (tmp_path / "README.md").write_text("atlas", encoding="utf-8")
    bad_plan = ExecutionPlan(
        goal="List a directory.",
        ordered_steps=(
            ExecutionStep(
                "step_1",
                "List directory.",
                "list_directory",
                arguments={"path": str(tmp_path)},
            ),
        ),
        estimated_steps=1,
        required_tools=("list_directory",),
        detected_risks=(),
        requires_confirmation=False,
        output={"items": StepOutputReference("step_1")},
        required_outputs=("entries",),
    )
    good_plan = ExecutionPlan(
        goal="List a directory.",
        ordered_steps=(
            ExecutionStep(
                "step_1",
                "List directory.",
                "list_directory",
                arguments={"path": str(tmp_path)},
            ),
        ),
        estimated_steps=1,
        required_tools=("list_directory",),
        detected_risks=(),
        requires_confirmation=False,
        output={"entries": StepOutputReference("step_1")},
        required_outputs=("entries",),
        output_validators={"entries": (OutputValidatorKind.NON_EMPTY_COLLECTION.value,)},
    )
    registry = _registry(ListDirectoryTool())
    library = ExecutionPlanLibrary(
        "atlas.test",
        (
            _workflow(bad_plan, "workflow.bad"),
            _workflow(good_plan, "workflow.good"),
        ),
        version="1.0",
    )
    service = _service_for_library(registry, library)

    result = service.execute(
        CapabilityExecutionRequest(
            required_tags=("demo",),
            require_unique_top_score=False,
            replanning_policy=ReplanningPolicy(
                enabled=True,
                max_replans=1,
                strategy=ReplanningStrategy.ALTERNATIVE_WORKFLOW,
                retryable_goal_reasons=(GoalVerificationReason.MISSING_REQUIRED_OUTPUTS.value,),
            ),
        )
    )

    assert result.status is CapabilityExecutionStatus.COMPLETED
    assert result.replanning_attempted is True
    assert result.replan_attempts == 1
    assert result.replanning_status == "REPLANNED"
    assert result.original_plan_signature != result.final_plan_signature
    assert result.goal_verification_result is not None
    assert result.goal_verification_result.satisfied is True
    assert result.output == {"entries": ("README.md",)}
    assert result.orchestration_result is not None
    assert result.orchestration_result.replanning_history[0]["workflow_plan_id"] == "workflow.good"


def test_service_can_activate_goal_driven_execution_policy() -> None:
    tool = SpyTool(output="done")
    plan = ExecutionPlan(
        goal="Execute goal-driven workflow.",
        ordered_steps=(ExecutionStep("step_1", "Run.", "demo.tool"),),
        estimated_steps=1,
        required_tools=("demo.tool",),
        detected_risks=(),
        requires_confirmation=False,
        output={"value": StepOutputReference("step_1")},
        required_outputs=("value",),
    )
    registry = _registry(tool)
    library = ExecutionPlanLibrary("atlas.test", (_workflow(plan),), version="1.0")
    service = _service_for_library(registry, library)

    result = service.execute(
        CapabilityExecutionRequest(
            required_tags=("demo",),
            goal_driven_policy=GoalDrivenExecutionPolicy(enabled=True, max_cycles=2),
        )
    )

    assert result.status is CapabilityExecutionStatus.COMPLETED
    assert result.goal_driven_status == "COMPLETED"
    assert result.orchestration_result is not None
    assert result.orchestration_result.goal_driven_result is not None
    assert len(result.orchestration_result.goal_driven_result.cycles) == 1
    assert tool.calls == 1


@pytest.mark.parametrize(
    ("planning_status", "execution_status"),
    (
        (CapabilityPlanningStatus.NO_CAPABILITY_CANDIDATES, CapabilityExecutionStatus.NO_CAPABILITY_CANDIDATES),
        (CapabilityPlanningStatus.CAPABILITY_AMBIGUOUS, CapabilityExecutionStatus.CAPABILITY_AMBIGUOUS),
        (CapabilityPlanningStatus.NO_WORKFLOW_CANDIDATES, CapabilityExecutionStatus.NO_WORKFLOW_CANDIDATES),
        (
            CapabilityPlanningStatus.WORKFLOW_BELOW_MINIMUM_SCORE,
            CapabilityExecutionStatus.WORKFLOW_BELOW_MINIMUM_SCORE,
        ),
        (CapabilityPlanningStatus.WORKFLOW_AMBIGUOUS, CapabilityExecutionStatus.WORKFLOW_AMBIGUOUS),
    ),
)
def test_service_maps_planning_stop_statuses_without_execution(
    planning_status: CapabilityPlanningStatus,
    execution_status: CapabilityExecutionStatus,
) -> None:
    tool = SpyTool()
    result = _service_with_static_planner(StaticPlanner(_decision(planning_status)), _registry(tool)).execute(
        CapabilityExecutionRequest()
    )

    assert result.status is execution_status
    assert result.execution_status is None
    assert result.output is None
    assert tool.calls == 0


def test_invalid_requests_are_rejected_or_reported() -> None:
    with pytest.raises(InvalidCapabilityExecutionRequestError):
        CapabilityExecutionRequest(capability_type=CapabilityType.TOOL)
    with pytest.raises(InvalidCapabilityExecutionRequestError):
        CapabilityExecutionRequest(metadata={"api_key": "secret"})

    service = _service_with_static_planner(StaticPlanner(_decision(CapabilityPlanningStatus.NO_CAPABILITY_CANDIDATES)), _registry())
    result = service.execute(object())  # type: ignore[arg-type]

    assert result.status is CapabilityExecutionStatus.INVALID_REQUEST
    assert result.error_code == "INVALID_REQUEST"


def test_validation_failure_and_execution_failure_do_not_leak_arguments() -> None:
    tool = SpyTool()
    invalid_plan = _plan(required_tools=())
    validation_failed = _service_with_static_planner(
        StaticPlanner(_decision(CapabilityPlanningStatus.SELECTED, plan=invalid_plan)),
        _registry(tool),
    ).execute(CapabilityExecutionRequest(metadata={"request_id": "safe"}))

    failing_tool = SpyTool(fail=True)
    execution_failed = _service_with_static_planner(
        StaticPlanner(_decision(CapabilityPlanningStatus.SELECTED, plan=_plan())),
        _registry(failing_tool),
    ).execute(CapabilityExecutionRequest(metadata={"request_id": "safe"}))

    assert validation_failed.status is CapabilityExecutionStatus.PLAN_VALIDATION_FAILED
    assert validation_failed.execution_status is None
    assert tool.calls == 0
    assert execution_failed.status is CapabilityExecutionStatus.EXECUTION_FAILED
    assert execution_failed.message == "Capability execution failed."
    assert "request_id" not in repr(execution_failed)


def test_execution_cancellation_is_mapped_without_running_tool() -> None:
    tool = SpyTool()
    result = _service_with_static_planner(
        StaticPlanner(_decision(CapabilityPlanningStatus.SELECTED, plan=_plan())),
        _registry(tool),
    ).execute(CapabilityExecutionRequest(control=ExecutionControl(should_cancel=lambda: True)))

    assert result.status is CapabilityExecutionStatus.CANCELLED
    assert result.execution_status == "cancelled"
    assert tool.calls == 0


def test_orchestrator_method_is_optional_and_delegates_once_without_touching_chat_flow() -> None:
    unavailable_orchestrator, unavailable_agent = _atlas_orchestrator()
    unavailable = unavailable_orchestrator.execute_capability(CapabilityExecutionRequest())

    expected = CapabilityExecutionResult(CapabilityExecutionStatus.COMPLETED, output={"ok": True})
    service = CountingService(expected)
    orchestrator, agent = _atlas_orchestrator(capability_execution_service=service)
    request = CapabilityExecutionRequest()
    delegated = orchestrator.execute_capability(request)
    chat = orchestrator.process_prompt("hola", confirm=lambda _prompt: "")

    assert unavailable == unavailable_capability_execution_result()
    assert unavailable_agent.calls == 0
    assert delegated is expected
    assert service.calls == 1
    assert service.requests == [request]
    assert chat == "respuesta anterior"
    assert agent.calls == 1


def test_bootstrap_builds_capability_service_reusing_validator_and_executor() -> None:
    orchestrator = Bootstrap.build()
    service = orchestrator._capability_execution_service  # type: ignore[attr-defined]
    structured = orchestrator._structured_execution_coordinator  # type: ignore[attr-defined]
    capability_orchestrator = service._capability_orchestrator  # type: ignore[union-attr]

    assert isinstance(service, CapabilityExecutionService)
    assert capability_orchestrator._execution_plan_validator is structured._validator
    assert capability_orchestrator._execution_plan_executor is structured._executor


def test_bootstrap_factory_can_execute_minimal_real_workflow() -> None:
    tool = SpyTool(output={"final": "ok"})
    registry = _registry(tool)
    library = ExecutionPlanLibrary("atlas.test", (_workflow(_plan()),), version="1.0")
    service = _service_for_library(registry, library)

    result = service.execute(
        CapabilityExecutionRequest(
            capability_id="workflow.atlas.test.workflow.demo.1.0",
            preferred_workflow_reference=ExecutionPlanReference("workflow.demo", "1.0"),
        )
    )

    assert result.status is CapabilityExecutionStatus.COMPLETED
    assert result.output == {"final": "ok"}
    assert tool.calls == 1
