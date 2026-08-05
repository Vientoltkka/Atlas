from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from core.agent_context import AgentContext, AgentContextBuilder
from core.agent_executor import AgentExecutor, AgentHandlerRegistry
from core.agent_registry import (
    AgentCapabilities,
    AgentContextPolicy,
    AgentDefinition,
    AgentLimits,
    AgentPermissions,
    AgentRegistry,
    AgentType,
)
from core.agent_resolver import AgentResolver
from core.capability_execution_service import (
    CapabilityExecutionRequest,
    CapabilityExecutionService,
    CapabilityExecutionStatus,
)
from core.capability_orchestrator import (
    AgentExecutionPolicy,
    CapabilityOrchestrationPolicy,
    CapabilityOrchestrationRequest,
    CapabilityOrchestrationStatus,
    CapabilityOrchestrator,
)
from core.capability_planner import (
    CapabilityPlanner,
    CapabilityPlanningDecision,
    CapabilityPlanningRequest,
    CapabilityPlanningStatus,
    capability_planning_request_signature,
)
from core.capability_resolver import CapabilityResolver
from core.execution_plan_executor import ExecutionPlanExecutor
from core.execution_plan_library import WorkflowDefinition
from core.execution_plan_registry import ExecutionPlanReference
from core.execution_plan_validator import ExecutionPlanValidator
from core.multi_capability_planner import MultiCapabilityPlanner, MultiCapabilityPlanningRequest
from core.planner import ExecutionPlan, ExecutionStep
from core.workflow_selector import WorkflowSelector
from core.skill_execution_context import SkillExecutionContext
from tools.base_tool import BaseTool
from tools.registry import ToolRegistry
from tools.tool_context import ToolContext


class SpyTool(BaseTool):
    def __init__(self, output: Any = "ok") -> None:
        self.calls = 0
        self._output = output

    @property
    def name(self) -> str:
        return "demo.tool"

    @property
    def description(self) -> str:
        return "Safe test tool."

    def execute(self, context: ToolContext) -> Any:
        del context
        self.calls += 1
        return self._output


class StaticPlanner(CapabilityPlanner):
    def __init__(self, decision: CapabilityPlanningDecision) -> None:
        super().__init__(CapabilityResolver(()), WorkflowSelector())
        self.decision = decision
        self.calls = 0

    def plan(self, request: CapabilityPlanningRequest) -> CapabilityPlanningDecision:
        del request
        self.calls += 1
        return self.decision


class FailingMultiCapabilityPlanner(MultiCapabilityPlanner):
    def __init__(self) -> None:
        super().__init__(execution_plan_libraries=())
        self.calls = 0

    def plan(self, request: MultiCapabilityPlanningRequest):
        del request
        self.calls += 1
        raise AssertionError("multi-capability planner must not run for agent execution")


@dataclass(frozen=True)
class EchoAgentHandler:
    agent_id: str = "atlas.agent.echo"

    def handle(self, context: AgentContext):
        return {
            "agent_id": context.agent_id,
            "value": context.structured_input.get("value"),
            "token": "hidden",
        }


def _tool_registry(tool: SpyTool) -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(tool)
    return registry


def _plan() -> ExecutionPlan:
    return ExecutionPlan(
        goal="Execute workflow.",
        ordered_steps=(ExecutionStep("step_1", "Run tool.", "demo.tool"),),
        estimated_steps=1,
        required_tools=("demo.tool",),
        detected_risks=(),
        requires_confirmation=False,
    )


def _decision(plan: ExecutionPlan) -> CapabilityPlanningDecision:
    workflow = WorkflowDefinition(
        reference=ExecutionPlanReference("workflow.demo", "1.0"),
        plan=plan,
        title="Demo workflow",
        description="Safe workflow.",
        category="demo",
        tags=("demo",),
    )
    return CapabilityPlanningDecision(
        status=CapabilityPlanningStatus.SELECTED,
        request_signature=capability_planning_request_signature(CapabilityPlanningRequest("Execute workflow")),
        capability_resolution_result=None,
        workflow_selection_result=None,
        selected_capability=None,
        selected_workflow=workflow,
        selected_workflow_reference=workflow.reference,
        plan=plan,
    )


def _agent_definition(
    *,
    enabled: bool = True,
) -> AgentDefinition:
    return AgentDefinition(
        agent_id="atlas.agent.echo",
        agent_type=AgentType.GENERAL,
        name="Echo agent",
        description="Deterministic echo agent.",
        permissions=AgentPermissions(requires_confirmation=False),
        limits=AgentLimits(max_steps=1, max_tool_calls=0),
        capabilities=AgentCapabilities(capabilities=("agent.echo",)),
        context_policy=AgentContextPolicy(allow_shared_context=True),
        enabled=enabled,
    )


def _agent_executor(
    *,
    enabled: bool = True,
    with_handler: bool = True,
) -> AgentExecutor:
    registry = AgentRegistry((_agent_definition(enabled=enabled),))
    handler_registry = AgentHandlerRegistry((EchoAgentHandler(),) if with_handler else ())
    return AgentExecutor(
        AgentResolver(registry),
        AgentContextBuilder(),
        handler_registry,
    )


def _orchestrator(
    planner: StaticPlanner,
    tool: SpyTool,
    *,
    agent_executor: AgentExecutor | None = None,
    observed: list | None = None,
) -> CapabilityOrchestrator:
    registry = _tool_registry(tool)
    return CapabilityOrchestrator(
        planner,
        ExecutionPlanValidator(registry),
        ExecutionPlanExecutor(registry),
        agent_executor=agent_executor,
        observer=observed.append if observed is not None else None,
    )


def _agent_policy(
    *,
    preferred_agent_id: str = "atlas.agent.echo",
) -> AgentExecutionPolicy:
    return AgentExecutionPolicy(
        enabled=True,
        allow_agent_execution=True,
        preferred_agent_id=preferred_agent_id,
        require_explicit_agent=True,
        fail_if_agent_missing=True,
        required_capability_ids=("agent.echo",),
    )


def test_disabled_agent_policy_preserves_existing_capability_flow() -> None:
    tool = SpyTool(output={"legacy": True})
    planner = StaticPlanner(_decision(_plan()))
    result = _orchestrator(planner, tool).orchestrate(
        CapabilityOrchestrationRequest(
            CapabilityPlanningRequest("Execute workflow"),
            inputs={"value": 7},
        )
    )

    assert result.status is CapabilityOrchestrationStatus.COMPLETED
    assert result.agent_execution_result is None
    assert result.checkpoint == {}
    assert result.metrics == {}
    assert planner.calls == 1
    assert tool.calls == 1
    assert "agent_pipeline_started" not in {event.name for event in result.events}


def test_enabled_agent_policy_runs_agent_pipeline_without_legacy_execution() -> None:
    tool = SpyTool()
    planner = StaticPlanner(_decision(_plan()))
    observed = []
    result = _orchestrator(
        planner,
        tool,
        agent_executor=_agent_executor(),
        observed=observed,
    ).orchestrate(
        CapabilityOrchestrationRequest(
            CapabilityPlanningRequest("Execute workflow"),
            policy=CapabilityOrchestrationPolicy(agent_execution_policy=_agent_policy()),
            inputs={"value": 42},
            metadata={"request_id": "safe"},
        )
    )

    assert result.status is CapabilityOrchestrationStatus.COMPLETED
    assert result.agent_execution_result is not None
    assert result.agent_execution_result.output == {"agent_id": "atlas.agent.echo", "value": 42}
    assert planner.calls == 0
    assert tool.calls == 0
    assert observed == list(result.events)
    assert [event.name for event in result.events] == [
        "capability_orchestration_started",
        "agent_pipeline_started",
        "agent_pipeline_completed",
        "capability_orchestration_completed",
    ]


def test_agent_pipeline_reports_missing_agent() -> None:
    tool = SpyTool()
    planner = StaticPlanner(_decision(_plan()))
    result = _orchestrator(planner, tool, agent_executor=_agent_executor()).orchestrate(
        CapabilityOrchestrationRequest(
            CapabilityPlanningRequest("Execute workflow"),
            policy=CapabilityOrchestrationPolicy(
                agent_execution_policy=_agent_policy(preferred_agent_id="atlas.agent.missing")
            ),
        )
    )

    assert result.status is CapabilityOrchestrationStatus.NO_CAPABILITY_CANDIDATES
    assert result.error_code in (None, "NO_AGENT_CANDIDATES")
    assert result.metrics == {
        "agent_pipeline_started": 1,
        "agent_pipeline_completed": 0,
        "agent_pipeline_failed": 1,
    }


def test_agent_pipeline_reports_disabled_agent() -> None:
    tool = SpyTool()
    planner = StaticPlanner(_decision(_plan()))
    result = _orchestrator(planner, tool, agent_executor=_agent_executor(enabled=False)).orchestrate(
        CapabilityOrchestrationRequest(
            CapabilityPlanningRequest("Execute workflow"),
            policy=CapabilityOrchestrationPolicy(agent_execution_policy=_agent_policy()),
        )
    )

    assert result.status is CapabilityOrchestrationStatus.EXECUTION_FAILED
    assert result.agent_execution_result is not None
    assert result.agent_execution_result.agent_id == "atlas.agent.echo"
    assert result.checkpoint["agent_status"] == "AGENT_DISABLED"


def test_agent_pipeline_reports_missing_handler() -> None:
    tool = SpyTool()
    planner = StaticPlanner(_decision(_plan()))
    result = _orchestrator(planner, tool, agent_executor=_agent_executor(with_handler=False)).orchestrate(
        CapabilityOrchestrationRequest(
            CapabilityPlanningRequest("Execute workflow"),
            policy=CapabilityOrchestrationPolicy(agent_execution_policy=_agent_policy()),
        )
    )

    assert result.status is CapabilityOrchestrationStatus.EXECUTION_FAILED
    assert result.agent_execution_result is not None
    assert result.agent_execution_result.status.value == "HANDLER_UNAVAILABLE"


def test_agent_pipeline_checkpoint_is_summary_only() -> None:
    tool = SpyTool()
    planner = StaticPlanner(_decision(_plan()))
    result = _orchestrator(planner, tool, agent_executor=_agent_executor()).orchestrate(
        CapabilityOrchestrationRequest(
            CapabilityPlanningRequest("Execute workflow"),
            policy=CapabilityOrchestrationPolicy(agent_execution_policy=_agent_policy()),
            inputs={"value": 5},
        )
    )

    assert result.checkpoint == {
        "agent_policy_enabled": True,
        "agent_policy_allow_agent_execution": True,
        "agent_policy_preferred_agent_id": "atlas.agent.echo",
        "agent_policy_require_explicit_agent": True,
        "agent_policy_fail_if_agent_missing": True,
        "agent_id": "atlas.agent.echo",
        "agent_status": "COMPLETED",
        "result_has_output": True,
        "result_output_key_count": 2,
        "result_output_keys": ("agent_id", "value"),
        "result_sanitized_output_fields": 1,
    }
    assert "hidden" not in repr(result.checkpoint)
    assert "token" not in repr(result.checkpoint)


def test_capability_execution_request_transports_agent_policy() -> None:
    tool = SpyTool()
    planner = StaticPlanner(_decision(_plan()))
    orchestrator = _orchestrator(planner, tool, agent_executor=_agent_executor())
    service = CapabilityExecutionService(orchestrator)

    result = service.execute(
        CapabilityExecutionRequest(
            agent_execution_policy=_agent_policy(),
            inputs={"value": 9},
            metadata={"request_id": "safe"},
        )
    )

    assert result.status is CapabilityExecutionStatus.COMPLETED
    assert result.output == {"agent_id": "atlas.agent.echo", "value": 9}
    assert result.execution_status == "COMPLETED"
    assert result.orchestration_result is not None
    assert result.orchestration_result.agent_execution_result is not None
    assert planner.calls == 0
    assert tool.calls == 0


def test_agent_policy_skips_multi_capability_path_in_service() -> None:
    tool = SpyTool()
    planner = StaticPlanner(_decision(_plan()))
    orchestrator = _orchestrator(planner, tool, agent_executor=_agent_executor())
    multi_planner = FailingMultiCapabilityPlanner()
    service = CapabilityExecutionService(orchestrator, multi_capability_planner=multi_planner)

    result = service.execute(
        CapabilityExecutionRequest(
            agent_execution_policy=_agent_policy(),
            inputs={"value": 11},
            required_outputs=("value",),
        )
    )

    assert result.status is CapabilityExecutionStatus.COMPLETED
    assert result.output == {"agent_id": "atlas.agent.echo", "value": 11}
    assert multi_planner.calls == 0


def test_capability_service_propagates_cancelled_context_to_agent_pipeline() -> None:
    tool = SpyTool()
    planner = StaticPlanner(_decision(_plan()))
    service = CapabilityExecutionService(
        _orchestrator(planner, tool, agent_executor=_agent_executor())
    )

    result = service.execute(
        CapabilityExecutionRequest(
            agent_execution_policy=_agent_policy(),
            execution_context=SkillExecutionContext(cancelled=True),
        )
    )

    assert result.status is CapabilityExecutionStatus.CANCELLED
    assert result.orchestration_result is not None
    assert result.orchestration_result.agent_execution_result is not None
    assert result.orchestration_result.agent_execution_result.status.value == "CANCELLED"
    assert planner.calls == 0
    assert tool.calls == 0
