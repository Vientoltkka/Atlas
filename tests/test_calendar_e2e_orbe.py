"""Focal E2E: conversational Calendar flow routed from the orchestrator.

Reproduces the Orbe E2E bugs:

1. "Que tengo manana" must reach calendar_list_events (read-only) instead of
   falling back to the direct chat reply.
2. A structured calendar create plan must be confirmed exactly once. A stale
   async approval left behind by a cancelled background goal must never steal
   the "confirmo" turn nor block the plan's cancel.

No Google access is used: the calendar tools are fakes bound to the same
registry/schemas the production bootstrap wires.
"""

from __future__ import annotations

from types import SimpleNamespace

from agents.registry import AgentRegistry
from bootstrap.bootstrap import Bootstrap
from core.async_task_scheduler import (
    InvalidApprovalError,
    PendingApproval,
    TaskStatus,
)
from core.execution_plan_executor import ExecutionPlanExecutor
from core.execution_plan_validator import ExecutionPlanValidator
from core.operational_request_router import (
    OperationalRequestRouter,
    RequestRoute,
)
from core.orchestrator import AtlasOrchestrator
from core.planner import Planner
from core.request_gateway import RequestGateway
from core.router import Router
from core.structured_execution import StructuredExecutionCoordinator
from memory.conversation import ConversationMemory
from tools.argument_schema import ArgumentSchemaRegistry, ArgumentValidator
from tools.base_tool import BaseTool
from tools.calendar.calendar_create_event_tool import (
    CALENDAR_CREATE_EVENT_ARGUMENTS_SCHEMA,
)
from tools.calendar.calendar_list_events_tool import (
    CALENDAR_LIST_EVENTS_ARGUMENTS_SCHEMA,
)
from tools.executor import ToolExecutor
from tools.registry import ToolRegistry


class CalendarSpyTool(BaseTool):
    def __init__(
        self,
        name: str,
        output: object,
        calls: list[str],
        *,
        requires_confirmation: bool = False,
    ) -> None:
        self._name = name
        self._output = output
        self._calls = calls
        self._requires_confirmation = requires_confirmation
        self.contexts: list[object] = []

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return f"Fake {self._name}."

    @property
    def requires_confirmation(self) -> bool:
        return self._requires_confirmation

    def execute(self, context):
        self._calls.append(self._name)
        self.contexts.append(context)
        return self._output


class ChatAgentFake:
    name = "chat"
    description = "fake chat"

    def __init__(self) -> None:
        self.calls = 0

    def run(self, model, messages):
        self.calls += 1
        return f"fallback:{messages[-1]['content']}"


class DirectRouteExecutorFake:
    """Stand-in for the production bounded direct-response route."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def execute(self, request, decision, **kwargs):
        self.calls.append(request.content)
        return SimpleNamespace(
            status="completed",
            text=f"chat directo: {request.content}",
        )


class RoutePresenterFake:
    def present(self, result) -> str:
        return str(getattr(result, "text", ""))


class StaleApprovalScheduler:
    """Scheduler stand-in holding one stale approval for a cancelled goal."""

    def __init__(self) -> None:
        self.approval = PendingApproval(
            confirmation_id="cid-stale",
            task_id="task-1",
            goal_id="goal-1",
            prompt="write_target",
        )
        self.approve_calls: list[str] = []

    def pending_approvals(self, goal_id=None):
        return [self.approval]

    def goal_status(self, goal_id):
        return TaskStatus.CANCELLED

    def approve(self, confirmation_id: str):
        self.approve_calls.append(confirmation_id)
        raise InvalidApprovalError(
            f"task {self.approval.task_id} is not waiting for approval."
        )

    def deny(self, confirmation_id: str):
        self.approve_calls.append(confirmation_id)
        raise InvalidApprovalError(
            f"task {self.approval.task_id} is not waiting for approval."
        )


def _calendar_tool_stack(
    calls: list[str],
) -> tuple[ToolRegistry, CalendarSpyTool, CalendarSpyTool]:
    registry = ToolRegistry()
    create_tool = CalendarSpyTool(
        "calendar_create_event",
        "id: evt-1 | summary: ATLAS E2E TEST 6",
        calls,
        requires_confirmation=True,
    )
    list_tool = CalendarSpyTool("calendar_list_events", {"events": []}, calls)
    registry.register(
        create_tool,
        arguments_schema=CALENDAR_CREATE_EVENT_ARGUMENTS_SCHEMA,
    )
    registry.register(
        list_tool,
        arguments_schema=CALENDAR_LIST_EVENTS_ARGUMENTS_SCHEMA,
    )
    return registry, create_tool, list_tool


def _build_orchestrator(calls: list[str], *, scheduler):
    registry, create_tool, list_tool = _calendar_tool_stack(calls)
    schema_registry = Bootstrap.build_argument_schema_registry()
    planner = Planner(
        tool_registry=registry,
        tool_selector=Bootstrap.build_tool_selector(registry),
        schema_registry=schema_registry,
        argument_validator=ArgumentValidator(schema_registry),
    )
    coordinator = StructuredExecutionCoordinator(
        planner=planner,
        validator=ExecutionPlanValidator(),
        executor=ExecutionPlanExecutor(registry, ToolExecutor(registry)),
    )
    agent = ChatAgentFake()
    agents = AgentRegistry()
    agents.register(agent)
    orchestrator = AtlasOrchestrator(
        planner=planner,
        router=Router(
            operational_router=OperationalRequestRouter(
                tool_registry=registry,
                agent_registry=agents,
            )
        ),
        model_manager=SimpleNamespace(
            choose_model=lambda agent_name: f"model:{agent_name}",
        ),
        memory=ConversationMemory(),
        registry=agents,
        write_file=SimpleNamespace(execute=lambda *_args: "written"),
        structured_execution_coordinator=coordinator,
        structured_execution_enabled=True,
        structured_plan_execution_enabled=True,
        operational_route_executor=DirectRouteExecutorFake(),
        route_execution_presenter=RoutePresenterFake(),
        async_task_scheduler=scheduler,
    )
    if scheduler is not None:
        orchestrator._background_goal_id = "goal-1"
    return orchestrator, coordinator, create_tool, list_tool, agent


def test_natural_tomorrow_question_routes_to_calendar_list_events() -> None:
    decision = OperationalRequestRouter(
        tool_registry=Bootstrap.build_tool_registry(),
    ).classify(RequestGateway().from_text("Que tengo manana"))

    assert decision.route is RequestRoute.SINGLE_TOOL
    assert decision.target_tool_name == "calendar_list_events"


def test_natural_tomorrow_question_executes_calendar_list_not_chat() -> None:
    calls: list[str] = []
    orchestrator, _, _, list_tool, agent = _build_orchestrator(
        calls,
        scheduler=None,
    )

    response = orchestrator.process_prompt(
        "Que tengo manana",
        confirm=lambda _prompt: "",
    )

    assert calls == ["calendar_list_events"]
    assert agent.calls == 0
    assert "Ejecucion completada" in response
    parameters = list_tool.contexts[0].parameters
    assert parameters["time_min"] < parameters["time_max"]


def test_create_plan_confirms_once_despite_stale_async_approval() -> None:
    calls: list[str] = []
    scheduler = StaleApprovalScheduler()
    (
        orchestrator,
        coordinator,
        create_tool,
        list_tool,
        agent,
    ) = _build_orchestrator(calls, scheduler=scheduler)

    pending = orchestrator.process_prompt(
        'Crea un evento "ATLAS E2E TEST 6" manana a las 12',
        confirm=lambda _prompt: "",
    )
    assert "pendiente de confirmacion" in pending
    assert calls == []
    assert coordinator.has_pending_execution()

    confirmed = orchestrator.process_prompt("confirmo", confirm=lambda _prompt: "")
    assert calls == ["calendar_create_event"]
    assert scheduler.approve_calls == []
    assert "Plan confirmado" in confirmed
    assert not coordinator.has_pending_execution()

    repeat = orchestrator.process_prompt("confirmo", confirm=lambda _prompt: "")
    assert calls == ["calendar_create_event"]
    assert "calendar_create_event" not in repeat or calls.count(
        "calendar_create_event"
    ) == 1

    listing = orchestrator.process_prompt(
        "Lista eventos del calendario de manana con maximo 20",
        confirm=lambda _prompt: "",
    )
    assert calls == ["calendar_create_event", "calendar_list_events"]
    assert list_tool.contexts[0].parameters["max_results"] == 20
    assert "Ejecucion completada" in listing
    assert not coordinator.has_pending_execution()


def test_cancel_pending_create_plan_never_executes_create() -> None:
    calls: list[str] = []
    scheduler = StaleApprovalScheduler()
    (
        orchestrator,
        coordinator,
        create_tool,
        list_tool,
        agent,
    ) = _build_orchestrator(calls, scheduler=scheduler)

    orchestrator.process_prompt(
        'Crea un evento "ATLAS E2E TEST 6" manana a las 12',
        confirm=lambda _prompt: "",
    )

    cancelled = orchestrator.process_prompt("cancela", confirm=lambda _prompt: "")

    assert calls == []
    assert "cancelado" in cancelled
    assert not coordinator.has_pending_execution()
