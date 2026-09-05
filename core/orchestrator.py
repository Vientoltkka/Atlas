"""Core orchestration module for Atlas."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
import inspect
from pathlib import Path
import re
import sys
import time
import unicodedata

from agents.registry import AgentRegistry
from agents.base_agent import AgentResponse
from agents.coding_agent import PendingCodingChangeError

from core.async_task_scheduler import (
    AsyncTaskScheduler,
    AsyncTaskSchedulerError,
    GoalBudget,
    TaskStatus,
    UnknownGoalError,
)
from core.background_goal_pump import BackgroundGoalPump
from core.conversational_autonomy import (
    AutonomousGoalRequest,
    DEFAULT_MAX_DURATION_SECONDS,
    DEFAULT_MAX_TASK_EXECUTIONS,
    detect_autonomous_goal,
    detect_structured_plan_objective,
    is_background_goal_cancel_request,
    is_background_goal_status_query,
)
from core.execution_plan_task_adapter import (
    ExecutionPlanTaskBridgeError,
    execution_plan_to_task_specs,
)
from core.execution_plan_validator import ExecutionPlanValidator
from core.multi_task_goal import detect_multi_task_goal
from core.atlas_router import (
    AtlasRouter,
    AtlasRoutingRequest,
    AtlasRoutingResult,
    AtlasRoutingStatus,
    AtlasRouteType,
)
from core.atlas_request_adapter import (
    AtlasRequestAdapter,
    StructuredAtlasRequest,
    adaptation_failure_to_routing_result,
    unavailable_atlas_request_adapter_result,
)
from core.atlas_request_classifier import (
    AtlasRequestClassifier,
    StructuredInput,
    classification_failure_to_routing_result,
    unavailable_atlas_request_classifier_result,
)
from core.atlas_request_normalizer import (
    AtlasRequestNormalizer,
    normalization_failure_to_routing_result,
    unavailable_atlas_request_normalizer_result,
)
from core.model_health import ModelHealthChecker
from core.model_inference import (
    InferenceFallbackExhaustedError,
    ModelHealthCheckError,
    ModelInferenceRunner,
    ModelSelectionError,
)
from core.model_manager import ModelManager
from core.model_selection_policy import ModelSelectionPolicy
from core.planner import Planner
from core.router import Router
from core.request_gateway import (
    AtlasRequest,
    RequestGateway,
)
from core import skill_intent
from core.operational_request_router import (
    MemoryOperation,
    RequestRoute,
    SystemCommand,
    RouteDecision,
)
from core.operational_route_executor import (
    OperationalRouteExecutor,
    RouteExecutionPresenter,
    RouteExecutionResult,
    RouteExecutionStatus,
)
from core.capability_execution_service import (
    CapabilityExecutionRequest,
    CapabilityExecutionResult,
    CapabilityExecutionService,
    unavailable_capability_execution_result,
)
from core.execution_plan_executor import ExecutionControl, ExecutionProgress
from core.execution_history import ExecutionSessionHistory
from core.execution_history_advisor import ExecutionHistoryAdvisor
from core.historical_plan_adjustment import HistoricalPlanAdjuster
from core.execution_strategy import ExecutionStrategySelector
from core.execution_authorization import (
    ExecutionAuthorizationGate,
    ExecutionDispatcher,
)
from core.hybrid_execution_planner import StructuredPlanningProgress
from core.structured_execution import (
    StructuredExecutionCoordinator,
    StructuredExecutionResponse,
)
from core.skill_executor import SkillExecutionRequest
from core.skill_resolver import SkillResolutionRequest, SkillResolutionStatus
from core.skill_system import SkillSystem
from tools.tool_context import ToolContext

from core.supervised_capability_gap import (
    MissingCapabilityProposal,
    SkillCreationResponse,
    SupervisedCapabilityGapDetector,
)
from core.self_improvement_conversation import SelfImprovementConversation

from memory.conversation import ConversationMemory
from use_cases.correction_interaction import CorrectionInteractionUseCase
from use_cases.desktop_interaction import DesktopInteractionUseCase
from use_cases.execution_conversation import ExecutionConversationController
from use_cases.refactoring_interaction import RefactoringInteractionUseCase
from use_cases.permanent_assistant import PermanentAssistantUseCase
from use_cases.speech_engine import SpeechInteractionUseCase
from use_cases.voice_conversation import VoiceConversationUseCase
from use_cases.wake_word_engine import WakeWordInteractionUseCase
from use_cases.write_file import WriteFileUseCase
from tools.registry import ToolRegistry
from tools.web_search import WebSearchError, WebSearchTimeoutError, WebSearchTool


@dataclass(frozen=True, slots=True)
class PendingAgentFollowUp:
    """One active specialist continuation in the current Atlas runtime."""

    agent_name: str
    original_prompt: str
    active: bool = True


class AtlasOrchestrator:
    """Main orchestrator for Atlas."""

    def __init__(
        self,
        planner: Planner,
        router: Router,
        model_manager: ModelManager,
        memory: ConversationMemory,
        registry: AgentRegistry,
        write_file: WriteFileUseCase,
        model_health_checker: ModelHealthChecker | None = None,
        model_selection_policy: ModelSelectionPolicy | None = None,
        refactoring_interaction: RefactoringInteractionUseCase | None = None,
        correction_interaction: CorrectionInteractionUseCase | None = None,
        desktop_interaction: DesktopInteractionUseCase | None = None,
        execution_conversation: ExecutionConversationController | None = None,
        speech_interaction: SpeechInteractionUseCase | None = None,
        wake_word_interaction: WakeWordInteractionUseCase | None = None,
        voice_conversation: VoiceConversationUseCase | None = None,
        permanent_assistant: PermanentAssistantUseCase | None = None,
        structured_execution_coordinator: StructuredExecutionCoordinator | None = None,
        capability_execution_service: CapabilityExecutionService | None = None,
        atlas_router: AtlasRouter | None = None,
        atlas_request_adapter: AtlasRequestAdapter | None = None,
        atlas_request_classifier: AtlasRequestClassifier | None = None,
        atlas_request_normalizer: AtlasRequestNormalizer | None = None,
        request_gateway: RequestGateway | None = None,
        operational_route_executor: OperationalRouteExecutor | None = None,
        route_execution_presenter: RouteExecutionPresenter | None = None,
        execution_history: ExecutionSessionHistory | None = None,
        execution_history_advisor: ExecutionHistoryAdvisor | None = None,
        historical_plan_adjuster: HistoricalPlanAdjuster | None = None,
        execution_strategy_selector: ExecutionStrategySelector | None = None,
        execution_authorization_gate: ExecutionAuthorizationGate | None = None,
        execution_dispatcher: ExecutionDispatcher | None = None,
        tool_registry: ToolRegistry | None = None,
        web_search_tool: WebSearchTool | None = None,
        skill_system: SkillSystem | None = None,
        capability_gap_detector: SupervisedCapabilityGapDetector | None = None,
        self_improvement_conversation: SelfImprovementConversation | None = None,
        structured_execution_enabled: bool = False,
        structured_plan_streaming_enabled: bool = False,
        structured_plan_execution_enabled: bool = False,
        structured_planning_progress_enabled: bool = True,
        project_root: Path | None = None,
        training_pdf_output_dir: Path | None = None,
        now_provider=None,
        async_task_scheduler: "AsyncTaskScheduler | None" = None,
        background_pump_interval_seconds: "float | None" = None,
    ) -> None:

        self._planner = planner
        self._web_search_tool = web_search_tool
        self._router = router
        self._model_manager = model_manager
        self._model_selection_policy = (
            model_selection_policy
            if model_selection_policy is not None
            else ModelSelectionPolicy()
        )
        self._model_inference_runner = ModelInferenceRunner(
            model_manager,
            health_checker=model_health_checker,
        )
        self._memory = memory
        self._registry = registry
        self._write_file = write_file
        self._refactoring_interaction = refactoring_interaction
        self._correction_interaction = correction_interaction
        self._desktop_interaction = desktop_interaction
        self._execution_conversation = execution_conversation
        self._speech_interaction = speech_interaction
        self._wake_word_interaction = wake_word_interaction
        self._voice_conversation = voice_conversation
        self._permanent_assistant = permanent_assistant
        self._structured_execution_coordinator = structured_execution_coordinator
        self._capability_execution_service = capability_execution_service
        self._atlas_router = atlas_router
        self._atlas_request_adapter = atlas_request_adapter
        self._atlas_request_classifier = atlas_request_classifier
        self._atlas_request_normalizer = atlas_request_normalizer
        self._request_gateway = request_gateway or RequestGateway(router=router)
        self._operational_route_executor = operational_route_executor
        self._route_execution_presenter = (
            route_execution_presenter or RouteExecutionPresenter()
        )
        self._execution_history = execution_history
        self._execution_history_advisor = execution_history_advisor
        self._historical_plan_adjuster = historical_plan_adjuster
        self._execution_strategy_selector = execution_strategy_selector
        self._execution_authorization_gate = execution_authorization_gate
        self._execution_dispatcher = execution_dispatcher
        self._tool_registry = tool_registry
        self._training_pdf_output_dir = training_pdf_output_dir or (
            Path("artifacts") / "documents"
        )
        self._skill_system = skill_system
        self._capability_gap_detector = capability_gap_detector
        self._pending_capability_proposal: MissingCapabilityProposal | None = None
        self._pending_agent_followup: PendingAgentFollowUp | None = None
        self._pending_skill_creation_proposal: SkillCreationResponse | None = None
        self._prepared_capability_proposal: MissingCapabilityProposal | None = None
        self._validated_capability_proposal: MissingCapabilityProposal | None = None
        self._last_structured_execution_response: (
            StructuredExecutionResponse | None
        ) = None
        self._structured_execution_enabled = structured_execution_enabled
        self._structured_plan_streaming_enabled = structured_plan_streaming_enabled
        self._structured_plan_execution_enabled = structured_plan_execution_enabled
        self._structured_planning_progress_enabled = structured_planning_progress_enabled
        self._structured_planning_active = False
        self._structured_execution_active = False
        self._execution_cancel_requested = False
        self._planning_progress_presenter = _PlanningProgressPresenter(
            self._print_atlas
        )
        self._execution_progress_presenter = _ExecutionProgressPresenter(
            self._print_atlas
        )
        self._project_root = project_root or Path(".")
        self._self_improvement_conversation = self_improvement_conversation or SelfImprovementConversation(self._project_root)
        self._now_provider = now_provider or (lambda: datetime.now().astimezone())
        self._async_task_scheduler = async_task_scheduler
        self._background_pump = (
            BackgroundGoalPump(
                async_task_scheduler,
                **(
                    {"interval_seconds": background_pump_interval_seconds}
                    if background_pump_interval_seconds is not None
                    else {}
                ),
            )
            if async_task_scheduler is not None
            else None
        )
        self._background_goal_id: "str | None" = None

    @property
    def async_task_scheduler(self) -> "AsyncTaskScheduler | None":
        """Expose the optional independent-task scheduler."""
        return self._async_task_scheduler

    def run_independent_tasks(
        self,
        description: str,
        tasks: list[dict[str, Any]],
        *,
        goal_id: str | None = None,
    ) -> "str | None":
        """Submit independent tasks and run every READY one without blocking."""
        if self._async_task_scheduler is None:
            return None
        goal = self._async_task_scheduler.submit_goal(
            description,
            tasks,
            goal_id=goal_id,
        )
        self._async_task_scheduler.run_ready()
        return goal

    def approve_async_task(self, confirmation_id: str) -> "str | None":
        """Resume exactly the task bound to this single-use approval token."""
        if self._async_task_scheduler is None:
            return None
        return self._async_task_scheduler.approve(confirmation_id)

    def deny_async_task(self, confirmation_id: str) -> "str | None":
        """Deny one pending approval; only its own task is affected."""
        if self._async_task_scheduler is None:
            return None
        return self._async_task_scheduler.deny(confirmation_id)

    def _handle_multi_task_objective(self, prompt: str) -> "str | None":
        """Route a conservative multi-task objective through the async scheduler."""
        if self._async_task_scheduler is None:
            return None
        goal_plan = detect_multi_task_goal(prompt)
        if goal_plan is None:
            return None
        goal_id = self.run_independent_tasks(goal_plan.description, goal_plan.tasks)
        if goal_id is None:
            return None
        response = self._describe_async_goal(
            goal_id,
            intro="Objetivo en marcha",
        )
        self._memory.add_user(prompt)
        self._memory.add_assistant(response)
        return response

    def _handle_pending_async_approval(self, prompt: str) -> "str | None":
        """Answer a bare confirmation bound to one pending async approval."""
        if self._async_task_scheduler is None:
            return None
        pending = self._async_task_scheduler.pending_approvals()
        if not pending:
            return None
        normalized = _normalize_confirmation_text(prompt)
        if normalized in _ASYNC_APPROVAL_YES_TOKENS:
            approval = pending[0]
            try:
                resumed = self.approve_async_task(approval.confirmation_id)
            except AsyncTaskSchedulerError:
                response = "Esa operación pendiente ya no está disponible."
                self._memory.add_user(prompt)
                self._memory.add_assistant(response)
                return response
            intro = (
                "Hecho, operación completada."
                if resumed
                else "Confirmación registrada."
            )
            response = self._describe_async_goal(approval.goal_id, intro=intro)
            self._memory.add_user(prompt)
            self._memory.add_assistant(response)
            return response
        if normalized in _ASYNC_APPROVAL_NO_TOKENS:
            approval = pending[0]
            try:
                denied = self.deny_async_task(approval.confirmation_id)
            except AsyncTaskSchedulerError:
                response = "Esa operación pendiente ya no está disponible."
                self._memory.add_user(prompt)
                self._memory.add_assistant(response)
                return response
            intro = (
                "Entendido, la operación quedó cancelada."
                if denied
                else "Entendido."
            )
            response = self._describe_async_goal(approval.goal_id, intro=intro)
            self._memory.add_user(prompt)
            self._memory.add_assistant(response)
            return response
        return None

    def _describe_async_goal(self, goal_id: str, *, intro: str) -> str:
        """Compose a short conversational progress report for one goal."""
        state = self._async_task_scheduler.goal(goal_id)
        lines = [f"{intro}: {state.description}."]
        for task in state.tasks.values():
            if task.status is TaskStatus.DONE:
                label = "hecho"
            elif task.status is TaskStatus.WAITING_APPROVAL:
                label = "pendiente de tu confirmación"
            elif task.status is TaskStatus.FAILED:
                label = f"fallida ({task.error or 'error'})"
            elif task.status is TaskStatus.BLOCKED:
                label = f"bloqueada ({task.error or 'pendiente cancelada'})"
            elif task.status is TaskStatus.CANCELLED:
                label = "cancelada"
            else:
                label = "en curso"
            lines.append(f"- {task.description}: {label}.")
        if any(
            task.status is TaskStatus.WAITING_APPROVAL
            for task in state.tasks.values()
        ):
            lines.append("Responde sí para autorizar la operación pendiente.")
        return "\n".join(lines)

    # -- prolonged autonomy (background pump) ------------------------------------

    def start_background_pump(self) -> None:
        """Start the cooperative background goal pump, if configured."""
        if self._background_pump is not None:
            self._background_pump.start()

    def stop_background_pump(self, timeout: float = 8.0) -> None:
        """Stop the background pump cleanly."""
        if self._background_pump is not None:
            self._background_pump.stop(timeout=timeout)

    def close(self) -> None:
        """Release background resources; never blocks shutdown."""
        self.stop_background_pump()

    def _handle_background_autonomy(self, prompt: str) -> "str | None":
        """Route explicit autonomy orders, status queries and cancellations."""
        if self._async_task_scheduler is None or self._background_pump is None:
            return None
        if is_background_goal_status_query(prompt):
            return self._describe_background_goal_status(prompt)
        if is_background_goal_cancel_request(prompt):
            return self._cancel_background_goal(prompt)
        structured_response = self._handle_structured_plan_execution(prompt)
        if structured_response is not None:
            return structured_response
        request = detect_autonomous_goal(prompt)
        if request is None:
            return None
        return self._start_background_goal(prompt, request)

    def _handle_structured_plan_execution(self, prompt: str) -> "str | None":
        """Explicit 'plan and execute' entry: plan → validate → convert → submit.

        No ``run_ready()`` is called here: after submission the existing
        BackgroundGoalPump keeps executing the converted tasks in background.
        """
        if self._async_task_scheduler is None or self._background_pump is None:
            return None
        objective = detect_structured_plan_objective(prompt)
        if objective is None:
            return None
        generation = self._planner.generate_execution_plan(
            objective,
            structured_planning=True,
        )
        plan = generation.plan
        if plan is None or not generation.success:
            response = (
                "No pude estructurar ese objetivo en un plan ejecutable y "
                "seguro. Dámelo como una cadena clara (por ejemplo: lee X, "
                "resume Y, escribe Z)."
            )
            self._memory.add_user(prompt)
            self._memory.add_assistant(response)
            return response
        try:
            task_specs = execution_plan_to_task_specs(
                plan,
                validator=ExecutionPlanValidator(self._tool_registry),
            )
        except ExecutionPlanTaskBridgeError as error:
            detail = "; ".join(error.errors[:3])
            response = (
                "El plan no puede convertirse en tareas de fondo seguras "
                f"({error.code}). {detail}"
            )
            self._memory.add_user(prompt)
            self._memory.add_assistant(response)
            return response
        goal_id = self._async_task_scheduler.submit_goal(
            plan.goal,
            list(task_specs),
            budget=GoalBudget(
                max_duration_seconds=DEFAULT_MAX_DURATION_SECONDS,
                max_task_executions=DEFAULT_MAX_TASK_EXECUTIONS,
            ),
        )
        self._background_goal_id = goal_id
        self.start_background_pump()
        response = self._describe_async_goal(
            goal_id,
            intro="Plan estructurado en marcha",
        )
        response += (
            "\nTrabajo en segundo plano sin que tengas que escribir más "
            "(di 'detén el trabajo' para parar)."
        )
        self._memory.add_user(prompt)
        self._memory.add_assistant(response)
        return response

    def _start_background_goal(
        self,
        prompt: str,
        request: "AutonomousGoalRequest",
    ) -> "str | None":
        objective = getattr(request, "objective", "")
        if not objective:
            return self._describe_background_goal_status(prompt)
        goal_plan = detect_multi_task_goal(objective)
        if goal_plan is None:
            response = (
                "Puedo trabajar en segundo plano, pero no supe estructurar "
                "ese objetivo en tareas concretas y seguras. Dámelo como una "
                "cadena clara (por ejemplo: lee X, resume Y, escribe Z)."
            )
            self._memory.add_user(prompt)
            self._memory.add_assistant(response)
            return response
        budget = GoalBudget(
            max_duration_seconds=request.max_duration_seconds,
            max_task_executions=request.max_task_executions,
        )
        goal_id = self._async_task_scheduler.submit_goal(
            goal_plan.description,
            goal_plan.tasks,
            budget=budget,
        )
        self._background_goal_id = goal_id
        self.start_background_pump()
        response = (
            "Objetivo en segundo plano: "
            f"{goal_plan.description}. Trabajaré de forma autónoma sin que "
            "tengas que escribir más mensajes (límite: "
            f"{int(request.max_duration_seconds // 60)} min). "
            "Pregunta 'cómo va el objetivo' o di 'detén el trabajo' para parar."
        )
        self._memory.add_user(prompt)
        self._memory.add_assistant(response)
        return response

    def _describe_background_goal_status(self, prompt: str) -> "str | None":
        goal_id = self._background_goal_id_or_recovered()
        if goal_id is None:
            return None
        try:
            summary = self._async_task_scheduler.goal_summary(goal_id)
        except UnknownGoalError:
            return None
        response = self._describe_async_goal(goal_id, intro="Estado del trabajo")
        remaining = summary.get("budget_remaining_seconds")
        if remaining is not None and remaining > 0:
            response += (
                f"\nPresupuesto: quedan unos {int(remaining // 60)} min."
            )
        if summary.get("cancelled"):
            response += "\nEl trabajo fue cancelado."
        self._memory.add_user(prompt)
        self._memory.add_assistant(response)
        return response

    def _background_goal_id_or_recovered(self) -> "str | None":
        """Return the active background goal, recovering a persisted one."""
        if self._background_goal_id is not None:
            return self._background_goal_id
        goal_id = self._active_persisted_background_goal_id()
        if goal_id is not None:
            self._background_goal_id = goal_id
        return goal_id

    def _active_persisted_background_goal_id(self) -> "str | None":
        """Resolve the newest non-terminal persisted goal after a restart."""
        scheduler = self._async_task_scheduler
        if scheduler is None:
            return None
        active: "list[tuple[datetime, str]]" = []
        for persisted_goal_id in scheduler.persisted_goal_ids():
            try:
                try:
                    state = scheduler.goal(persisted_goal_id)
                    status = scheduler.goal_status(persisted_goal_id)
                except UnknownGoalError:
                    state = scheduler.load_goal(persisted_goal_id)
                    status = scheduler.goal_status(persisted_goal_id)
            except AsyncTaskSchedulerError:
                continue
            if status in {TaskStatus.DONE, TaskStatus.CANCELLED}:
                continue
            active.append((state.created_at, persisted_goal_id))
        if not active:
            return None
        active.sort()
        return active[-1][1]

    def _cancel_background_goal(self, prompt: str) -> "str | None":
        goal_id = self._background_goal_id_or_recovered()
        if goal_id is None:
            return None
        try:
            status = self._async_task_scheduler.goal_status(goal_id)
        except UnknownGoalError:
            return None
        description = self._async_task_scheduler.goal(goal_id).description
        if status is TaskStatus.CANCELLED:
            return "El trabajo en segundo plano ya estaba detenido."
        if status is TaskStatus.DONE:
            return (
                f"El trabajo en segundo plano ya había terminado: {description}."
            )
        self._async_task_scheduler.cancel_goal(goal_id)
        response = (
            f"Trabajo detenido: {description}. Las acciones ya completadas "
            "no se revierten y las confirmaciones pendientes quedan sin usar."
        )
        self._memory.add_user(prompt)
        self._memory.add_assistant(response)
        return response

    @property
    def execution_history(self) -> ExecutionSessionHistory | None:
        """Expose the read-only internal execution-history query API."""
        return self._execution_history

    @property
    def pending_agent_followup(self) -> "PendingAgentFollowUp | None":
        """Return the active agent continuation, when one was requested."""
        return self._pending_agent_followup

    def add_supervision_state_listener(self, listener) -> None:
        """Attach a passive observer to real supervised execution state."""
        if self._structured_execution_coordinator is not None:
            self._structured_execution_coordinator.add_supervision_state_listener(listener)

    @property
    def execution_history_advisor(self) -> ExecutionHistoryAdvisor | None:
        """Expose consultation-only recommendations for planning callers."""
        return self._execution_history_advisor

    @property
    def historical_plan_adjuster(self) -> HistoricalPlanAdjuster | None:
        """Expose the validated post-planning historical adjustment stage."""
        return self._historical_plan_adjuster

    @property
    def execution_strategy_selector(self) -> ExecutionStrategySelector | None:
        """Expose deterministic strategy selection after plan validation."""
        return self._execution_strategy_selector

    @property
    def execution_authorization_gate(self) -> ExecutionAuthorizationGate | None:
        """Expose the deterministic pre-dispatch authorization boundary."""
        return self._execution_authorization_gate

    @property
    def execution_dispatcher(self) -> ExecutionDispatcher | None:
        """Expose one-shot dispatch records without granting execution."""
        return self._execution_dispatcher

    @property
    def last_structured_execution_response(
        self,
    ) -> StructuredExecutionResponse | None:
        """Return the latest high-level structured result for observability."""
        return self._last_structured_execution_response

    def start(self) -> None:

        startup_response = self._load_persisted_structured_execution()
        if startup_response is not None:
            self._print_atlas(startup_response.message)

        while True:

            try:
                prompt = input("Tú: ")
            except EOFError:
                print("\nHasta pronto.")
                break
            except KeyboardInterrupt:
                print("\n\nInterrupcion recibida. Hasta pronto.")
                break

            prompt = prompt.strip()

            if not prompt:
                self._print_atlas(
                    "La peticion esta vacia. Escribe una instruccion o 'salir'."
                )
                continue

            if prompt.casefold() in ("exit", "quit", "salir"):
                print("\nHasta pronto.")
                break

            tool_catalog = self._tool_catalog_response(prompt)
            if tool_catalog is not None:
                self._print_atlas(tool_catalog)
                continue

            capability_status = self._capability_status_response(prompt)
            if capability_status is not None:
                self._print_atlas(capability_status)
                continue

            request = self._request_gateway.from_text(prompt, user_id="local-user")
            skill_response = self._process_skill_request(request)
            if skill_response is not None:
                self._print_atlas(skill_response)
                continue

            memory_response = self._process_memory_request(request)
            if memory_response is not None:
                self._print_atlas(memory_response)
                continue

            history_response = self._process_execution_history_request(request)
            if history_response is not None:
                self._print_atlas(history_response)
                continue

            if self._desktop_interaction is not None:
                desktop_response = self._desktop_interaction.execute(
                    request.content,
                    confirm=input,
                )
                if desktop_response is not None:
                    self._print_atlas(desktop_response)
                    continue

            direct_response = self._process_direct_conversation(
                request,
                status_sink=self._print_atlas,
            )
            if direct_response is not None:
                self._print_atlas(direct_response)
                continue

            structured_response = self._handle_structured_execution(
                request.content,
                request=request,
            )
            if structured_response is not None:
                visible = self._present_structured_execution(structured_response)
                self._remember_structured_turn(request.content, structured_response)
                self._print_atlas(visible)
                continue

            if self._execution_conversation is not None:
                outcome = self._execution_conversation.handle(prompt)

                if not outcome.direct_response_required:
                    self._print_atlas(outcome.text)
                    continue

            if self._voice_conversation is not None:
                execute_voice = self._voice_conversation.execute
                voice_kwargs = {
                    "prompt": prompt,
                    "process_text": lambda text: self.process_voice_prompt(
                        text,
                        confirm=input,
                    ),
                    "status_sink": self._print_atlas,
                }
                if _accepts_keyword(execute_voice, "process_text_stream"):
                    voice_kwargs["process_text_stream"] = (
                        lambda text, fragment_sink: self.process_voice_prompt(
                            text,
                            confirm=input,
                            on_model_fragment=fragment_sink,
                        )
                    )
                voice_result = execute_voice(**voice_kwargs)

                if voice_result is not None:
                    continue

            if self._wake_word_interaction is not None:
                wake_word_response = self._wake_word_interaction.execute(prompt)

                if wake_word_response is not None:
                    print()
                    print("Atlas:")
                    print(wake_word_response)
                    print()

                    continue

            if self._speech_interaction is not None:
                speech_response = self._speech_interaction.execute(prompt)

                if speech_response is not None:
                    print()
                    print("Atlas:")
                    print(speech_response)
                    print()

                    continue

            response = self._process_prompt_without_execution(
                prompt,
                confirm=input,
            )

            self._print_atlas(response)

    def start_voice(
        self,
        state_listener=None,
        status_sink=None,
        typed_input=None,
    ) -> None:
        """Start manual voice conversation mode without wake word."""
        print("Atlas iniciado en modo voz.")
        print()

        if self._voice_conversation is None:
            print("Atlas:")
            print("Modo de voz no disponible.")
            print()
            return

        execute_manual = self._voice_conversation.execute_manual
        voice_kwargs = {
            "process_text": lambda text: self.process_voice_prompt(
                text,
                confirm=input,
            ),
            "status_sink": status_sink or print,
            "typed_input": self._read_typed_exit_command,
        }
        if state_listener is not None and _accepts_keyword(
            execute_manual, "state_listener"
        ):
            voice_kwargs["state_listener"] = state_listener
        if typed_input is not None and _accepts_keyword(
            execute_manual, "typed_input"
        ):
            voice_kwargs["typed_input"] = typed_input
        if _accepts_keyword(execute_manual, "process_text_stream"):
            voice_kwargs["process_text_stream"] = (
                lambda text, fragment_sink: self.process_voice_prompt(
                    text,
                    confirm=input,
                    on_model_fragment=fragment_sink,
                )
            )
        execute_manual(**voice_kwargs)

    def start_assistant(self, state_listener=None) -> None:
        """Start permanent assistant mode with wake word."""
        if self._permanent_assistant is None:
            print("Atlas:")
            print("Modo asistente permanente no disponible.")
            print()
            return

        run = self._permanent_assistant.run
        assistant_kwargs = {
            "process_text": lambda text: self.process_voice_prompt(
                text,
                confirm=input,
            ),
            "status_sink": print,
            "typed_input": self._read_typed_exit_command,
        }
        if state_listener is not None and _accepts_keyword(run, "state_listener"):
            assistant_kwargs["state_listener"] = state_listener
        run(**assistant_kwargs)

    def list_microphones(self) -> str:
        """Return available input microphones."""
        if self._speech_interaction is None:
            return "Modo de voz no disponible."

        return self._speech_interaction.list_microphones_text()

    def process_prompt(
        self,
        prompt: str,
        confirm,
    ) -> str:
        """Process text through the normal Atlas flow."""
        self_improvement_response = self._self_improvement_conversation.handle(prompt)
        if self_improvement_response is not None:
            return self_improvement_response
        if (
            self._execution_conversation is not None
            and getattr(
                self._execution_conversation,
                "pending_confirmation_id",
                None,
            ) is not None
        ):
            return self._execution_conversation.handle(prompt).text
        closure_response = self._handle_validated_capability_closure(prompt)
        if closure_response is not None:
            return closure_response

        application_response = self._handle_prepared_capability_application(prompt)
        if application_response is not None:
            return application_response

        pending_skill_response = self._handle_pending_skill_creation(prompt)
        if pending_skill_response is not None:
            return pending_skill_response

        pending_response = self._handle_pending_capability_proposal(prompt)
        if pending_response is not None:
            return pending_response

        if self._capability_gap_detector is not None:
            skill_creation = self._capability_gap_detector.skill_creation_response_for(prompt)
            if skill_creation is not None:
                if skill_creation.status == "CREATE_PROPOSAL":
                    self._pending_skill_creation_proposal = skill_creation
                    return skill_creation.present()
                return skill_creation.present()

        temperature_response = self._temperature_conversion_response(prompt)
        if temperature_response is not None:
            return temperature_response

        tool_catalog = self._tool_catalog_response(prompt)
        if tool_catalog is not None:
            return tool_catalog

        web_research = self._web_research_response(prompt)
        if web_research is not None:
            return web_research

        capability_status = self._capability_status_response(prompt)
        if capability_status is not None:
            return capability_status

        if self._capability_gap_detector is not None:
            proposal = self._capability_gap_detector.proposal_for(prompt)
            if proposal is not None:
                self._pending_capability_proposal = proposal
                return proposal.present()

        request = self._request_gateway.from_text(prompt)
        followup_response = self._process_pending_agent_followup(request)
        if followup_response is not None:
            return followup_response
        skill_response = self._process_skill_request(request)
        if skill_response is not None:
            return skill_response

        memory_response = self._process_memory_request(request)
        if memory_response is not None:
            return memory_response

        history_response = self._process_execution_history_request(request)
        if history_response is not None:
            return history_response

        async_approval_response = self._handle_pending_async_approval(prompt)
        if async_approval_response is not None:
            return async_approval_response

        background_response = self._handle_background_autonomy(prompt)
        if background_response is not None:
            return background_response

        multi_task_response = self._handle_multi_task_objective(prompt)
        if multi_task_response is not None:
            return multi_task_response

        # Specialized agents bypass the legacy execution-conversation gate
        # while reusing Atlas' existing model health/fallback pipeline.
        if self._desktop_interaction is not None and self._execution_conversation is not None:
            filesystem_request = self._desktop_interaction.filesystem_tool_request(
                request.content,
            )
            if filesystem_request is not None:
                tool_name, arguments, confirmation_text = filesystem_request
                outcome = self._execution_conversation.handle_registered_tool(
                    tool_name,
                    arguments,
                    original_text=request.content,
                    confirmation_text=confirmation_text,
                )
                return outcome.text
            activation_request = _conversational_activation_window_title(
                request.content,
            )
            if activation_request is not None:
                activation_title, activation_strict = activation_request
                resolved = self._desktop_interaction.resolve_window_for_activation(
                    activation_title,
                )
                if isinstance(resolved, str):
                    if activation_strict:
                        return resolved
                else:
                    target_handle, target_title = resolved
                    outcome = self._execution_conversation.handle_registered_tool(
                        "desktop.bring_window_to_front",
                        {"handle": target_handle},
                        original_text=request.content,
                        confirmation_text=(
                            f"Voy a activar '{target_title}'. ¿Confirmas?"
                        ),
                        pending_target_handle=target_handle,
                    )
                    return outcome.text
            copy_text = self._desktop_interaction._extract_clipboard_copy_text(  # noqa: SLF001
                request.content,
            )
            if copy_text is not None:
                outcome = self._execution_conversation.handle_registered_tool(
                    "desktop.copy_clipboard_text",
                    {"text": copy_text},
                    original_text=request.content,
                    confirmation_text="Voy a copiar ese texto al portapapeles. ¿Confirmas?",
                )
                return outcome.text
            if _is_conversational_type_text_request(request.content):
                content = self._desktop_interaction._extract_text_to_type(  # noqa: SLF001
                    request.content,
                )
                target_handle = self._desktop_interaction.capture_external_foreground_handle()
                outcome = self._execution_conversation.handle_registered_tool(
                    "desktop.type_text",
                    {"text": content},
                    original_text=request.content,
                    confirmation_text="Voy a escribir en la ventana activa. ¿Confirmas?",
                    pending_target_handle=target_handle,
                )
                return outcome.text
            if _is_conversational_paste_request(request.content):
                target_handle = self._desktop_interaction.capture_external_foreground_handle()
                outcome = self._execution_conversation.handle_registered_tool(
                    "desktop.paste_clipboard",
                    {},
                    original_text=request.content,
                    confirmation_text="Voy a pegar el contenido del portapapeles en la ventana activa. ¿Confirmas?",
                    pending_target_handle=target_handle,
                )
                return outcome.text
            close_request = self._desktop_interaction.close_application_tool_request(
                request.content,
            )
            if close_request is not None:
                tool_name, arguments, confirmation_text = close_request
                outcome = self._execution_conversation.handle_registered_tool(
                    tool_name,
                    arguments,
                    original_text=request.content,
                    confirmation_text=confirmation_text,
                )
                return outcome.text
        if self._desktop_interaction is not None:
            desktop_response = self._desktop_interaction.execute(
                request.content,
                confirm=confirm,
            )
            if desktop_response is not None:
                return desktop_response

        specialist_decision = self.classify_request(request)
        if specialist_decision.route is RequestRoute.AGENT_DELEGATION:
            agent_name = specialist_decision.target_agent_name
            if agent_name is None:
                raise RuntimeError("Specialist route did not select an agent.")
            specialist_response = self._run_specialist_agent(
                agent_name,
                request.content,
            )
            if agent_name == "training" and _requests_pdf_export(request.content):
                if self._execution_conversation is None:
                    return specialist_response.text
                outcome = self._execution_conversation.handle_registered_tool(
                    "training.create_pdf",
                    {
                        "content": specialist_response.text,
                        "output_dir": str(self._training_pdf_output_dir),
                    },
                    original_text=request.content,
                    confirmation_text="Voy a crear y abrir el PDF. ¿Confirmas?",
                )
                return f"{specialist_response.text}\n\n{outcome.text}"
            return specialist_response.text


        if self._execution_conversation is not None:
            last_conversation_result = self._execution_conversation.last_result
            if (
                _is_bare_confirmation_token(request.content)
                and last_conversation_result is not None
                and last_conversation_result.confirmation_id is not None
                and not (
                    self._structured_execution_coordinator is not None
                    and self._structured_execution_coordinator.has_pending_execution()
                )
            ):
                outcome = self._execution_conversation.handle(request.content)
                if not outcome.direct_response_required:
                    return outcome.text

        direct_response = self._process_direct_conversation(request)
        if direct_response is not None:
            return direct_response

        structured_response = self._handle_structured_execution(
            request.content,
            request=request,
        )
        if structured_response is not None:
            visible = self._present_structured_execution(structured_response)
            self._remember_structured_turn(request.content, structured_response)
            return visible

        if self._execution_conversation is not None:
            outcome = self._execution_conversation.handle(request.content)

            if not outcome.direct_response_required:
                return outcome.text

        return self._process_prompt_without_execution(
            request.content,
            confirm,
            request=request,
        )

    def _process_pending_agent_followup(self, request: AtlasRequest) -> str | None:
        """Continue one explicitly requested agent follow-up before normal routing."""
        pending = self._pending_agent_followup
        if pending is None:
            return None
        if _normalize_confirmation_text(request.content) in {
            "cancelar",
            "cancela",
            "cancelalo",
            "cancelala",
            "olvidalo",
        }:
            self._pending_agent_followup = None
            return "Seguimiento del agente cancelado."

        decision = self.classify_request(request)
        if _replaces_pending_agent_followup(decision, pending.agent_name):
            self._pending_agent_followup = None
            return None

        return self._run_specialist_agent(pending.agent_name, request.content).text

    def _run_specialist_agent(
        self,
        agent_name: str,
        prompt: str,
    ) -> AgentResponse:
        """Run one specialist and update the bounded continuation state."""
        specialist_agent = self._registry.get(agent_name)
        if specialist_agent is None:
            self._pending_agent_followup = None
            raise RuntimeError(f"Agent '{agent_name}' is not registered.")
        model_task = "coding" if agent_name in {"code", "coding"} else "chat"
        self._memory.add_user(prompt)
        messages = self._memory.history()
        preflight = getattr(specialist_agent, "preflight", None)
        raw_response = preflight(messages) if callable(preflight) else None
        if raw_response is None:
            try:
                raw_response = self._model_inference_runner.run(
                    self._model_selection_policy.create_request(task=model_task),
                    lambda selected_model: specialist_agent.run(
                        model=selected_model,
                        messages=messages,
                    ),
                )
            except (ModelHealthCheckError, InferenceFallbackExhaustedError) as error:
                local_fallback = getattr(
                    specialist_agent, "local_calculation_fallback", None
                )
                raw_response = local_fallback(messages) if callable(local_fallback) else None
                if raw_response is None:
                    raise error
            except ModelSelectionError as error:
                if self._model_selection_policy != ModelSelectionPolicy():
                    raise
                raw_response = specialist_agent.run(
                    model=self._model_manager.choose_model(
                        model_task,
                        selection_result=error.result,
                    ),
                    messages=messages,
                )
        response = (
            raw_response
            if isinstance(raw_response, AgentResponse)
            else AgentResponse(text=raw_response)
        )
        self._memory.add_assistant(response.text)
        self._pending_agent_followup = (
            PendingAgentFollowUp(
                agent_name=agent_name,
                original_prompt=(
                    self._pending_agent_followup.original_prompt
                    if self._pending_agent_followup is not None
                    and self._pending_agent_followup.agent_name == agent_name
                    else prompt
                ),
            )
            if response.requires_follow_up
            else None
        )
        return response

    def _handle_pending_skill_creation(self, prompt: str) -> str | None:
        proposal = self._pending_skill_creation_proposal
        if proposal is None:
            return None
        normalized = _normalize_confirmation_text(prompt)
        if normalized in {"no", "n", "cancelar", "cancela"}:
            self._pending_skill_creation_proposal = None
            return "Creación de Skill cancelada. No se han realizado cambios."
        expected = _normalize_confirmation_text(
            f"AUTORIZAR {proposal.skill_id} {proposal.authorization_token}"
        )
        if normalized != expected:
            if normalized.startswith("autorizar "):
                return "UNSUPPORTED_FOR_SAFE_CREATION: autorización no corresponde a la propuesta activa."
            return None
        self._pending_skill_creation_proposal = None
        if self._capability_gap_detector is None:
            return "UNSUPPORTED_FOR_SAFE_CREATION: detector no disponible."
        authorization = f"AUTORIZAR {proposal.skill_id} {proposal.authorization_token}"
        return self._capability_gap_detector.apply_declarative_skill(
            proposal,
            authorization,
            self._project_root,
        )

    def _handle_pending_capability_proposal(self, prompt: str) -> str | None:
        """Prepare or cancel a pending capability proposal without applying it."""
        proposal = self._pending_capability_proposal
        if proposal is None:
            return None

        normalized = unicodedata.normalize("NFD", prompt)
        normalized = "".join(
            character
            for character in normalized
            if unicodedata.category(character) != "Mn"
        ).casefold().strip()
        if normalized in {"no", "n", "cancelar", "cancela"}:
            self._pending_capability_proposal = None
            return "Preparación de mejora cancelada. No se han realizado cambios."
        if normalized not in {"si", "s", "si, por favor", "si por favor", "vale", "ok", "de acuerdo", "adelante"}:
            return None

        self._pending_capability_proposal = None
        self._prepared_capability_proposal = proposal
        coding_agent = self._registry.get("coding")
        prepare = getattr(coding_agent, "prepare_capability_plan", None)
        if not callable(prepare):
            return "No hay un CodingAgent disponible para preparar la mejora. No se han realizado cambios."
        return prepare(
            capability_id=proposal.capability_id,
            implementation=proposal.minimum_scope,
            planned_files=proposal.planned_files,
            focused_tests=proposal.focused_tests,
            risk=proposal.risk,
        ) + "\n¿Autorizas APLICAR este plan? [s/N]"
    def _handle_validated_capability_closure(self, prompt: str) -> str | None:
        """Require final human approval before closing a validated capability."""
        proposal = self._validated_capability_proposal
        if proposal is None:
            return None
        normalized = _normalize_confirmation_text(prompt)
        if normalized not in {"si", "s", "vale", "ok", "de acuerdo", "adelante", "no", "n", "cancelar", "cancela"}:
            return None
        coding_agent = self._registry.get("coding")
        close = getattr(coding_agent, "close_validated_capability_plan", None)
        if not callable(close):
            return "No hay un CodingAgent disponible para cerrar la mejora."
        approved = normalized in {"si", "s", "vale", "ok", "de acuerdo", "adelante"}
        response = close(proposal.capability_id, approved=approved)
        if approved and getattr(coding_agent, "capability_validation_status", None) != "VALIDATED":
            self._validated_capability_proposal = None
        elif not approved:
            self._validated_capability_proposal = None
        return response

    def _handle_prepared_capability_application(self, prompt: str) -> str | None:
        """Require a second explicit authorization before applying a prepared plan."""
        proposal = self._prepared_capability_proposal
        if proposal is None:
            return None
        normalized = _normalize_confirmation_text(prompt)
        if normalized in {"no", "n", "cancelar", "cancela"}:
            self._prepared_capability_proposal = None
            return "Aplicación de mejora cancelada. No se han realizado cambios."
        if normalized not in {"si", "s", "vale", "ok", "de acuerdo", "adelante"}:
            return None
        self._prepared_capability_proposal = None
        coding_agent = self._registry.get("coding")
        apply = getattr(coding_agent, "apply_prepared_capability_plan", None)
        if not callable(apply):
            return "No hay un CodingAgent disponible para aplicar la mejora. No se han realizado cambios."
        response = apply(proposal.capability_id)
        if getattr(coding_agent, "capability_validation_status", None) == "VALIDATED":
            self._validated_capability_proposal = proposal
        return response
    def _temperature_conversion_response(self, prompt: str) -> str | None:
        """Run the registered deterministic temperature capability when requested."""
        if self._tool_registry is None:
            return None
        match = re.fullmatch(r"\s*convierte\s+([-+]?\d+(?:[.,]\d+)?)\s+grados?\s+celsius\s+a\s+fahrenheit\.?\s*", prompt, re.IGNORECASE)
        if match is None or not self._tool_registry.exists("temperature_conversion"):
            return None
        celsius = float(match.group(1).replace(",", "."))
        fahrenheit = self._tool_registry.get("temperature_conversion").execute(ToolContext({"celsius": celsius}))
        return f"{celsius:g} grados Celsius equivalen a {fahrenheit:g} grados Fahrenheit."

    def _web_research_response(self, prompt: str) -> str | None:
        """Run one bounded web search only for explicit web or research requests."""
        request = _web_research_request(prompt)
        if request is None:
            return None
        query, limit = request
        tool = self._web_search_tool
        if tool is None and self._tool_registry is not None and self._tool_registry.exists("web_search"):
            candidate = self._tool_registry.get("web_search")
            tool = candidate if isinstance(candidate, WebSearchTool) else None
        if tool is None:
            return "La búsqueda web no está disponible en esta instalación."
        try:
            results = tool.search(query, max_results=limit)
        except WebSearchTimeoutError:
            return "La búsqueda web agotó el tiempo de espera. Inténtalo de nuevo más tarde."
        except WebSearchError:
            return "No se pudo completar la búsqueda web. Inténtalo de nuevo más tarde."
        except ValueError:
            return "Necesito un tema concreto para buscar en internet."
        if not results:
            return f"No encontré resultados web para: {query}."
        evidence = [
            f"- {result.title}: {result.snippet or 'Sin resumen disponible.'}"
            for result in results
        ]
        sources = [
            f"{index}. {result.title} ({result.source})\n   {result.url}"
            for index, result in enumerate(results, start=1)
        ]
        return (
            f"Resumen de la búsqueda sobre \"{query}\":\n"
            + "\n".join(evidence)
            + "\n\nFuentes:\n"
            + "\n".join(sources)
        )
    def execute_capability(
        self,
        request: CapabilityExecutionRequest,
    ) -> CapabilityExecutionResult:
        """Execute one explicit structured capability request."""
        if self._capability_execution_service is None:
            return unavailable_capability_execution_result()

        return self._capability_execution_service.execute(request)

    def classify_prompt(
        self,
        prompt: str,
    ) -> RouteDecision:
        """Classify text input through the request gateway without executing it."""
        return self.classify_request(self._request_gateway.from_text(prompt))

    def classify_request(
        self,
        request: AtlasRequest,
    ) -> RouteDecision:
        """Classify an AtlasRequest without executing the selected route."""
        classify_request = getattr(self._router, "classify_request", None)
        if not callable(classify_request):
            raise RuntimeError("Router does not support request classification.")
        return classify_request(request)

    def process_request(
        self,
        request: AtlasRequest,
    ) -> RouteExecutionResult:
        """Classify and execute one existing AtlasRequest without recreating it."""
        if not isinstance(request, AtlasRequest):
            raise TypeError("request must be an AtlasRequest.")
        if self._operational_route_executor is None:
            raise RuntimeError("OperationalRouteExecutor is not configured.")
        decision = self.classify_request(request)
        return self._operational_route_executor.execute(request, decision)

    def process_prompt_result(
        self,
        prompt: str,
    ) -> RouteExecutionResult:
        """Process text through Gateway, operational routing and route execution."""
        return self.process_request(self._request_gateway.from_text(prompt))

    def process_voice_prompt_result(
        self,
        prompt: str,
        **voice_metadata,
    ) -> RouteExecutionResult:
        """Process non-empty voice transcription through the same operational flow."""
        return self.process_request(
            self._request_gateway.from_voice(prompt, **voice_metadata)
        )

    def present_route_execution(
        self,
        result: RouteExecutionResult,
    ) -> str:
        """Convert one operational result to safe temporary visible text."""
        return self._route_execution_presenter.present(result)

    def route_request(
        self,
        request: AtlasRoutingRequest,
    ) -> AtlasRoutingResult:
        """Route one explicit structured Atlas request."""
        if self._atlas_router is None:
            return AtlasRoutingResult(
                status=AtlasRoutingStatus.SERVICE_UNAVAILABLE,
                route_type=(
                    request.route_type
                    if isinstance(request, AtlasRoutingRequest)
                    else AtlasRouteType.UNKNOWN
                ),
                request_id=(
                    request.request_id
                    if isinstance(request, AtlasRoutingRequest)
                    else None
                ),
                error_code="ATLAS_ROUTER_UNAVAILABLE",
                message="Atlas router is not configured.",
            )

        return self._atlas_router.route(request)

    def route_structured_request(
        self,
        request: StructuredAtlasRequest,
    ) -> AtlasRoutingResult:
        """Adapt and route one already-classified structured request."""
        if self._atlas_request_adapter is None:
            return unavailable_atlas_request_adapter_result(request)

        adaptation = self._atlas_request_adapter.adapt(request)
        if not adaptation.adapted or adaptation.routing_request is None:
            return adaptation_failure_to_routing_result(adaptation)

        return self.route_request(adaptation.routing_request)

    def route_structured_input(
        self,
        structured_input: StructuredInput,
    ) -> AtlasRoutingResult:
        """Normalize, classify, adapt, and route one already-structured input."""
        if self._atlas_request_normalizer is None:
            return unavailable_atlas_request_normalizer_result(structured_input)

        normalization = self._atlas_request_normalizer.normalize(structured_input)
        if not normalization.normalized or normalization.structured_input is None:
            return normalization_failure_to_routing_result(normalization)

        if self._atlas_request_classifier is None:
            return unavailable_atlas_request_classifier_result(normalization.structured_input)

        classification = self._atlas_request_classifier.classify(normalization.structured_input)
        if not classification.classified or classification.structured_request is None:
            return classification_failure_to_routing_result(classification)

        return self.route_structured_request(classification.structured_request)

    def confirm_structured_execution(
        self,
        confirmation_token: str,
        *,
        objective: str | None = None,
    ) -> StructuredExecutionResponse:
        """Continue a pending structured execution without regenerating its plan."""
        if self._structured_execution_coordinator is None:
            return StructuredExecutionResponse(
                handled=True,
                status="structured_execution_unavailable",
                message="Ejecucion estructurada no disponible.",
                error_code="STRUCTURED_EXECUTION_UNAVAILABLE",
                error="structured execution coordinator is not configured",
            )

        if not self._structured_plan_execution_enabled:
            return StructuredExecutionResponse(
                handled=True,
                status="structured_plan_execution_disabled",
                message=(
                    "Ejecucion estructurada desactivada. "
                    "El plan no se ha ejecutado."
                ),
                error_code="STRUCTURED_PLAN_EXECUTION_DISABLED",
            )

        if self._structured_execution_active:
            return StructuredExecutionResponse(
                handled=True,
                status="execution_in_progress",
                message="Atlas está ejecutando un plan.",
                error_code="STRUCTURED_EXECUTION_IN_PROGRESS",
            )

        self._execution_progress_presenter.reset()
        self._execution_cancel_requested = False
        self._structured_execution_active = True
        try:
            return self._structured_execution_coordinator.confirm(
                confirmation_token,
                objective=objective,
                control=self._execution_control(),
                on_execution_progress=self.on_execution_progress,
            )
        except KeyboardInterrupt:
            self._execution_cancel_requested = True
            return StructuredExecutionResponse(
                handled=True,
                status="cancelled",
                message="Ejecución cancelada.",
                error_code="EXECUTION_CANCELLED",
            )
        finally:
            self._structured_execution_active = False

    def _process_prompt_without_execution(
        self,
        prompt: str,
        confirm,
        *,
        request: AtlasRequest | None = None,
    ) -> str:
        """Process text through the pre-existing conversational flow."""
        request = request or self._request_gateway.from_text(prompt)
        prompt = request.content
        coding_agent = self._registry.get("coding")

        match = re.fullmatch(r"\s*aplicar\s+([A-Za-z0-9_-]+)\s*", prompt, re.IGNORECASE)
        if match is not None:
            authorize = getattr(coding_agent, "authorize_pending_change", None)
            if not callable(authorize):
                return "No hay una propuesta de código pendiente para aplicar."
            try:
                change = authorize(match.group(1))
            except PendingCodingChangeError as error:
                return str(error)
            try:
                result = self._write_file.execute(str(change.path), change.proposed_content)
            except Exception as error:
                return "La autorización fue consumida, pero no se pudo escribir la propuesta. Motivo: " + str(error)
            return f"Cambio aplicado en '{change.relative_path}'.\n{result}"

        if self._correction_interaction is not None:
            correction_response = self._correction_interaction.execute(
                prompt=prompt,
                project_root=self._project_root,
                choose_model=self._model_manager.choose_model,
                confirm=confirm,
            )

            if correction_response is not None:
                return correction_response

        if self._refactoring_interaction is not None:
            refactoring_response = self._refactoring_interaction.execute(
                prompt=prompt,
                project_root=self._project_root,
                confirm=confirm,
            )

            if refactoring_response is not None:
                return refactoring_response

        self._memory.add_user(prompt)
        plan = self._planner.create_plan(prompt)
        route_request = getattr(self._router, "route_request", None)
        if callable(route_request):
            try:
                agent_name = route_request(request, plan=plan)
            except TypeError:
                agent_name = route_request(request)
        else:
            agent_name = self._router.route(plan)
        agent = self._registry.get(agent_name)

        if agent is None:
            raise RuntimeError(
                f"Agent '{agent_name}' is not registered."
            )

        messages = self._memory.history()
        supports_fallback = all(
            callable(getattr(self._model_manager, name, None))
            for name in ("select_model", "select_fallback")
        )
        if supports_fallback:
            try:
                response = self._model_inference_runner.run(
                    self._model_selection_policy.create_request(task=agent_name),
                    lambda selected_model: agent.run(
                        model=selected_model,
                        messages=messages,
                    ),
                )
            except ModelSelectionError as error:
                if self._model_selection_policy != ModelSelectionPolicy():
                    raise
                response = agent.run(
                    model=self._model_manager.choose_model(
                        agent_name,
                        selection_result=error.result,
                    ),
                    messages=messages,
                )
        else:
            response = agent.run(
                model=self._model_manager.choose_model(agent_name),
                messages=messages,
            )
        self._memory.add_assistant(response)

        return response

    def _process_skill_request(self, request: AtlasRequest) -> str | None:
        """Resolve and execute an explicitly requested registered Skill."""
        if self._skill_system is None:
            return None

        skill_id = _requested_skill_id(request.content, self._skill_system)
        if skill_id is None:
            return None

        resolution = self._skill_system.skill_resolver.resolve(
            SkillResolutionRequest(required_skill_ids=(skill_id,))
        )
        if (
            resolution.status is not SkillResolutionStatus.RESOLVED
            or resolution.selected_skill is None
        ):
            return "No pude resolver la Skill solicitada."

        execution = self._skill_system.skill_executor.execute(
            SkillExecutionRequest(
                skill=resolution.selected_skill,
                inputs=_skill_inputs_from_text(
                    request.content,
                    resolution.selected_skill,
                ),
                metadata={"source": "text", "request_id": request.request_id},
            )
        )
        if not execution.completed:
            return execution.safe_message or "No pude ejecutar la Skill solicitada."

        return _present_skill_output(execution.output)

    def _handle_structured_execution(
        self,
        prompt: str,
        *,
        request: AtlasRequest | None = None,
    ) -> StructuredExecutionResponse | None:
        request = request or self._request_gateway.from_text(prompt)
        route_decision = self._structured_route_decision(request)
        if (
            route_decision is not None
            and route_decision.target_tool_name in {"read_file", "write_file"}
            and _contains_windows_absolute_path(prompt)
        ):
            # Windows absolute paths keep the supervised single-tool flow.
            return None
        if (
            route_decision is not None
            and route_decision.target_tool_name
            in {"gmail_list", "gmail_read", "gmail_send"}
        ):
            # Gmail keeps the supervised conversation flow with its own
            # confirmation and presentation.
            return None
        response = self._handle_structured_execution_core(
            prompt,
            request_id=request.request_id,
        )
        if response is None or not response.handled:
            return response
        response = replace(
            response,
            original_request=request.content,
            route_decision=route_decision,
        )
        self._last_structured_execution_response = response
        return response

    def _handle_structured_execution_core(
        self,
        prompt: str,
        *,
        request_id: str | None = None,
    ) -> StructuredExecutionResponse | None:
        if self._structured_execution_coordinator is None:
            return None

        if self._structured_planning_active:
            return StructuredExecutionResponse(
                handled=True,
                status="planning_in_progress",
                message="Atlas está generando un plan.",
                error_code="STRUCTURED_PLANNING_IN_PROGRESS",
            )

        if self._structured_execution_active:
            return StructuredExecutionResponse(
                handled=True,
                status="execution_in_progress",
                message="Atlas está ejecutando un plan.",
                error_code="STRUCTURED_EXECUTION_IN_PROGRESS",
            )

        if not self._structured_execution_enabled:
            if self._structured_execution_coordinator.has_pending_execution():
                return self._structured_execution_coordinator.cancel_pending()
            return None

        if _is_structured_resume_cancel_intent(prompt):
            return self._structured_execution_coordinator.discard_resumable_execution()

        if _is_structured_resume_intent(prompt):
            return self._handle_resumable_structured_execution()

        if self._structured_execution_coordinator.has_pending_execution():
            pending_response = self._handle_pending_structured_execution(prompt)
            if pending_response is not None:
                return pending_response

        use_streaming = self._structured_plan_streaming_enabled
        execute_plan = self._structured_plan_execution_enabled
        on_progress = (
            self.on_planning_progress
            if use_streaming and self._structured_planning_progress_enabled
            else None
        )
        self._planning_progress_presenter.reset()
        self._execution_progress_presenter.reset()
        self._execution_cancel_requested = False
        self._structured_planning_active = True
        self._structured_execution_active = execute_plan
        try:
            response = self._structured_execution_coordinator.handle(
                prompt,
                on_planning_progress=on_progress,
                planning_control=ExecutionControl(),
                control=self._execution_control() if execute_plan else None,
                on_execution_progress=(
                    self.on_execution_progress if execute_plan else None
                ),
                execute_after_planning=execute_plan,
                request_id=request_id,
            )
        except KeyboardInterrupt:
            self._execution_cancel_requested = True
            if execute_plan:
                return StructuredExecutionResponse(
                    handled=True,
                    status="cancelled",
                    message="Ejecución cancelada.",
                    error_code="EXECUTION_CANCELLED",
                )
            return StructuredExecutionResponse(
                handled=True,
                status="planning_cancelled",
                message="Planificación cancelada.",
                error_code="STRUCTURED_PLAN_PROVIDER_CANCELLED",
            )
        finally:
            self._structured_planning_active = False
            self._structured_execution_active = False

        if not response.handled:
            return None

        return response

    def _structured_route_decision(
        self,
        request: AtlasRequest,
    ) -> RouteDecision | None:
        """Classify once before structured planning when the router supports it."""
        try:
            return self.classify_request(request)
        except RuntimeError:
            return None

    @staticmethod
    def _present_structured_execution(
        response: StructuredExecutionResponse,
    ) -> str:
        report = response.operational_report
        if report is None or report.status.value == "USER_ACTION_REQUIRED":
            return response.message

        if report.status.value in {"COMPLETED", "COMPLETED_WITH_RECOVERY"}:
            confirmation_intent = _classify_structured_confirmation_intent(
                response.original_request or ""
            )
            prefix = "Plan confirmado. " if confirmation_intent == "confirm" else ""
            correction_status = report.objective_correction.get("status")
            verification_status = report.goal_verification_status
            if correction_status == "VERIFIED_AFTER_CORRECTION":
                outcome = "Objetivo corregido y verificado."
            elif verification_status == "VERIFIED":
                outcome = "Objetivo verificado."
            elif verification_status == "NOT_VERIFIED":
                outcome = (
                    "Ejecucion completada, pero el objetivo no fue verificado."
                )
            elif verification_status == "PARTIALLY_VERIFIED":
                outcome = (
                    "Ejecucion completada, pero el objetivo solo fue "
                    "verificado parcialmente."
                )
            elif verification_status == "INCONCLUSIVE":
                outcome = (
                    "Ejecucion completada, pero no hay evidencia suficiente "
                    "para verificar el objetivo."
                )
            else:
                outcome = "Ejecucion completada."
            lines = [prefix + outcome]
            for step in report.steps:
                if step.result:
                    label = step.tool_name or step.description
                    lines.append(f"{label}: {step.result}")
            if report.warnings:
                lines.append("Aviso: " + report.warnings[0])
            return "\n".join(lines)

        if report.status.value == "FAILED":
            error = next((step.error for step in report.steps if step.error), None)
            if error:
                return f"No pude completar la accion. Error: {error}"
            return "No pude completar la accion. Consulta el log para mas detalle."

        if report.status.value == "CANCELLED":
            return "La ejecucion fue cancelada. No se ejecuto ninguna accion pendiente."

        return response.message

    def _load_persisted_structured_execution(self) -> StructuredExecutionResponse | None:
        if (
            self._structured_execution_coordinator is None
            or not self._structured_execution_enabled
        ):
            return None

        response = (
            self._structured_execution_coordinator.load_persisted_resumable_execution()
        )
        if response.status == "resumable_execution_loaded":
            return response
        if response.status == "resumable_execution_invalid":
            return response
        return None

    def on_planning_progress(
        self,
        progress: StructuredPlanningProgress,
    ) -> None:
        """Receive safe structured-planning progress metadata."""
        self._planning_progress_presenter.handle(progress)

    def on_execution_progress(
        self,
        progress: ExecutionProgress,
    ) -> None:
        """Receive safe execution progress metadata."""
        if progress.phase == "preparing":
            self._structured_planning_active = False
            self._structured_execution_active = True
        self._execution_progress_presenter.handle(progress)

    def _execution_control(self) -> ExecutionControl:
        return ExecutionControl(
            should_cancel=lambda: self._execution_cancel_requested,
            cancellation_reason="Ejecución cancelada por el usuario.",
        )

    def _handle_resumable_structured_execution(self) -> StructuredExecutionResponse:
        assert self._structured_execution_coordinator is not None

        if not self._structured_plan_execution_enabled:
            return StructuredExecutionResponse(
                handled=True,
                status="structured_plan_execution_disabled",
                message=(
                    "Ejecucion estructurada desactivada. "
                    "El plan no se ha ejecutado."
                ),
                error_code="STRUCTURED_PLAN_EXECUTION_DISABLED",
            )

        state = self._structured_execution_coordinator.resumable_execution()
        if state is None:
            return self._structured_execution_coordinator.resume_pending_execution()

        total_steps = len(state.original_plan.ordered_steps)
        next_index = _first_pending_step_index(
            state.original_plan.ordered_steps,
            state.pending_step_ids,
        )
        self._print_atlas("Reanudando ejecución...")
        self._print_atlas(
            f"Se conservarán {len(state.completed_step_ids)} pasos completados."
        )
        if next_index is not None:
            self._print_atlas(
                f"Continuando desde el paso {next_index} de {total_steps}..."
            )
        self._print_atlas("Ejecución reanudada.")

        self._execution_progress_presenter.reset()
        self._execution_cancel_requested = False
        self._structured_execution_active = True
        try:
            response = self._structured_execution_coordinator.resume_pending_execution(
                control=self._execution_control(),
                on_execution_progress=self.on_execution_progress,
            )
        except KeyboardInterrupt:
            self._execution_cancel_requested = True
            return StructuredExecutionResponse(
                handled=True,
                status="cancelled",
                message="Ejecución cancelada.",
                error_code="EXECUTION_CANCELLED",
            )
        finally:
            self._structured_execution_active = False

        if response.status == "rejected":
            self._print_atlas("No se puede reanudar la ejecución.")
            if response.error_code == "VALIDATION_MISMATCH":
                self._print_atlas("La ejecución pendiente ya no es válida.")

        return response

    def _handle_pending_structured_execution(
        self,
        prompt: str,
    ) -> StructuredExecutionResponse:
        assert self._structured_execution_coordinator is not None

        intent = _classify_structured_confirmation_intent(prompt)
        if intent == "confirm":
            if not self._structured_plan_execution_enabled:
                return StructuredExecutionResponse(
                    handled=True,
                    status="structured_plan_execution_disabled",
                    message=(
                        "Ejecucion estructurada desactivada. "
                        "El plan no se ha ejecutado."
                    ),
                    error_code="STRUCTURED_PLAN_EXECUTION_DISABLED",
                )

            self._execution_progress_presenter.reset()
            self._execution_cancel_requested = False
            self._structured_execution_active = True
            try:
                response = self._structured_execution_coordinator.confirm_pending(
                    control=self._execution_control(),
                    on_execution_progress=self.on_execution_progress,
                )
            except KeyboardInterrupt:
                self._execution_cancel_requested = True
                return StructuredExecutionResponse(
                    handled=True,
                    status="cancelled",
                    message="Ejecución cancelada.",
                    error_code="EXECUTION_CANCELLED",
                )
            finally:
                self._structured_execution_active = False
            if response.status == "completed":
                return _with_message_prefix(
                    response,
                    "Plan confirmado. ",
                )
            return response

        if intent == "cancel":
            return self._structured_execution_coordinator.cancel_pending()

        if intent == "show":
            return self._structured_execution_coordinator.show_pending()

        return StructuredExecutionResponse(
            handled=True,
            status="confirmation_ambiguous",
            message=(
                "No he ejecutado nada. Hay un plan pendiente. Responde "
                "'confirmo', 'cancela' o 'muestrame el plan'."
            ),
            plan=(
                self._structured_execution_coordinator.pending_execution().plan
                if self._structured_execution_coordinator.pending_execution() is not None
                else None
            ),
            validation_result=(
                self._structured_execution_coordinator.pending_execution().validation_result
                if self._structured_execution_coordinator.pending_execution() is not None
                else None
            ),
            requires_confirmation=True,
            error_code="CONFIRMATION_AMBIGUOUS",
            error="pending structured execution requires an explicit response",
        )

    def process_voice_prompt(
        self,
        prompt: str,
        confirm,
        *,
        on_model_fragment=None,
    ) -> str:
        """Route transcribed voice text before falling back to the model."""
        prompt = _strip_voice_invocation_prefix(prompt)
        request = self._request_gateway.from_voice(prompt)
        routing_text = self._voice_routing_text(request.content)
        route_voice_command = getattr(self._router, "route_voice_command", None)
        voice_route = (
            route_voice_command(routing_text)
            if callable(route_voice_command)
            else None
        )

        if voice_route == "voice_time":
            return f"Son las {self._time_words(self._now_provider())}."

        if voice_route == "voice_date":
            return f"Hoy es {self._date_words(self._now_provider())}."

        if voice_route == "voice_datetime":
            now = self._now_provider()
            return (
                f"Son las {self._time_words(now)} del "
                f"{self._date_words(now)}."
            )

        if voice_route == "voice_open_notepad":
            return self._execute_voice_desktop_command("Abre Bloc de notas", confirm)

        if voice_route == "voice_open_vscode":
            return self._execute_voice_desktop_command(
                "Abre Visual Studio Code",
                confirm,
            )

        tool_catalog = self._tool_catalog_response(request.content)
        if tool_catalog is not None:
            return tool_catalog

        capability_status = self._capability_status_response(request.content)
        if capability_status is not None:
            return capability_status

        memory_response = self._process_memory_request(request)
        if memory_response is not None:
            return memory_response

        direct_response = self._process_direct_conversation(
            request,
            raise_on_failure=True,
            output_fragment_sink=on_model_fragment,
        )
        if direct_response is not None:
            return direct_response

        return self.process_prompt(request.content, confirm=confirm)

    def _execute_voice_desktop_command(
        self,
        prompt: str,
        confirm,
    ) -> str:
        """Execute a router-approved voice command through existing tools."""
        if self._desktop_interaction is None:
            return "Herramienta de escritorio no disponible."

        response = self._desktop_interaction.execute(prompt, confirm=confirm)

        if response is None:
            return "Herramienta no disponible para esta frase."

        return response

    def _voice_routing_text(
        self,
        prompt: str,
    ) -> str:
        """Remove voice-only response instructions before router matching."""
        normalized_newlines = prompt.replace("\r\n", "\n")
        marker = "\n\nResponde en "

        if marker in normalized_newlines:
            return normalized_newlines.split(marker, 1)[0].strip()

        return prompt.strip()

    def _capability_status_response(self, prompt: str) -> str | None:
        """Return deterministic status for common daily-use capability questions."""
        normalized = _normalize_confirmation_text(prompt)
        queries = {
            "que puedes hacer",
            "que capacidades estan disponibles",
            "tienes voz",
            "puedes leer archivos",
            "puedes escribir archivos",
        }
        if normalized not in queries:
            return None

        registry = self._tool_registry
        read_available = registry is not None and registry.exists("read_file")
        write_available = registry is not None and registry.exists("write_file")
        write_confirmation = False
        if write_available and registry is not None:
            write_confirmation = registry.descriptor("write_file").requires_confirmation

        if normalized == "tienes voz":
            return (
                "Voz: capacidad opcional no configurada para esta sesion de texto. "
                "El banner de inicio muestra su estado operativo."
            )
        if normalized == "puedes leer archivos":
            return (
                "Si. read_file esta disponible para leer archivos permitidos."
                if read_available
                else "No. read_file no esta disponible en el registro activo."
            )
        if normalized == "puedes escribir archivos":
            if not write_available:
                return "No. write_file no esta disponible en el registro activo."
            suffix = " Requiere confirmacion explicita." if write_confirmation else ""
            return "Si. write_file esta disponible para escribir archivos permitidos." + suffix

        tool_count = len(registry.list()) if registry is not None else 0
        return "\n".join(
            (
                "Capacidades actuales:",
                "- Texto: disponible.",
                "- Voz: opcional; no configurada para esta sesion de texto.",
                f"- Herramientas registradas y disponibles: {tool_count}.",
                "- Lectura de archivos: disponible." if read_available else "- Lectura de archivos: no disponible.",
                (
                    "- Escritura de archivos: disponible; requiere confirmacion."
                    if write_available and write_confirmation
                    else "- Escritura de archivos: disponible."
                    if write_available
                    else "- Escritura de archivos: no disponible."
                ),
            )
        )

    def _process_memory_request(self, request: AtlasRequest) -> str | None:
        """Execute one explicit memory intent through the shared operational route."""
        if self._operational_route_executor is None:
            return None
        decision = self._structured_route_decision(request)
        if decision is None or decision.route is not RequestRoute.MEMORY_QUERY:
            return None

        executable_request = request
        if (
            decision.memory_operation
            in {
                MemoryOperation.STORE,
                MemoryOperation.FORGET,
                MemoryOperation.UPDATE,
            }
            and not request.safety_context.allow_side_effects
            and not request.safety_context.contains_sensitive_data
        ):
            executable_request = replace(
                request,
                safety_context=replace(
                    request.safety_context,
                    allow_side_effects=True,
                ),
            )
        result = self._operational_route_executor.execute(
            executable_request,
            decision,
        )
        response = self._route_execution_presenter.present(result)
        self._memory.add_user(request.content)
        self._memory.add_assistant(response)
        return response

    def _process_execution_history_request(self, request: AtlasRequest) -> str | None:
        """Execute only read-only execution-history commands before conversation."""
        if self._operational_route_executor is None:
            return None
        decision = self._structured_route_decision(request)
        if (
            decision is None
            or decision.route is not RequestRoute.SYSTEM_COMMAND
            or decision.system_command not in {
                SystemCommand.LIST_EXECUTIONS,
                SystemCommand.EXECUTION_DETAIL,
            }
        ):
            return None
        result = self._operational_route_executor.execute(request, decision)
        response = self._route_execution_presenter.present(result)
        self._memory.add_user(request.content)
        self._memory.add_assistant(response)
        return response

    def _process_direct_conversation(
        self,
        request: AtlasRequest,
        *,
        status_sink=None,
        raise_on_failure: bool = False,
        output_fragment_sink=None,
    ) -> str | None:
        """Use the existing bounded direct-response route for conversational turns."""
        if self._operational_route_executor is None:
            return None
        if (
            self._structured_execution_coordinator is not None
            and self._structured_execution_coordinator.has_pending_execution()
        ):
            return None
        if self._execution_conversation is not None and (
            self._execution_conversation.pending_clarification is not None
            or self._execution_conversation.pending_confirmation_id is not None
        ):
            return None

        decision = self._structured_route_decision(request)
        if decision is None:
            return None
        is_reference = _is_conversational_reference_query(request.content)
        if is_reference and not self._has_execution_context():
            response = (
                "Necesito que aclares a que archivo, resultado o error te refieres."
            )
            self._memory.add_user(request.content)
            self._memory.add_assistant(response)
            return response
        if decision.route is not RequestRoute.DIRECT_RESPONSE and not is_reference:
            return None
        if decision.route is not RequestRoute.DIRECT_RESPONSE:
            decision = replace(
                decision,
                route=RequestRoute.DIRECT_RESPONSE,
                reason="Conversational reference with bounded current-session context.",
                target_tool_name=None,
                target_agent_name=None,
                target_session_id=None,
                requires_confirmation=False,
                requires_clarification=False,
                clarification_question=None,
                fallback_route=None,
                system_command=None,
                memory_operation=None,
            )

        if callable(status_sink):
            status_sink("Procesando...")
        result = self._operational_route_executor.execute(
            request,
            decision,
            output_fragment_sink=output_fragment_sink,
        )
        if raise_on_failure and result.status is RouteExecutionStatus.FAILED:
            safe_cause = result.error.safe_cause if result.error is not None else ""
            if "timeout" in safe_cause.casefold():
                raise TimeoutError("La respuesta directa del modelo agotó su timeout.")
            raise RuntimeError(
                result.error.summary
                if result.error is not None
                else "Direct response failed."
            )
        response = self._route_execution_presenter.present(result)
        self._memory.add_user(request.content)
        self._memory.add_assistant(response)
        return response

    def _remember_structured_turn(
        self,
        prompt: str,
        response: StructuredExecutionResponse,
    ) -> None:
        """Keep one bounded safe execution summary in existing temporary memory."""
        self._memory.add_user(prompt)
        self._memory.add_assistant(self._structured_context_summary(response))

    @staticmethod
    def _structured_context_summary(response: StructuredExecutionResponse) -> str:
        report = response.operational_report
        if report is None:
            return _bounded_context_text(
                "Contexto de ejecucion: " + response.message,
            )

        parts = [
            "Contexto de ejecucion:",
            f"Objetivo: {report.objective}",
            f"Estado: {report.status.value}.",
        ]
        for step in report.steps:
            label = step.tool_name or step.description
            if label:
                parts.append(f"Herramienta: {label}.")
            if step.result:
                parts.append(f"Resultado: {step.result}")
            if step.error:
                parts.append(f"Error: {step.error}")
        return _bounded_context_text(" ".join(parts))

    def _has_execution_context(self) -> bool:
        return any(
            message.get("role") == "assistant"
            and message.get("content", "").startswith("Contexto de ejecucion:")
            for message in self._memory.history()
        )

    def _tool_catalog_response(self, prompt: str) -> str | None:
        """Render the active registry without routing or executing tools."""

        if not _is_tool_catalog_query(prompt):
            return None
        if self._tool_registry is None:
            return "El catalogo de herramientas no esta disponible."

        descriptors = self._tool_registry.descriptors()
        lines = [f"Herramientas disponibles ({len(descriptors)}):"]
        for descriptor in descriptors:
            description = " ".join(descriptor.description.split())[:160]
            confirmation = (
                " Requiere confirmacion."
                if descriptor.requires_confirmation
                else ""
            )
            lines.append(
                f"- {descriptor.name}: {description}{confirmation}"
            )
        lines.append(
            "El registro activo solo contiene herramientas disponibles."
        )
        return "\n".join(lines)

    def _print_atlas(
        self,
        response: str,
    ) -> None:
        text = str(response).replace("\ufeff", "")
        encoding = sys.stdout.encoding or "utf-8"
        text = text.encode(encoding, errors="replace").decode(
            encoding,
            errors="replace",
        )
        print()
        print("Atlas:")
        print(text)
        print()

    def _read_typed_exit_command(self) -> str | None:
        """Read a typed exit command when a console line is already available."""
        try:
            import msvcrt
        except ImportError:
            return None

        if not msvcrt.kbhit():
            return None

        characters: list[str] = []

        while msvcrt.kbhit():
            character = msvcrt.getwch()

            if character in ("\r", "\n"):
                break

            characters.append(character)

        text = "".join(characters).strip()
        return text or None

    def _time_words(
        self,
        now: datetime,
    ) -> str:
        """Return natural Spanish time words for voice responses."""
        hour = now.hour
        minute = now.minute
        period = "de la madrugada"

        if 6 <= hour < 12:
            period = "de la mañana"
        elif 12 <= hour < 20:
            period = "de la tarde"
        elif hour >= 20:
            period = "de la noche"

        spoken_hour = hour % 12

        if spoken_hour == 0:
            spoken_hour = 12

        return (
            f"{self._number_words(spoken_hour)} y "
            f"{self._number_words(minute)} {period}"
        )

    def _date_words(
        self,
        now: datetime,
    ) -> str:
        """Return natural Spanish date words for voice responses."""
        weekdays = (
            "lunes",
            "martes",
            "miércoles",
            "jueves",
            "viernes",
            "sábado",
            "domingo",
        )
        months = (
            "enero",
            "febrero",
            "marzo",
            "abril",
            "mayo",
            "junio",
            "julio",
            "agosto",
            "septiembre",
            "octubre",
            "noviembre",
            "diciembre",
        )

        return (
            f"{weekdays[now.weekday()]}, {now.day} de "
            f"{months[now.month - 1]} de {now.year}"
        )

    def _number_words(
        self,
        value: int,
    ) -> str:
        """Return Spanish words for the limited clock range."""
        units = (
            "cero",
            "una",
            "dos",
            "tres",
            "cuatro",
            "cinco",
            "seis",
            "siete",
            "ocho",
            "nueve",
            "diez",
            "once",
            "doce",
            "trece",
            "catorce",
            "quince",
            "dieciséis",
            "diecisiete",
            "dieciocho",
            "diecinueve",
            "veinte",
            "veintiuna",
            "veintidós",
            "veintitrés",
            "veinticuatro",
            "veinticinco",
            "veintiséis",
            "veintisiete",
            "veintiocho",
            "veintinueve",
        )

        if 0 <= value < len(units):
            return units[value]

        tens = {
            30: "treinta",
            40: "cuarenta",
            50: "cincuenta",
        }
        ten = value - (value % 10)
        unit = value % 10

        if unit == 0:
            return tens[ten]

        return f"{tens[ten]} y {units[unit]}"


def _web_research_request(prompt: str) -> tuple[str, int] | None:
    """Recognize explicit web/research prompts without capturing local searches."""
    normalized = unicodedata.normalize("NFD", prompt)
    normalized = "".join(
        character for character in normalized if unicodedata.category(character) != "Mn"
    ).casefold().strip()
    if normalized.startswith("investiga ") or normalized.startswith("research "):
        query = re.sub(r"^(?:investiga|research)\s+(?:sobre\s+)?", "", prompt, flags=re.IGNORECASE).strip()
        return (query, 5) if query else None
    if not normalized.startswith("busca "):
        return None
    web_markers = ("en internet", "en la web", "online", "informacion actual", "noticias", "actual")
    if not any(marker in normalized for marker in web_markers):
        return None
    query = re.sub(
        r"^\s*busca\s+(?:(?:en\s+)?internet|en\s+la\s+web|online)\s+",
        "",
        prompt,
        flags=re.IGNORECASE,
    )
    query = re.sub(
        r"^\s*(?:informaci[oó]n\s+actual|noticias)\s+(?:sobre\s+)?",
        "",
        query,
        flags=re.IGNORECASE,
    ).strip()
    return (query, 3) if query else None


class _PlanningProgressPresenter:
    """Render safe structured-planning progress without exposing model content."""

    _RECEIVING_INTERVAL_SECONDS = 5.0

    def __init__(
        self,
        sink,
    ) -> None:
        self._sink = sink
        self._printed_phases: set[str] = set()
        self._first_token_printed = False
        self._receiving_printed = False
        self._last_receiving_printed_at = 0.0

    def handle(
        self,
        progress: StructuredPlanningProgress,
    ) -> None:
        message = self._message_for(progress)
        if message is None:
            return
        self._sink(message)

    def reset(self) -> None:
        self._printed_phases.clear()
        self._first_token_printed = False
        self._receiving_printed = False
        self._last_receiving_printed_at = 0.0

    def _message_for(
        self,
        progress: StructuredPlanningProgress,
    ) -> str | None:
        if progress.phase == "preparing":
            return self._once("preparing", "Preparando el plan...")
        if progress.phase == "waiting_model":
            return self._once("waiting_model", "Esperando al modelo...")
        if progress.phase == "receiving" and not self._first_token_printed:
            self._first_token_printed = True
            self._last_receiving_printed_at = time.monotonic()
            return "El modelo ha comenzado a responder."
        if progress.phase == "receiving":
            now = time.monotonic()
            if self._receiving_printed:
                return None
            if now - self._last_receiving_printed_at < self._RECEIVING_INTERVAL_SECONDS:
                return None
            self._receiving_printed = True
            self._last_receiving_printed_at = now
            return "Generando el plan..."
        if progress.phase == "completed":
            return self._once("completed", "Plan generado.")
        if progress.phase == "failed":
            return self._once("failed", "No se pudo generar el plan.")
        return None

    def _once(
        self,
        phase: str,
        message: str,
    ) -> str | None:
        if phase in self._printed_phases:
            return None
        self._printed_phases.add(phase)
        return message


class _ExecutionProgressPresenter:
    """Render safe execution progress without exposing tool internals."""

    def __init__(
        self,
        sink,
    ) -> None:
        self._sink = sink
        self._printed_start = False
        self._printed_completed = False
        self._printed_terminal: set[str] = set()

    def reset(self) -> None:
        self._printed_start = False
        self._printed_completed = False
        self._printed_terminal.clear()

    def handle(
        self,
        progress: ExecutionProgress,
    ) -> None:
        for message in self._messages_for(progress):
            self._sink(message)

    def _messages_for(
        self,
        progress: ExecutionProgress,
    ) -> tuple[str, ...]:
        if progress.phase == "preparing":
            if self._printed_start:
                return ()
            self._printed_start = True
            return (
                "Plan validado.",
                "Iniciando ejecución...",
                "Preparando la ejecución...",
            )

        if progress.phase == "step_started":
            return (self._step_message(progress, "Ejecutando paso"),)

        if progress.phase == "step_completed":
            return (self._step_done_message(progress, "completado"),)

        if progress.phase == "step_failed":
            return (self._step_done_message(progress, "ha fallado"),)

        if progress.phase == "step_retry_scheduled":
            return (self._retry_message(progress),)

        if progress.phase == "step_completed_after_retry":
            return (
                self._step_done_message(
                    progress,
                    "se completó tras un reintento",
                ),
            )

        if progress.phase == "step_retry_exhausted":
            return (self._step_done_message(progress, "agotó sus intentos"),)

        if progress.phase == "interrupted":
            return self._terminal_once("interrupted", "Ejecución interrumpida.")

        if progress.phase == "cancelled":
            return self._terminal_once("cancelled", "Ejecución cancelada.")

        if progress.phase == "completed":
            if self._printed_completed:
                return ()
            self._printed_completed = True
            return ("Ejecución completada.",)

        return ()

    def _terminal_once(
        self,
        phase: str,
        message: str,
    ) -> tuple[str, ...]:
        if phase in self._printed_terminal:
            return ()
        self._printed_terminal.add(phase)
        return (message,)

    def _step_message(
        self,
        progress: ExecutionProgress,
        prefix: str,
    ) -> str:
        index = progress.step_index or 0
        total = progress.total_steps or index
        return f"{prefix} {index} de {total}..."

    def _step_done_message(
        self,
        progress: ExecutionProgress,
        suffix: str,
    ) -> str:
        index = progress.step_index or 0
        return f"Paso {index} {suffix}."

    def _retry_message(
        self,
        progress: ExecutionProgress,
    ) -> str:
        index = progress.step_index or 0
        attempt = progress.attempt_number or 0
        maximum = progress.max_attempts or attempt
        return f"Reintentando paso {index}, intento {attempt} de {maximum}..."


def _classify_structured_confirmation_intent(
    prompt: str,
) -> str:
    """Classify short conversational replies to a pending structured plan."""
    normalized = _normalize_confirmation_text(prompt)

    confirm_phrases = {
        "si",
        "si adelante",
        "confirmo",
        "confirma",
        "adelante",
        "hazlo",
        "ejecuta",
        "de acuerdo",
        "vale",
        "vale hazlo",
        "correcto",
        "acepto",
        "puedes hacerlo",
    }
    cancel_phrases = {
        "no",
        "cancela",
        "cancelar",
        "deten el plan",
        "no lo hagas",
        "dejalo",
        "olvidalo",
        "rechazar",
        "no confirmo",
        "no cancela",
    }
    show_phrases = {
        "que vas a hacer",
        "muestrame el plan",
        "muestra el plan",
        "repite el plan",
        "que se ejecutara",
        "cuales son los riesgos",
        "que riesgos hay",
        "que herramientas usaras",
    }

    if normalized in confirm_phrases:
        return "confirm"

    if normalized in cancel_phrases:
        return "cancel"

    if normalized in show_phrases:
        return "show"

    return "ambiguous"


def _requested_skill_id(prompt: str, skill_system: SkillSystem) -> str | None:
    return skill_intent.requested_skill_id(prompt, skill_system)


_VOICE_INVOCATION_PREFIX_PATTERN = re.compile(
    r"^atlas\b[ ,.;:!?-]*",
    re.IGNORECASE,
)


def _strip_voice_invocation_prefix(prompt: str) -> str:
    """Drop one leading "Atlas, ..." invocation prefix from transcribed voice."""
    stripped = _VOICE_INVOCATION_PREFIX_PATTERN.sub("", prompt.strip(), count=1).strip()
    return stripped or prompt.strip()


def _skill_inputs_from_text(prompt: str, skill) -> dict[str, object]:
    return skill_intent.skill_inputs_from_text(prompt, skill)


def _present_skill_output(output) -> str:
    return skill_intent.present_skill_output(output)

def _is_structured_resume_intent(
    prompt: str,
) -> bool:
    normalized = _normalize_confirmation_text(prompt)
    return normalized in {
        "reanuda",
        "continua",
        "continuar ejecucion",
        "sigue con el plan",
        "retoma",
        "retoma la ejecucion",
    }


def _is_structured_resume_cancel_intent(
    prompt: str,
) -> bool:
    normalized = _normalize_confirmation_text(prompt)
    return normalized in {
        "cancela la ejecucion pendiente",
        "cancelar la ejecucion pendiente",
        "descarta el plan",
        "descartar el plan",
    }


def _first_pending_step_index(
    steps,
    pending_step_ids: tuple[str, ...],
) -> int | None:
    if not pending_step_ids:
        return None

    first_pending = pending_step_ids[0]
    for index, step in enumerate(steps, start=1):
        if step.id == first_pending:
            return index

    return None


def _is_conversational_reference_query(prompt: str) -> bool:
    normalized = _normalize_confirmation_text(prompt)
    markers = (
        "lo que acabas de leer",
        "ese archivo",
        "archivo has leido",
        "error anterior",
        "ultimo resultado",
        "ultima herramienta",
        "vuelve a hacerlo",
        "repitelo",
        "haz lo mismo",
    )
    return any(marker in normalized for marker in markers)


def _bounded_context_text(value: str, limit: int = 1200) -> str:
    normalized = " ".join(value.split())
    sensitive_markers = (
        "api_key",
        "api key",
        "authorization",
        "bearer",
        "credential",
        "password",
        "private_key",
        "secret",
        "token",
    )
    if any(marker in normalized.casefold() for marker in sensitive_markers):
        return "Contexto de ejecucion: [redacted]"
    return normalized[:limit]


def _accepts_keyword(callable_object, name: str) -> bool:
    """Return whether a callable accepts one explicit or arbitrary keyword."""
    try:
        parameters = inspect.signature(callable_object).parameters
    except (TypeError, ValueError):
        return False
    return name in parameters or any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )


def _is_tool_catalog_query(prompt: str) -> bool:
    normalized = _normalize_confirmation_text(prompt)
    return normalized in {
        "que herramientas tienes",
        "lista tus herramientas",
        "que puedes ejecutar",
        "muestrame tus capacidades disponibles",
    }


_WINDOWS_ABSOLUTE_PATH_PATTERN = re.compile(r"\b[A-Za-z]:[\\/]\S")


def _contains_windows_absolute_path(prompt: str) -> bool:
    return bool(_WINDOWS_ABSOLUTE_PATH_PATTERN.search(prompt))


def _replaces_pending_agent_followup(
    decision: RouteDecision,
    pending_agent_name: str,
) -> bool:
    """Allow an explicit operational request to supersede an agent follow-up."""
    if (
        decision.route is RequestRoute.AGENT_DELEGATION
        and decision.target_agent_name != pending_agent_name
    ):
        return True
    return decision.route in {
        RequestRoute.MEMORY_QUERY,
        RequestRoute.SINGLE_TOOL,
        RequestRoute.AUTONOMOUS_EXECUTION,
        RequestRoute.RESUME_EXECUTION,
        RequestRoute.SYSTEM_COMMAND,
    }


def _normalize_confirmation_text(
    text: str,
) -> str:
    normalized = unicodedata.normalize("NFKD", text.strip().lower())
    without_accents = "".join(
        character
        for character in normalized
        if unicodedata.category(character) != "Mn"
    )
    without_punctuation = re.sub(r"[^\w\s]", " ", without_accents)
    return " ".join(without_punctuation.split())


_BARE_CONFIRMATION_TOKENS = {
    "si",
    "s",
    "confirmo",
    "confirmar",
    "vale",
    "ok",
    "no",
    "n",
    "cancela",
    "cancelar",
    "olvidalo",
}


_ASYNC_APPROVAL_YES_TOKENS = {
    "si",
    "s",
    "confirmo",
    "confirmar",
    "vale",
    "ok",
    "adelante",
    "continua",
    "continuar",
}

_ASYNC_APPROVAL_NO_TOKENS = {
    "no",
    "n",
    "cancela",
    "cancelar",
    "olvidalo",
    "rechaza",
    "descarta",
}


def _is_bare_confirmation_token(prompt: str) -> bool:
    """Recognize stray confirmation words that belong to the supervised flow."""
    return _normalize_confirmation_text(prompt) in _BARE_CONFIRMATION_TOKENS


def _with_message_prefix(
    response: StructuredExecutionResponse,
    prefix: str,
) -> StructuredExecutionResponse:
    return replace(response, message=prefix + response.message)


def _conversational_activation_window_title(prompt: str) -> tuple[str, bool] | None:
    """Return (title, strict) for the narrowly supported activation commands.

    ``strict`` commands (activa/ve a/cambia a) surface resolution errors; the
    loose ``pon <title>`` variant falls back to conversation when no window
    matches, so unrelated "pon ..." requests keep their previous behavior.
    """
    normalized = " ".join(prompt.strip().split())
    folded = normalized.casefold()
    for prefix in ("activa ", "ve a ", "cambia a ", "pon "):
        if not folded.startswith(prefix):
            continue
        title = normalized[len(prefix) :].strip()
        if prefix == "pon " and title.casefold().endswith(" delante"):
            title = title[: -len(" delante")].strip()
        if not title:
            return None
        return title, prefix != "pon "
    return None

def _is_conversational_type_text_request(prompt: str) -> bool:
    """Recognize active-window typing requests, excluding file writes."""
    normalized = " ".join(prompt.strip().lower().split())
    if not normalized.startswith("escribe ") and not normalized.startswith("escribe:"):
        return False
    if re.search(r"\ben\s+(?:la\s+)?(?:ventana\s+)?[^ ]+\.[a-z0-9]{1,8}$", normalized):
        return False
    return len(normalized) > len("escribe")


def _is_conversational_paste_request(prompt: str) -> bool:
    """Recognize the active-window clipboard paste command."""
    normalized = " ".join(prompt.strip().lower().split())
    return normalized in {
        "pega",
        "pega el texto",
        "pega el portapapeles",
        "pega el contenido del portapapeles",
        "paste clipboard",
    }

def _requests_pdf_export(text: str) -> bool:
    normalized = unicodedata.normalize("NFD", text.casefold())
    normalized = "".join(
        character for character in normalized if not unicodedata.combining(character)
    )
    return "pdf" in normalized and any(
        marker in normalized for marker in ("guarda", "exporta", "crea")
    )
