"""Core orchestration module for Atlas."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import re
import time
import unicodedata

from agents.registry import AgentRegistry

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
from core.model_manager import ModelManager
from core.planner import Planner
from core.router import Router
from core.request_gateway import (
    AtlasRequest,
    RequestGateway,
)
from core.operational_request_router import RouteDecision
from core.operational_route_executor import (
    OperationalRouteExecutor,
    RouteExecutionPresenter,
    RouteExecutionResult,
)
from core.capability_execution_service import (
    CapabilityExecutionRequest,
    CapabilityExecutionResult,
    CapabilityExecutionService,
    unavailable_capability_execution_result,
)
from core.execution_plan_executor import ExecutionControl, ExecutionProgress
from core.execution_history import ExecutionSessionHistory
from core.hybrid_execution_planner import StructuredPlanningProgress
from core.structured_execution import (
    StructuredExecutionCoordinator,
    StructuredExecutionResponse,
)

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
        structured_execution_enabled: bool = False,
        structured_plan_streaming_enabled: bool = False,
        structured_plan_execution_enabled: bool = False,
        structured_planning_progress_enabled: bool = True,
        project_root: Path | None = None,
        now_provider=None,
    ) -> None:

        self._planner = planner
        self._router = router
        self._model_manager = model_manager
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
        self._now_provider = now_provider or (lambda: datetime.now().astimezone())

    @property
    def execution_history(self) -> ExecutionSessionHistory | None:
        """Expose the read-only internal execution-history query API."""
        return self._execution_history

    def start(self) -> None:

        print("Atlas iniciado correctamente.")
        print()
        startup_response = self._load_persisted_structured_execution()
        if startup_response is not None:
            self._print_atlas(startup_response.message)

        while True:

            prompt = input("Tú: ")

            if prompt.lower() in ("exit", "quit", "salir"):
                print("\nHasta pronto.")
                break

            structured_response = self._handle_structured_execution(prompt)
            if structured_response is not None:
                self._print_atlas(structured_response.message)
                continue

            if self._execution_conversation is not None:
                outcome = self._execution_conversation.handle(prompt)

                if not outcome.direct_response_required:
                    self._print_atlas(outcome.text)
                    continue

            if self._voice_conversation is not None:
                voice_result = self._voice_conversation.execute(
                    prompt=prompt,
                    process_text=lambda text: self.process_voice_prompt(
                        text,
                        confirm=input,
                    ),
                    status_sink=self._print_atlas,
                )

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

    def start_voice(self) -> None:
        """Start manual voice conversation mode without wake word."""
        print("Atlas iniciado en modo voz.")
        print()

        if self._voice_conversation is None:
            print("Atlas:")
            print("Modo de voz no disponible.")
            print()
            return

        self._voice_conversation.execute_manual(
            process_text=lambda text: self.process_voice_prompt(
                text,
                confirm=input,
            ),
            status_sink=print,
            typed_input=self._read_typed_exit_command,
        )

    def start_assistant(self) -> None:
        """Start permanent assistant mode with wake word."""
        if self._permanent_assistant is None:
            print("Atlas:")
            print("Modo asistente permanente no disponible.")
            print()
            return

        self._permanent_assistant.run(
            process_text=lambda text: self.process_voice_prompt(
                text,
                confirm=input,
            ),
            status_sink=print,
            typed_input=self._read_typed_exit_command,
        )

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
        request = self._request_gateway.from_text(prompt)
        structured_response = self._handle_structured_execution(request.content)
        if structured_response is not None:
            return structured_response.message

        if self._execution_conversation is not None:
            outcome = self._execution_conversation.handle(request.content)

            if not outcome.direct_response_required:
                return outcome.text

        return self._process_prompt_without_execution(
            request.content,
            confirm,
            request=request,
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

        if (
            prompt.strip().lower() == "s"
            and coding_agent is not None
            and coding_agent.generated_path is not None
        ):
            result = self._write_file.execute(
                coding_agent.generated_path,
                coding_agent.generated_content,
            )
            coding_agent.clear_generated()

            return result

        if self._desktop_interaction is not None:
            desktop_response = self._desktop_interaction.execute(
                prompt,
                confirm=confirm,
            )

            if desktop_response is not None:
                return desktop_response

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

        model = self._model_manager.choose_model(
            agent_name
        )
        response = agent.run(
            model=model,
            messages=self._memory.history(),
        )
        self._memory.add_assistant(response)

        return response

    def _handle_structured_execution(
        self,
        prompt: str,
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
    ) -> str:
        """Route transcribed voice text before falling back to the model."""
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

    def _print_atlas(
        self,
        response: str,
    ) -> None:
        print()
        print("Atlas:")
        print(response)
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


def _with_message_prefix(
    response: StructuredExecutionResponse,
    prefix: str,
) -> StructuredExecutionResponse:
    return StructuredExecutionResponse(
        handled=response.handled,
        status=response.status,
        message=prefix + response.message,
        plan=response.plan,
        validation_result=response.validation_result,
        execution_result=response.execution_result,
        requires_confirmation=response.requires_confirmation,
        confirmation_token=response.confirmation_token,
        error_code=response.error_code,
        error=response.error,
        resumable_state=response.resumable_state,
        partial_state=response.partial_state,
        operational_report=response.operational_report,
    )
