"""Core orchestration module for Atlas."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from agents.registry import AgentRegistry

from core.model_manager import ModelManager
from core.planner import Planner
from core.router import Router

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
        self._project_root = project_root or Path(".")
        self._now_provider = now_provider or (lambda: datetime.now().astimezone())

    def start(self) -> None:

        print("Atlas iniciado correctamente.")
        print()

        while True:

            prompt = input("Tú: ")

            if prompt.lower() in ("exit", "quit", "salir"):
                print("\nHasta pronto.")
                break

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
        if self._execution_conversation is not None:
            outcome = self._execution_conversation.handle(prompt)

            if not outcome.direct_response_required:
                return outcome.text

        return self._process_prompt_without_execution(
            prompt,
            confirm,
        )

    def _process_prompt_without_execution(
        self,
        prompt: str,
        confirm,
    ) -> str:
        """Process text through the pre-existing conversational flow."""
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

    def process_voice_prompt(
        self,
        prompt: str,
        confirm,
    ) -> str:
        """Route transcribed voice text before falling back to the model."""
        routing_text = self._voice_routing_text(prompt)
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

        return self.process_prompt(prompt, confirm=confirm)

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
