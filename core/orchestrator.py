"""Core orchestration module for Atlas."""

from __future__ import annotations

from pathlib import Path

from agents.registry import AgentRegistry

from core.model_manager import ModelManager
from core.planner import Planner
from core.router import Router

from memory.conversation import ConversationMemory
from use_cases.correction_interaction import CorrectionInteractionUseCase
from use_cases.desktop_interaction import DesktopInteractionUseCase
from use_cases.refactoring_interaction import RefactoringInteractionUseCase
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
        speech_interaction: SpeechInteractionUseCase | None = None,
        wake_word_interaction: WakeWordInteractionUseCase | None = None,
        voice_conversation: VoiceConversationUseCase | None = None,
        project_root: Path | None = None,
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
        self._speech_interaction = speech_interaction
        self._wake_word_interaction = wake_word_interaction
        self._voice_conversation = voice_conversation
        self._project_root = project_root or Path(".")

    def start(self) -> None:

        print("Atlas iniciado correctamente.")
        print()

        while True:

            prompt = input("Tú: ")

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

                print()
                print("Atlas:")
                print(result)
                print()

                continue

            if prompt.lower() in ("exit", "quit", "salir"):
                print("\nHasta pronto.")
                break

            if self._voice_conversation is not None:
                voice_result = self._voice_conversation.execute(
                    prompt=prompt,
                    process_text=lambda text: self.process_prompt(
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

            response = self.process_prompt(
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
            process_text=lambda text: self.process_prompt(
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
