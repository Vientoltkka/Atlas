"""Voice close-application confirmation must follow the same supervised
conversational routing as text, with a single-use confirmation.

Regression: with the real Bootstrap wiring, the voice desktop gate ran
``DesktopInteraction.execute`` directly for "cierra la calculadora", whose
interactive ``confirm`` callable reads stdin inside the voice loop (EOF), and
the short affirmation "sí" was discarded by the transcription gate, so the
close never executed and the pending confirmation was never resolved.
"""
from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace

from agents.registry import AgentRegistry
from bootstrap.bootstrap import Bootstrap
from core.operational_route_executor import (
    OperationalRouteExecutor,
    build_default_route_handlers,
)
from core.orchestrator import AtlasOrchestrator
from core.router import Router
from memory.conversation import ConversationMemory
from tools.desktop.desktop_tools import (
    CloseApplicationTool,
    CloseWindowTool,
    ListProcessesTool,
    ListWindowsTool,
)
from tools.desktop.windows_controller import ProcessInfo
from tools.executor import ToolExecutor
from tools.registry import ToolRegistry
from use_cases.desktop_interaction import DesktopInteractionUseCase
from use_cases.execution_conversation import ExecutionConversationController
from use_cases.speech_engine import SpeechTranscriptionResult
from use_cases.voice_conversation import VoiceConversationUseCase


def speech_result(text: str) -> SpeechTranscriptionResult:
    return SpeechTranscriptionResult(
        text=text,
        language="es",
        audio_duration_seconds=1.0,
        processing_duration_seconds=0.2,
        provider="fake-local",
        microphone_name="Fake Mic",
        completed=True,
        cancelled=False,
        no_speech_detected=False,
        warnings=(),
        summary="fake",
        samples_count=16000,
        rms=0.01,
    )


class FakeSpeechEngine:
    def __init__(self, results: list[str]) -> None:
        self._results = list(results)

    def warm_up(self) -> None:
        pass

    def default_microphone(self):
        return SimpleNamespace(index=0, name="Fake Mic")

    def active_microphone(self):
        return SimpleNamespace(index=1, name="Fake Mic")

    def prepare_stream(self, _settings=None):
        return self.active_microphone()

    def transcribe_once(self, capture_settings=None):
        if not self._results:
            return speech_result("")
        return speech_result(self._results.pop(0))


class FakeWakeWordEngine:
    def wait_for_wake_word(self, status_sink=None):
        from use_cases.wake_word_engine import WakeWordDetectionResult

        return WakeWordDetectionResult(
            wake_word="Atlas",
            detected=True,
            attempts=1,
            elapsed_seconds=0.5,
        )


class FakeSpeechOutputEngine:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def speak(self, text: str) -> None:
        self.calls.append(text)

    def close(self) -> None:
        pass

    def warm_up(self) -> None:
        pass


def _process(pid: int, name: str, *titles: str) -> ProcessInfo:
    return ProcessInfo(
        pid=pid,
        name=name,
        executable_path=None,
        window_titles=tuple(titles),
        is_running=True,
    )


class FakeController:
    def __init__(self) -> None:
        self.close_requests: list[int] = []
        self.close_window_requests: list[int] = []

    def list_processes(self, query: str) -> list[ProcessInfo]:
        normalized = query.strip().lower()
        candidates = {normalized}
        if normalized in {"calculadora", "calculator"}:
            candidates.add("calculatorapp")
        if "calculatorapp" in normalized:
            candidates.add("calculatorapp")
        return [_process(4321, "CalculatorApp.exe")]

    def list_windows(self):
        return [{"handle": 555, "title": "Calculadora", "process_id": 999}]

    def close_window(self, handle: int) -> None:
        self.close_window_requests.append(handle)

    def close_process_windows(self, pid: int) -> int:
        self.close_requests.append(pid)
        return 1


def build_real_wired_orchestrator(controller: FakeController):
    registry = ToolRegistry()
    registry.register(ListProcessesTool(controller))
    registry.register(CloseApplicationTool(controller))
    registry.register(ListWindowsTool(controller))
    registry.register(CloseWindowTool(controller))
    executor = ToolExecutor(registry)
    conversation = ExecutionConversationController(
        Bootstrap.build_execution_coordinator(tool_registry=registry, executor=executor)
    )
    route_executor = OperationalRouteExecutor(
        build_default_route_handlers(
            direct_responder=lambda _request: (
                "No puedo controlar la pantalla de tu dispositivo. "
                "Solo puedo conversar contigo."
            ),
            memory=ConversationMemory(),
            agent_registry=AgentRegistry(),
        )
    )
    orchestrator = AtlasOrchestrator(
        planner=SimpleNamespace(
            create_plan=lambda _prompt: (_ for _ in ()).throw(
                AssertionError("model fallback must not run")
            )
        ),
        router=Router(),
        model_manager=SimpleNamespace(choose_model=lambda _agent: "unused"),
        memory=ConversationMemory(),
        registry=AgentRegistry(),
        write_file=SimpleNamespace(execute=lambda *_args: "unused"),
        desktop_interaction=DesktopInteractionUseCase(executor),
        execution_conversation=conversation,
        operational_route_executor=route_executor,
        now_provider=lambda: datetime(2026, 7, 15, 18, 12).astimezone(),
    )
    return orchestrator, conversation


def run_voice_turn(orchestrator, utterance: str) -> str:
    """Run one manual voice session utterance through the real entrypoint."""
    output = FakeSpeechOutputEngine()
    speech = FakeSpeechEngine([utterance, "adios Atlas"])
    voice = VoiceConversationUseCase(
        speech_engine=speech,
        wake_word_engine=FakeWakeWordEngine(),
        speech_output_engine=output,
        diagnostics_enabled=False,
        clock=lambda: 0.0,
        session_id_factory=lambda: "session-1",
    )
    voice.execute_manual(
        process_text=lambda text: orchestrator.process_voice_prompt(
            text,
            confirm=input,
        ),
        status_sink=lambda _message: None,
        typed_input=lambda: None,
    )
    assert output.calls, "voice turn must produce a spoken response"
    return output.calls[0]


def test_voice_close_asks_confirmation_yes_closes_once_second_yes_reuses_nothing():
    controller = FakeController()
    orchestrator, conversation = build_real_wired_orchestrator(controller)
    confirm = lambda _prompt: ""  # noqa: E731

    text_pending = orchestrator.process_prompt("cierra la calculadora", confirm=confirm)
    text_confirmation = orchestrator.process_prompt("sí", confirm=confirm)
    text_second = orchestrator.process_prompt("sí", confirm=confirm)

    assert controller.close_window_requests == [555]
    assert conversation.pending_confirmation_id is None

    controller2 = FakeController()
    orchestrator2, conversation2 = build_real_wired_orchestrator(controller2)

    # A) voice asks the exact same supervised confirmation as text
    ask = run_voice_turn(orchestrator2, "cierra la calculadora")
    assert ask == text_pending
    assert "¿Confirmas?" in ask
    assert controller2.close_window_requests == []
    assert conversation2.pending_confirmation_id is not None

    # B) one "sí" resolves the pending confirmation and closes exactly once
    yes1 = run_voice_turn(orchestrator2, "sí")
    assert yes1 == text_confirmation
    assert controller2.close_window_requests == [555]
    assert conversation2.pending_confirmation_id is None

    # C) the second "sí" must not reuse the confirmation: no extra close
    yes2 = run_voice_turn(orchestrator2, "sí")
    assert controller2.close_window_requests == [555]
    assert conversation2.pending_confirmation_id is None

    # D) text and voice follow identical routing and responses
    assert yes2 == text_second


def test_voice_close_rejected_on_no_without_closing():
    controller = FakeController()
    orchestrator, conversation = build_real_wired_orchestrator(controller)

    spoken = [
        run_voice_turn(orchestrator, "cierra la calculadora"),
        run_voice_turn(orchestrator, "no"),
    ]

    assert "¿Confirmas?" in spoken[0]
    assert "cancel" in spoken[1].lower()
    assert controller.close_window_requests == []
    assert controller.close_requests == []
    assert conversation.pending_confirmation_id is None
