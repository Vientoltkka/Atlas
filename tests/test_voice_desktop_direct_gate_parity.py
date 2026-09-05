"""Voice desktop commands must follow the same operational routing as text.

Regression: with the real Bootstrap wiring (operational_route_executor
present), the direct-conversation gate in ``process_voice_prompt`` consumed
"maximiza la calculadora" before DesktopInteraction could execute it,
answering conversationally instead of running desktop.maximize_window.
"""
from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace

import pytest

from agents.registry import AgentRegistry
from core.operational_route_executor import (
    OperationalRouteExecutor,
    build_default_route_handlers,
)
from core.orchestrator import AtlasOrchestrator
from core.router import Router
from memory.conversation import ConversationMemory
from use_cases.desktop_interaction import DesktopInteractionUseCase
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


class RecordingExecutor:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def execute(self, tool_name, context):
        self.calls.append((tool_name, dict(context.parameters)))
        if tool_name == "desktop.list_windows":
            return [{"handle": 10, "title": "Calculadora", "rect": (0, 0, 400, 300)}]
        if tool_name == "desktop.get_window_rect":
            return (0, 0, 400, 300)
        if tool_name == "desktop.get_screen_size":
            return (1920, 1080)
        return "ok"


def build_real_wired_orchestrator(executor: RecordingExecutor) -> AtlasOrchestrator:
    """Wire the orchestrator exactly like Bootstrap does.

    The direct responder mimics the real conversational model answer that
    previously consumed the desktop command before DesktopInteraction.
    """
    direct_reply = (
        "No puedo controlar la pantalla de tu dispositivo. "
        "Solo puedo conversar contigo."
    )
    route_executor = OperationalRouteExecutor(
        build_default_route_handlers(
            direct_responder=lambda _request: direct_reply,
            memory=ConversationMemory(),
            agent_registry=AgentRegistry(),
        )
    )
    return AtlasOrchestrator(
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
        operational_route_executor=route_executor,
        now_provider=lambda: datetime(2026, 7, 15, 18, 12).astimezone(),
    )


def run_voice_turn(utterance: str) -> list[tuple[str, dict]]:
    executor = RecordingExecutor()
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
        process_text=lambda text: build_real_wired_orchestrator(
            executor,
        ).process_voice_prompt(
            text,
            confirm=lambda _prompt: "",
        ),
        status_sink=lambda _message: None,
        typed_input=lambda: None,
    )
    return executor.calls


@pytest.mark.parametrize(
    ("utterance", "tool", "arguments"),
    (
        ("abre la calculadora", "desktop.open_application", {"application": "calculadora"}),
        ("maximiza la calculadora", "desktop.maximize_window", {"handle": 10}),
        ("minimiza la calculadora", "desktop.minimize_window", {"handle": 10}),
        ("restaura la calculadora", "desktop.restore_window", {"handle": 10}),
    ),
)
def test_voice_desktop_command_uses_same_operational_routing_as_text(
    utterance: str,
    tool: str,
    arguments: dict,
) -> None:
    executor = RecordingExecutor()
    orchestrator = build_real_wired_orchestrator(executor)

    text_response = orchestrator.process_prompt(
        utterance,
        confirm=lambda _prompt: "",
    )
    text_calls = [
        (name, dict(parameters))
        for name, parameters in executor.calls
        if name == tool
    ]

    voice_executor_calls = run_voice_turn(utterance)
    voice_calls = [
        (name, dict(parameters))
        for name, parameters in voice_executor_calls
        if name == tool
    ]

    assert text_calls, "text path must execute the desktop tool"
    assert voice_calls == text_calls, (
        "voice transcription must follow the same operational routing as text"
    )
