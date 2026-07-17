from __future__ import annotations

from use_cases.permanent_assistant import (
    PermanentAssistantState,
    PermanentAssistantUseCase,
)
from use_cases.voice_conversation import (
    VoiceConversationResult,
    VoiceConversationSession,
)
from use_cases.wake_word_engine import WakeWordDetectionResult


def wake_result(
    detected: bool = True,
    cancelled: bool = False,
    configuration_error: bool = False,
    warnings: tuple[str, ...] = (),
) -> WakeWordDetectionResult:
    return WakeWordDetectionResult(
        wake_word="Atlas",
        detected=detected,
        attempts=1,
        elapsed_seconds=0.1,
        cancelled=cancelled,
        configuration_error=configuration_error,
        warnings=warnings,
    )


def turn_result(
    reason: str = "turn_completed",
    successful_turns: int = 1,
    summary: str = "Turno completado.",
) -> VoiceConversationResult:
    session = VoiceConversationSession(
        session_id="fake-session",
        started_at=0.0,
        successful_turns=successful_turns,
        active=False,
        ended_reason=reason,
        summary=summary,
    )
    return VoiceConversationResult(session=session, messages=[])


class FakeWakeWordEngine:
    def __init__(
        self,
        results: list[WakeWordDetectionResult] | None = None,
        events: list[str] | None = None,
        interrupt: bool = False,
    ) -> None:
        self._results = list(results or [])
        self.events = events
        self.calls = 0
        self.interrupt = interrupt

    def wait_for_wake_word(self) -> WakeWordDetectionResult:
        self.calls += 1

        if self.events is not None:
            self.events.append("wake")

        if self.interrupt:
            raise KeyboardInterrupt

        if not self._results:
            return wake_result(cancelled=True)

        return self._results.pop(0)


class FakeVoiceConversation:
    def __init__(
        self,
        results: list[VoiceConversationResult] | None = None,
        events: list[str] | None = None,
    ) -> None:
        self._results = list(results or [])
        self.events = events
        self.warm_up_calls = 0
        self.close_calls = 0
        self.discard_calls = 0
        self.turn_calls = 0

    def warm_up_resources(self) -> None:
        self.warm_up_calls += 1

        if self.events is not None:
            self.events.append("warm")

    def close_resources(self) -> None:
        self.close_calls += 1

        if self.events is not None:
            self.events.append("close")

    def discard_residual_audio(self) -> None:
        self.discard_calls += 1

        if self.events is not None:
            self.events.append("discard")

    def is_close_command(self, text: str) -> bool:
        return text.strip().lower() in {"salir", "terminar", "cancelar"}

    def execute_assistant_turn(
        self,
        process_text,
        status_sink=None,
        initial_text: str | None = None,
    ) -> VoiceConversationResult:
        self.turn_calls += 1

        if self.events is not None:
            self.events.append("turn")

        process_text(initial_text or f"pregunta {self.turn_calls}")

        if not self._results:
            return turn_result()

        return self._results.pop(0)


def make_assistant(
    wake: FakeWakeWordEngine,
    voice: FakeVoiceConversation,
    max_consecutive_errors: int = 3,
) -> PermanentAssistantUseCase:
    return PermanentAssistantUseCase(
        wake_word_engine=wake,
        voice_conversation=voice,
        max_consecutive_errors=max_consecutive_errors,
        cooldown_seconds=0.0,
    )


def test_wake_word_conversation_returns_to_waiting() -> None:
    events: list[str] = []
    wake = FakeWakeWordEngine([wake_result(), wake_result(cancelled=True)], events)
    voice = FakeVoiceConversation([turn_result()], events)
    assistant = make_assistant(wake, voice)
    processed: list[str] = []

    result = assistant.run(process_text=processed.append)

    assert result.conversations == 1
    assert result.stopped_reason == "cancelled"
    assert wake.calls == 2
    assert voice.turn_calls == 1
    assert voice.discard_calls == 1
    assert processed == ["pregunta 1"]
    assert events == ["warm", "wake", "discard", "turn", "wake", "close"]
    assert result.states.count(PermanentAssistantState.WAITING_WAKE_WORD) == 2


def test_two_consecutive_activations_are_independent_conversations() -> None:
    wake = FakeWakeWordEngine(
        [wake_result(), wake_result(), wake_result(cancelled=True)]
    )
    voice = FakeVoiceConversation([turn_result(), turn_result()])
    assistant = make_assistant(wake, voice)
    processed: list[str] = []

    result = assistant.run(process_text=processed.append)

    assert result.conversations == 2
    assert voice.turn_calls == 2
    assert processed == ["pregunta 1", "pregunta 2"]


def test_detector_does_not_listen_while_tts_turn_is_running() -> None:
    events: list[str] = []
    wake = FakeWakeWordEngine([wake_result(), wake_result(cancelled=True)], events)
    voice = FakeVoiceConversation([turn_result()], events)
    assistant = make_assistant(wake, voice)

    assistant.run(process_text=lambda _text: "respuesta")

    assert events == ["warm", "wake", "discard", "turn", "wake", "close"]


def test_activation_and_request_in_one_phrase_is_processed_without_second_listen() -> None:
    wake = FakeWakeWordEngine(
        [
            WakeWordDetectionResult(
                wake_word="Atlas",
                detected=True,
                attempts=1,
                elapsed_seconds=0.1,
                phrase=type(
                    "Phrase",
                    (),
                    {"completed": True, "text": "que hora es"},
                )(),
            ),
            wake_result(cancelled=True),
        ]
    )
    voice = FakeVoiceConversation([turn_result()])
    assistant = make_assistant(wake, voice)
    processed: list[str] = []

    result = assistant.run(process_text=processed.append)

    assert result.conversations == 1
    assert processed == ["que hora es"]


def test_silence_does_not_end_permanent_process() -> None:
    wake = FakeWakeWordEngine([wake_result(), wake_result(cancelled=True)])
    voice = FakeVoiceConversation(
        [turn_result(reason="no_speech", successful_turns=0, summary="silencio")]
    )
    assistant = make_assistant(wake, voice)

    result = assistant.run(process_text=lambda _text: "unexpected")

    assert result.stopped_reason == "cancelled"
    assert result.conversations == 0
    assert result.recoverable_errors == 1
    assert wake.calls == 2


def test_recoverable_failure_returns_to_waiting_wake_word() -> None:
    wake = FakeWakeWordEngine([wake_result(), wake_result(cancelled=True)])
    voice = FakeVoiceConversation(
        [
            turn_result(
                reason="recoverable_error",
                successful_turns=0,
                summary="fallo temporal de STT",
            )
        ]
    )
    assistant = make_assistant(wake, voice)

    result = assistant.run(process_text=lambda _text: "unexpected")

    assert result.stopped_reason == "cancelled"
    assert result.recoverable_errors == 1
    assert wake.calls == 2


def test_explicit_exit_command_stops_cycle() -> None:
    wake = FakeWakeWordEngine([wake_result()])
    voice = FakeVoiceConversation(
        [turn_result(reason="explicit_close", successful_turns=0)]
    )
    assistant = make_assistant(wake, voice)

    result = assistant.run(process_text=lambda _text: "unexpected")

    assert result.stopped_reason == "explicit_close"
    assert wake.calls == 1
    assert voice.close_calls == 1


def test_typed_exit_command_stops_before_next_wake_word() -> None:
    wake = FakeWakeWordEngine([])
    voice = FakeVoiceConversation([])
    assistant = make_assistant(wake, voice)

    result = assistant.run(
        process_text=lambda _text: "unexpected",
        typed_input=lambda: "salir",
    )

    assert result.stopped_reason == "explicit_close"
    assert wake.calls == 0
    assert voice.close_calls == 1


def test_keyboard_interrupt_releases_resources() -> None:
    wake = FakeWakeWordEngine(interrupt=True)
    voice = FakeVoiceConversation([])
    assistant = make_assistant(wake, voice)

    result = assistant.run(process_text=lambda _text: "unexpected")

    assert result.stopped_reason == "keyboard_interrupt"
    assert voice.close_calls == 1
    assert assistant.state == PermanentAssistantState.STOPPING


def test_too_many_recoverable_errors_closes_with_clear_message() -> None:
    wake = FakeWakeWordEngine(
        [
            wake_result(detected=False, warnings=("timeout",)),
            wake_result(detected=False, warnings=("timeout",)),
        ]
    )
    voice = FakeVoiceConversation([])
    assistant = make_assistant(wake, voice, max_consecutive_errors=2)
    messages: list[str] = []

    result = assistant.run(
        process_text=lambda _text: "unexpected",
        status_sink=messages.append,
    )

    assert result.stopped_reason == "too_many_errors"
    assert "Demasiados errores consecutivos. Cerrando asistente permanente." in messages
