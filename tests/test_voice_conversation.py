from __future__ import annotations

from types import SimpleNamespace

import pytest

from agents.registry import AgentRegistry
from core.orchestrator import AtlasOrchestrator
from use_cases.speech_engine import SpeechTranscriptionResult
from use_cases.voice_conversation import VoiceConversationUseCase
from use_cases.wake_word_engine import WakeWordDetectionResult, WakeWordEngine


def speech_result(
    text: str,
    completed: bool = True,
    cancelled: bool = False,
    no_speech: bool = False,
    warnings: tuple[str, ...] = (),
) -> SpeechTranscriptionResult:
    return SpeechTranscriptionResult(
        text=text,
        language="es" if completed else None,
        audio_duration_seconds=1.0 if completed else 0.0,
        processing_duration_seconds=0.2 if completed else 0.0,
        provider="fake-local",
        microphone_name="Fake Mic",
        completed=completed,
        cancelled=cancelled,
        no_speech_detected=no_speech,
        warnings=warnings,
        summary="fake",
    )


class FakeSpeechEngine:
    def __init__(self, results: list[SpeechTranscriptionResult]) -> None:
        self._results = list(results)
        self.transcribe_calls = 0
        self.warm_up_calls = 0
        self.default_microphone_calls = 0
        self.active_microphone_calls = 0
        self.prepare_stream_calls = 0
        self.prepared_before_wake = False
        self.active_microphone_index = 1
        self.active_microphone_name = "Microphone Array (Realtek(R) Audio)"
        self.audio_saved = False
        self.tts_calls = 0
        self.sound_played = False
        self.cloud_calls = 0
        self.stream_closed = True

    def warm_up(self) -> None:
        self.warm_up_calls += 1

    def default_microphone(self):
        self.default_microphone_calls += 1
        return SimpleNamespace(index=0, name="Asignador de sonido Microsoft - Input")

    def active_microphone(self):
        self.active_microphone_calls += 1
        return SimpleNamespace(
            index=self.active_microphone_index,
            name=self.active_microphone_name,
        )

    def prepare_stream(self, _settings=None):
        self.prepare_stream_calls += 1
        self.prepared_before_wake = True
        return self.active_microphone()

    def transcribe_once(self) -> SpeechTranscriptionResult:
        self.transcribe_calls += 1

        if not self._results:
            return speech_result(
                "",
                completed=False,
                no_speech=True,
                warnings=("sin audio",),
            )

        return self._results.pop(0)


class FakeWakeWordEngine:
    def __init__(
        self,
        detected: bool = True,
        cancelled: bool = False,
        speech: FakeSpeechEngine | None = None,
    ) -> None:
        self.calls = 0
        self.detected = detected
        self.cancelled = cancelled
        self.speech = speech
        self.called_after_prepare = False

    def wait_for_wake_word(self, status_sink=None) -> WakeWordDetectionResult:
        self.calls += 1
        self.called_after_prepare = (
            self.speech.prepared_before_wake if self.speech is not None else True
        )
        return WakeWordDetectionResult(
            wake_word="Atlas",
            detected=self.detected,
            attempts=1,
            elapsed_seconds=0.5,
            cancelled=self.cancelled,
        )


class FakeClock:
    def __init__(self, values: list[float]) -> None:
        self._values = list(values)
        self._last = values[-1] if values else 0.0

    def __call__(self) -> float:
        if self._values:
            self._last = self._values.pop(0)

        return self._last


def make_use_case(
    speech: FakeSpeechEngine,
    wake: FakeWakeWordEngine | None = None,
    clock=None,
    max_turns: int = 5,
    max_consecutive_no_speech: int = 2,
    max_session_duration: float = 600.0,
    conversation_idle_timeout: float = 25.0,
) -> VoiceConversationUseCase:
    return VoiceConversationUseCase(
        speech_engine=speech,
        wake_word_engine=wake or FakeWakeWordEngine(),
        conversation_idle_timeout=conversation_idle_timeout,
        max_session_duration=max_session_duration,
        max_turns=max_turns,
        max_consecutive_no_speech=max_consecutive_no_speech,
        clock=clock or (lambda: 0.0),
        session_id_factory=lambda: "session-1",
    )


def test_activates_by_explicit_command_waits_wake_and_starts_session() -> None:
    speech = FakeSpeechEngine([speech_result("adios Atlas")])
    wake = FakeWakeWordEngine(speech=speech)
    use_case = make_use_case(speech, wake)
    statuses: list[str] = []

    result = use_case.execute(
        "activa conversacion por voz",
        process_text=lambda text: f"respuesta a {text}",
        status_sink=statuses.append,
    )

    assert result is not None
    assert wake.calls == 1
    assert result.session.session_id == "session-1"
    assert result.session.ended_reason == "explicit_close"
    assert statuses[:8] == [
        "Estado: inicializando.",
        "Inicializando microfono y modelo...",
        "Microfono activo:",
        "1 - Microphone Array (Realtek(R) Audio)",
        "Esperando la palabra de activacion...",
        "Estado: listo para wake word.",
        'Di "Atlas" ahora...',
        "Estado: wake word detectada.",
    ]
    assert "Conversacion iniciada." in statuses


def test_shared_selected_microphone_is_used_by_voice_conversation_and_wake() -> None:
    speech = FakeSpeechEngine([speech_result("adios Atlas")])
    wake = FakeWakeWordEngine(speech=speech)
    statuses: list[str] = []

    result = make_use_case(speech, wake).execute(
        "activa conversacion por voz",
        process_text=lambda text: text,
        status_sink=statuses.append,
    )

    assert result is not None
    assert speech.default_microphone_calls == 0
    assert speech.active_microphone_calls >= 2
    assert speech.prepare_stream_calls == 1
    assert wake.called_after_prepare is True
    assert "1 - Microphone Array (Realtek(R) Audio)" in statuses
    assert "Asignador de sonido Microsoft - Input" not in statuses


def test_stream_is_prepared_before_asking_for_atlas() -> None:
    speech = FakeSpeechEngine([speech_result("adios Atlas")])
    wake = FakeWakeWordEngine(speech=speech)
    statuses: list[str] = []

    result = make_use_case(speech, wake).execute(
        "activa conversacion por voz",
        process_text=lambda text: text,
        status_sink=statuses.append,
    )

    assert result is not None
    assert wake.called_after_prepare is True
    assert statuses.index("Microfono activo:") < statuses.index('Di "Atlas" ahora...')
    assert statuses.index("Estado: listo para wake word.") < statuses.index('Di "Atlas" ahora...')


def test_wake_timeout_ends_voice_mode_without_chat_fallback() -> None:
    speech = FakeSpeechEngine([])
    wake = FakeWakeWordEngine(detected=False, speech=speech)

    result = make_use_case(speech, wake).execute(
        "activa conversacion por voz",
        process_text=lambda _text: (_ for _ in ()).throw(
            AssertionError("chat flow must not run")
        ),
    )

    assert result is not None
    assert result.session.ended_reason == "wake_word_timeout"
    assert 'No se detectó "Atlas". Modo de voz finalizado.' in result.messages


def test_second_activation_is_intercepted_again_after_timeout() -> None:
    speech = FakeSpeechEngine([])
    wake = FakeWakeWordEngine(detected=False, speech=speech)
    use_case = make_use_case(speech, wake)

    first = use_case.execute("activa conversacion por voz", process_text=lambda text: text)
    second = use_case.execute("activa conversacion por voz", process_text=lambda text: text)

    assert first is not None
    assert second is not None
    assert wake.calls == 2


def test_falls_back_to_default_microphone_when_no_explicit_selection() -> None:
    class DefaultOnlySpeech(FakeSpeechEngine):
        def active_microphone(self):
            self.active_microphone_calls += 1
            return self.default_microphone()

    speech = DefaultOnlySpeech([speech_result("adios Atlas")])
    statuses: list[str] = []

    result = make_use_case(speech, FakeWakeWordEngine(speech=speech)).execute(
        "activa conversacion por voz",
        process_text=lambda text: text,
        status_sink=statuses.append,
    )

    assert result is not None
    assert "0 - Asignador de sonido Microsoft - Input" in statuses


def test_statuses_are_separated_cleanly() -> None:
    speech = FakeSpeechEngine([speech_result("adios Atlas")])
    statuses: list[str] = []

    result = make_use_case(speech, FakeWakeWordEngine(speech=speech)).execute(
        "activa conversacion por voz",
        process_text=lambda text: text,
        status_sink=statuses.append,
    )

    assert result is not None
    assert "Estado: inicializando." in statuses
    assert "Estado: listo para wake word." in statuses
    assert "Estado: wake word detectada." in statuses
    assert "Estado: conversacion activa." in statuses
    assert "Estado: conversacion finalizada." in statuses


def test_ignores_unknown_command_and_does_not_activate_microphone() -> None:
    speech = FakeSpeechEngine([])
    use_case = make_use_case(speech)

    assert use_case.execute("hola", process_text=lambda text: text) is None
    assert speech.warm_up_calls == 0
    assert speech.transcribe_calls == 0


def test_warm_up_once_and_does_not_generate_transcription() -> None:
    speech = FakeSpeechEngine([speech_result("termina")])
    use_case = make_use_case(speech)

    result = use_case.execute("inicia modo voz", process_text=lambda text: text)

    assert result is not None
    assert speech.warm_up_calls == 1
    assert speech.transcribe_calls == 1
    assert result.session.total_turns == 0


def test_first_turn_retries_once_on_initial_no_speech() -> None:
    speech = FakeSpeechEngine(
        [
            speech_result("", completed=False, no_speech=True, warnings=("silencio",)),
            speech_result("que puedes hacer"),
            speech_result("termina"),
        ]
    )
    processed: list[str] = []
    statuses: list[str] = []
    use_case = make_use_case(speech)

    result = use_case.execute(
        "conversacion por voz",
        process_text=lambda text: processed.append(text) or "respuesta escrita",
        status_sink=statuses.append,
    )

    assert result is not None
    assert "No se detecto voz. Vuelvo a escuchar..." in statuses
    assert speech.transcribe_calls == 3
    assert processed == ["que puedes hacer"]
    assert result.session.successful_turns == 1


def test_first_turn_does_not_retry_more_than_once() -> None:
    speech = FakeSpeechEngine(
        [
            speech_result("", completed=False, no_speech=True, warnings=("silencio",)),
            speech_result("", completed=False, no_speech=True, warnings=("silencio",)),
        ]
    )
    use_case = make_use_case(speech, max_consecutive_no_speech=1)

    result = use_case.execute("start voice conversation", process_text=lambda text: text)

    assert result is not None
    assert speech.transcribe_calls == 2
    assert result.session.ended_reason == "idle_timeout"
    assert result.session.failed_turns == 1


def test_does_not_retry_critical_first_error() -> None:
    speech = FakeSpeechEngine(
        [
            speech_result(
                "",
                completed=False,
                warnings=("modelo no disponible",),
            )
        ]
    )
    use_case = make_use_case(speech)

    result = use_case.execute("activa conversacion por voz", process_text=lambda text: text)

    assert result is not None
    assert speech.transcribe_calls == 1
    assert result.session.ended_reason == "critical_error"


def test_processes_phrase_by_normal_flow_and_preserves_text_exactly() -> None:
    text = "  Que archivos dependen de Router?  "
    speech = FakeSpeechEngine([speech_result(text), speech_result("termina")])
    processed: list[str] = []
    use_case = make_use_case(speech)

    result = use_case.execute(
        "activa conversacion por voz",
        process_text=lambda phrase: processed.append(phrase) or "respuesta",
    )

    assert result is not None
    assert processed == ["Que archivos dependen de Router?"]
    assert result.session.transcript_history == ["Que archivos dependen de Router?"]
    assert result.session.response_history == ["respuesta"]


def test_shows_text_response_continues_second_turn_without_wake_word() -> None:
    speech = FakeSpeechEngine(
        [
            speech_result("primer turno"),
            speech_result("segundo turno"),
            speech_result("adios Atlas"),
        ]
    )
    wake = FakeWakeWordEngine()
    use_case = make_use_case(speech, wake)

    result = use_case.execute(
        "activa conversacion por voz",
        process_text=lambda phrase: f"respuesta {phrase}",
    )

    assert result is not None
    assert wake.calls == 1
    assert result.session.total_turns == 2
    assert result.session.successful_turns == 2
    assert result.session.transcript_history == ["primer turno", "segundo turno"]
    assert result.session.response_history == [
        "respuesta primer turno",
        "respuesta segundo turno",
    ]
    assert "respuesta segundo turno" in "\n".join(result.messages)


@pytest.mark.parametrize(
    "command",
    ["termina", "adios Atlas", "stop", "stop listening", "end conversation"],
)
def test_close_commands_are_not_sent_to_router(command: str) -> None:
    speech = FakeSpeechEngine([speech_result(command)])
    processed: list[str] = []
    use_case = make_use_case(speech)

    result = use_case.execute(
        "inicia modo voz",
        process_text=lambda phrase: processed.append(phrase) or "unexpected",
    )

    assert result is not None
    assert processed == []
    assert result.session.ended_reason == "explicit_close"
    assert result.session.summary == "Conversacion finalizada."


def test_ends_by_inactivity_max_duration_and_max_turns() -> None:
    idle = make_use_case(
        FakeSpeechEngine([speech_result("hola")]),
        clock=FakeClock([0.0, 0.0, 30.0]),
        conversation_idle_timeout=10.0,
    ).execute("activa conversacion por voz", process_text=lambda text: "respuesta")
    duration = make_use_case(
        FakeSpeechEngine([speech_result("hola")]),
        clock=FakeClock([0.0, 0.0, 601.0]),
        max_session_duration=600.0,
        conversation_idle_timeout=700.0,
    ).execute("activa conversacion por voz", process_text=lambda text: "respuesta")
    max_turns = make_use_case(
        FakeSpeechEngine([speech_result("uno"), speech_result("dos")]),
        max_turns=1,
    ).execute("activa conversacion por voz", process_text=lambda text: "respuesta")

    assert idle is not None
    assert duration is not None
    assert max_turns is not None
    assert idle.session.ended_reason == "idle_timeout"
    assert duration.session.ended_reason == "max_session_duration"
    assert max_turns.session.ended_reason == "max_turns"


def test_ctrl_c_cancels_session_and_stream_is_closed() -> None:
    speech = FakeSpeechEngine([])
    speech.transcribe_once = lambda: (_ for _ in ()).throw(KeyboardInterrupt())
    use_case = make_use_case(speech)

    result = use_case.execute("activa conversacion por voz", process_text=lambda text: text)

    assert result is not None
    assert result.session.ended_reason == "cancelled"
    assert speech.stream_closed is True


def test_transcription_cancelled_ends_session() -> None:
    speech = FakeSpeechEngine(
        [speech_result("", completed=False, cancelled=True, warnings=("cancelada",))]
    )
    use_case = make_use_case(speech)

    result = use_case.execute("activa conversacion por voz", process_text=lambda text: text)

    assert result is not None
    assert result.session.ended_reason == "cancelled"


def test_non_critical_turn_failure_allows_continuation() -> None:
    speech = FakeSpeechEngine(
        [
            speech_result("primer turno"),
            speech_result("segundo turno"),
            speech_result("termina"),
        ]
    )
    calls = 0

    def process(text: str) -> str:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("fallo temporal")
        return f"respuesta {text}"

    result = make_use_case(speech).execute(
        "activa conversacion por voz",
        process_text=process,
    )

    assert result is not None
    assert result.session.failed_turns == 1
    assert result.session.successful_turns == 1
    assert result.session.ended_reason == "explicit_close"


def test_critical_turn_failure_ends_session() -> None:
    speech = FakeSpeechEngine([speech_result("primer turno")])

    result = make_use_case(speech).execute(
        "activa conversacion por voz",
        process_text=lambda _text: (_ for _ in ()).throw(
            RuntimeError("permisos denegados")
        ),
    )

    assert result is not None
    assert result.session.ended_reason == "critical_error"


def test_privacy_no_audio_tts_sound_or_cloud() -> None:
    speech = FakeSpeechEngine([speech_result("termina")])

    result = make_use_case(speech).execute(
        "activa conversacion por voz",
        process_text=lambda text: text,
    )

    assert result is not None
    assert speech.audio_saved is False
    assert speech.tts_calls == 0
    assert speech.sound_played is False
    assert speech.cloud_calls == 0


def test_real_wake_word_engine_can_detect_without_capturing_phrase() -> None:
    speech = FakeSpeechEngine([speech_result("Atlas"), speech_result("primer turno")])
    wake = WakeWordEngine(
        speech,
        wake_word="Atlas",
        timeout_seconds=5.0,
        capture_phrase_after_detection=False,
    )

    result = wake.wait_for_wake_word()

    assert result.detected is True
    assert result.phrase is None
    assert speech.transcribe_calls == 1


def test_orchestrator_integrates_voice_mode_before_router(monkeypatch, capsys) -> None:
    class FailingRouter:
        def route(self, _plan):  # pragma: no cover - must not be called
            raise AssertionError("router should not run for activation command")

    inputs = iter(["activa conversacion por voz", "salir"])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(inputs))
    orchestrator = AtlasOrchestrator(
        planner=SimpleNamespace(create_plan=lambda _prompt: object()),
        router=FailingRouter(),
        model_manager=SimpleNamespace(choose_model=lambda _agent: "unused"),
        memory=SimpleNamespace(add_user=lambda _prompt: None, add_assistant=lambda _response: None, history=list),
        registry=AgentRegistry(),
        write_file=SimpleNamespace(execute=lambda *_args: "unused"),
        voice_conversation=SimpleNamespace(
            execute=lambda prompt, process_text, status_sink: (
                status_sink("Esperando la palabra de activacion...") or object()
            )
            if prompt == "activa conversacion por voz"
            else None
        ),
    )

    orchestrator.start()
    output = capsys.readouterr().out

    assert "Esperando la palabra de activacion..." in output


def test_orchestrator_activation_never_reaches_chat_general(monkeypatch) -> None:
    inputs = iter(["activa conversacion por voz", "salir"])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(inputs))
    routed: list[str] = []
    orchestrator = AtlasOrchestrator(
        planner=SimpleNamespace(create_plan=lambda prompt: routed.append(prompt) or object()),
        router=SimpleNamespace(route=lambda _plan: "chat"),
        model_manager=SimpleNamespace(choose_model=lambda _agent: "unused"),
        memory=SimpleNamespace(add_user=lambda _prompt: None, add_assistant=lambda _response: None, history=list),
        registry=AgentRegistry(),
        write_file=SimpleNamespace(execute=lambda *_args: "unused"),
        voice_conversation=SimpleNamespace(
            execute=lambda prompt, process_text, status_sink: object()
            if prompt == "activa conversacion por voz"
            else None
        ),
    )

    orchestrator.start()

    assert routed == []


def test_voice_activation_ignores_accidental_prompt_prefix() -> None:
    speech = FakeSpeechEngine([speech_result("adios Atlas")])
    wake = FakeWakeWordEngine(speech=speech)

    result = make_use_case(speech, wake).execute(
        "Tú: activa conversacion por voz",
        process_text=lambda text: text,
    )

    assert result is not None
    assert wake.calls == 1


def test_orchestrator_does_not_print_duplicate_prompt(monkeypatch, capsys) -> None:
    inputs = iter(["activa conversacion por voz", "salir"])
    monkeypatch.setattr("builtins.input", lambda prompt: next(inputs))
    orchestrator = AtlasOrchestrator(
        planner=SimpleNamespace(create_plan=lambda _prompt: object()),
        router=SimpleNamespace(route=lambda _plan: "chat"),
        model_manager=SimpleNamespace(choose_model=lambda _agent: "unused"),
        memory=SimpleNamespace(add_user=lambda _prompt: None, add_assistant=lambda _response: None, history=list),
        registry=AgentRegistry(),
        write_file=SimpleNamespace(execute=lambda *_args: "unused"),
        voice_conversation=SimpleNamespace(
            execute=lambda prompt, process_text, status_sink: object()
            if prompt == "activa conversacion por voz"
            else None
        ),
    )

    orchestrator.start()
    output = capsys.readouterr().out

    assert "Tú: Tú:" not in output


def test_bootstrap_injects_voice_conversation() -> None:
    from bootstrap.bootstrap import Bootstrap

    orchestrator = Bootstrap.build()

    assert orchestrator._voice_conversation is not None
