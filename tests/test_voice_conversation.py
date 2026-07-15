from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace

import numpy as np
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
    average_log_probability: float | None = None,
    no_speech_probability: float | None = None,
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
        average_log_probability=average_log_probability,
        no_speech_probability=no_speech_probability,
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
        self.capture_settings_seen = []

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

    def transcribe_once(self, capture_settings=None) -> SpeechTranscriptionResult:
        self.transcribe_calls += 1
        self.capture_settings_seen.append(capture_settings)

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


class FakeSpeechOutputEngine:
    def __init__(
        self,
        fail: bool = False,
        fail_once: bool = False,
        events: list[str] | None = None,
    ) -> None:
        self.calls: list[str] = []
        self.closed = False
        self.fail = fail
        self.fail_once = fail_once
        self.warm_up_calls = 0
        self.events = events

    def speak(self, text: str) -> None:
        if self.events is not None:
            self.events.append(f"tts start:{text}")

        self.calls.append(text)

        if self.fail or self.fail_once:
            self.fail_once = False
            raise RuntimeError("motor TTS roto")

        if self.events is not None:
            self.events.append(f"tts end:{text}")

    def close(self) -> None:
        self.closed = True

    def warm_up(self) -> None:
        self.warm_up_calls += 1


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
    output: FakeSpeechOutputEngine | None = None,
    clock=None,
    max_turns: int = 5,
    max_consecutive_no_speech: int = 2,
    max_session_duration: float = 600.0,
    conversation_idle_timeout: float = 25.0,
    diagnostics_enabled: bool = True,
    now_provider=None,
) -> VoiceConversationUseCase:
    return VoiceConversationUseCase(
        speech_engine=speech,
        wake_word_engine=wake or FakeWakeWordEngine(),
        speech_output_engine=output,
        conversation_idle_timeout=conversation_idle_timeout,
        max_session_duration=max_session_duration,
        max_turns=max_turns,
        max_consecutive_no_speech=max_consecutive_no_speech,
        diagnostics_enabled=diagnostics_enabled,
        clock=clock or (lambda: 0.0),
        now_provider=now_provider,
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


def test_manual_voice_speaks_after_valid_transcription_without_wake_word() -> None:
    speech = FakeSpeechEngine([speech_result("hola atlas"), speech_result("salir")])
    wake = FakeWakeWordEngine()
    output = FakeSpeechOutputEngine()
    processed: list[str] = []

    result = make_use_case(speech, wake=wake, output=output).execute_manual(
        process_text=lambda text: processed.append(text) or "respuesta hablada",
    )

    assert result.session.successful_turns == 1
    assert wake.calls == 0
    assert processed == [
        "hola atlas\n\nResponde en español, de forma natural y concisa."
    ]
    assert output.calls == ["respuesta hablada"]
    assert output.closed is True
    assert output.warm_up_calls == 1
    assert "respuesta hablada" in "\n".join(result.messages)


def test_manual_voice_uses_shorter_turn_capture_settings() -> None:
    speech = FakeSpeechEngine([speech_result("pregunta"), speech_result("salir")])

    result = make_use_case(speech).execute_manual(
        process_text=lambda _text: "respuesta",
    )

    assert result.session.successful_turns == 1
    assert speech.capture_settings_seen
    settings = speech.capture_settings_seen[0]
    assert settings.max_duration == 10.0
    assert settings.initial_silence_timeout == 5.0
    assert settings.trailing_silence == 1.0
    assert settings.minimum_audio_duration == 0.3


def test_manual_voice_applies_calibrated_noise_threshold() -> None:
    class CalibratingSpeech(FakeSpeechEngine):
        def calibrate_noise_threshold(self, capture_settings=None, duration_seconds=0.5):
            assert duration_seconds == 0.5
            assert capture_settings.initial_silence_timeout == 5.0
            return 0.011

    speech = CalibratingSpeech([speech_result("pregunta"), speech_result("salir")])

    result = make_use_case(speech).execute_manual(
        process_text=lambda _text: "respuesta",
    )

    assert result.session.successful_turns == 1
    assert speech.capture_settings_seen[0].speech_threshold == 0.011


def test_manual_voice_rejects_incompatible_wdm_ks_before_session_loop() -> None:
    class WdmKsSpeech(FakeSpeechEngine):
        def validate_active_microphone(self, capture_settings=None):
            raise RuntimeError(
                "Dispositivo incompatible para captura bloqueante: "
                "11 - Varios micrófonos (WDM-KS). Prueba con 1 - Microphone Array (MME)."
            )

    speech = WdmKsSpeech([speech_result("no debe escucharse")])
    output: list[str] = []

    result = make_use_case(
        speech,
        diagnostics_enabled=False,
    ).execute_manual(
        process_text=lambda _text: "unexpected",
        status_sink=output.append,
    )

    assert result.session.ended_reason == "critical_error"
    assert speech.transcribe_calls == 0
    assert "Esperando voz..." not in output
    assert len([message for message in result.messages if "WDM-KS" in message]) == 1


def test_open_error_9999_is_reported_once_and_does_not_loop() -> None:
    class BrokenSpeech(FakeSpeechEngine):
        def validate_active_microphone(self, capture_settings=None):
            raise RuntimeError(
                "Fallo al abrir stream del microfono 11 - Varios micrófonos "
                "(WDM-KS): Error opening InputStream: PaErrorCode -9999 "
                "Blocking API not supported yet."
            )

    speech = BrokenSpeech([speech_result("no debe escucharse")])

    result = make_use_case(speech).execute_manual(
        process_text=lambda _text: "unexpected",
    )

    messages = "\n".join(result.messages)
    assert result.session.ended_reason == "critical_error"
    assert result.session.consecutive_no_speech == 0
    assert speech.transcribe_calls == 0
    assert messages.count("PaErrorCode -9999") == 1


def test_device_failure_does_not_count_as_timeout() -> None:
    class BrokenSpeech(FakeSpeechEngine):
        def validate_active_microphone(self, capture_settings=None):
            raise RuntimeError("Fallo al abrir stream del microfono")

    speech = BrokenSpeech([speech_result("", completed=False, no_speech=True)])

    result = make_use_case(
        speech,
        max_consecutive_no_speech=1,
    ).execute_manual(process_text=lambda _text: "unexpected")

    assert result.session.ended_reason == "critical_error"
    assert result.session.failed_turns == 0
    assert result.session.consecutive_no_speech == 0


def test_calibration_runs_once_and_second_phrase_reuses_same_threshold() -> None:
    class CalibratingSpeech(FakeSpeechEngine):
        def __init__(self, results):
            super().__init__(results)
            self.calibration_calls = 0

        def calibrate_noise_threshold(self, capture_settings=None, duration_seconds=0.5):
            self.calibration_calls += 1
            return 0.011

    speech = CalibratingSpeech(
        [speech_result("primera"), speech_result("segunda"), speech_result("salir")]
    )

    result = make_use_case(speech).execute_manual(
        process_text=lambda text: f"respuesta {text.splitlines()[0]}",
    )

    assert result.session.successful_turns == 2
    assert speech.calibration_calls == 1
    assert speech.capture_settings_seen[0].speech_threshold == 0.011
    assert speech.capture_settings_seen[1].speech_threshold == 0.011


def test_rejects_high_no_speech_probability_transcription() -> None:
    speech = FakeSpeechEngine(
        [
            speech_result(
                "nice",
                average_log_probability=-1.5,
                no_speech_probability=0.9,
            ),
            speech_result("salir"),
        ]
    )
    processed: list[str] = []

    result = make_use_case(speech, max_consecutive_no_speech=2).execute_manual(
        process_text=lambda text: processed.append(text) or "unexpected",
    )

    assert result.session.failed_turns == 1
    assert processed == []
    assert "No entendí la frase. Inténtalo de nuevo." in result.messages


def test_trims_residual_punctuation_before_processing() -> None:
    speech = FakeSpeechEngine([speech_result("  ¿pregunta valida?  "), speech_result("salir")])
    prompts: list[str] = []

    result = make_use_case(speech).execute_manual(
        process_text=lambda text: prompts.append(text) or "respuesta",
    )

    assert result.session.successful_turns == 1
    assert result.session.transcript_history == ["pregunta valida"]
    assert prompts == [
        "pregunta valida\n\nResponde en español, de forma natural y concisa."
    ]


def test_rejects_low_value_noise_transcription() -> None:
    speech = FakeSpeechEngine(
        [
            speech_result(
                "thank you",
                average_log_probability=-0.9,
                no_speech_probability=0.4,
            ),
            speech_result("cancelar"),
        ]
    )
    output = FakeSpeechOutputEngine()

    result = make_use_case(speech, output=output).execute_manual(
        process_text=lambda _text: "unexpected",
    )

    assert result.session.successful_turns == 0
    assert result.session.failed_turns == 1
    assert output.calls == []


def test_spanish_voice_question_adds_spanish_instruction() -> None:
    speech = FakeSpeechEngine([speech_result("cuentame algo"), speech_result("salir")])
    prompts: list[str] = []

    result = make_use_case(speech).execute_manual(
        process_text=lambda text: prompts.append(text) or "respuesta en espanol",
    )

    assert result.session.successful_turns == 1
    assert "Responde en español, de forma natural y concisa." in prompts[0]


def test_explicit_english_voice_question_does_not_force_spanish() -> None:
    speech = FakeSpeechEngine(
        [speech_result("answer in English, say hello"), speech_result("salir")]
    )
    prompts: list[str] = []

    result = make_use_case(speech).execute_manual(
        process_text=lambda text: prompts.append(text) or "Hello.",
    )

    assert result.session.response_history == ["Hello."]
    assert prompts == ["answer in English, say hello"]


def test_current_time_uses_mocked_clock_and_does_not_call_model() -> None:
    speech = FakeSpeechEngine([speech_result("que hora es"), speech_result("salir")])
    output = FakeSpeechOutputEngine()
    model_calls = 0

    def process(_text: str) -> str:
        nonlocal model_calls
        model_calls += 1
        return "modelo no debe responder"

    result = make_use_case(
        speech,
        output=output,
        now_provider=lambda: datetime(2026, 7, 15, 18, 12).astimezone(),
    ).execute_manual(process_text=process)

    assert model_calls == 0
    assert result.session.response_history == ["Son las seis y doce de la tarde."]
    assert output.calls == ["Son las seis y doce de la tarde."]


def test_mistranscribed_short_time_question_uses_mocked_clock_and_does_not_call_model() -> None:
    speech = FakeSpeechEngine(
        [speech_result("que ahora es ahora mismo atras"), speech_result("salir")]
    )
    model_calls = 0

    def process(_text: str) -> str:
        nonlocal model_calls
        model_calls += 1
        return "modelo no debe responder"

    result = make_use_case(
        speech,
        now_provider=lambda: datetime(2026, 7, 15, 18, 12).astimezone(),
    ).execute_manual(process_text=process)

    assert model_calls == 0
    assert result.session.response_history == ["Son las seis y doce de la tarde."]


def test_current_date_uses_mocked_clock_and_does_not_call_model() -> None:
    speech = FakeSpeechEngine([speech_result("cual es la fecha"), speech_result("salir")])
    model_calls = 0

    def process(_text: str) -> str:
        nonlocal model_calls
        model_calls += 1
        return "modelo no debe responder"

    result = make_use_case(
        speech,
        now_provider=lambda: datetime(2026, 7, 15, 18, 12).astimezone(),
    ).execute_manual(process_text=process)

    assert model_calls == 0
    assert result.session.response_history == [
        "Hoy es miércoles, 15 de julio de 2026."
    ]


def test_current_date_and_time_uses_mocked_clock_and_does_not_call_model() -> None:
    speech = FakeSpeechEngine(
        [speech_result("fecha y hora actual"), speech_result("salir")]
    )
    model_calls = 0

    def process(_text: str) -> str:
        nonlocal model_calls
        model_calls += 1
        return "modelo no debe responder"

    result = make_use_case(
        speech,
        now_provider=lambda: datetime(2026, 7, 15, 18, 12).astimezone(),
    ).execute_manual(process_text=process)

    assert model_calls == 0
    assert result.session.response_history == [
        "Son las seis y doce de la tarde del miércoles, 15 de julio de 2026."
    ]


def test_three_consecutive_timeouts_end_manual_session() -> None:
    speech = FakeSpeechEngine(
        [
            speech_result("", completed=False, no_speech=True),
            speech_result("", completed=False, no_speech=True),
            speech_result("", completed=False, no_speech=True),
        ]
    )

    result = make_use_case(
        speech,
        max_consecutive_no_speech=3,
    ).execute_manual(process_text=lambda _text: "unexpected")

    assert result.session.ended_reason == "idle_timeout"
    assert speech.transcribe_calls == 3


def test_single_timeout_does_not_end_manual_session() -> None:
    speech = FakeSpeechEngine(
        [
            speech_result("", completed=False, no_speech=True),
            speech_result("pregunta valida"),
            speech_result("salir"),
        ]
    )

    result = make_use_case(
        speech,
        max_consecutive_no_speech=3,
    ).execute_manual(process_text=lambda _text: "respuesta")

    assert result.session.ended_reason == "explicit_close"
    assert result.session.successful_turns == 1


def test_slow_isolated_timeouts_do_not_trigger_absolute_idle_before_counter() -> None:
    speech = FakeSpeechEngine(
        [
            speech_result("", completed=False, no_speech=True),
            speech_result("", completed=False, no_speech=True),
            speech_result("salir"),
        ]
    )

    result = make_use_case(
        speech,
        clock=FakeClock([0.0, 0.0, 0.0, 30.0, 31.0, 61.0, 62.0]),
        conversation_idle_timeout=25.0,
        max_consecutive_no_speech=3,
    ).execute_manual(process_text=lambda _text: "unexpected")

    assert result.session.ended_reason == "explicit_close"
    assert speech.transcribe_calls == 3


def test_valid_question_resets_timeout_counter() -> None:
    speech = FakeSpeechEngine(
        [
            speech_result("", completed=False, no_speech=True),
            speech_result("pregunta valida"),
            speech_result("", completed=False, no_speech=True),
            speech_result("", completed=False, no_speech=True),
            speech_result("salir"),
        ]
    )

    result = make_use_case(
        speech,
        max_consecutive_no_speech=3,
    ).execute_manual(process_text=lambda _text: "respuesta")

    assert result.session.ended_reason == "explicit_close"
    assert result.session.successful_turns == 1
    assert result.session.failed_turns == 3


def test_clean_console_when_diagnostics_disabled() -> None:
    speech = FakeSpeechEngine([speech_result("que hora es"), speech_result("salir")])
    output: list[str] = []

    make_use_case(
        speech,
        diagnostics_enabled=False,
        now_provider=lambda: datetime(2026, 7, 15, 18, 12).astimezone(),
    ).execute_manual(
        process_text=lambda _text: "modelo no debe responder",
        status_sink=output.append,
    )

    assert output[:3] == [
        "Esperando voz...",
        "Tú: que hora es",
        "Atlas: Son las seis y doce de la tarde.",
    ]
    assert "Transcripcion:" not in output
    assert "Respuesta:" not in output
    assert "Estado: inicializando." not in output


def test_voice_resources_are_loaded_once_for_session() -> None:
    speech = FakeSpeechEngine(
        [speech_result("primera"), speech_result("segunda"), speech_result("salir")]
    )
    output = FakeSpeechOutputEngine()

    result = make_use_case(speech, output=output).execute_manual(
        process_text=lambda text: f"respuesta {text}",
    )

    assert result.session.successful_turns == 2
    assert speech.warm_up_calls == 1
    assert output.warm_up_calls == 1


def test_two_consecutive_responses_are_both_spoken_once() -> None:
    speech = FakeSpeechEngine(
        [speech_result("primera"), speech_result("segunda"), speech_result("salir")]
    )
    output = FakeSpeechOutputEngine()

    result = make_use_case(speech, output=output).execute_manual(
        process_text=lambda text: f"respuesta {text.splitlines()[0]}",
    )

    assert result.session.successful_turns == 2
    assert output.calls == ["respuesta primera", "respuesta segunda"]


def test_next_turn_starts_after_tts_finishes() -> None:
    events: list[str] = []

    class EventSpeech(FakeSpeechEngine):
        def transcribe_once(self, capture_settings=None) -> SpeechTranscriptionResult:
            events.append("listen")
            return super().transcribe_once(capture_settings)

    speech = EventSpeech(
        [speech_result("primera"), speech_result("segunda"), speech_result("salir")]
    )
    output = FakeSpeechOutputEngine(events=events)

    make_use_case(speech, output=output).execute_manual(
        process_text=lambda text: f"respuesta {text.splitlines()[0]}",
    )

    assert events == [
        "listen",
        "tts start:respuesta primera",
        "tts end:respuesta primera",
        "listen",
        "tts start:respuesta segunda",
        "tts end:respuesta segunda",
        "listen",
    ]


def test_tts_failure_allows_next_turn_to_speak() -> None:
    speech = FakeSpeechEngine(
        [speech_result("primera"), speech_result("segunda"), speech_result("salir")]
    )
    output = FakeSpeechOutputEngine(fail_once=True)

    result = make_use_case(speech, output=output).execute_manual(
        process_text=lambda text: f"respuesta {text.splitlines()[0]}",
    )

    assert result.session.successful_turns == 2
    assert output.calls == ["respuesta primera", "respuesta segunda"]
    assert "TTS falló:" in "\n".join(result.messages)
    assert result.session.ended_reason == "explicit_close"


def test_tts_diagnostics_are_hidden_when_diagnostics_disabled() -> None:
    speech = FakeSpeechEngine([speech_result("pregunta"), speech_result("salir")])
    output = FakeSpeechOutputEngine()
    console: list[str] = []

    make_use_case(
        speech,
        output=output,
        diagnostics_enabled=False,
    ).execute_manual(
        process_text=lambda _text: "respuesta",
        status_sink=console.append,
    )

    assert "TTS iniciado" not in console
    assert "TTS finalizado" not in console
    assert output.calls == ["respuesta"]


def test_tts_is_called_once_per_non_empty_response() -> None:
    speech = FakeSpeechEngine([speech_result("pregunta"), speech_result("terminar")])
    output = FakeSpeechOutputEngine()

    result = make_use_case(speech, output=output).execute_manual(
        process_text=lambda _text: "respuesta unica",
    )

    assert result.session.successful_turns == 1
    assert output.calls == ["respuesta unica"]


def test_tts_is_not_called_for_empty_response() -> None:
    speech = FakeSpeechEngine([speech_result("pregunta"), speech_result("cancelar")])
    output = FakeSpeechOutputEngine()

    result = make_use_case(speech, output=output).execute_manual(
        process_text=lambda _text: "",
    )

    assert result.session.successful_turns == 1
    assert result.session.response_history == ["Respuesta vacia."]
    assert output.calls == []


@pytest.mark.parametrize("command", ["salir", "terminar", "cancelar"])
def test_manual_voice_exits_by_spoken_commands(command: str) -> None:
    speech = FakeSpeechEngine([speech_result(command)])
    output = FakeSpeechOutputEngine()

    result = make_use_case(speech, output=output).execute_manual(
        process_text=lambda _text: "unexpected",
    )

    assert result.session.ended_reason == "explicit_close"
    assert result.session.total_turns == 0
    assert output.calls == []


@pytest.mark.parametrize("command", ["salir", "terminar", "cancelar"])
def test_manual_voice_exits_by_typed_commands(command: str) -> None:
    speech = FakeSpeechEngine([speech_result("no debe escucharse")])
    output = FakeSpeechOutputEngine()

    result = make_use_case(speech, output=output).execute_manual(
        process_text=lambda _text: "unexpected",
        typed_input=lambda: command,
    )

    assert result.session.ended_reason == "explicit_close"
    assert speech.transcribe_calls == 0
    assert output.calls == []


def test_manual_voice_timeout_does_not_call_tts() -> None:
    speech = FakeSpeechEngine(
        [speech_result("", completed=False, no_speech=True, warnings=("silencio",))]
    )
    output = FakeSpeechOutputEngine()

    result = make_use_case(
        speech,
        output=output,
        max_consecutive_no_speech=1,
    ).execute_manual(process_text=lambda _text: "unexpected")

    assert result.session.ended_reason == "idle_timeout"
    assert output.calls == []


def test_tts_error_keeps_textual_flow_alive() -> None:
    speech = FakeSpeechEngine([speech_result("primera"), speech_result("terminar")])
    output = FakeSpeechOutputEngine(fail=True)

    result = make_use_case(speech, output=output).execute_manual(
        process_text=lambda text: f"respuesta {text}",
    )

    messages = "\n".join(result.messages)
    assert result.session.successful_turns == 1
    assert "respuesta primera" in messages
    assert "TTS falló:" in messages
    assert result.session.ended_reason == "explicit_close"


def test_timeout_counter_starts_after_tts_finishes() -> None:
    speech = FakeSpeechEngine(
        [
            speech_result("primera"),
            speech_result("", completed=False, no_speech=True),
            speech_result("salir"),
        ]
    )
    output = FakeSpeechOutputEngine()

    result = make_use_case(
        speech,
        output=output,
        clock=FakeClock([0.0, 0.0, 0.0, 100.0, 101.0, 102.0, 103.0]),
        conversation_idle_timeout=25.0,
        max_consecutive_no_speech=3,
    ).execute_manual(process_text=lambda _text: "respuesta")

    assert result.session.ended_reason == "explicit_close"
    assert output.calls == ["respuesta"]


def test_voice_response_removes_internal_json_block_before_print_and_tts() -> None:
    speech = FakeSpeechEngine([speech_result("muestra formato"), speech_result("salir")])
    output = FakeSpeechOutputEngine()
    raw_response = "\n".join(
        [
            "Es hora de la respuesta del sistema; aqui tienes el formato actual:",
            "",
            "```json",
            '{ "time": "14:59" }',
            "```",
        ]
    )

    result = make_use_case(speech, output=output).execute_manual(
        process_text=lambda _text: raw_response,
    )

    messages = "\n".join(result.messages)
    assert "respuesta del sistema" not in messages
    assert "```json" not in messages
    assert result.session.response_history == ["Hora: 14:59."]
    assert output.calls == ["Hora: 14:59."]


def test_manual_voice_processes_second_question_in_same_session() -> None:
    speech = FakeSpeechEngine(
        [
            speech_result("primera pregunta"),
            speech_result("segunda pregunta"),
            speech_result("salir"),
        ]
    )
    output = FakeSpeechOutputEngine()

    result = make_use_case(speech, output=output).execute_manual(
        process_text=lambda text: f"respuesta {text.splitlines()[0]}",
    )

    assert result.session.transcript_history == [
        "primera pregunta",
        "segunda pregunta",
    ]
    assert output.calls == [
        "respuesta primera pregunta",
        "respuesta segunda pregunta",
    ]


def test_real_wake_word_engine_can_detect_without_capturing_phrase() -> None:
    class Provider:
        sample_rate = 16_000
        frame_length = 2

        def __init__(self) -> None:
            self.closed = False

        def initialize(self) -> None:
            pass

        def process_frame(self, _frame) -> bool:
            return True

        def close(self) -> None:
            self.closed = True

    class Speech:
        def __init__(self) -> None:
            self.transcribe_calls = 0

        def iter_pcm_frames(self, sample_rate: int, frame_length: int):
            assert sample_rate == 16_000
            assert frame_length == 2
            yield np.ones(2, dtype=np.int16)

        def transcribe_once(self):
            self.transcribe_calls += 1
            return speech_result("primer turno")

    speech = Speech()
    wake = WakeWordEngine(
        speech,
        provider=Provider(),
        wake_word="Atlas",
        timeout_seconds=5.0,
        capture_phrase_after_detection=False,
    )

    result = wake.wait_for_wake_word()

    assert result.detected is True
    assert result.phrase is None
    assert speech.transcribe_calls == 0


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
