from __future__ import annotations

from types import SimpleNamespace

import pytest

from core.orchestrator import AtlasOrchestrator
from use_cases.speech_engine import (
    MicrophoneInfo,
    SpeechCaptureSettings,
    SpeechEngineUseCase,
    SpeechTranscriptionResult,
    SoundDeviceAudioCapture,
)
from use_cases.wake_word_engine import (
    WakeWordDetectionResult,
    WakeWordEngine,
    WakeWordInteractionUseCase,
)


def speech_result(
    text: str,
    completed: bool = True,
    cancelled: bool = False,
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
        no_speech_detected=not completed and not cancelled,
        warnings=warnings,
        summary="fake",
    )


class FakeSpeechEngine:
    def __init__(self, results: list[SpeechTranscriptionResult]) -> None:
        self._results = list(results)
        self.calls = 0

    def transcribe_once(self) -> SpeechTranscriptionResult:
        self.calls += 1

        if not self._results:
            return speech_result("", completed=False, warnings=("sin audio",))

        return self._results.pop(0)


class ConfigAwareFakeSpeechEngine(FakeSpeechEngine):
    def __init__(self, results: list[SpeechTranscriptionResult]) -> None:
        super().__init__(results)
        self.capture_settings: list[SpeechCaptureSettings | None] = []

    def transcribe_once(
        self,
        capture_settings: SpeechCaptureSettings | None = None,
    ) -> SpeechTranscriptionResult:
        self.capture_settings.append(capture_settings)
        return super().transcribe_once()


class FakeClock:
    def __init__(self, values: list[float]) -> None:
        self._values = list(values)
        self._last = values[-1] if values else 0.0

    def __call__(self) -> float:
        if self._values:
            self._last = self._values.pop(0)

        return self._last


def test_waits_until_wake_word_then_transcribes_phrase() -> None:
    speech = FakeSpeechEngine(
        [
            speech_result("ruido de fondo"),
            speech_result("Atlas"),
            speech_result("abre el proyecto"),
        ]
    )
    engine = WakeWordEngine(speech, wake_word="Atlas", timeout_seconds=10.0)

    result = engine.wait_for_wake_word()

    assert result.detected is True
    assert result.attempts == 2
    assert result.phrase is not None
    assert result.phrase.text == "abre el proyecto"
    assert speech.calls == 3


def test_wake_word_is_configurable_and_accent_insensitive() -> None:
    speech = FakeSpeechEngine(
        [
            speech_result("oye atlassian"),
            speech_result("hola Átlas"),
            speech_result("frase final"),
        ]
    )
    engine = WakeWordEngine(speech, wake_word="Atlas", timeout_seconds=10.0)

    result = engine.wait_for_wake_word()

    assert result.detected is True
    assert result.attempts == 2
    assert result.phrase is not None
    assert result.phrase.text == "frase final"


@pytest.mark.parametrize(
    "text",
    [
        "Atlas",
        "Atlas.",
        "ATLÁS",
        "Atlas abre Visual Studio Code",
    ],
)
def test_detects_safe_wake_word_variants(text: str) -> None:
    speech = FakeSpeechEngine([speech_result(text)])
    engine = WakeWordEngine(
        speech,
        wake_word="Atlas",
        timeout_seconds=10.0,
        capture_phrase_after_detection=False,
    )

    result = engine.wait_for_wake_word()

    assert result.detected is True
    assert result.attempts_log[-1].raw_text == text


@pytest.mark.parametrize(
    "text",
    [
        "atrás",
        "altas",
        "atlassian",
        "preatlas",
        "atlasico",
    ],
)
def test_rejects_unsafe_wake_word_matches(text: str) -> None:
    speech = FakeSpeechEngine([speech_result(text)])
    clock = FakeClock([0.0, 0.0, 2.0])
    engine = WakeWordEngine(
        speech,
        wake_word="Atlas",
        timeout_seconds=1.0,
        capture_phrase_after_detection=False,
        clock=clock,
    )

    result = engine.wait_for_wake_word()

    assert result.detected is False
    assert result.attempts_log[0].reason == "la transcripcion no contiene la wake word"


def test_timeout_is_configurable_and_does_not_capture_phrase() -> None:
    speech = FakeSpeechEngine(
        [
            speech_result("hola"),
            speech_result("sin palabra clave"),
        ]
    )
    clock = FakeClock([0.0, 0.0, 1.0, 2.0])
    sleeps: list[float] = []
    engine = WakeWordEngine(
        speech,
        wake_word="Atlas",
        timeout_seconds=1.5,
        poll_interval_seconds=0.25,
        clock=clock,
        sleeper=sleeps.append,
    )

    result = engine.wait_for_wake_word()

    assert result.detected is False
    assert result.phrase is None
    assert result.attempts == 2
    assert sleeps == [0.25, 0.25]
    assert "timeout de wake word alcanzado" in result.warnings


def test_uses_wake_capture_settings_and_retries_once_after_no_speech() -> None:
    speech = ConfigAwareFakeSpeechEngine(
        [
            speech_result("", completed=False, warnings=("silencio",)),
            speech_result("Atlas"),
        ]
    )
    statuses: list[str] = []
    engine = WakeWordEngine(
        speech,
        wake_word="Atlas",
        timeout_seconds=10.0,
        capture_phrase_after_detection=False,
    )

    result = engine.wait_for_wake_word(status_sink=statuses.append)

    assert result.detected is True
    assert speech.calls == 2
    assert len(speech.capture_settings) == 2
    assert speech.capture_settings[0] is not None
    assert speech.capture_settings[0].max_duration == 4.0
    assert speech.capture_settings[0].initial_silence_timeout == 7.0
    assert speech.capture_settings[0].trailing_silence == 1.2
    assert speech.capture_settings[0].speech_threshold == 0.008
    assert speech.capture_settings[0].minimum_audio_duration == 0.8
    assert "No te he oido. Repite 'Atlas'." in statuses


def test_does_not_retry_empty_more_than_once() -> None:
    speech = ConfigAwareFakeSpeechEngine(
        [
            speech_result("", completed=False, warnings=("silencio",)),
            speech_result("", completed=False, warnings=("silencio",)),
        ]
    )
    clock = FakeClock([0.0, 0.0, 0.5, 2.0])
    engine = WakeWordEngine(
        speech,
        wake_word="Atlas",
        timeout_seconds=1.0,
        capture_phrase_after_detection=False,
        clock=clock,
    )

    result = engine.wait_for_wake_word()

    assert result.detected is False
    assert speech.calls == 2


def test_continues_waiting_after_wrong_phrase_and_reports_diagnostic() -> None:
    speech = FakeSpeechEngine(
        [
            speech_result("altas"),
            speech_result("Atlas"),
        ]
    )
    statuses: list[str] = []
    engine = WakeWordEngine(
        speech,
        wake_word="Atlas",
        timeout_seconds=10.0,
        capture_phrase_after_detection=False,
    )

    result = engine.wait_for_wake_word(status_sink=statuses.append)

    assert result.detected is True
    assert speech.calls == 2
    assert "Texto bruto: altas" in statuses[0]
    assert "Motivo: la transcripcion no contiene la wake word" in statuses[0]


def test_temporary_wake_settings_preserve_selected_microphone(monkeypatch: pytest.MonkeyPatch) -> None:
    capture = SoundDeviceAudioCapture()
    provider = SimpleNamespace(
        name="fake-local",
        transcribe=lambda _samples, _sample_rate: SimpleNamespace(
            text="Atlas",
            language="es",
            processing_duration_seconds=0.1,
            provider="fake-local",
        ),
    )
    engine = SpeechEngineUseCase(capture, provider)
    fake_sd = SimpleNamespace(
        default=SimpleNamespace(device=(0, None)),
        query_devices=lambda: [
            {"name": "Default Mic", "max_input_channels": 1},
            {"name": "Selected Mic", "max_input_channels": 1},
        ],
    )
    monkeypatch.setattr(capture, "_sounddevice", lambda: fake_sd)
    monkeypatch.setattr(
        capture,
        "capture_phrase",
        lambda settings=None: SimpleNamespace(
            samples=__import__("numpy").ones(10, dtype=__import__("numpy").float32),
            sample_rate=16_000,
            duration_seconds=1.0,
            microphone_name=capture.selected_or_default_microphone().name,
            completed=True,
            cancelled=False,
            no_speech_detected=False,
            warnings=(),
        ),
    )

    selected = engine.select_microphone(1)
    result = engine.transcribe_once(
        capture_settings=SpeechCaptureSettings(speech_threshold=0.001),
    )

    assert selected.name == "Selected Mic"
    assert result.microphone_name == "Selected Mic"


def test_cancelled_detection_stops_waiting() -> None:
    speech = FakeSpeechEngine(
        [
            speech_result(
                "",
                completed=False,
                cancelled=True,
                warnings=("captura cancelada por el usuario",),
            )
        ]
    )
    engine = WakeWordEngine(speech, wake_word="Atlas", timeout_seconds=10.0)

    result = engine.wait_for_wake_word()

    assert result.cancelled is True
    assert result.detected is False
    assert result.attempts == 1


def test_invalid_configuration_is_rejected() -> None:
    speech = FakeSpeechEngine([])

    with pytest.raises(ValueError):
        WakeWordEngine(speech, wake_word=" ")

    with pytest.raises(ValueError):
        WakeWordEngine(speech, timeout_seconds=0)

    with pytest.raises(ValueError):
        WakeWordEngine(speech, poll_interval_seconds=-1)


def test_interaction_formats_transcription_without_execution() -> None:
    speech = FakeSpeechEngine(
        [
            speech_result("Atlas"),
            speech_result("abre visual studio code"),
        ]
    )
    interaction = WakeWordInteractionUseCase(
        WakeWordEngine(speech, wake_word="Atlas", timeout_seconds=10.0)
    )

    response = interaction.execute("atlas")

    assert "Wake word detectada: Atlas" in str(response)
    assert "Transcripcion:" in str(response)
    assert "abre visual studio code" in str(response)
    assert "La orden transcrita no se ejecuto." in str(response)


def test_interaction_reports_timeout_and_ignores_unknown_command() -> None:
    timeout_result = WakeWordDetectionResult(
        wake_word="Atlas",
        detected=False,
        attempts=0,
        elapsed_seconds=30.0,
        warnings=("timeout de wake word alcanzado",),
    )
    interaction = WakeWordInteractionUseCase(
        SimpleNamespace(wait_for_wake_word=lambda: timeout_result)
    )

    assert interaction.execute("hola") is None
    assert (
        interaction.execute("wake word")
        == "No se detecto la palabra de activacion Atlas antes del tiempo limite."
    )


def test_interaction_formats_failed_phrase_after_wake_word() -> None:
    result = WakeWordDetectionResult(
        wake_word="Atlas",
        detected=True,
        attempts=1,
        elapsed_seconds=1.0,
        phrase=speech_result("", completed=False, warnings=("modelo no disponible",)),
    )
    interaction = WakeWordInteractionUseCase(
        SimpleNamespace(wait_for_wake_word=lambda: result)
    )

    response = interaction.execute("modo atlas")

    assert "Wake word detectada: Atlas" in str(response)
    assert "modelo no disponible" in str(response)
    assert "La orden transcrita no se ejecuto." in str(response)


def test_orchestrator_wake_word_runs_before_router_and_agent(monkeypatch, capsys) -> None:
    class FailingRouter:
        def route(self, _plan):  # pragma: no cover - must not be called
            raise AssertionError("router should not receive wake word flow")

    response = "\n".join(
        [
            "Wake word detectada: Atlas",
            "",
            "Transcripcion:",
            "abre visual studio code",
            "",
            "La orden transcrita no se ejecuto.",
        ]
    )
    inputs = iter(["atlas", "salir"])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(inputs))
    orchestrator = AtlasOrchestrator(
        planner=SimpleNamespace(create_plan=lambda _prompt: object()),
        router=FailingRouter(),
        model_manager=SimpleNamespace(choose_model=lambda _agent: "unused"),
        memory=SimpleNamespace(add_user=lambda _prompt: None, add_assistant=lambda _response: None, history=list),
        registry=SimpleNamespace(get=lambda _name: None),
        write_file=SimpleNamespace(execute=lambda *_args: "unused"),
        wake_word_interaction=SimpleNamespace(execute=lambda prompt: response if prompt == "atlas" else None),
    )

    orchestrator.start()
    output = capsys.readouterr().out

    assert "Wake word detectada: Atlas" in output
    assert "La orden transcrita no se ejecuto." in output


def test_bootstrap_injects_wake_word_interaction() -> None:
    from bootstrap.bootstrap import Bootstrap

    orchestrator = Bootstrap.build()

    assert orchestrator._wake_word_interaction is not None
