from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from agents.registry import AgentRegistry
from core.model_manager import ModelManager
from core.orchestrator import AtlasOrchestrator
from core.planner import Planner
from core.router import Router
from memory.conversation import ConversationMemory
from use_cases.speech_engine import (
    AudioCaptureResult,
    FasterWhisperSpeechToTextProvider,
    MicrophoneInfo,
    ProviderTranscriptionResult,
    SoundDeviceAudioCapture,
    SpeechEngineUseCase,
    SpeechInteractionUseCase,
)


class FakeCapture:
    def __init__(self) -> None:
        self.selected_index: int | None = None
        self.capture_calls = 0
        self.raise_error: Exception | None = None
        self.result = AudioCaptureResult(
            samples=np.ones(16_000, dtype=np.float32) * 0.1,
            sample_rate=16_000,
            duration_seconds=1.0,
            microphone_name="Fake Mic",
            completed=True,
        )
        self.microphones = [
            MicrophoneInfo(0, "Fake Mic", True, 1),
            MicrophoneInfo(2, "Second Mic", False, 1),
        ]

    def list_microphones(self) -> list[MicrophoneInfo]:
        return self.microphones

    def default_microphone(self) -> MicrophoneInfo:
        if not self.microphones:
            raise RuntimeError("No hay microfonos de entrada disponibles.")
        return next((mic for mic in self.microphones if mic.is_default), self.microphones[0])

    def select_microphone(self, index: int) -> MicrophoneInfo:
        for microphone in self.microphones:
            if microphone.index == index:
                self.selected_index = index
                return microphone
        raise ValueError(f"Microfono inexistente: {index}")

    def capture_phrase(self) -> AudioCaptureResult:
        self.capture_calls += 1
        if self.raise_error is not None:
            raise self.raise_error
        return self.result


class FakeProvider:
    name = "fake-local"

    def __init__(self) -> None:
        self.calls = 0
        self.loaded = False
        self.network_called = False
        self.raise_error: Exception | None = None
        self.text = "Atlas abre Visual Studio Code"

    def transcribe(
        self,
        samples: np.ndarray,
        sample_rate: int,
    ) -> ProviderTranscriptionResult:
        self.calls += 1
        self.loaded = True
        if self.raise_error is not None:
            raise self.raise_error
        return ProviderTranscriptionResult(
            text=self.text,
            language="es",
            processing_duration_seconds=0.25,
            provider=self.name,
        )


def test_lists_devices_default_and_selects_valid_index() -> None:
    capture = FakeCapture()
    engine = SpeechEngineUseCase(capture, FakeProvider())

    microphones = engine.list_microphones()
    default = engine.default_microphone()
    selected = engine.select_microphone(2)

    assert [mic.name for mic in microphones] == ["Fake Mic", "Second Mic"]
    assert default.index == 0
    assert selected.name == "Second Mic"
    assert capture.selected_index == 2


def test_rejects_invalid_index_and_empty_device_list() -> None:
    capture = FakeCapture()
    engine = SpeechEngineUseCase(capture, FakeProvider())

    with pytest.raises(ValueError):
        engine.select_microphone(99)

    capture.microphones = []

    with pytest.raises(RuntimeError):
        engine.default_microphone()


def test_sounddevice_device_listing_uses_input_devices(monkeypatch: pytest.MonkeyPatch) -> None:
    capture = SoundDeviceAudioCapture()
    fake_sd = SimpleNamespace(
        default=SimpleNamespace(device=(1, None)),
        query_devices=lambda: [
            {"name": "Output", "max_input_channels": 0},
            {"name": "Input", "max_input_channels": 2},
        ],
    )
    monkeypatch.setattr(capture, "_sounddevice", lambda: fake_sd)

    microphones = capture.list_microphones()

    assert len(microphones) == 1
    assert microphones[0].index == 1
    assert microphones[0].is_default is True


def test_capture_audio_mono_sample_rate_and_phrase_boundaries() -> None:
    capture = SoundDeviceAudioCapture(
        sample_rate=10,
        initial_silence_timeout=1.0,
        trailing_silence=0.2,
        chunk_duration=0.1,
        speech_threshold=0.05,
    )
    silence = np.zeros((1, 1), dtype=np.float32)
    voice = np.ones((1, 1), dtype=np.float32) * 0.2
    result = capture.capture_from_chunks(
        [silence, silence, voice, voice, silence, silence],
        "Fake Mic",
    )

    assert result.completed is True
    assert result.sample_rate == 10
    assert result.microphone_name == "Fake Mic"
    assert result.samples.ndim == 1
    assert result.duration_seconds == pytest.approx(0.4)


def test_capture_limits_max_duration() -> None:
    capture = SoundDeviceAudioCapture(
        sample_rate=10,
        max_duration=0.3,
        chunk_duration=0.1,
        speech_threshold=0.05,
    )
    voice = np.ones(1, dtype=np.float32) * 0.2

    result = capture.capture_from_chunks([voice, voice, voice, voice], "Fake Mic")

    assert "Duracion maxima alcanzada." in result.warnings


def test_detects_initial_silence_and_no_speech() -> None:
    capture = SoundDeviceAudioCapture(
        sample_rate=10,
        initial_silence_timeout=0.3,
        chunk_duration=0.1,
    )
    silence = np.zeros(1, dtype=np.float32)

    result = capture.capture_from_chunks([silence, silence, silence], "Fake Mic")

    assert result.completed is False
    assert result.no_speech_detected is True


def test_cancels_with_keyboard_interrupt(monkeypatch: pytest.MonkeyPatch) -> None:
    capture = SoundDeviceAudioCapture()
    monkeypatch.setattr(capture, "selected_or_default_microphone", lambda: MicrophoneInfo(0, "Mic", True, 1))
    monkeypatch.setattr(capture, "_sounddevice", lambda: SimpleNamespace(InputStream=lambda **_: (_ for _ in ()).throw(KeyboardInterrupt())))

    result = capture.capture_phrase()

    assert result.cancelled is True
    assert result.completed is False


def test_transcribes_valid_audio_and_returns_structured_result() -> None:
    capture = FakeCapture()
    provider = FakeProvider()
    engine = SpeechEngineUseCase(capture, provider)

    result = engine.transcribe_once()

    assert result.completed is True
    assert result.text == "Atlas abre Visual Studio Code"
    assert result.language == "es"
    assert result.audio_duration_seconds == 1.0
    assert result.processing_duration_seconds == 0.25
    assert result.provider == "fake-local"
    assert result.microphone_name == "Fake Mic"


def test_empty_transcription_no_speech_and_capture_errors_are_controlled() -> None:
    empty_provider = FakeProvider()
    empty_provider.text = ""
    empty = SpeechEngineUseCase(FakeCapture(), empty_provider).transcribe_once()

    no_speech_capture = FakeCapture()
    no_speech_capture.result = AudioCaptureResult(
        samples=np.array([], dtype=np.float32),
        sample_rate=16_000,
        duration_seconds=0.5,
        microphone_name="Fake Mic",
        completed=False,
        no_speech_detected=True,
    )
    no_speech = SpeechEngineUseCase(no_speech_capture, FakeProvider()).transcribe_once()

    busy_capture = FakeCapture()
    busy_capture.raise_error = RuntimeError("dispositivo ocupado")
    busy = SpeechEngineUseCase(busy_capture, FakeProvider()).transcribe_once()

    assert empty.completed is False
    assert "transcripcion vacia" in empty.warnings
    assert no_speech.no_speech_detected is True
    assert "dispositivo ocupado" in busy.summary


def test_permission_model_missing_and_load_failures_are_controlled() -> None:
    permission_capture = FakeCapture()
    permission_capture.raise_error = PermissionError("permisos denegados")
    permission = SpeechEngineUseCase(permission_capture, FakeProvider()).transcribe_once()

    provider = FakeProvider()
    provider.raise_error = RuntimeError("modelo no disponible")
    model_missing = SpeechEngineUseCase(FakeCapture(), provider).transcribe_once()

    assert "permisos denegados" in permission.summary
    assert "modelo no disponible" in model_missing.summary


def test_provider_is_lazy_reused_and_local_only() -> None:
    provider = FakeProvider()
    engine = SpeechEngineUseCase(FakeCapture(), provider)

    assert provider.loaded is False
    first = engine.transcribe_once()
    second = engine.transcribe_once()

    assert first.completed is True
    assert second.completed is True
    assert provider.loaded is True
    assert provider.calls == 2
    assert provider.network_called is False


def test_faster_whisper_temp_file_removed_on_success_and_error(tmp_path: Path) -> None:
    class Segment:
        text = " hola "

    class Info:
        language = "es"

    class Model:
        def __init__(self, fail: bool = False) -> None:
            self.fail = fail

        def transcribe(self, *_args, **_kwargs):
            if self.fail:
                raise RuntimeError("boom")
            return ([Segment()], Info())

    provider = FasterWhisperSpeechToTextProvider()
    path = tmp_path / "audio.wav"
    provider._model = Model()
    provider._write_temp_wav = lambda _samples, _rate: path
    path.write_bytes(b"temp")

    result = provider.transcribe(np.zeros(10, dtype=np.float32), 16_000)

    assert result.text == "hola"
    assert not path.exists()

    provider._model = Model(fail=True)
    provider._write_temp_wav = lambda _samples, _rate: path
    path.write_bytes(b"temp")

    with pytest.raises(RuntimeError):
        provider.transcribe(np.zeros(10, dtype=np.float32), 16_000)

    assert not path.exists()


def test_interprets_spanish_and_english_listen_commands() -> None:
    for command in (
        "escucha una frase",
        "escuchame",
        "transcribe mi voz",
        "listen to me",
    ):
        interaction = SpeechInteractionUseCase(
            SpeechEngineUseCase(FakeCapture(), FakeProvider())
        )
        response = interaction.execute(command)

        assert "Transcripcion:" in str(response)
        assert "La orden transcrita no se ejecuto." in str(response)


def test_interprets_microphone_commands() -> None:
    capture = FakeCapture()
    interaction = SpeechInteractionUseCase(SpeechEngineUseCase(capture, FakeProvider()))

    listed = interaction.execute("lista los microfonos")
    selected = interaction.execute("usa el microfono 2")
    english = interaction.execute("list microphones")

    assert "Microfonos disponibles:" in str(listed)
    assert "Microfono seleccionado: 2 - Second Mic" == selected
    assert "Second Mic" in str(english)


def test_does_not_execute_text_or_call_router(monkeypatch: pytest.MonkeyPatch) -> None:
    capture = FakeCapture()
    provider = FakeProvider()
    interaction = SpeechInteractionUseCase(SpeechEngineUseCase(capture, provider))

    response = interaction.execute("escuchame")

    assert "Atlas abre Visual Studio Code" in str(response)
    assert capture.capture_calls == 1
    assert provider.calls == 1


def test_orchestrator_speech_runs_before_router_and_agent(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    class FailingRouter(Router):
        def route(self, plan):  # pragma: no cover - must not be called
            raise AssertionError("router should not receive transcription")

    inputs = iter(["escuchame", "salir"])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(inputs))
    orchestrator = AtlasOrchestrator(
        planner=Planner(),
        router=FailingRouter(),
        model_manager=ModelManager(),
        memory=ConversationMemory(),
        registry=AgentRegistry(),
        write_file=SimpleNamespace(execute=lambda *_: "unused"),
        speech_interaction=SpeechInteractionUseCase(
            SpeechEngineUseCase(FakeCapture(), FakeProvider())
        ),
    )

    orchestrator.start()
    output = capsys.readouterr().out

    assert "Transcripcion:" in output
    assert "La orden transcrita no se ejecuto." in output


def test_bootstrap_injects_speech_interaction() -> None:
    from bootstrap.bootstrap import Bootstrap

    orchestrator = Bootstrap.build()

    assert orchestrator._speech_interaction is not None
