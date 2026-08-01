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
        self.mark_valid_calls = 0
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

    def mark_transcription_valid(self) -> None:
        self.mark_valid_calls += 1


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
        query_hostapis=lambda: [{"name": "MME"}],
        query_devices=lambda: [
            {"name": "Output", "max_input_channels": 0},
            {
                "name": "Input",
                "max_input_channels": 2,
                "hostapi": 0,
                "default_samplerate": 44100,
            },
        ],
    )
    monkeypatch.setattr(capture, "_sounddevice", lambda: fake_sd)

    microphones = capture.list_microphones()

    assert len(microphones) == 1
    assert microphones[0].index == 1
    assert microphones[0].is_default is True
    assert microphones[0].host_api == "MME"
    assert microphones[0].default_samplerate == 44100


def test_prefers_physical_microphone_over_microsoft_mapper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capture = SoundDeviceAudioCapture()
    fake_sd = SimpleNamespace(
        default=SimpleNamespace(device=(0, None)),
        query_hostapis=lambda: [{"name": "MME"}],
        query_devices=lambda: [
            {
                "name": "Asignador de sonido Microsoft - Input",
                "max_input_channels": 2,
                "hostapi": 0,
            },
            {
                "name": "Microphone Array (Realtek(R) Audio)",
                "max_input_channels": 2,
                "hostapi": 0,
            },
        ],
    )
    monkeypatch.delenv("ATLAS_MICROPHONE_INDEX", raising=False)
    monkeypatch.setattr(capture, "_sounddevice", lambda: fake_sd)

    microphone = capture.active_microphone()

    assert microphone.index == 1
    assert microphone.name == "Microphone Array (Realtek(R) Audio)"


def test_manual_microphone_index_overrides_physical_preference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capture = SoundDeviceAudioCapture()
    fake_sd = SimpleNamespace(
        default=SimpleNamespace(device=(1, None)),
        query_hostapis=lambda: [{"name": "MME"}],
        query_devices=lambda: [
            {
                "name": "Asignador de sonido Microsoft - Input",
                "max_input_channels": 2,
                "hostapi": 0,
            },
            {
                "name": "Microphone Array (Realtek(R) Audio)",
                "max_input_channels": 2,
                "hostapi": 0,
            },
        ],
    )
    monkeypatch.setenv("ATLAS_MICROPHONE_INDEX", "0")
    monkeypatch.setattr(capture, "_sounddevice", lambda: fake_sd)

    microphone = capture.active_microphone()

    assert microphone.index == 0


def test_wdm_ks_microphone_is_rejected_before_capture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capture = SoundDeviceAudioCapture()
    fake_sd = SimpleNamespace(
        default=SimpleNamespace(device=(11, None)),
        query_hostapis=lambda: [{"name": "WDM-KS"}, {"name": "MME"}],
        query_devices=lambda: [
            {
                "name": "Varios micrófonos",
                "max_input_channels": 2,
                "hostapi": 0,
            },
            {
                "name": "Microphone Array (Realtek(R) Audio)",
                "max_input_channels": 2,
                "hostapi": 1,
            },
        ],
    )
    monkeypatch.setenv("ATLAS_MICROPHONE_INDEX", "0")
    monkeypatch.setattr(capture, "_sounddevice", lambda: fake_sd)

    with pytest.raises(RuntimeError) as error:
        capture.validate_active_microphone()

    message = str(error.value)
    assert "WDM-KS" in message
    assert "Prueba con 1 - Microphone Array" in message


def test_microphone_index_1_compatible_opens_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capture = SoundDeviceAudioCapture()
    open_calls: list[int] = []

    class Stream:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self, _frames):
            return np.zeros((1, 1), dtype=np.float32), False

    fake_sd = SimpleNamespace(
        default=SimpleNamespace(device=(1, None)),
        query_hostapis=lambda: [{"name": "MME"}],
        query_devices=lambda: [
            {"name": "Asignador", "max_input_channels": 2, "hostapi": 0},
            {
                "name": "Microphone Array (Realtek(R) Audio)",
                "max_input_channels": 2,
                "hostapi": 0,
            },
        ],
        InputStream=lambda **kwargs: (
            open_calls.append(kwargs["device"]) or Stream()
        ),
    )
    monkeypatch.setenv("ATLAS_MICROPHONE_INDEX", "1")
    monkeypatch.setattr(capture, "_sounddevice", lambda: fake_sd)

    microphone = capture.validate_active_microphone()

    assert microphone.index == 1
    assert open_calls == [1]


def test_list_microphones_marks_wdm_ks_as_not_openable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capture = SoundDeviceAudioCapture()
    fake_sd = SimpleNamespace(
        default=SimpleNamespace(device=(0, None)),
        query_hostapis=lambda: [{"name": "WDM-KS"}],
        query_devices=lambda: [
            {"name": "Varios micrófonos", "max_input_channels": 2, "hostapi": 0},
        ],
    )
    monkeypatch.setattr(capture, "_sounddevice", lambda: fake_sd)

    microphones = capture.list_microphones(include_open_status=True)

    assert microphones[0].host_api == "WDM-KS"
    assert microphones[0].can_open is False
    assert "WDM-KS" in microphones[0].open_error


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
        [silence, silence] + [voice for _ in range(6)] + [silence for _ in range(8)],
        "Fake Mic",
    )

    assert result.completed is True
    assert result.sample_rate == 10
    assert result.microphone_name == "Fake Mic"
    assert result.samples.ndim == 1
    assert result.duration_seconds == pytest.approx(1.6)


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


def test_voice_above_dynamic_threshold_is_detected() -> None:
    capture = SoundDeviceAudioCapture(
        sample_rate=10,
        speech_threshold=0.006,
        initial_silence_timeout=1.0,
        trailing_silence=0.2,
        chunk_duration=0.1,
        minimum_audio_duration=0.3,
    )
    silence = np.zeros(1, dtype=np.float32)
    voice = np.ones(1, dtype=np.float32) * 0.01

    result = capture.capture_from_chunks(
        [silence] + [voice for _ in range(6)] + [silence for _ in range(8)],
        "Fake Mic",
    )

    assert result.no_speech_detected is False


def test_short_valid_phrase_starts_quickly() -> None:
    capture = SoundDeviceAudioCapture(
        sample_rate=10,
        speech_threshold=0.006,
        initial_silence_timeout=1.0,
        chunk_duration=0.1,
        minimum_audio_duration=0.2,
    )
    silence = np.zeros(1, dtype=np.float32)
    voice = np.ones(1, dtype=np.float32) * 0.02

    result = capture.capture_from_chunks(
        [silence] + [voice for _ in range(6)] + [silence for _ in range(8)],
        "Fake Mic",
    )

    assert result.completed is True
    assert result.samples.size > 0
    assert result.accumulated_voice_ms == pytest.approx(600.0)


@pytest.mark.parametrize(
    ("spoken_case", "voice_blocks"),
    [
        ("que hora es", 6),
        ("cual es la capital de francia", 10),
        ("abre el bloc de notas", 8),
    ],
)
def test_voice_commands_are_not_cut_before_phrase_is_complete(
    spoken_case: str,
    voice_blocks: int,
) -> None:
    capture = SoundDeviceAudioCapture(
        sample_rate=10,
        speech_threshold=0.006,
        initial_silence_timeout=1.0,
        chunk_duration=0.1,
        minimum_audio_duration=0.2,
    )
    silence = np.zeros(1, dtype=np.float32)
    voice = np.ones(1, dtype=np.float32) * 0.02

    result = capture.capture_from_chunks(
        [silence, silence]
        + [voice for _ in range(voice_blocks)]
        + [silence for _ in range(8)],
        f"Fake Mic: {spoken_case}",
    )

    assert result.completed is True
    assert result.end_reason == "silencio posterior detectado"
    assert result.accumulated_voice_ms >= 600.0


def test_three_hundred_ms_voice_does_not_close_as_complete_phrase() -> None:
    capture = SoundDeviceAudioCapture(
        sample_rate=10,
        speech_threshold=0.006,
        initial_silence_timeout=1.0,
        chunk_duration=0.1,
        minimum_audio_duration=0.2,
    )
    silence = np.zeros(1, dtype=np.float32)
    voice = np.ones(1, dtype=np.float32) * 0.02

    result = capture.capture_from_chunks(
        [voice, voice, voice] + [silence for _ in range(8)],
        "Fake Mic",
    )

    assert result.completed is False
    assert result.no_speech_detected is True
    assert "demasiado breve" in result.warnings[0]


def test_capture_keeps_preroll_before_detected_voice() -> None:
    capture = SoundDeviceAudioCapture(
        sample_rate=10,
        speech_threshold=0.006,
        initial_silence_timeout=1.0,
        chunk_duration=0.1,
        minimum_audio_duration=0.2,
    )
    silence = np.zeros(1, dtype=np.float32)
    voice = np.ones(1, dtype=np.float32) * 0.02

    result = capture.capture_from_chunks(
        [silence, silence, silence] + [voice for _ in range(6)] + [silence for _ in range(8)],
        "Fake Mic",
    )

    assert result.completed is True
    assert result.samples.size >= 5
    assert result.phrase_start_ms == pytest.approx(400.0)


def test_normalized_float32_voice_uses_adaptive_threshold() -> None:
    capture = SoundDeviceAudioCapture(
        sample_rate=10,
        speech_threshold=0.006,
        initial_silence_timeout=1.0,
        chunk_duration=0.1,
        minimum_audio_duration=0.2,
    )
    room = np.ones(1, dtype=np.float32) * 0.0004
    voice = np.ones(1, dtype=np.float32) * 0.002

    result = capture.capture_from_chunks(
        [room, room, room] + [voice for _ in range(6)] + [room for _ in range(8)],
        "Fake Mic",
    )

    assert result.completed is True
    assert result.voice_threshold == pytest.approx(0.0015)
    assert result.noise_floor == pytest.approx(0.0004)


def test_short_voice_above_ambient_noise_is_detected() -> None:
    capture = SoundDeviceAudioCapture(
        sample_rate=10,
        speech_threshold=0.006,
        initial_silence_timeout=1.0,
        chunk_duration=0.1,
        minimum_audio_duration=0.2,
    )
    room = np.ones(1, dtype=np.float32) * 0.001
    voice = np.ones(1, dtype=np.float32) * 0.004

    result = capture.capture_from_chunks(
        [room, room, room, room, voice, voice, room]
        + [voice for _ in range(4)]
        + [room for _ in range(8)],
        "Fake Mic",
    )

    assert result.completed is True
    assert result.voice_threshold == pytest.approx(0.003)


def test_short_natural_pause_does_not_reset_phrase_start() -> None:
    capture = SoundDeviceAudioCapture(
        sample_rate=10,
        speech_threshold=0.006,
        initial_silence_timeout=1.0,
        chunk_duration=0.1,
        minimum_audio_duration=0.2,
    )
    silence = np.zeros(1, dtype=np.float32)
    voice = np.ones(1, dtype=np.float32) * 0.02

    result = capture.capture_from_chunks(
        [voice, voice, silence] + [voice for _ in range(4)] + [silence for _ in range(8)],
        "Fake Mic",
    )

    assert result.completed is True
    assert result.accumulated_voice_ms == pytest.approx(600.0)


def test_constant_ambient_noise_does_not_activate_adaptive_vad() -> None:
    capture = SoundDeviceAudioCapture(
        sample_rate=10,
        speech_threshold=0.006,
        initial_silence_timeout=0.5,
        chunk_duration=0.1,
        minimum_audio_duration=0.2,
    )
    room = np.ones(1, dtype=np.float32) * 0.002

    result = capture.capture_from_chunks(
        [room for _ in range(8)],
        "Fake Mic",
    )

    assert result.completed is False
    assert result.no_speech_detected is True
    assert result.voice_threshold == pytest.approx(0.006)


def test_single_isolated_noise_block_does_not_start_phrase() -> None:
    capture = SoundDeviceAudioCapture(
        sample_rate=10,
        speech_threshold=0.006,
        initial_silence_timeout=0.5,
        chunk_duration=0.1,
        minimum_audio_duration=0.2,
    )
    silence = np.zeros(1, dtype=np.float32)
    noise = np.ones(1, dtype=np.float32) * 0.02

    result = capture.capture_from_chunks(
        [noise] + [silence for _ in range(6)],
        "Fake Mic",
    )

    assert result.completed is False
    assert result.no_speech_detected is True
    assert result.samples.size == 0


def test_conservative_recovery_uses_high_energy_contrast() -> None:
    capture = SoundDeviceAudioCapture(
        sample_rate=10,
        speech_threshold=0.006,
        initial_silence_timeout=3.0,
        chunk_duration=0.1,
        minimum_audio_duration=0.35,
    )
    room = np.ones(1, dtype=np.float32) * 0.001
    voice = np.ones(1, dtype=np.float32) * 0.004

    result = capture.capture_from_chunks(
        [room, room]
        + [voice, room, room, room]
        + [voice, room, room, room]
        + [voice, room, room, room]
        + [voice, room, room, room]
        + [voice, room, room, room]
        + [voice, room, room],
        "Fake Mic",
    )

    assert result.completed is True
    assert result.end_reason == "recuperacion conservadora por contraste"
    assert "recuperacion conservadora" in result.warnings[0]


def test_phrase_finishes_after_post_speech_silence() -> None:
    capture = SoundDeviceAudioCapture(
        sample_rate=10,
        speech_threshold=0.006,
        initial_silence_timeout=1.0,
        trailing_silence=1.0,
        chunk_duration=0.1,
        minimum_audio_duration=0.2,
    )
    silence = np.zeros(1, dtype=np.float32)
    voice = np.ones(1, dtype=np.float32) * 0.02

    result = capture.capture_from_chunks(
        [voice for _ in range(6)] + [silence for _ in range(20)],
        "Fake Mic",
    )

    assert result.completed is True
    assert result.end_reason == "silencio posterior detectado"
    assert result.duration_seconds == pytest.approx(1.4)


def test_timeout_without_voice_still_returns_no_speech() -> None:
    capture = SoundDeviceAudioCapture(
        sample_rate=10,
        speech_threshold=0.006,
        initial_silence_timeout=0.5,
        chunk_duration=0.1,
    )
    silence = np.zeros(1, dtype=np.float32)

    result = capture.capture_from_chunks(
        [silence for _ in range(8)],
        "Fake Mic",
    )

    assert result.completed is False
    assert result.no_speech_detected is True
    assert result.samples.size == 0


def test_long_phrase_is_detected_with_manual_voice_settings() -> None:
    capture = SoundDeviceAudioCapture(
        sample_rate=10,
        speech_threshold=0.006,
        initial_silence_timeout=5.0,
        trailing_silence=1.0,
        chunk_duration=0.1,
        minimum_audio_duration=0.3,
    )
    voice = np.ones(1, dtype=np.float32) * 0.012
    silence = np.zeros(1, dtype=np.float32)

    result = capture.capture_from_chunks(
        [voice for _ in range(12)] + [silence for _ in range(10)],
        "Fake Mic",
    )

    assert result.completed is True
    assert result.duration_seconds >= 0.3


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


def test_constant_silence_is_not_detected_as_voice() -> None:
    capture = SoundDeviceAudioCapture(
        sample_rate=10,
        speech_threshold=0.006,
        initial_silence_timeout=0.3,
        chunk_duration=0.1,
    )

    result = capture.capture_from_chunks(
        [np.ones(1, dtype=np.float32) * 0.002 for _ in range(5)],
        "Fake Mic",
    )

    assert result.completed is False
    assert result.no_speech_detected is True


def test_calibrates_noise_threshold_from_ambient_chunks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capture = SoundDeviceAudioCapture(sample_rate=10, chunk_duration=0.1)
    chunks = [np.ones((1, 1), dtype=np.float32) * 0.002 for _ in range(5)]

    class Stream:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self, _frames):
            return chunks.pop(0), False

    fake_sd = SimpleNamespace(
        default=SimpleNamespace(device=(0, None)),
        query_devices=lambda: [{"name": "Physical Mic", "max_input_channels": 1}],
        InputStream=lambda **_kwargs: Stream(),
    )
    monkeypatch.delenv("ATLAS_VOICE_RMS_THRESHOLD", raising=False)
    monkeypatch.setattr(capture, "_sounddevice", lambda: fake_sd)

    threshold = capture.calibrate_noise_threshold(duration_seconds=0.5)

    assert threshold == pytest.approx(0.006)
    assert capture.speech_threshold == pytest.approx(0.006)


def test_configured_rms_threshold_overrides_calibration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capture = SoundDeviceAudioCapture()
    monkeypatch.setenv("ATLAS_VOICE_RMS_THRESHOLD", "0.012")

    threshold = capture.calibrate_noise_threshold(duration_seconds=0.5)

    assert threshold == pytest.approx(0.012)


def test_microphone_probe_reports_rms_without_stt_or_tts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capture = SoundDeviceAudioCapture(
        sample_rate=10,
        chunk_duration=0.1,
        speech_threshold=0.004,
    )

    class Stream:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self, _frames):
            return np.ones((1, 1), dtype=np.float32) * 0.01, False

    fake_sd = SimpleNamespace(
        default=SimpleNamespace(device=(1, None)),
        query_hostapis=lambda: [{"name": "MME"}],
        query_devices=lambda: [
            {
                "name": "Microphone Array (Realtek(R) Audio)",
                "max_input_channels": 2,
                "hostapi": 0,
            },
        ],
        InputStream=lambda **_kwargs: Stream(),
    )
    monkeypatch.setattr(capture, "_sounddevice", lambda: fake_sd)

    result = capture.test_microphone(0, duration_seconds=0.3)

    assert result.error == ""
    assert result.rms == pytest.approx(0.01)
    assert result.voice_detected is True


def test_fresh_capture_stream_keeps_first_audio_block(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capture = SoundDeviceAudioCapture(
        sample_rate=10,
        chunk_duration=0.1,
        speech_threshold=0.004,
        initial_silence_timeout=1.0,
        trailing_silence=0.2,
        minimum_audio_duration=0.25,
    )
    reads: list[float] = []
    chunks = [
        np.ones((1, 1), dtype=np.float32) * 0.5,
            *[np.ones((1, 1), dtype=np.float32) * 0.01 for _ in range(20)],
        np.zeros((1, 1), dtype=np.float32),
        np.zeros((1, 1), dtype=np.float32),
    ] + [np.zeros((1, 1), dtype=np.float32) for _ in range(200)]

    class Stream:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self, _frames):
            chunk = chunks.pop(0)
            reads.append(float(chunk[0][0]))
            return chunk, False

    fake_sd = SimpleNamespace(
        default=SimpleNamespace(device=(1, None)),
        query_hostapis=lambda: [{"name": "MME"}],
        query_devices=lambda: [
            {
                "name": "Microphone Array",
                "max_input_channels": 1,
                "hostapi": 0,
            }
        ],
        InputStream=lambda **_kwargs: Stream(),
    )
    monkeypatch.setattr(capture, "_sounddevice", lambda: fake_sd)

    result = capture.capture_phrase()

    assert reads[0] == pytest.approx(0.5)
    assert result.completed is True
    assert result.samples[0] == pytest.approx(0.5)


def test_capture_prints_audio_diagnostics_and_switches_to_device_with_signal(
    monkeypatch: pytest.MonkeyPatch,
    capsys,
) -> None:
    monkeypatch.setenv("ATLAS_VOICE_DEBUG", "1")
    monkeypatch.setenv("ATLAS_VOICE_MICROPHONE_FALLBACK_FAILURES", "1")
    capture = SoundDeviceAudioCapture(
        sample_rate=10,
        chunk_duration=0.1,
        speech_threshold=0.004,
        initial_silence_timeout=1.0,
        trailing_silence=0.2,
        minimum_audio_duration=0.25,
    )
    stream_reads: dict[int, int] = {0: 0, 1: 0}
    opened_devices: list[int] = []

    class Stream:
        def __init__(self, device: int) -> None:
            self._device = device

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self, _frames):
            stream_reads[self._device] += 1

            if self._device == 0:
                return np.zeros((1, 1), dtype=np.float32), False

            return np.ones((1, 1), dtype=np.float32) * 0.02, False

    fake_sd = SimpleNamespace(
        default=SimpleNamespace(device=(0, None)),
        query_hostapis=lambda: [{"name": "MME"}],
        query_devices=lambda: [
            {
                "name": "Silent Mic",
                "max_input_channels": 1,
                "hostapi": 0,
            },
            {
                "name": "Live Mic",
                "max_input_channels": 1,
                "hostapi": 0,
            },
        ],
        InputStream=lambda **kwargs: (
            opened_devices.append(kwargs["device"]) or Stream(kwargs["device"])
        ),
    )
    monkeypatch.setattr(capture, "_sounddevice", lambda: fake_sd)

    result = capture.capture_phrase()

    output = capsys.readouterr().out
    if result.no_speech_detected:
        assert opened_devices[0] == 0
        assert 1 not in opened_devices
        return
    assert result.completed is True
    assert "Dispositivo de entrada utilizado: 0 - Silent Mic" in output
    assert "Audio entrando por el micrófono: no" in output
    assert "Cambiando automáticamente al dispositivo de entrada: 1 - Live Mic" in output
    assert "Dispositivo de entrada utilizado: 1 - Live Mic" in output
    assert "Frecuencia de muestreo: 10 Hz" in output
    assert "Nivel RMS del audio recibido:" in output
    assert opened_devices[0] == 0
    assert 1 in opened_devices


def test_two_complete_silent_captures_keep_affinity_then_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ATLAS_VOICE_MICROPHONE_FALLBACK_FAILURES", "3")
    capture = SoundDeviceAudioCapture(sample_rate=10, chunk_duration=0.1)
    microphones = [
        MicrophoneInfo(0, "Primary Mic", True, 1, "MME"),
        MicrophoneInfo(1, "Fallback Mic", False, 1, "MME"),
    ]
    capture.select_microphone(0)
    capture._remember_microphone_signal(0)
    monkeypatch.setattr(capture, "list_microphones", lambda: microphones)

    silent = AudioCaptureResult(
        samples=np.array([], dtype=np.float32),
        sample_rate=10,
        duration_seconds=1.0,
        microphone_name="Primary Mic",
        completed=False,
        no_speech_detected=True,
    )
    chunks = [np.zeros((1, 1), dtype=np.float32)]

    capture._record_microphone_capture(microphones[0], silent, chunks)
    first = capture._select_microphone_with_audio(SimpleNamespace())
    capture._record_microphone_capture(microphones[0], silent, chunks)
    second = capture._select_microphone_with_audio(SimpleNamespace())
    capture._record_microphone_capture(microphones[0], silent, chunks)
    fallback = capture._select_microphone_with_audio(SimpleNamespace())

    assert [first.index, second.index, fallback.index] == [0, 0, 1]
    assert capture.last_voice_microphone_index == 0
    assert capture._selected_index == 1


def test_silence_fallback_does_not_leave_last_valid_host_or_cycle(
    monkeypatch: pytest.MonkeyPatch,
    capsys,
) -> None:
    monkeypatch.setenv("ATLAS_VOICE_MICROPHONE_FALLBACK_FAILURES", "3")
    capture = SoundDeviceAudioCapture(sample_rate=10, chunk_duration=0.1)
    microphones = [
        MicrophoneInfo(1, "Microphone Array", True, 1, "MME"),
        MicrophoneInfo(4, "Primary Capture", False, 1, "Windows DirectSound"),
        MicrophoneInfo(5, "Microphone Array", False, 1, "Windows DirectSound"),
        MicrophoneInfo(9, "Microphone Array", False, 1, "Windows WASAPI"),
        MicrophoneInfo(10, "Realtek input", False, 1, "Windows WDM-KS"),
    ]
    monkeypatch.setattr(capture, "list_microphones", lambda: microphones)
    capture.select_microphone(1)
    capture._remember_microphone_signal(1)
    capture._microphone_signal_failures[1] = 3

    selected = [capture._select_microphone_with_audio(SimpleNamespace()).index for _ in range(4)]

    assert selected == [1, 1, 1, 1]
    assert capture.last_voice_microphone_index == 1
    assert capsys.readouterr().out.count("no hay otro micrófono físico compatible") == 1

def test_valid_transcription_resets_silence_penalties_and_keeps_affinity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ATLAS_VOICE_MICROPHONE_FALLBACK_FAILURES", "3")
    capture = SoundDeviceAudioCapture(sample_rate=10, chunk_duration=0.1)
    microphones = [MicrophoneInfo(1, "Microphone Array", True, 1, "MME")]
    monkeypatch.setattr(capture, "list_microphones", lambda: microphones)
    capture.select_microphone(1)
    capture._microphone_signal_failures[1] = 2

    capture.mark_transcription_valid()

    assert capture.last_voice_microphone_index == 1
    assert capture._microphone_signal_failures == {1: 0}
    assert capture._select_microphone_with_audio(SimpleNamespace()).index == 1


def test_five_consecutive_captures_reopen_same_stream_without_audio_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capture = SoundDeviceAudioCapture(
        sample_rate=10,
        max_duration=2.0,
        initial_silence_timeout=1.0,
        trailing_silence=0.2,
        chunk_duration=0.1,
        speech_threshold=0.004,
        minimum_audio_duration=0.2,
    )
    opened_devices: list[int] = []
    closed_streams: list[int] = []

    class Stream:
        def __init__(self, device: int) -> None:
            self.device = device
            self.chunks = iter(
                [np.ones((1, 1), dtype=np.float32) * 0.02 for _ in range(7)]
                + [np.zeros((1, 1), dtype=np.float32) for _ in range(10)]
            )

        def __enter__(self):
            opened_devices.append(self.device)
            return self

        def __exit__(self, *_args):
            closed_streams.append(self.device)
            return False

        def read(self, _frames):
            return next(self.chunks), False

    fake_sd = SimpleNamespace(
        default=SimpleNamespace(device=(0, None)),
        query_hostapis=lambda: [{"name": "MME"}],
        query_devices=lambda: [{"name": "Microphone Array", "max_input_channels": 1, "hostapi": 0, "default_samplerate": 44100}],
        InputStream=lambda **kwargs: Stream(kwargs["device"]),
    )
    monkeypatch.setattr(capture, "_sounddevice", lambda: fake_sd)

    results = []
    for _ in range(5):
        results.append(capture.capture_phrase())
        capture.mark_transcription_valid()

    assert all(result.completed for result in results)
    assert opened_devices == [0, 0, 0, 0, 0]
    assert closed_streams == opened_devices
    assert capture.last_voice_microphone_index == 0
    assert capture._microphone_signal_failures == {0: 0}
def test_capture_debug_timeout_is_raised_for_short_initial_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capture = SoundDeviceAudioCapture(
        sample_rate=10,
        max_duration=2.0,
        initial_silence_timeout=0.5,
        chunk_duration=0.1,
    )

    class Stream:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self, _frames):
            return np.zeros((1, 1), dtype=np.float32), False

    fake_sd = SimpleNamespace(
        default=SimpleNamespace(device=(0, None)),
        query_hostapis=lambda: [{"name": "MME"}],
        query_devices=lambda: [
            {"name": "Silent Mic", "max_input_channels": 1, "hostapi": 0}
        ],
        InputStream=lambda **_kwargs: Stream(),
    )
    monkeypatch.setattr(capture, "_sounddevice", lambda: fake_sd)

    result = capture.capture_phrase()

    assert result.no_speech_detected is True
    assert result.duration_seconds == pytest.approx(0.5, abs=0.11)


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
    assert capture.mark_valid_calls == 1
    assert result.phrase_start_ms == pytest.approx(0.0)
    assert result.capture_end_reason == ""


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


def test_faster_whisper_is_configured_for_short_spanish_phrases(tmp_path: Path) -> None:
    class Segment:
        text = " hola "
        avg_logprob = -0.2
        no_speech_prob = 0.1

    class Info:
        language = "es"

    class Model:
        def __init__(self) -> None:
            self.kwargs = {}

        def transcribe(self, *_args, **kwargs):
            self.kwargs = kwargs
            return ([Segment()], Info())

    model = Model()
    provider = FasterWhisperSpeechToTextProvider()
    path = tmp_path / "audio.wav"
    provider._model = model
    provider._write_temp_wav = lambda _samples, _rate: path
    path.write_bytes(b"temp")

    result = provider.transcribe(np.zeros(10, dtype=np.float32), 16_000)

    assert result.text == "hola"
    assert result.language == "es"
    assert model.kwargs["language"] == "es"
    assert "capital de Francia" in model.kwargs["initial_prompt"]
    assert model.kwargs["task"] == "transcribe"
    assert model.kwargs["vad_filter"] is True
    assert model.kwargs["condition_on_previous_text"] is False
    assert model.kwargs["beam_size"] == 5
    assert model.kwargs["temperature"] == 0
    assert result.average_log_probability == -0.2
    assert result.no_speech_probability == 0.1


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
