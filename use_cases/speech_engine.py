"""Speech capture and transcription use cases."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import os
from pathlib import Path
import tempfile
import time
import traceback
from typing import Callable, Protocol
import wave

import numpy as np


@dataclass(frozen=True)
class MicrophoneInfo:
    """Input audio device information."""

    index: int
    name: str
    is_default: bool
    channels: int
    host_api: str = ""
    default_samplerate: float | None = None
    can_open: bool | None = None
    open_error: str = ""


@dataclass(frozen=True)
class MicrophoneTestResult:
    """Result of a direct microphone capture probe."""

    microphone: MicrophoneInfo
    rms: float
    voice_detected: bool
    duration_seconds: float
    error: str = ""


@dataclass(frozen=True)
class AudioCaptureResult:
    """Captured phrase audio."""

    samples: np.ndarray
    sample_rate: int
    duration_seconds: float
    microphone_name: str
    completed: bool
    cancelled: bool = False
    no_speech_detected: bool = False
    warnings: tuple[str, ...] = field(default_factory=tuple)
    phrase_start_ms: float = 0.0
    accumulated_voice_ms: float = 0.0
    end_reason: str = ""
    voice_threshold: float = 0.0
    noise_floor: float = 0.0


@dataclass(frozen=True)
class StreamReadDiagnostics:
    """Raw stream read diagnostics for one phrase capture."""

    callbacks_received: int
    read_blocks: int
    block_lengths: tuple[int, ...]
    total_buffer_length: int
    ndarray_size_before_vad: int
    dtype: str
    overflow_count: int
    drained_block_length: int


@dataclass(frozen=True)
class SpeechCaptureSettings:
    """Temporary audio capture settings."""

    max_duration: float | None = None
    initial_silence_timeout: float | None = None
    trailing_silence: float | None = None
    chunk_duration: float | None = None
    speech_threshold: float | None = None
    minimum_audio_duration: float | None = None


@dataclass(frozen=True)
class ProviderTranscriptionResult:
    """Provider-level transcription result."""

    text: str
    language: str | None
    processing_duration_seconds: float
    provider: str
    average_log_probability: float | None = None
    no_speech_probability: float | None = None


@dataclass(frozen=True)
class SpeechTranscriptionResult:
    """Structured result returned to Atlas."""

    text: str
    language: str | None
    audio_duration_seconds: float
    processing_duration_seconds: float
    provider: str
    microphone_name: str
    completed: bool
    cancelled: bool
    no_speech_detected: bool
    warnings: tuple[str, ...] = field(default_factory=tuple)
    summary: str = ""
    average_log_probability: float | None = None
    no_speech_probability: float | None = None
    samples_count: int = 0
    rms: float = 0.0
    exception_traceback: str = ""
    phrase_start_ms: float = 0.0
    accumulated_voice_ms: float = 0.0
    capture_end_reason: str = ""


class SpeechToTextProvider(Protocol):
    """Minimal speech-to-text provider interface."""

    name: str

    def transcribe(
        self,
        samples: np.ndarray,
        sample_rate: int,
    ) -> ProviderTranscriptionResult:
        """Transcribe mono audio samples."""


class FasterWhisperSpeechToTextProvider:
    """Local faster-whisper provider with lazy model loading."""

    name = "local-faster-whisper"

    def __init__(
        self,
        model_size: str = "small",
        device: str = "cpu",
        compute_type: str = "int8",
        language: str | None = None,
        initial_prompt: str | None = None,
        min_confidence: float | None = None,
        max_no_speech_probability: float | None = None,
    ) -> None:
        self._model_size = os.getenv("ATLAS_STT_MODEL", model_size).strip() or model_size
        self._device = device
        self._compute_type = compute_type
        self._language = language or os.getenv("ATLAS_STT_LANGUAGE", "es").strip() or "es"
        self._initial_prompt = initial_prompt or os.getenv(
            "ATLAS_STT_INITIAL_PROMPT",
            "Atlas, hora, fecha, capital de Francia, abrir Bloc de notas, abrir VS Code",
        )
        self._min_confidence = (
            min_confidence
            if min_confidence is not None
            else _read_float("ATLAS_STT_MIN_CONFIDENCE", 0.35, 0.0, 1.0)
        )
        self._max_no_speech_probability = (
            max_no_speech_probability
            if max_no_speech_probability is not None
            else _read_float("ATLAS_STT_MAX_NO_SPEECH_PROBABILITY", 0.65, 0.0, 1.0)
        )
        self._model = None

    def transcribe(
        self,
        samples: np.ndarray,
        sample_rate: int,
    ) -> ProviderTranscriptionResult:
        """Transcribe samples with a lazily loaded local model."""
        started = time.monotonic()
        model = self._load_model()
        path = self._write_temp_wav(samples, sample_rate)

        try:
            segments, info = model.transcribe(
                str(path),
                language=self._language,
                initial_prompt=self._initial_prompt,
                task="transcribe",
                vad_filter=True,
                condition_on_previous_text=False,
                beam_size=5,
                temperature=0,
            )
            segment_list = list(segments)
            text = " ".join(segment.text.strip() for segment in segment_list).strip()
            language = getattr(info, "language", None)
            average_log_probability = self._average_log_probability(segment_list)
            no_speech_probability = self._max_no_speech_probability_value(segment_list)
        finally:
            path.unlink(missing_ok=True)

        return ProviderTranscriptionResult(
            text=text,
            language=language,
            processing_duration_seconds=time.monotonic() - started,
            provider=self.name,
            average_log_probability=average_log_probability,
            no_speech_probability=no_speech_probability,
        )

    @property
    def language(self) -> str:
        """Forced transcription language."""
        return self._language

    @property
    def initial_prompt(self) -> str:
        """Initial transcription prompt for short Atlas voice commands."""
        return self._initial_prompt

    @property
    def min_confidence(self) -> float:
        """Minimum accepted confidence for short voice mode phrases."""
        return self._min_confidence

    @property
    def max_no_speech_probability(self) -> float:
        """Maximum accepted no-speech probability."""
        return self._max_no_speech_probability

    def _average_log_probability(
        self,
        segments,
    ) -> float | None:
        values = [
            float(value)
            for value in (getattr(segment, "avg_logprob", None) for segment in segments)
            if value is not None
        ]

        if not values:
            return None

        return sum(values) / len(values)

    def _max_no_speech_probability_value(
        self,
        segments,
    ) -> float | None:
        values = [
            float(value)
            for value in (
                getattr(segment, "no_speech_prob", None) for segment in segments
            )
            if value is not None
        ]

        if not values:
            return None

        return max(values)

    def _load_model(self):
        """Load faster-whisper once."""
        if self._model is not None:
            return self._model

        try:
            from faster_whisper import WhisperModel
        except ImportError as error:
            raise RuntimeError(
                "Dependencia no disponible: faster-whisper."
            ) from error

        try:
            self._model = WhisperModel(
                self._model_size,
                device=self._device,
                compute_type=self._compute_type,
            )
        except Exception as error:
            raise RuntimeError(f"No se pudo cargar el modelo de voz: {error}") from error

        return self._model

    def _write_temp_wav(
        self,
        samples: np.ndarray,
        sample_rate: int,
    ) -> Path:
        """Write a temporary WAV file and return its path."""
        with tempfile.NamedTemporaryFile(
            suffix=".wav",
            delete=False,
        ) as handle:
            path = Path(handle.name)

        clipped = np.clip(samples, -1.0, 1.0)
        pcm = (clipped * 32767).astype(np.int16)

        with wave.open(str(path), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(sample_rate)
            wav.writeframes(pcm.tobytes())

        return path


class SoundDeviceAudioCapture:
    """Capture one spoken phrase with sounddevice."""

    _INITIAL_SILENCE_TIMEOUT_WARNING = "No se detecto voz antes del tiempo limite."

    _GENERIC_MICROPHONE_TERMS = (
        "asignador de sonido microsoft",
        "microsoft sound mapper",
    )
    _BLOCKING_UNSUPPORTED_HOST_APIS = {"wdm-ks"}
    _DEFAULT_INPUT_SIGNAL_RMS = 0.0005
    _INITIAL_VOICE_SECONDS = 0.20
    _INITIAL_SILENCE_TOLERANCE_SECONDS = 0.20
    _TRAILING_SILENCE_SECONDS = 0.80
    _MIN_FINAL_VOICE_SECONDS = 0.60
    _MIN_FINAL_VOICE_BLOCKS = 4
    _ADAPTIVE_NOISE_SECONDS = 0.4
    _FLOAT32_MIN_VOICE_RMS = 0.0015
    _NOISE_MULTIPLIER = 3.0
    _RECOVERY_CONTRAST_MULTIPLIER = 2.2
    _SHORT_UTTERANCE_MIN_BLOCKS = 3
    _SHORT_UTTERANCE_MIN_SECONDS = 0.25
    _SHORT_UTTERANCE_THRESHOLD_MULTIPLIER = 1.5
    _SHORT_UTTERANCE_NOISE_MULTIPLIER = 3.0
    _SHORT_UTTERANCE_PADDING_SECONDS = 0.20
    _PRE_ROLL_SECONDS = 0.45
    # After a phrase starts, a later block only counts as voice when it
    # reaches this fraction of the phrase's own median loudness (and the
    # base threshold). This keeps sustained low-level echo/noise from
    # being misclassified as continued speech and holding the turn open.
    _POST_PHRASE_ECHO_FACTOR = 0.35

    def __init__(
        self,
        sample_rate: int = 16_000,
        max_duration: float = 15.0,
        initial_silence_timeout: float = 5.0,
        trailing_silence: float = 1.0,
        chunk_duration: float = 0.1,
        speech_threshold: float | None = None,
        minimum_audio_duration: float = 0.0,
    ) -> None:
        self.sample_rate = sample_rate
        self.max_duration = max_duration
        self.initial_silence_timeout = initial_silence_timeout
        self.trailing_silence = trailing_silence
        self.chunk_duration = chunk_duration
        self.speech_threshold = (
            speech_threshold
            if speech_threshold is not None
            else _read_float("ATLAS_VOICE_RMS_THRESHOLD", 0.004, 0.001, 0.05)
        )
        self.minimum_audio_duration = minimum_audio_duration
        self._selected_index: int | None = None
        self._last_voice_microphone_index: int | None = None
        self._microphone_signal_failures: dict[int, int] = {}
        self._fallback_notice_indices: set[int] = set()
        self._microphone_fallback_failures = int(
            _read_float(
                "ATLAS_VOICE_MICROPHONE_FALLBACK_FAILURES",
                3.0,
                1.0,
                10.0,
            )
        )
        self._empty_capture_diagnostics = 0

    def list_microphones(
        self,
        include_open_status: bool = False,
    ) -> list[MicrophoneInfo]:
        """List available input devices."""
        sd = self._sounddevice()
        devices = sd.query_devices()
        host_apis = self._host_api_names(sd)
        default_index = self._default_input_index(sd)
        microphones: list[MicrophoneInfo] = []

        for index, device in enumerate(devices):
            channels = int(device.get("max_input_channels", 0))

            if channels <= 0:
                continue

            microphone = MicrophoneInfo(
                index=index,
                name=" ".join(
                    str(device.get("name", f"Microfono {index}")).split()
                ),
                is_default=index == default_index,
                channels=channels,
                host_api=host_apis.get(int(device.get("hostapi", -1)), ""),
                default_samplerate=self._device_samplerate(device),
            )

            if include_open_status:
                microphone = self._with_open_status(microphone)

            microphones.append(microphone)

        return microphones

    def default_microphone(self) -> MicrophoneInfo:
        """Return the default input microphone."""
        microphones = self.list_microphones()

        if not microphones:
            raise RuntimeError("No hay microfonos de entrada disponibles.")

        preferred = self._preferred_physical_microphone(microphones)

        if preferred is not None:
            return preferred

        for microphone in microphones:
            if microphone.is_default:
                return microphone

        return microphones[0]

    def select_microphone(
        self,
        index: int,
    ) -> MicrophoneInfo:
        """Select an input device by index."""
        for microphone in self.list_microphones():
            if microphone.index == index:
                self._selected_index = index
                return microphone

        raise ValueError(f"Microfono inexistente: {index}")

    def active_microphone(self) -> MicrophoneInfo:
        """Return selected microphone or the default input device."""
        return self.selected_or_default_microphone()

    def selected_or_default_microphone(self) -> MicrophoneInfo:
        """Return the selected microphone or the default input."""
        if self._selected_index is not None:
            return self.select_microphone(self._selected_index)

        env_index = os.getenv("ATLAS_MICROPHONE_INDEX", "").strip()

        if env_index:
            try:
                return self.select_microphone(int(env_index))
            except ValueError:
                pass

        return self.default_microphone()

    def validate_active_microphone(
        self,
        settings: SpeechCaptureSettings | None = None,
    ) -> MicrophoneInfo:
        """Validate that the active microphone can be opened for blocking capture."""
        microphone = self.selected_or_default_microphone()

        if self._is_blocking_unsupported(microphone):
            raise RuntimeError(self._unsupported_microphone_message(microphone))

        self._open_probe_stream(microphone, settings)
        return microphone

    def test_microphone(
        self,
        index: int,
        duration_seconds: float = 3.0,
    ) -> MicrophoneTestResult:
        """Open a microphone directly and measure RMS without Whisper or TTS."""
        microphone = self.select_microphone(index)

        try:
            if self._is_blocking_unsupported(microphone):
                raise RuntimeError(self._unsupported_microphone_message(microphone))

            samples = self._capture_fixed_duration(microphone, duration_seconds)
        except Exception as error:
            return MicrophoneTestResult(
                microphone=microphone,
                rms=0.0,
                voice_detected=False,
                duration_seconds=0.0,
                error=str(error),
            )

        rms = self._rms(samples)
        return MicrophoneTestResult(
            microphone=microphone,
            rms=rms,
            voice_detected=rms >= self.speech_threshold,
            duration_seconds=len(samples) / self.sample_rate,
        )

    def calibrate_noise_threshold(
        self,
        settings: SpeechCaptureSettings | None = None,
        duration_seconds: float = 0.5,
    ) -> float:
        """Estimate ambient noise and return a safe speech RMS threshold."""
        configured = os.getenv("ATLAS_VOICE_RMS_THRESHOLD", "").strip()

        if configured:
            self.speech_threshold = _read_float(
                "ATLAS_VOICE_RMS_THRESHOLD",
                self.speech_threshold,
                0.001,
                0.05,
            )
            return self.speech_threshold

        original = self._snapshot_settings()
        self._apply_settings(settings)

        try:
            microphone = self.selected_or_default_microphone()
            sd = self._sounddevice()
            frames = max(1, int(self.sample_rate * self.chunk_duration))
            max_chunks = max(1, math.ceil(duration_seconds / self.chunk_duration))
            rms_values: list[float] = []

            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="float32",
                device=microphone.index,
            ) as stream:
                for _ in range(max_chunks):
                    data, _overflowed = stream.read(frames)
                    rms_values.append(self._rms(self._mono_float32(data)))

            noise_floor = max(rms_values) if rms_values else 0.0
            threshold = min(max(noise_floor * 3.0, 0.006), 0.03)
            self.speech_threshold = threshold
            return threshold
        finally:
            restored = dict(original)
            restored["speech_threshold"] = self.speech_threshold
            self._restore_settings(restored)

    def prepare_stream(
        self,
        settings: SpeechCaptureSettings | None = None,
    ) -> MicrophoneInfo:
        """Open and close the active input stream without retaining audio."""
        original = self._snapshot_settings()
        self._apply_settings(settings)

        try:
            microphone = self.selected_or_default_microphone()
            sd = self._sounddevice()
            frames = max(1, int(self.sample_rate * self.chunk_duration))

            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="float32",
                device=microphone.index,
            ) as stream:
                stream.read(frames)

            return microphone
        finally:
            self._restore_settings(original)

    def iter_pcm_frames(
        self,
        sample_rate: int,
        frame_length: int,
    ):
        """Yield signed 16-bit mono PCM frames from the active microphone."""
        microphone = self.selected_or_default_microphone()
        sd = self._sounddevice()

        with sd.InputStream(
            samplerate=sample_rate,
            channels=1,
            dtype="int16",
            blocksize=frame_length,
            device=microphone.index,
        ) as stream:
            while True:
                data, _overflowed = stream.read(frame_length)
                yield np.asarray(data, dtype=np.int16).reshape(-1)

    def capture_phrase(
        self,
        settings: SpeechCaptureSettings | None = None,
    ) -> AudioCaptureResult:
        """Capture one phrase from a fresh stream on the affinity microphone."""
        original = self._snapshot_settings()
        self._apply_settings(settings)
        self._apply_debug_timeouts()

        try:
            sd = self._sounddevice()
            microphone = self._select_microphone_with_audio(sd)
            frames = int(self.sample_rate * self.chunk_duration)

            try:
                return self._capture_open_stream(sd, microphone, frames)
            except KeyboardInterrupt:
                return AudioCaptureResult(
                    samples=np.array([], dtype=np.float32),
                    sample_rate=self.sample_rate,
                    duration_seconds=0.0,
                    microphone_name=microphone.name,
                    completed=False,
                    cancelled=True,
                    warnings=("captura cancelada por el usuario",),
                )
            except Exception as error:
                self._microphone_signal_failures[microphone.index] = self._microphone_fallback_failures
                alternative = self._first_open_microphone(exclude_index=microphone.index)
                if alternative is None:
                    raise RuntimeError(
                        f"Fallo al abrir o leer el stream del microfono: {error}"
                    ) from error
                self._selected_index = alternative.index
                print(
                    "Cambiando automáticamente al dispositivo de entrada tras "
                    f"fallo real de apertura: {alternative.index} - {alternative.name}"
                )
                try:
                    return self._capture_open_stream(sd, alternative, frames)
                except Exception as fallback_error:
                    raise RuntimeError(
                        "Fallo al abrir o leer el stream del microfono de fallback: "
                        f"{fallback_error}"
                    ) from fallback_error
        except KeyboardInterrupt:
            return AudioCaptureResult(
                samples=np.array([], dtype=np.float32),
                sample_rate=self.sample_rate,
                duration_seconds=0.0,
                microphone_name="desconocido",
                completed=False,
                cancelled=True,
                warnings=("captura cancelada por el usuario",),
            )
        finally:
            self._restore_settings(original)
    def _apply_debug_timeouts(self) -> None:
        if self.max_duration < self.initial_silence_timeout:
            self.max_duration = self.initial_silence_timeout + 1.0

    def _select_microphone_with_audio(
        self,
        sd,
    ) -> MicrophoneInfo:
        """Keep affinity unless complete captures reached the fallback limit."""
        microphone = self.selected_or_default_microphone()
        if self._is_blocking_unsupported(microphone):
            raise RuntimeError(self._unsupported_microphone_message(microphone))

        failures = self._microphone_signal_failures.get(microphone.index, 0)
        if failures < self._microphone_fallback_failures:
            return microphone

        blocked_indices = {
            index
            for index, count in self._microphone_signal_failures.items()
            if count >= self._microphone_fallback_failures
        }
        alternative = self._first_fallback_candidate(
            exclude_index=microphone.index,
            excluded_indices=blocked_indices,
            preferred_host_api=microphone.host_api,
        )

        if alternative is None:
            if microphone.index not in self._fallback_notice_indices:
                print(
                    "Audio de micrófono: se mantiene el último dispositivo válido; "
                    "no hay otro micrófono físico compatible."
                )
                self._fallback_notice_indices.add(microphone.index)
            return microphone

        self._selected_index = alternative.index
        print(
            "Cambiando automáticamente al dispositivo de entrada: "
            f"{alternative.index} - {alternative.name}"
        )
        return alternative
    @property
    def last_voice_microphone_index(self) -> int | None:
        """Return the most recent device that produced a valid voice signal."""
        return self._last_voice_microphone_index

    def _record_microphone_capture(
        self,
        microphone: MicrophoneInfo,
        capture: AudioCaptureResult,
        chunks: list[np.ndarray],
    ) -> None:
        rms_values = [self._rms(self._mono_float32(chunk)) for chunk in chunks]
        rms = max(rms_values) if rms_values else 0.0
        self._print_capture_diagnostic(
            microphone,
            {"rms": rms, "has_audio": rms >= self._input_signal_threshold()},
        )

        if capture.completed and not capture.no_speech_detected and capture.samples.size > 0:
            self._selected_index = microphone.index
            self._last_voice_microphone_index = microphone.index
            return

        failures = self._microphone_signal_failures.get(microphone.index, 0) + 1
        self._microphone_signal_failures[microphone.index] = failures
        if failures < self._microphone_fallback_failures:
            print(
                "No se detectó voz; se mantiene el micrófono actual "
                f"({failures}/{self._microphone_fallback_failures})."
            )

    def mark_transcription_valid(self) -> None:
        """Confirm affinity and reset temporary penalties after valid STT."""
        if self._selected_index is not None:
            self._remember_microphone_signal(self._selected_index)

    def _remember_microphone_signal(self, index: int) -> None:
        self._selected_index = index
        self._last_voice_microphone_index = index
        self._microphone_signal_failures.clear()
        self._microphone_signal_failures[index] = 0
        self._fallback_notice_indices.clear()

    def _capture_open_stream(
        self,
        sd,
        microphone: MicrophoneInfo,
        frames: int,
    ) -> AudioCaptureResult:
        """Capture with one fresh stream; closing the context releases it."""
        with sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            device=microphone.index,
        ) as stream:
            chunks, diagnostics = self._read_stream_chunks(stream, frames, 0)
            capture = self.capture_from_chunks(chunks, microphone.name)
            self._print_stream_read_diagnostic(diagnostics, capture, chunks)
            self._record_microphone_capture(microphone, capture, chunks)
            return capture
    def _first_fallback_candidate(
        self,
        exclude_index: int,
        excluded_indices: set[int] | None = None,
        preferred_host_api: str = "",
    ) -> MicrophoneInfo | None:
        excluded_indices = excluded_indices or set()
        normalized_host_api = _normalize_text(preferred_host_api)
        candidates = [
            microphone
            for microphone in self.list_microphones()
            if microphone.index != exclude_index
            and microphone.index not in excluded_indices
            and not self._is_blocking_unsupported(microphone)
            and not self._is_generic_microphone(microphone.name)
            and (
                not normalized_host_api
                or _normalize_text(microphone.host_api) == normalized_host_api
            )
        ]
        return candidates[0] if candidates else None
    def _first_open_microphone(
        self,
        exclude_index: int,
    ) -> MicrophoneInfo | None:
        candidates = [
            microphone
            for microphone in self.list_microphones()
            if microphone.index != exclude_index
            and not self._is_blocking_unsupported(microphone)
        ]
        candidates.sort(key=lambda item: self._is_generic_microphone(item.name))

        for microphone in candidates:
            try:
                self._open_probe_stream(microphone, None)
            except Exception:
                continue
            return microphone

        return None
    def _print_capture_diagnostic(
        self,
        microphone: MicrophoneInfo,
        diagnostic: dict[str, float | bool],
    ) -> None:
        if not self._voice_debug_enabled():
            return

        host_api = f" ({microphone.host_api})" if microphone.host_api else ""
        print(
            "Dispositivo de entrada utilizado: "
            f"{microphone.index} - {microphone.name}{host_api}"
        )
        print(f"Frecuencia de muestreo: {self.sample_rate} Hz")
        print(f"Nivel RMS del audio recibido: {float(diagnostic['rms']):.6f}")
        print(
            "Audio entrando por el micrófono: "
            f"{'sí' if diagnostic['has_audio'] else 'no'}"
        )

    def _input_signal_threshold(self) -> float:
        return _read_float(
            "ATLAS_AUDIO_INPUT_MIN_RMS",
            self._DEFAULT_INPUT_SIGNAL_RMS,
            0.0,
            0.05,
        )

    def capture_from_chunks(
        self,
        chunks: list[np.ndarray],
        microphone_name: str = "fake microphone",
    ) -> AudioCaptureResult:
        """Capture a phrase from controlled audio chunks."""
        captured: list[np.ndarray] = []
        pending_voice: list[np.ndarray] = []
        pre_voice_buffer: list[np.ndarray] = []
        mono_chunks = [self._mono_float32(chunk) for chunk in chunks]
        block_rms_values = [self._rms(chunk) for chunk in mono_chunks]
        noise_floor = self._noise_floor(block_rms_values)
        voice_threshold = self._adaptive_voice_threshold(noise_floor)
        voice_started = False
        voice_started_at = 0.0
        accumulated_voice = 0.0
        pending_silence = 0.0
        pending_voice_blocks = 0
        voice_blocks = 0
        elapsed = 0.0
        silence_after_voice = 0.0
        warnings: list[str] = []
        end_reason = "fin de bloques"
        required_voice_duration = self._initial_voice_duration()
        tolerated_initial_silence = self._initial_silence_tolerance()
        trailing_silence = self._trailing_silence_duration()
        phrase_levels: list[float] = []

        for mono, rms in zip(mono_chunks, block_rms_values):
            chunk_duration = len(mono) / self.sample_rate
            chunk_started_at = elapsed
            elapsed += chunk_duration

            if voice_started:
                # Post-start classification adapts to the phrase's own
                # loudness so residual echo/noise cannot hold the turn.
                phrase_levels.append(rms)
                effective_threshold = max(
                    voice_threshold,
                    float(np.median(phrase_levels)) * self._POST_PHRASE_ECHO_FACTOR,
                )
                is_voice = rms >= effective_threshold
            else:
                is_voice = rms >= voice_threshold

            if not voice_started:
                if is_voice:
                    pending_voice.append(mono)
                    pending_voice_blocks += 1
                    voice_blocks += 1
                    accumulated_voice += chunk_duration
                    pending_silence = 0.0

                    if (
                        pending_voice_blocks >= 2
                        and accumulated_voice >= required_voice_duration
                    ):
                        voice_started = True
                        voice_started_at = chunk_started_at
                        captured.extend(pre_voice_buffer)
                        captured.extend(pending_voice)
                        pending_voice = []
                        pre_voice_buffer = []
                        silence_after_voice = 0.0
                else:
                    if pending_voice and pending_silence < tolerated_initial_silence:
                        pending_voice.append(mono)
                        pending_silence += chunk_duration
                    else:
                        pre_voice_buffer.append(mono)
                        self._trim_pre_roll(pre_voice_buffer)
                        pending_voice = []
                        pending_silence = 0.0
                        pending_voice_blocks = 0
                        accumulated_voice = 0.0

                if not voice_started and elapsed >= self.initial_silence_timeout:
                    return self._no_speech_result(
                        microphone_name,
                        elapsed,
                        self._INITIAL_SILENCE_TIMEOUT_WARNING,
                        voice_threshold=voice_threshold,
                        noise_floor=noise_floor,
                    )
            else:
                captured.append(mono)
                if is_voice:
                    voice_blocks += 1
                    accumulated_voice += chunk_duration
                    silence_after_voice = 0.0
                else:
                    silence_after_voice += chunk_duration

                captured_duration = sum(len(item) for item in captured) / self.sample_rate

                if (
                    silence_after_voice + 1e-9 >= trailing_silence
                    and captured_duration >= self.minimum_audio_duration
                    and self._has_complete_voice(accumulated_voice, voice_blocks)
                ):
                    end_reason = "silencio posterior detectado"
                    break

            if elapsed >= self.max_duration:
                warnings.append("Duracion maxima alcanzada.")
                end_reason = "duracion maxima alcanzada"
                break

        if not self._has_complete_voice(accumulated_voice, voice_blocks):
            short_utterance = self._recover_short_utterance(
                mono_chunks,
                block_rms_values,
                voice_threshold,
                noise_floor,
            )

            if short_utterance is not None:
                samples, start_ms, voice_ms = short_utterance
                warnings.append(
                    "short utterance conservado: voz consecutiva con contraste suficiente"
                )
                return AudioCaptureResult(
                    samples=samples,
                    sample_rate=self.sample_rate,
                    duration_seconds=len(samples) / self.sample_rate,
                    microphone_name=microphone_name,
                    completed=True,
                    warnings=tuple(warnings),
                    phrase_start_ms=start_ms,
                    accumulated_voice_ms=voice_ms,
                    end_reason="short utterance por contraste",
                    voice_threshold=voice_threshold,
                    noise_floor=noise_floor,
                )

        if not voice_started:
            recovered = self._recover_contrasting_phrase(
                mono_chunks,
                block_rms_values,
                voice_threshold,
                noise_floor,
            )

            if recovered is not None:
                samples, start_ms, voice_ms = recovered
                warnings.append(
                    "recuperacion conservadora: contraste de energia suficiente"
                )
                return AudioCaptureResult(
                    samples=samples,
                    sample_rate=self.sample_rate,
                    duration_seconds=len(samples) / self.sample_rate,
                    microphone_name=microphone_name,
                    completed=True,
                    warnings=tuple(warnings),
                    phrase_start_ms=start_ms,
                    accumulated_voice_ms=voice_ms,
                    end_reason="recuperacion conservadora por contraste",
                    voice_threshold=voice_threshold,
                    noise_floor=noise_floor,
                )

            return self._no_speech_result(
                microphone_name,
                elapsed,
                "No se detecto ninguna frase.",
                voice_threshold=voice_threshold,
                noise_floor=noise_floor,
            )

        if not self._has_complete_voice(accumulated_voice, voice_blocks):
            warnings.append("Voz demasiado breve para formar una frase completa.")
            return AudioCaptureResult(
                samples=np.array([], dtype=np.float32),
                sample_rate=self.sample_rate,
                duration_seconds=elapsed,
                microphone_name=microphone_name,
                completed=False,
                no_speech_detected=True,
                warnings=tuple(warnings),
                voice_threshold=voice_threshold,
                noise_floor=noise_floor,
            )

        samples = np.concatenate(captured).astype(np.float32)

        return AudioCaptureResult(
            samples=samples,
            sample_rate=self.sample_rate,
            duration_seconds=len(samples) / self.sample_rate,
            microphone_name=microphone_name,
            completed=True,
            warnings=tuple(warnings),
            phrase_start_ms=voice_started_at * 1000.0,
            accumulated_voice_ms=accumulated_voice * 1000.0,
            end_reason=end_reason,
            voice_threshold=voice_threshold,
            noise_floor=noise_floor,
        )

    def _read_stream_chunks(
        self,
        stream,
        frames: int,
        drained_block_length: int = 0,
    ) -> tuple[list[np.ndarray], StreamReadDiagnostics]:
        chunks: list[np.ndarray] = []
        block_lengths: list[int] = []
        overflow_count = 0
        max_chunks = math.ceil(self.max_duration / self.chunk_duration)

        for _ in range(max_chunks):
            data, overflowed = stream.read(frames)
            chunks.append(data)
            mono = self._mono_float32(data)
            block_lengths.append(len(mono))
            if overflowed:
                overflow_count += 1

            partial = self.capture_from_chunks(chunks)
            if partial.completed and partial.end_reason in {
                "silencio posterior detectado",
                "short utterance por contraste",
            }:
                break
            if partial.warnings == (self._INITIAL_SILENCE_TIMEOUT_WARNING,):
                break

        total_buffer_length = sum(block_lengths)
        raw_buffer = self._concatenate_chunks(chunks)

        return chunks, StreamReadDiagnostics(
            callbacks_received=0,
            read_blocks=len(chunks),
            block_lengths=tuple(block_lengths),
            total_buffer_length=total_buffer_length,
            ndarray_size_before_vad=int(raw_buffer.size),
            dtype=str(raw_buffer.dtype),
            overflow_count=overflow_count,
            drained_block_length=drained_block_length,
        )

    def _drain_input_buffer(
        self,
        stream,
        frames: int,
    ) -> int:
        try:
            data, _overflowed = stream.read(frames)
        except Exception:
            raise

        return len(self._mono_float32(data))

    def _concatenate_chunks(
        self,
        chunks: list[np.ndarray],
    ) -> np.ndarray:
        if not chunks:
            return np.array([], dtype=np.float32)

        mono_chunks = [self._mono_float32(chunk) for chunk in chunks]
        non_empty = [chunk for chunk in mono_chunks if len(chunk) > 0]

        if not non_empty:
            return np.array([], dtype=np.float32)

        return np.concatenate(non_empty).astype(np.float32)

    def _print_stream_read_diagnostic(
        self,
        diagnostics: StreamReadDiagnostics,
        capture: AudioCaptureResult,
        chunks: list[np.ndarray],
    ) -> None:
        if not self._voice_debug_enabled():
            return

        if capture.samples.size == 0:
            self._empty_capture_diagnostics += 1
        else:
            self._empty_capture_diagnostics = 0

        if self._empty_capture_diagnostics > 3:
            print(
                "[speech-debug] captura sin voz util repetida; "
                "se omite diagnostico extenso temporalmente"
            )
            time.sleep(0.2)
            return

        block_lengths_text = ", ".join(
            str(length) for length in diagnostics.block_lengths
        )
        if len(block_lengths_text) > 240:
            block_lengths_text = f"{block_lengths_text[:240]}..."

        block_rms_values = [
            self._rms(self._mono_float32(chunk))
            for chunk in chunks
        ]
        noise_floor = capture.noise_floor or self._noise_floor(block_rms_values)
        voice_threshold = (
            capture.voice_threshold
            or self._adaptive_voice_threshold(noise_floor)
        )
        voice_block_rms = [
            rms for rms in block_rms_values if rms >= voice_threshold
        ]
        voice_blocks = len(voice_block_rms)
        rms_stats = self._block_rms_stats(block_rms_values)
        reason = "captura valida enviada a STT"

        if capture.samples.size == 0:
            if diagnostics.total_buffer_length == 0:
                reason = "stream.read devolvio bloques vacios"
            elif voice_blocks == 0:
                reason = "el buffer tenia muestras pero ninguna supero el umbral de voz"
            else:
                reason = (
                    "hubo bloques con voz, pero no alcanzaron la duracion minima "
                    "para iniciar la frase"
                )

        print("[speech-debug] callbacks recibidos: 0 (InputStream usa stream.read)")
        print(f"[speech-debug] bloques leidos por stream.read: {diagnostics.read_blocks}")
        print(
            "[speech-debug] longitud de cada bloque: "
            f"[{block_lengths_text}]"
        )
        print(
            "[speech-debug] longitud total del buffer leido: "
            f"{diagnostics.total_buffer_length}"
        )
        print(
            "[speech-debug] tamano ndarray antes de VAD/STT: "
            f"{diagnostics.ndarray_size_before_vad}"
        )
        print(f"[speech-debug] dtype ndarray antes de VAD/STT: {diagnostics.dtype}")
        print(
            "[speech-debug] bloque descartado al limpiar buffer inicial: "
            f"{diagnostics.drained_block_length}"
        )
        print(f"[speech-debug] overflows detectados: {diagnostics.overflow_count}")
        print(f"[speech-debug] RMS minimo bloques: {rms_stats['min']:.6f}")
        print(f"[speech-debug] RMS maximo bloques: {rms_stats['max']:.6f}")
        print(f"[speech-debug] RMS medio bloques: {rms_stats['mean']:.6f}")
        print(f"[speech-debug] RMS p50 bloques: {rms_stats['p50']:.6f}")
        print(f"[speech-debug] RMS p75 bloques: {rms_stats['p75']:.6f}")
        print(f"[speech-debug] RMS p90 bloques: {rms_stats['p90']:.6f}")
        print(f"[speech-debug] RMS p95 bloques: {rms_stats['p95']:.6f}")
        print(f"[speech-debug] noise floor adaptativo: {noise_floor:.6f}")
        print(f"[speech-debug] umbral de voz aplicado: {voice_threshold:.6f}")
        print(
            "[speech-debug] RMS bloques clasificados como voz: "
            f"{self._format_rms_values(voice_block_rms)}"
        )
        print(f"[speech-debug] bloques con voz segun umbral: {voice_blocks}")
        print(
            "[speech-debug] ms hasta detectar inicio de frase: "
            f"{capture.phrase_start_ms:.1f}"
        )
        print(
            "[speech-debug] duracion de voz acumulada: "
            f"{capture.accumulated_voice_ms:.1f} ms"
        )
        print(f"[speech-debug] motivo de finalizacion: {capture.end_reason or 'n/a'}")
        print(
            "[speech-debug] tamano ndarray final antes del STT: "
            f"{int(capture.samples.size)}"
        )
        print(f"[speech-debug] motivo si termina en cero muestras: {reason}")

    def _initial_voice_duration(self) -> float:
        configured = self.minimum_audio_duration or self._INITIAL_VOICE_SECONDS
        return min(max(configured, self._INITIAL_VOICE_SECONDS), 0.35)

    def _initial_silence_tolerance(self) -> float:
        return max(self.chunk_duration, self._INITIAL_SILENCE_TOLERANCE_SECONDS)

    def _trailing_silence_duration(self) -> float:
        """Effective trailing-silence window for closing a turn.

        Respects the configured ``trailing_silence`` within safe bounds:
        a floor avoids cutting words prematurely; a ceiling guarantees
        the turn closes even with residual noise or TTS echo.
        """
        configured = self.trailing_silence
        if configured is None or configured <= 0:
            configured = self._TRAILING_SILENCE_SECONDS
        return min(max(float(configured), 0.3), 1.5)

    def _has_complete_voice(
        self,
        accumulated_voice: float,
        voice_blocks: int,
    ) -> bool:
        return (
            accumulated_voice >= self._MIN_FINAL_VOICE_SECONDS
            and voice_blocks >= self._MIN_FINAL_VOICE_BLOCKS
        )

    def _trim_pre_roll(
        self,
        chunks: list[np.ndarray],
    ) -> None:
        max_samples = max(1, int(self.sample_rate * self._PRE_ROLL_SECONDS))

        while sum(len(chunk) for chunk in chunks) > max_samples and len(chunks) > 1:
            chunks.pop(0)

    def _noise_floor(
        self,
        block_rms_values: list[float],
    ) -> float:
        if not block_rms_values:
            return 0.0

        values = np.asarray(block_rms_values, dtype=np.float32)
        ambient_blocks = max(
            1,
            math.ceil(self._ADAPTIVE_NOISE_SECONDS / self.chunk_duration),
        )
        ambient = values[:ambient_blocks]
        ambient_median = float(np.median(ambient))
        global_p25 = float(np.percentile(values, 25))

        return min(ambient_median, global_p25)

    def _adaptive_voice_threshold(
        self,
        noise_floor: float,
    ) -> float:
        adaptive = max(
            self._FLOAT32_MIN_VOICE_RMS,
            noise_floor * self._NOISE_MULTIPLIER,
        )
        configured_cap = max(
            self._FLOAT32_MIN_VOICE_RMS,
            min(max(self.speech_threshold, 0.0), 1.0),
        )

        return min(adaptive, configured_cap)

    def _recover_short_utterance(
        self,
        mono_chunks: list[np.ndarray],
        block_rms_values: list[float],
        voice_threshold: float,
        noise_floor: float,
    ) -> tuple[np.ndarray, float, float] | None:
        if not mono_chunks or len(mono_chunks) != len(block_rms_values):
            return None

        short_threshold = max(
            self._FLOAT32_MIN_VOICE_RMS
            * self._SHORT_UTTERANCE_THRESHOLD_MULTIPLIER,
            voice_threshold,
            noise_floor * self._SHORT_UTTERANCE_NOISE_MULTIPLIER,
        )
        runs: list[tuple[int, int, float]] = []
        run_start: int | None = None

        for index, rms in enumerate(block_rms_values):
            if rms >= short_threshold:
                if run_start is None:
                    run_start = index
                continue

            if run_start is not None:
                duration = sum(
                    len(mono_chunks[item]) / self.sample_rate
                    for item in range(run_start, index)
                )
                runs.append((run_start, index, duration))
                run_start = None

        if run_start is not None:
            duration = sum(
                len(mono_chunks[item]) / self.sample_rate
                for item in range(run_start, len(mono_chunks))
            )
            runs.append((run_start, len(mono_chunks), duration))

        def trailing_quiet_duration(run_end: int) -> float:
            duration = 0.0
            for index in range(run_end, len(mono_chunks)):
                if block_rms_values[index] >= short_threshold:
                    break
                duration += len(mono_chunks[index]) / self.sample_rate
            return duration

        required_trailing_silence = self._trailing_silence_duration()
        eligible = [
            run
            for run in runs
            if run[1] - run[0] >= self._SHORT_UTTERANCE_MIN_BLOCKS
            and run[2] >= self._SHORT_UTTERANCE_MIN_SECONDS
            and trailing_quiet_duration(run[1]) + 1e-9
            >= required_trailing_silence
        ]
        if not eligible:
            return None

        voice_start, voice_end, voice_duration = max(
            eligible,
            key=lambda run: (run[2], run[1] - run[0]),
        )
        padding_blocks = max(
            1,
            math.ceil(
                self._SHORT_UTTERANCE_PADDING_SECONDS / self.chunk_duration
            ),
        )
        capture_start = max(0, voice_start - padding_blocks)
        capture_end = min(len(mono_chunks), voice_end + padding_blocks)
        samples = np.concatenate(
            mono_chunks[capture_start:capture_end]
        ).astype(np.float32)
        start_ms = (
            sum(len(chunk) for chunk in mono_chunks[:voice_start])
            / self.sample_rate
            * 1000.0
        )

        return samples, start_ms, voice_duration * 1000.0

    def _recover_contrasting_phrase(
        self,
        mono_chunks: list[np.ndarray],
        block_rms_values: list[float],
        voice_threshold: float,
        noise_floor: float,
    ) -> tuple[np.ndarray, float, float] | None:
        if not mono_chunks or not block_rms_values:
            return None

        peak = max(block_rms_values)
        contrast_threshold = max(
            self._FLOAT32_MIN_VOICE_RMS,
            noise_floor * self._RECOVERY_CONTRAST_MULTIPLIER,
        )

        if peak < contrast_threshold:
            return None

        candidate_indexes = [
            index
            for index, rms in enumerate(block_rms_values)
            if rms >= contrast_threshold
        ]

        if len(candidate_indexes) < 2:
            return None

        start = max(0, candidate_indexes[0] - 1)
        end = min(
            len(mono_chunks),
            candidate_indexes[-1]
            + 1
            + math.ceil(self._trailing_silence_duration() / self.chunk_duration),
        )
        samples = np.concatenate(mono_chunks[start:end]).astype(np.float32)
        voice_ms = sum(
            len(mono_chunks[index]) / self.sample_rate
            for index in candidate_indexes
        ) * 1000.0

        if voice_ms < self._MIN_FINAL_VOICE_SECONDS * 1000.0:
            return None

        return samples, start * self.chunk_duration * 1000.0, voice_ms

    def _block_rms_stats(
        self,
        block_rms_values: list[float],
    ) -> dict[str, float]:
        if not block_rms_values:
            return {
                "min": 0.0,
                "max": 0.0,
                "mean": 0.0,
                "p50": 0.0,
                "p75": 0.0,
                "p90": 0.0,
                "p95": 0.0,
            }

        values = np.asarray(block_rms_values, dtype=np.float32)
        return {
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "mean": float(np.mean(values)),
            "p50": float(np.percentile(values, 50)),
            "p75": float(np.percentile(values, 75)),
            "p90": float(np.percentile(values, 90)),
            "p95": float(np.percentile(values, 95)),
        }

    def _format_rms_values(
        self,
        values: list[float],
    ) -> str:
        if not values:
            return "[]"

        text = ", ".join(f"{value:.6f}" for value in values)
        if len(text) > 240:
            text = f"{text[:240]}..."

        return f"[{text}]"

    def _voice_debug_enabled(self) -> bool:
        return _read_bool("ATLAS_VOICE_DEBUG", False)

    def _capture_fixed_duration(
        self,
        microphone: MicrophoneInfo,
        duration_seconds: float,
    ) -> np.ndarray:
        sd = self._sounddevice()
        frames = max(1, int(self.sample_rate * self.chunk_duration))
        max_chunks = max(1, math.ceil(duration_seconds / self.chunk_duration))
        chunks: list[np.ndarray] = []

        with sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            device=microphone.index,
        ) as stream:
            for _ in range(max_chunks):
                data, _overflowed = stream.read(frames)
                chunks.append(self._mono_float32(data))

        if not chunks:
            return np.array([], dtype=np.float32)

        return np.concatenate(chunks).astype(np.float32)

    def _open_probe_stream(
        self,
        microphone: MicrophoneInfo,
        settings: SpeechCaptureSettings | None,
    ) -> None:
        original = self._snapshot_settings()
        self._apply_settings(settings)

        try:
            sd = self._sounddevice()
            frames = max(1, int(self.sample_rate * self.chunk_duration))

            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="float32",
                device=microphone.index,
            ) as stream:
                stream.read(frames)
        except Exception as error:
            raise RuntimeError(
                self._open_error_message(microphone, error)
            ) from error
        finally:
            self._restore_settings(original)

    def _with_open_status(
        self,
        microphone: MicrophoneInfo,
    ) -> MicrophoneInfo:
        can_open = True
        open_error = ""

        if self._is_blocking_unsupported(microphone):
            can_open = False
            open_error = "WDM-KS no compatible con captura bloqueante"
        else:
            try:
                self._open_probe_stream(microphone, None)
            except Exception as error:
                can_open = False
                open_error = str(error)

        return MicrophoneInfo(
            index=microphone.index,
            name=microphone.name,
            is_default=microphone.is_default,
            channels=microphone.channels,
            host_api=microphone.host_api,
            default_samplerate=microphone.default_samplerate,
            can_open=can_open,
            open_error=open_error,
        )

    def _no_speech_result(
        self,
        microphone_name: str,
        duration: float,
        warning: str,
        voice_threshold: float = 0.0,
        noise_floor: float = 0.0,
    ) -> AudioCaptureResult:
        return AudioCaptureResult(
            samples=np.array([], dtype=np.float32),
            sample_rate=self.sample_rate,
            duration_seconds=duration,
            microphone_name=microphone_name,
            completed=False,
            no_speech_detected=True,
            warnings=(warning,),
            voice_threshold=voice_threshold,
            noise_floor=noise_floor,
        )

    def _mono_float32(
        self,
        chunk: np.ndarray,
    ) -> np.ndarray:
        array = np.asarray(chunk, dtype=np.float32)

        if array.ndim == 2:
            array = array[:, 0]

        return array.reshape(-1)

    def _rms(
        self,
        chunk: np.ndarray,
    ) -> float:
        if len(chunk) == 0:
            return 0.0

        return float(np.sqrt(np.mean(np.square(chunk))))

    def _sounddevice(self):
        try:
            import sounddevice as sd
        except ImportError as error:
            raise RuntimeError("Dependencia no disponible: sounddevice.") from error

        return sd

    def _snapshot_settings(self) -> dict[str, float]:
        return {
            "max_duration": self.max_duration,
            "initial_silence_timeout": self.initial_silence_timeout,
            "trailing_silence": self.trailing_silence,
            "chunk_duration": self.chunk_duration,
            "speech_threshold": self.speech_threshold,
            "minimum_audio_duration": self.minimum_audio_duration,
        }

    def _apply_settings(
        self,
        settings: SpeechCaptureSettings | None,
    ) -> None:
        if settings is None:
            return

        for name in self._snapshot_settings():
            value = getattr(settings, name)

            if value is not None:
                setattr(self, name, value)

    def _restore_settings(
        self,
        settings: dict[str, float],
    ) -> None:
        for name, value in settings.items():
            setattr(self, name, value)

    def _default_input_index(
        self,
        sd,
    ) -> int | None:
        default = getattr(sd, "default", None)
        device = getattr(default, "device", None)

        if isinstance(device, (list, tuple)) and len(device) >= 1:
            value = device[0]
            return int(value) if value is not None and int(value) >= 0 else None

        return None

    def _host_api_names(
        self,
        sd,
    ) -> dict[int, str]:
        try:
            host_apis = sd.query_hostapis()
        except Exception:
            return {}

        return {
            index: str(host_api.get("name", ""))
            for index, host_api in enumerate(host_apis)
        }

    def _device_samplerate(
        self,
        device,
    ) -> float | None:
        value = device.get("default_samplerate")

        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _preferred_physical_microphone(
        self,
        microphones: list[MicrophoneInfo],
    ) -> MicrophoneInfo | None:
        physical = [
            microphone
            for microphone in microphones
            if not self._is_generic_microphone(microphone.name)
            and not self._is_blocking_unsupported(microphone)
        ]

        if not physical:
            return None

        for microphone in physical:
            if microphone.is_default:
                return microphone

        return physical[0]

    def _is_generic_microphone(
        self,
        name: str,
    ) -> bool:
        normalized = _normalize_text(name)
        return any(term in normalized for term in self._GENERIC_MICROPHONE_TERMS)

    def _is_blocking_unsupported(
        self,
        microphone: MicrophoneInfo,
    ) -> bool:
        normalized = _normalize_text(microphone.host_api)
        return any(
            host_api in normalized
            for host_api in self._BLOCKING_UNSUPPORTED_HOST_APIS
        )

    def _unsupported_microphone_message(
        self,
        microphone: MicrophoneInfo,
    ) -> str:
        recommendation = self._recommended_alternative_message(microphone)
        return (
            f"Dispositivo incompatible para captura bloqueante: "
            f"{microphone.index} - {microphone.name} ({microphone.host_api}). "
            "WDM-KS no es compatible con este flujo de InputStream. "
            f"{recommendation}"
        ).strip()

    def _open_error_message(
        self,
        microphone: MicrophoneInfo,
        error: Exception,
    ) -> str:
        recommendation = self._recommended_alternative_message(microphone)
        return (
            f"Fallo al abrir stream del microfono {microphone.index} - "
            f"{microphone.name} ({microphone.host_api}): {error}. "
            f"{recommendation}"
        ).strip()

    def _recommended_alternative_message(
        self,
        microphone: MicrophoneInfo,
    ) -> str:
        alternative = self._recommended_alternative(microphone)

        if alternative is None:
            return "Usa un indice MME, DirectSound o WASAPI visible en --list-microphones."

        return (
            f"Prueba con {alternative.index} - {alternative.name} "
            f"({alternative.host_api})."
        )

    def _recommended_alternative(
        self,
        microphone: MicrophoneInfo,
    ) -> MicrophoneInfo | None:
        safe_host_apis = {"mme", "directsound", "wasapi"}
        target_name = _normalize_text(microphone.name)
        microphones = self.list_microphones()
        candidates = [
            item
            for item in microphones
            if _normalize_text(item.host_api) in safe_host_apis
            and item.index != microphone.index
            and not self._is_generic_microphone(item.name)
        ]

        for item in candidates:
            if _normalize_text(item.name) == target_name:
                return item

        for item in candidates:
            if "microphone array" in _normalize_text(item.name):
                return item

        return candidates[0] if candidates else None


class SpeechEngineUseCase:
    """Capture and transcribe one user phrase."""

    def __init__(
        self,
        capture: SoundDeviceAudioCapture,
        provider: SpeechToTextProvider,
    ) -> None:
        self._capture = capture
        self._provider = provider

    def list_microphones(
        self,
        include_open_status: bool = False,
    ) -> list[MicrophoneInfo]:
        """List input microphones."""
        try:
            return self._capture.list_microphones(include_open_status)
        except TypeError:
            return self._capture.list_microphones()

    def default_microphone(self) -> MicrophoneInfo:
        """Return default microphone."""
        return self._capture.default_microphone()

    def active_microphone(self) -> MicrophoneInfo:
        """Return selected microphone or default microphone."""
        active_microphone = getattr(self._capture, "active_microphone", None)

        if callable(active_microphone):
            return active_microphone()

        return self.default_microphone()

    def validate_active_microphone(
        self,
        capture_settings: SpeechCaptureSettings | None = None,
    ) -> MicrophoneInfo:
        """Validate that the active microphone can open before voice mode starts."""
        validate = getattr(self._capture, "validate_active_microphone", None)

        if not callable(validate):
            return self.active_microphone()

        return validate(capture_settings)

    def test_microphone(
        self,
        index: int,
        duration_seconds: float = 3.0,
    ) -> MicrophoneTestResult:
        """Run a direct microphone probe without STT or TTS."""
        test = getattr(self._capture, "test_microphone", None)

        if not callable(test):
            raise RuntimeError("La prueba de microfono no esta disponible.")

        return test(index, duration_seconds)

    def select_microphone(
        self,
        index: int,
    ) -> MicrophoneInfo:
        """Select a microphone."""
        return self._capture.select_microphone(index)

    def warm_up(self) -> None:
        """Prepare microphone lookup and the local provider."""
        self.active_microphone()
        load_model = getattr(self._provider, "_load_model", None)

        if callable(load_model):
            load_model()

    def prepare_stream(
        self,
        capture_settings: SpeechCaptureSettings | None = None,
    ) -> MicrophoneInfo:
        """Prepare the active microphone stream before capture."""
        prepare_stream = getattr(self._capture, "prepare_stream", None)

        if callable(prepare_stream):
            return prepare_stream(capture_settings)

        return self.active_microphone()

    def calibrate_noise_threshold(
        self,
        capture_settings: SpeechCaptureSettings | None = None,
        duration_seconds: float = 0.5,
    ) -> float | None:
        """Calibrate the capture threshold when supported by the capture backend."""
        calibrate = getattr(self._capture, "calibrate_noise_threshold", None)

        if not callable(calibrate):
            return None

        return calibrate(capture_settings, duration_seconds)

    def iter_pcm_frames(
        self,
        sample_rate: int,
        frame_length: int,
    ):
        """Yield PCM frames from the selected microphone."""
        iter_pcm_frames = getattr(self._capture, "iter_pcm_frames", None)

        if not callable(iter_pcm_frames):
            raise RuntimeError("La captura PCM no esta disponible.")

        return iter_pcm_frames(sample_rate, frame_length)

    def transcribe_once(
        self,
        capture_settings: SpeechCaptureSettings | None = None,
        stage_sink: Callable[[str], None] | None = None,
    ) -> SpeechTranscriptionResult:
        """Capture one phrase and transcribe it."""
        started = time.monotonic()

        try:
            try:
                capture = self._capture.capture_phrase(capture_settings)
            except TypeError:
                capture = self._capture.capture_phrase()
        except Exception as error:
            return self._failed_result(
                str(error),
                exception_traceback=traceback.format_exc(),
            )

        if capture.cancelled:
            return self._capture_failure(capture, cancelled=True)

        if capture.no_speech_detected or len(capture.samples) == 0:
            return self._capture_failure(capture, no_speech=True)

        if stage_sink is not None:
            stage_sink("transcribing")

        try:
            provider_result = self._provider.transcribe(
                capture.samples,
                capture.sample_rate,
            )
        except Exception as error:
            return self._failed_result(
                str(error),
                microphone_name=capture.microphone_name,
                audio_duration=capture.duration_seconds,
                samples_count=self._sample_count(capture.samples),
                rms=self._samples_rms(capture.samples),
                exception_traceback=traceback.format_exc(),
            )

        text = provider_result.text.strip()

        if not text:
            return SpeechTranscriptionResult(
                text="",
                language=provider_result.language,
                audio_duration_seconds=capture.duration_seconds,
                processing_duration_seconds=time.monotonic() - started,
                provider=provider_result.provider,
                microphone_name=capture.microphone_name,
                completed=False,
                cancelled=False,
                no_speech_detected=False,
                warnings=capture.warnings + ("transcripcion vacia",),
                summary="La transcripcion esta vacia.",
                average_log_probability=provider_result.average_log_probability,
                no_speech_probability=provider_result.no_speech_probability,
                samples_count=self._sample_count(capture.samples),
                rms=self._samples_rms(capture.samples),
                phrase_start_ms=capture.phrase_start_ms,
                accumulated_voice_ms=capture.accumulated_voice_ms,
                capture_end_reason=capture.end_reason,
            )

        warnings = list(capture.warnings)
        confidence = self._confidence(provider_result.average_log_probability)

        if (
            provider_result.no_speech_probability is not None
            and provider_result.no_speech_probability
            > self._provider_threshold("max_no_speech_probability", 0.65)
        ):
            warnings.append("probabilidad alta de silencio")

        if (
            confidence is not None
            and confidence < self._provider_threshold("min_confidence", 0.35)
        ):
            warnings.append("confianza baja de transcripcion")

        result = SpeechTranscriptionResult(
            text=text,
            language=provider_result.language,
            audio_duration_seconds=capture.duration_seconds,
            processing_duration_seconds=provider_result.processing_duration_seconds,
            provider=provider_result.provider,
            microphone_name=capture.microphone_name,
            completed=True,
            cancelled=False,
            no_speech_detected=False,
            warnings=tuple(warnings),
            summary="Transcripcion completada.",
            average_log_probability=provider_result.average_log_probability,
            no_speech_probability=provider_result.no_speech_probability,
            samples_count=self._sample_count(capture.samples),
            rms=self._samples_rms(capture.samples),
            phrase_start_ms=capture.phrase_start_ms,
            accumulated_voice_ms=capture.accumulated_voice_ms,
            capture_end_reason=capture.end_reason,
        )

        mark_valid = getattr(self._capture, "mark_transcription_valid", None)
        if callable(mark_valid):
            mark_valid()
        return result
    def _confidence(
        self,
        average_log_probability: float | None,
    ) -> float | None:
        if average_log_probability is None:
            return None

        return max(0.0, min(1.0, math.exp(average_log_probability)))

    def _provider_threshold(
        self,
        name: str,
        default: float,
    ) -> float:
        value = getattr(self._provider, name, default)

        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _capture_failure(
        self,
        capture: AudioCaptureResult,
        cancelled: bool = False,
        no_speech: bool = False,
    ) -> SpeechTranscriptionResult:
        return SpeechTranscriptionResult(
            text="",
            language=None,
            audio_duration_seconds=capture.duration_seconds,
            processing_duration_seconds=0.0,
            provider=self._provider.name,
            microphone_name=capture.microphone_name,
            completed=False,
            cancelled=cancelled,
            no_speech_detected=no_speech,
            warnings=capture.warnings,
            summary="No se detecto ninguna frase antes del tiempo limite."
            if no_speech
            else "Captura cancelada.",
            average_log_probability=None,
            no_speech_probability=None,
            samples_count=self._sample_count(capture.samples),
            rms=self._samples_rms(capture.samples),
            phrase_start_ms=capture.phrase_start_ms,
            accumulated_voice_ms=capture.accumulated_voice_ms,
            capture_end_reason=capture.end_reason,
        )

    def _failed_result(
        self,
        warning: str,
        microphone_name: str = "",
        audio_duration: float = 0.0,
        samples_count: int = 0,
        rms: float = 0.0,
        exception_traceback: str = "",
    ) -> SpeechTranscriptionResult:
        return SpeechTranscriptionResult(
            text="",
            language=None,
            audio_duration_seconds=audio_duration,
            processing_duration_seconds=0.0,
            provider=self._provider.name,
            microphone_name=microphone_name,
            completed=False,
            cancelled=False,
            no_speech_detected=False,
            warnings=(warning,),
            summary=f"No se pudo transcribir la frase: {warning}",
            average_log_probability=None,
            no_speech_probability=None,
            samples_count=samples_count,
            rms=rms,
            exception_traceback=exception_traceback,
        )

    def _sample_count(
        self,
        samples: np.ndarray,
    ) -> int:
        return int(np.asarray(samples).size)

    def _samples_rms(
        self,
        samples: np.ndarray,
    ) -> float:
        array = np.asarray(samples, dtype=np.float32)

        if array.size == 0:
            return 0.0

        return float(np.sqrt(np.mean(np.square(array))))


class SpeechInteractionUseCase:
    """Interpret speech commands for the interactive Atlas flow."""

    _LIST_COMMANDS = {
        "lista los microfonos",
        "muestra los microfonos",
        "list microphones",
    }
    _LISTEN_COMMANDS = {
        "escucha una frase",
        "escuchame",
        "transcribe mi voz",
        "listen to me",
    }

    def __init__(
        self,
        speech_engine: SpeechEngineUseCase,
    ) -> None:
        self._speech_engine = speech_engine

    def execute(
        self,
        prompt: str,
    ) -> str | None:
        """Execute a speech command or return None."""
        normalized = self._normalize(prompt)

        try:
            if normalized in self._LIST_COMMANDS:
                return self._format_microphones(self._speech_engine.list_microphones())

            if normalized.startswith("usa el microfono "):
                return self._select_microphone(normalized, "usa el microfono ")

            if normalized.startswith("use microphone "):
                return self._select_microphone(normalized, "use microphone ")

            if normalized in self._LISTEN_COMMANDS:
                result = self._speech_engine.transcribe_once()
                return self._format_transcription(result)
        except Exception as error:
            return f"Error: {error}"

        return None

    def list_microphones_text(self) -> str:
        """Return a user-facing list of input microphones."""
        return self._format_microphones(
            self._speech_engine.list_microphones(include_open_status=True)
        )

    def test_microphone_text(
        self,
        index: int,
    ) -> str:
        """Return a user-facing direct microphone test result."""
        result = self._speech_engine.test_microphone(index, duration_seconds=3.0)
        lines = [
            f"Microfono probado: {self._format_microphone(result.microphone)}",
        ]

        if result.error:
            lines.append(f"Error: {result.error}")
            return "\n".join(lines)

        lines.append(f"Duracion: {result.duration_seconds:.1f} s")
        lines.append(f"RMS: {result.rms:.4f}")
        lines.append(
            "Voz detectada: si" if result.voice_detected else "Voz detectada: no"
        )
        return "\n".join(lines)

    def _select_microphone(
        self,
        normalized: str,
        prefix: str,
    ) -> str:
        value = normalized[len(prefix) :].strip()

        if not value.isdigit():
            raise ValueError("Indice de microfono invalido.")

        microphone = self._speech_engine.select_microphone(int(value))
        return f"Microfono seleccionado: {microphone.index} - {microphone.name}"

    def _format_microphones(
        self,
        microphones: list[MicrophoneInfo],
    ) -> str:
        if not microphones:
            return "No hay microfonos de entrada disponibles."

        lines = ["Microfonos disponibles:"]

        for microphone in microphones:
            lines.append(self._format_microphone(microphone))

        return "\n".join(lines)

    def _format_microphone(
        self,
        microphone: MicrophoneInfo,
    ) -> str:
        suffix = " (predeterminado)" if microphone.is_default else ""
        samplerate = (
            f"{microphone.default_samplerate:.0f} Hz"
            if microphone.default_samplerate is not None
            else "frecuencia desconocida"
        )
        open_status = ""

        if microphone.can_open is True:
            open_status = " | abre: si"
        elif microphone.can_open is False:
            open_status = f" | abre: no ({microphone.open_error})"

        return (
            f"{microphone.index}. {microphone.name}{suffix} | "
            f"host API: {microphone.host_api or 'desconocida'} | "
            f"canales: {microphone.channels} | {samplerate}{open_status}"
        )

    def _format_transcription(
        self,
        result: SpeechTranscriptionResult,
    ) -> str:
        if result.no_speech_detected:
            return "No se detecto ninguna frase antes del tiempo limite."

        if result.cancelled:
            return "Captura de voz cancelada."

        if not result.completed:
            warning = "; ".join(result.warnings)
            return f"No se pudo transcribir la frase: {warning}"

        return "\n".join(
            [
                "Escuchando...",
                "",
                "Transcripcion:",
                result.text,
                "",
                f"Duracion de audio: {result.audio_duration_seconds:.1f} s",
                f"Tiempo de procesamiento: {result.processing_duration_seconds:.1f} s",
                f"Proveedor: {result.provider}",
                f"Microfono: {result.microphone_name}",
                "",
                "La orden transcrita no se ejecuto.",
            ]
        )

    def _normalize(
        self,
        text: str,
    ) -> str:
        import unicodedata

        normalized = unicodedata.normalize("NFKD", text.strip().lower())
        without_accents = "".join(
            character
            for character in normalized
            if unicodedata.category(character) != "Mn"
        )

        return " ".join(without_accents.split())


def _read_float(
    name: str,
    default: float,
    minimum: float,
    maximum: float,
) -> float:
    raw = os.getenv(name, "").strip()

    if not raw:
        return default

    try:
        value = float(raw)
    except ValueError:
        return default

    return min(max(value, minimum), maximum)


def _read_bool(
    name: str,
    default: bool,
) -> bool:
    raw = os.getenv(name, "").strip().lower()

    if not raw:
        return default

    return raw in {"1", "true", "yes", "on", "si", "sí"}


def _normalize_text(
    text: str,
) -> str:
    import unicodedata

    normalized = unicodedata.normalize("NFKD", text.strip().lower())
    without_accents = "".join(
        character
        for character in normalized
        if unicodedata.category(character) != "Mn"
    )
    return " ".join(without_accents.split())
