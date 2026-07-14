"""Speech capture and transcription use cases."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import math
import tempfile
import time
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


@dataclass(frozen=True)
class ProviderTranscriptionResult:
    """Provider-level transcription result."""

    text: str
    language: str | None
    processing_duration_seconds: float
    provider: str


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
    ) -> None:
        self._model_size = model_size
        self._device = device
        self._compute_type = compute_type
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
                language="es",
                vad_filter=False,
            )
            text = " ".join(segment.text.strip() for segment in segments).strip()
            language = getattr(info, "language", None)
        finally:
            path.unlink(missing_ok=True)

        return ProviderTranscriptionResult(
            text=text,
            language=language,
            processing_duration_seconds=time.monotonic() - started,
            provider=self.name,
        )

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

    def __init__(
        self,
        sample_rate: int = 16_000,
        max_duration: float = 15.0,
        initial_silence_timeout: float = 5.0,
        trailing_silence: float = 1.0,
        chunk_duration: float = 0.1,
        speech_threshold: float = 0.015,
    ) -> None:
        self.sample_rate = sample_rate
        self.max_duration = max_duration
        self.initial_silence_timeout = initial_silence_timeout
        self.trailing_silence = trailing_silence
        self.chunk_duration = chunk_duration
        self.speech_threshold = speech_threshold
        self._selected_index: int | None = None

    def list_microphones(self) -> list[MicrophoneInfo]:
        """List available input devices."""
        sd = self._sounddevice()
        devices = sd.query_devices()
        default_index = self._default_input_index(sd)
        microphones: list[MicrophoneInfo] = []

        for index, device in enumerate(devices):
            channels = int(device.get("max_input_channels", 0))

            if channels <= 0:
                continue

            microphones.append(
                MicrophoneInfo(
                    index=index,
                    name=str(device.get("name", f"Microfono {index}")),
                    is_default=index == default_index,
                    channels=channels,
                )
            )

        return microphones

    def default_microphone(self) -> MicrophoneInfo:
        """Return the default input microphone."""
        microphones = self.list_microphones()

        if not microphones:
            raise RuntimeError("No hay microfonos de entrada disponibles.")

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

    def selected_or_default_microphone(self) -> MicrophoneInfo:
        """Return the selected microphone or the default input."""
        if self._selected_index is not None:
            return self.select_microphone(self._selected_index)

        return self.default_microphone()

    def capture_phrase(self) -> AudioCaptureResult:
        """Capture one phrase from the selected microphone."""
        microphone = self.selected_or_default_microphone()
        sd = self._sounddevice()
        frames = int(self.sample_rate * self.chunk_duration)

        try:
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="float32",
                device=microphone.index,
            ) as stream:
                chunks = self._read_stream_chunks(stream, frames)
                return self.capture_from_chunks(chunks, microphone.name)
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
            raise RuntimeError(f"No se pudo capturar audio: {error}") from error

    def capture_from_chunks(
        self,
        chunks: list[np.ndarray],
        microphone_name: str = "fake microphone",
    ) -> AudioCaptureResult:
        """Capture a phrase from controlled audio chunks."""
        captured: list[np.ndarray] = []
        voice_started = False
        elapsed = 0.0
        silence_after_voice = 0.0
        warnings: list[str] = []

        for chunk in chunks:
            mono = self._mono_float32(chunk)
            elapsed += len(mono) / self.sample_rate
            rms = self._rms(mono)
            is_voice = rms >= self.speech_threshold

            if not voice_started:
                if is_voice:
                    voice_started = True
                    captured.append(mono)
                    silence_after_voice = 0.0
                elif elapsed >= self.initial_silence_timeout:
                    return self._no_speech_result(
                        microphone_name,
                        elapsed,
                        "No se detecto voz antes del tiempo limite.",
                    )
            else:
                captured.append(mono)
                silence_after_voice = 0.0 if is_voice else (
                    silence_after_voice + len(mono) / self.sample_rate
                )

                if silence_after_voice >= self.trailing_silence:
                    break

            if elapsed >= self.max_duration:
                warnings.append("Duracion maxima alcanzada.")
                break

        if not voice_started:
            return self._no_speech_result(
                microphone_name,
                elapsed,
                "No se detecto ninguna frase.",
            )

        samples = np.concatenate(captured).astype(np.float32)

        return AudioCaptureResult(
            samples=samples,
            sample_rate=self.sample_rate,
            duration_seconds=len(samples) / self.sample_rate,
            microphone_name=microphone_name,
            completed=True,
            warnings=tuple(warnings),
        )

    def _read_stream_chunks(
        self,
        stream,
        frames: int,
    ) -> list[np.ndarray]:
        chunks: list[np.ndarray] = []
        max_chunks = math.ceil(self.max_duration / self.chunk_duration)

        for _ in range(max_chunks):
            data, _overflowed = stream.read(frames)
            chunks.append(data)

        return chunks

    def _no_speech_result(
        self,
        microphone_name: str,
        duration: float,
        warning: str,
    ) -> AudioCaptureResult:
        return AudioCaptureResult(
            samples=np.array([], dtype=np.float32),
            sample_rate=self.sample_rate,
            duration_seconds=duration,
            microphone_name=microphone_name,
            completed=False,
            no_speech_detected=True,
            warnings=(warning,),
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


class SpeechEngineUseCase:
    """Capture and transcribe one user phrase."""

    def __init__(
        self,
        capture: SoundDeviceAudioCapture,
        provider: SpeechToTextProvider,
    ) -> None:
        self._capture = capture
        self._provider = provider

    def list_microphones(self) -> list[MicrophoneInfo]:
        """List input microphones."""
        return self._capture.list_microphones()

    def default_microphone(self) -> MicrophoneInfo:
        """Return default microphone."""
        return self._capture.default_microphone()

    def select_microphone(
        self,
        index: int,
    ) -> MicrophoneInfo:
        """Select a microphone."""
        return self._capture.select_microphone(index)

    def transcribe_once(self) -> SpeechTranscriptionResult:
        """Capture one phrase and transcribe it."""
        started = time.monotonic()

        try:
            capture = self._capture.capture_phrase()
        except Exception as error:
            return self._failed_result(str(error))

        if capture.cancelled:
            return self._capture_failure(capture, cancelled=True)

        if capture.no_speech_detected or len(capture.samples) == 0:
            return self._capture_failure(capture, no_speech=True)

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
            )

        return SpeechTranscriptionResult(
            text=text,
            language=provider_result.language,
            audio_duration_seconds=capture.duration_seconds,
            processing_duration_seconds=provider_result.processing_duration_seconds,
            provider=provider_result.provider,
            microphone_name=capture.microphone_name,
            completed=True,
            cancelled=False,
            no_speech_detected=False,
            warnings=capture.warnings,
            summary="Transcripcion completada.",
        )

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
        )

    def _failed_result(
        self,
        warning: str,
        microphone_name: str = "",
        audio_duration: float = 0.0,
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
        )


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
            suffix = " (predeterminado)" if microphone.is_default else ""
            lines.append(f"{microphone.index}. {microphone.name}{suffix}")

        return "\n".join(lines)

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
