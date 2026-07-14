"""Wake word detection use cases."""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
import time
import unicodedata
from typing import Callable, Protocol

import numpy as np

from use_cases.speech_engine import SpeechEngineUseCase, SpeechTranscriptionResult


@dataclass(frozen=True)
class WakeWordDetectionResult:
    """Result of waiting for a wake word and optionally capturing one phrase."""

    wake_word: str
    detected: bool
    attempts: int
    elapsed_seconds: float
    phrase: SpeechTranscriptionResult | None = None
    cancelled: bool = False
    configuration_error: bool = False
    warnings: tuple[str, ...] = field(default_factory=tuple)


class WakeWordProvider(Protocol):
    """Minimal acoustic wake word provider interface."""

    @property
    def sample_rate(self) -> int:
        """Required sample rate."""

    @property
    def frame_length(self) -> int:
        """Required frame length."""

    def initialize(self) -> None:
        """Initialize provider resources."""

    def process_frame(
        self,
        pcm_frame: np.ndarray,
    ) -> bool:
        """Return True when the wake word is detected."""

    def close(self) -> None:
        """Release provider resources."""


class OpenWakeWordProvider:
    """openWakeWord-based local wake word provider."""

    _MISSING_CONFIG_MESSAGE = (
        "Wake word no configurada. "
        "Define ATLAS_WAKE_WORD_MODEL_PATH con un modelo .onnx local."
    )
    _SAMPLE_RATE = 16_000
    _FRAME_LENGTH = 1_280

    def __init__(
        self,
        model_path: str | Path | None = None,
        sensitivity: float | str | None = None,
        model_factory: Callable[..., object] | None = None,
    ) -> None:
        self._model_path = Path(model_path) if model_path is not None else None
        self._sensitivity = sensitivity
        self._model_factory = model_factory
        self._model = None

    @classmethod
    def from_environment(cls) -> "OpenWakeWordProvider":
        """Create provider using environment variables only."""
        return cls(
            model_path=os.environ.get("ATLAS_WAKE_WORD_MODEL_PATH"),
            sensitivity=os.environ.get("ATLAS_WAKE_WORD_SENSITIVITY"),
        )

    @property
    def sample_rate(self) -> int:
        """Required sample rate."""
        return self._SAMPLE_RATE

    @property
    def frame_length(self) -> int:
        """Required frame length."""
        return self._FRAME_LENGTH

    def initialize(self) -> None:
        """Initialize openWakeWord lazily."""
        if self._model is not None:
            return

        model_path, _sensitivity = self._validated_configuration()
        factory = self._model_factory or self._load_model_factory()
        self._model = factory(
            wakeword_models=[str(model_path)],
            inference_framework="onnx",
        )

    def process_frame(
        self,
        pcm_frame: np.ndarray,
    ) -> bool:
        """Process one mono PCM frame using acoustic score only."""
        self._ensure_initialized()
        frame = np.asarray(pcm_frame, dtype=np.int16).reshape(-1)
        _model_path, sensitivity = self._validated_configuration()

        if len(frame) == 0:
            raise RuntimeError("Frame PCM invalido para openWakeWord.")

        predictions = self._model.predict(frame)

        if not isinstance(predictions, dict):
            raise RuntimeError("Prediccion invalida de openWakeWord.")

        scores = [float(score) for score in predictions.values()]
        return bool(scores) and max(scores) >= sensitivity

    def close(self) -> None:
        """Reset provider state without unloading the model."""
        if self._model is None:
            return

        reset = getattr(self._model, "reset", None)

        if callable(reset):
            reset()

    def _validated_configuration(self) -> tuple[Path, float]:
        model_path = self._model_path
        sensitivity = self._validated_sensitivity()

        if model_path is None:
            raise RuntimeError(self._MISSING_CONFIG_MESSAGE)

        resolved_model_path = model_path.expanduser().resolve()

        if resolved_model_path.suffix.lower() != ".onnx":
            raise RuntimeError("ATLAS_WAKE_WORD_MODEL_PATH debe apuntar a un archivo .onnx.")

        if not resolved_model_path.is_file():
            raise RuntimeError("ATLAS_WAKE_WORD_MODEL_PATH no existe.")

        return resolved_model_path, sensitivity

    def _validated_sensitivity(self) -> float:
        try:
            sensitivity = 0.55 if self._sensitivity is None else float(self._sensitivity)
        except (TypeError, ValueError) as error:
            raise RuntimeError(
                "ATLAS_WAKE_WORD_SENSITIVITY debe estar entre 0.0 y 1.0."
            ) from error

        if not 0.0 <= sensitivity <= 1.0:
            raise RuntimeError("ATLAS_WAKE_WORD_SENSITIVITY debe estar entre 0.0 y 1.0.")

        return sensitivity

    def _load_model_factory(self):
        try:
            from openwakeword.model import Model
        except ImportError as error:
            raise RuntimeError(
                "Dependencia no disponible: openwakeword."
            ) from error

        return Model

    def _ensure_initialized(self) -> None:
        if self._model is None:
            self.initialize()


class WakeWordEngine:
    """Wait for a local acoustic wake word, then optionally capture one phrase."""

    def __init__(
        self,
        speech_engine: SpeechEngineUseCase,
        provider: WakeWordProvider,
        wake_word: str = "Atlas",
        timeout_seconds: float = 30.0,
        capture_phrase_after_detection: bool = True,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not wake_word.strip():
            raise ValueError("La wake word no puede estar vacia.")

        if timeout_seconds <= 0:
            raise ValueError("El timeout debe ser mayor que cero.")

        self._speech_engine = speech_engine
        self._provider = provider
        self._wake_word = wake_word.strip()
        self._timeout_seconds = timeout_seconds
        self._capture_phrase_after_detection = capture_phrase_after_detection
        self._clock = clock

    @property
    def wake_word(self) -> str:
        """Configured wake word."""
        return self._wake_word

    @property
    def timeout_seconds(self) -> float:
        """Maximum wait time."""
        return self._timeout_seconds

    def wait_for_wake_word(
        self,
        status_sink: Callable[[str], None] | None = None,
    ) -> WakeWordDetectionResult:
        """Process PCM frames until wake word, timeout, cancellation, or error."""
        started = self._clock()
        frames_processed = 0
        frame_iterator = None

        try:
            self._provider.initialize()
            frame_iterator = self._speech_engine.iter_pcm_frames(
                sample_rate=self._provider.sample_rate,
                frame_length=self._provider.frame_length,
            )

            for frame in frame_iterator:
                elapsed = self._clock() - started

                if elapsed >= self._timeout_seconds:
                    return WakeWordDetectionResult(
                        wake_word=self._wake_word,
                        detected=False,
                        attempts=frames_processed,
                        elapsed_seconds=elapsed,
                        warnings=("timeout de wake word alcanzado",),
                    )

                frames_processed += 1

                if self._provider.process_frame(frame):
                    phrase = (
                        self._speech_engine.transcribe_once()
                        if self._capture_phrase_after_detection
                        else None
                    )
                    return WakeWordDetectionResult(
                        wake_word=self._wake_word,
                        detected=True,
                        attempts=frames_processed,
                        elapsed_seconds=self._clock() - started,
                        phrase=phrase,
                    )
        except KeyboardInterrupt:
            return WakeWordDetectionResult(
                wake_word=self._wake_word,
                detected=False,
                attempts=frames_processed,
                elapsed_seconds=self._clock() - started,
                cancelled=True,
                warnings=("espera de wake word cancelada",),
            )
        except RuntimeError as error:
            message = str(error)
            configuration_error = message.startswith("Wake word no configurada.") or (
                message.startswith("ATLAS_WAKE_WORD_MODEL_PATH")
                or message.startswith("ATLAS_WAKE_WORD_SENSITIVITY")
            )

            return WakeWordDetectionResult(
                wake_word=self._wake_word,
                detected=False,
                attempts=frames_processed,
                elapsed_seconds=self._clock() - started,
                configuration_error=configuration_error,
                warnings=(message,),
            )
        finally:
            close_iterator = getattr(frame_iterator, "close", None)

            if callable(close_iterator):
                close_iterator()

            self._provider.close()

        return WakeWordDetectionResult(
            wake_word=self._wake_word,
            detected=False,
            attempts=frames_processed,
            elapsed_seconds=self._clock() - started,
            warnings=("flujo de audio finalizado",),
        )


class WakeWordInteractionUseCase:
    """Interpret console commands that enter wake word mode."""

    _COMMANDS = {
        "atlas",
        "wake word",
        "activar wake word",
        "espera atlas",
        "modo atlas",
    }

    def __init__(
        self,
        wake_word_engine: WakeWordEngine,
    ) -> None:
        self._wake_word_engine = wake_word_engine

    def execute(
        self,
        prompt: str,
    ) -> str | None:
        """Run wake word mode for supported commands."""
        if self._normalize(prompt) not in self._COMMANDS:
            return None

        result = self._wake_word_engine.wait_for_wake_word()
        return self._format_result(result)

    def _format_result(
        self,
        result: WakeWordDetectionResult,
    ) -> str:
        if result.cancelled:
            return "Espera de wake word cancelada."

        if result.configuration_error:
            return result.warnings[0]

        if not result.detected:
            return (
                "No se detecto la palabra de activacion "
                f"{result.wake_word} antes del tiempo limite."
            )

        if result.phrase is None:
            return f"Wake word detectada: {result.wake_word}\n\nNo se capturo ninguna frase."

        if not result.phrase.completed:
            warning = "; ".join(result.phrase.warnings)
            return "\n".join(
                [
                    f"Wake word detectada: {result.wake_word}",
                    "",
                    f"No se pudo transcribir la frase: {warning}",
                    "",
                    "La orden transcrita no se ejecuto.",
                ]
            )

        return "\n".join(
            [
                f"Wake word detectada: {result.wake_word}",
                "",
                "Transcripcion:",
                result.phrase.text,
                "",
                f"Duracion de audio: {result.phrase.audio_duration_seconds:.1f} s",
                f"Tiempo de procesamiento: {result.phrase.processing_duration_seconds:.1f} s",
                f"Proveedor: {result.phrase.provider}",
                f"Microfono: {result.phrase.microphone_name}",
                "",
                "La orden transcrita no se ejecuto.",
            ]
        )

    def _normalize(
        self,
        text: str,
    ) -> str:
        normalized = unicodedata.normalize("NFKD", text.strip().lower())
        without_accents = "".join(
            character
            for character in normalized
            if unicodedata.category(character) != "Mn"
        )

        return " ".join(without_accents.split())
