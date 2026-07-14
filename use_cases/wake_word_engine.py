"""Wake word detection use cases."""

from __future__ import annotations

from dataclasses import dataclass, field
import time
import unicodedata
from typing import Callable

from use_cases.speech_engine import SpeechEngineUseCase, SpeechTranscriptionResult


@dataclass(frozen=True)
class WakeWordDetectionResult:
    """Result of waiting for a wake word and capturing one phrase."""

    wake_word: str
    detected: bool
    attempts: int
    elapsed_seconds: float
    phrase: SpeechTranscriptionResult | None = None
    cancelled: bool = False
    warnings: tuple[str, ...] = field(default_factory=tuple)


class WakeWordEngine:
    """Wait for a local wake word, then capture one phrase."""

    def __init__(
        self,
        speech_engine: SpeechEngineUseCase,
        wake_word: str = "Atlas",
        timeout_seconds: float = 30.0,
        poll_interval_seconds: float = 0.0,
        clock: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        if not wake_word.strip():
            raise ValueError("La wake word no puede estar vacia.")

        if timeout_seconds <= 0:
            raise ValueError("El timeout debe ser mayor que cero.")

        if poll_interval_seconds < 0:
            raise ValueError("El intervalo de espera no puede ser negativo.")

        self._speech_engine = speech_engine
        self._wake_word = wake_word.strip()
        self._timeout_seconds = timeout_seconds
        self._poll_interval_seconds = poll_interval_seconds
        self._clock = clock
        self._sleeper = sleeper

    @property
    def wake_word(self) -> str:
        """Configured wake word."""
        return self._wake_word

    @property
    def timeout_seconds(self) -> float:
        """Maximum wait time."""
        return self._timeout_seconds

    def wait_for_wake_word(self) -> WakeWordDetectionResult:
        """Wait until the wake word is detected, then transcribe one phrase."""
        started = self._clock()
        attempts = 0

        while True:
            elapsed = self._clock() - started

            if elapsed >= self._timeout_seconds:
                return WakeWordDetectionResult(
                    wake_word=self._wake_word,
                    detected=False,
                    attempts=attempts,
                    elapsed_seconds=elapsed,
                    warnings=("timeout de wake word alcanzado",),
                )

            attempts += 1
            candidate = self._speech_engine.transcribe_once()

            if candidate.cancelled:
                return WakeWordDetectionResult(
                    wake_word=self._wake_word,
                    detected=False,
                    attempts=attempts,
                    elapsed_seconds=self._clock() - started,
                    cancelled=True,
                    warnings=candidate.warnings,
                )

            if candidate.completed and self._contains_wake_word(candidate.text):
                phrase = self._speech_engine.transcribe_once()
                return WakeWordDetectionResult(
                    wake_word=self._wake_word,
                    detected=True,
                    attempts=attempts,
                    elapsed_seconds=self._clock() - started,
                    phrase=phrase,
                )

            if self._poll_interval_seconds > 0:
                self._sleeper(self._poll_interval_seconds)

    def _contains_wake_word(
        self,
        text: str,
    ) -> bool:
        normalized_text = self._normalize(text)
        normalized_wake_word = self._normalize(self._wake_word)
        tokens = normalized_text.split()

        return normalized_wake_word in tokens

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
