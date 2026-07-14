"""Wake word detection use cases."""

from __future__ import annotations

from dataclasses import dataclass, field
import time
import unicodedata
from typing import Callable

from use_cases.speech_engine import (
    SpeechCaptureSettings,
    SpeechEngineUseCase,
    SpeechTranscriptionResult,
)


@dataclass(frozen=True)
class WakeWordAttempt:
    """Diagnostic information for one wake word attempt."""

    raw_text: str
    normalized_text: str
    audio_duration_seconds: float
    microphone_name: str
    accepted: bool
    reason: str


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
    attempts_log: tuple[WakeWordAttempt, ...] = field(default_factory=tuple)


class WakeWordEngine:
    """Wait for a local wake word, then capture one phrase."""

    def __init__(
        self,
        speech_engine: SpeechEngineUseCase,
        wake_word: str = "Atlas",
        timeout_seconds: float = 30.0,
        capture_phrase_after_detection: bool = True,
        capture_settings: SpeechCaptureSettings | None = None,
        max_empty_retries: int = 1,
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

        if max_empty_retries < 0:
            raise ValueError("El maximo de reintentos no puede ser negativo.")

        self._speech_engine = speech_engine
        self._wake_word = wake_word.strip()
        self._timeout_seconds = timeout_seconds
        self._capture_phrase_after_detection = capture_phrase_after_detection
        self._capture_settings = capture_settings or SpeechCaptureSettings(
            max_duration=4.0,
            initial_silence_timeout=7.0,
            trailing_silence=1.2,
            chunk_duration=0.1,
            speech_threshold=0.008,
            minimum_audio_duration=0.8,
        )
        self._max_empty_retries = max_empty_retries
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

    def wait_for_wake_word(
        self,
        status_sink: Callable[[str], None] | None = None,
    ) -> WakeWordDetectionResult:
        """Wait until the wake word is detected, then transcribe one phrase."""
        started = self._clock()
        attempts = 0
        empty_retries = 0
        attempts_log: list[WakeWordAttempt] = []

        while True:
            elapsed = self._clock() - started

            if elapsed >= self._timeout_seconds:
                return WakeWordDetectionResult(
                    wake_word=self._wake_word,
                    detected=False,
                    attempts=attempts,
                    elapsed_seconds=elapsed,
                    warnings=("timeout de wake word alcanzado",),
                    attempts_log=tuple(attempts_log),
                )

            attempts += 1
            candidate = self._transcribe_wake_word()

            if candidate.cancelled:
                return WakeWordDetectionResult(
                    wake_word=self._wake_word,
                    detected=False,
                    attempts=attempts,
                    elapsed_seconds=self._clock() - started,
                    cancelled=True,
                    warnings=candidate.warnings,
                    attempts_log=tuple(attempts_log),
                )

            accepted = candidate.completed and self._contains_wake_word(candidate.text)
            attempt = self._build_attempt(candidate, accepted)
            attempts_log.append(attempt)

            if status_sink is not None and not accepted:
                status_sink(self._format_rejection(attempt))

            if accepted:
                phrase = (
                    self._speech_engine.transcribe_once()
                    if self._capture_phrase_after_detection
                    else None
                )
                return WakeWordDetectionResult(
                    wake_word=self._wake_word,
                    detected=True,
                    attempts=attempts,
                    elapsed_seconds=self._clock() - started,
                    phrase=phrase,
                    attempts_log=tuple(attempts_log),
                )

            if self._is_empty_or_no_speech(candidate) and empty_retries < self._max_empty_retries:
                empty_retries += 1

                if status_sink is not None:
                    status_sink("No te he oido. Repite 'Atlas'.")

                continue

            if self._poll_interval_seconds > 0:
                self._sleeper(self._poll_interval_seconds)

    def _transcribe_wake_word(self) -> SpeechTranscriptionResult:
        try:
            return self._speech_engine.transcribe_once(
                capture_settings=self._capture_settings,
            )
        except TypeError:
            return self._speech_engine.transcribe_once()

    def _contains_wake_word(
        self,
        text: str,
    ) -> bool:
        normalized_text = self._normalize(text)
        normalized_wake_word = self._normalize(self._wake_word)
        tokens = normalized_text.split()

        return normalized_wake_word in tokens

    def _is_empty_or_no_speech(
        self,
        candidate: SpeechTranscriptionResult,
    ) -> bool:
        return candidate.no_speech_detected or not candidate.text.strip()

    def _build_attempt(
        self,
        candidate: SpeechTranscriptionResult,
        accepted: bool,
    ) -> WakeWordAttempt:
        normalized = self._normalize(candidate.text)
        reason = "wake word detectada"

        if not accepted:
            if candidate.no_speech_detected:
                reason = "no se detecto voz"
            elif not candidate.text.strip():
                reason = "transcripcion vacia"
            else:
                reason = "la transcripcion no contiene la wake word"

        return WakeWordAttempt(
            raw_text=candidate.text,
            normalized_text=normalized,
            audio_duration_seconds=candidate.audio_duration_seconds,
            microphone_name=candidate.microphone_name,
            accepted=accepted,
            reason=reason,
        )

    def _format_rejection(
        self,
        attempt: WakeWordAttempt,
    ) -> str:
        return "\n".join(
            [
                "Wake word no reconocida.",
                f"Texto bruto: {attempt.raw_text or '<vacio>'}",
                f"Texto normalizado: {attempt.normalized_text or '<vacio>'}",
                f"Duracion de audio: {attempt.audio_duration_seconds:.1f} s",
                f"Microfono: {attempt.microphone_name or '<desconocido>'}",
                f"Motivo: {attempt.reason}",
            ]
        )

    def _normalize(
        self,
        text: str,
    ) -> str:
        normalized = unicodedata.normalize("NFKD", text.strip().lower())
        characters: list[str] = []

        for character in normalized:
            category = unicodedata.category(character)

            if category == "Mn":
                continue

            if category[0] in {"L", "N"}:
                characters.append(character)
            else:
                characters.append(" ")

        return " ".join("".join(characters).split())


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
