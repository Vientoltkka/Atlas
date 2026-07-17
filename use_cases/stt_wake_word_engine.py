"""STT-based provisional wake word detection for Atlas."""

from __future__ import annotations

from dataclasses import replace
import re
import time
import unicodedata
from typing import Callable

from use_cases.speech_engine import SpeechCaptureSettings, SpeechEngineUseCase
from use_cases.wake_word_engine import WakeWordDetectionResult


class SttWakeWordEngine:
    """Detect the Atlas wake word from short local STT captures."""

    _WAKE_WORD = "atlas"
    _WORD_PATTERN = re.compile(r"\batlas\b", re.IGNORECASE)

    def __init__(
        self,
        speech_engine: SpeechEngineUseCase,
        wake_word: str = "Atlas",
        capture_settings: SpeechCaptureSettings | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not wake_word.strip():
            raise ValueError("La wake word no puede estar vacia.")

        self._speech_engine = speech_engine
        self._wake_word = wake_word.strip()
        self._capture_settings = capture_settings or SpeechCaptureSettings(
            max_duration=2.2,
            initial_silence_timeout=1.4,
            trailing_silence=0.45,
            chunk_duration=0.1,
            speech_threshold=0.004,
            minimum_audio_duration=0.25,
        )
        self._clock = clock

    @property
    def wake_word(self) -> str:
        """Configured wake word."""
        return self._wake_word

    def wait_for_wake_word(
        self,
        status_sink: Callable[[str], None] | None = None,
    ) -> WakeWordDetectionResult:
        """Capture one short fragment and return whether it contains Atlas."""
        del status_sink
        started = self._clock()

        try:
            transcription = self._speech_engine.transcribe_once(self._capture_settings)
        except KeyboardInterrupt:
            return WakeWordDetectionResult(
                wake_word=self._wake_word,
                detected=False,
                attempts=1,
                elapsed_seconds=self._clock() - started,
                cancelled=True,
                warnings=("espera STT cancelada",),
            )
        except Exception as error:
            return WakeWordDetectionResult(
                wake_word=self._wake_word,
                detected=False,
                attempts=1,
                elapsed_seconds=self._clock() - started,
                warnings=(str(error),),
            )

        if transcription.cancelled:
            return WakeWordDetectionResult(
                wake_word=self._wake_word,
                detected=False,
                attempts=1,
                elapsed_seconds=self._clock() - started,
                cancelled=True,
                warnings=transcription.warnings,
            )

        if not transcription.completed or transcription.no_speech_detected:
            return WakeWordDetectionResult(
                wake_word=self._wake_word,
                detected=False,
                attempts=1,
                elapsed_seconds=self._clock() - started,
                phrase=transcription,
                warnings=transcription.warnings or ("sin activacion STT",),
            )

        activation = self.extract_activation(transcription.text)

        if activation is None:
            return WakeWordDetectionResult(
                wake_word=self._wake_word,
                detected=False,
                attempts=1,
                elapsed_seconds=self._clock() - started,
                phrase=transcription,
                warnings=("sin activacion STT",),
            )

        phrase = replace(transcription, text=activation)
        return WakeWordDetectionResult(
            wake_word=self._wake_word,
            detected=True,
            attempts=1,
            elapsed_seconds=self._clock() - started,
            phrase=phrase,
        )

    @classmethod
    def extract_activation(
        cls,
        text: str,
    ) -> str | None:
        """Return text after the full-word wake word, or an empty string."""
        normalized = cls.normalize_text(text)
        match = cls._WORD_PATTERN.search(normalized)

        if match is None:
            return None

        return normalized[match.end() :].strip(" ,.;:!?")

    @classmethod
    def normalize_text(
        cls,
        text: str,
    ) -> str:
        """Normalize STT text for wake-word matching."""
        normalized = unicodedata.normalize("NFKD", text.strip().lower())
        without_accents = "".join(
            character
            for character in normalized
            if unicodedata.category(character) != "Mn"
        )
        cleaned = re.sub(r"[^a-z0-9áéíóúüñ ,.;:!?¿¡'-]+", " ", without_accents)
        return " ".join(cleaned.split())
