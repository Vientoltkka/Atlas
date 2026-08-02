"""Local speech output use cases."""

from __future__ import annotations

from dataclasses import dataclass
import os
import threading
import time
from typing import Any, Protocol


@dataclass(frozen=True)
class SpeechOutputSettings:
    """Text-to-speech runtime settings."""

    rate: int = 180
    volume: float = 1.0
    voice: str | None = None


@dataclass(frozen=True)
class SpeechOutputMetrics:
    """Monotonic timings for one local TTS delivery."""

    synthesis_seconds: float = 0.0
    playback_seconds: float = 0.0
    barge_in_detected: bool = False
    tts_cancel_latency_ms: float = 0.0
    barge_in_to_stt_ms: float = 0.0
    tts_cancel_confirmed: bool = False


class SpeechOutputEngine(Protocol):
    """Minimal local text-to-speech interface."""

    def speak(
        self,
        text: str,
    ) -> None:
        """Speak text locally."""

    def warm_up(self) -> None:
        """Initialize speech resources before the first response."""

    def cancel(self) -> bool:
        """Cancel active playback once and report whether it was active."""

    def close(self) -> None:
        """Release speech resources."""


class Pyttsx3SpeechOutputEngine:
    """Local pyttsx3 speech output backed by Windows SAPI5."""

    _CANCEL_CONFIRM_TIMEOUT_SECONDS = 0.2

    def __init__(
        self,
        settings: SpeechOutputSettings | None = None,
    ) -> None:
        self._settings = settings or SpeechOutputSettings()
        self._engine: Any | None = None
        self._engine_lock = threading.RLock()
        self._cancel_requested = threading.Event()
        self._playback_active = False
        self._external_loop_active = False

    @classmethod
    def from_environment(cls) -> "Pyttsx3SpeechOutputEngine":
        """Build the engine from optional ATLAS_TTS_* environment variables."""
        return cls(
            SpeechOutputSettings(
                rate=_read_int("ATLAS_TTS_RATE", 175, minimum=80, maximum=320),
                volume=_read_float(
                    "ATLAS_TTS_VOLUME",
                    1.0,
                    minimum=0.0,
                    maximum=1.0,
                ),
                voice=_read_optional_text("ATLAS_TTS_VOICE"),
            )
        )

    def speak(
        self,
        text: str,
    ) -> None:
        """Speak text with the configured local voice."""
        self.speak_with_metrics(text)

    def speak_with_metrics(
        self,
        text: str,
    ) -> SpeechOutputMetrics:
        """Speak text and return real synthesis/playback timings."""
        clean_text = text.strip()

        if not clean_text:
            return SpeechOutputMetrics()

        self._cancel_requested.clear()
        synthesis_started = time.monotonic()
        engine = self._load_engine()

        try:
            self._stop_if_busy(engine)
            engine.say(clean_text)
            synthesis_seconds = time.monotonic() - synthesis_started
            playback_started = time.monotonic()
            use_external_loop = self._supports_external_loop(engine)
            with self._engine_lock:
                self._playback_active = True
                self._external_loop_active = use_external_loop
            cancel_confirmed = self._run_playback(engine, use_external_loop)
            playback_seconds = time.monotonic() - playback_started
        except Exception as error:
            self._discard_failed_engine(engine)
            raise RuntimeError(str(error)) from error
        finally:
            with self._engine_lock:
                self._playback_active = False
                self._external_loop_active = False
                if self._engine is engine:
                    if not self._cancel_requested.is_set():
                        self._release_engine(engine)
                    self._engine = None

        return SpeechOutputMetrics(
            synthesis_seconds=synthesis_seconds,
            playback_seconds=playback_seconds,
            tts_cancel_confirmed=cancel_confirmed,
        )

    def warm_up(self) -> None:
        """Validate pyttsx3 without acquiring an audio device before capture."""
        try:
            import pyttsx3  # noqa: F401
        except ImportError as error:
            raise RuntimeError("Dependencia no disponible: pyttsx3.") from error

    def cancel(self) -> bool:
        """Stop active playback once."""
        with self._engine_lock:
            if (
                self._engine is None
                or not self._playback_active
                or self._cancel_requested.is_set()
            ):
                return False
            self._cancel_requested.set()
            engine = self._engine
            external_loop_active = self._external_loop_active

        if not external_loop_active:
            stop = getattr(engine, "stop", None)
            if callable(stop):
                stop()
        return True

    def close(self) -> None:
        """Stop any pending speech and release a warmed engine."""
        if self.cancel():
            return

        with self._engine_lock:
            engine = self._engine
            self._engine = None
            self._playback_active = False
            self._external_loop_active = False

        if engine is not None:
            self._release_engine(engine)

    def _load_engine(self):
        with self._engine_lock:
            if self._engine is not None:
                return self._engine

            try:
                import pyttsx3
            except ImportError as error:
                raise RuntimeError("Dependencia no disponible: pyttsx3.") from error

            try:
                engine = pyttsx3.init()
                engine.setProperty("rate", self._settings.rate)
                engine.setProperty("volume", self._settings.volume)
                self._select_voice(engine, self._settings.voice)
            except Exception as error:
                raise RuntimeError(f"No se pudo inicializar TTS local: {error}") from error

            self._engine = engine
            return engine

    def _supports_external_loop(
        self,
        engine,
    ) -> bool:
        return all(
            callable(getattr(engine, name, None))
            for name in ("startLoop", "iterate", "endLoop", "isBusy")
        )

    def _run_playback(
        self,
        engine,
        use_external_loop: bool,
    ) -> bool:
        if not use_external_loop:
            engine.runAndWait()
            is_busy = getattr(engine, "isBusy", None)
            return bool(
                self._cancel_requested.is_set()
                and callable(is_busy)
                and not is_busy()
            )

        cancel_started: float | None = None
        cancel_confirmed = False
        engine.startLoop(False)
        try:
            while engine.isBusy():
                if self._cancel_requested.is_set() and cancel_started is None:
                    engine.stop()
                    cancel_started = time.monotonic()

                engine.iterate()

                if cancel_started is not None:
                    if not engine.isBusy():
                        cancel_confirmed = True
                        break
                    if (
                        time.monotonic() - cancel_started
                        >= self._CANCEL_CONFIRM_TIMEOUT_SECONDS
                    ):
                        break

                time.sleep(0.01)
        finally:
            engine.endLoop()

        return cancel_confirmed

    def _stop_if_busy(
        self,
        engine,
    ) -> None:
        is_busy = getattr(engine, "isBusy", None)

        if callable(is_busy) and is_busy():
            stop = getattr(engine, "stop", None)

            if callable(stop):
                stop()

    def _discard_failed_engine(
        self,
        engine,
    ) -> None:
        self._release_engine(engine)
        self._engine = None

    def _release_engine(
        self,
        engine,
    ) -> None:
        stop = getattr(engine, "stop", None)

        if callable(stop):
            try:
                stop()
            except Exception:
                pass

    def _select_voice(
        self,
        engine,
        requested_voice: str | None,
    ) -> None:
        voices = engine.getProperty("voices") or []

        if not requested_voice:
            self._select_preferred_spanish_voice(engine, voices)
            return

        normalized_requested = requested_voice.casefold()

        for voice in voices:
            voice_id = str(getattr(voice, "id", ""))
            voice_name = str(getattr(voice, "name", ""))

            if (
                voice_id.casefold() == normalized_requested
                or normalized_requested in voice_name.casefold()
            ):
                engine.setProperty("voice", voice_id)
                return

    def _select_preferred_spanish_voice(
        self,
        engine,
        voices,
    ) -> None:
        preferred_terms = (
            "spanish",
            "espanol",
            "castellano",
            "sabina",
            "helena",
            "pablo",
            "laura",
        )

        for voice in voices:
            voice_id = str(getattr(voice, "id", ""))
            voice_name = str(getattr(voice, "name", ""))
            haystack = f"{voice_id} {voice_name}".casefold()

            if any(term in haystack for term in preferred_terms):
                engine.setProperty("voice", voice_id)
                return


def _read_optional_text(
    name: str,
) -> str | None:
    value = os.getenv(name, "").strip()
    return value or None


def _read_int(
    name: str,
    default: int,
    minimum: int,
    maximum: int,
) -> int:
    raw = os.getenv(name, "").strip()

    if not raw:
        return default

    try:
        value = int(raw)
    except ValueError:
        return default

    return min(max(value, minimum), maximum)


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
