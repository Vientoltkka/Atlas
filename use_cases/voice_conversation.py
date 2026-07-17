"""Continuous voice conversation use case."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime
import json
import os
import re
import time
import unicodedata
from typing import Callable
from uuid import uuid4

from use_cases.speech_engine import (
    SpeechCaptureSettings,
    SpeechEngineUseCase,
    SpeechTranscriptionResult,
)
from use_cases.speech_output_engine import SpeechOutputEngine
from use_cases.wake_word_engine import WakeWordEngine


@dataclass
class VoiceConversationTurn:
    """One voice conversation turn."""

    number: int
    transcription: str
    response: str
    audio_duration_seconds: float
    processing_duration_seconds: float
    success: bool
    timestamp: float
    error: str = ""


@dataclass
class VoiceConversationSession:
    """Voice conversation session state."""

    session_id: str
    started_at: float
    total_turns: int = 0
    successful_turns: int = 0
    failed_turns: int = 0
    consecutive_no_speech: int = 0
    active: bool = True
    ended_reason: str = ""
    transcript_history: list[str] = field(default_factory=list)
    response_history: list[str] = field(default_factory=list)
    turns: list[VoiceConversationTurn] = field(default_factory=list)
    summary: str = ""


@dataclass
class VoiceConversationResult:
    """Result returned after a voice conversation session."""

    session: VoiceConversationSession
    messages: list[str]


class VoiceConversationUseCase:
    """Run a controlled continuous voice conversation."""

    _CODE_BLOCK_PATTERN = re.compile(r"```(?:[a-zA-Z0-9_-]+)?\s*(.*?)```", re.DOTALL)
    _COMMANDS = {
        "activa conversacion por voz",
        "inicia modo voz",
        "conversacion por voz",
        "start voice conversation",
    }
    _CLOSE_COMMANDS = {
        "termina",
        "terminar",
        "terminar conversacion",
        "cancelar",
        "salir",
        "adios",
        "adios atlas",
        "stop",
        "stop listening",
        "end conversation",
    }
    _CRITICAL_ERROR_HINTS = (
        "permiso",
        "deneg",
        "microfono inexistente",
        "no hay microfonos",
        "modelo no disponible",
        "dependencia no disponible",
        "cancelada",
        "cancelado",
        "dispositivo incompatible",
        "fallo al abrir stream",
        "error opening inputstream",
        "paerrorcode -9999",
        "wdm-ks",
    )
    _LOW_VALUE_NOISE_PHRASES = {
        "nice",
        "thanks",
        "thank you",
    }
    _EDGE_PUNCTUATION = " \t\r\n.,;:!?¿¡\"'`()[]{}"
    _SPANISH_RESPONSE_INSTRUCTION = (
        "Responde en español, de forma natural y concisa."
    )

    def __init__(
        self,
        speech_engine: SpeechEngineUseCase,
        wake_word_engine: WakeWordEngine | None,
        speech_output_engine: SpeechOutputEngine | None = None,
        conversation_idle_timeout: float = 25.0,
        max_session_duration: float = 600.0,
        max_turns: int = 20,
        max_consecutive_no_speech: int = 2,
        wake_capture_settings: SpeechCaptureSettings | None = None,
        turn_capture_settings: SpeechCaptureSettings | None = None,
        diagnostics_enabled: bool | None = None,
        clock: Callable[[], float] = time.monotonic,
        now_provider: Callable[[], datetime] | None = None,
        session_id_factory: Callable[[], str] | None = None,
    ) -> None:
        if conversation_idle_timeout <= 0:
            raise ValueError("El timeout de inactividad debe ser mayor que cero.")
        if max_session_duration <= 0:
            raise ValueError("La duracion maxima debe ser mayor que cero.")
        if max_turns <= 0:
            raise ValueError("El maximo de turnos debe ser mayor que cero.")
        if max_consecutive_no_speech <= 0:
            raise ValueError("El maximo de silencios debe ser mayor que cero.")

        self._speech_engine = speech_engine
        self._wake_word_engine = wake_word_engine
        self._speech_output_engine = speech_output_engine
        self._conversation_idle_timeout = conversation_idle_timeout
        self._max_session_duration = max_session_duration
        self._max_turns = max_turns
        self._max_consecutive_no_speech = max_consecutive_no_speech
        self._wake_capture_settings = wake_capture_settings or SpeechCaptureSettings(
            max_duration=4.0,
            initial_silence_timeout=7.0,
            trailing_silence=1.2,
            chunk_duration=0.1,
            speech_threshold=0.008,
            minimum_audio_duration=0.8,
        )
        self._turn_capture_settings = turn_capture_settings or SpeechCaptureSettings(
            max_duration=10.0,
            initial_silence_timeout=5.0,
            trailing_silence=1.0,
            chunk_duration=0.1,
            speech_threshold=_read_float("ATLAS_VOICE_RMS_THRESHOLD", 0.004, 0.001, 0.05),
            minimum_audio_duration=0.3,
        )
        self._diagnostics_enabled = (
            diagnostics_enabled
            if diagnostics_enabled is not None
            else _read_bool("ATLAS_VOICE_DIAGNOSTICS", False)
        )
        self._clock = clock
        self._now_provider = now_provider or (lambda: datetime.now().astimezone())
        self._session_id_factory = session_id_factory or (lambda: str(uuid4()))

    def execute(
        self,
        prompt: str,
        process_text: Callable[[str], str],
        status_sink: Callable[[str], None] | None = None,
    ) -> VoiceConversationResult | None:
        """Run voice conversation mode for explicit activation commands."""
        if self._normalize(prompt) not in self._COMMANDS:
            return None
        if self._wake_word_engine is None:
            return self.execute_manual(process_text, status_sink)

        messages: list[str] = []

        def emit(message: str, diagnostic: bool = True) -> None:
            messages.append(message)

            if status_sink is not None and (self._diagnostics_enabled or not diagnostic):
                status_sink(message)

        session = VoiceConversationSession(
            session_id=self._session_id_factory(),
            started_at=self._clock(),
        )

        try:
            emit("Estado: inicializando.")
            emit("Inicializando microfono y modelo...")
            self._warm_up()
            active_microphone = self._active_microphone()
            emit("Microfono activo:")
            emit(f"{active_microphone.index} - {active_microphone.name}")
            self._prepare_stream()
            emit("Esperando la palabra de activacion...")
            emit("Estado: listo para wake word.")
            emit('Di "Atlas" ahora...')
            wake_word = self._wake_word_engine.wait_for_wake_word(
                status_sink=emit,
            )

            if wake_word.cancelled:
                self._end_session(session, "cancelled", "Conversacion cancelada.")
                emit("Conversacion cancelada.")
                return VoiceConversationResult(session=session, messages=messages)

            if wake_word.configuration_error:
                self._end_session(
                    session,
                    "configuration_error",
                    wake_word.warnings[0] if wake_word.warnings else "Wake word no configurada.",
                )
                emit("Estado: configuration_error.")
                emit(session.summary)
                return VoiceConversationResult(session=session, messages=messages)

            if not wake_word.detected:
                self._end_session(
                    session,
                    "wake_word_timeout",
                    'No se detectó "Atlas". Modo de voz finalizado.',
                )
                emit("Estado: conversacion finalizada.")
                emit('No se detectó "Atlas". Modo de voz finalizado.')
                return VoiceConversationResult(session=session, messages=messages)

            emit("Estado: wake word detectada.")
            emit("Wake word detectada.")
            emit("Estado: conversacion activa.")
            emit("Conversacion iniciada.")
            self._run_turns(
                session,
                process_text,
                emit,
                enforce_spanish_response=False,
                retry_initial_no_speech=True,
            )
        except KeyboardInterrupt:
            self._end_session(session, "cancelled", "Conversacion cancelada.")
            emit("Conversacion cancelada.")
        except Exception as error:
            self._end_session(
                session,
                "critical_error",
                f"Conversacion finalizada por error critico: {error}",
            )
            emit(session.summary)
        finally:
            self._close_resources()

        return VoiceConversationResult(session=session, messages=messages)

    def execute_manual(
        self,
        process_text: Callable[[str], str],
        status_sink: Callable[[str], None] | None = None,
        typed_input: Callable[[], str | None] | None = None,
    ) -> VoiceConversationResult:
        """Run voice conversation mode without wake-word activation."""
        messages: list[str] = []

        def emit(message: str, diagnostic: bool = True) -> None:
            messages.append(message)

            if status_sink is not None and (self._diagnostics_enabled or not diagnostic):
                status_sink(message)

        session = VoiceConversationSession(
            session_id=self._session_id_factory(),
            started_at=self._clock(),
        )

        try:
            emit("Estado: inicializando.")
            active_microphone = self._active_microphone()
            self._validate_active_microphone()
            emit("Microfono activo:")
            emit(f"{active_microphone.index} - {active_microphone.name}")
            self._calibrate_turn_capture_settings()
            emit("Inicializando microfono, modelo y TTS local...")
            self._warm_up()
            emit("Estado: conversacion activa.")
            emit("Conversacion de voz manual iniciada.")
            self._run_turns(
                session,
                process_text,
                emit,
                typed_input,
                enforce_spanish_response=True,
                retry_initial_no_speech=False,
            )
        except KeyboardInterrupt:
            self._end_session(session, "cancelled", "Conversacion cancelada.")
            emit("Conversacion cancelada.")
        except Exception as error:
            self._end_session(
                session,
                "critical_error",
                f"Conversacion finalizada por error critico: {error}",
            )
            emit(session.summary)
        finally:
            self._close_resources()

        return VoiceConversationResult(session=session, messages=messages)

    def warm_up_resources(self) -> None:
        """Load speech and TTS resources for a long-running assistant session."""
        self._active_microphone()
        self._validate_active_microphone()
        self._calibrate_turn_capture_settings()
        self._warm_up()

    def close_resources(self) -> None:
        """Release speech and TTS resources owned by the voice flow."""
        self._close_resources()

    def discard_residual_audio(self) -> None:
        """Open and drain the input stream before returning to wake-word mode."""
        self._prepare_turn_stream()

    def is_close_command(
        self,
        text: str,
    ) -> bool:
        """Return True when text asks Atlas to stop listening."""
        return self._is_close_command(text)

    def execute_assistant_turn(
        self,
        process_text: Callable[[str], str],
        status_sink: Callable[[str], None] | None = None,
        initial_text: str | None = None,
    ) -> VoiceConversationResult:
        """Process one assistant turn after an external wake-word detection."""
        messages: list[str] = []

        def emit(message: str, diagnostic: bool = True) -> None:
            messages.append(message)

            if status_sink is not None and (self._diagnostics_enabled or not diagnostic):
                status_sink(message)

        session = VoiceConversationSession(
            session_id=self._session_id_factory(),
            started_at=self._clock(),
        )

        if initial_text is not None and initial_text.strip():
            return self._execute_assistant_text_turn(
                session,
                initial_text,
                process_text,
                emit,
            )

        transcription = self._transcribe_turn()

        if transcription.cancelled:
            self._end_session(session, "cancelled", "Conversacion cancelada.")
            emit("Conversacion cancelada.", diagnostic=True)
            return VoiceConversationResult(session=session, messages=messages)

        if transcription.no_speech_detected:
            session.failed_turns += 1
            session.consecutive_no_speech += 1
            self._end_session(session, "no_speech", "No se detecto ninguna frase.")
            emit("No se detecto ninguna frase.", diagnostic=True)
            return VoiceConversationResult(session=session, messages=messages)

        if not transcription.completed:
            session.failed_turns += 1
            error = "; ".join(transcription.warnings) or transcription.summary
            reason = "critical_error" if self._is_critical_error(error) else "recoverable_error"
            self._end_session(
                session,
                reason,
                f"No se pudo procesar el turno: {error}",
            )
            emit(session.summary, diagnostic=True)
            return VoiceConversationResult(session=session, messages=messages)

        text = self._accepted_transcription_text(
            transcription,
            trim_edge_punctuation=True,
        )

        if text is None:
            session.failed_turns += 1
            session.consecutive_no_speech += 1
            self._end_session(session, "no_speech", "No entendi la frase.")
            emit("No entendi la frase.", diagnostic=True)
            return VoiceConversationResult(session=session, messages=messages)

        if self._is_close_command(text):
            self._end_session(session, "explicit_close", "Conversacion finalizada.")
            emit("Conversacion finalizada.", diagnostic=True)
            return VoiceConversationResult(session=session, messages=messages)

        self._process_successful_turn(
            session,
            transcription,
            text,
            process_text,
            emit,
            enforce_spanish_response=True,
        )

        if session.active:
            self._end_session(session, "turn_completed", "Turno completado.")

        return VoiceConversationResult(session=session, messages=messages)

    def _execute_assistant_text_turn(
        self,
        session: VoiceConversationSession,
        text: str,
        process_text: Callable[[str], str],
        emit: Callable[[str, bool], None],
    ) -> VoiceConversationResult:
        cleaned_text = self._clean_transcription_text(
            text,
            trim_edge_punctuation=True,
        )

        if not cleaned_text:
            self._end_session(session, "no_speech", "No entendi la frase.")
            emit("No entendi la frase.", diagnostic=True)
            return VoiceConversationResult(session=session, messages=[])

        if self._is_close_command(cleaned_text):
            self._end_session(session, "explicit_close", "Conversacion finalizada.")
            emit("Conversacion finalizada.", diagnostic=True)
            return VoiceConversationResult(session=session, messages=[])

        transcription = SpeechTranscriptionResult(
            text=cleaned_text,
            language="es",
            audio_duration_seconds=0.0,
            processing_duration_seconds=0.0,
            provider="stt-wake-word",
            microphone_name="wake-word",
            completed=True,
            cancelled=False,
            no_speech_detected=False,
        )
        self._process_successful_turn(
            session,
            transcription,
            cleaned_text,
            process_text,
            emit,
            enforce_spanish_response=True,
        )

        if session.active:
            self._end_session(session, "turn_completed", "Turno completado.")

        return VoiceConversationResult(session=session, messages=[])

    def _warm_up(self) -> None:
        warm_up = getattr(self._speech_engine, "warm_up", None)

        if callable(warm_up):
            warm_up()
        else:
            self._speech_engine.default_microphone()

        if self._speech_output_engine is not None:
            warm_up_output = getattr(self._speech_output_engine, "warm_up", None)

            if callable(warm_up_output):
                warm_up_output()

    def _active_microphone(self):
        active_microphone = getattr(self._speech_engine, "active_microphone", None)

        if callable(active_microphone):
            return active_microphone()

        return self._speech_engine.default_microphone()

    def _prepare_stream(self) -> None:
        prepare_stream = getattr(self._speech_engine, "prepare_stream", None)

        if callable(prepare_stream):
            prepare_stream(self._wake_capture_settings)

    def _prepare_turn_stream(self) -> None:
        prepare_stream = getattr(self._speech_engine, "prepare_stream", None)

        if callable(prepare_stream):
            prepare_stream(self._turn_capture_settings)

    def _run_turns(
        self,
        session: VoiceConversationSession,
        process_text: Callable[[str], str],
        emit: Callable[[str, bool], None],
        typed_input: Callable[[], str | None] | None = None,
        enforce_spanish_response: bool = False,
        retry_initial_no_speech: bool = False,
    ) -> None:
        first_listen = True
        last_activity = self._clock()

        while session.active:
            typed_command = typed_input() if typed_input is not None else None

            if typed_command is not None and self._is_close_command(typed_command):
                self._end_session(
                    session,
                    "explicit_close",
                    "Conversacion finalizada.",
                )
                emit("Estado: conversacion finalizada.")
                emit("Conversacion finalizada.")
                break

            now = self._clock()

            if now - last_activity >= self._conversation_idle_timeout:
                self._end_session(
                    session,
                    "idle_timeout",
                    "Conversacion finalizada por inactividad.",
                )
                emit(session.summary)
                break

            if now - session.started_at >= self._max_session_duration:
                self._end_session(
                    session,
                    "max_session_duration",
                    "Conversacion finalizada por duracion maxima.",
                )
                emit(session.summary)
                break

            if session.total_turns >= self._max_turns:
                self._end_session(
                    session,
                    "max_turns",
                    "Conversacion finalizada por maximo de turnos.",
                )
                emit(session.summary)
                break

            emit("Esperando voz...", diagnostic=False)
            transcription = self._transcribe_turn()

            if (
                retry_initial_no_speech
                and first_listen
                and self._should_retry_initial_listen(transcription)
            ):
                emit("No se detecto voz. Vuelvo a escuchar...", diagnostic=True)
                transcription = self._transcribe_turn()

            first_listen = False

            if transcription.cancelled:
                self._end_session(session, "cancelled", "Conversacion cancelada.")
                emit("Estado: conversacion finalizada.", diagnostic=True)
                emit("Conversacion cancelada.", diagnostic=True)
                break

            if transcription.no_speech_detected:
                session.consecutive_no_speech += 1
                session.failed_turns += 1
                last_activity = self._clock()

                if (
                    session.consecutive_no_speech
                    >= self._max_consecutive_no_speech
                ):
                    self._end_session(
                        session,
                        "idle_timeout",
                        "Conversacion finalizada por inactividad.",
                    )
                    emit("Estado: conversacion finalizada.", diagnostic=True)
                    emit(session.summary, diagnostic=True)
                    break

                emit("No se detecto ninguna frase.", diagnostic=True)
                continue

            if not transcription.completed:
                session.failed_turns += 1
                last_activity = self._clock()
                error = "; ".join(transcription.warnings) or transcription.summary

                if self._is_critical_error(error):
                    self._end_session(
                        session,
                        "critical_error",
                        f"Conversacion finalizada por error critico: {error}",
                    )
                    emit("Estado: conversacion finalizada.", diagnostic=True)
                    emit(session.summary, diagnostic=True)
                    break

                emit(f"No se pudo procesar el turno: {error}", diagnostic=True)
                continue

            accepted_text = self._accepted_transcription_text(
                transcription,
                trim_edge_punctuation=enforce_spanish_response,
            )

            if accepted_text is None:
                session.consecutive_no_speech += 1
                session.failed_turns += 1
                last_activity = self._clock()

                if session.consecutive_no_speech >= self._max_consecutive_no_speech:
                    self._end_session(
                        session,
                        "idle_timeout",
                        "Conversacion finalizada por inactividad.",
                    )
                    emit("Estado: conversacion finalizada.", diagnostic=True)
                    emit(session.summary, diagnostic=True)
                    break

                emit("No entendí la frase. Inténtalo de nuevo.", diagnostic=False)
                continue

            session.consecutive_no_speech = 0
            text = accepted_text
            last_activity = self._clock()

            if self._is_close_command(text):
                self._end_session(
                    session,
                    "explicit_close",
                    "Conversacion finalizada.",
                )
                emit("Estado: conversacion finalizada.", diagnostic=True)
                emit("Conversacion finalizada.", diagnostic=True)
                break

            self._process_successful_turn(
                session,
                transcription,
                text,
                process_text,
                emit,
                enforce_spanish_response,
            )
            last_activity = self._clock()

    def _process_successful_turn(
        self,
        session: VoiceConversationSession,
        transcription: SpeechTranscriptionResult,
        text: str,
        process_text: Callable[[str], str],
        emit: Callable[[str, bool], None],
        enforce_spanish_response: bool = False,
    ) -> None:
        turn_number = session.total_turns + 1

        try:
            deterministic_response = (
                self._deterministic_datetime_response(text)
                if enforce_spanish_response
                else None
            )
            response = deterministic_response or process_text(
                self._prompt_for_voice(text, enforce_spanish_response)
            )
        except Exception as error:
            response = ""
            session.failed_turns += 1
            session.total_turns += 1
            session.transcript_history.append(text)
            turn = VoiceConversationTurn(
                number=turn_number,
                transcription=text,
                response="",
                audio_duration_seconds=transcription.audio_duration_seconds,
                processing_duration_seconds=transcription.processing_duration_seconds,
                success=False,
                timestamp=self._clock(),
                error=str(error),
            )
            session.turns.append(turn)

            if self._is_critical_error(str(error)):
                self._end_session(
                    session,
                    "critical_error",
                    f"Conversacion finalizada por error critico: {error}",
                )
                emit(session.summary, diagnostic=True)
            else:
                emit(f"No se pudo procesar el turno: {error}", diagnostic=True)

            return

        response = self._format_response_for_voice(response or "")
        should_speak_response = bool(response.strip())

        if not response:
            response = "Respuesta vacia."

        session.total_turns += 1
        session.successful_turns += 1
        session.transcript_history.append(text)
        session.response_history.append(response)
        session.turns.append(
            VoiceConversationTurn(
                number=turn_number,
                transcription=text,
                response=response,
                audio_duration_seconds=transcription.audio_duration_seconds,
                processing_duration_seconds=transcription.processing_duration_seconds,
                success=True,
                timestamp=self._clock(),
            )
        )
        if self._diagnostics_enabled:
            emit("Transcripcion:", diagnostic=True)
            emit(text, diagnostic=True)
            emit("Respuesta:", diagnostic=True)
            emit(response, diagnostic=True)
        else:
            emit(f"Tú: {text}", diagnostic=False)
            emit(f"Atlas: {response}", diagnostic=False)

        if should_speak_response:
            self._speak_response(response, emit)
            time.sleep(0.2)

    def _speak_response(
        self,
        response: str,
        emit: Callable[[str, bool], None],
    ) -> None:
        if self._speech_output_engine is None or not response.strip():
            return

        try:
            emit("TTS iniciado", diagnostic=True)
            self._speech_output_engine.speak(response)
            emit("TTS finalizado", diagnostic=True)
        except Exception as error:
            emit(f"TTS falló: {error}", diagnostic=True)

    def _calibrate_turn_capture_settings(self) -> None:
        calibrate = getattr(self._speech_engine, "calibrate_noise_threshold", None)

        if not callable(calibrate):
            return

        try:
            threshold = calibrate(self._turn_capture_settings, 0.5)
        except Exception:
            return

        if threshold is None:
            return

        self._turn_capture_settings = replace(
            self._turn_capture_settings,
            speech_threshold=float(threshold),
        )

    def _validate_active_microphone(self) -> None:
        validate = getattr(self._speech_engine, "validate_active_microphone", None)

        if callable(validate):
            validate(self._turn_capture_settings)

    def _close_resources(self) -> None:
        close_speech = getattr(self._speech_engine, "close", None)

        if callable(close_speech):
            close_speech()

        if self._speech_output_engine is not None:
            try:
                self._speech_output_engine.close()
            except Exception:
                pass

    def _transcribe_turn(self) -> SpeechTranscriptionResult:
        try:
            return self._speech_engine.transcribe_once(self._turn_capture_settings)
        except TypeError:
            return self._speech_engine.transcribe_once()

    def _accepted_transcription_text(
        self,
        transcription: SpeechTranscriptionResult,
        trim_edge_punctuation: bool = False,
    ) -> str | None:
        text = self._clean_transcription_text(
            transcription.text,
            trim_edge_punctuation=trim_edge_punctuation,
        )

        if not text:
            return None

        useful = self._useful_text(text)
        confidence = self._confidence(transcription.average_log_probability)
        no_speech_probability = transcription.no_speech_probability

        if (
            no_speech_probability is not None
            and no_speech_probability
            > self._stt_threshold("ATLAS_STT_MAX_NO_SPEECH_PROBABILITY", 0.65)
        ):
            return None

        if (
            confidence is not None
            and confidence < self._stt_threshold("ATLAS_STT_MIN_CONFIDENCE", 0.35)
            and len(useful) < 8
        ):
            return None

        if (
            self._normalize(text) in self._LOW_VALUE_NOISE_PHRASES
            and (
                confidence is None
                or confidence < 0.75
                or (
                    no_speech_probability is not None
                    and no_speech_probability > 0.35
                )
            )
        ):
            return None

        if len(useful) < 2 and not self._is_close_command(text):
            return None

        return text

    def _clean_transcription_text(
        self,
        text: str,
        trim_edge_punctuation: bool = False,
    ) -> str:
        cleaned = " ".join(text.strip().split())

        if trim_edge_punctuation:
            return cleaned.strip(self._EDGE_PUNCTUATION)

        return cleaned

    def _useful_text(
        self,
        text: str,
    ) -> str:
        normalized = self._normalize(text)
        return re.sub(r"[^a-z0-9]+", "", normalized)

    def _confidence(
        self,
        average_log_probability: float | None,
    ) -> float | None:
        if average_log_probability is None:
            return None

        import math

        return max(0.0, min(1.0, math.exp(average_log_probability)))

    def _stt_threshold(
        self,
        name: str,
        default: float,
    ) -> float:
        raw = os.getenv(name, "").strip()

        if not raw:
            return default

        try:
            return float(raw)
        except ValueError:
            return default

    def _prompt_for_voice(
        self,
        text: str,
        enforce_spanish_response: bool,
    ) -> str:
        if not enforce_spanish_response or self._explicitly_requests_other_language(text):
            return text

        return f"{text}\n\n{self._SPANISH_RESPONSE_INSTRUCTION}"

    def _explicitly_requests_other_language(
        self,
        text: str,
    ) -> bool:
        normalized = self._normalize(text)
        return any(
            phrase in normalized
            for phrase in (
                "en ingles",
                "in english",
                "respond in english",
                "answer in english",
            )
        )

    def _deterministic_datetime_response(
        self,
        text: str,
    ) -> str | None:
        normalized = self._normalize(text)
        asks_time = self._asks_current_time(normalized)
        asks_date = self._asks_current_date(normalized)

        if not asks_time and not asks_date:
            return None

        now = self._now_provider()

        if asks_time and asks_date:
            return (
                f"Son las {self._time_words(now)} del "
                f"{self._date_words(now)}."
            )

        if asks_time:
            return f"Son las {self._time_words(now)}."

        return f"Hoy es {self._date_words(now)}."

    def _asks_current_time(
        self,
        normalized: str,
    ) -> bool:
        if any(
            phrase in normalized
            for phrase in (
                "que hora es",
                "dime la hora",
                "hora actual",
            )
        ):
            return True

        words = normalized.split()
        return (
            "ahora" in words
            and "es" in words
            and (
                normalized.startswith("que ")
                or "actual" in words
                or "mismo" in words
            )
        )

    def _asks_current_date(
        self,
        normalized: str,
    ) -> bool:
        return any(
            phrase in normalized
            for phrase in (
                "que dia es",
                "cual es la fecha",
                "fecha actual",
                "fecha y hora",
                "que fecha es",
            )
        )

    def _time_words(
        self,
        now: datetime,
    ) -> str:
        hour = now.hour
        minute = now.minute
        period = "de la madrugada"

        if 6 <= hour < 12:
            period = "de la mañana"
        elif 12 <= hour < 20:
            period = "de la tarde"
        elif hour >= 20:
            period = "de la noche"

        spoken_hour = hour % 12

        if spoken_hour == 0:
            spoken_hour = 12

        return (
            f"{self._number_words(spoken_hour)} y "
            f"{self._number_words(minute)} {period}"
        )

    def _date_words(
        self,
        now: datetime,
    ) -> str:
        weekdays = (
            "lunes",
            "martes",
            "miércoles",
            "jueves",
            "viernes",
            "sábado",
            "domingo",
        )
        months = (
            "enero",
            "febrero",
            "marzo",
            "abril",
            "mayo",
            "junio",
            "julio",
            "agosto",
            "septiembre",
            "octubre",
            "noviembre",
            "diciembre",
        )

        return (
            f"{weekdays[now.weekday()]}, {now.day} de "
            f"{months[now.month - 1]} de {now.year}"
        )

    def _number_words(
        self,
        value: int,
    ) -> str:
        units = (
            "cero",
            "una",
            "dos",
            "tres",
            "cuatro",
            "cinco",
            "seis",
            "siete",
            "ocho",
            "nueve",
            "diez",
            "once",
            "doce",
            "trece",
            "catorce",
            "quince",
            "dieciséis",
            "diecisiete",
            "dieciocho",
            "diecinueve",
            "veinte",
            "veintiuna",
            "veintidós",
            "veintitrés",
            "veinticuatro",
            "veinticinco",
            "veintiséis",
            "veintisiete",
            "veintiocho",
            "veintinueve",
        )

        if 0 <= value < len(units):
            return units[value]

        tens = {
            30: "treinta",
            40: "cuarenta",
            50: "cincuenta",
        }
        ten = value - (value % 10)
        unit = value % 10

        if unit == 0:
            return tens[ten]

        return f"{tens[ten]} y {units[unit]}"

    def _format_response_for_voice(
        self,
        response: str,
    ) -> str:
        raw_response = response.strip()

        if not raw_response:
            return ""

        json_summary = self._extract_json_summary(raw_response)
        without_code_blocks = self._CODE_BLOCK_PATTERN.sub("", raw_response)
        cleaned_lines: list[str] = []

        for line in without_code_blocks.splitlines():
            cleaned = line.strip()

            if not cleaned:
                continue

            normalized = self._normalize(cleaned)

            if normalized == "respuesta:":
                continue

            if (
                "respuesta del sistema" in normalized
                or "formato actual" in normalized
                or "bloque json" in normalized
            ):
                continue

            cleaned_lines.append(cleaned)

        cleaned_response = "\n".join(cleaned_lines).strip()

        if cleaned_response:
            return cleaned_response

        if json_summary:
            return json_summary

        return raw_response

    def _extract_json_summary(
        self,
        response: str,
    ) -> str:
        for match in self._CODE_BLOCK_PATTERN.finditer(response):
            content = match.group(1).strip()

            try:
                payload = json.loads(content)
            except json.JSONDecodeError:
                continue

            if isinstance(payload, dict):
                return self._summarize_json_object(payload)

        return ""

    def _summarize_json_object(
        self,
        payload: dict,
    ) -> str:
        if not payload:
            return ""

        direct_answer = self._first_existing_value(
            payload,
            ("respuesta", "answer", "message", "content", "texto", "text"),
        )

        if direct_answer:
            return direct_answer

        time_value = self._first_existing_value(payload, ("hora", "time"))

        if time_value:
            return f"Hora: {time_value}."

        parts: list[str] = []

        for key, value in payload.items():
            if isinstance(value, (str, int, float, bool)):
                parts.append(f"{key}: {value}")

        return ". ".join(parts[:3]) + "." if parts else ""

    def _first_existing_value(
        self,
        payload: dict,
        keys: tuple[str, ...],
    ) -> str:
        normalized_payload = {
            str(key).casefold(): value
            for key, value in payload.items()
        }

        for key in keys:
            value = normalized_payload.get(key)

            if isinstance(value, (str, int, float, bool)):
                text = str(value).strip()

                if text:
                    return text

        return ""

    def _should_retry_initial_listen(
        self,
        result: SpeechTranscriptionResult,
    ) -> bool:
        if result.cancelled:
            return False

        return result.no_speech_detected or (
            not result.completed
            and not self._is_critical_error("; ".join(result.warnings))
            and not result.text.strip()
        )

    def _is_close_command(
        self,
        text: str,
    ) -> bool:
        return self._normalize(text) in self._CLOSE_COMMANDS

    def _is_critical_error(
        self,
        text: str,
    ) -> bool:
        normalized = self._normalize(text)
        return any(hint in normalized for hint in self._CRITICAL_ERROR_HINTS)

    def _end_session(
        self,
        session: VoiceConversationSession,
        reason: str,
        summary: str,
    ) -> None:
        session.active = False
        session.ended_reason = reason
        session.summary = summary

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

        collapsed = " ".join(without_accents.split())

        if collapsed.startswith("tu: "):
            return collapsed[4:].strip()

        if collapsed == "tu:":
            return ""

        return collapsed


def _read_bool(
    name: str,
    default: bool,
) -> bool:
    raw = os.getenv(name, "").strip().lower()

    if not raw:
        return default

    return raw in {"1", "true", "yes", "s", "si", "sí"}


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
