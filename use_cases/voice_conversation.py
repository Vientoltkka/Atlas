"""Continuous voice conversation use case."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime
from difflib import SequenceMatcher
from enum import Enum
import json
import os
import queue
import re
import threading
import time
import unicodedata
from typing import Callable
from uuid import uuid4

from use_cases.speech_engine import (
    SpeechCaptureSettings,
    SpeechEngineUseCase,
    SpeechTranscriptionResult,
)
from use_cases.speech_output_engine import SpeechOutputEngine, SpeechOutputMetrics
from use_cases.wake_word_engine import WakeWordEngine


class VoiceConversationState(str, Enum):
    """Explicit lifecycle states for one voice session."""

    STARTING = "STARTING"
    READY = "READY"
    LISTENING = "LISTENING"
    TRANSCRIBING = "TRANSCRIBING"
    PROCESSING = "PROCESSING"
    SPEAKING = "SPEAKING"
    RECOVERING = "RECOVERING"
    STOPPING = "STOPPING"
    STOPPED = "STOPPED"
    DEGRADED = "DEGRADED"


class VoiceModelTimeoutError(TimeoutError):
    """Raised when the bounded voice model supervisor expires."""


@dataclass(frozen=True)
class VoiceTurnMetrics:
    """Privacy-safe monotonic timings for one voice turn."""

    voice_start_seconds: float = 0.0
    capture_seconds: float = 0.0
    stt_seconds: float = 0.0
    atlas_seconds: float = 0.0
    model_seconds: float = 0.0
    first_token_seconds: float = 0.0
    first_audio_seconds: float = 0.0
    post_first_audio_seconds: float = 0.0
    synthesis_seconds: float = 0.0
    playback_seconds: float = 0.0
    barge_in_detected: bool = False
    tts_cancel_latency_ms: float = 0.0
    barge_in_to_stt_ms: float = 0.0
    total_seconds: float = 0.0


@dataclass
class _StreamingVoiceDelivery:
    """Mutable per-turn state owned by the supervised model worker."""

    buffer: str = ""
    metrics: SpeechOutputMetrics = field(default_factory=SpeechOutputMetrics)
    first_fragment_seconds: float | None = None
    first_audio_at: float | None = None
    tts_started: bool = False
    interrupted: bool = False


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
    outcome: str = "completed"
    metrics: VoiceTurnMetrics = field(default_factory=VoiceTurnMetrics)


@dataclass
class VoiceConversationSession:
    """Voice conversation session state."""

    session_id: str
    started_at: float
    total_turns: int = 0
    successful_turns: int = 0
    failed_turns: int = 0
    consecutive_no_speech: int = 0
    consecutive_model_timeouts: int = 0
    active: bool = True
    ended_reason: str = ""
    transcript_history: list[str] = field(default_factory=list)
    response_history: list[str] = field(default_factory=list)
    turns: list[VoiceConversationTurn] = field(default_factory=list)
    summary: str = ""
    state: VoiceConversationState = VoiceConversationState.STARTING
    states: list[VoiceConversationState] = field(
        default_factory=lambda: [VoiceConversationState.STARTING]
    )


@dataclass
class VoiceConversationResult:
    """Result returned after a voice conversation session."""

    session: VoiceConversationSession
    messages: list[str]


class VoiceConversationUseCase:
    """Run a controlled continuous voice conversation."""

    _TTS_WORKER_JOIN_TIMEOUT_SECONDS = 0.25
    _MAX_EMPTY_BARGE_IN_CAPTURES = 2
    _EMPTY_BARGE_IN_BACKOFF_SECONDS = 0.05
    _MIN_STREAM_SEGMENT_CHARACTERS = 24
    _MIN_TTS_ECHO_PREFIX_TOKENS = 2
    _MIN_TTS_ECHO_PREFIX_CHARACTERS = 12
    _MIN_TTS_ECHO_TOKEN_SIMILARITY = 0.8
    _MIN_TTS_ECHO_CONTENT_TOKEN_CHARACTERS = 5
    _MIN_TTS_ECHO_COMMON_PREFIX_CHARACTERS = 6
    _MIN_TTS_ECHO_CONTENT_COVERAGE = 0.75
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
        "exit",
        "quit",
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
            max_duration=_read_float("ATLAS_VOICE_MAX_DURATION", 8.0, 1.0, 60.0),
            initial_silence_timeout=_read_float(
                "ATLAS_VOICE_INITIAL_SILENCE_TIMEOUT", 3.0, 0.2, 30.0
            ),
            trailing_silence=_read_float(
                "ATLAS_VOICE_TRAILING_SILENCE", 0.75, 0.1, 5.0
            ),
            chunk_duration=0.1,
            speech_threshold=_read_float("ATLAS_VOICE_RMS_THRESHOLD", 0.004, 0.001, 0.05),
            minimum_audio_duration=0.2,
        )
        self._diagnostics_enabled = (
            diagnostics_enabled
            if diagnostics_enabled is not None
            else (
                _read_bool("ATLAS_VOICE_DIAGNOSTICS", False)
                or _read_bool("ATLAS_VOICE_DEBUG", False)
            )
        )
        self._voice_debug_enabled = _read_bool("ATLAS_VOICE_DEBUG", False)
        self._metrics_enabled = _read_bool(
            "ATLAS_VOICE_METRICS",
            self._diagnostics_enabled or self._voice_debug_enabled,
        )
        self._tts_speaking = False
        self._tts_warmup_error = ""
        self._model_workers: set[threading.Thread] = set()
        self._tts_workers: set[threading.Thread] = set()
        self._pending_barge_in: SpeechTranscriptionResult | None = None
        self._barge_in_enabled = False
        self._model_timeout_seconds = _read_float(
            "ATLAS_VOICE_MODEL_TIMEOUT",
            135.0,
            0.1,
            600.0,
        )
        self._max_consecutive_model_timeouts = int(
            _read_float(
                "ATLAS_VOICE_MAX_CONSECUTIVE_TIMEOUTS",
                2.0,
                1.0,
                10.0,
            )
        )
        self._clock = clock
        self._now_provider = now_provider or (lambda: datetime.now().astimezone())
        self._session_id_factory = session_id_factory or (lambda: str(uuid4()))

    def execute(
        self,
        prompt: str,
        process_text: Callable[[str], str],
        status_sink: Callable[[str], None] | None = None,
        process_text_stream=None,
    ) -> VoiceConversationResult | None:
        """Run voice conversation mode for explicit activation commands."""
        if self._normalize(prompt) not in self._COMMANDS:
            return None
        if self._wake_word_engine is None:
            return self.execute_manual(
                process_text,
                status_sink,
                process_text_stream=process_text_stream,
            )

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
                process_text_stream=process_text_stream,
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
        process_text_stream=None,
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
            self._set_state(session, VoiceConversationState.STARTING, emit, force=True)
            active_microphone = self._active_microphone()
            self._validate_active_microphone()
            emit("Microfono activo:")
            emit(f"{active_microphone.index} - {active_microphone.name}")
            self._calibrate_turn_capture_settings()
            emit("Inicializando microfono, modelo y TTS local...")
            self._warm_up()
            if self._tts_warmup_error:
                self._set_state(session, VoiceConversationState.DEGRADED, emit)
                emit(
                    "TTS no disponible: "
                    f"{self._tts_warmup_error}. La sesion continuara solo por texto.",
                    diagnostic=False,
                )
            self._set_state(session, VoiceConversationState.READY, emit)
            emit("Conversacion de voz manual iniciada.")
            self._run_turns(
                session,
                process_text,
                emit,
                typed_input=typed_input,
                enforce_spanish_response=True,
                retry_initial_no_speech=False,
                keep_listening_on_recoverable=True,
                voice_debug=self._voice_debug_enabled,
                process_text_stream=process_text_stream,
            )
        except KeyboardInterrupt:
            self._end_session(session, "cancelled", "Conversacion cancelada.")
            emit("Conversacion cancelada.")
        except Exception as error:
            self._set_state(session, VoiceConversationState.DEGRADED, emit)
            self._end_session(
                session,
                "critical_error",
                f"Conversacion finalizada por error critico: {error}",
            )
            emit(session.summary)
        finally:
            self._set_state(session, VoiceConversationState.STOPPING, emit)
            self._close_resources()
            self._set_state(session, VoiceConversationState.STOPPED, emit)

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

        if self._is_close_command(transcription.text):
            self._end_session(session, "explicit_close", "Conversacion finalizada.")
            emit("Conversacion finalizada.", diagnostic=True)
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

        self._tts_warmup_error = ""
        if self._speech_output_engine is not None:
            warm_up_output = getattr(self._speech_output_engine, "warm_up", None)

            if callable(warm_up_output):
                try:
                    warm_up_output()
                except Exception as error:
                    self._tts_warmup_error = str(error) or type(error).__name__

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
        process_text_stream=None,
        typed_input: Callable[[], str | None] | None = None,
        enforce_spanish_response: bool = False,
        retry_initial_no_speech: bool = False,
        keep_listening_on_recoverable: bool = False,
        voice_debug: bool = False,
    ) -> None:
        first_listen = True
        last_activity = self._clock()
        self._barge_in_enabled = True

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

            turn_started = time.monotonic()
            pending_barge_in = self._take_pending_barge_in()
            from_barge_in = pending_barge_in is not None
            self._set_state(session, VoiceConversationState.LISTENING, emit)
            self._emit_voice_flow("comenzar siguiente turno", emit, voice_debug)
            if from_barge_in:
                transcription = pending_barge_in
                self._emit_voice_flow(
                    f"BARGE_IN dispatch query={transcription.text!r}",
                    emit,
                    voice_debug,
                )
            else:
                emit("Esperando voz...", diagnostic=False)
                transcription = self._transcribe_turn(session, emit)
            if transcription.no_speech_detected or not transcription.completed:
                self._emit_voice_debug_transcription(transcription, emit, voice_debug)

            if (
                retry_initial_no_speech
                and first_listen
                and not from_barge_in
                and self._should_retry_initial_listen(transcription)
            ):
                emit("No se detecto voz. Vuelvo a escuchar...", diagnostic=True)
                self._set_state(session, VoiceConversationState.LISTENING, emit)
                transcription = self._transcribe_turn(session, emit)
                if transcription.no_speech_detected or not transcription.completed:
                    self._emit_voice_debug_transcription(transcription, emit, voice_debug)

            first_listen = False

            if transcription.cancelled:
                self._end_session(session, "cancelled", "Conversacion cancelada.")
                emit("Estado: conversacion finalizada.", diagnostic=True)
                emit("Conversacion cancelada.", diagnostic=True)
                break

            if transcription.no_speech_detected:
                self._set_state(session, VoiceConversationState.RECOVERING, emit)
                session.consecutive_no_speech += 1
                session.failed_turns += 1
                last_activity = self._clock()

                if (
                    not keep_listening_on_recoverable
                    and
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

                self._emit_voice_debug_discard(
                    "timeout recuperable de captura",
                    emit,
                    voice_debug,
                )
                emit("No se detecto ninguna frase.", diagnostic=True)
                continue

            if not transcription.completed:
                self._set_state(session, VoiceConversationState.RECOVERING, emit)
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

                self._emit_voice_debug_discard(
                    error or "error recuperable del STT",
                    emit,
                    voice_debug,
                    transcription.exception_traceback,
                )
                emit(f"No se pudo procesar el turno: {error}", diagnostic=True)
                continue

            if self._is_close_command(transcription.text):
                self._end_session(
                    session,
                    "explicit_close",
                    "Conversacion finalizada.",
                )
                emit("Estado: conversacion finalizada.", diagnostic=True)
                emit("Conversacion finalizada.", diagnostic=True)
                break

            if from_barge_in:
                accepted_text = transcription.text
            else:
                accepted_text = self._accepted_transcription_text(
                    transcription,
                    trim_edge_punctuation=enforce_spanish_response,
                )

            if accepted_text is None:
                self._set_state(session, VoiceConversationState.RECOVERING, emit)
                self._emit_voice_debug_transcription(transcription, emit, voice_debug)
                self._emit_voice_debug_discard(
                    self._transcription_discard_reason(
                        transcription,
                        trim_edge_punctuation=enforce_spanish_response,
                    ),
                    emit,
                    voice_debug,
                )
                session.consecutive_no_speech += 1
                session.failed_turns += 1
                last_activity = self._clock()

                if (
                    not keep_listening_on_recoverable
                    and session.consecutive_no_speech >= self._max_consecutive_no_speech
                ):
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
                voice_debug,
                turn_started,
                process_text_stream,
            )
            if from_barge_in:
                self._emit_voice_flow(
                    f"BARGE_IN completed query={text!r}",
                    emit,
                    voice_debug,
                )
            if session.active:
                self._set_state(session, VoiceConversationState.READY, emit)
            last_activity = self._clock()

    def _process_successful_turn(
        self,
        session: VoiceConversationSession,
        transcription: SpeechTranscriptionResult,
        text: str,
        process_text: Callable[[str], str],
        emit: Callable[[str, bool], None],
        enforce_spanish_response: bool = False,
        voice_debug: bool = False,
        turn_started: float | None = None,
        process_text_stream=None,
    ) -> None:
        turn_number = session.total_turns + 1
        prompt_for_voice = self._prompt_for_voice(text, enforce_spanish_response)
        route_label = self._voice_flow_route(text)
        success = True
        error_text = ""
        outcome = "completed"
        model_duration = 0.0
        stream_delivery: _StreamingVoiceDelivery | None = None
        turn_started = time.monotonic() if turn_started is None else turn_started
        self._set_state(session, VoiceConversationState.PROCESSING, emit)
        processing_started = time.monotonic()

        try:
            self._emit_voice_flow(f"STT recibido: {text!r}", emit, voice_debug)
            self._emit_voice_flow(
                f"ruta seleccionada: {route_label}",
                emit,
                voice_debug,
            )
            if self._normalize(text) == "hoy":
                response = "¿Quieres saber la hora o la fecha?"
            elif route_label == "modelo":
                self._emit_voice_flow("antes de llamar al modelo", emit, voice_debug)
                model_started = time.monotonic()
                try:
                    if (
                        callable(process_text_stream)
                        and self._speech_output_engine is not None
                        and not self._tts_warmup_error
                    ):
                        stream_delivery = _StreamingVoiceDelivery()
                        response = self._process_text_for_voice(
                            prompt_for_voice,
                            process_text,
                            route_label,
                            process_text_stream=process_text_stream,
                            fragment_sink=lambda fragment: self._handle_stream_fragment(
                                fragment,
                                stream_delivery,
                                session,
                                emit,
                                voice_debug,
                                processing_started,
                            ),
                        )
                    else:
                        response = self._process_text_for_voice(
                            prompt_for_voice,
                            process_text,
                            route_label,
                        )
                finally:
                    model_duration = time.monotonic() - model_started
                    self._emit_voice_flow(
                        "despues de llamar al modelo",
                        emit,
                        voice_debug,
                    )
            else:
                response = self._process_text_for_voice(
                    prompt_for_voice,
                    process_text,
                    route_label,
                )
            processing_duration = time.monotonic() - processing_started
            self._emit_voice_flow(
                f"respuesta obtenida: {response!r}",
                emit,
                voice_debug,
            )
        except KeyboardInterrupt:
            raise
        except TimeoutError as error:
            processing_duration = time.monotonic() - processing_started
            error_text = "model_timeout"
            outcome = "model_timeout"
            response = "La respuesta está tardando demasiado. Inténtalo de nuevo."
            success = False
            session.failed_turns += 1
            session.consecutive_model_timeouts += 1
            self._set_state(session, VoiceConversationState.RECOVERING, emit)
            self._emit_voice_flow(
                f"timeout del modelo: {error}",
                emit,
                voice_debug,
            )
        except BaseException as error:
            processing_duration = time.monotonic() - processing_started
            error_text = f"{type(error).__name__}: {error}"
            outcome = "model_failure"
            response = f"Error en flujo post-STT: {error_text}"
            self._emit_voice_flow(
                f"respuesta obtenida: {response!r}",
                emit,
                voice_debug,
            )
            success = False
            session.failed_turns += 1
            self._set_state(session, VoiceConversationState.RECOVERING, emit)

        if outcome != "model_timeout":
            session.consecutive_model_timeouts = 0

        response = self._format_response_for_voice(
            response if response is not None else ""
        )
        should_speak_response = bool(response.strip())

        if not response:
            response = "No pude generar una respuesta. Inténtalo de nuevo."
            should_speak_response = True
            outcome = "empty_response"
            success = False
            error_text = "empty_response"
            session.failed_turns += 1
            self._set_state(session, VoiceConversationState.RECOVERING, emit)

        if stream_delivery is None:
            output_metrics = self._deliver_voice_response(
                session,
                text,
                response,
                should_speak_response,
                emit,
                voice_debug,
            )
        else:
            self._deliver_voice_response(
                session,
                text,
                response,
                False,
                emit,
                voice_debug,
            )
            output_metrics = self._finish_streaming_delivery(
                stream_delivery,
                session,
                emit,
                voice_debug,
                should_speak_response,
            )
        turn_finished = time.monotonic()
        first_token_seconds = (
            stream_delivery.first_fragment_seconds
            if stream_delivery is not None
            and stream_delivery.first_fragment_seconds is not None
            else 0.0
        )
        first_audio_seconds = (
            transcription.processing_duration_seconds
            + max(0.0, stream_delivery.first_audio_at - processing_started)
            if stream_delivery is not None and stream_delivery.first_audio_at is not None
            else transcription.processing_duration_seconds
            + processing_duration
            + output_metrics.first_audio_seconds
        )
        post_first_audio_seconds = (
            max(0.0, turn_finished - stream_delivery.first_audio_at)
            if stream_delivery is not None and stream_delivery.first_audio_at is not None
            else max(0.0, output_metrics.playback_seconds - output_metrics.first_audio_seconds)
        )
        metrics = VoiceTurnMetrics(
            voice_start_seconds=max(0.0, transcription.phrase_start_ms / 1000.0),
            capture_seconds=max(0.0, transcription.audio_duration_seconds),
            stt_seconds=max(0.0, transcription.processing_duration_seconds),
            atlas_seconds=max(0.0, processing_duration),
            model_seconds=max(0.0, model_duration),
            first_token_seconds=max(0.0, first_token_seconds),
            first_audio_seconds=max(0.0, first_audio_seconds),
            post_first_audio_seconds=max(0.0, post_first_audio_seconds),
            synthesis_seconds=max(0.0, output_metrics.synthesis_seconds),
            playback_seconds=max(0.0, output_metrics.playback_seconds),
            barge_in_detected=output_metrics.barge_in_detected,
            tts_cancel_latency_ms=max(0.0, output_metrics.tts_cancel_latency_ms),
            barge_in_to_stt_ms=max(0.0, output_metrics.barge_in_to_stt_ms),
            total_seconds=max(0.0, turn_finished - turn_started),
        )

        session.total_turns += 1
        if success:
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
                success=success,
                timestamp=self._clock(),
                error=error_text,
                outcome=outcome,
                metrics=metrics,
            )
        )
        self._emit_turn_metrics(metrics, emit, voice_debug)
        self._emit_voice_flow("turno terminado", emit, voice_debug)

        if (
            session.active
            and session.consecutive_model_timeouts
            >= self._max_consecutive_model_timeouts
        ):
            self._end_session(
                session,
                "model_timeout_limit",
                "Conversación finalizada tras varios timeouts consecutivos del modelo.",
            )
            emit("Estado: conversacion finalizada.", diagnostic=True)
            emit(session.summary, diagnostic=True)

    def _deliver_voice_response(
        self,
        session: VoiceConversationSession,
        text: str,
        response: str,
        should_speak_response: bool,
        emit: Callable[[str, bool], None],
        voice_debug: bool = False,
    ) -> SpeechOutputMetrics:
        """Print and speak exactly one final Atlas response for a voice turn."""
        if voice_debug:
            emit(f"Tú: {text}", diagnostic=False)
            emit(f"Atlas: {response}", diagnostic=False)
        elif self._diagnostics_enabled:
            emit("Transcripcion:", diagnostic=True)
            emit(text, diagnostic=True)
            emit("Respuesta:", diagnostic=True)
            emit(response, diagnostic=True)
        else:
            emit(f"Tú: {text}", diagnostic=False)
            emit(f"Atlas: {response}", diagnostic=False)

        output_metrics = SpeechOutputMetrics()
        if should_speak_response:
            if self._speech_output_engine is None or self._tts_warmup_error:
                self._set_state(session, VoiceConversationState.DEGRADED, emit)
            else:
                self._set_state(session, VoiceConversationState.SPEAKING, emit)
                self._emit_voice_flow("inicio TTS", emit, voice_debug)
                output_metrics = self._speak_response(
                    response,
                    session,
                    emit,
                    voice_debug,
                )
                self._emit_voice_flow("fin TTS", emit, voice_debug)
                if not output_metrics.barge_in_detected:
                    time.sleep(0.2)
                self._emit_voice_flow("vuelta a escucha", emit, voice_debug)

        return output_metrics

    def _handle_stream_fragment(
        self,
        fragment: str,
        delivery: _StreamingVoiceDelivery,
        session: VoiceConversationSession,
        emit: Callable[[str, bool], None],
        voice_debug: bool,
        processing_started: float,
    ) -> bool:
        if delivery.interrupted:
            return False
        if not fragment:
            return True
        if delivery.first_fragment_seconds is None:
            delivery.first_fragment_seconds = max(
                0.0,
                time.monotonic() - processing_started,
            )
        delivery.buffer += fragment
        segments, delivery.buffer = self._take_stream_segments(delivery.buffer)
        for segment in segments:
            self._speak_stream_segment(
                segment,
                delivery,
                session,
                emit,
                voice_debug,
            )
            if delivery.interrupted:
                return False
        return True

    def _finish_streaming_delivery(
        self,
        delivery: _StreamingVoiceDelivery,
        session: VoiceConversationSession,
        emit: Callable[[str, bool], None],
        voice_debug: bool,
        should_speak_response: bool,
    ) -> SpeechOutputMetrics:
        if (
            should_speak_response
            and delivery.buffer.strip()
            and not delivery.interrupted
        ):
            self._speak_stream_segment(
                delivery.buffer,
                delivery,
                session,
                emit,
                voice_debug,
            )
        delivery.buffer = ""

        if delivery.tts_started:
            self._emit_voice_flow("fin TTS", emit, voice_debug)
            if not delivery.metrics.barge_in_detected:
                time.sleep(0.2)
            self._emit_voice_flow("vuelta a escucha", emit, voice_debug)
        return delivery.metrics

    def _speak_stream_segment(
        self,
        segment: str,
        delivery: _StreamingVoiceDelivery,
        session: VoiceConversationSession,
        emit: Callable[[str, bool], None],
        voice_debug: bool,
    ) -> None:
        spoken = self._format_response_for_voice(segment)
        if not spoken:
            return
        if not delivery.tts_started:
            delivery.tts_started = True
            self._set_state(session, VoiceConversationState.SPEAKING, emit)
            self._emit_voice_flow("inicio TTS", emit, voice_debug)

        segment_started = time.monotonic()
        metrics = self._speak_response(
            spoken,
            session,
            emit,
            voice_debug,
        )
        if delivery.first_audio_at is None:
            delivery.first_audio_at = (
                segment_started + max(0.0, metrics.first_audio_seconds)
            )
        delivery.metrics = self._merge_speech_metrics(delivery.metrics, metrics)
        if metrics.barge_in_detected:
            delivery.interrupted = True

    def _take_stream_segments(self, text: str) -> tuple[list[str], str]:
        segments: list[str] = []
        consumed = 0
        boundary_pattern = re.compile(
            r"""[.!?](?:["')\]]*)?(?:\s+|$)|[;:](?:\s+|$)"""
        )
        for match in boundary_pattern.finditer(text):
            candidate = text[consumed:match.end()].strip()
            if (
                len(self._format_response_for_voice(candidate))
                < self._MIN_STREAM_SEGMENT_CHARACTERS
            ):
                continue
            segments.append(candidate)
            consumed = match.end()
        return segments, text[consumed:]

    def _merge_speech_metrics(
        self,
        current: SpeechOutputMetrics,
        new: SpeechOutputMetrics,
    ) -> SpeechOutputMetrics:
        first_audio_seconds = (
            current.first_audio_seconds
            if current.synthesis_seconds > 0.0 or current.playback_seconds > 0.0
            else new.first_audio_seconds
        )
        return SpeechOutputMetrics(
            synthesis_seconds=current.synthesis_seconds + new.synthesis_seconds,
            first_audio_seconds=first_audio_seconds,
            playback_seconds=current.playback_seconds + new.playback_seconds,
            barge_in_detected=current.barge_in_detected or new.barge_in_detected,
            tts_cancel_latency_ms=max(
                current.tts_cancel_latency_ms,
                new.tts_cancel_latency_ms,
            ),
            barge_in_to_stt_ms=max(
                current.barge_in_to_stt_ms,
                new.barge_in_to_stt_ms,
            ),
            tts_cancel_confirmed=(
                current.tts_cancel_confirmed or new.tts_cancel_confirmed
            ),
        )

    def _emit_turn_metrics(
        self,
        metrics: VoiceTurnMetrics,
        emit: Callable[[str, bool], None],
        voice_debug: bool,
    ) -> None:
        if self._metrics_enabled:
            emit("Métricas de latencia del turno:", diagnostic=False)
            emit(f"Inicio de voz: {metrics.voice_start_seconds * 1000:.0f} ms", diagnostic=False)
            emit(f"Captura: {metrics.capture_seconds * 1000:.0f} ms", diagnostic=False)
            emit(f"STT: {metrics.stt_seconds * 1000:.0f} ms", diagnostic=False)
            emit(f"Atlas: {metrics.atlas_seconds * 1000:.0f} ms", diagnostic=False)
            emit(f"Modelo: {metrics.model_seconds * 1000:.0f} ms", diagnostic=False)
            emit(f"Primer token: {metrics.first_token_seconds * 1000:.0f} ms", diagnostic=False)
            emit(f"Primer audio: {metrics.first_audio_seconds * 1000:.0f} ms", diagnostic=False)
            emit(f"Tras primer audio: {metrics.post_first_audio_seconds * 1000:.0f} ms", diagnostic=False)
            emit(f"Síntesis TTS: {metrics.synthesis_seconds * 1000:.0f} ms", diagnostic=False)
            emit(f"Reproducción: {metrics.playback_seconds * 1000:.0f} ms", diagnostic=False)
            emit(f"Total: {metrics.total_seconds * 1000:.0f} ms", diagnostic=False)
            if metrics.barge_in_detected:
                emit("barge_in_detected: true", diagnostic=False)
                emit(
                    f"tts_cancel_latency_ms: {metrics.tts_cancel_latency_ms:.0f}",
                    diagnostic=False,
                )
                emit(
                    f"barge_in_to_stt_ms: {metrics.barge_in_to_stt_ms:.0f}",
                    diagnostic=False,
                )

        if voice_debug:
            self._emit_voice_flow(
                "latencia espera_voz="
                f"{metrics.voice_start_seconds:.3f}s "
                f"captura={metrics.capture_seconds:.3f}s "
                f"stt={metrics.stt_seconds:.3f}s "
                f"procesamiento={metrics.atlas_seconds:.3f}s "
                f"primer_token={metrics.first_token_seconds:.3f}s "
                f"primer_audio={metrics.first_audio_seconds:.3f}s "
                f"post_audio={metrics.post_first_audio_seconds:.3f}s "
                f"tts={metrics.synthesis_seconds + metrics.playback_seconds:.3f}s "
                f"total={metrics.total_seconds:.3f}s",
                emit,
                voice_debug,
            )
    def _process_text_for_voice(
        self,
        prompt: str,
        process_text: Callable[[str], str],
        route_label: str,
        process_text_stream=None,
        fragment_sink=None,
    ) -> str:
        if route_label != "modelo":
            return process_text(prompt)

        return self._process_text_with_timeout(
            prompt,
            process_text,
            process_text_stream=process_text_stream,
            fragment_sink=fragment_sink,
        )

    def _process_text_with_timeout(
        self,
        prompt: str,
        process_text: Callable[[str], str],
        *,
        process_text_stream=None,
        fragment_sink=None,
    ) -> str:
        result_queue: queue.Queue[tuple[bool, str | BaseException]] = queue.Queue(maxsize=1)

        def run() -> None:
            try:
                value = (
                    process_text_stream(prompt, fragment_sink)
                    if callable(process_text_stream) and callable(fragment_sink)
                    else process_text(prompt)
                )
                result_queue.put((True, value))
            except BaseException as error:
                result_queue.put((False, error))
            finally:
                self._model_workers.discard(threading.current_thread())

        worker = threading.Thread(target=run, daemon=True, name="atlas-voice-model")
        self._model_workers.add(worker)
        worker.start()

        try:
            ok, value = result_queue.get(timeout=self._model_timeout_seconds)
        except queue.Empty:
            raise VoiceModelTimeoutError(
                f"supervisor agotado tras {self._model_timeout_seconds:.1f} s"
            )

        if ok:
            if value is None:
                return ""

            return str(value)

        if isinstance(value, KeyboardInterrupt):
            raise value

        if isinstance(value, Exception):
            raise value

        if isinstance(value, BaseException):
            raise RuntimeError(f"{type(value).__name__}: {value}") from value

        raise RuntimeError(str(value))

    def _voice_flow_route(
        self,
        text: str,
    ) -> str:
        normalized = re.sub(r"[^\w\s]", " ", self._normalize(text))
        normalized = " ".join(normalized.split())
        compact = re.sub(r"[^a-z0-9]+", "", normalized)

        if (
            normalized == "hoy"
            or self._asks_current_time(normalized)
            or self._asks_current_date(normalized)
        ):
            return "local"

        if any(
            marker in normalized
            for marker in (
                "notepad",
                "visual studio code",
            )
        ) or any(
            marker in compact
            for marker in (
                "blocdenotas",
                "vscode",
                "visualstudiocode",
            )
        ):
            return "herramienta"

        return "modelo"

    def _emit_voice_flow(
        self,
        message: str,
        emit: Callable[[str, bool], None],
        enabled: bool,
    ) -> None:
        if enabled:
            emit(f"[voice-flow] {message}", diagnostic=False)

    def _speak_response(
        self,
        response: str,
        session: VoiceConversationSession,
        emit: Callable[[str, bool], None],
        voice_debug: bool = False,
    ) -> SpeechOutputMetrics:
        if self._speech_output_engine is None or not response.strip():
            return SpeechOutputMetrics()

        self._tts_speaking = True
        try:
            emit("TTS iniciado", diagnostic=True)
            cancel = getattr(self._speech_output_engine, "cancel", None)
            if self._barge_in_enabled and callable(cancel):
                metrics = self._speak_with_barge_in(
                    response,
                    emit,
                    voice_debug,
                )
            else:
                metrics = self._invoke_speech_output(response)
            emit("TTS finalizado", diagnostic=True)
            return metrics
        except KeyboardInterrupt:
            raise
        except Exception as error:
            self._set_state(session, VoiceConversationState.DEGRADED, emit)
            emit(f"Error TTS: {error}", diagnostic=False)
            return SpeechOutputMetrics()
        finally:
            self._tts_speaking = False

    def _invoke_speech_output(
        self,
        response: str,
    ) -> SpeechOutputMetrics:
        if self._speech_output_engine is None:
            return SpeechOutputMetrics()

        speak_with_metrics = getattr(
            self._speech_output_engine,
            "speak_with_metrics",
            None,
        )
        if callable(speak_with_metrics):
            return speak_with_metrics(response)

        playback_started = time.monotonic()
        self._speech_output_engine.speak(response)
        return SpeechOutputMetrics(
            playback_seconds=time.monotonic() - playback_started,
        )

    def _speak_with_barge_in(
        self,
        response: str,
        emit: Callable[[str, bool], None],
        voice_debug: bool,
    ) -> SpeechOutputMetrics:
        result_queue: queue.Queue[tuple[bool, object]] = queue.Queue(maxsize=1)
        cancel_attempted = False
        cancel_request_accepted = False
        forced_stop_confirmed = False
        empty_capture_attempts = 0
        detected = False
        cancel_latency_ms = 0.0
        barge_in_to_stt_ms = 0.0

        def run_tts() -> None:
            try:
                result_queue.put((True, self._invoke_speech_output(response)))
            except BaseException as error:
                result_queue.put((False, error))
            finally:
                self._tts_workers.discard(threading.current_thread())

        worker = threading.Thread(
            target=run_tts,
            name="atlas-voice-tts",
            daemon=True,
        )
        self._tts_workers.add(worker)
        worker.start()
        worker.join(timeout=0.01)

        try:
            while worker.is_alive():
                self._emit_voice_flow(
                    "BARGE_IN capture_started",
                    emit,
                    voice_debug,
                )
                stt_started = time.monotonic()
                transcription = self._transcribe_barge_in_turn()
                stt_finished = time.monotonic()
                speech_detected = bool(
                    transcription.completed
                    and not transcription.no_speech_detected
                    and transcription.text.strip()
                )
                self._emit_voice_flow(
                    f"BARGE_IN speech_detected={speech_detected}",
                    emit,
                    voice_debug,
                )
                self._emit_voice_flow(
                    f"BARGE_IN transcript={transcription.text!r}",
                    emit,
                    voice_debug,
                )

                if transcription.cancelled:
                    self._cancel_speech_output()
                    cancel_attempted = True
                    worker.join()
                    raise KeyboardInterrupt

                if (
                    transcription.no_speech_detected
                    and transcription.samples_count == 0
                ):
                    empty_capture_attempts += 1
                    self._emit_voice_flow(
                        "BARGE_IN no_audio "
                        f"attempt={empty_capture_attempts}/"
                        f"{self._MAX_EMPTY_BARGE_IN_CAPTURES}",
                        emit,
                        voice_debug,
                    )
                    if empty_capture_attempts >= self._MAX_EMPTY_BARGE_IN_CAPTURES:
                        self._emit_voice_flow(
                            "BARGE_IN monitoring_stopped reason=no_audio",
                            emit,
                            voice_debug,
                        )
                        break
                    time.sleep(self._EMPTY_BARGE_IN_BACKOFF_SECONDS)
                    continue

                empty_capture_attempts = 0
                clean_text = self._clean_transcription_text(transcription.text)
                echo_prefix_removed, clean_text = self._strip_tts_echo_prefix(
                    clean_text,
                    response,
                )
                if echo_prefix_removed:
                    self._emit_voice_flow(
                        "BARGE_IN tts_echo_prefix_removed=True",
                        emit,
                        voice_debug,
                    )
                    self._emit_voice_flow(
                        f"BARGE_IN cleaned_transcript={clean_text!r}",
                        emit,
                        voice_debug,
                    )
                accepted_text = self._accepted_barge_in_text(
                    replace(transcription, text=clean_text),
                    response,
                )
                if accepted_text is None:
                    worker.join(timeout=0.01)
                    continue

                interruption_recognized, _remainder = (
                    self._strip_interruption_prefix(clean_text)
                )
                self._emit_voice_flow(
                    f"BARGE_IN interruption_recognized={interruption_recognized}",
                    emit,
                    voice_debug,
                )
                if accepted_text:
                    self._pending_barge_in = replace(
                        transcription,
                        text=accepted_text,
                    )
                    self._emit_voice_flow(
                        f"BARGE_IN pending_query={accepted_text!r}",
                        emit,
                        voice_debug,
                    )
                detected = True
                barge_in_to_stt_ms = max(
                    0.0,
                    (stt_finished - stt_started) * 1000.0
                    - max(0.0, transcription.phrase_start_ms),
                )
                cancel_started = time.monotonic()
                cancel_request_accepted = self._cancel_speech_output()
                self._emit_voice_flow(
                    f"BARGE_IN stop_requested={cancel_request_accepted}",
                    emit,
                    voice_debug,
                )
                cancel_attempted = True
                worker.join(timeout=self._TTS_WORKER_JOIN_TIMEOUT_SECONDS)
                if worker.is_alive():
                    self._emit_voice_flow(
                        "BARGE_IN force_close_requested=True",
                        emit,
                        voice_debug,
                    )
                    force_close_requested = self._force_close_speech_output()
                    worker.join(timeout=self._TTS_WORKER_JOIN_TIMEOUT_SECONDS)
                    forced_stop_confirmed = bool(
                        force_close_requested and not worker.is_alive()
                    )
                cancel_latency_ms = max(
                    0.0,
                    (time.monotonic() - cancel_started) * 1000.0,
                )
                self._emit_voice_flow(
                    f"barge-in detectado: {accepted_text!r}",
                    emit,
                    voice_debug,
                )
                break

            worker.join(timeout=self._TTS_WORKER_JOIN_TIMEOUT_SECONDS)
            if worker.is_alive():
                output_metrics = SpeechOutputMetrics()
            else:
                completed, payload = result_queue.get_nowait()
                if not completed:
                    raise payload
                output_metrics = payload
            if not isinstance(output_metrics, SpeechOutputMetrics):
                output_metrics = SpeechOutputMetrics()
            if detected:
                cancel_confirmed = bool(
                    output_metrics.tts_cancel_confirmed
                    or forced_stop_confirmed
                )
                tts_stopped = bool(
                    cancel_request_accepted
                    and not worker.is_alive()
                    and cancel_confirmed
                )
                self._emit_voice_flow(
                    f"BARGE_IN tts_stopped={tts_stopped}",
                    emit,
                    voice_debug,
                )
            return replace(
                output_metrics,
                barge_in_detected=detected,
                tts_cancel_confirmed=(
                    output_metrics.tts_cancel_confirmed
                    or forced_stop_confirmed
                ),
                tts_cancel_latency_ms=cancel_latency_ms,
                barge_in_to_stt_ms=barge_in_to_stt_ms,
            )
        except BaseException:
            if not cancel_attempted:
                self._cancel_speech_output()
            worker.join(timeout=self._TTS_WORKER_JOIN_TIMEOUT_SECONDS)
            raise
        finally:
            if not worker.is_alive():
                self._tts_workers.discard(worker)

    def _transcribe_barge_in_turn(self) -> SpeechTranscriptionResult:
        max_duration = min(self._turn_capture_settings.max_duration, 3.0)
        settings = replace(
            self._turn_capture_settings,
            max_duration=max_duration,
            initial_silence_timeout=max_duration,
            minimum_audio_duration=min(
                self._turn_capture_settings.minimum_audio_duration,
                0.2,
            ),
        )
        try:
            return self._speech_engine.transcribe_once(settings)
        except TypeError:
            return self._speech_engine.transcribe_once()

    def _accepted_barge_in_text(
        self,
        transcription: SpeechTranscriptionResult,
        spoken_response: str,
    ) -> str | None:
        if (
            transcription.cancelled
            or transcription.no_speech_detected
            or not transcription.completed
        ):
            return None

        clean_text = self._clean_transcription_text(transcription.text)
        if not clean_text:
            return None
        if self._is_close_command(clean_text):
            return clean_text

        has_interruption_prefix, candidate_text = (
            self._strip_interruption_prefix(clean_text)
        )
        if has_interruption_prefix and not candidate_text:
            return ""
        normalized_clean = self._normalize_echo_text(clean_text)
        normalized_response = self._normalize_echo_text(spoken_response)
        if normalized_clean and normalized_clean in normalized_response:
            return None
        if has_interruption_prefix:
            echo_removed, residual_segments = self._strip_tts_echo_fragments(
                candidate_text,
                spoken_response,
            )
            if echo_removed:
                human_candidates = self._human_barge_in_candidates(
                    transcription,
                    residual_segments,
                )
                if len(human_candidates) != 1:
                    return ""
                return human_candidates[0]
            clean_text = candidate_text

        if self._is_tts_echo(clean_text, spoken_response):
            return None

        return self._accepted_transcription_text(
            replace(transcription, text=clean_text),
            trim_edge_punctuation=False,
        )

    def _strip_interruption_prefix(
        self,
        text: str,
    ) -> tuple[bool, str]:
        match = re.match(r"^\s*para\b", text, flags=re.IGNORECASE)
        if match is None:
            return False, text

        remainder = text[match.end():].lstrip()
        remainder = re.sub(r"^[,.;:!¡-]+\s*", "", remainder)
        if not self._useful_text(remainder):
            return True, ""
        return True, remainder.strip()

    def _strip_tts_echo_prefix(
        self,
        transcription: str,
        spoken_response: str,
    ) -> tuple[bool, str]:
        """Remove only a strong leading match with the active TTS segment."""
        if not self._tts_speaking:
            return False, transcription

        transcription_words = list(re.finditer(r"\w+", transcription))
        response_words = list(re.finditer(r"\w+", spoken_response))
        matched_tokens = 0

        for response_start in range(len(response_words)):
            candidate_tokens = 0
            for transcription_word, response_word in zip(
                transcription_words,
                response_words[response_start:],
            ):
                captured = self._normalize_echo_text(transcription_word.group())
                spoken = self._normalize_echo_text(response_word.group())
                if not captured or not spoken:
                    break
                if captured != spoken:
                    similarity = SequenceMatcher(
                        None,
                        captured,
                        spoken,
                        autojunk=False,
                    ).ratio()
                    if similarity < self._MIN_TTS_ECHO_TOKEN_SIMILARITY:
                        break
                candidate_tokens += 1
            matched_tokens = max(matched_tokens, candidate_tokens)

        if matched_tokens < self._MIN_TTS_ECHO_PREFIX_TOKENS:
            return False, transcription

        matched_characters = sum(
            len(self._normalize_echo_text(word.group()))
            for word in transcription_words[:matched_tokens]
        )
        if matched_characters < self._MIN_TTS_ECHO_PREFIX_CHARACTERS:
            return False, transcription

        residual = transcription[transcription_words[matched_tokens - 1].end():]
        residual = re.sub(r"^\s*[,.;:-]\s*", "", residual)
        return True, residual.strip()

    def _strip_tts_echo_fragments(
        self,
        transcription: str,
        spoken_response: str,
    ) -> tuple[bool, list[str]]:
        """Remove only exact multi-token blocks from the active TTS text."""
        transcription_words = list(re.finditer(r"\w+", transcription))
        response_words = list(re.finditer(r"\w+", spoken_response))
        transcription_tokens = [
            self._normalize_echo_text(word.group())
            for word in transcription_words
        ]
        response_tokens = [
            self._normalize_echo_text(word.group())
            for word in response_words
        ]
        echo_blocks = []
        for block in SequenceMatcher(
            None,
            transcription_tokens,
            response_tokens,
            autojunk=False,
        ).get_matching_blocks():
            if block.size < self._MIN_TTS_ECHO_PREFIX_TOKENS:
                continue
            matched_characters = sum(
                len(token)
                for token in transcription_tokens[block.a:block.a + block.size]
            )
            if matched_characters < self._MIN_TTS_ECHO_PREFIX_CHARACTERS:
                continue
            echo_blocks.append(block)

        if not echo_blocks:
            return False, [transcription]

        residual_segments: list[str] = []
        cursor = 0
        for block in echo_blocks:
            start = transcription_words[block.a].start()
            end = transcription_words[block.a + block.size - 1].end()
            residual_segments.append(transcription[cursor:start])
            cursor = end
        residual_segments.append(transcription[cursor:])
        return True, residual_segments

    def _human_barge_in_candidates(
        self,
        transcription: SpeechTranscriptionResult,
        residual_segments: list[str],
    ) -> list[str]:
        candidates: list[str] = []
        seen: set[str] = set()
        for segment in residual_segments:
            parts = re.split(r"\bpara\s*[,.;:!?-]+\s*", segment, flags=re.IGNORECASE)
            for part in parts:
                candidate = part.strip(" \t\r\n,;:")
                words = re.findall(r"\w+", candidate)
                if not candidate or (
                    len(words) < 2 and not self._is_close_command(candidate)
                ):
                    continue
                accepted = self._accepted_transcription_text(
                    replace(transcription, text=candidate),
                    trim_edge_punctuation=False,
                )
                normalized = self._normalize_echo_text(accepted or "")
                if accepted and normalized not in seen:
                    seen.add(normalized)
                    candidates.append(accepted)
        return candidates

    def _is_tts_echo(
        self,
        transcription: str,
        spoken_response: str,
    ) -> bool:
        normalized_transcription = self._normalize_echo_text(transcription)
        normalized_response = self._normalize_echo_text(spoken_response)
        if not normalized_transcription or not normalized_response:
            return False
        if normalized_transcription in normalized_response:
            return True

        transcription_tokens = normalized_transcription.split()
        response_tokens = normalized_response.split()
        if not transcription_tokens:
            return False

        content_tokens = [
            token
            for token in transcription_tokens
            if len(token) >= self._MIN_TTS_ECHO_CONTENT_TOKEN_CHARACTERS
        ]
        if content_tokens:
            matched_content = sum(
                any(
                    self._tts_echo_tokens_match(captured, spoken)
                    for spoken in response_tokens
                )
                for captured in content_tokens
            )
            if (
                matched_content / len(content_tokens)
                >= self._MIN_TTS_ECHO_CONTENT_COVERAGE
            ):
                return True

        transcription_token_set = set(transcription_tokens)
        response_token_set = set(response_tokens)
        overlap = len(transcription_token_set & response_token_set)
        return overlap / len(transcription_token_set) >= 0.8

    def _tts_echo_tokens_match(self, captured: str, spoken: str) -> bool:
        if captured == spoken:
            return True

        shorter_length = min(len(captured), len(spoken))
        if shorter_length < self._MIN_TTS_ECHO_CONTENT_TOKEN_CHARACTERS:
            return False

        common_prefix = 0
        for captured_character, spoken_character in zip(captured, spoken):
            if captured_character != spoken_character:
                break
            common_prefix += 1

        return (
            common_prefix >= self._MIN_TTS_ECHO_COMMON_PREFIX_CHARACTERS
            and common_prefix / shorter_length >= 0.7
        )

    def _normalize_echo_text(self, text: str) -> str:
        normalized = self._normalize(text)
        without_punctuation = "".join(
            " " if unicodedata.category(character).startswith("P") else character
            for character in normalized
        )
        return " ".join(without_punctuation.split())

    def _take_pending_barge_in(self) -> SpeechTranscriptionResult | None:
        pending = self._pending_barge_in
        self._pending_barge_in = None
        return pending

    def _cancel_speech_output(self) -> bool:
        if self._speech_output_engine is None:
            return False
        cancel = getattr(self._speech_output_engine, "cancel", None)
        if not callable(cancel):
            return False
        try:
            return bool(cancel())
        except Exception:
            return False

    def _force_close_speech_output(self) -> bool:
        if self._speech_output_engine is None:
            return False
        close = getattr(self._speech_output_engine, "close", None)
        if not callable(close):
            return False
        try:
            close()
            return True
        except Exception:
            return False

    def _calibrate_turn_capture_settings(self) -> None:
        calibrate = getattr(self._speech_engine, "calibrate_noise_threshold", None)

        if not callable(calibrate):
            return

        try:
            threshold = calibrate(self._turn_capture_settings, 0.4)
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
        self._barge_in_enabled = False
        self._pending_barge_in = None
        if self._speech_output_engine is not None:
            try:
                self._speech_output_engine.close()
            except Exception:
                pass
        self._join_tts_workers()
        self._join_model_workers()

        close_speech = getattr(self._speech_engine, "close", None)
        if callable(close_speech):
            close_speech()

    def _join_tts_workers(self) -> None:
        for worker in tuple(self._tts_workers):
            if worker is threading.current_thread():
                continue
            worker.join(timeout=self._TTS_WORKER_JOIN_TIMEOUT_SECONDS)
            if not worker.is_alive():
                self._tts_workers.discard(worker)

    def _join_model_workers(self) -> None:
        for worker in tuple(self._model_workers):
            if worker is threading.current_thread():
                continue
            worker.join(timeout=0.5)
            if not worker.is_alive():
                self._model_workers.discard(worker)
    def _transcribe_turn(
        self,
        session: VoiceConversationSession | None = None,
        emit: Callable[[str, bool], None] | None = None,
    ) -> SpeechTranscriptionResult:
        def stage_sink(stage: str) -> None:
            if stage == "transcribing" and session is not None:
                self._set_state(session, VoiceConversationState.TRANSCRIBING, emit)

        try:
            return self._speech_engine.transcribe_once(
                self._turn_capture_settings,
                stage_sink=stage_sink,
            )
        except TypeError:
            if session is not None:
                self._set_state(session, VoiceConversationState.TRANSCRIBING, emit)
            try:
                return self._speech_engine.transcribe_once(self._turn_capture_settings)
            except TypeError:
                return self._speech_engine.transcribe_once()

    def _emit_voice_debug_transcription(
        self,
        transcription: SpeechTranscriptionResult,
        emit: Callable[[str, bool], None],
        enabled: bool,
    ) -> None:
        if not enabled:
            return

        self._emit_voice_debug(
            f"duracion audio capturado: {transcription.audio_duration_seconds:.3f}s",
            emit,
            enabled,
        )
        self._emit_voice_debug(
            f"numero de muestras: {transcription.samples_count}",
            emit,
            enabled,
        )
        self._emit_voice_debug(
            f"RMS final: {transcription.rms:.6f}",
            emit,
            enabled,
        )
        self._emit_voice_debug(
            f"texto exacto STT repr: {transcription.text!r}",
            emit,
            enabled,
        )

        if transcription.exception_traceback:
            self._emit_voice_debug(
                "excepcion completa:\n" + transcription.exception_traceback,
                emit,
                enabled,
            )

    def _emit_voice_debug_discard(
        self,
        reason: str,
        emit: Callable[[str, bool], None],
        enabled: bool,
        exception_traceback: str = "",
    ) -> None:
        if not enabled:
            return

        self._emit_voice_debug(
            f"texto descartado: si | motivo: {reason}",
            emit,
            enabled,
        )

        if exception_traceback:
            self._emit_voice_debug(
                "excepcion completa:\n" + exception_traceback,
                emit,
                enabled,
            )

    def _emit_voice_debug(
        self,
        message: str,
        emit: Callable[[str, bool], None],
        enabled: bool,
    ) -> None:
        if enabled:
            emit(f"[voice-debug] {message}", diagnostic=False)

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

        normalized = self._normalize(text)
        allowed_short_intents = {
            "hora",
            "fecha",
            "hoy",
            "salir",
            "atras",
            "vs code",
            "vscode",
            "bloc de notas",
        }

        if (
            len(useful) <= 3
            and normalized not in allowed_short_intents
            and not self._is_close_command(text)
        ):
            return None

        return text

    def _transcription_discard_reason(
        self,
        transcription: SpeechTranscriptionResult,
        trim_edge_punctuation: bool = False,
    ) -> str:
        text = self._clean_transcription_text(
            transcription.text,
            trim_edge_punctuation=trim_edge_punctuation,
        )

        if not text:
            return "STT devolvio texto vacio"

        useful = self._useful_text(text)
        confidence = self._confidence(transcription.average_log_probability)
        no_speech_probability = transcription.no_speech_probability

        if (
            no_speech_probability is not None
            and no_speech_probability
            > self._stt_threshold("ATLAS_STT_MAX_NO_SPEECH_PROBABILITY", 0.65)
        ):
            return (
                "probabilidad de silencio alta: "
                f"{no_speech_probability:.3f}"
            )

        if (
            confidence is not None
            and confidence < self._stt_threshold("ATLAS_STT_MIN_CONFIDENCE", 0.35)
            and len(useful) < 8
        ):
            return f"confianza baja en texto corto: {confidence:.3f}"

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
            return "frase corta de bajo valor filtrada"

        if len(useful) < 2 and not self._is_close_command(text):
            return "texto util demasiado corto"

        normalized = self._normalize(text)
        allowed_short_intents = {
            "hora",
            "fecha",
            "hoy",
            "salir",
            "atras",
            "vs code",
            "vscode",
            "bloc de notas",
        }

        if (
            len(useful) <= 3
            and normalized not in allowed_short_intents
            and not self._is_close_command(text)
        ):
            return "texto corto sin intencion local util"

        return "motivo de descarte no identificado"

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
        words = normalized.split()

        if words == ["hora"] or words == ["hora", "es"]:
            return True

        if words in (
            ["que", "hora"],
            ["que", "hora", "es"],
            ["que", "hora", "es", "hoy"],
            ["dime", "la", "hora"],
        ):
            return True

        if any(
            phrase in normalized
            for phrase in (
                "que hora es",
                "dime la hora",
                "hora actual",
            )
        ):
            return True

        return (
            "hora" in words
            and any(marker in words for marker in ("actual", "ahora"))
            and (
                normalized.startswith("que ")
                or "dime" in words
                or "actual" in words
                or "mismo" in words
            )
        )

    def _asks_current_date(
        self,
        normalized: str,
    ) -> bool:
        words = normalized.split()

        if words == ["fecha"] or words == ["fecha", "hoy"]:
            return True

        if words in (
            ["que", "fecha", "es", "hoy"],
            ["que", "dia", "es", "hoy"],
            ["dime", "la", "fecha"],
        ):
            return True

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
        normalized = self._normalize(text)
        without_punctuation = "".join(
            " " if unicodedata.category(character).startswith("P") else character
            for character in normalized
        )
        return " ".join(without_punctuation.split()) in self._CLOSE_COMMANDS

    def _is_critical_error(
        self,
        text: str,
    ) -> bool:
        normalized = self._normalize(text)
        return any(hint in normalized for hint in self._CRITICAL_ERROR_HINTS)

    def _set_state(
        self,
        session: VoiceConversationSession,
        state: VoiceConversationState,
        emit: Callable[[str, bool], None] | None = None,
        force: bool = False,
    ) -> None:
        if session.state == state and not force:
            return
        session.state = state
        if not session.states or session.states[-1] != state:
            session.states.append(state)
        if emit is not None:
            emit(f"Estado: {state.value}", diagnostic=True)
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
