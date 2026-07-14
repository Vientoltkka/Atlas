"""Continuous voice conversation use case."""

from __future__ import annotations

from dataclasses import dataclass, field
import time
import unicodedata
from typing import Callable
from uuid import uuid4

from use_cases.speech_engine import (
    SpeechCaptureSettings,
    SpeechEngineUseCase,
    SpeechTranscriptionResult,
)
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

    _COMMANDS = {
        "activa conversacion por voz",
        "inicia modo voz",
        "conversacion por voz",
        "start voice conversation",
    }
    _CLOSE_COMMANDS = {
        "termina",
        "terminar conversacion",
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
    )

    def __init__(
        self,
        speech_engine: SpeechEngineUseCase,
        wake_word_engine: WakeWordEngine,
        conversation_idle_timeout: float = 25.0,
        max_session_duration: float = 600.0,
        max_turns: int = 20,
        max_consecutive_no_speech: int = 2,
        wake_capture_settings: SpeechCaptureSettings | None = None,
        clock: Callable[[], float] = time.monotonic,
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
        self._clock = clock
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

        messages: list[str] = []

        def emit(message: str) -> None:
            messages.append(message)

            if status_sink is not None:
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
            self._run_turns(session, process_text, emit)
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

        return VoiceConversationResult(session=session, messages=messages)

    def _warm_up(self) -> None:
        warm_up = getattr(self._speech_engine, "warm_up", None)

        if callable(warm_up):
            warm_up()
            return

        self._speech_engine.default_microphone()

    def _active_microphone(self):
        active_microphone = getattr(self._speech_engine, "active_microphone", None)

        if callable(active_microphone):
            return active_microphone()

        return self._speech_engine.default_microphone()

    def _prepare_stream(self) -> None:
        prepare_stream = getattr(self._speech_engine, "prepare_stream", None)

        if callable(prepare_stream):
            prepare_stream(self._wake_capture_settings)

    def _run_turns(
        self,
        session: VoiceConversationSession,
        process_text: Callable[[str], str],
        emit: Callable[[str], None],
    ) -> None:
        first_listen = True
        last_activity = self._clock()

        while session.active:
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

            emit("Escuchando siguiente frase...")
            transcription = self._speech_engine.transcribe_once()

            if first_listen and self._should_retry_initial_listen(transcription):
                emit("No se detecto voz. Vuelvo a escuchar...")
                transcription = self._speech_engine.transcribe_once()

            first_listen = False

            if transcription.cancelled:
                self._end_session(session, "cancelled", "Conversacion cancelada.")
                emit("Estado: conversacion finalizada.")
                emit("Conversacion cancelada.")
                break

            if transcription.no_speech_detected:
                session.consecutive_no_speech += 1
                session.failed_turns += 1

                if (
                    session.consecutive_no_speech
                    >= self._max_consecutive_no_speech
                ):
                    self._end_session(
                        session,
                        "idle_timeout",
                        "Conversacion finalizada por inactividad.",
                    )
                    emit("Estado: conversacion finalizada.")
                    emit(session.summary)
                    break

                emit("No se detecto ninguna frase.")
                continue

            if not transcription.completed:
                session.failed_turns += 1
                error = "; ".join(transcription.warnings) or transcription.summary

                if self._is_critical_error(error):
                    self._end_session(
                        session,
                        "critical_error",
                        f"Conversacion finalizada por error critico: {error}",
                    )
                    emit("Estado: conversacion finalizada.")
                    emit(session.summary)
                    break

                emit(f"No se pudo procesar el turno: {error}")
                continue

            session.consecutive_no_speech = 0
            text = transcription.text.strip()
            last_activity = self._clock()

            if self._is_close_command(text):
                self._end_session(
                    session,
                    "explicit_close",
                    "Conversacion finalizada.",
                )
                emit("Estado: conversacion finalizada.")
                emit("Conversacion finalizada.")
                break

            self._process_successful_turn(session, transcription, process_text, emit)

    def _process_successful_turn(
        self,
        session: VoiceConversationSession,
        transcription: SpeechTranscriptionResult,
        process_text: Callable[[str], str],
        emit: Callable[[str], None],
    ) -> None:
        text = transcription.text.strip()
        turn_number = session.total_turns + 1

        try:
            response = process_text(text)
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
                emit(session.summary)
            else:
                emit(f"No se pudo procesar el turno: {error}")

            return

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
        emit("Transcripcion:")
        emit(text)
        emit("Respuesta:")
        emit(response)

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
