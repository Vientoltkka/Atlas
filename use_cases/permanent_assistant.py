"""Permanent Atlas assistant state controller."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import time
from typing import Callable

from use_cases.voice_conversation import VoiceConversationUseCase
from use_cases.wake_word_engine import WakeWordEngine


class PermanentAssistantState(str, Enum):
    """Runtime states for the permanent assistant loop."""

    INITIALIZING = "INITIALIZING"
    WAITING_WAKE_WORD = "WAITING_WAKE_WORD"
    LISTENING = "LISTENING"
    PROCESSING = "PROCESSING"
    SPEAKING = "SPEAKING"
    STOPPING = "STOPPING"


@dataclass
class PermanentAssistantResult:
    """Summary of one permanent assistant run."""

    stopped_reason: str
    conversations: int = 0
    recoverable_errors: int = 0
    states: list[PermanentAssistantState] = field(default_factory=list)


class PermanentAssistantUseCase:
    """Keep Atlas waiting for a wake word and process one request per activation."""

    def __init__(
        self,
        wake_word_engine: WakeWordEngine,
        voice_conversation: VoiceConversationUseCase,
        max_consecutive_errors: int = 5,
        cooldown_seconds: float = 0.2,
    ) -> None:
        if max_consecutive_errors <= 0:
            raise ValueError("El maximo de errores consecutivos debe ser mayor que cero.")
        if cooldown_seconds < 0:
            raise ValueError("La pausa posterior al TTS no puede ser negativa.")

        self._wake_word_engine = wake_word_engine
        self._voice_conversation = voice_conversation
        self._max_consecutive_errors = max_consecutive_errors
        self._cooldown_seconds = cooldown_seconds
        self._state = PermanentAssistantState.INITIALIZING

    @property
    def state(self) -> PermanentAssistantState:
        """Current assistant state."""
        return self._state

    def run(
        self,
        process_text: Callable[[str], str],
        status_sink: Callable[[str], None] | None = None,
        typed_input: Callable[[], str | None] | None = None,
    ) -> PermanentAssistantResult:
        """Run until Ctrl+C, explicit exit command, or repeated recoverable errors."""
        result = PermanentAssistantResult(stopped_reason="")
        consecutive_errors = 0

        def emit(message: str) -> None:
            if status_sink is not None:
                status_sink(message)

        def set_state(state: PermanentAssistantState) -> None:
            self._state = state
            result.states.append(state)

        try:
            set_state(PermanentAssistantState.INITIALIZING)
            emit("Atlas iniciado en modo asistente permanente.")
            self._voice_conversation.warm_up_resources()

            while True:
                typed_command = typed_input() if typed_input is not None else None

                if typed_command is not None and self._is_exit_command(typed_command):
                    result.stopped_reason = "explicit_close"
                    break

                set_state(PermanentAssistantState.WAITING_WAKE_WORD)
                emit('Esperando palabra de activacion "Atlas"...')
                wake_word = self._wake_word_engine.wait_for_wake_word()

                if wake_word.cancelled:
                    result.stopped_reason = "cancelled"
                    break

                if wake_word.configuration_error:
                    result.stopped_reason = "configuration_error"
                    emit(wake_word.warnings[0] if wake_word.warnings else "Wake word no configurada.")
                    break

                if not wake_word.detected:
                    if self._is_passive_miss(wake_word.warnings):
                        continue

                    consecutive_errors += 1
                    result.recoverable_errors += 1
                    self._emit_recoverable_error(
                        emit,
                        wake_word.warnings[0] if wake_word.warnings else "wake word no detectada",
                    )

                    if consecutive_errors >= self._max_consecutive_errors:
                        result.stopped_reason = "too_many_errors"
                        emit("Demasiados errores consecutivos. Cerrando asistente permanente.")
                        break

                    continue

                consecutive_errors = 0
                emit("Wake word detectada.")
                self._discard_residual_audio(emit)
                initial_text = self._initial_request_text(wake_word)
                set_state(PermanentAssistantState.LISTENING)
                emit("Escuchando petición...")
                set_state(PermanentAssistantState.PROCESSING)
                turn = self._voice_conversation.execute_assistant_turn(
                    process_text=process_text,
                    status_sink=emit,
                    initial_text=initial_text,
                )
                set_state(PermanentAssistantState.SPEAKING)

                if turn.session.ended_reason == "explicit_close":
                    result.stopped_reason = "explicit_close"
                    break

                if turn.session.ended_reason == "cancelled":
                    result.stopped_reason = "cancelled"
                    break

                if turn.session.ended_reason == "critical_error":
                    consecutive_errors += 1
                    result.recoverable_errors += 1
                    emit(turn.session.summary)
                elif turn.session.successful_turns > 0:
                    result.conversations += 1
                    consecutive_errors = 0
                else:
                    consecutive_errors += 1
                    result.recoverable_errors += 1
                    self._emit_recoverable_error(emit, turn.session.summary)

                if consecutive_errors >= self._max_consecutive_errors:
                    result.stopped_reason = "too_many_errors"
                    emit("Demasiados errores consecutivos. Cerrando asistente permanente.")
                    break

        except KeyboardInterrupt:
            result.stopped_reason = "keyboard_interrupt"
            emit("Cierre solicitado por Ctrl+C.")
        finally:
            set_state(PermanentAssistantState.STOPPING)
            self._voice_conversation.close_resources()
            if not result.stopped_reason:
                result.stopped_reason = "stopped"
            emit("Asistente permanente detenido.")

        return result

    def _discard_residual_audio(
        self,
        emit: Callable[[str], None],
    ) -> None:
        try:
            if self._cooldown_seconds:
                time.sleep(self._cooldown_seconds)
            self._voice_conversation.discard_residual_audio()
        except Exception as error:
            self._emit_recoverable_error(emit, f"No se pudo limpiar audio residual: {error}")

    def _emit_recoverable_error(
        self,
        emit: Callable[[str], None],
        message: str,
    ) -> None:
        if message:
            emit(f"Error recuperable: {message}")

    def _is_exit_command(
        self,
        text: str,
    ) -> bool:
        return self._voice_conversation.is_close_command(text)

    def _initial_request_text(
        self,
        wake_word,
    ) -> str | None:
        phrase = getattr(wake_word, "phrase", None)

        if phrase is None or not getattr(phrase, "completed", False):
            return None

        text = str(getattr(phrase, "text", "")).strip()
        return text or None

    def _is_passive_miss(
        self,
        warnings: tuple[str, ...],
    ) -> bool:
        return any(
            warning in {
                "sin activacion STT",
                "timeout de wake word alcanzado",
                "flujo de audio finalizado",
            }
            for warning in warnings
        )
