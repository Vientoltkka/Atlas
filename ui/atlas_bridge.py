"""Qt bridge between Atlas core threads and the Orbe UI."""
from __future__ import annotations

from PySide6.QtCore import QObject, Signal

from use_cases.ui_state_mapper import map_to_orb_state


class AtlasUiBridge(QObject):
    """Thread-safe signal source for Orbe's voice and chat callbacks."""

    state_changed = Signal(object)
    voice_state_changed = Signal(str)
    voice_disconnected = Signal()
    message_received = Signal(str)
    user_message_received = Signal(str)
    transcription_received = Signal(str)
    response_received = Signal(str)
    error_received = Signal(str)
    session_finished = Signal()

    def __init__(self) -> None:
        super().__init__()
        self._quit_on_finish = False

    @property
    def quit_on_finish(self) -> bool:
        return self._quit_on_finish

    def request_quit_on_finish(self) -> None:
        self._quit_on_finish = True

    def on_state(self, state) -> None:
        """Core listener entry point; safe from any worker thread."""
        try:
            visual = map_to_orb_state(state)
        except ValueError:
            return
        self.state_changed.emit(visual)
        self.voice_state_changed.emit(str(state.value if hasattr(state, "value") else state))
        self.message_received.emit(f"Estado: {visual.value}")

    def on_message(self, message: str) -> None:
        text = str(message).strip()
        if not text:
            return
        if text.startswith("Tú:"):
            self.transcription_received.emit(text[3:].strip())
        elif text.startswith("Atlas:"):
            self.response_received.emit(text[6:].strip())
        else:
            self.message_received.emit(text)

    def on_user_message(self, message: str) -> None:
        text = str(message).strip()
        if text:
            self.user_message_received.emit(text)

    def on_response(self, response: str) -> None:
        text = str(response).strip()
        if text:
            self.response_received.emit(text)

    def on_error(self, error: BaseException | str) -> None:
        text = str(error).strip()
        if text:
            self.error_received.emit(text)

    def on_voice_error(self, error: BaseException | str) -> None:
        self.on_error(error)
        self.voice_state_changed.emit("ERROR")

    def notify_session_finished(self) -> None:
        """Called when the voice worker exits, regardless of its outcome."""
        self.voice_disconnected.emit()
        self.session_finished.emit()