"""Qt bridge between Atlas core threads and the Orbe UI (V4.3-I3).

The bridge converts real ``VoiceConversationState`` values into visual
states and crosses threads via Qt signals. It never touches STT, TTS or
capture components: the UI stays a pure rendering layer.
"""

from __future__ import annotations

from PySide6.QtCore import QObject, Signal

from use_cases.ui_state_mapper import map_to_orb_state


class AtlasUiBridge(QObject):
    """Signal source fed by the core ``state_listener`` callback."""

    state_changed = Signal(object)
    message_received = Signal(str)
    session_finished = Signal()

    def __init__(self) -> None:
        super().__init__()
        self._quit_on_finish = False

    @property
    def quit_on_finish(self) -> bool:
        """Whether finishing the session should close the application."""
        return self._quit_on_finish

    def request_quit_on_finish(self) -> None:
        """Mark that the pending session end must close the app."""
        self._quit_on_finish = True

    def on_state(self, state) -> None:
        """Core listener entry point; safe to call from any thread."""
        try:
            visual = map_to_orb_state(state)
        except ValueError:
            return
        self.state_changed.emit(visual)
        self.message_received.emit(f"Estado: {visual.value}")

    def on_message(self, message: str) -> None:
        text = str(message).strip()
        if text:
            self.message_received.emit(text)

    def notify_session_finished(self) -> None:
        """Called when the voice session thread ends (any reason)."""
        self.session_finished.emit()
