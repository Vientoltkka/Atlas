"""Thread-safe controller joining the Orbe UI to Atlas orchestration."""
from __future__ import annotations

import logging
import threading

from PySide6.QtCore import QObject, Signal

from ui.atlas_bridge import AtlasUiBridge
from ui.windows_hotkey import WindowsGlobalHotkey
from use_cases.ui_state_mapper import OrbVisualState


class _ChatHotkeyBridge(QObject):
    """Marshal native hotkey callbacks back onto the Qt UI thread."""

    activated = Signal()


class _VoiceSessionBridge(QObject):
    """Tag queued worker callbacks so stale sessions cannot repaint the UI."""

    state_received = Signal(int, object)
    message_received = Signal(int, object)


class OrbeController:
    """Own UI lifecycle; workers emit signals and never touch widgets."""

    def __init__(
        self,
        *,
        atlas,
        application,
        orb,
        transcript_panel,
        logger=None,
        hotkey_factory=None,
    ):
        self._atlas = atlas
        self._application = application
        self._orb = orb
        self._transcript_panel = transcript_panel
        self._logger = logger or logging.getLogger(__name__)
        self._bridge = AtlasUiBridge()
        self._stop_event = threading.Event()
        self._session_thread = None
        self._session_generation = 0
        self._chat_threads: set[threading.Thread] = set()
        self._hotkey_factory = hotkey_factory
        self._chat_hotkey = None
        self._chat_hotkey_bridge = _ChatHotkeyBridge()
        self._voice_session_bridge = _VoiceSessionBridge()
        self._voice_session_bridge.state_received.connect(self._on_voice_state)
        self._voice_session_bridge.message_received.connect(self._on_voice_message)
        self._bridge.state_changed.connect(self._orb.apply_state)
        self._bridge.voice_state_changed.connect(self._transcript_panel.set_voice_state)
        self._bridge.voice_disconnected.connect(self._transcript_panel.set_voice_disconnected)
        self._bridge.message_received.connect(self._transcript_panel.append_message)
        self._bridge.user_message_received.connect(self._transcript_panel.append_user)
        self._bridge.transcription_received.connect(self._transcript_panel.append_transcription)
        self._bridge.response_received.connect(self._transcript_panel.append_response)
        self._bridge.error_received.connect(self._transcript_panel.append_error)
        self._bridge.session_finished.connect(self._on_session_finished)
        subscribe_supervision = getattr(self._atlas, "add_supervision_state_listener", None)
        if callable(subscribe_supervision):
            subscribe_supervision(self._bridge.on_supervision_state)
        self._orb.stop_requested.connect(self.stop)
        self._orb.quit_requested.connect(self.request_quit)
        self._transcript_panel.send_requested.connect(self.submit_text)
        self._transcript_panel.attachment_send_requested.connect(self.submit_attachment_notice)
        self._transcript_panel.close_requested.connect(self.hide_chat)
        self._transcript_panel.voice_start_requested.connect(self.start_voice)
        self._transcript_panel.voice_stop_requested.connect(self.stop)
        self._transcript_panel.voice_retry_requested.connect(self.retry_voice)
        self._chat_hotkey_bridge.activated.connect(self.show_chat)

    @property
    def bridge(self):
        return self._bridge

    def start(self, *, start_voice: bool = True, show_on_start: bool = True) -> None:
        self._transcript_panel.set_hide_on_close(not start_voice)
        if show_on_start:
            self.show_chat()
        else:
            # Windows autostart hides only the chat panel; the Atlas core remains visible.
            self._transcript_panel.hide()
            self._orb.show()
        if not start_voice:
            self._application.setQuitOnLastWindowClosed(False)
            self._start_chat_hotkey()
        if start_voice:
            self.start_voice()

    def show_chat(self) -> None:
        """Show and focus the existing chat windows without starting voice."""
        self._orb.show()
        self._transcript_panel.show()
        self._transcript_panel.raise_()
        self._transcript_panel.activateWindow()

    def hide_chat(self) -> None:
        """Hide the chat pair while preserving the running controller."""
        self._transcript_panel.hide()
        self._orb.hide()

    def start_voice(self) -> None:
        """Start a new voice session only when no prior session is running."""
        if self._session_thread is not None and self._session_thread.is_alive():
            return
        self._stop_event = threading.Event()
        self._session_generation += 1
        generation = self._session_generation
        self._orb.apply_state(OrbVisualState.STARTING)
        self._transcript_panel.set_voice_state("STARTING")
        self._session_thread = threading.Thread(
            target=self._run_session,
            args=(generation, self._stop_event),
            daemon=True,
            name="atlas-orbe-voice",
        )
        self._session_thread.start()

    def retry_voice(self) -> None:
        """Retry a completed voice session without affecting text chat."""
        self.start_voice()

    def run(self, *, start_voice: bool = True, show_on_start: bool = True) -> int:
        self.start(start_voice=start_voice, show_on_start=show_on_start)
        try:
            return self._application.exec()
        finally:
            self._stop_chat_hotkey()
            self.stop()
            self.join()

    def stop(self) -> None:
        """Acknowledge cancellation in the UI without waiting for a worker."""
        self._stop_event.set()
        self._session_generation += 1
        self._orb.apply_state(OrbVisualState.IDLE)
        self._transcript_panel.set_voice_disconnected()

    def request_quit(self) -> None:
        self.stop()
        self._bridge.request_quit_on_finish()

    def submit_text(self, prompt: str) -> None:
        text = str(prompt).strip()
        if not text:
            return
        self._bridge.on_user_message(text)
        worker = threading.Thread(
            target=self._run_text_prompt, args=(text,), daemon=True, name="atlas-orbe-text"
        )
        self._chat_threads.add(worker)
        worker.start()

    def submit_attachment_notice(self, prompt: str, attachment) -> None:
        """Keep attachment handling local until chat attachment analysis is wired."""
        text = str(prompt).strip()
        if not text:
            return
        self._bridge.on_user_message(text)
        self._bridge.on_response(
            f"Archivo adjunto: {attachment.name}. "
            "El análisis de archivos desde el chat todavía no está habilitado."
        )

    def join(self, timeout: float = 8.0) -> None:
        if self._session_thread is not None and self._session_thread.is_alive():
            self._session_thread.join(timeout=timeout)
        self.join_chat(timeout=timeout)

    def join_chat(self, timeout: float = 8.0) -> None:
        for worker in tuple(self._chat_threads):
            worker.join(timeout=timeout)

    @staticmethod
    def _typed_input(stop_event):
        return "salir" if stop_event.is_set() else None

    def _run_session(self, generation: int, stop_event: threading.Event) -> None:
        def is_current() -> bool:
            return generation == self._session_generation and not stop_event.is_set()

        def on_state(state) -> None:
            self._voice_session_bridge.state_received.emit(generation, state)

        def on_message(message) -> None:
            self._voice_session_bridge.message_received.emit(generation, message)

        try:
            self._atlas.start_voice(
                state_listener=on_state,
                status_sink=on_message,
                typed_input=lambda: self._typed_input(stop_event),
            )
        except Exception as error:
            self._logger.error("Fallo en la sesion de voz del Orbe: %s", error)
            if is_current():
                self._bridge.on_voice_error("La sesion de voz ha terminado por un error.")
        finally:
            if is_current():
                self._bridge.notify_session_finished()

    def _on_voice_state(self, generation: int, state) -> None:
        if generation == self._session_generation:
            self._bridge.on_state(state)

    def _on_voice_message(self, generation: int, message) -> None:
        if generation == self._session_generation:
            self._bridge.on_message(message)

    def _run_text_prompt(self, prompt: str) -> None:
        try:
            self._bridge.on_response(self._atlas.process_prompt(prompt))
        except Exception as error:
            self._logger.error("Fallo en el chat textual del Orbe: %s", error)
            self._bridge.on_error("No se pudo procesar el mensaje textual.")
        finally:
            self._chat_threads.discard(threading.current_thread())

    def _on_session_finished(self) -> None:
        if self._bridge.quit_on_finish:
            self._application.quit()
        else:
            self._orb.apply_state(OrbVisualState.IDLE)

    def _start_chat_hotkey(self) -> None:
        if self._chat_hotkey is None:
            factory = self._hotkey_factory or WindowsGlobalHotkey
            self._chat_hotkey = factory(self._request_chat_visible, logger=self._logger)
        self._chat_hotkey.start()

    def _stop_chat_hotkey(self) -> None:
        if self._chat_hotkey is not None:
            self._chat_hotkey.stop()

    def _request_chat_visible(self) -> None:
        self._chat_hotkey_bridge.activated.emit()
