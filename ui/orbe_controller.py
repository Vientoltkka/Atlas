"""Thread-safe controller joining the Orbe UI to Atlas orchestration."""
from __future__ import annotations

import logging
import threading

from ui.atlas_bridge import AtlasUiBridge
from use_cases.ui_state_mapper import OrbVisualState


class OrbeController:
    """Own UI lifecycle; workers emit signals and never touch widgets."""

    def __init__(self, *, atlas, application, orb, transcript_panel, logger=None):
        self._atlas = atlas
        self._application = application
        self._orb = orb
        self._transcript_panel = transcript_panel
        self._logger = logger or logging.getLogger(__name__)
        self._bridge = AtlasUiBridge()
        self._stop_event = threading.Event()
        self._session_thread = None
        self._chat_threads: set[threading.Thread] = set()
        self._bridge.state_changed.connect(self._orb.apply_state)
        self._bridge.voice_state_changed.connect(self._transcript_panel.set_voice_state)
        self._bridge.voice_disconnected.connect(self._transcript_panel.set_voice_disconnected)
        self._bridge.message_received.connect(self._transcript_panel.append_message)
        self._bridge.user_message_received.connect(self._transcript_panel.append_user)
        self._bridge.transcription_received.connect(self._transcript_panel.append_transcription)
        self._bridge.response_received.connect(self._transcript_panel.append_response)
        self._bridge.error_received.connect(self._transcript_panel.append_error)
        self._bridge.session_finished.connect(self._on_session_finished)
        self._orb.stop_requested.connect(self.stop)
        self._orb.quit_requested.connect(self.request_quit)
        self._transcript_panel.send_requested.connect(self.submit_text)
        self._transcript_panel.voice_start_requested.connect(self.start_voice)
        self._transcript_panel.voice_stop_requested.connect(self.stop)
        self._transcript_panel.voice_retry_requested.connect(self.retry_voice)

    @property
    def bridge(self):
        return self._bridge

    def start(self) -> None:
        self._orb.show()
        self._transcript_panel.show()
        self.start_voice()

    def start_voice(self) -> None:
        """Start a new voice session only when no prior session is running."""
        if self._session_thread is not None and self._session_thread.is_alive():
            return
        self._stop_event = threading.Event()
        self._session_thread = threading.Thread(
            target=self._run_session, daemon=True, name="atlas-orbe-voice"
        )
        self._session_thread.start()

    def retry_voice(self) -> None:
        """Retry a completed voice session without affecting text chat."""
        self.start_voice()

    def run(self) -> int:
        self.start()
        exit_code = self._application.exec()
        self.stop()
        self.join()
        return exit_code

    def stop(self) -> None:
        self._stop_event.set()

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

    def join(self, timeout: float = 8.0) -> None:
        if self._session_thread is not None and self._session_thread.is_alive():
            self._session_thread.join(timeout=timeout)
        self.join_chat(timeout=timeout)

    def join_chat(self, timeout: float = 8.0) -> None:
        for worker in tuple(self._chat_threads):
            worker.join(timeout=timeout)

    def _typed_input(self):
        return "salir" if self._stop_event.is_set() else None

    def _run_session(self) -> None:
        try:
            self._atlas.start_voice(
                state_listener=self._bridge.on_state,
                status_sink=self._bridge.on_message,
                typed_input=self._typed_input,
            )
        except Exception as error:
            self._logger.error("Fallo en la sesion de voz del Orbe: %s", error)
            self._bridge.on_voice_error("La sesion de voz ha terminado por un error.")
        finally:
            self._bridge.notify_session_finished()

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