"""Thread-safe controller joining the Orbe UI to Atlas orchestration."""
from __future__ import annotations

import logging
import os
import threading

from PySide6.QtCore import QObject, Signal, Slot

from ui.atlas_bridge import AtlasUiBridge
from ui.windows_hotkey import WindowsGlobalHotkey
from use_cases.ui_state_mapper import OrbVisualState


class _OrbHotkeyBridge(QObject):
    """Marshal native hotkey callbacks back onto the Qt UI thread."""

    activated = Signal()

    def __init__(self, show_orb, logger) -> None:
        super().__init__()
        self._show_orb = show_orb
        self._logger = logger
        self.activated.connect(self._show_orb_on_ui_thread)

    @Slot()
    def _show_orb_on_ui_thread(self) -> None:
        self._logger.info("Callback de Ctrl+Espacio en hilo UI")
        self._show_orb()


class _EscapeEventFilter(QObject):
    """Last-resort ESC: hides the whole Atlas interface from any window.

    Installed on the QApplication so ESC works no matter which Atlas
    window (chat, orb, controls) holds focus. It never consumes the
    event, so widgets keep their native ESC behavior.
    """

    def __init__(self, hide_interface, logger) -> None:
        super().__init__()
        self._hide_interface = hide_interface
        self._logger = logger

    def eventFilter(self, obj, event) -> bool:  # noqa: N802 (Qt API)
        try:
            from PySide6.QtCore import QEvent, Qt

            if event.type() == QEvent.Type.KeyPress and event.key() == Qt.Key.Key_Escape:
                self._logger.info("ESC recibido; ocultando interfaz Atlas")
                self._hide_interface()
        except Exception:
            pass
        return False


class _VoiceSessionBridge(QObject):
    """Tag queued worker callbacks so stale sessions cannot repaint the UI."""

    state_received = Signal(int, object)
    message_received = Signal(int, object)


def _stage_v2_requested() -> bool:
    """Feature flag ATLAS_STAGE_V2: truthy values route Ctrl+Espacio to V2."""
    return os.environ.get("ATLAS_STAGE_V2", "").strip().lower() in {"1", "true", "yes", "on"}


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
        fullscreen_factory=None,
        stage_v2_factory=None,
    ):
        self._atlas = atlas
        self._application = application
        self._orb = orb
        self._transcript_panel = transcript_panel
        self._logger = logger or logging.getLogger(__name__)
        # Fullscreen stage (V1): hosts the compact orb over a holographic
        # stage; created only when possible so tests with plain fakes pass.
        if fullscreen_factory is None:
            def fullscreen_factory():
                from ui.orbe_stage import OrbeStage

                return OrbeStage(orb, atlas=self._atlas)
        self._stage = None
        try:
            self._stage = fullscreen_factory()
        except Exception:
            self._stage = None
            self._logger.warning("Overlay fullscreen no disponible; se usa el orbe compacto")
        # Interactive layer (X + capability menu): the ONLY clickable surface
        # of the fullscreen composition. The stage itself is click-through.
        self._controls = None
        if self._stage is not None:
            try:
                from ui.orbe_stage import OrbeStageControls

                self._controls = OrbeStageControls(self._stage)
            except Exception:
                self._controls = None
                self._logger.warning("Capa de controles del stage no disponible")
        for overlay in (self._stage, self._controls):
            if overlay is None:
                continue
            overlay.chat_requested.connect(self.show_chat)
            overlay.voice_requested.connect(self.toggle_voice)
            overlay.quit_requested.connect(self.request_quit)
            overlay.capability_selected.connect(self.open_capability)
            if hasattr(overlay, "hide_requested"):
                overlay.hide_requested.connect(self.hide_interface)
        # Stage V2 (experimental): when ATLAS_STAGE_V2 is truthy, Ctrl+Espacio
        # shows ONLY AtlasStageV2; the legacy orb/stage/controls stay hidden.
        self._stage_v2 = None
        if _stage_v2_requested():
            factory = stage_v2_factory

            if factory is None:
                def factory():
                    from ui.stage_v2.renderer import AtlasStageV2

                    return AtlasStageV2()
            try:
                self._stage_v2 = factory()
            except Exception:
                self._stage_v2 = None
                self._logger.warning("Stage V2 no disponible; se usa el overlay legacy")
            if self._stage_v2 is not None and hasattr(self._stage_v2, "chat_requested"):
                self._stage_v2.chat_requested.connect(self._show_chat_from_stage_v2)
        self._bridge = AtlasUiBridge()
        self._stop_event = threading.Event()
        self._session_thread = None
        self._session_generation = 0
        self._chat_threads: set[threading.Thread] = set()
        self._hotkey_factory = hotkey_factory
        self._chat_hotkey = None
        # Ctrl+Espacio now TOGGLES: visible -> hide everything, hidden -> show.
        self._chat_hotkey_bridge = _OrbHotkeyBridge(self.toggle_interface, self._logger)
        # ESC must work GLOBALLY while Atlas is visible (the overlay usually
        # has no keyboard focus on Windows); it is unregistered when hidden.
        self._escape_hotkey = None
        self._escape_hotkey_bridge = _OrbHotkeyBridge(self.hide_interface, self._logger)
        self._escape_filter = _EscapeEventFilter(self.hide_interface, self._logger)
        try:
            self._application.installEventFilter(self._escape_filter)
        except Exception:
            self._logger.warning("Filtro ESC de aplicacion no disponible")
        self._voice_session_bridge = _VoiceSessionBridge()
        self._voice_session_bridge.state_received.connect(self._on_voice_state)
        self._voice_session_bridge.message_received.connect(self._on_voice_message)
        self._last_voice_visual_state = OrbVisualState.IDLE
        self._supervision_visual_override = None
        self._bridge.voice_visual_state_changed.connect(self._apply_voice_visual_state)
        self._bridge.supervision_visual_state_changed.connect(self._apply_supervision_visual_state)
        self._bridge.voice_state_changed.connect(self._transcript_panel.set_voice_state)
        self._bridge.voice_disconnected.connect(self._transcript_panel.set_voice_disconnected)
        self._bridge.message_received.connect(self._transcript_panel.append_message)
        self._bridge.user_message_received.connect(self._transcript_panel.append_user)
        self._bridge.transcription_received.connect(self._transcript_panel.append_transcription)
        self._bridge.response_received.connect(self._transcript_panel.append_response)
        self._bridge.error_received.connect(self._on_error_received)
        self._bridge.session_finished.connect(self._on_session_finished)
        subscribe_supervision = getattr(self._atlas, "add_supervision_state_listener", None)
        if callable(subscribe_supervision):
            subscribe_supervision(self._bridge.on_supervision_state)
        self._orb.stop_requested.connect(self.stop)
        self._orb.quit_requested.connect(self.request_quit)
        self._orb.chat_requested.connect(self.show_chat)
        self._orb.voice_requested.connect(self.toggle_voice)
        self._orb.context_menu.capability_selected.connect(self.open_capability)
        # Tray and Alt+F4 safety exits (real orb window only).
        if hasattr(self._orb, "show_requested"):
            self._orb.show_requested.connect(self.show_orb)
        if hasattr(self._orb, "hide_requested"):
            self._orb.hide_requested.connect(self.hide_interface)
        if hasattr(self._orb, "close_requested"):
            self._orb.close_requested.connect(self.hide_interface)
        if self._stage is not None and hasattr(self._orb, "visual_state_changed"):
            self._orb.visual_state_changed.connect(self._stage.apply_state)
            self._stage.apply_state(self._orb.state)
        self._transcript_panel.send_requested.connect(self.submit_text)
        self._transcript_panel.attachment_send_requested.connect(self.submit_attachment_notice)
        self._transcript_panel.close_requested.connect(self.hide_chat)
        self._transcript_panel.voice_start_requested.connect(self.start_voice)
        self._transcript_panel.voice_stop_requested.connect(self.stop)
        self._transcript_panel.voice_retry_requested.connect(self.retry_voice)

    @property
    def bridge(self):
        return self._bridge

    @property
    def stage(self):
        """Fullscreen holographic stage (None when unavailable)."""
        return self._stage

    @property
    def controls(self):
        """Interactive layer of the stage (None when unavailable)."""
        return self._controls

    @property
    def stage_v2(self):
        """Experimental Stage V2 (None unless ATLAS_STAGE_V2 is enabled)."""
        return self._stage_v2

    def start(
        self,
        *,
        start_voice: bool = True,
        show_on_start: bool = False,
        start_hidden: bool = False,
    ) -> None:
        self._transcript_panel.set_hide_on_close(not start_voice)
        if show_on_start:
            self.show_chat()
        else:
            self._transcript_panel.hide()
            if start_hidden:
                self._orb.hide()
                self._hide_stage()
                self._logger.info("Autoarranque oculto: Orbe y chat ocultos")
            else:
                self._orb.show()
                self._show_stage()
        if not start_voice:
            self._application.setQuitOnLastWindowClosed(False)
        # Failsafe hotkey in EVERY mode: Ctrl+Espacio must always toggle.
        self._start_chat_hotkey()
        if start_voice:
            self.start_voice()

    def show_chat(self) -> None:
        """Show and focus the existing chat windows without starting voice."""
        self._show_chat_without_overlap()

    def _show_chat_from_stage_v2(self) -> None:
        """Open the real chat from Stage V2, reusing the legacy show_chat flow.

        Stage V2 is fullscreen and stays on top, so it MUST hide first:
        otherwise the chat would open BEHIND it and look like a dead button.
        On failure Stage V2 is restored and a brief in-stage error is shown.
        """
        stage = self._stage_v2
        was_visible = stage is not None and stage.isVisible()
        if was_visible:
            stage.hide()
        try:
            self.show_chat()
        except Exception as error:
            self._logger.error("Fallo al abrir el chat desde Stage V2: %s", error)
            if was_visible and stage is not None and not stage.isVisible():
                stage.show()
                restore_on_top = getattr(stage, "raise_", None)
                if callable(restore_on_top):
                    restore_on_top()
            set_error = getattr(stage, "set_status_error", None)
            if callable(set_error):
                set_error("CHAT: error al abrir el chat")
            return
        # Chat visible: bring it to front and give it focus when the
        # existing panel exposes those APIs.
        bring_to_front = getattr(self._transcript_panel, "raise_", None)
        if callable(bring_to_front):
            bring_to_front()
        set_focus_window = getattr(self._transcript_panel, "activateWindow", None)
        if callable(set_focus_window):
            set_focus_window()

    # Capability menu options reuse the existing chat routing: the option only
    # opens the chat and pre-fills the domain prefix; the real router in
    # core.operational_request_router / agents resolves the request.
    _CAPABILITY_PREFIXES = {
        "coding": "Coding: ",
        "proyectos": "Proyectos: ",
        "entrenamiento": "Entrenamiento: ",
        "nutricion": "Nutrición: ",
        "salud": "Salud: ",
        "calendario": "Calendario: ",
        "control_pc": "Control PC: ",
        "automatizacion": "Automatización: ",
        "investigacion": "Investigación: ",
        "legal": "Legal: ",
        "finanzas": "Finanzas: ",
        "agentes": "Agentes: ",
        "mas_herramientas": "Más herramientas: ",
    }

    def open_capability(self, capability_id: str) -> None:
        """Open the chat prepared for one existing Atlas capability domain."""
        self.show_chat()
        prefix = self._CAPABILITY_PREFIXES.get(str(capability_id))
        if prefix is not None:
            self._transcript_panel.prefill_input(prefix)

    def show_orb(self) -> None:
        """Muestra el overlay Atlas completo: orbe + stage + controles."""
        if self._stage_v2 is not None:
            if not self._stage_v2.isVisible():
                self._stage_v2.show()
                self._stage_v2.raise_()
            self._set_escape_hotkey(True)
            self._logger.info("Overlay Atlas mostrado (Stage V2)")
            return
        self._orb.show()
        self._orb.raise_()
        self._show_stage()
        self._set_escape_hotkey(True)
        self._logger.info("Overlay Atlas mostrado (orbe + stage + controles)")

    def hide_interface(self) -> None:
        """Salida segura e inmediata: oculta TODO Atlas.

        Funciona en cualquier estado (LISTENING, PROCESSING, SPEAKING,
        AUTOMATION, AUTHORIZATION, ERROR) porque solo toca visibilidad de
        ventanas; nunca depende de la voz ni de un worker.
        """
        self._hide_stage()
        self._orb.hide()
        self._transcript_panel.hide()
        self._set_escape_hotkey(False)
        self._logger.info("Interfaz Atlas oculta (salida segura)")

    def toggle_interface(self) -> None:
        """Ctrl+Espacio: si Atlas es visible lo oculta; si está oculto lo muestra."""
        if self._overlay_visible():
            self.hide_interface()
        else:
            self.show_orb()

    def _overlay_visible(self) -> bool:
        if self._stage_v2 is not None and self._stage_v2.isVisible():
            return True
        if self._orb.isVisible() or self._transcript_panel.isVisible():
            return True
        stage_visible = self._stage is not None and self._stage.isVisible()
        controls_visible = self._controls is not None and self._controls.isVisible()
        return stage_visible or controls_visible

    def _show_chat_without_overlap(self) -> None:
        """Show the existing chat once and resolve only an initial overlap.

        The fullscreen stage is topmost, so it must hide first: otherwise
        the chat would open BEHIND it and look like a dead button.
        """
        self._hide_stage()
        self._orb.show()
        self._transcript_panel.show()
        self._position_chat_without_overlap()
        self._transcript_panel.raise_()
        self._transcript_panel.activateWindow()
        self._set_escape_hotkey(True)

    def _position_chat_without_overlap(self) -> None:
        """Keep the initial orb position unless the newly shown panel covers it."""
        if not self._orb.frameGeometry().intersects(self._transcript_panel.frameGeometry()):
            return
        screen = self._orb.screen() or self._transcript_panel.screen()
        if screen is None:
            return
        bounds = screen.availableGeometry()
        panel = self._transcript_panel
        orb_geometry = self._orb.frameGeometry()
        gap = 16
        y = max(
            bounds.top(),
            min(orb_geometry.center().y() - panel.height() // 2, bounds.bottom() - panel.height() + 1),
        )
        for x in (orb_geometry.right() + gap + 1, orb_geometry.left() - gap - panel.width()):
            if bounds.left() <= x and x + panel.width() <= bounds.right() + 1:
                panel.move(x, y)
                return
        # On narrow displays preserve the readable chat position and move the orb once.
        self._orb.reposition_beside(panel)
        if not self._orb.frameGeometry().intersects(panel.frameGeometry()):
            return

        # If neither window fits beside its current position, anchor the chat
        # and shift the orb to its side while keeping both windows on screen.
        panel_x = bounds.left()
        orb_x = panel_x + panel.width() + gap
        if orb_x + self._orb.width() <= bounds.right() + 1:
            panel_y = max(bounds.top(), min(panel.y(), bounds.bottom() - panel.height() + 1))
            orb_y = max(
                bounds.top(),
                min(panel_y + (panel.height() - self._orb.height()) // 2, bounds.bottom() - self._orb.height() + 1),
            )
            panel.move(panel_x, panel_y)
            self._orb.move(orb_x, orb_y)

    def hide_chat(self) -> None:
        """Hide the chat pair while preserving the running controller."""
        self._transcript_panel.hide()
        self._orb.hide()
        self._hide_stage()
        self._set_escape_hotkey(False)

    def start_voice(self) -> None:
        """Start a new voice session only when no prior session is running."""
        if self._session_thread is not None and self._session_thread.is_alive():
            return
        self._orb.set_voice_active(True)
        self._set_stage_voice_active(True)
        self._stop_event = threading.Event()
        self._session_generation += 1
        generation = self._session_generation
        self._apply_voice_visual_state(OrbVisualState.STARTING)
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

    def toggle_voice(self) -> None:
        """Reuse the existing start/stop flow selected from the orb popup."""
        if self._session_thread is not None and self._session_thread.is_alive():
            self.stop()
        else:
            self.start_voice()

    def run(
        self,
        *,
        start_voice: bool = True,
        show_on_start: bool = True,
        start_hidden: bool = False,
    ) -> int:
        self.start(
            start_voice=start_voice,
            show_on_start=show_on_start,
            start_hidden=start_hidden,
        )
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
        self._orb.set_voice_active(False)
        self._set_stage_voice_active(False)
        self._apply_voice_visual_state(OrbVisualState.IDLE)
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

    def _on_error_received(self, error: str) -> None:
        """Surface a real error in the chat and as a discrete stage signal."""
        self._transcript_panel.append_error(error)
        if self._stage is not None:
            self._stage.set_last_error(str(error))

    def _run_text_prompt(self, prompt: str) -> None:
        try:
            self._bridge.on_response(self._atlas.process_prompt(prompt))
        except Exception as error:
            self._logger.error("Fallo en el chat textual del Orbe: %s", error)
            self._bridge.on_error("No se pudo procesar el mensaje textual.")
        finally:
            self._chat_threads.discard(threading.current_thread())

    def _on_session_finished(self) -> None:
        self._orb.set_voice_active(False)
        self._set_stage_voice_active(False)
        if self._bridge.quit_on_finish:
            self._application.quit()
        else:
            self._apply_voice_visual_state(OrbVisualState.IDLE)

    def _apply_voice_visual_state(self, state) -> None:
        """Remember voice state without displacing an active supervision cue."""
        self._last_voice_visual_state = OrbVisualState(state)
        if self._supervision_visual_override is None:
            self._orb.apply_state(self._last_voice_visual_state)

    def _apply_supervision_visual_state(self, state) -> None:
        """Give approval and execution their required visual precedence."""
        visual_state = OrbVisualState(state)
        if visual_state in {OrbVisualState.AUTHORIZATION, OrbVisualState.AUTOMATION}:
            self._supervision_visual_override = visual_state
            self._orb.apply_state(visual_state)
            return
        self._supervision_visual_override = None
        self._orb.apply_state(self._last_voice_visual_state)

    # -- fullscreen stage -------------------------------------------------

    def _show_stage(self) -> None:
        if self._stage is not None and not self._stage.isVisible():
            self._stage.show()
            self._stage.raise_()
        self._show_controls()
        # The real orb stays ABOVE the translucent stage: it is the
        # interactive núcleo and its selective hit-testing handles clicks.
        self._orb.raise_()

    def _hide_stage(self) -> None:
        if self._stage_v2 is not None and self._stage_v2.isVisible():
            self._stage_v2.hide()
        self._hide_controls()
        if self._stage is not None and self._stage.isVisible():
            self._stage.hide()

    def _show_controls(self) -> None:
        if self._controls is not None and not self._controls.isVisible():
            self._controls.show()

    def _hide_controls(self) -> None:
        if self._controls is not None and self._controls.isVisible():
            self._controls.hide()

    def _set_stage_voice_active(self, active: bool) -> None:
        if self._stage is not None:
            self._stage.set_voice_active(active)

    def _start_chat_hotkey(self) -> None:
        if self._chat_hotkey is None:
            self._chat_hotkey = self._make_hotkey(
                self._request_orb_visible,
                virtual_key=0x20,   # VK_SPACE
                modifiers=0x0002,   # MOD_CONTROL
            )
        registered = self._chat_hotkey.start()
        if registered is True:
            self._logger.info("Listener global Ctrl+Espacio activo")
        elif registered is False:
            self._logger.warning("Listener global Ctrl+Espacio no quedo registrado")

    def _stop_chat_hotkey(self) -> None:
        if self._chat_hotkey is not None:
            self._chat_hotkey.stop()

    def _make_hotkey(self, callback, *, virtual_key: int, modifiers: int):
        """Build a global hotkey; the real Windows one takes vk/modifiers."""
        from functools import partial

        factory = self._hotkey_factory or WindowsGlobalHotkey
        if factory is WindowsGlobalHotkey:
            factory = partial(factory, virtual_key=virtual_key, modifiers=modifiers)
        return factory(callback, logger=self._logger)

    def _set_escape_hotkey(self, enabled: bool) -> None:
        """Register plain ESC globally ONLY while Atlas is visible.

        Windows gives the fullscreen overlay no keyboard focus, so the Qt
        event filter never sees ESC when another app holds focus; a real
        Win32 hotkey does. Hidden Atlas must never intercept ESC.
        """
        if enabled:
            if self._escape_hotkey is None:
                self._escape_hotkey = self._make_hotkey(
                    self._request_escape_hide, virtual_key=0x1B, modifiers=0x0000
                )
            registered = self._escape_hotkey.start()
            if registered is False:
                self._logger.warning("Listener global ESC no quedo registrado")
        elif self._escape_hotkey is not None:
            self._escape_hotkey.stop()
            self._escape_hotkey = None

    def _request_orb_visible(self) -> None:
        self._logger.info("Callback de Ctrl+Espacio recibido; solicitando Orbe en hilo UI")
        self._chat_hotkey_bridge.activated.emit()

    def _request_escape_hide(self) -> None:
        self._logger.info("Callback de ESC recibido; solicitando ocultar en hilo UI")
        self._escape_hotkey_bridge.activated.emit()
