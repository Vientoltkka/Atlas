"""Integration tests for V4.3-I3: real core states reaching the orb."""

from __future__ import annotations

import os
import threading
from datetime import datetime
from types import SimpleNamespace

import pytest

pytest.importorskip("PySide6")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import main as atlas_main
from use_cases.voice_conversation import (
    VoiceConversationSession,
    VoiceConversationState,
    VoiceConversationUseCase,
)
from ui.atlas_bridge import AtlasUiBridge
from ui.orbe_app import create_application, create_orb_window


@pytest.fixture(scope="module")
def qapp():
    app = create_application([])
    yield app


def _drain_events(app, rounds: int = 20) -> None:
    for _ in range(rounds):
        app.processEvents()


def make_core_with_listener(listener):
    use_case = VoiceConversationUseCase(
        speech_engine=SimpleNamespace(),
        wake_word_engine=None,
        speech_output_engine=None,
        diagnostics_enabled=False,
        clock=lambda: 0.0,
        now_provider=lambda: datetime.now().astimezone(),
    )
    session = VoiceConversationSession(session_id="ui-test", started_at=0.0)
    session.state = VoiceConversationState.READY
    return use_case, session


def test_real_core_state_reaches_the_orb(qapp) -> None:
    from use_cases.ui_state_mapper import OrbVisualState

    orb = create_orb_window()
    bridge = AtlasUiBridge()
    bridge.state_changed.connect(orb.apply_state)

    use_case, session = make_core_with_listener(bridge.on_state)
    # Wire the same hook the real execute_manual wires (session-scoped).
    use_case._session_state_listener = bridge.on_state

    use_case._set_state(session, VoiceConversationState.LISTENING)
    _drain_events(qapp)
    assert orb.state is OrbVisualState.LISTENING

    use_case._set_state(session, VoiceConversationState.PROCESSING)
    _drain_events(qapp)
    assert orb.state is OrbVisualState.PROCESSING
    orb.close()


def test_bridge_crosses_threads_via_signals(qapp) -> None:
    from use_cases.ui_state_mapper import OrbVisualState

    orb = create_orb_window()
    bridge = AtlasUiBridge()
    bridge.state_changed.connect(orb.apply_state)

    def emit_from_thread() -> None:
        bridge.on_state(VoiceConversationState.SPEAKING)

    thread = threading.Thread(target=emit_from_thread)
    thread.start()
    while thread.is_alive():
        qapp.processEvents()
    qapp.processEvents()

    assert orb.state is OrbVisualState.SPEAKING
    orb.close()


def test_unknown_core_state_is_ignored_by_bridge(qapp) -> None:
    messages: list[str] = []
    bridge = AtlasUiBridge()
    bridge.message_received.connect(messages.append)

    bridge.on_state("ESTADO_INVENTADO")

    _drain_events(qapp)
    assert messages == []


def test_requested_mode_ui_and_regressions() -> None:
    args_ui = SimpleNamespace(
        list_microphones=False,
        test_microphone=None,
        whatsapp_webhook=False,
        ui=True,
        voice=False,
        assistant=False,
    )
    args_voice = SimpleNamespace(
        list_microphones=False,
        test_microphone=None,
        whatsapp_webhook=False,
        ui=False,
        voice=True,
        assistant=False,
    )
    args_assistant = SimpleNamespace(
        list_microphones=False,
        test_microphone=None,
        whatsapp_webhook=False,
        ui=False,
        voice=False,
        assistant=True,
    )

    assert atlas_main._requested_mode(args_ui) == "ui"
    assert atlas_main._requested_mode(args_voice) == "voice"
    assert atlas_main._requested_mode(args_assistant) == "assistant"


def test_stop_order_produces_close_command_for_session() -> None:
    stop_event = threading.Event()
    typed_input = lambda: "salir" if stop_event.is_set() else None  # noqa: E731

    assert typed_input() is None
    stop_event.set()
    assert typed_input() == "salir"


def test_quit_flow_marks_and_quits_only_on_finish(qapp) -> None:
    quit_called: list[int] = []
    bridge = AtlasUiBridge()
    orb = create_orb_window()

    stop_event = threading.Event()
    app_proxy = SimpleNamespace(quit=lambda: quit_called.append(1))

    def on_session_finished() -> None:
        if bridge.quit_on_finish:
            app_proxy.quit()

    bridge.session_finished.connect(on_session_finished)
    orb.quit_requested.connect(stop_event.set)
    orb.quit_requested.connect(bridge.request_quit_on_finish)

    # User picks "Salir": stop requested, quit armed but NOT executed yet.
    orb.quit_requested.emit()
    assert stop_event.is_set()
    assert bridge.quit_on_finish is True
    assert quit_called == []

    # Session ends through its natural path -> only then Qt quits.
    bridge.notify_session_finished()
    _drain_events(qapp)
    assert quit_called == [1]
    orb.close()


def test_detener_stops_session_without_closing_app(qapp) -> None:
    quit_called: list[int] = []
    bridge = AtlasUiBridge()
    bridge.session_finished.connect(lambda: quit_called.append(1))

    stop_event = threading.Event()
    orb = create_orb_window()
    orb.stop_requested.connect(stop_event.set)

    orb.stop_requested.emit()
    assert stop_event.is_set()
    assert bridge.quit_on_finish is False
    orb.close()


def test_transcript_panel_appends_without_session_logic(qapp) -> None:
    from ui.orbe_app import create_transcript_panel

    panel = create_transcript_panel()
    panel.append_message("Estado: LISTENING")
    panel.append_message("Estado: PROCESSING")
    assert "LISTENING" in panel._view.toPlainText()
    panel.close()


def test_controller_delivers_worker_events_only_through_qt_signals(qapp) -> None:
    from ui.orbe_controller import OrbeController
    from use_cases.ui_state_mapper import OrbVisualState
    from ui.orbe_app import create_transcript_panel

    class FakeAtlas:
        def start_voice(self, *, state_listener, status_sink, typed_input) -> None:
            state_listener(VoiceConversationState.LISTENING)
            status_sink("Tú: prueba desde worker")
            status_sink("Atlas: respuesta desde worker")
            assert typed_input() is None

    orb = create_orb_window()
    panel = create_transcript_panel()
    controller = OrbeController(
        atlas=FakeAtlas(), application=qapp, orb=orb, transcript_panel=panel
    )
    controller.start()
    controller.join(timeout=2)
    _drain_events(qapp)

    assert orb.state is OrbVisualState.IDLE
    content = panel._view.toPlainText()
    assert "prueba desde worker" in content
    assert "Atlas: respuesta desde worker" in content
    orb.close()
    panel.close()

def test_text_chat_submits_without_blocking_and_renders_response(qapp) -> None:
    from ui.orbe_controller import OrbeController
    from ui.orbe_app import create_transcript_panel

    started = threading.Event()
    release = threading.Event()

    class FakeAtlas:
        def start_voice(self, **_kwargs) -> None:
            return None

        def process_prompt(self, prompt: str) -> str:
            started.set()
            assert release.wait(timeout=2)
            return f"eco: {prompt}"

    orb = create_orb_window()
    panel = create_transcript_panel()
    controller = OrbeController(
        atlas=FakeAtlas(), application=qapp, orb=orb, transcript_panel=panel
    )
    panel._input.setText("hola Orbe")
    panel._send_button.click()
    _drain_events(qapp)

    assert started.wait(timeout=1)
    assert "Usuario: hola Orbe" in panel._view.toPlainText()
    assert "eco: hola Orbe" not in panel._view.toPlainText()

    release.set()
    controller.join_chat(timeout=2)
    _drain_events(qapp)
    assert "Atlas: eco: hola Orbe" in panel._view.toPlainText()
    orb.close()
    panel.close()


def test_text_chat_errors_are_rendered_by_bridge_signal(qapp) -> None:
    from ui.orbe_controller import OrbeController
    from ui.orbe_app import create_transcript_panel

    class FakeAtlas:
        def process_prompt(self, _prompt: str) -> str:
            raise RuntimeError("fallo esperado")

    orb = create_orb_window()
    panel = create_transcript_panel()
    controller = OrbeController(
        atlas=FakeAtlas(), application=qapp, orb=orb, transcript_panel=panel
    )
    controller.submit_text("provoca fallo")
    controller.join_chat(timeout=2)
    _drain_events(qapp)

    assert "Error: No se pudo procesar el mensaje textual." in panel._view.toPlainText()
    orb.close()
    panel.close()

def test_voice_controls_stop_and_retry_without_touching_chat(qapp) -> None:
    from ui.orbe_controller import OrbeController
    from ui.orbe_app import create_transcript_panel

    entered = threading.Event()
    calls: list[int] = []

    class FakeAtlas:
        def start_voice(self, *, state_listener, status_sink, typed_input) -> None:
            calls.append(1)
            state_listener(VoiceConversationState.TRANSCRIBING)
            state_listener(VoiceConversationState.PROCESSING)
            state_listener(VoiceConversationState.SPEAKING)
            entered.set()
            while typed_input() is None:
                threading.Event().wait(0.01)

    orb = create_orb_window()
    panel = create_transcript_panel()
    controller = OrbeController(
        atlas=FakeAtlas(), application=qapp, orb=orb, transcript_panel=panel
    )
    controller.start()
    assert entered.wait(timeout=1)
    _drain_events(qapp)
    assert panel._voice_status.text() == "TTS"

    panel._voice_stop_button.click()
    controller.join(timeout=2)
    _drain_events(qapp)
    assert panel._voice_status.text() == "Desconectado"

    entered.clear()
    panel._voice_retry_button.click()
    assert entered.wait(timeout=1)
    assert len(calls) == 2
    controller.stop()
    controller.join(timeout=2)
    orb.close()
    panel.close()


def test_voice_statuses_are_distinguishable_through_bridge_signals(qapp) -> None:
    from ui.orbe_app import create_transcript_panel

    panel = create_transcript_panel()
    bridge = AtlasUiBridge()
    bridge.voice_state_changed.connect(panel.set_voice_state)
    bridge.voice_disconnected.connect(panel.set_voice_disconnected)

    for state, label in (
        (VoiceConversationState.TRANSCRIBING, "STT"),
        (VoiceConversationState.PROCESSING, "Procesando"),
        (VoiceConversationState.SPEAKING, "TTS"),
        (VoiceConversationState.DEGRADED, "Error"),
    ):
        bridge.on_state(state)
        _drain_events(qapp)
        assert panel._voice_status.text() == label

    bridge.notify_session_finished()
    _drain_events(qapp)
    assert panel._voice_status.text() == "Desconectado"
    panel.close()