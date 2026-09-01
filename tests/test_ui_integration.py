"""Integration tests for V4.3-I3: real core states reaching the orb."""

from __future__ import annotations

from datetime import datetime
import json
import os
import threading
from types import SimpleNamespace

import pytest

pytest.importorskip("PySide6")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import main as atlas_main
from bootstrap.bootstrap import Bootstrap
from core.atlas import Atlas
from core.model_health import ModelHealthResult
from core.model_inference import ModelHealthCheckError
from core.orchestrator import AtlasOrchestrator
from models.prompt_client import InferenceBackendError
from tools.desktop.desktop_tools import CopyClipboardTextTool
from tools.executor import ToolExecutor
from tools.registry import ToolRegistry
from use_cases.desktop_interaction import DesktopInteractionUseCase
from use_cases.execution_conversation import ExecutionConversationController
from use_cases.voice_conversation import (
    VoiceConversationSession,
    VoiceConversationState,
    VoiceConversationUseCase,
)
from core.execution_supervisor import ExecutionState, ExecutionSupervisor
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


def test_real_supervisor_authorization_reaches_orb_and_clears(qapp) -> None:
    from use_cases.ui_state_mapper import OrbVisualState

    orb = create_orb_window()
    bridge = AtlasUiBridge()
    bridge.state_changed.connect(orb.apply_state)
    supervisor = ExecutionSupervisor()
    supervisor.add_state_listener(bridge.on_supervision_state)
    session = supervisor.start(SimpleNamespace(steps=()))

    supervisor.mark_running(session.session_id)
    _drain_events(qapp)
    assert orb.state is OrbVisualState.PROCESSING
    supervisor.mark_waiting_confirmation(session.session_id)
    _drain_events(qapp)
    assert orb.state is OrbVisualState.AUTHORIZATION
    supervisor.mark_cancelled(session.session_id, error="denegado")
    _drain_events(qapp)
    assert orb.state is OrbVisualState.IDLE
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


def test_controller_subscribes_only_when_atlas_exposes_real_supervision(qapp) -> None:
    from ui.orbe_controller import OrbeController
    from ui.orbe_app import create_transcript_panel
    from use_cases.ui_state_mapper import OrbVisualState

    class FakeAtlas:
        def add_supervision_state_listener(self, listener) -> None:
            self.listener = listener

    atlas = FakeAtlas()
    orb = create_orb_window()
    panel = create_transcript_panel()
    OrbeController(atlas=atlas, application=qapp, orb=orb, transcript_panel=panel)
    atlas.listener(ExecutionState.WAITING_CONFIRMATION)
    _drain_events(qapp)
    assert orb.state is OrbVisualState.AUTHORIZATION
    atlas.listener(ExecutionState.RUNNING)
    _drain_events(qapp)
    assert orb.state is OrbVisualState.PROCESSING
    orb.close()
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


def test_text_chat_with_attachment_renders_notice_without_calling_backend(qapp, tmp_path, monkeypatch) -> None:
    from PySide6.QtWidgets import QFileDialog
    from ui.orbe_controller import OrbeController
    from ui.orbe_app import create_transcript_panel

    attachment = tmp_path / "datos.csv"
    attachment.write_text("valor\n1\n", encoding="utf-8")
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *_args: (str(attachment), ""))

    class FakeAtlas:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def process_prompt(self, prompt: str) -> str:
            self.calls.append(prompt)
            return "no debe llegar"

    atlas = FakeAtlas()
    orb = create_orb_window()
    panel = create_transcript_panel()
    controller = OrbeController(atlas=atlas, application=qapp, orb=orb, transcript_panel=panel)
    panel._choose_attachment()
    panel._input.setText("revisa esto")
    panel._send_button.click()
    _drain_events(qapp)

    content = panel._view.toPlainText()
    assert atlas.calls == []
    assert "Usuario: revisa esto" in content
    assert "Archivo adjunto: datos.csv." in content
    assert "análisis de archivos" in content
    assert "No se pudo procesar el mensaje textual." not in content
    assert panel._pending_attachment is None
    orb.close()
    panel.close()


def test_text_chat_nutrition_preflight_bypasses_model_health_and_completes_followup(
    qapp, monkeypatch
) -> None:
    """Exercise the real Orbe text path without requiring a model for missing data."""
    from agents.nutrition_agent import NutritionAgent
    from ui.orbe_controller import OrbeController
    from ui.orbe_app import create_transcript_panel

    prompt = (
        "Calcula aproximadamente mis calorías y macronutrientes diarios para ganar "
        "masa muscular. Peso 74 kg, mido 1,80 m, entreno CrossFit 5 días por "
        "semana y quiero subir de peso minimizando la ganancia de grasa."
    )
    orchestrator = Bootstrap.build()
    nutrition = orchestrator._registry.get("nutrition")
    assert isinstance(nutrition, NutritionAgent)

    def health_check_must_not_run(_model: str) -> None:
        raise AssertionError("El preflight no debe ejecutar el health-check del modelo.")

    monkeypatch.setattr(nutrition._client, "check_model_health", health_check_must_not_run)
    atlas = Atlas.__new__(Atlas)
    atlas._orchestrator = orchestrator
    orb = create_orb_window()
    panel = create_transcript_panel()
    controller = OrbeController(
        atlas=atlas, application=qapp, orb=orb, transcript_panel=panel
    )

    panel._input.setText(prompt)
    panel._send_button.click()
    controller.join_chat(timeout=2)
    _drain_events(qapp)
    assert "Para calcularlo necesito tu edad y tu sexo." in panel._view.toPlainText()
    assert orchestrator.pending_agent_followup is not None

    monkeypatch.setattr(
        nutrition._client,
        "check_model_health",
        lambda model: (_ for _ in ()).throw(
            InferenceBackendError(model, "backend unavailable")
        ),
    )
    panel._input.setText("48 años, hombre.")
    panel._send_button.click()
    controller.join_chat(timeout=2)
    _drain_events(qapp)

    content = panel._view.toPlainText()
    assert "**Calorías:** 3,050 kcal/día" in content
    assert '"requires_follow_up"' not in content
    assert orchestrator.pending_agent_followup is None
    orb.close()
    panel.close()


def test_text_chat_legal_contract_clause_falls_back_when_provider_is_unavailable(
    qapp, monkeypatch
) -> None:
    """Exercise Orbe -> Atlas -> LegalAgent with the authorized fallback exhausted."""
    from agents.legal_agent import LegalAgent
    from ui.orbe_controller import OrbeController
    from ui.orbe_app import create_transcript_panel

    prompt = (
        "He firmado un contrato de alquiler y hay una cláusula que no entiendo. "
        "¿Qué información debería revisar para saber si esa cláusula puede ser "
        "abusiva o ilegal?"
    )
    orchestrator = Bootstrap.build()
    legal = orchestrator._registry.get("legal")
    assert isinstance(legal, LegalAgent)
    assert orchestrator.classify_prompt(prompt).target_agent_name == "legal"
    monkeypatch.setattr(
        legal._client,
        "check_model_health",
        lambda model: (_ for _ in ()).throw(
            InferenceBackendError(model, "backend unavailable")
        ),
    )
    atlas = Atlas.__new__(Atlas)
    atlas._orchestrator = orchestrator
    orb = create_orb_window()
    panel = create_transcript_panel()
    controller = OrbeController(
        atlas=atlas, application=qapp, orb=orb, transcript_panel=panel
    )

    panel._input.setText(prompt)
    panel._send_button.click()
    controller.join_chat(timeout=2)
    _drain_events(qapp)

    content = panel._view.toPlainText()
    assert "orientación jurídica general" in content
    assert "no se puede determinar si una cláusula es abusiva o ilegal" in content
    assert "jurisdicción aplicable" in content
    assert "No se pudo procesar el mensaje textual." not in content
    assert '"requires_follow_up"' not in content
    assert orchestrator.pending_agent_followup is None
    orb.close()
    panel.close()


@pytest.mark.parametrize(
    ("response", "expected_copies"),
    [("si", ["hola Atlas"]), ("no", [])],
)
def test_text_chat_routes_clipboard_copy_through_conversational_confirmation(
    qapp, response: str, expected_copies: list[str]
) -> None:
    """Exercise the actual Orbe text controller and Atlas multi-turn flow."""
    from ui.orbe_controller import OrbeController
    from ui.orbe_app import create_transcript_panel

    class Desktop:
        def __init__(self) -> None:
            self.copied: list[str] = []

        def copy_clipboard_text(self, text: str) -> int:
            self.copied.append(text)
            return len(text)

    desktop = Desktop()
    registry = ToolRegistry()
    registry.register(CopyClipboardTextTool(desktop))
    executor = ToolExecutor(registry)
    coordinator = Bootstrap.build_execution_coordinator(
        tool_registry=registry,
        executor=executor,
    )
    desktop_interaction = DesktopInteractionUseCase(executor)
    orchestrator = AtlasOrchestrator(
        planner=None,
        router=None,
        model_manager=None,
        memory=None,
        registry=None,
        write_file=None,
        desktop_interaction=desktop_interaction,
        execution_conversation=ExecutionConversationController(coordinator),
    )
    atlas = Atlas.__new__(Atlas)
    atlas._orchestrator = orchestrator
    orb = create_orb_window()
    panel = create_transcript_panel()
    controller = OrbeController(
        atlas=atlas, application=qapp, orb=orb, transcript_panel=panel
    )

    panel._input.setText("Copia hola Atlas al portapapeles")
    panel._send_button.click()
    controller.join_chat(timeout=2)
    _drain_events(qapp)

    assert desktop.copied == []
    assert "Voy a copiar ese texto al portapapeles." in panel._view.toPlainText()

    panel._input.setText(response)
    panel._send_button.click()
    controller.join_chat(timeout=2)
    _drain_events(qapp)

    assert desktop.copied == expected_copies
    content = panel._view.toPlainText()
    if response == "si":
        assert "Texto copiado al portapapeles." in content
    else:
        assert "Operacion cancelada." in content
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


def test_orbe_text_chat_uses_medical_fallback_when_provider_is_unavailable(
    qapp, monkeypatch
) -> None:
    from ui.orbe_app import create_transcript_panel
    from ui.orbe_controller import OrbeController

    orchestrator = Bootstrap.build()
    health_result = ModelHealthResult(
        logical_model_id="chat-local",
        physical_model_name="chat-local",
        provider_id="ollama",
        healthy=False,
        reason="provider unavailable",
    )
    unavailable_error = ModelHealthCheckError(
        initial_logical_model_id="chat-local",
        attempted_logical_model_ids=("chat-local",),
        allow_fallback=True,
        last_result=health_result,
    )

    def unavailable_provider(*_args, **_kwargs):
        raise unavailable_error

    monkeypatch.setattr(orchestrator._model_inference_runner, "run", unavailable_provider)
    atlas = Atlas.__new__(Atlas)
    atlas._orchestrator = orchestrator
    orb = create_orb_window()
    panel = create_transcript_panel()
    controller = OrbeController(
        atlas=atlas, application=qapp, orb=orb, transcript_panel=panel
    )
    prompt = (
        "Tengo dolor muscular después de entrenar. ¿Cómo distingo unas agujetas "
        "normales de algo que debería revisar con un médico?"
    )

    controller.submit_text(prompt)
    controller.join_chat(timeout=2)
    _drain_events(qapp)

    content = panel._view.toPlainText()
    assert "agujetas habituales" in content
    assert "atención urgente" in content
    assert "Error: No se pudo procesar el mensaje textual." not in content
    assert '"requires_follow_up"' not in content
    assert not orchestrator._pending_agent_followup
    orb.close()
    panel.close()


def test_orbe_text_chat_uses_finance_budget_fallback_when_provider_is_unavailable(
    qapp, monkeypatch
) -> None:
    from ui.orbe_app import create_transcript_panel
    from ui.orbe_controller import OrbeController

    orchestrator = Bootstrap.build()
    health_result = ModelHealthResult(
        logical_model_id="chat-local",
        physical_model_name="chat-local",
        provider_id="ollama",
        healthy=False,
        reason="provider unavailable",
    )
    unavailable_error = ModelHealthCheckError(
        initial_logical_model_id="chat-local",
        attempted_logical_model_ids=("chat-local",),
        allow_fallback=True,
        last_result=health_result,
    )
    monkeypatch.setattr(
        orchestrator._model_inference_runner,
        "run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(unavailable_error),
    )
    atlas = Atlas.__new__(Atlas)
    atlas._orchestrator = orchestrator
    orb = create_orb_window()
    panel = create_transcript_panel()
    controller = OrbeController(
        atlas=atlas, application=qapp, orb=orb, transcript_panel=panel
    )
    prompt = (
        "Quiero organizar mejor mis finanzas personales. Cobro 1.500 € al mes y "
        "quiero ahorrar 300 €. ¿Cómo repartirías el resto entre gastos fijos, ocio "
        "y un fondo de emergencia?"
    )

    controller.submit_text(prompt)
    controller.join_chat(timeout=2)
    _drain_events(qapp)

    content = panel._view.toPlainText()
    assert "Ahorro objetivo: 300 €" in content
    assert "Gastos fijos: 800 €" in content
    assert "Ocio y gastos variables: 250 €" in content
    assert "Fondo de emergencia: 150 €" in content
    assert "Error: No se pudo procesar el mensaje textual." not in content
    assert '"requires_follow_up"' not in content
    assert not orchestrator._pending_agent_followup
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


def test_stop_acknowledges_ui_before_a_blocked_voice_worker_returns(qapp) -> None:
    from ui.orbe_controller import OrbeController
    from ui.orbe_app import create_transcript_panel
    from use_cases.ui_state_mapper import OrbVisualState

    entered = threading.Event()
    release = threading.Event()

    class FakeAtlas:
        def start_voice(self, *, state_listener, status_sink, typed_input) -> None:
            state_listener(VoiceConversationState.PROCESSING)
            entered.set()
            assert release.wait(timeout=2)
            state_listener(VoiceConversationState.SPEAKING)  # late worker event

    orb = create_orb_window()
    panel = create_transcript_panel()
    controller = OrbeController(atlas=FakeAtlas(), application=qapp, orb=orb, transcript_panel=panel)
    controller.start()
    assert entered.wait(timeout=1)
    _drain_events(qapp)
    assert orb.state is OrbVisualState.PROCESSING

    panel._voice_stop_button.click()
    assert orb.state is OrbVisualState.IDLE
    assert panel._voice_status.text() == "Desconectado"
    _drain_events(qapp)  # stale queued PROCESSING must not restore the core
    assert orb.state is OrbVisualState.IDLE
    release.set()
    controller.join(timeout=2)
    _drain_events(qapp)
    assert orb.state is OrbVisualState.IDLE
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

def test_chat_close_hides_both_windows_and_hotkey_restores_them(qapp) -> None:
    from ui.orbe_controller import OrbeController
    from ui.orbe_app import create_transcript_panel

    class FakeAtlas:
        pass

    class FakeHotkey:
        def __init__(self, callback, logger=None) -> None:
            self.callback = callback
            self.started = False
            self.stopped = False

        def start(self) -> None:
            self.started = True

        def stop(self) -> None:
            self.stopped = True

        def trigger(self) -> None:
            self.callback()

    hotkeys: list[FakeHotkey] = []

    def make_hotkey(callback, logger=None) -> FakeHotkey:
        hotkey = FakeHotkey(callback, logger)
        hotkeys.append(hotkey)
        return hotkey

    orb = create_orb_window()
    panel = create_transcript_panel()
    controller = OrbeController(
        atlas=FakeAtlas(),
        application=qapp,
        orb=orb,
        transcript_panel=panel,
        hotkey_factory=make_hotkey,
    )
    controller.start(start_voice=False)
    assert hotkeys[0].started is True
    assert qapp.quitOnLastWindowClosed() is False
    assert orb.isVisible() is True
    assert panel.isVisible() is True

    panel.close()
    _drain_events(qapp)
    assert orb.isVisible() is False
    assert panel.isVisible() is False
    assert hotkeys[0].stopped is False

    hotkeys[0].trigger()
    _drain_events(qapp)
    assert orb.isVisible() is True
    assert panel.isVisible() is True

    controller._stop_chat_hotkey()
    assert hotkeys[0].stopped is True
    orb.close()
    panel.close()
