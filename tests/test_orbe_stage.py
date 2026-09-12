"""Focal safety tests for the fullscreen stage (Orbe V5, NO-GO fix).

Offscreen-only. Contract under test: the fullscreen stage is a pure
visual, permanently click-through layer; the ONLY interactive surface is
``OrbeStageControls`` (masked buttons); and every failsafe exit
(ESC / Ctrl+Espacio / X / hide) works in any Atlas state.
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

pytest.importorskip("PySide6")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QEvent, QPoint, Qt
from PySide6.QtGui import QKeyEvent

from ui import orbe_app, orbe_controller, orbe_stage
from use_cases.ui_state_mapper import OrbVisualState


@pytest.fixture(scope="module")
def qapp():
    return orbe_app.create_application([])


@pytest.fixture()
def stage(qapp):
    orb = orbe_app.create_orb_window()
    window = orbe_stage.OrbeStage(orb)
    yield window
    window.hide()
    orb.close()
    window.deleteLater()


@pytest.fixture()
def controls(qapp, stage):
    layer = orbe_stage.OrbeStageControls(stage)
    layer.resize(800, 600)
    layer._relayout()
    yield layer
    layer.hide()
    layer.deleteLater()


@pytest.fixture()
def controller(qapp):
    orb = orbe_app.create_orb_window()
    panel = orbe_app.create_transcript_panel()
    instance = orbe_controller.OrbeController(
        atlas=SimpleNamespace(), application=qapp, orb=orb, transcript_panel=panel
    )
    yield instance
    instance.hide_interface()
    orb.close()
    panel.close()


def _drain(qapp) -> None:
    qapp.processEvents()


def _escape_event() -> QKeyEvent:
    return QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Escape, Qt.KeyboardModifier.NoModifier)


# ------------------------------------------------------------------ metrics


def test_real_metrics_are_safe_or_none() -> None:
    memory = orbe_stage.system_memory_percent()
    cpu = orbe_stage.system_cpu_percent()
    assert memory is None or 0.0 <= memory <= 100.0
    assert cpu is None or 0.0 <= cpu <= 100.0


def test_agent_names_read_the_real_atlas_core() -> None:
    atlas = SimpleNamespace(agents={"PLANIFICADOR": object(), "CODING": object()})
    assert orbe_stage._agent_names(atlas) == ["PLANIFICADOR", "CODING"]
    assert orbe_stage._agent_names(SimpleNamespace()) == []


# ------------------------------------------------------------------ A: fondo


def test_a_stage_background_never_captures_input(stage) -> None:
    """The fullscreen stage is visual-only: Qt-level and design-level."""
    assert stage.testAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
    # The legacy polling hit-test and its giant disc hitbox are gone.
    assert not hasattr(stage, "clickable_at")
    assert getattr(stage, "_hit_timer", None) is None


def test_stage_is_frameless_translucent_overlay(stage) -> None:
    assert bool(stage.windowFlags() & Qt.WindowType.FramelessWindowHint)
    assert stage.testAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)


# ------------------------------------------------------------------ B: controles


def test_b_controls_buttons_are_the_only_interactive_surface(controls) -> None:
    received: list[str] = []
    controls.capability_selected.connect(received.append)
    for index, (capability_id, _label) in enumerate(orbe_stage._CAPABILITY_OPTIONS):
        if capability_id in ("chat", "voz"):
            continue
        controls._menu_buttons[index].click()
    assert received == [
        "control_pc", "entrenamiento", "nutricion", "salud",
        "coding", "proyectos", "mas_herramientas",
    ]


def test_b_mask_limits_input_to_real_buttons(controls) -> None:
    mask = controls.mask()
    close_rect = controls.close_button_rect()
    assert mask.contains(close_rect.center())
    for rect in controls.menu_button_rects():
        assert mask.contains(rect.center())
    # Empty areas (left middle, centre) pass clicks to Windows.
    assert not mask.contains(QPoint(10, controls.height() // 2))
    assert not mask.contains(QPoint(controls.width() // 2, controls.height() // 3))


def test_b_menu_chat_and_voice_use_dedicated_signals(controls) -> None:
    chats: list[bool] = []
    voices: list[bool] = []
    controls.chat_requested.connect(lambda: chats.append(True))
    controls.voice_requested.connect(lambda: voices.append(True))
    controls._menu_buttons[0].click()  # chat
    controls._menu_buttons[1].click()  # voz
    assert chats == [True]
    assert voices == [True]


# ------------------------------------------------------------------ states


def test_stage_starts_idle_and_mirrors_real_states(stage) -> None:
    assert stage.state is OrbVisualState.IDLE
    identity = id(stage)
    for state in (
        OrbVisualState.LISTENING,
        OrbVisualState.PROCESSING,
        OrbVisualState.SPEAKING,
        OrbVisualState.AUTOMATION,
        OrbVisualState.IDLE,
    ):
        stage.apply_state(state)
        assert stage.state is state
    assert id(stage) == identity


def test_all_states_render_without_errors(stage) -> None:
    for state in OrbVisualState:
        stage.apply_state(state)
        stage.repaint()  # paintEvent must not raise for any state
        assert stage.state is state


def test_voice_active_and_error_signal_are_pure_visual_state(stage) -> None:
    stage.set_voice_active(True)
    stage.set_last_error("fallo de red simulado")
    stage.repaint()
    stage.set_voice_active(False)
    stage.set_last_error("")
    stage.repaint()


def test_automation_panel_is_red_only_while_executing(stage) -> None:
    stage.apply_state(OrbVisualState.IDLE)
    assert stage._state_accent() != (255, 82, 82)
    stage.apply_state(OrbVisualState.AUTOMATION)
    assert stage._state_accent() == (255, 82, 82)


# ------------------------------------------------------------------ C: ESC


def test_c_escape_hides_stage_and_orb(qapp, controller) -> None:
    controller.show_orb()
    assert controller.stage.isVisible()
    controller._escape_filter.eventFilter(None, _escape_event())
    assert not controller.stage.isVisible()
    assert not controller._orb.isVisible()


def test_c_controls_escape_emits_hide_requested(controls) -> None:
    received: list[bool] = []
    controls.hide_requested.connect(lambda: received.append(True))
    controls.keyPressEvent(_escape_event())
    assert received == [True]


# ------------------------------------------------------------------ D: Ctrl+Espacio


def test_d_ctrl_space_toggles_the_whole_interface(qapp, controller) -> None:
    controller._chat_hotkey_bridge.activated.emit()  # show
    _drain(qapp)
    assert controller.stage.isVisible()
    assert controller._orb.isVisible()
    assert controller.controls.isVisible()
    controller._chat_hotkey_bridge.activated.emit()  # hide
    _drain(qapp)
    assert not controller.stage.isVisible()
    assert not controller._orb.isVisible()
    assert not controller.controls.isVisible()


# ------------------------------------------------------------------ E: botón X


def test_e_close_button_hides_the_interface(qapp, controller) -> None:
    controller.show_orb()
    assert controller.stage.isVisible()
    controller.controls._close_button.click()
    assert not controller.stage.isVisible()
    assert not controller._orb.isVisible()


# ------------------------------------------------------------------ F: CHAT


def test_f_chat_opens_the_real_panel_in_front_of_the_stage(qapp, controller) -> None:
    controller.show_orb()
    controller.controls.chat_requested.emit()
    _drain(qapp)
    # The real existing transcript panel is visible and usable.
    assert controller._transcript_panel.isVisible()
    # The topmost stage must step aside so the chat is not trapped behind it.
    assert not controller.stage.isVisible()


def test_f_capability_routes_through_existing_chat(qapp, controller) -> None:
    controller.controls.capability_selected.emit("coding")
    _drain(qapp)
    assert controller._transcript_panel.isVisible()
    assert controller._transcript_panel._input.text().startswith("Coding: ")


# ------------------------------------------------------------------ G: estados críticos


def test_g_escape_and_hotkey_work_during_automation(qapp, controller) -> None:
    controller.show_orb()
    controller._orb.apply_state(OrbVisualState.AUTOMATION)
    _drain(qapp)
    controller._escape_filter.eventFilter(None, _escape_event())
    assert not controller.stage.isVisible()

    controller._chat_hotkey_bridge.activated.emit()  # show again
    _drain(qapp)
    controller._orb.apply_state(OrbVisualState.PROCESSING)
    _drain(qapp)
    controller._chat_hotkey_bridge.activated.emit()  # toggle hides
    _drain(qapp)
    assert not controller.stage.isVisible()


# ------------------------------------------------------------------ H: timers


def test_h_hide_stops_stage_timers(stage) -> None:
    stage.show()
    assert stage._timer.isActive()
    stage.hide()
    assert not stage._timer.isActive()
    assert not stage._metrics_timer.isActive()
    assert not stage._clock_timer.isActive()


def test_h_stage_close_never_destroys_the_overlay(stage) -> None:
    stage.show()
    stage.close()
    assert not stage.isVisible()  # hidden, not destroyed: no zombie, no crash
