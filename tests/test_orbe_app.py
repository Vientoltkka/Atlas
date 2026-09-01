"""Offscreen tests for the V4.3-I2 orb shell (PySide6)."""

from __future__ import annotations

import os

import pytest

pytest.importorskip("PySide6")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from use_cases.ui_state_mapper import OrbVisualState
from ui import orbe_app


@pytest.fixture(scope="module")
def qapp():
    app = orbe_app.create_application([])
    yield app


@pytest.fixture()
def orb(qapp):
    window = orbe_app.create_orb_window()
    yield window
    window.close()


def test_color_mapping_is_deterministic_and_complete() -> None:
    for state in OrbVisualState:
        first = orbe_app.color_for_state(state)
        second = orbe_app.color_for_state(state)
        assert first == second
        assert len(first) == 4  # RGBA


def test_primary_visual_states_use_the_fixed_semantic_palette() -> None:
    assert orbe_app.color_for_state(OrbVisualState.IDLE)[:3] == (56, 185, 255)
    assert orbe_app.color_for_state(OrbVisualState.PROCESSING)[:3] == (168, 102, 255)
    assert orbe_app.color_for_state(OrbVisualState.SPEAKING)[:3] == (72, 238, 148)
    assert orbe_app.color_for_state(OrbVisualState.AUTOMATION)[:3] == (255, 72, 72)
    assert orbe_app.color_for_state(OrbVisualState.AUTHORIZATION)[:3] == (255, 174, 52)


def test_demo_cycle_covers_every_visual_state() -> None:
    assert set(orbe_app.DEMO_STATE_CYCLE) == set(OrbVisualState)


def test_orb_starts_idle(orb) -> None:
    assert orb.state is OrbVisualState.IDLE


def test_apply_state_updates_visual_state(orb) -> None:
    orb.apply_state(OrbVisualState.LISTENING)
    assert orb.state is OrbVisualState.LISTENING
    orb.apply_state("SPEAKING")  # raw string accepted like the mapper
    assert orb.state is OrbVisualState.SPEAKING


def test_all_states_are_renderable_without_errors(orb, qtbot=None) -> None:
    for state in OrbVisualState:
        orb.apply_state(state)
        orb.repaint()  # paintEvent must not raise for any state
        assert orb.state is state


def test_window_is_frameless_translucent_and_on_top(qapp, orb) -> None:
    from PySide6.QtCore import Qt

    flags = orb.windowFlags()
    assert bool(flags & Qt.WindowType.FramelessWindowHint)
    assert bool(flags & Qt.WindowType.WindowStaysOnTopHint)
    assert orb.testAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)


def test_transcript_panel_is_a_normal_window_and_not_on_top(qapp) -> None:
    from PySide6.QtCore import Qt

    panel = orbe_app.create_transcript_panel()
    flags = panel.windowFlags()

    assert flags & Qt.WindowType.WindowType_Mask == Qt.WindowType.Window
    assert not flags & Qt.WindowType.WindowStaysOnTopHint
    panel.close()


def test_chat_input_is_multiline_and_enter_submits_but_shift_enter_inserts_newline(qapp) -> None:
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest

    panel = orbe_app.create_transcript_panel()
    sent: list[str] = []
    panel.send_requested.connect(sent.append)
    panel._input.setText("primera linea")
    QTest.keyClick(panel._input, Qt.Key.Key_Return, Qt.KeyboardModifier.ShiftModifier)
    assert panel._input.text() == "primera linea\n"
    QTest.keyClick(panel._input, Qt.Key.Key_Return)
    assert sent == ["primera linea"]
    assert panel._input.text() == ""
    QTest.keyClick(panel._input, Qt.Key.Key_Backspace)
    assert panel._input.text() == ""
    panel.close()


def test_empty_chat_input_does_not_send(qapp) -> None:
    panel = orbe_app.create_transcript_panel()
    sent: list[str] = []
    panel.send_requested.connect(sent.append)
    panel._submit_input()
    assert sent == []
    panel.close()


def test_transcript_uses_distinct_escaped_user_and_atlas_turns(qapp) -> None:
    panel = orbe_app.create_transcript_panel()
    panel.append_user("<b>sin formato</b>\nsegunda linea")
    panel.append_response("respuesta")
    rendered = panel._view.toHtml()
    text = panel._view.toPlainText()
    assert "Usuario:" in text
    assert "Atlas:" in text
    assert "&lt;b&gt;sin formato&lt;/b&gt;" in rendered
    assert "segunda linea" in text
    assert "#123b5c" in rendered
    assert "#162235" in rendered
    assert rendered.count("<table") == 2
    panel.close()


def test_transcript_hides_repetitive_voice_states_but_keeps_errors_and_transcription(qapp) -> None:
    panel = orbe_app.create_transcript_panel()
    for message in (
        "Estado: STARTING",
        "Estado: IDLE",
        "Estado: LISTENING",
        "Estado: PROCESSING",
        "Estado: RECOVERING",
        "Esperando voz...",
    ):
        panel.append_message(message)
    panel.append_transcription("Qué tal está el día hoy")
    panel.append_error("Error real de voz")
    text = panel._view.toPlainText()
    assert "Estado:" not in text
    assert "Esperando voz" not in text
    assert "Tú: Qué tal está el día hoy" in text
    assert "Error: Error real de voz" in text
    panel.close()


def test_transcript_autoscrolls_only_when_reader_is_at_the_end(qapp) -> None:
    panel = orbe_app.create_transcript_panel()
    panel.show()
    for index in range(40):
        panel.append_response(f"mensaje {index}")
    scrollbar = panel._view.verticalScrollBar()
    assert scrollbar.value() == scrollbar.maximum()
    scrollbar.setValue(0)
    panel.append_response("mensaje mientras leo arriba")
    assert scrollbar.value() == 0
    scrollbar.setValue(scrollbar.maximum())
    panel.append_response("mensaje al final")
    assert scrollbar.value() == scrollbar.maximum()
    panel.close()


def test_attachment_selection_shows_preview_and_remove_clears_it(qapp, tmp_path, monkeypatch) -> None:
    from PySide6.QtWidgets import QFileDialog

    attachment = tmp_path / "nota.txt"
    attachment.write_text("contenido", encoding="utf-8")
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *_args: (str(attachment), ""))
    panel = orbe_app.create_transcript_panel()
    panel._choose_attachment()
    assert panel._pending_attachment.name == "nota.txt"
    assert panel._pending_attachment.media_type == "text/plain"
    assert panel._pending_attachment.size_bytes == len("contenido")
    assert "nota.txt" in panel._attachment_details.text()
    assert "text/plain" in panel._attachment_details.text()
    panel._clear_attachment()
    assert panel._pending_attachment is None
    assert panel._attachment_preview.isHidden()
    panel.close()

def test_drag_moves_window(qapp, orb) -> None:
    from PySide6.QtCore import QEvent, QPointF, Qt
    from PySide6.QtGui import QMouseEvent

    start = orb.frameGeometry().topLeft()

    press = QMouseEvent(
        QEvent.Type.MouseButtonPress,
        QPointF(10, 10),
        QPointF(float(start.x()) + 10, float(start.y()) + 10),
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    orb.mousePressEvent(press)

    target = start + type(start)(50, 40)
    move = QMouseEvent(
        QEvent.Type.MouseMove,
        QPointF(float(target.x()) + 10, float(target.y()) + 10),
        QPointF(float(target.x()) + 10, float(target.y()) + 10),
        Qt.MouseButton.NoButton,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    orb.mouseMoveEvent(move)
    orb.mouseReleaseEvent(
        QMouseEvent(
            QEvent.Type.MouseButtonRelease,
            QPointF(0, 0),
            QPointF(0, 0),
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.NoButton,
            Qt.KeyboardModifier.NoModifier,
        )
    )

    assert orb.frameGeometry().topLeft() == target
    assert not orb.context_menu.isVisible()
    centre = orb.frameGeometry().center()
    orb.apply_state(OrbVisualState.PROCESSING)
    assert orb.frameGeometry().center() == centre


def test_orb_click_toggles_the_custom_context_menu(qapp, orb) -> None:
    from PySide6.QtCore import QEvent, QPointF, Qt
    from PySide6.QtGui import QMouseEvent

    point = orb.frameGeometry().topLeft() + type(orb.pos())(20, 20)
    press = QMouseEvent(
        QEvent.Type.MouseButtonPress, QPointF(20, 20), QPointF(point),
        Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier,
    )
    release = QMouseEvent(
        QEvent.Type.MouseButtonRelease, QPointF(20, 20), QPointF(point),
        Qt.MouseButton.LeftButton, Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier,
    )
    orb.mousePressEvent(press)
    orb.mouseReleaseEvent(release)
    assert orb.context_menu.isVisible()
    orb.mousePressEvent(press)
    orb.mouseReleaseEvent(release)
    assert not orb.context_menu.isVisible()


def test_context_menu_stays_on_screen_and_moves_left_at_right_edge(qapp, orb) -> None:
    bounds = orb.screen().availableGeometry()
    orb.move(bounds.right() - orb.width() + 1, bounds.center().y() - orb.height() // 2)

    orb.toggle_context_menu()

    menu = orb.context_menu
    geometry = menu.frameGeometry()
    assert geometry.left() >= bounds.left()
    assert geometry.right() <= bounds.right()
    assert geometry.top() >= bounds.top()
    assert geometry.bottom() <= bounds.bottom()
    assert geometry.right() < orb.frameGeometry().left()
    menu.hide()


def test_context_menu_actions_emit_existing_orb_signals_and_close(qapp, orb) -> None:
    calls: list[str] = []
    orb.chat_requested.connect(lambda: calls.append("chat"))
    orb.voice_requested.connect(lambda: calls.append("voice"))
    orb.quit_requested.connect(lambda: calls.append("quit"))
    menu = orb.context_menu

    for button, expected in (
        (menu._chat_button, "chat"),
        (menu._voice_button, "voice"),
        (menu._quit_button, "quit"),
    ):
        orb.toggle_context_menu()
        assert menu.isVisible()
        button.click()
        assert calls[-1] == expected
        assert not menu.isVisible()


def test_context_menu_voice_label_reflects_real_controller_state(orb) -> None:
    orb.set_voice_active(True)
    assert orb.context_menu._voice_button.text() == "Detener voz"
    orb.set_voice_active(False)
    assert orb.context_menu._voice_button.text() == "Modo Voz"


def test_fixed_orb_size(orb) -> None:
    assert orbe_app.CORE_RADIUS_FACTOR == pytest.approx(0.275)
    assert orb.width() == orbe_app.ORB_SIZE
    assert 350 <= orbe_app.ORB_SIZE <= 370
    assert orb.height() == orbe_app.ORB_SIZE


def test_orb_repositions_beside_an_overlapping_transcript_panel(qapp, orb) -> None:
    from PySide6.QtCore import Qt

    panel = orbe_app.create_transcript_panel()
    panel.move(400, 100)
    orb.move(420, 100)
    panel.show()

    orb.reposition_beside(panel)

    assert not orb.frameGeometry().intersects(panel.frameGeometry())
    assert orb.windowFlags() & Qt.WindowType.WindowStaysOnTopHint
    panel.close()


# ---------------------------------------------------------------------------
# V4.3-I4: animations, tray, position preferences
# ---------------------------------------------------------------------------


def test_authorization_uses_distinct_amber_animation_without_extra_timer(orb) -> None:
    assert orbe_app.color_for_state(OrbVisualState.AUTHORIZATION)[:3] == (255, 174, 52)
    assert orbe_app.animation_period(OrbVisualState.AUTHORIZATION) == 2.4
    orb.apply_state(OrbVisualState.AUTHORIZATION)
    assert orb._timer.isActive()


def test_listening_keeps_the_core_stable_while_animation_phase_advances() -> None:
    first = orbe_app.animation_frame(OrbVisualState.LISTENING, 0.0)
    second = orbe_app.animation_frame(OrbVisualState.LISTENING, 0.8)
    assert first["scale"] == second["scale"] == 1.0
    assert first["rotation_deg"] != second["rotation_deg"]


def test_animation_rate_is_capped_for_the_desktop_widget() -> None:
    assert orbe_app.ANIMATION_FPS <= 24


def test_degraded_stays_compact_and_visually_separate_from_authorization(orb) -> None:
    orb.apply_state(OrbVisualState.DEGRADED)
    assert orb.width() == orbe_app.ORB_SIZE
    assert orbe_app.color_for_state(OrbVisualState.DEGRADED) != orbe_app.color_for_state(OrbVisualState.AUTHORIZATION)


def test_orb_owns_only_its_existing_animation_timer() -> None:
    source = __import__("inspect").getsource(orbe_app.create_orb_window)
    assert source.count("self._timer = QTimer(self)") == 1


def test_idle_is_compact_and_active_states_are_much_larger(qapp, orb) -> None:
    assert orbe_app.size_for_state(OrbVisualState.IDLE) == 360
    for state in (OrbVisualState.LISTENING, OrbVisualState.PROCESSING, OrbVisualState.SPEAKING, OrbVisualState.AUTHORIZATION, OrbVisualState.AUTOMATION):
        orb.apply_state(state)
        assert orb.width() == orb.height() == orbe_app.size_for_state(state)
        assert orb.width() >= 430


def test_initial_position_centers_orb_on_available_screen(orb) -> None:
    bounds = orb.screen().availableGeometry()
    assert orb.frameGeometry().center() == bounds.center()


def test_visible_chat_does_not_displace_initial_orb_centre(qapp, orb) -> None:
    panel = orbe_app.create_transcript_panel()
    panel.show()
    orb.show()
    qapp.processEvents()

    assert orb.frameGeometry().center() == orb.screen().availableGeometry().center()
    panel.close()


def test_resize_keeps_centre_and_clamps_to_available_screen(orb) -> None:
    bounds = orb.screen().availableGeometry()
    centre = orb.frameGeometry().center()
    orb.apply_state(OrbVisualState.PROCESSING)
    assert orb.frameGeometry().center() == centre
    orb.apply_state(OrbVisualState.SPEAKING)
    assert orb.frameGeometry().center() == centre

    orb.move(bounds.left(), bounds.top())
    orb.apply_state(OrbVisualState.AUTOMATION)
    geometry = orb.frameGeometry()
    assert geometry.left() >= bounds.left()
    assert geometry.top() >= bounds.top()
    assert geometry.right() <= bounds.right()
    assert geometry.bottom() <= bounds.bottom()


def test_active_resize_repositions_beside_visible_transcript(qapp, orb) -> None:
    from PySide6.QtCore import QRect

    class Panel:
        def frameGeometry(self):  # noqa: N802 (Qt API shape)
            return QRect(100, 100, 100, 100)

        def screen(self):
            return orb.screen()

    panel = Panel()
    orb.move(100, 100)
    orb.apply_state(OrbVisualState.PROCESSING)
    orb.reposition_beside(panel)
    assert not orb.frameGeometry().intersects(panel.frameGeometry())

def test_animation_frame_is_pure_and_deterministic() -> None:
    first = orbe_app.animation_frame(OrbVisualState.LISTENING, 0.4)
    second = orbe_app.animation_frame(OrbVisualState.LISTENING, 0.4)
    assert first == second
    assert set(first.keys()) == {"scale", "alpha_factor", "rotation_deg"}


def test_animation_frames_stay_within_sane_bounds_for_every_state() -> None:
    for state in OrbVisualState:
        for step in range(24):
            frame = orbe_app.animation_frame(state, step * 0.1)
            assert 0.85 <= frame["scale"] <= 1.10
            assert 0.30 <= frame["alpha_factor"] <= 1.05
            assert 0.0 <= frame["rotation_deg"] <= 360.0


def test_static_states_have_no_animation() -> None:
    for elapsed in (0.0, 0.7, 3.3):
        for state in (OrbVisualState.DEGRADED,):
            frame = orbe_app.animation_frame(state, elapsed)
            assert frame["scale"] == 1.0
            assert frame["alpha_factor"] == 1.0
            assert frame["rotation_deg"] == 0.0


def test_processing_rotates_and_listening_pulses() -> None:
    quarter = orbe_app.animation_frame(OrbVisualState.PROCESSING, 0.5)
    three_quarters = orbe_app.animation_frame(OrbVisualState.PROCESSING, 1.5)
    # 2 s period: half period -> ~180 degrees of rotation.
    assert quarter["rotation_deg"] == pytest.approx(90.0, abs=2.0)
    assert three_quarters["rotation_deg"] == pytest.approx(270.0, abs=2.0)

    peak = orbe_app.animation_frame(OrbVisualState.SPEAKING, 0.225)
    trough = orbe_app.animation_frame(OrbVisualState.SPEAKING, 0.675)
    assert peak["scale"] > trough["scale"]


def test_apply_state_resets_animation_clock(qapp, orb) -> None:
    from PySide6.QtCore import QTimer

    orb.apply_state(OrbVisualState.LISTENING)
    assert orb._timer.isActive()
    orb.apply_state(OrbVisualState.DEGRADED)  # static state stops the timer
    assert not orb._timer.isActive()


def test_tray_is_created_with_detener_and_salir(qapp, orb) -> None:
    from PySide6.QtWidgets import QSystemTrayIcon

    if not QSystemTrayIcon.isSystemTrayAvailable():
        pytest.skip("system tray not available in this environment")
    assert orb.tray is not None
    actions = [action.text() for action in orb.tray.contextMenu().actions()]
    assert "Detener" in actions
    assert "Salir" in actions


def test_tray_menu_actions_emit_orb_signals(qapp, orb) -> None:
    from PySide6.QtWidgets import QSystemTrayIcon

    if not QSystemTrayIcon.isSystemTrayAvailable():
        pytest.skip("system tray not available in this environment")
    stop_calls: list[int] = []
    quit_calls: list[int] = []
    orb.stop_requested.connect(lambda: stop_calls.append(1))
    orb.quit_requested.connect(lambda: quit_calls.append(1))

    actions = {action.text(): action for action in orb.tray.contextMenu().actions()}
    actions["Detener"].trigger()
    actions["Salir"].trigger()

    assert stop_calls == [1]
    assert quit_calls == [1]


def test_position_roundtrip_with_settings(qapp, tmp_path) -> None:
    from PySide6.QtCore import QSettings

    settings_file = tmp_path / "orb.ini"
    settings = QSettings(str(settings_file), QSettings.Format.IniFormat)
    orb = orbe_app.create_orb_window(settings=settings)
    orb.move(120, 90)
    orb.save_position()

    second = orbe_app.create_orb_window(settings=settings)
    second.restore_position()
    assert second.frameGeometry().topLeft().x() == 120
    assert second.frameGeometry().topLeft().y() == 90
    orb.close()
    second.close()


def test_restore_clamps_positions_outside_screen(qapp, tmp_path) -> None:
    from PySide6.QtCore import QSettings

    settings_file = tmp_path / "orb-clamp.ini"
    settings = QSettings(str(settings_file), QSettings.Format.IniFormat)
    settings.setValue("pos_x", -5000)
    settings.setValue("pos_y", 90000)

    orb = orbe_app.create_orb_window(settings=settings)
    orb.restore_position()
    point = orb.frameGeometry().topLeft()
    screen_bounds = orb.screen().availableGeometry()
    assert point.x() >= screen_bounds.left()
    assert point.y() >= screen_bounds.top()
    orb.close()


def test_close_event_saves_position(qapp, tmp_path) -> None:
    from PySide6.QtCore import QSettings

    settings_file = tmp_path / "orb-close.ini"
    settings = QSettings(str(settings_file), QSettings.Format.IniFormat)
    orb = orbe_app.create_orb_window(settings=settings)
    orb.move(55, 66)
    orb.close()

    assert int(settings.value("pos_x")) == 55
    assert int(settings.value("pos_y")) == 66
