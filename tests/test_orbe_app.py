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


def test_fixed_orb_size(orb) -> None:
    assert orbe_app.CORE_RADIUS_FACTOR == pytest.approx(0.275)
    assert orb.width() == orbe_app.ORB_SIZE
    assert 270 <= orbe_app.ORB_SIZE <= 290
    assert orb.height() == orbe_app.ORB_SIZE


def test_orb_repositions_beside_an_overlapping_transcript_panel(qapp, orb) -> None:
    from PySide6.QtCore import Qt

    panel = orbe_app.create_transcript_panel()
    panel.move(100, 100)
    orb.move(120, 100)
    panel.show()

    orb.reposition_beside(panel)

    assert not orb.frameGeometry().intersects(panel.frameGeometry())
    assert orb.windowFlags() & Qt.WindowType.WindowStaysOnTopHint
    panel.close()


# ---------------------------------------------------------------------------
# V4.3-I4: animations, tray, position preferences
# ---------------------------------------------------------------------------


def test_authorization_uses_distinct_amber_animation_without_extra_timer(orb) -> None:
    assert orbe_app.color_for_state(OrbVisualState.AUTHORIZATION)[:3] == (255, 178, 58)
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
    assert orbe_app.size_for_state(OrbVisualState.IDLE) == 280
    for state in (OrbVisualState.LISTENING, OrbVisualState.PROCESSING, OrbVisualState.SPEAKING, OrbVisualState.AUTHORIZATION):
        orb.apply_state(state)
        assert orb.width() == orb.height() == orbe_app.size_for_state(state)
        assert orb.width() >= 430


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
