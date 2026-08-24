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
    assert orb.width() == orbe_app.ORB_SIZE
    assert orb.height() == orbe_app.ORB_SIZE
