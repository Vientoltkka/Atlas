"""Atlas Orbe shell (V4.3-I2): frameless translucent state window.

Pure visual layer over the core: the orb only renders an
``OrbVisualState`` produced by ``use_cases.ui_state_mapper``. No audio,
STT, TTS or orchestrator access happens here. The Qt import is local to
this GUI module so the rest of Atlas never depends on PySide6.
"""

from __future__ import annotations

import sys
from typing import Sequence

from use_cases.ui_state_mapper import OrbVisualState


ORB_SIZE = 140

_STATE_COLORS: dict[OrbVisualState, tuple[int, int, int, int]] = {
    OrbVisualState.IDLE: (96, 125, 255, 210),
    OrbVisualState.STARTING: (70, 90, 200, 160),
    OrbVisualState.LISTENING: (64, 170, 255, 230),
    OrbVisualState.PROCESSING: (255, 180, 60, 230),
    OrbVisualState.SPEAKING: (70, 220, 140, 235),
    OrbVisualState.RECOVERING: (255, 120, 70, 220),
    OrbVisualState.DEGRADED: (200, 150, 80, 150),
    OrbVisualState.STOPPING: (120, 120, 130, 140),
}

DEMO_STATE_CYCLE: Sequence[OrbVisualState] = (
    OrbVisualState.IDLE,
    OrbVisualState.STARTING,
    OrbVisualState.LISTENING,
    OrbVisualState.PROCESSING,
    OrbVisualState.SPEAKING,
    OrbVisualState.RECOVERING,
    OrbVisualState.DEGRADED,
    OrbVisualState.STOPPING,
)


def color_for_state(state: OrbVisualState) -> tuple[int, int, int, int]:
    """Deterministic RGBA for one visual state."""
    return _STATE_COLORS[state]


def create_application(argv: list[str] | None = None):
    """Create (or reuse) the QApplication with HiDPI defaults."""
    from PySide6.QtWidgets import QApplication

    existing = QApplication.instance()
    if existing is not None:
        return existing
    return QApplication(argv if argv is not None else sys.argv)


def create_orb_window():
    """Build the orb window; requires a live QApplication."""
    from PySide6.QtCore import Qt, Signal
    from PySide6.QtGui import QColor, QPainter
    from PySide6.QtWidgets import QMenu, QWidget

    class OrbWindow(QWidget):
        """Frameless translucent always-on-top circular state indicator."""

        stop_requested = Signal()
        quit_requested = Signal()

        def __init__(self) -> None:
            super().__init__()
            self.setWindowTitle("Atlas")
            self.setFixedSize(ORB_SIZE, ORB_SIZE)
            self.setWindowFlags(
                Qt.WindowType.FramelessWindowHint
                | Qt.WindowType.WindowStaysOnTopHint
                | Qt.WindowType.Tool
            )
            self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
            self._state = OrbVisualState.IDLE
            self._drag_offset = None

        @property
        def state(self) -> OrbVisualState:
            return self._state

        def apply_state(self, state: OrbVisualState) -> None:
            self._state = OrbVisualState(state)
            self.update()

        def mousePressEvent(self, event) -> None:  # noqa: N802 (Qt API)
            if event.button() == Qt.MouseButton.LeftButton:
                self._drag_offset = (
                    event.globalPosition().toPoint() - self.frameGeometry().topLeft()
                )

        def mouseMoveEvent(self, event) -> None:  # noqa: N802 (Qt API)
            if self._drag_offset is not None and event.buttons() & Qt.MouseButton.LeftButton:
                self.move(event.globalPosition().toPoint() - self._drag_offset)

        def mouseReleaseEvent(self, event) -> None:  # noqa: N802 (Qt API)
            self._drag_offset = None

        def contextMenuEvent(self, event) -> None:  # noqa: N802 (Qt API)
            menu = QMenu(self)
            menu.addAction("Detener", self.stop_requested.emit)
            menu.addAction("Salir", self.quit_requested.emit)
            menu.exec(event.globalPos())

        def paintEvent(self, event) -> None:  # noqa: N802 (Qt API)
            painter = QPainter(self)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            red, green, blue, alpha = color_for_state(self._state)
            margin = ORB_SIZE // 10
            diameter = ORB_SIZE - 2 * margin
            painter.setBrush(QColor(red, green, blue, alpha))
            painter.setPen(QColor(0, 0, 0, 0))
            painter.drawEllipse(margin, margin, diameter, diameter)
            painter.end()

    return OrbWindow()


def create_transcript_panel():
    """Minimal always-on-top panel showing session messages (no logic)."""
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QTextBrowser, QVBoxLayout, QWidget

    class TranscriptPanel(QWidget):
        def __init__(self) -> None:
            super().__init__()
            self.setWindowTitle("Atlas - transcripcion")
            self.setWindowFlags(
                Qt.WindowType.Tool
                | Qt.WindowType.WindowStaysOnTopHint
            )
            self.resize(360, 220)
            layout = QVBoxLayout(self)
            layout.setContentsMargins(0, 0, 0, 0)
            self._view = QTextBrowser(self)
            self._view.setReadOnly(True)
            layout.addWidget(self._view)

        def append_message(self, message: str) -> None:
            self._view.append(str(message))

    return TranscriptPanel()


def run_demo(interval_ms: int = 900) -> int:
    """Cycle every visual state for manual verification (I2 acceptance)."""
    from PySide6.QtCore import QTimer

    app = create_application()
    orb = create_orb_window()
    orb.show()
    cycle = list(DEMO_STATE_CYCLE)
    index = {"value": 0}
    orb.apply_state(cycle[0])

    def advance() -> None:
        index["value"] = (index["value"] + 1) % len(cycle)
        orb.apply_state(cycle[index["value"]])

    timer = QTimer()
    timer.timeout.connect(advance)
    timer.start(interval_ms)
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(run_demo())
