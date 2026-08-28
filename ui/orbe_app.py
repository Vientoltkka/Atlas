"""Atlas Orbe shell (V4.3-I2/I3/I4): frameless translucent state window.

Pure visual layer over the core: the orb only renders an
``OrbVisualState`` produced by ``use_cases.ui_state_mapper``. No audio,
STT, TTS or orchestrator access happens here. The Qt import is local to
this GUI module so the rest of Atlas never depends on PySide6.

I4 adds per-state fine animation, a tray icon with Detener/Salir and
window position preferences. States, colors/alpha and the existing
signals are unchanged.
"""

from __future__ import annotations

import math
import sys
import time
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

# Per-state fine animation tuning (period seconds; None = static).
_ANIMATION_PERIODS: dict[OrbVisualState, float | None] = {
    OrbVisualState.IDLE: 3.0,        # slow breathing scale
    OrbVisualState.STARTING: 1.2,    # soft fade pulse
    OrbVisualState.LISTENING: 1.6,   # breathing pulse
    OrbVisualState.PROCESSING: 2.0,  # inner arc rotation
    OrbVisualState.SPEAKING: 0.9,    # glow alpha pulse
    OrbVisualState.RECOVERING: 0.6,  # quick blink
    OrbVisualState.DEGRADED: None,   # static dim
    OrbVisualState.STOPPING: 1.5,    # gentle shrink/fade
}

ANIMATION_FPS = 30


def color_for_state(state: OrbVisualState) -> tuple[int, int, int, int]:
    """Deterministic RGBA for one visual state."""
    return _STATE_COLORS[state]


def animation_period(state: OrbVisualState) -> float | None:
    """Animation period in seconds, or None when the state is static."""
    return _ANIMATION_PERIODS[state]


def animation_frame(
    state: OrbVisualState,
    elapsed_seconds: float,
) -> dict[str, float]:
    """Deterministic animation frame for one state at one instant.

    Returns scale (relative to base), alpha_factor (multiplier of the
    base alpha) and rotation_deg (inner marker rotation). Pure function.
    """
    period = _ANIMATION_PERIODS[state]
    if period is None or period <= 0:
        return {"scale": 1.0, "alpha_factor": 1.0, "rotation_deg": 0.0}

    phase = (elapsed_seconds % period) / period
    wave = math.sin(2 * math.pi * phase)

    if state is OrbVisualState.IDLE:
        scale = 1.0 + 0.03 * wave
        alpha_factor = 1.0
        rotation_deg = 0.0
    elif state is OrbVisualState.STARTING:
        scale = 1.0
        alpha_factor = 0.75 + 0.25 * (wave * 0.5 + 0.5)
        rotation_deg = 0.0
    elif state is OrbVisualState.LISTENING:
        scale = 1.0 + 0.05 * wave
        alpha_factor = 1.0
        rotation_deg = 0.0
    elif state is OrbVisualState.PROCESSING:
        scale = 1.0
        alpha_factor = 1.0
        rotation_deg = 360.0 * phase
    elif state is OrbVisualState.SPEAKING:
        scale = 1.02 + 0.03 * (wave * 0.5 + 0.5)
        alpha_factor = 0.85 + 0.15 * (wave * 0.5 + 0.5)
        rotation_deg = 0.0
    elif state is OrbVisualState.RECOVERING:
        scale = 1.0
        alpha_factor = 0.55 + 0.45 * abs(wave)
        rotation_deg = 0.0
    else:  # STOPPING
        scale = 1.0 - 0.06 * phase
        alpha_factor = 1.0 - 0.35 * phase
        rotation_deg = 0.0

    return {
        "scale": round(scale, 4),
        "alpha_factor": round(alpha_factor, 4),
        "rotation_deg": round(rotation_deg, 2),
    }


def create_application(argv: list[str] | None = None):
    """Create (or reuse) the QApplication with HiDPI defaults."""
    from PySide6.QtWidgets import QApplication

    existing = QApplication.instance()
    if existing is not None:
        return existing
    return QApplication(argv if argv is not None else sys.argv)


def create_orb_window(settings=None):
    """Build the orb window; requires a live QApplication.

    ``settings`` is an optional QSettings-like object used to persist
    the window position (keys ``pos_x`` / ``pos_y``).
    """
    from PySide6.QtCore import Qt, Signal, QTimer, Signal
    from PySide6.QtGui import QColor, QIcon, QPainter, QPixmap
    from PySide6.QtWidgets import (
        QMenu,
        QSystemTrayIcon,
        QWidget,
    )

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
            self._settings = settings
            self._animation_started_at = time.monotonic()

            self._timer = QTimer(self)
            self._timer.timeout.connect(self._on_animation_tick)

            self._tray = None
            if QSystemTrayIcon.isSystemTrayAvailable():
                self._build_tray()

            self.restore_position()

        @property
        def state(self) -> OrbVisualState:
            return self._state

        @property
        def tray(self):
            return self._tray

        def apply_state(self, state: OrbVisualState) -> None:
            self._state = OrbVisualState(state)
            self._animation_started_at = time.monotonic()
            self._update_tray_icon()
            self._update_timer()
            self.update()

        def _update_timer(self) -> None:
            if animation_period(self._state) is None:
                self._timer.stop()
            else:
                self._timer.start(int(1000 / ANIMATION_FPS))

        def _on_animation_tick(self) -> None:
            elapsed = time.monotonic() - self._animation_started_at
            self._last_frame = animation_frame(self._state, elapsed)
            self.update()

        def current_frame(self) -> dict[str, float]:
            """Latest deterministic animation frame for rendering."""
            elapsed = time.monotonic() - self._animation_started_at
            return animation_frame(self._state, elapsed)

        # -- tray ------------------------------------------------------

        def _build_tray(self) -> None:
            menu = QMenu()
            menu.addAction("Detener", self.stop_requested.emit)
            menu.addAction("Salir", self.quit_requested.emit)
            tray = QSystemTrayIcon(self._render_state_icon(), self)
            tray.setContextMenu(menu)
            tray.setToolTip("Atlas")
            tray.show()
            self._tray = tray

        def _render_state_icon(self) -> QPixmap:
            red, green, blue, alpha = color_for_state(self._state)
            pixmap = QPixmap(64, 64)
            pixmap.fill(Qt.GlobalColor.transparent)
            painter = QPainter(pixmap)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            painter.setBrush(QColor(red, green, blue, alpha))
            painter.setPen(QColor(0, 0, 0, 0))
            painter.drawEllipse(8, 8, 48, 48)
            painter.end()
            return pixmap

        def _update_tray_icon(self) -> None:
            if self._tray is not None:
                self._tray.setIcon(QIcon(self._render_state_icon()))

        # -- position preferences ---------------------------------------

        def save_position(self) -> None:
            """Persist the current window position when settings exist."""
            if self._settings is None:
                return
            geometry = self.frameGeometry().topLeft()
            self._settings.setValue("pos_x", int(geometry.x()))
            self._settings.setValue("pos_y", int(geometry.y()))

        def restore_position(self) -> None:
            """Restore a previously saved position within screen bounds."""
            if self._settings is None:
                return
            x = self._settings.value("pos_x")
            y = self._settings.value("pos_y")
            if x is None or y is None:
                return
            try:
                target_x, target_y = int(x), int(y)
            except (TypeError, ValueError):
                return
            screen = self.screen()
            bounds = screen.availableGeometry() if screen is not None else None
            if bounds is not None:
                target_x = max(bounds.left(), min(target_x, bounds.right() - ORB_SIZE))
                target_y = max(bounds.top(), min(target_y, bounds.bottom() - ORB_SIZE))
            self.move(target_x, target_y)

        # -- interaction -----------------------------------------------

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

        def closeEvent(self, event) -> None:  # noqa: N802 (Qt API)
            self.save_position()
            super().closeEvent(event)

        # -- painting ---------------------------------------------------

        def paintEvent(self, event) -> None:  # noqa: N802 (Qt API)
            frame = self.current_frame()
            red, green, blue, alpha = color_for_state(self._state)
            alpha = max(0, min(255, int(alpha * frame["alpha_factor"])))
            margin = int((ORB_SIZE // 10) * (2.0 - frame["scale"]))
            diameter = max(1, ORB_SIZE - 2 * margin)
            center = ORB_SIZE // 2

            painter = QPainter(self)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            painter.setBrush(QColor(red, green, blue, alpha))
            painter.setPen(QColor(0, 0, 0, 0))
            painter.drawEllipse(center - diameter // 2, center - diameter // 2, diameter, diameter)

            rotation = frame["rotation_deg"]
            if rotation > 0.0:
                painter.translate(center, center)
                painter.rotate(rotation)
                painter.setBrush(QColor(255, 255, 255, 90))
                painter.drawPie(
                    -diameter // 2, -diameter // 2, diameter, diameter, 0, 60 * 16
                )
            painter.end()

    return OrbWindow()


def create_transcript_panel():
    """Small transcript panel with chat and minimal voice controls."""
    from PySide6.QtCore import Qt, Signal
    from PySide6.QtWidgets import QHBoxLayout, QLabel, QLineEdit, QPushButton, QTextBrowser, QVBoxLayout, QWidget

    class TranscriptPanel(QWidget):
        send_requested = Signal(str)
        close_requested = Signal()
        voice_start_requested = Signal()
        voice_stop_requested = Signal()
        voice_retry_requested = Signal()
        _VOICE_STATUS = {"STARTING": "Iniciando voz", "LISTENING": "Escuchando", "TRANSCRIBING": "STT", "PROCESSING": "Procesando", "SPEAKING": "TTS", "RECOVERING": "Reintentando", "DEGRADED": "Error", "ERROR": "Error", "STOPPING": "Deteniendo", "STOPPED": "Desconectado"}

        def __init__(self) -> None:
            super().__init__()
            self._hide_on_close = False
            self.setWindowTitle("Atlas - transcripcion")
            self.setWindowFlags(Qt.WindowType.Tool | Qt.WindowType.WindowStaysOnTopHint)
            self.resize(360, 220)
            layout = QVBoxLayout(self)
            layout.setContentsMargins(0, 0, 0, 0)
            self._voice_status = QLabel("Desconectado", self)
            layout.addWidget(self._voice_status)
            voice_layout = QHBoxLayout()
            self._voice_start_button = QPushButton("Iniciar voz", self)
            self._voice_stop_button = QPushButton("Detener voz", self)
            self._voice_retry_button = QPushButton("Reintentar", self)
            self._voice_start_button.clicked.connect(self.voice_start_requested.emit)
            self._voice_stop_button.clicked.connect(self.voice_stop_requested.emit)
            self._voice_retry_button.clicked.connect(self.voice_retry_requested.emit)
            voice_layout.addWidget(self._voice_start_button)
            voice_layout.addWidget(self._voice_stop_button)
            voice_layout.addWidget(self._voice_retry_button)
            layout.addLayout(voice_layout)
            self._view = QTextBrowser(self)
            self._view.setReadOnly(True)
            layout.addWidget(self._view)
            input_layout = QHBoxLayout()
            self._input = QLineEdit(self)
            self._input.setPlaceholderText("Escribe un mensaje...")
            self._send_button = QPushButton("Enviar", self)
            self._input.returnPressed.connect(self._submit_input)
            self._send_button.clicked.connect(self._submit_input)
            input_layout.addWidget(self._input)
            input_layout.addWidget(self._send_button)
            layout.addLayout(input_layout)

        def _submit_input(self) -> None:
            text = self._input.text().strip()
            if text:
                self._input.clear()
                self.send_requested.emit(text)

        def set_hide_on_close(self, enabled: bool) -> None:
            """Configure the chat-only close behavior without changing voice UI."""
            self._hide_on_close = bool(enabled)

        def closeEvent(self, event) -> None:  # noqa: N802 (Qt API)
            if self._hide_on_close:
                self.close_requested.emit()
                event.ignore()
                return
            super().closeEvent(event)

        def set_voice_state(self, state: str) -> None:
            self._voice_status.setText(self._VOICE_STATUS.get(str(state), str(state)))

        def set_voice_disconnected(self) -> None:
            self._voice_status.setText("Desconectado")

        def append_message(self, message: str) -> None:
            self._view.append(str(message))

        def append_user(self, message: str) -> None:
            self.append_message(f"Usuario: {message}")

        def append_transcription(self, transcription: str) -> None:
            self.append_message(f"Tú: {transcription}")

        def append_response(self, response: str) -> None:
            self.append_message(f"Atlas: {response}")

        def append_error(self, error: str) -> None:
            self.append_message(f"Error: {error}")

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
