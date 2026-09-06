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

import html
import math
import mimetypes
import sys
import time
import uuid
from typing import Sequence

from use_cases.ui_state_mapper import OrbVisualState


ORB_SIZE = 360
CORE_RADIUS_FACTOR = 0.29
_ACTIVE_ORB_SIZES: dict[OrbVisualState, int] = {
    OrbVisualState.LISTENING: 460,
    OrbVisualState.PROCESSING: 480,
    OrbVisualState.SPEAKING: 490,
    OrbVisualState.AUTHORIZATION: 490,
    OrbVisualState.AUTOMATION: 490,
}

_STATE_COLORS: dict[OrbVisualState, tuple[int, int, int, int]] = {
    OrbVisualState.IDLE: (56, 185, 255, 230),
    OrbVisualState.STARTING: (105, 225, 255, 235),
    OrbVisualState.LISTENING: (80, 225, 255, 245),
    OrbVisualState.PROCESSING: (168, 102, 255, 245),
    OrbVisualState.SPEAKING: (72, 238, 148, 250),
    OrbVisualState.AUTHORIZATION: (255, 174, 52, 250),
    OrbVisualState.AUTOMATION: (255, 72, 72, 250),
    OrbVisualState.RECOVERING: (60, 175, 235, 220),
    OrbVisualState.DEGRADED: (78, 100, 128, 145),
    OrbVisualState.STOPPING: (80, 145, 180, 160),
}

DEMO_STATE_CYCLE: Sequence[OrbVisualState] = (
    OrbVisualState.IDLE,
    OrbVisualState.STARTING,
    OrbVisualState.LISTENING,
    OrbVisualState.PROCESSING,
    OrbVisualState.SPEAKING,
    OrbVisualState.AUTHORIZATION,
    OrbVisualState.AUTOMATION,
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
    OrbVisualState.SPEAKING: 0.9,    # quick luminous halo pulse
    OrbVisualState.AUTHORIZATION: 2.4,  # slow human-approval pulse
    OrbVisualState.AUTOMATION: 1.35,  # active supervised execution
    OrbVisualState.RECOVERING: 0.6,  # quick blink
    OrbVisualState.DEGRADED: None,   # static dim
    OrbVisualState.STOPPING: 1.5,    # gentle shrink/fade
}

ANIMATION_FPS = 24

_VISUAL_PROFILES: dict[OrbVisualState, dict[str, float]] = {
    # One procedural renderer; these controls give every primary state its own motion language.
    OrbVisualState.IDLE: {
        "ring_activity": 0.16, "ring_speed": 0.34, "ring_angle": 0.86, "ring_amplitude": 0.88,
        "pulse_strength": 0.34, "halo_intensity": 0.72, "halo_strength": 0.72,
        "core_intensity": 0.85, "core_pulse": 0.54, "particle_intensity": 0.48,
        "segment_activity": 0.45, "base_intensity": 0.60,
        "wave_activity": 0.0, "ping_intensity": 0.0, "node_energy": 0.35, "spark_intensity": 0.0,
    },
    OrbVisualState.PROCESSING: {
        "ring_activity": 1.00, "ring_speed": 1.00, "ring_angle": 1.20, "ring_amplitude": 1.08,
        "pulse_strength": 0.66, "halo_intensity": 1.12, "halo_strength": 1.12,
        "core_intensity": 1.18, "core_pulse": 0.86, "particle_intensity": 0.82,
        "segment_activity": 1.00, "base_intensity": 0.88,
        "wave_activity": 0.85, "ping_intensity": 0.15, "node_energy": 0.85, "spark_intensity": 0.35,
    },
    OrbVisualState.SPEAKING: {
        "ring_activity": 0.30, "ring_speed": 0.64, "ring_angle": 0.98, "ring_amplitude": 1.00,
        "pulse_strength": 1.00, "halo_intensity": 1.28, "halo_strength": 1.28,
        "core_intensity": 1.12, "core_pulse": 1.18, "particle_intensity": 0.86,
        "segment_activity": 0.72, "base_intensity": 1.04,
        "wave_activity": 1.00, "ping_intensity": 0.0, "node_energy": 0.55, "spark_intensity": 0.0,
    },
    OrbVisualState.AUTOMATION: {
        "ring_activity": 1.34, "ring_speed": 1.42, "ring_angle": 1.34, "ring_amplitude": 1.16,
        "pulse_strength": 0.92, "halo_intensity": 1.22, "halo_strength": 1.22,
        "core_intensity": 1.22, "core_pulse": 1.04, "particle_intensity": 0.96,
        "segment_activity": 1.25, "base_intensity": 1.12,
        "wave_activity": 0.35, "ping_intensity": 0.30, "node_energy": 1.00, "spark_intensity": 1.00,
    },
    OrbVisualState.AUTHORIZATION: {
        "ring_activity": 0.25, "ring_speed": 0.42, "ring_angle": 0.78, "ring_amplitude": 0.82,
        "pulse_strength": 0.52, "halo_intensity": 0.92, "halo_strength": 0.92,
        "core_intensity": 1.04, "core_pulse": 0.70, "particle_intensity": 0.58,
        "segment_activity": 0.56, "base_intensity": 0.78,
        "wave_activity": 0.10, "ping_intensity": 1.00, "node_energy": 0.45, "spark_intensity": 0.0,
    },
}


def color_for_state(state: OrbVisualState) -> tuple[int, int, int, int]:
    """Deterministic RGBA for one visual state."""
    return _STATE_COLORS[state]


def animation_period(state: OrbVisualState) -> float | None:
    """Animation period in seconds, or None when the state is static."""
    return _ANIMATION_PERIODS[state]


def visual_profile(state: OrbVisualState) -> dict[str, float]:
    """Return lightweight per-state render controls for the shared orb."""
    return dict(_VISUAL_PROFILES.get(OrbVisualState(state), _VISUAL_PROFILES[OrbVisualState.IDLE]))


def size_for_state(state: OrbVisualState) -> int:
    """Return the compact idle size or the deliberately larger active size."""
    return _ACTIVE_ORB_SIZES.get(OrbVisualState(state), ORB_SIZE)


def animation_frame(
    state: OrbVisualState,
    elapsed_seconds: float,
) -> dict[str, float]:
    """Return the lightweight deterministic frame used by the core renderer."""
    period = _ANIMATION_PERIODS[state]
    if period is None or period <= 0:
        return {"scale": 1.0, "alpha_factor": 1.0, "rotation_deg": 0.0}

    phase = (elapsed_seconds % period) / period
    wave = math.sin(2 * math.pi * phase)
    profile = visual_profile(state)

    if state is OrbVisualState.IDLE:
        scale = 1.0 + 0.024 * profile["pulse_strength"] * wave
        alpha_factor = 1.0
        rotation_deg = 18.0 * phase
    elif state is OrbVisualState.STARTING:
        scale = 1.0
        alpha_factor = 0.75 + 0.25 * (wave * 0.5 + 0.5)
        rotation_deg = 30.0 * phase
    elif state is OrbVisualState.LISTENING:
        scale = 1.0  # voice waves move outside; the nucleus remains stable
        alpha_factor = 1.0
        rotation_deg = 360.0 * phase
    elif state is OrbVisualState.PROCESSING:
        scale = 1.0 + 0.012 * profile["pulse_strength"] * wave
        alpha_factor = 0.92 + 0.08 * (wave * 0.5 + 0.5)
        rotation_deg = 360.0 * phase
    elif state is OrbVisualState.SPEAKING:
        scale = 1.018 + 0.048 * profile["pulse_strength"] * (wave * 0.5 + 0.5)
        alpha_factor = 0.85 + 0.15 * (wave * 0.5 + 0.5)
        rotation_deg = 360.0 * phase
    elif state is OrbVisualState.AUTHORIZATION:
        scale = 1.0 + 0.022 * profile["pulse_strength"] * wave
        alpha_factor = 0.78 + 0.20 * (wave * 0.5 + 0.5)
        rotation_deg = 360.0 * phase
    elif state is OrbVisualState.AUTOMATION:
        scale = 1.0 + 0.035 * profile["pulse_strength"] * wave
        alpha_factor = 0.88 + 0.12 * (wave * 0.5 + 0.5)
        rotation_deg = 360.0 * phase
    elif state is OrbVisualState.RECOVERING:
        scale = 1.0
        alpha_factor = 0.55 + 0.45 * abs(wave)
        rotation_deg = 15.0 * phase
    else:  # STOPPING
        scale = 1.0 - 0.06 * phase
        alpha_factor = 1.0 - 0.35 * phase
        rotation_deg = 12.0 * phase

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
    from PySide6.QtCore import QPointF, QRectF, QSize, Qt, Signal, QTimer
    from PySide6.QtGui import (
        QConicalGradient,
        QColor,
        QIcon,
        QLinearGradient,
        QPainter,
        QPainterPath,
        QPen,
        QPixmap,
        QRadialGradient,
    )
    from PySide6.QtWidgets import (
        QFrame,
        QHBoxLayout,
        QLabel,
        QMenu,
        QPushButton,
        QSystemTrayIcon,
        QVBoxLayout,
        QWidget,
    )

    class OrbContextMenu(QWidget):
        """Small translucent popup for actions already owned by the controller."""

        chat_requested = Signal()
        voice_requested = Signal()
        quit_requested = Signal()

        def __init__(self, parent=None) -> None:
            super().__init__(parent)
            self.setWindowFlags(Qt.WindowType.Popup | Qt.WindowType.FramelessWindowHint)
            self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
            self.setFixedWidth(224)
            self._voice_active = False

            layout = QVBoxLayout(self)
            layout.setContentsMargins(12, 11, 12, 11)
            layout.setSpacing(4)
            self._title = QLabel("MENU ATLAS", self)
            self._title.setStyleSheet("color: #bdeeff; font-size: 10px; font-weight: 700; letter-spacing: 1.4px;")
            layout.addWidget(self._title)
            self._add_separator(layout)
            self._chat_button = self._add_action(layout, "Abrir Chat", self.chat_requested, "chat")
            self._voice_button = self._add_action(layout, "Modo Voz", self.voice_requested, "voice", voice_status=True)
            self._add_separator(layout)
            self._quit_button = self._add_action(layout, "Salir", self.quit_requested, "quit", danger=True)

        def _add_separator(self, layout) -> None:
            separator = QFrame(self)
            separator.setFrameShape(QFrame.Shape.HLine)
            separator.setStyleSheet("color: rgba(85, 185, 255, 105);")
            layout.addWidget(separator)

        def _add_action(self, layout, text: str, signal, icon_name: str, *, danger: bool = False, voice_status: bool = False):
            button = QPushButton(text, self)
            color = "#ffb6b6" if danger else "#e7f8ff"
            hover = "rgba(255, 78, 78, 50)" if danger else "rgba(52, 173, 255, 52)"
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            button.setIcon(QIcon(self._action_icon(icon_name, "#ff9c9c" if danger else "#84d8ff")))
            button.setIconSize(QSize(18, 18))
            button.setStyleSheet(
                "QPushButton {"
                f"color: {color}; background: transparent; border: 1px solid transparent;"
                "border-radius: 7px; padding: 8px 9px; text-align: left; font-size: 12px;"
                "}"
                f"QPushButton:hover {{ background: {hover}; border-color: rgba(96, 202, 255, 130); }}"
            )
            button.clicked.connect(signal.emit)
            button.clicked.connect(self.hide)
            if voice_status:
                row = QFrame(self)
                row_layout = QHBoxLayout(row)
                row_layout.setContentsMargins(0, 0, 0, 0)
                row_layout.setSpacing(6)
                row_layout.addWidget(button, 1)
                self._voice_indicator = QLabel(row)
                self._voice_indicator.setFixedSize(8, 8)
                self._voice_indicator.setToolTip("Voz inactiva")
                row_layout.addWidget(self._voice_indicator)
                layout.addWidget(row)
                self._set_voice_indicator(False)
            else:
                layout.addWidget(button)
            return button

        def _action_icon(self, icon_name: str, color: str) -> QPixmap:
            pixmap = QPixmap(18, 18)
            pixmap.fill(Qt.GlobalColor.transparent)
            painter = QPainter(pixmap)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            pen = QPen(QColor(color), 1.7)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            if icon_name == "chat":
                painter.drawRoundedRect(2, 3, 14, 10, 3, 3)
                painter.drawLine(6, 13, 5, 16)
                painter.drawLine(5, 16, 9, 13)
            elif icon_name == "voice":
                painter.drawRoundedRect(6, 2, 6, 10, 3, 3)
                painter.drawArc(3, 7, 12, 8, 0, -180 * 16)
                painter.drawLine(9, 15, 9, 17)
                painter.drawLine(6, 17, 12, 17)
            else:
                painter.drawArc(3, 3, 12, 12, 45 * 16, 270 * 16)
                painter.drawLine(9, 1, 9, 9)
            painter.end()
            return pixmap

        def _set_voice_indicator(self, active: bool) -> None:
            color = "#45ee94" if active else "#758496"
            self._voice_indicator.setStyleSheet(f"background: {color}; border-radius: 4px;")
            self._voice_indicator.setToolTip("Voz activa" if active else "Voz inactiva")

        def set_voice_active(self, active: bool) -> None:
            self._voice_active = active
            self._voice_button.setText("Detener voz" if active else "Modo Voz")
            self._set_voice_indicator(active)

        def show_beside(self, orb) -> None:
            self.adjustSize()
            screen = orb.screen()
            if screen is None:
                return
            bounds = screen.availableGeometry()
            gap = 12
            right_x = orb.frameGeometry().right() + gap + 1
            left_x = orb.frameGeometry().left() - gap - self.width()
            x = right_x if right_x + self.width() <= bounds.right() + 1 else left_x
            x = max(bounds.left(), min(x, bounds.right() - self.width() + 1))
            y = orb.frameGeometry().center().y() - self.height() // 2
            y = max(bounds.top(), min(y, bounds.bottom() - self.height() + 1))
            self.move(x, y)
            self.show()
            self.raise_()

        def paintEvent(self, event) -> None:  # noqa: N802 (Qt API)
            painter = QPainter(self)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            path = QPainterPath()
            path.addRoundedRect(self.rect().adjusted(2, 2, -2, -2), 10, 10)
            painter.setPen(QPen(QColor(87, 204, 255, 205), 1.0))
            painter.setBrush(QColor(3, 15, 36, 242))
            painter.drawPath(path)
            painter.setPen(QPen(QColor(80, 195, 255, 46), 5.0))
            painter.drawPath(path)
            painter.end()

    class OrbWindow(QWidget):
        """Frameless translucent always-on-top circular state indicator."""

        stop_requested = Signal()
        quit_requested = Signal()
        chat_requested = Signal()
        voice_requested = Signal()

        def __init__(self) -> None:
            super().__init__()
            self.setWindowTitle("Atlas")
            initial_size = self._bounded_size(ORB_SIZE)
            self.setFixedSize(initial_size, initial_size)
            self.setWindowFlags(
                Qt.WindowType.FramelessWindowHint
                | Qt.WindowType.WindowStaysOnTopHint
                | Qt.WindowType.Tool
            )
            self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
            self._state = OrbVisualState.IDLE
            self._drag_offset = None
            self._press_global = None
            self._dragging = False
            self._voice_active = False
            self._settings = settings
            self._animation_started_at = time.monotonic()
            self._emblem_path = self._build_atlas_emblem()
            self._context_menu = OrbContextMenu(self)
            self._context_menu.chat_requested.connect(self.chat_requested.emit)
            self._context_menu.voice_requested.connect(self.voice_requested.emit)
            self._context_menu.quit_requested.connect(self.quit_requested.emit)

            self._timer = QTimer(self)
            self._timer.timeout.connect(self._on_animation_tick)

            self._tray = None
            if QSystemTrayIcon.isSystemTrayAvailable():
                self._build_tray()

            self.restore_position()
            # Start the animation loop for the initial IDLE state.
            self._update_timer()

        @property
        def state(self) -> OrbVisualState:
            return self._state

        @property
        def tray(self):
            return self._tray

        def apply_state(self, state: OrbVisualState) -> None:
            self._state = OrbVisualState(state)
            self._resize_for_state()
            self._animation_started_at = time.monotonic()
            self._update_tray_icon()
            self._update_timer()
            self.update()

        @property
        def context_menu(self):
            return self._context_menu

        def set_voice_active(self, active: bool) -> None:
            self._voice_active = active
            self._context_menu.set_voice_active(active)

        def toggle_context_menu(self) -> None:
            if self._context_menu.isVisible():
                self._context_menu.hide()
            else:
                self._context_menu.set_voice_active(self._voice_active)
                self._context_menu.show_beside(self)

        def _resize_for_state(self) -> None:
            """Resize around the current centre while keeping it on screen."""
            target_size = self._bounded_size(size_for_state(self._state))
            if self.width() == target_size:
                return
            centre = self.frameGeometry().center()
            self.setFixedSize(target_size, target_size)
            self._emblem_path = self._build_atlas_emblem()
            self.move(
                centre.x() - (target_size - 1) // 2,
                centre.y() - (target_size - 1) // 2,
            )
            self._clamp_to_available_geometry()

        def _bounded_size(self, requested_size: int) -> int:
            screen = self.screen()
            if screen is None:
                return requested_size
            bounds = screen.availableGeometry()
            return max(1, min(requested_size, bounds.width(), bounds.height()))

        def _clamp_to_available_geometry(self) -> None:
            screen = self.screen()
            if screen is None:
                return
            bounds = screen.availableGeometry()
            x = max(bounds.left(), min(self.x(), bounds.right() - self.width() + 1))
            y = max(bounds.top(), min(self.y(), bounds.bottom() - self.height() + 1))
            self.move(x, y)

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
                self._center_on_available_geometry()
                return
            x = self._settings.value("pos_x")
            y = self._settings.value("pos_y")
            if x is None or y is None:
                self._center_on_available_geometry()
                return
            try:
                target_x, target_y = int(x), int(y)
            except (TypeError, ValueError):
                return
            self.move(target_x, target_y)
            self._clamp_to_available_geometry()

        def _center_on_available_geometry(self) -> None:
            screen = self.screen()
            if screen is None:
                return
            bounds = screen.availableGeometry()
            self.move(
                bounds.center().x() - (self.width() - 1) // 2,
                bounds.center().y() - (self.height() - 1) // 2,
            )
            self._clamp_to_available_geometry()

        def reposition_beside(self, panel) -> None:
            """Move beside a visible transcript panel only when they overlap."""
            panel_geometry = panel.frameGeometry()
            if not self.frameGeometry().intersects(panel_geometry):
                return
            screen = panel.screen() or self.screen()
            if screen is None:
                return
            bounds = screen.availableGeometry()
            size = self.width()
            target_y = max(bounds.top(), min(panel_geometry.center().y() - size // 2, bounds.bottom() - size + 1))
            gap = 16
            for target_x in (panel_geometry.right() + gap + 1, panel_geometry.left() - gap - size):
                if bounds.left() <= target_x and target_x + size <= bounds.right() + 1:
                    self.move(target_x, target_y)
                    return
            target_x = max(bounds.left(), min(panel_geometry.center().x() - size // 2, bounds.right() - size + 1))
            for target_y in (panel_geometry.top() - gap - size, panel_geometry.bottom() + gap + 1):
                if bounds.top() <= target_y and target_y + size <= bounds.bottom() + 1:
                    self.move(target_x, target_y)
                    return

        def _avoid_visible_transcript_overlap(self) -> None:
            from PySide6.QtWidgets import QApplication
            for widget in QApplication.topLevelWidgets():
                if widget.objectName() == "atlasTranscriptPanel" and widget.isVisible():
                    self.reposition_beside(widget)
                    return

        # -- interaction -----------------------------------------------

        def mousePressEvent(self, event) -> None:  # noqa: N802 (Qt API)
            if event.button() == Qt.MouseButton.LeftButton:
                self._drag_offset = (
                    event.globalPosition().toPoint() - self.frameGeometry().topLeft()
                )
                self._press_global = event.globalPosition().toPoint()
                self._dragging = False

        def mouseMoveEvent(self, event) -> None:  # noqa: N802 (Qt API)
            if self._drag_offset is not None and event.buttons() & Qt.MouseButton.LeftButton:
                if not self._dragging:
                    distance = (event.globalPosition().toPoint() - self._press_global).manhattanLength()
                    if distance < 10:
                        return
                    self._dragging = True
                    self._context_menu.hide()
                self.move(event.globalPosition().toPoint() - self._drag_offset)

        def mouseReleaseEvent(self, event) -> None:  # noqa: N802 (Qt API)
            if event.button() == Qt.MouseButton.LeftButton and not self._dragging:
                self.toggle_context_menu()
            self._drag_offset = None
            self._press_global = None
            self._dragging = False

        def contextMenuEvent(self, event) -> None:  # noqa: N802 (Qt API)
            self.toggle_context_menu()
            event.accept()

        def closeEvent(self, event) -> None:  # noqa: N802 (Qt API)
            self.save_position()
            super().closeEvent(event)

        # -- painting ---------------------------------------------------

        def _build_atlas_emblem(self) -> QPainterPath:
            """Create the compact Atlas chevron and its detached lower triangle."""
            size = self.width()
            path = QPainterPath()
            # Two diagonal arms form an open chevron; no horizontal A crossbar is used.
            path.moveTo(size * 0.500, size * 0.427)
            path.lineTo(size * 0.414, size * 0.558)
            path.lineTo(size * 0.449, size * 0.569)
            path.lineTo(size * 0.500, size * 0.480)
            path.closeSubpath()
            path.moveTo(size * 0.500, size * 0.427)
            path.lineTo(size * 0.586, size * 0.558)
            path.lineTo(size * 0.551, size * 0.569)
            path.lineTo(size * 0.500, size * 0.480)
            path.closeSubpath()
            # The isolated lower triangle keeps generous negative space inside the core.
            path.moveTo(size * 0.500, size * 0.578)
            path.lineTo(size * 0.470, size * 0.624)
            path.lineTo(size * 0.530, size * 0.624)
            path.closeSubpath()
            return path

        # -- painting ---------------------------------------------------

        _ORBIT_SPECS = (
            # (tilt_deg, squash, radius_factor, speed, front segments (start_deg, span_deg))
            (-26.0, 0.44, 0.450, 1.00, ((196.0, 74.0), (282.0, 50.0), (340.0, 16.0))),
            (36.0, 0.26, 0.478, -0.58, ((188.0, 88.0), (292.0, 62.0))),
            (-58.0, 0.56, 0.452, 0.76, ((182.0, 60.0), (254.0, 30.0), (296.0, 58.0))),
            (10.0, 0.20, 0.490, -0.40, ((200.0, 96.0), (308.0, 44.0))),
        )
        _ORBIT_BACK_SEGMENTS = ((22.0, 66.0), (112.0, 58.0))
        _ORBIT_MODULES = ((226.0, 318.0), (240.0,), (300.0, 208.0), (264.0,))
        _SHELL_PLATES = ((12.0, 46.0), (74.0, 30.0), (118.0, 52.0), (188.0, 24.0), (226.0, 58.0), (300.0, 34.0))
        _BAND_SEGMENTS = ((18.0, 58.0), (94.0, 24.0), (136.0, 46.0), (200.0, 16.0), (238.0, 64.0), (322.0, 26.0))
        _BAND_NODES = (80.0, 122.0, 190.0, 268.0, 348.0)
        _STATE_ORBIT_SPEEDS = {
            OrbVisualState.AUTOMATION: (1.34, 2.10, -1.15, 1.72),
            OrbVisualState.PROCESSING: (1.00, 1.58, -0.82, 1.24),
            OrbVisualState.SPEAKING: (0.30, 0.20, -0.14, 0.24),
            OrbVisualState.AUTHORIZATION: (0.25, -0.15, 0.10, 0.18),
            OrbVisualState.LISTENING: (0.18, -0.12, 0.08, 0.14),
        }

        @staticmethod
        def _arc_rect(radius: float) -> QRectF:
            return QRectF(-radius, -radius, radius * 2.0, radius * 2.0)

        @staticmethod
        def _orbit_point(radius: float, squash: float, tilt_deg: float, angle_deg: float) -> tuple[float, float]:
            """Screen-space point on a tilted squashed orbital ellipse."""
            rad = math.radians(angle_deg)
            x = radius * math.cos(rad)
            y = -radius * math.sin(rad) * squash
            tilt = math.radians(tilt_deg)
            return (x * math.cos(tilt) - y * math.sin(tilt), x * math.sin(tilt) + y * math.cos(tilt))

        def paintEvent(self, event) -> None:  # noqa: N802 (Qt API)
            """Render the layered volumetric Atlas holo-device with QPainter."""
            elapsed = time.monotonic() - self._animation_started_at
            frame = animation_frame(self._state, elapsed)
            red, green, blue, base_alpha = color_for_state(self._state)
            alpha = max(0, min(255, int(base_alpha * frame["alpha_factor"])))
            size = self.width()
            center = size // 2
            profile = visual_profile(self._state)
            core_scale = 1.0 + (frame["scale"] - 1.0) * profile["core_pulse"]
            core_radius = max(20.0, size * CORE_RADIUS_FACTOR * core_scale)
            palette = {
                OrbVisualState.PROCESSING: ((187, 116, 255), (75, 38, 132), (215, 175, 255), (165, 94, 255), (245, 232, 255), (16, 5, 44)),
                OrbVisualState.SPEAKING: ((80, 235, 157), (20, 111, 69), (152, 255, 205), (57, 220, 128), (222, 255, 237), (4, 42, 27)),
                OrbVisualState.AUTHORIZATION: ((255, 181, 60), (166, 95, 15), (255, 224, 142), (255, 194, 70), (255, 245, 204), (46, 22, 2)),
                OrbVisualState.AUTOMATION: ((255, 78, 78), (145, 26, 35), (255, 163, 163), (242, 62, 69), (255, 232, 232), (44, 5, 10)),
            }
            halo_rgb, ring_dim_rgb, ring_light_rgb, ring_bright_rgb, ring_peak_rgb, depth_rgb = palette.get(
                self._state, ((70, 205, 255), (42, 142, 255), (120, 225, 255), (53, 179, 255), (216, 252, 255), (2, 12, 40))
            )
            c = {
                "state": self._state,
                "size": size,
                "center": center,
                "r": core_radius,
                "alpha": alpha,
                "base_alpha": int(alpha * profile["base_intensity"]),
                "frame": frame,
                "profile": profile,
                "period": _ANIMATION_PERIODS[self._state],
                "elapsed": elapsed,
                "rgb": (red, green, blue),
                "halo": halo_rgb,
                "dim": ring_dim_rgb,
                "light": ring_light_rgb,
                "bright": ring_bright_rgb,
                "peak": ring_peak_rgb,
                "depth": depth_rgb,
            }

            painter = QPainter(self)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            # Order: ambient -> orbit backs -> beam/base -> device -> nucleus -> orbit fronts -> activity.
            self._draw_ambient(painter, c)
            self._draw_orbits(painter, c, front=False)
            self._draw_beam_and_base(painter, c)
            self._draw_device(painter, c)
            self._draw_core_particles(painter, c)
            self._draw_emblem(painter, c)
            self._draw_orbits(painter, c, front=True)
            self._draw_state_activity(painter, c)
            self._draw_starfield(painter, c)
            painter.end()

        def _draw_ambient(self, painter, c) -> None:
            """Dark atmospheric halo that fades to fully transparent over the desktop."""
            center, alpha = c["center"], c["alpha"]
            ambient_radius = c["size"] * 0.55
            ambient_rgb = tuple(min(255, int(d + h * 0.22)) for d, h in zip(c["depth"], c["halo"]))
            glow = c["profile"]["halo_strength"]
            gradient = QRadialGradient(center, center, ambient_radius)
            gradient.setColorAt(0.0, QColor(*ambient_rgb, int(alpha * 0.34 * glow)))
            gradient.setColorAt(0.40, QColor(*ambient_rgb, int(alpha * 0.20 * glow)))
            gradient.setColorAt(0.75, QColor(*ambient_rgb, int(alpha * 0.08 * glow)))
            gradient.setColorAt(1.0, QColor(*ambient_rgb, 0))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(gradient)
            painter.drawEllipse(self._arc_rect(ambient_radius).translated(QPointF(center, center)))

        def _draw_orbits(self, painter, c, *, front: bool) -> None:
            """3-4 thick segmented 3D orbits: dim halves behind, bright halves in front."""
            center, size = c["center"], c["size"]
            alpha, profile, frame = c["alpha"], c["profile"], c["frame"]
            halo, dim, light, bright, peak = c["halo"], c["dim"], c["light"], c["bright"], c["peak"]
            speeds = self._STATE_ORBIT_SPEEDS.get(c["state"], (0.16, -0.10, 0.07, 0.12))
            gain = profile["ring_activity"] * profile["ring_speed"]
            widths = (4.6, 3.6, 2.8)
            for index, (tilt, squash, factor, speed, segments) in enumerate(self._ORBIT_SPECS):
                radius = min(
                    size * factor * (0.97 + 0.05 * profile["ring_amplitude"]),
                    size * 0.495,
                )
                rotation = frame["rotation_deg"] * speed * gain
                painter.save()
                painter.translate(center, center)
                painter.rotate(tilt + rotation)
                painter.scale(1.0, squash)
                orbit_rect = self._arc_rect(radius)
                if front:
                    for seg_index, (start, span) in enumerate(segments):
                        width = widths[seg_index % len(widths)] * (0.72 + 0.38 * profile["segment_activity"])
                        # Dark underside first: the tube gains physical thickness.
                        painter.setPen(QPen(QColor(*c["depth"], int(alpha * 0.55)), width + 2.0))
                        painter.drawArc(orbit_rect, int(start * 16), int(span * 16))
                        painter.setPen(QPen(QColor(*halo, int(alpha * 0.12)), width * 2.4))
                        painter.drawArc(orbit_rect, int(start * 16), int(span * 16))
                        painter.setPen(QPen(QColor(*light, int(alpha * 0.82)), width))
                        painter.drawArc(orbit_rect, int(start * 16), int(span * 16))
                        painter.setPen(QPen(QColor(*bright, int(alpha)), max(1.2, width * 0.50)))
                        painter.drawArc(orbit_rect, int((start + 2) * 16), int((span - 4) * 16))
                        painter.setPen(QPen(QColor(255, 255, 255, int(alpha * 0.60)), max(1.0, width * 0.35)))
                        painter.drawArc(orbit_rect, int(start * 16), int(min(9.0, span * 0.35) * 16))
                else:
                    for start, span in self._ORBIT_BACK_SEGMENTS:
                        painter.setPen(QPen(QColor(*dim, int(alpha * 0.34)), 1.6))
                        painter.drawArc(orbit_rect, int(start * 16), int(span * 16))
                painter.restore()
                if not front:
                    continue
                # Small technology modules ride the front arc of each orbit.
                for module_angle in self._ORBIT_MODULES[index]:
                    mx, my = self._orbit_point(radius, squash, tilt + rotation, module_angle)
                    if my < -c["r"] * 0.15 or math.hypot(mx, my) < c["r"] * 1.18:
                        continue  # the module would read as being behind the device
                    px, py = center + mx, center + my
                    painter.setPen(Qt.PenStyle.NoPen)
                    painter.setBrush(QColor(*c["depth"], int(alpha * 0.88)))
                    painter.drawEllipse(QPointF(px, py), 6.4, 2.9)
                    painter.setBrush(QColor(*light, int(alpha * 0.92)))
                    painter.drawEllipse(QPointF(px, py - 0.9), 5.2, 2.2)
                    painter.setBrush(QColor(255, 255, 255, int(alpha * 0.75)))
                    painter.drawEllipse(QPointF(px - 1.6, py - 1.5), 1.5, 0.9)
                # Luminous node closing the first front segment of each orbit.
                node_angle = segments[0][0] + segments[0][1]
                nx, ny = self._orbit_point(radius, squash, tilt + rotation, node_angle)
                node_flash = 1.0
                if profile["node_energy"] >= 0.6:
                    node_flash = 0.55 + 0.45 * (math.sin(c["elapsed"] * 11.0 + index * 1.9) * 0.5 + 0.5)
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QColor(*peak, int(alpha * 0.30 * node_flash)))
                painter.drawEllipse(QPointF(center + nx, center + ny), 5.5, 5.5)
                painter.setBrush(QColor(*bright, int(alpha * 0.92 * node_flash)))
                painter.drawEllipse(QPointF(center + nx, center + ny), 2.4, 2.4)

        def _draw_beam_and_base(self, painter, c) -> None:
            """Projection beam, segmented holographic platform and reflection pool."""
            size, center = c["size"], c["center"]
            alpha, projection, peak = c["base_alpha"], c["bright"], c["peak"]
            base_y = size * 0.815
            beam_top = center + c["r"] * 0.72
            half_top = max(6.0, c["r"] * 0.30)
            half_bottom = max(14.0, size * 0.205)
            beam = QPainterPath()
            beam.moveTo(center - half_top, beam_top)
            beam.lineTo(center + half_top, beam_top)
            beam.lineTo(center + half_bottom, base_y)
            beam.lineTo(center - half_bottom, base_y)
            beam.closeSubpath()
            projection_gradient = QLinearGradient(center, beam_top, center, base_y)
            projection_gradient.setColorAt(0.0, QColor(*projection, int(alpha * 0.08)))
            projection_gradient.setColorAt(0.45, QColor(*projection, int(alpha * 0.20)))
            projection_gradient.setColorAt(0.88, QColor(*projection, int(alpha * 0.32)))
            projection_gradient.setColorAt(1.0, QColor(*projection, int(alpha * 0.26)))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(projection_gradient)
            painter.drawPath(beam)
            painter.setPen(QPen(QColor(*peak, int(alpha * 0.30)), 2.2))
            painter.drawLine(QPointF(center, beam_top), QPointF(center, base_y))

            # Reflection pool below the platform.
            painter.save()
            painter.translate(center, base_y + size * 0.02)
            painter.scale(1.0, 0.20)
            pool = QRadialGradient(0, 0, size * 0.30)
            pool.setColorAt(0.0, QColor(*projection, int(alpha * 0.26)))
            pool.setColorAt(0.55, QColor(*projection, int(alpha * 0.10)))
            pool.setColorAt(1.0, QColor(*projection, 0))
            painter.setBrush(pool)
            painter.drawEllipse(self._arc_rect(size * 0.30))
            painter.restore()

            # Layered platform: segmented outer ring, mid rings, luminous centre.
            painter.save()
            painter.translate(center, base_y)
            painter.scale(1.0, 0.30)
            outer = size * 0.30
            dash_pen = QPen(QColor(*projection, int(alpha * 0.62)), 3.2, Qt.PenStyle.CustomDashLine)
            dash_pen.setDashPattern((0.40, 0.10))
            painter.setPen(dash_pen)
            painter.drawEllipse(self._arc_rect(outer))
            mid = size * 0.225
            painter.setPen(QPen(QColor(*projection, int(alpha * 0.52)), 2.0))
            painter.drawEllipse(self._arc_rect(mid))
            inner = size * 0.155
            painter.setPen(QPen(QColor(*projection, int(alpha * 0.40)), 1.2))
            painter.drawEllipse(self._arc_rect(inner))
            hot = size * 0.09
            hot_gradient = QRadialGradient(0, 0, hot)
            hot_gradient.setColorAt(0.0, QColor(240, 252, 255, min(255, int(alpha * 0.95))))
            hot_gradient.setColorAt(0.45, QColor(*peak, int(alpha * 0.55)))
            hot_gradient.setColorAt(1.0, QColor(*projection, 0))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(hot_gradient)
            painter.drawEllipse(self._arc_rect(hot))
            painter.restore()

        def _draw_device(self, painter, c) -> None:
            """The device body: shell plates, tech rings, segment band, glass sphere, inner glow."""
            center, size = c["center"], c["size"]
            r = c["r"]
            alpha = c["alpha"]
            profile = c["profile"]
            halo, dim, light, bright, peak, depth = (c[key] for key in ("halo", "dim", "light", "bright", "peak", "depth"))
            painter.save()
            painter.translate(center, center)

            # A. Outer shell: dark faceted band with metallic plates and edge glints.
            band_outer = r * 1.15
            band_inner = r * 1.005
            band = QPainterPath()
            band.addEllipse(self._arc_rect(band_outer))
            hole = QPainterPath()
            hole.addEllipse(self._arc_rect(band_inner))
            shell_gradient = QRadialGradient(0, -r * 0.30, band_outer)
            shell_gradient.setColorAt(0.0, QColor(min(255, depth[0] + 20), min(255, depth[1] + 20), min(255, depth[2] + 26), int(alpha * 0.94)))
            shell_gradient.setColorAt(0.55, QColor(*depth, int(alpha * 0.97)))
            shell_gradient.setColorAt(1.0, QColor(min(255, depth[0] + 12), min(255, depth[1] + 12), min(255, depth[2] + 16), int(alpha * 0.92)))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(shell_gradient)
            painter.drawPath(band.subtracted(hole))
            plate_r = r * 1.078
            top_r = r * 1.132
            for start, span in self._SHELL_PLATES:
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.setPen(QPen(QColor(*halo, int(alpha * 0.30)), r * 0.105))
                painter.drawArc(self._arc_rect(plate_r), int(start * 16), int(span * 16))
                painter.setPen(QPen(QColor(*light, int(alpha * 0.26)), 1.3))
                painter.drawArc(self._arc_rect(top_r), int((start + 1) * 16), int((span - 2) * 16))
                painter.setPen(QPen(QColor(255, 255, 255, int(alpha * 0.42)), 1.0))
                painter.drawArc(self._arc_rect(top_r), int((start + span - 5) * 16), int(5 * 16))

            # B. Thin concentric technology rings (solid, dashed, bright segments).
            painter.setBrush(Qt.BrushStyle.NoBrush)
            ring_r = r * 1.205
            painter.setPen(QPen(QColor(*halo, int(alpha * 0.26)), 1.0))
            painter.drawEllipse(self._arc_rect(ring_r))
            dash_pen = QPen(QColor(*light, int(alpha * 0.44)), 1.5, Qt.PenStyle.CustomDashLine)
            dash_pen.setDashPattern((0.045, 0.030))
            painter.setPen(dash_pen)
            painter.drawEllipse(self._arc_rect(r * 1.245))
            for start, span in ((30.0, 40.0), (142.0, 24.0), (252.0, 56.0)):
                painter.setPen(QPen(QColor(*bright, int(alpha * 0.55)), 1.1))
                painter.drawArc(self._arc_rect(ring_r), int(start * 16), int(span * 16))

            # C. Discontinuous equatorial segment band with varied lengths and gaps.
            seg_r = r * 1.315
            for start, span in self._BAND_SEGMENTS:
                painter.setPen(QPen(QColor(*dim, int(alpha * 0.82)), max(3.0, r * 0.070)))
                painter.drawArc(self._arc_rect(seg_r), int(start * 16), int(span * 16))
                painter.setPen(QPen(QColor(*bright, int(alpha * 0.85)), max(1.2, r * 0.040)))
                painter.drawArc(self._arc_rect(seg_r - 1.5), int((start + 1.5) * 16), int((span - 3) * 16))
                painter.setPen(QPen(QColor(255, 255, 255, int(alpha * 0.72)), 1.1))
                painter.drawArc(self._arc_rect(seg_r), int(start * 16), int(5 * 16))

            # D. Luminous nodes sitting in the gaps of the band.
            node_energy = profile["node_energy"]
            for node_index, angle in enumerate(self._BAND_NODES):
                rad = math.radians(angle)
                nx, ny = seg_r * math.cos(rad), -seg_r * math.sin(rad)
                flash = 1.0
                if node_energy >= 0.6:
                    flash = 0.50 + 0.50 * (math.sin(c["elapsed"] * 11.0 + node_index * 2.1) * 0.5 + 0.5)
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QColor(*peak, int(alpha * 0.26 * node_energy * flash)))
                painter.drawEllipse(QPointF(nx, ny), 5.5, 5.5)
                painter.setBrush(QColor(*bright, int(alpha * 0.90 * (0.55 + 0.45 * node_energy) * flash)))
                painter.drawEllipse(QPointF(nx, ny), 2.2, 2.2)

            # E. Glass/energy sphere: offset light, dark rim, faceted conical sheen.
            sphere_gradient = QRadialGradient(-r * 0.24, -r * 0.30, r * 1.28)
            state = c["state"]
            state_rgb = c["rgb"]
            if state is OrbVisualState.AUTHORIZATION:
                sphere_gradient.setColorAt(0.0, QColor(26, 13, 2, min(255, alpha + 14)))
                sphere_gradient.setColorAt(0.50, QColor(64, 32, 4, int(alpha * 0.96)))
                sphere_gradient.setColorAt(0.82, QColor(150, 84, 10, int(alpha * 0.70)))
            elif state is OrbVisualState.AUTOMATION:
                sphere_gradient.setColorAt(0.0, QColor(30, 3, 9, min(255, alpha + 14)))
                sphere_gradient.setColorAt(0.50, QColor(66, 7, 16, int(alpha * 0.96)))
                sphere_gradient.setColorAt(0.82, QColor(140, 28, 33, int(alpha * 0.72)))
            elif state is OrbVisualState.PROCESSING:
                sphere_gradient.setColorAt(0.0, QColor(16, 4, 38, min(255, alpha + 14)))
                sphere_gradient.setColorAt(0.50, QColor(36, 10, 74, int(alpha * 0.96)))
                sphere_gradient.setColorAt(0.82, QColor(96, 42, 158, int(alpha * 0.70)))
            elif state is OrbVisualState.SPEAKING:
                sphere_gradient.setColorAt(0.0, QColor(2, 26, 20, min(255, alpha + 14)))
                sphere_gradient.setColorAt(0.50, QColor(4, 60, 38, int(alpha * 0.96)))
                sphere_gradient.setColorAt(0.82, QColor(16, 116, 68, int(alpha * 0.70)))
            else:
                sphere_gradient.setColorAt(0.0, QColor(2, 9, 26, min(255, alpha + 14)))
                sphere_gradient.setColorAt(0.50, QColor(5, 28, 68, int(alpha * 0.96)))
                sphere_gradient.setColorAt(0.82, QColor(14, 84, 150, int(alpha * 0.70)))
            sphere_gradient.setColorAt(1.0, QColor(*state_rgb, 0))
            painter.setBrush(sphere_gradient)
            painter.setPen(QPen(QColor(*halo, int(alpha * 0.50)), 2.0))
            painter.drawEllipse(self._arc_rect(r))

            vignette = QRadialGradient(0, r * 0.08, r)
            vignette.setColorAt(0.0, QColor(*depth, 0))
            vignette.setColorAt(0.55, QColor(*depth, 0))
            vignette.setColorAt(0.86, QColor(*depth, int(alpha * 0.45)))
            vignette.setColorAt(1.0, QColor(*depth, int(alpha * 0.72)))
            painter.setBrush(vignette)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(self._arc_rect(r))

            sheen = QConicalGradient(0, 0, -48.0)
            sheen.setColorAt(0.00, QColor(255, 255, 255, int(alpha * 0.06)))
            sheen.setColorAt(0.16, QColor(*halo, 0))
            sheen.setColorAt(0.52, QColor(*depth, int(alpha * 0.12)))
            sheen.setColorAt(0.74, QColor(*halo, 0))
            sheen.setColorAt(1.00, QColor(255, 255, 255, int(alpha * 0.06)))
            painter.setBrush(sheen)
            painter.drawEllipse(self._arc_rect(r))

            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(QPen(QColor(*peak, int(alpha * 0.42)), 1.2))
            painter.drawEllipse(QRectF(-r + 1, -r + 1, (r - 1) * 2, (r - 1) * 2))
            for inner_factor, opacity, width in ((0.84, 0.22, 1.2), (0.66, 0.30, 1.0)):
                painter.setPen(QPen(QColor(*bright, int(alpha * opacity)), width))
                painter.drawEllipse(self._arc_rect(r * inner_factor))
            for start, span in ((24.0, 30.0), (150.0, 20.0), (262.0, 36.0)):
                painter.setPen(QPen(QColor(*peak, int(alpha * 0.45)), 1.0))
                painter.drawArc(self._arc_rect(r * 0.84), int(start * 16), int(span * 16))

            gloss = QRadialGradient(-r * 0.34, -r * 0.38, r * 0.62)
            gloss.setColorAt(0.0, QColor(*peak, int(alpha * 0.14)))
            gloss.setColorAt(0.36, QColor(*halo, int(alpha * 0.05)))
            gloss.setColorAt(1.0, QColor(*halo, 0))
            painter.setBrush(gloss)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(self._arc_rect(r))

            # F. Deep inner glow and hot plasma centre behind the emblem.
            energy_r = r * 0.46
            energy = QRadialGradient(-energy_r * 0.18, -energy_r * 0.20, energy_r)
            energy.setColorAt(0.0, QColor(*peak, int(alpha * 0.42 * profile["core_intensity"])))
            energy.setColorAt(0.28, QColor(*light, int(alpha * 0.28 * profile["core_intensity"])))
            energy.setColorAt(0.62, QColor(*dim, int(alpha * 0.16)))
            energy.setColorAt(1.0, QColor(*dim, 0))
            painter.setBrush(energy)
            painter.drawEllipse(self._arc_rect(energy_r))
            plasma_r = r * 0.26
            plasma = QRadialGradient(0, r * 0.04, plasma_r)
            plasma.setColorAt(0.0, QColor(226, 250, 255, min(255, int(alpha * 0.78 * profile["core_intensity"]))))
            plasma.setColorAt(0.35, QColor(*peak, int(alpha * 0.40 * profile["core_intensity"])))
            plasma.setColorAt(1.0, QColor(*peak, 0))
            painter.setBrush(plasma)
            painter.drawEllipse(self._arc_rect(plasma_r))

            painter.restore()

        def _draw_core_particles(self, painter, c) -> None:
            """Deterministic particles orbiting inside the glass sphere."""
            center, r = c["center"], c["r"]
            alpha, profile = c["alpha"], c["profile"]
            intensity = profile["particle_intensity"]
            painter.setPen(Qt.PenStyle.NoPen)
            for orbit_factor, phase_offset, speed, dot, base_opacity in (
                (0.30, 0.0, 0.9, 1.6, 0.72), (0.44, 1.4, -0.7, 1.3, 0.60),
                (0.55, 2.8, 0.5, 1.6, 0.66), (0.68, 4.2, -0.45, 1.3, 0.55),
                (0.78, 5.5, 0.35, 1.6, 0.48), (0.62, 0.9, -0.6, 1.0, 0.62),
                (0.50, 3.6, 0.65, 1.0, 0.70), (0.85, 2.2, -0.3, 1.0, 0.42),
                (0.38, 5.0, 0.55, 1.0, 0.64), (0.72, 1.1, 0.4, 1.3, 0.52),
            ):
                angle = (
                    phase_offset
                    + c["elapsed"] * speed * (0.4 + 0.9 * intensity)
                    + c["frame"]["rotation_deg"] * 0.02
                )
                px = center + math.cos(angle) * r * orbit_factor
                py = center + math.sin(angle) * r * orbit_factor * 0.92
                painter.setBrush(QColor(*c["peak"], int(alpha * base_opacity * intensity)))
                painter.drawEllipse(QPointF(px, py), dot, dot)

        def _draw_emblem(self, painter, c) -> None:
            """The compact Atlas chevron shares the light of the nucleus."""
            center, size, alpha = c["center"], c["size"], c["alpha"]
            emblem = self._emblem_path
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(*c["depth"], int(alpha * 0.78)))
            painter.save()
            painter.translate(2.0, 2.6)
            painter.drawPath(emblem)
            painter.restore()
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(QPen(QColor(*c["halo"], int(alpha * 0.46)), 4.0))
            painter.drawPath(emblem)
            painter.setPen(QPen(QColor(*c["peak"], min(255, alpha + 4)), 1.1))
            painter.drawPath(emblem)
            emblem_gradient = QRadialGradient(center, center - size * 0.02, size * 0.20)
            emblem_gradient.setColorAt(0.0, QColor(244, 252, 255, min(255, alpha + 16)))
            emblem_gradient.setColorAt(0.52, QColor(*c["peak"], min(255, alpha + 10)))
            emblem_gradient.setColorAt(1.0, QColor(*c["light"], min(255, int(alpha * 0.90))))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(emblem_gradient)
            painter.drawPath(emblem)

        def _draw_state_activity(self, painter, c) -> None:
            """Per-state behaviour: voice bars, pulse rings, attention pings, sparks."""
            center, r = c["center"], c["r"]
            alpha, profile, state = c["alpha"], c["profile"], c["state"]
            period, elapsed = c["period"], c["elapsed"]
            painter.setBrush(Qt.BrushStyle.NoBrush)

            # Lateral voice bars: processing and speaking read as audio activity.
            wave = profile["wave_activity"]
            if wave > 0.02:
                speed = 9.5 if state is OrbVisualState.PROCESSING else 6.2
                for side in (-1, 1):
                    for bar in range(7):
                        level = abs(math.sin(elapsed * speed + bar * 1.25 + (0.7 if side > 0 else 0.0)))
                        envelope = (1.0 - 0.085 * bar) * wave * (0.30 + 0.70 * level)
                        height = max(2.0, r * 0.46 * envelope)
                        x = center + side * (r * (1.06 + 0.085 * bar))
                        painter.setPen(Qt.PenStyle.NoPen)
                        painter.setBrush(QColor(*c["peak"], int(alpha * (0.25 + 0.55 * envelope))))
                        painter.drawRoundedRect(QRectF(x - 1.7, center - height, 3.4, height * 2), 1.7, 1.7)
                painter.setBrush(Qt.BrushStyle.NoBrush)

            if state is OrbVisualState.SPEAKING and period:
                # Voice pulse: rings born in the nucleus expand while speaking.
                for offset in (0.0, 0.45):
                    p = ((elapsed / period) + offset) % 1.0
                    radius = r * (1.04 + 0.52 * p)
                    painter.setPen(QPen(QColor(*c["peak"], int(alpha * 0.40 * (1.0 - p))), 2.0))
                    painter.drawEllipse(self._arc_rect(radius).translated(QPointF(center, center)))

            ping = profile["ping_intensity"]
            if ping > 0.02 and period:
                # Authorization: concentric attention circles asking for approval.
                for offset in (0.0, 0.5):
                    p = ((elapsed / period) + offset) % 1.0
                    radius = r * (1.10 + 0.48 * p)
                    painter.setPen(QPen(QColor(*c["peak"], int(alpha * 0.34 * ping * (1.0 - p))), 1.8))
                    painter.drawEllipse(self._arc_rect(radius).translated(QPointF(center, center)))

            if state is OrbVisualState.LISTENING:
                for offset, opacity in ((0.00, 0.30), (0.34, 0.20), (0.67, 0.12)):
                    wave_p = (math.sin(c["frame"]["rotation_deg"] * 0.105 - offset * math.tau) + 1.0) * 0.5
                    radius = r * (1.42 + 0.10 * wave_p)
                    painter.setPen(QPen(QColor(*c["halo"], int(alpha * opacity * wave_p)), 2.0))
                    painter.drawEllipse(self._arc_rect(radius).translated(QPointF(center, center)))

            # Automation: quick bright arclets orbiting the running device.
            spark = profile["spark_intensity"]
            if spark > 0.02:
                radius = c["size"] * 0.452
                for index in range(4):
                    angle = (elapsed * 165.0 * (1.0 if index % 2 == 0 else -1.0) + index * 97.0) % 360.0
                    painter.save()
                    painter.translate(center, center)
                    painter.rotate(angle)
                    painter.setPen(QPen(QColor(*c["peak"], int(alpha * 0.75 * spark)), 1.8))
                    painter.drawArc(self._arc_rect(radius), 0, 15 * 16)
                    painter.setPen(QPen(QColor(255, 255, 255, int(alpha * 0.85 * spark)), 1.0))
                    painter.drawArc(self._arc_rect(radius), 0, 5 * 16)
                    painter.restore()

        def _draw_starfield(self, painter, c) -> None:
            """Sparse stellar specks riding the outer orbits, as in the reference."""
            if c["state"] is OrbVisualState.DEGRADED:
                return
            center, size = c["center"], c["size"]
            painter.setPen(Qt.PenStyle.NoPen)
            for index, (offset, radius_factor, squash, tilt, speck_radius) in enumerate((
                (0.0, 0.47, 0.44, 0.0, 1.6), (1.6, 0.52, 0.26, 201.0, 1.1),
                (2.9, 0.50, 0.56, 57.0, 1.6), (4.1, 0.42, 0.20, -41.0, 1.1),
                (5.3, 0.55, 0.40, 96.0, 1.1), (0.9, 0.44, 0.48, 152.0, 1.1),
                (3.6, 0.57, 0.30, 224.0, 1.6),
            )):
                speck_angle = math.radians(offset + c["frame"]["rotation_deg"] * (0.45 + 0.16 * index))
                speck_orbit = size * radius_factor
                x0 = math.cos(speck_angle) * speck_orbit
                y0 = math.sin(speck_angle) * speck_orbit * squash
                tilt_radians = math.radians(tilt)
                speck_x = int(center + x0 * math.cos(tilt_radians) - y0 * math.sin(tilt_radians))
                speck_y = int(center + x0 * math.sin(tilt_radians) + y0 * math.cos(tilt_radians))
                shimmer = 0.40 + 0.60 * ((math.sin(speck_angle * 2.0) + 1.0) * 0.5)
                painter.setBrush(QColor(*c["peak"], int(c["alpha"] * 0.60 * shimmer * c["profile"]["particle_intensity"])))
                dot = int(speck_radius)
                painter.drawEllipse(speck_x - dot, speck_y - dot, dot * 2, dot * 2)
    return OrbWindow()


def create_transcript_panel():
    """Small transcript panel with chat and minimal voice controls."""
    from PySide6.QtCore import Qt, Signal
    from PySide6.QtGui import QTextCursor
    from PySide6.QtWidgets import (
        QFileDialog,
        QFrame,
        QHBoxLayout,
        QLabel,
        QPlainTextEdit,
        QPushButton,
        QTextBrowser,
        QVBoxLayout,
        QWidget,
    )

    class ChatInput(QPlainTextEdit):
        """Multiline input that keeps Enter as the chat submission shortcut."""

        submit_requested = Signal()

        def setText(self, text: str) -> None:  # noqa: N802 (QLineEdit compatibility)
            self.setPlainText(text)
            self.moveCursor(QTextCursor.MoveOperation.End)

        def text(self) -> str:
            return self.toPlainText()

        def keyPressEvent(self, event) -> None:  # noqa: N802 (Qt API)
            if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
                if event.modifiers() == Qt.KeyboardModifier.NoModifier:
                    self.submit_requested.emit()
                    event.accept()
                    return
            super().keyPressEvent(event)

    class TranscriptPanel(QWidget):
        send_requested = Signal(str)
        attachment_send_requested = Signal(str, object)
        close_requested = Signal()
        voice_start_requested = Signal()
        voice_stop_requested = Signal()
        voice_retry_requested = Signal()
        _VOICE_STATUS = {"STARTING": "Iniciando voz", "LISTENING": "Escuchando", "TRANSCRIBING": "STT", "PROCESSING": "Procesando", "SPEAKING": "TTS", "RECOVERING": "Reintentando", "DEGRADED": "Error", "ERROR": "Error", "STOPPING": "Deteniendo", "STOPPED": "Desconectado"}
        _HIDDEN_SYSTEM_MESSAGES = frozenset(
            {
                "Estado: STARTING",
                "Estado: IDLE",
                "Estado: LISTENING",
                "Estado: PROCESSING",
                "Estado: RECOVERING",
                "Esperando voz...",
            }
        )

        def __init__(self) -> None:
            super().__init__()
            self._hide_on_close = False
            self.setWindowTitle("Atlas - transcripcion")
            self.setObjectName("atlasTranscriptPanel")
            self.setWindowFlags(Qt.WindowType.Window)
            self.resize(380, 440)
            self.setMinimumSize(360, 400)
            self.setStyleSheet(
                "QWidget#atlasTranscriptPanel { background: #060d1a; }"
                "QLabel { color: #9fd8ff; font-size: 12px; font-weight: 600; }"
                "QPushButton { background: #10233c; color: #d8efff; border: 1px solid #2d4b69; "
                "border-radius: 6px; padding: 6px 12px; font-size: 12px; }"
                "QPushButton:hover { background: #173154; border-color: #3f6d99; }"
                "QPushButton:pressed { background: #1d3f6b; }"
            )
            layout = QVBoxLayout(self)
            layout.setContentsMargins(12, 12, 12, 12)
            layout.setSpacing(8)
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
            self._view.setStyleSheet(
                "QTextBrowser { background: #0b1220; color: #e6edf7; border: 1px solid #24344a; "
                "border-radius: 8px; font-size: 15px; padding: 8px; }"
            )
            layout.addWidget(self._view)
            self._pending_attachment = None
            self._attachment_preview = QFrame(self)
            self._attachment_preview.setFrameShape(QFrame.Shape.StyledPanel)
            self._attachment_preview.setStyleSheet(
                "QFrame { background: #132238; border: 1px solid #2f5b82; border-radius: 6px; }"
            )
            attachment_layout = QHBoxLayout(self._attachment_preview)
            self._attachment_icon = QLabel("[archivo]", self._attachment_preview)
            self._attachment_details = QLabel(self._attachment_preview)
            self._attachment_remove_button = QPushButton("X", self._attachment_preview)
            self._attachment_remove_button.setToolTip("Quitar adjunto")
            self._attachment_remove_button.setFixedWidth(30)
            self._attachment_remove_button.clicked.connect(self._clear_attachment)
            attachment_layout.addWidget(self._attachment_icon)
            attachment_layout.addWidget(self._attachment_details, 1)
            attachment_layout.addWidget(self._attachment_remove_button)
            self._attachment_preview.hide()
            layout.addWidget(self._attachment_preview)
            input_layout = QHBoxLayout()
            self._attachment_button = QPushButton("+", self)
            self._attachment_button.setToolTip("Adjuntar archivo")
            self._attachment_button.clicked.connect(self._choose_attachment)
            self._input = ChatInput(self)
            self._input.setPlaceholderText("Escribe un mensaje...")
            self._input.setFixedHeight(96)
            self._input.setStyleSheet(
                "QPlainTextEdit { background: #101a2a; color: #edf5ff; border: 1px solid #2d4b69; "
                "border-radius: 7px; font-size: 15px; padding: 8px; }"
            )
            self._send_button = QPushButton("Enviar", self)
            self._input.submit_requested.connect(self._submit_input)
            self._send_button.clicked.connect(self._submit_input)
            input_layout.addWidget(self._attachment_button)
            input_layout.addWidget(self._input)
            input_layout.addWidget(self._send_button)
            layout.addLayout(input_layout)

        def _submit_input(self) -> None:
            text = self._input.text().strip()
            if not text:
                return
            self._input.clear()
            if self._pending_attachment is not None:
                attachment = self._pending_attachment
                self._clear_attachment()
                self.attachment_send_requested.emit(text, attachment)
                return
            self.send_requested.emit(text)

        def _choose_attachment(self) -> None:
            path, _selected_filter = QFileDialog.getOpenFileName(self, "Seleccionar archivo")
            if not path:
                return
            from pathlib import Path
            from core.request_gateway import RequestAttachment

            file_path = Path(path)
            try:
                size_bytes = file_path.stat().st_size
            except OSError:
                return
            media_type = mimetypes.guess_type(str(file_path))[0] or "application/octet-stream"
            self._pending_attachment = RequestAttachment(
                attachment_id=uuid.uuid4().hex,
                name=file_path.name,
                media_type=media_type,
                size_bytes=size_bytes,
                local_reference=str(file_path),
            )
            self._attachment_details.setText(
                f"{file_path.name}\n{media_type} · {self._format_size(size_bytes)}"
            )
            self._attachment_preview.show()

        @staticmethod
        def _format_size(size_bytes: int) -> str:
            if size_bytes < 1024:
                return f"{size_bytes} B"
            if size_bytes < 1024 * 1024:
                return f"{size_bytes / 1024:.1f} KB"
            return f"{size_bytes / (1024 * 1024):.1f} MB"

        def _clear_attachment(self) -> None:
            self._pending_attachment = None
            self._attachment_details.clear()
            self._attachment_preview.hide()

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
            text = str(message).strip()
            if text in self._HIDDEN_SYSTEM_MESSAGES:
                return
            self._append_turn("Sistema", text, "#162235", "#334761")

        def append_user(self, message: str) -> None:
            self._append_turn("Usuario", message, "#123b5c", "#2b6d9a")

        def append_transcription(self, transcription: str) -> None:
            self._append_turn("Tú", transcription, "#123b5c", "#2b6d9a")

        def append_response(self, response: str) -> None:
            self._append_turn("Atlas", response, "#162235", "#38526f")

        def append_error(self, error: str) -> None:
            self._append_turn("Error", error, "#48202a", "#9b4759")

        def _append_turn(self, sender: str, message: str, background: str, border: str) -> None:
            scrollbar = self._view.verticalScrollBar()
            follow_tail = scrollbar.value() >= scrollbar.maximum() - 24
            previous_scroll_value = scrollbar.value()
            safe_sender = html.escape(str(sender))
            safe_message = html.escape(str(message)).replace("\n", "<br>")
            self._view.moveCursor(QTextCursor.MoveOperation.End)
            self._view.insertHtml(
                f'<table width="100%" cellspacing="0" cellpadding="0" '
                f'style="margin-top: 6px; margin-bottom: 14px;">'
                f'<tr><td bgcolor="{background}" style="border: 1px solid {border}; '
                f'padding: 10px 12px;">'
                f'<span style="color: #8fd3ff; font-weight: 700;">{safe_sender}:</span> '
                f'<span style="color: #eef5ff;">{safe_message}</span>'
                f'</td></tr></table>'
            )
            self._view.insertHtml("<br>")
            if follow_tail:
                scrollbar.setValue(scrollbar.maximum())
            else:
                scrollbar.setValue(previous_scroll_value)

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
