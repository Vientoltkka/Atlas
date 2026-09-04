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
CORE_RADIUS_FACTOR = 0.275
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
    },
    OrbVisualState.PROCESSING: {
        "ring_activity": 1.00, "ring_speed": 1.00, "ring_angle": 1.20, "ring_amplitude": 1.08,
        "pulse_strength": 0.66, "halo_intensity": 1.12, "halo_strength": 1.12,
        "core_intensity": 1.18, "core_pulse": 0.86, "particle_intensity": 0.82,
        "segment_activity": 1.00, "base_intensity": 0.88,
    },
    OrbVisualState.SPEAKING: {
        "ring_activity": 0.30, "ring_speed": 0.64, "ring_angle": 0.98, "ring_amplitude": 1.00,
        "pulse_strength": 1.00, "halo_intensity": 1.28, "halo_strength": 1.28,
        "core_intensity": 1.12, "core_pulse": 1.18, "particle_intensity": 0.86,
        "segment_activity": 0.72, "base_intensity": 1.04,
    },
    OrbVisualState.AUTOMATION: {
        "ring_activity": 1.34, "ring_speed": 1.42, "ring_angle": 1.34, "ring_amplitude": 1.16,
        "pulse_strength": 0.92, "halo_intensity": 1.22, "halo_strength": 1.22,
        "core_intensity": 1.22, "core_pulse": 1.04, "particle_intensity": 0.96,
        "segment_activity": 1.25, "base_intensity": 1.12,
    },
    OrbVisualState.AUTHORIZATION: {
        "ring_activity": 0.25, "ring_speed": 0.42, "ring_angle": 0.78, "ring_amplitude": 0.82,
        "pulse_strength": 0.52, "halo_intensity": 0.92, "halo_strength": 0.92,
        "core_intensity": 1.04, "core_pulse": 0.70, "particle_intensity": 0.58,
        "segment_activity": 0.56, "base_intensity": 0.78,
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
    from PySide6.QtCore import QSize, Qt, Signal, QTimer
    from PySide6.QtGui import QColor, QIcon, QLinearGradient, QPainter, QPainterPath, QPen, QPixmap, QRadialGradient
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
            """Create the Atlas chevron and its detached lower triangle."""
            size = self.width()
            path = QPainterPath()
            # Two diagonal arms form an open chevron; no horizontal A crossbar is used.
            path.moveTo(size * 0.500, size * 0.405)
            path.lineTo(size * 0.395, size * 0.565)
            path.lineTo(size * 0.438, size * 0.578)
            path.lineTo(size * 0.500, size * 0.470)
            path.closeSubpath()
            path.moveTo(size * 0.500, size * 0.405)
            path.lineTo(size * 0.605, size * 0.565)
            path.lineTo(size * 0.562, size * 0.578)
            path.lineTo(size * 0.500, size * 0.470)
            path.closeSubpath()
            # The isolated lower triangle keeps generous negative space inside the core.
            path.moveTo(size * 0.500, size * 0.590)
            path.lineTo(size * 0.464, size * 0.646)
            path.lineTo(size * 0.536, size * 0.646)
            path.closeSubpath()
            return path

        def paintEvent(self, event) -> None:  # noqa: N802 (Qt API)
            """Render the layered, fully procedural Atlas energy core."""
            frame = self.current_frame()
            red, green, blue, base_alpha = color_for_state(self._state)
            alpha = max(0, min(255, int(base_alpha * frame["alpha_factor"])))
            size = self.width()
            center = size // 2
            profile = visual_profile(self._state)
            core_scale = 1.0 + (frame["scale"] - 1.0) * profile["core_pulse"]
            core_radius = max(18, int(size * CORE_RADIUS_FACTOR * core_scale))
            orbit_radius = max(28, int(size * 0.335 * frame["scale"] * profile["ring_amplitude"]))
            halo_radius = max(38, int(size * 0.435 * frame["scale"] * profile["halo_strength"]))
            phase = math.radians(frame["rotation_deg"])
            palette = {
                OrbVisualState.PROCESSING: ((187, 116, 255), (75, 38, 132), (215, 175, 255), (165, 94, 255), (245, 232, 255)),
                OrbVisualState.SPEAKING: ((80, 235, 157), (20, 111, 69), (152, 255, 205), (57, 220, 128), (222, 255, 237)),
                OrbVisualState.AUTHORIZATION: ((255, 181, 60), (166, 95, 15), (255, 224, 142), (255, 194, 70), (255, 245, 204)),
                OrbVisualState.AUTOMATION: ((255, 78, 78), (145, 26, 35), (255, 163, 163), (242, 62, 69), (255, 232, 232)),
            }
            halo_rgb, ring_dim_rgb, ring_light_rgb, ring_bright_rgb, ring_peak_rgb = palette.get(
                self._state, ((70, 205, 255), (42, 142, 255), (120, 225, 255), (53, 179, 255), (216, 252, 255))
            )
            active_glow = profile["halo_strength"]
            projection_rgb = ring_bright_rgb
            base_alpha = int(alpha * profile["base_intensity"])

            painter = QPainter(self)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            painter.setBrush(Qt.BrushStyle.NoBrush)

            # Narrow halo shells frame the structure without flattening the orbital silhouette.
            for multiplier, opacity, width in (
                (1.10, 0.030 * active_glow, 12.0),
                (1.03, 0.060 * active_glow, 8.0),
                (0.96, 0.105 * active_glow, 4.5),
                (0.89, 0.22 * active_glow, 2.0),
            ):
                radius = int(halo_radius * multiplier)
                painter.setPen(QPen(QColor(*halo_rgb, int(alpha * opacity)), width))
                painter.drawEllipse(center - radius, center - radius, radius * 2, radius * 2)

            # Projection stays behind the sphere, so the core looks suspended rather than painted on.
            base_y = int(size * 0.79)
            beam_top = center + int(core_radius * 0.54)
            projection = QLinearGradient(center, beam_top, center, base_y)
            projection.setColorAt(0.0, QColor(*projection_rgb, int(base_alpha * 0.34)))
            projection.setColorAt(0.62, QColor(*projection_rgb, int(base_alpha * 0.11)))
            projection.setColorAt(1.0, QColor(*projection_rgb, 0))
            beam_width = max(16, int(size * 0.105))
            beam = QPainterPath()
            beam.moveTo(center - beam_width * 0.28, beam_top)
            beam.lineTo(center + beam_width * 0.28, beam_top)
            beam.lineTo(center + beam_width, base_y)
            beam.lineTo(center - beam_width, base_y)
            beam.closeSubpath()
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(projection)
            painter.drawPath(beam)
            painter.setPen(QPen(QColor(*projection_rgb, int(base_alpha * 0.060)), max(10.0, size * 0.046)))
            painter.drawLine(center, beam_top, center, base_y)
            painter.setPen(QPen(QColor(*projection_rgb, int(base_alpha * 0.23)), max(3.0, size * 0.011)))
            painter.drawLine(center, beam_top, center, base_y)
            for index, (multiplier, squash, opacity, width) in enumerate(((0.14, 0.023, 1.00, 3.4), (0.21, 0.034, 0.60, 2.2), (0.29, 0.044, 0.28, 1.4))):
                half_width = int(size * multiplier)
                half_height = max(2, int(size * squash))
                painter.setPen(QPen(QColor(*projection_rgb, int(base_alpha * opacity)), width))
                painter.drawEllipse(center - half_width, base_y - half_height, half_width * 2, half_height * 2)
                if index < 2:
                    painter.setPen(QPen(QColor(*ring_peak_rgb, int(base_alpha * opacity)), max(1.0, width * 0.55)))
                    painter.drawArc(center - half_width, base_y - half_height, half_width * 2, half_height * 2, (28 + index * 82) * 16, 46 * 16)

            orbit_speed = (
                (1.34, 2.10, -1.15)
                if self._state is OrbVisualState.AUTOMATION
                else (1.00, 1.58, -0.82)
                if self._state is OrbVisualState.PROCESSING
                else (0.30, 0.20, -0.14)
                if self._state is OrbVisualState.SPEAKING
                else (0.25, -0.15, 0.10)
                if self._state is OrbVisualState.AUTHORIZATION
                else (0.18, -0.12, 0.08)
                if self._state is OrbVisualState.LISTENING
                else (0.16, -0.10, 0.07)
            )
            orbit_speed = tuple(speed * profile["ring_activity"] * profile["ring_speed"] for speed in orbit_speed)
            ring_specs = (
                (frame["rotation_deg"] * orbit_speed[0], 0.46 * profile["ring_angle"], 1.10, 18),
                (frame["rotation_deg"] * orbit_speed[1] + 57.0, 0.62 * profile["ring_angle"], 1.00, 126),
                (frame["rotation_deg"] * orbit_speed[2] + 119.0, 0.34 * profile["ring_angle"], 0.90, 247),
            )
            # Dim back segments establish that each orbit continues behind the core.
            for angle, squash, radius_factor, start_angle in ring_specs:
                radius = int(orbit_radius * radius_factor)
                painter.save()
                painter.translate(center, center)
                painter.rotate(angle)
                painter.scale(1.0, squash)
                painter.setPen(QPen(QColor(*ring_dim_rgb, int(alpha * 0.38)), 4.2))
                painter.drawArc(-radius, -radius, radius * 2, radius * 2, start_angle * 16, 112 * 16)
                painter.setPen(QPen(QColor(*ring_light_rgb, int(alpha * 0.20)), 1.6))
                painter.drawArc(-radius, -radius, radius * 2, radius * 2, (start_angle + 154) * 16, 58 * 16)
                painter.restore()

            # A layered shell, offset light source and dark rim give the 2D core spherical volume.
            shell_gradient = QRadialGradient(
                center - core_radius * 0.24,
                center - core_radius * 0.30,
                core_radius * 1.28,
            )
            if self._state is OrbVisualState.AUTHORIZATION:
                shell_gradient.setColorAt(0.0, QColor(38, 20, 3, min(255, alpha + 18)))
                shell_gradient.setColorAt(0.48, QColor(83, 43, 4, int(alpha * 0.96)))
                shell_gradient.setColorAt(0.77, QColor(184, 103, 15, int(alpha * 0.82)))
            elif self._state is OrbVisualState.AUTOMATION:
                shell_gradient.setColorAt(0.0, QColor(46, 4, 12, min(255, alpha + 18)))
                shell_gradient.setColorAt(0.48, QColor(104, 10, 24, int(alpha * 0.96)))
                shell_gradient.setColorAt(0.77, QColor(212, 42, 50, int(alpha * 0.86)))
            elif self._state is OrbVisualState.PROCESSING:
                shell_gradient.setColorAt(0.0, QColor(23, 5, 52, min(255, alpha + 18)))
                shell_gradient.setColorAt(0.48, QColor(57, 16, 112, int(alpha * 0.96)))
                shell_gradient.setColorAt(0.77, QColor(145, 63, 228, int(alpha * 0.84)))
            elif self._state is OrbVisualState.SPEAKING:
                shell_gradient.setColorAt(0.0, QColor(3, 39, 29, min(255, alpha + 18)))
                shell_gradient.setColorAt(0.48, QColor(5, 93, 58, int(alpha * 0.96)))
                shell_gradient.setColorAt(0.77, QColor(23, 177, 103, int(alpha * 0.84)))
            else:
                shell_gradient.setColorAt(0.0, QColor(3, 13, 38, min(255, alpha + 18)))
                shell_gradient.setColorAt(0.48, QColor(7, 42, 104, int(alpha * 0.96)))
                shell_gradient.setColorAt(0.77, QColor(20, 122, 214, int(alpha * 0.84)))
            shell_gradient.setColorAt(1.0, QColor(red, green, blue, 0))
            painter.setBrush(shell_gradient)
            painter.setPen(QPen(QColor(*halo_rgb, int(alpha * 0.56)), 2.2))
            painter.drawEllipse(center - core_radius, center - core_radius, core_radius * 2, core_radius * 2)

            gloss_radius = max(8, int(core_radius * 0.62))
            gloss_gradient = QRadialGradient(center - core_radius * 0.34, center - core_radius * 0.38, gloss_radius)
            gloss_gradient.setColorAt(0.0, QColor(*ring_peak_rgb, int(alpha * 0.24)))
            gloss_gradient.setColorAt(0.36, QColor(*halo_rgb, int(alpha * 0.08)))
            gloss_gradient.setColorAt(1.0, QColor(*halo_rgb, 0))
            painter.setBrush(gloss_gradient)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(center - core_radius, center - core_radius, core_radius * 2, core_radius * 2)

            for multiplier, opacity, width in ((0.95, 0.12, 4.8), (0.85, 0.21, 2.7), (0.73, 0.31, 1.4)):
                radius = int(core_radius * multiplier)
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.setPen(QPen(QColor(*ring_bright_rgb, int(alpha * opacity)), width))
                painter.drawEllipse(center - radius, center - radius, radius * 2, radius * 2)
            painter.setPen(QPen(QColor(*ring_peak_rgb, int(alpha * 0.64)), 1.9))
            for start_angle, span_angle in ((18, 36), (151, 28), (275, 42)):
                painter.drawArc(center - core_radius, center - core_radius, core_radius * 2, core_radius * 2, start_angle * 16, span_angle * 16)

            energy_radius = max(10, int(core_radius * 0.64))
            energy_gradient = QRadialGradient(center - energy_radius * 0.18, center - energy_radius * 0.20, energy_radius)
            energy_gradient.setColorAt(0.0, QColor(*ring_peak_rgb, int(alpha * 0.78 * profile["core_intensity"])))
            energy_gradient.setColorAt(0.22, QColor(*ring_light_rgb, int(alpha * 0.62 * profile["core_intensity"])))
            energy_gradient.setColorAt(0.58, QColor(*ring_dim_rgb, int(alpha * 0.34)))
            energy_gradient.setColorAt(1.0, QColor(*ring_dim_rgb, 0))
            painter.setBrush(energy_gradient)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(center - energy_radius, center - energy_radius, energy_radius * 2, energy_radius * 2)
            # A hot plasma centre behind the emblem matches the reference luminous core.
            plasma_radius = max(6, int(core_radius * 0.55))
            plasma_gradient = QRadialGradient(center, center + core_radius * 0.05, plasma_radius)
            plasma_gradient.setColorAt(0.0, QColor(226, 250, 255, min(255, int(alpha * 1.05 * profile["core_intensity"]))))
            plasma_gradient.setColorAt(0.35, QColor(*ring_peak_rgb, int(alpha * 0.62 * profile["core_intensity"])))
            plasma_gradient.setColorAt(1.0, QColor(*ring_peak_rgb, 0))
            painter.setBrush(plasma_gradient)
            painter.drawEllipse(center - plasma_radius, center - plasma_radius, plasma_radius * 2, plasma_radius * 2)

            if self._state is OrbVisualState.LISTENING:
                # Three cheap exterior ripples make listening distinct from idle.
                for offset, opacity in ((0.00, 0.30), (0.34, 0.20), (0.67, 0.12)):
                    wave = (math.sin(phase - offset * math.tau) + 1.0) * 0.5
                    radius = int(halo_radius * (0.90 + 0.22 * wave))
                    painter.setBrush(Qt.BrushStyle.NoBrush)
                    painter.setPen(QPen(QColor(*halo_rgb, int(alpha * opacity * wave)), 2.0))
                    painter.drawEllipse(center - radius, center - radius, radius * 2, radius * 2)

            # Fine energy points make the interior feel active rather than uniformly filled.
            for x_factor, y_factor, radius, opacity in (
                (-0.28, -0.18, 2, 0.72), (0.23, -0.27, 1, 0.58),
                (0.31, 0.16, 2, 0.48), (-0.17, 0.30, 1, 0.66),
                (0.04, 0.11, 1, 0.78),
            ):
                painter.setBrush(QColor(*ring_peak_rgb, int(alpha * opacity * profile["particle_intensity"])))
                painter.drawEllipse(
                    int(center + energy_radius * x_factor) - radius,
                    int(center + energy_radius * y_factor) - radius,
                    radius * 2,
                    radius * 2,
                )

            # Bright front segments and sparse nodes complete the orbital depth cue.
            for index, (angle, squash, radius_factor, start_angle) in enumerate(ring_specs):
                radius = int(orbit_radius * radius_factor)
                painter.save()
                painter.translate(center, center)
                painter.rotate(angle)
                painter.scale(1.0, squash)
                painter.setPen(QPen(QColor(*ring_bright_rgb, int(alpha * 0.95)), 6.5 * profile["segment_activity"]))
                painter.drawArc(-radius, -radius, radius * 2, radius * 2, (start_angle + 218) * 16, 64 * 16)
                painter.drawArc(-radius, -radius, radius * 2, radius * 2, (start_angle + 278) * 16, 40 * 16)
                painter.setPen(QPen(QColor(*ring_peak_rgb, min(255, alpha + 8)), 2.4))
                painter.drawArc(-radius, -radius, radius * 2, radius * 2, (start_angle + 240) * 16, 22 * 16)
                painter.setPen(QPen(QColor(*ring_peak_rgb, int(alpha * 0.82)), 1.1))
                painter.drawArc(-radius, -radius, radius * 2, radius * 2, (start_angle + 222) * 16, 9 * 16)
                painter.drawArc(-radius, -radius, radius * 2, radius * 2, (start_angle + 269) * 16, 7 * 16)
                node_angle = math.radians(start_angle + 252 + index * 17)
                node_x, node_y = int(math.cos(node_angle) * radius), int(math.sin(node_angle) * radius)
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QColor(*ring_peak_rgb, int(alpha * 0.96)))
                painter.drawEllipse(node_x - 3, node_y - 3, 6, 6)
                painter.restore()

            # The unchanged Atlas geometry gets a dark extrusion plus two luminous passes.
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(2, 18, 48, int(alpha * 0.82)))
            painter.save()
            painter.translate(2.0, 3.0)
            painter.drawPath(self._emblem_path)
            painter.restore()
            painter.setPen(QPen(QColor(*halo_rgb, int(alpha * 0.40)), 5.0))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawPath(self._emblem_path)
            painter.setPen(QPen(QColor(*ring_peak_rgb, min(255, alpha + 4)), 1.2))
            painter.drawPath(self._emblem_path)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(*ring_peak_rgb, min(255, alpha + 12)))
            painter.drawPath(self._emblem_path)

            # Compact, irregular near-core particles use the existing animation phase only.
            if self._state is not OrbVisualState.DEGRADED:
                for x, y, offset, radius in (
                    (0.27, 0.37, 0.0, 2), (0.73, 0.31, 1.0, 1),
                    (0.71, 0.67, 2.0, 2), (0.29, 0.70, 3.0, 1),
                    (0.23, 0.53, 4.0, 1), (0.77, 0.50, 5.0, 2),
                    (0.37, 0.24, 2.6, 1), (0.63, 0.22, 4.6, 1),
                    (0.42, 0.76, 1.7, 1), (0.59, 0.75, 3.8, 1),
                ):
                    shimmer = 0.36 + 0.64 * ((math.sin(phase + offset) + 1.0) * 0.5)
                    painter.setPen(Qt.PenStyle.NoPen)
                    painter.setBrush(QColor(*ring_peak_rgb, int(alpha * 0.54 * shimmer * profile["particle_intensity"])))
                    painter.drawEllipse(int(size * x) - radius, int(size * y) - radius, radius * 2, radius * 2)

            painter.end()
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
