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


ORB_SIZE = 280
_ACTIVE_ORB_SIZES: dict[OrbVisualState, int] = {
    OrbVisualState.LISTENING: 460,
    OrbVisualState.PROCESSING: 500,
    OrbVisualState.SPEAKING: 460,
    OrbVisualState.AUTHORIZATION: 500,
}

_STATE_COLORS: dict[OrbVisualState, tuple[int, int, int, int]] = {
    OrbVisualState.IDLE: (56, 185, 255, 220),
    OrbVisualState.STARTING: (105, 225, 255, 235),
    OrbVisualState.LISTENING: (80, 225, 255, 245),
    OrbVisualState.PROCESSING: (68, 178, 255, 245),
    OrbVisualState.SPEAKING: (82, 235, 255, 250),
    OrbVisualState.AUTHORIZATION: (255, 178, 58, 245),
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
    OrbVisualState.RECOVERING: 0.6,  # quick blink
    OrbVisualState.DEGRADED: None,   # static dim
    OrbVisualState.STOPPING: 1.5,    # gentle shrink/fade
}

ANIMATION_FPS = 24


def color_for_state(state: OrbVisualState) -> tuple[int, int, int, int]:
    """Deterministic RGBA for one visual state."""
    return _STATE_COLORS[state]


def animation_period(state: OrbVisualState) -> float | None:
    """Animation period in seconds, or None when the state is static."""
    return _ANIMATION_PERIODS[state]


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

    if state is OrbVisualState.IDLE:
        scale = 1.0 + 0.03 * wave
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
        scale = 1.0
        alpha_factor = 1.0
        rotation_deg = 360.0 * phase
    elif state is OrbVisualState.SPEAKING:
        scale = 1.02 + 0.03 * (wave * 0.5 + 0.5)
        alpha_factor = 0.85 + 0.15 * (wave * 0.5 + 0.5)
        rotation_deg = 360.0 * phase
    elif state is OrbVisualState.AUTHORIZATION:
        scale = 1.0 + 0.015 * wave
        alpha_factor = 0.78 + 0.20 * (wave * 0.5 + 0.5)
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
    from PySide6.QtCore import Qt, Signal, QTimer
    from PySide6.QtGui import QColor, QIcon, QPainter, QPainterPath, QPen, QPixmap, QRadialGradient
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
            self._emblem_path = self._build_atlas_emblem()

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
            self._resize_for_state()
            self._animation_started_at = time.monotonic()
            self._update_tray_icon()
            self._update_timer()
            self.update()

        def _resize_for_state(self) -> None:
            """Resize around the current centre, then keep clear of the chat."""
            target_size = size_for_state(self._state)
            if self.width() == target_size:
                return
            centre = self.frameGeometry().center()
            self.setFixedSize(target_size, target_size)
            self._emblem_path = self._build_atlas_emblem()
            self.move(centre.x() - target_size // 2, centre.y() - target_size // 2)
            self._avoid_visible_transcript_overlap()

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
                target_x = max(bounds.left(), min(target_x, bounds.right() - self.width() + 1))
                target_y = max(bounds.top(), min(target_y, bounds.bottom() - self.height() + 1))
            self.move(target_x, target_y)

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

        def showEvent(self, event) -> None:  # noqa: N802 (Qt API)
            super().showEvent(event)
            QTimer.singleShot(0, self._avoid_visible_transcript_overlap)

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

        def _build_atlas_emblem(self) -> QPainterPath:
            """Create the cached, bar-free geometric Atlas emblem silhouette."""
            size = self.width()
            path = QPainterPath()
            # Two filled, inclined arms leave a triangular inner cut: this is a symbol, not a typographic A.
            path.moveTo(size * 0.50, size * 0.405)
            path.lineTo(size * 0.402, size * 0.635)
            path.lineTo(size * 0.447, size * 0.620)
            path.lineTo(size * 0.500, size * 0.485)
            path.closeSubpath()
            path.moveTo(size * 0.50, size * 0.405)
            path.lineTo(size * 0.598, size * 0.635)
            path.lineTo(size * 0.553, size * 0.620)
            path.lineTo(size * 0.500, size * 0.485)
            path.closeSubpath()
            return path

        def paintEvent(self, event) -> None:  # noqa: N802 (Qt API)
            """Render the layered, fully procedural Atlas energy core."""
            frame = self.current_frame()
            red, green, blue, base_alpha = color_for_state(self._state)
            alpha = max(0, min(255, int(base_alpha * frame["alpha_factor"])))
            size = self.width()
            center = size // 2
            core_radius = max(18, int(size * 0.245 * frame["scale"]))
            orbit_radius = max(28, int(size * 0.335 * frame["scale"]))
            halo_radius = max(38, int(size * 0.435 * frame["scale"]))
            phase = math.radians(frame["rotation_deg"])
            is_authorization = self._state is OrbVisualState.AUTHORIZATION
            halo_rgb = (255, 181, 60) if is_authorization else (70, 205, 255)
            ring_dim_rgb = (166, 95, 15) if is_authorization else (42, 142, 255)
            ring_light_rgb = (255, 224, 142) if is_authorization else (120, 225, 255)
            ring_bright_rgb = (255, 194, 70) if is_authorization else (53, 179, 255)
            ring_peak_rgb = (255, 245, 204) if is_authorization else (216, 252, 255)

            painter = QPainter(self)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            painter.setBrush(Qt.BrushStyle.NoBrush)

            # Broad, low-alpha strokes create a soft halo without costly blur.
            for multiplier, opacity, width in (
                (1.03, 0.045, 10.0), (0.95, 0.075, 7.0),
                (0.86, 0.12, 4.0), (0.77, 0.18, 2.0),
            ):
                radius = int(halo_radius * multiplier)
                painter.setPen(QPen(QColor(*halo_rgb, int(alpha * opacity)), width))
                painter.drawEllipse(center - radius, center - radius, radius * 2, radius * 2)

            orbit_speed = (
                (1.0, 1.58, -0.82)
                if self._state is OrbVisualState.PROCESSING
                else (0.24, 0.16, -0.11)
                if self._state is OrbVisualState.SPEAKING
                else (0.18, -0.12, 0.08)
                if self._state is OrbVisualState.LISTENING
                else (0.34, -0.21, 0.15)
            )
            ring_specs = (
                (frame["rotation_deg"] * orbit_speed[0], 0.46, 1.00, 18),
                (frame["rotation_deg"] * orbit_speed[1] + 57.0, 0.62, 0.88, 126),
                (frame["rotation_deg"] * orbit_speed[2] + 119.0, 0.34, 0.76, 247),
            )
            # Dim back segments establish that each orbit continues behind the core.
            for angle, squash, radius_factor, start_angle in ring_specs:
                radius = int(orbit_radius * radius_factor)
                painter.save()
                painter.translate(center, center)
                painter.rotate(angle)
                painter.scale(1.0, squash)
                painter.setPen(QPen(QColor(*ring_dim_rgb, int(alpha * 0.24)), 3.2))
                painter.drawArc(-radius, -radius, radius * 2, radius * 2, start_angle * 16, 112 * 16)
                painter.setPen(QPen(QColor(*ring_light_rgb, int(alpha * 0.13)), 1.4))
                painter.drawArc(-radius, -radius, radius * 2, radius * 2, (start_angle + 158) * 16, 54 * 16)
                painter.restore()

            # Dark shell shifts to amber only while real human approval is pending.
            shell_gradient = QRadialGradient(center, center, core_radius)
            if is_authorization:
                shell_gradient.setColorAt(0.0, QColor(38, 20, 3, min(255, alpha + 18)))
                shell_gradient.setColorAt(0.48, QColor(83, 43, 4, int(alpha * 0.96)))
                shell_gradient.setColorAt(0.77, QColor(184, 103, 15, int(alpha * 0.82)))
            else:
                shell_gradient.setColorAt(0.0, QColor(3, 13, 38, min(255, alpha + 18)))
                shell_gradient.setColorAt(0.48, QColor(5, 31, 78, int(alpha * 0.96)))
                shell_gradient.setColorAt(0.77, QColor(15, 104, 188, int(alpha * 0.82)))
            shell_gradient.setColorAt(1.0, QColor(red, green, blue, 0))
            painter.setBrush(shell_gradient)
            painter.setPen(QPen(QColor(*halo_rgb, int(alpha * 0.52)), 2.2))
            painter.drawEllipse(center - core_radius, center - core_radius, core_radius * 2, core_radius * 2)

            for multiplier, opacity, width in ((0.90, 0.22, 4.0), (0.79, 0.34, 2.5), (0.68, 0.48, 1.5)):
                radius = int(core_radius * multiplier)
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.setPen(QPen(QColor(*ring_bright_rgb, int(alpha * opacity)), width))
                painter.drawEllipse(center - radius, center - radius, radius * 2, radius * 2)

            energy_radius = max(10, int(core_radius * 0.64))
            energy_gradient = QRadialGradient(center - energy_radius * 0.18, center - energy_radius * 0.20, energy_radius)
            energy_gradient.setColorAt(0.0, QColor(172, 248, 255, int(alpha * 0.82)))
            energy_gradient.setColorAt(0.22, QColor(52, 206, 255, int(alpha * 0.68)))
            energy_gradient.setColorAt(0.58, QColor(12, 79, 173, int(alpha * 0.38)))
            energy_gradient.setColorAt(1.0, QColor(7, 27, 74, 0))
            painter.setBrush(energy_gradient)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(center - energy_radius, center - energy_radius, energy_radius * 2, energy_radius * 2)

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
                painter.setBrush(QColor(170, 245, 255, int(alpha * opacity)))
                painter.drawEllipse(
                    int(center + energy_radius * x_factor) - radius,
                    int(center + energy_radius * y_factor) - radius,
                    radius * 2,
                    radius * 2,
                )

            # Bright front segments complete the three-dimensional orbit illusion.
            for angle, squash, radius_factor, start_angle in ring_specs:
                radius = int(orbit_radius * radius_factor)
                painter.save()
                painter.translate(center, center)
                painter.rotate(angle)
                painter.scale(1.0, squash)
                painter.setPen(QPen(QColor(*ring_bright_rgb, int(alpha * 0.80)), 4.2))
                painter.drawArc(-radius, -radius, radius * 2, radius * 2, (start_angle + 224) * 16, 82 * 16)
                painter.setPen(QPen(QColor(*ring_peak_rgb, int(alpha * 0.94)), 2.5))
                painter.drawArc(-radius, -radius, radius * 2, radius * 2, (start_angle + 246) * 16, 30 * 16)
                painter.restore()

            # A compact Atlas emblem stays inside the nucleus and gains a cyan-blue glow.
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(37, 152, 255, int(alpha * 0.38)))
            painter.drawPath(self._emblem_path)
            painter.setBrush(QColor(218, 252, 255, min(255, alpha + 8)))
            painter.drawPath(self._emblem_path)

            # Six-to-ten restrained exterior particles use the existing animation phase only.
            if self._state is not OrbVisualState.DEGRADED:
                for x, y, offset, radius in (
                    (0.18, 0.33, 0.0, 2), (0.82, 0.28, 1.0, 1),
                    (0.76, 0.73, 2.0, 2), (0.24, 0.75, 3.0, 1),
                    (0.13, 0.53, 4.0, 1), (0.87, 0.52, 5.0, 2),
                    (0.34, 0.17, 2.6, 1), (0.66, 0.18, 4.6, 1),
                ):
                    shimmer = 0.36 + 0.64 * ((math.sin(phase + offset) + 1.0) * 0.5)
                    painter.setPen(Qt.PenStyle.NoPen)
                    painter.setBrush(QColor(185, 248, 255, int(alpha * 0.42 * shimmer)))
                    painter.drawEllipse(int(size * x) - radius, int(size * y) - radius, radius * 2, radius * 2)

            # A subtle vertical beam and three compact ellipses ground the hologram below the core.
            base_y = int(size * 0.79)
            beam_top = center + int(core_radius * 0.58)
            painter.setPen(QPen(QColor(47, 183, 255, int(alpha * 0.08)), 18.0))
            painter.drawLine(center, beam_top, center, base_y)
            painter.setPen(QPen(QColor(105, 235, 255, int(alpha * 0.20)), 5.0))
            painter.drawLine(center, beam_top, center, base_y)
            for half_width, half_height, opacity, width in ((15, 3, 0.62, 2.2), (22, 5, 0.34, 1.6), (30, 7, 0.16, 1.1)):
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.setPen(QPen(QColor(92, 224, 255, int(alpha * opacity)), width))
                painter.drawEllipse(center - half_width, base_y - half_height, half_width * 2, half_height * 2)

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
            self.setObjectName("atlasTranscriptPanel")
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
