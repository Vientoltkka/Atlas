"""Atlas MASTER HUD: the approved master visual is the interface.

The 1536x1024 master image is used verbatim as the visual composition.
Interaction is layered on top with transparent hitboxes aligned in design
coordinates (DESIGN_WIDTH/DESIGN_HEIGHT) and transformed by a uniform
``scale = min(w/1536, h/1024)`` so circles stay glued to the artwork at any
resolution. The sphere states (IDLE/LISTENING/THINKING/SPEAKING/...) are
lightweight overlays drawn ABOVE the master, never a redraw of it.
"""
from __future__ import annotations

import math
import time
from pathlib import Path

from use_cases.ui_state_mapper import OrbVisualState

DESIGN_WIDTH = 1536
DESIGN_HEIGHT = 1024
MASTER_ASSET = Path(__file__).resolve().parent / "assets" / "master" / "atlas_master.png"

SPHERE_CENTER = (768.0, 486.0)
SPHERE_RADIUS = 215.0
CIRCLE_HIT_RADIUS = 52.0

CAPABILITY_CIRCLES: dict[str, tuple[float, float]] = {
    # Claves = ids de capacidad EXISTENTES en OrbeController (routing real).
    "chat": (768.0, 155.0),
    "nutricion": (974.0, 232.0),
    "calendario": (1077.0, 372.0),
    "salud": (1081.0, 549.0),
    "agentes": (970.0, 707.0),
    "proyectos": (768.0, 752.0),
    "coding": (563.0, 707.0),
    "finanzas": (461.0, 549.0),
    "control_pc": (456.0, 372.0),
    "entrenamiento": (561.0, 232.0),
}

_HOVER_FADE_MS = 160
_ANIMATION_FPS = 20
_STATE_RING_COLORS = {
    OrbVisualState.IDLE: (56, 185, 255),
    OrbVisualState.STARTING: (105, 225, 255),
    OrbVisualState.LISTENING: (90, 205, 255),
    OrbVisualState.PROCESSING: (60, 150, 255),
    OrbVisualState.SPEAKING: (150, 225, 255),
    OrbVisualState.AUTHORIZATION: (255, 174, 52),
    OrbVisualState.AUTOMATION: (255, 82, 82),
    OrbVisualState.RECOVERING: (60, 175, 235),
    OrbVisualState.DEGRADED: (78, 100, 128),
    OrbVisualState.STOPPING: (80, 145, 180),
}
_ANIMATED_STATES = frozenset(
    {
        OrbVisualState.STARTING,
        OrbVisualState.LISTENING,
        OrbVisualState.PROCESSING,
        OrbVisualState.SPEAKING,
        OrbVisualState.AUTHORIZATION,
        OrbVisualState.AUTOMATION,
        OrbVisualState.RECOVERING,
        OrbVisualState.STOPPING,
    }
)


def master_asset_available() -> bool:
    return MASTER_ASSET.is_file()


def create_master_hud():
    """Build the MASTER HUD when the approved asset exists."""
    if not master_asset_available():
        return None
    return AtlasMasterHud()


def _build_hud_class():
    from PySide6.QtCore import QPointF, QRectF, Qt, QTimer, Signal
    from PySide6.QtGui import (
        QColor,
        QPainter,
        QPainterPath,
        QPen,
        QPixmap,
        QRadialGradient,
    )
    from PySide6.QtWidgets import QWidget

    class AtlasMasterHud(QWidget):
        capability_selected = Signal(str)
        sphere_clicked = Signal()
        hide_requested = Signal()

        def __init__(self) -> None:
            super().__init__()
            self.setWindowTitle("Atlas")
            self.setWindowFlags(
                Qt.WindowType.FramelessWindowHint | Qt.WindowType.Tool
            )
            self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
            self.setMouseTracking(True)
            self._master = QPixmap(str(MASTER_ASSET))
            self._scaled: QPixmap | None = None
            self._scaled_key: tuple[int, int] | None = None
            self._scale = 1.0
            self._offset_x = 0.0
            self._offset_y = 0.0
            self._hovered: str | None = None
            self._hover_strength = 0.0
            self._press_until = 0.0
            self._state = OrbVisualState.IDLE
            self._state_started_at = time.monotonic()
            self._animation_timer = QTimer(self)
            self._animation_timer.setInterval(int(1000 / _ANIMATION_FPS))
            self._animation_timer.timeout.connect(self._on_animation_tick)

        # -- state ------------------------------------------------------

        @property
        def state(self) -> OrbVisualState:
            return self._state

        def apply_state(self, state) -> None:
            new_state = OrbVisualState(state)
            if new_state == self._state:
                return
            self._state = new_state
            self._state_started_at = time.monotonic()
            self._update_animation_timer()
            self.update()

        def _update_animation_timer(self) -> None:
            should_run = (
                self.isVisible() and self._state in _ANIMATED_STATES
            )
            if should_run and not self._animation_timer.isActive():
                self._animation_timer.start()
            elif not should_run and self._animation_timer.isActive():
                self._animation_timer.stop()

        def _on_animation_tick(self) -> None:
            self.update()

        # -- geometry ---------------------------------------------------

        def showEvent(self, event) -> None:  # noqa: N802 (Qt API)
            super().showEvent(event)
            self._fit_to_screen()
            self._update_animation_timer()

        def hideEvent(self, event) -> None:  # noqa: N802 (Qt API)
            super().hideEvent(event)
            self._update_animation_timer()

        def _fit_to_screen(self) -> None:
            screen = self.screen()
            if screen is None:
                return
            bounds = screen.availableGeometry()
            scale = min(
                bounds.width() / DESIGN_WIDTH,
                bounds.height() / DESIGN_HEIGHT,
            )
            width = max(1, int(DESIGN_WIDTH * scale))
            height = max(1, int(DESIGN_HEIGHT * scale))
            self.setGeometry(
                bounds.center().x() - width // 2,
                bounds.center().y() - height // 2,
                width,
                height,
            )

        def resizeEvent(self, event) -> None:  # noqa: N802 (Qt API)
            super().resizeEvent(event)
            self._scale = min(
                self.width() / DESIGN_WIDTH, self.height() / DESIGN_HEIGHT
            )
            self._offset_x = (self.width() - DESIGN_WIDTH * self._scale) / 2
            self._offset_y = (self.height() - DESIGN_HEIGHT * self._scale) / 2
            key = (self.width(), self.height())
            if key != self._scaled_key:
                self._scaled = self._master.scaled(
                    int(DESIGN_WIDTH * self._scale),
                    int(DESIGN_HEIGHT * self._scale),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
                self._scaled_key = key

        def _to_design(self, x: float, y: float) -> tuple[float, float]:
            return (
                (x - self._offset_x) / self._scale,
                (y - self._offset_y) / self._scale,
            )

        def _hit_capability(self, dx: float, dy: float) -> str | None:
            for capability_id, (cx, cy) in CAPABILITY_CIRCLES.items():
                if math.hypot(dx - cx, dy - cy) <= CIRCLE_HIT_RADIUS:
                    return capability_id
            return None

        # -- interaction -------------------------------------------------

        def mouseMoveEvent(self, event) -> None:  # noqa: N802 (Qt API)
            dx, dy = self._to_design(event.position().x(), event.position().y())
            hovered = self._hit_capability(dx, dy)
            inside_sphere = math.hypot(dx - SPHERE_CENTER[0], dy - SPHERE_CENTER[1]) <= SPHERE_RADIUS
            cursor_shape = (
                Qt.CursorShape.PointingHandCursor
                if hovered or inside_sphere
                else Qt.CursorShape.ArrowCursor
            )
            if self.cursor().shape() != cursor_shape:
                self.setCursor(cursor_shape)
            if hovered != self._hovered:
                self._hovered = hovered
                self._hover_started_at = time.monotonic()
                if not self._animation_timer.isActive():
                    self._hover_timer = getattr(self, "_hover_timer", None)
                    if self._hover_timer is None:
                        self._hover_timer = QTimer(self)
                        self._hover_timer.timeout.connect(self._on_hover_tick)
                    self._hover_timer.start(16)

        def _on_hover_tick(self) -> None:
            target = 1.0 if self._hovered is not None else 0.0
            step = 1000.0 / (_HOVER_FADE_MS * _ANIMATION_FPS)
            current = self._hover_strength
            if abs(current - target) <= step:
                self._hover_strength = target
                if self._hover_timer is not None:
                    self._hover_timer.stop()
            else:
                self._hover_strength = current + step if target > current else current - step
            self.update()

        def mousePressEvent(self, event) -> None:  # noqa: N802 (Qt API)
            if event.button() != Qt.MouseButton.LeftButton:
                return
            dx, dy = self._to_design(event.position().x(), event.position().y())
            capability_id = self._hit_capability(dx, dy)
            if capability_id is not None:
                self._press_until = time.monotonic() + 0.22
                self.capability_selected.emit(capability_id)
                return
            if math.hypot(dx - SPHERE_CENTER[0], dy - SPHERE_CENTER[1]) <= SPHERE_RADIUS:
                self._press_until = time.monotonic() + 0.22
                self.sphere_clicked.emit()

        def keyPressEvent(self, event) -> None:  # noqa: N802 (Qt API)
            from PySide6.QtCore import Qt as _Qt

            if event.key() == _Qt.Key.Key_Escape:
                self.hide_requested.emit()
                return
            super().keyPressEvent(event)

        # -- painting ----------------------------------------------------

        def paintEvent(self, event) -> None:  # noqa: N802 (Qt API)
            painter = QPainter(self)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
            if self._scaled is not None:
                painter.drawPixmap(
                    int(self._offset_x), int(self._offset_y), self._scaled
                )
            self._draw_state_overlay(painter)
            if self._hover_strength > 0.005 or time.monotonic() < self._press_until:
                self._draw_hover_glow(painter)
            painter.end()

        def _draw_state_overlay(self, painter: QPainter) -> None:
            state = self._state
            if state is OrbVisualState.IDLE:
                return
            elapsed = time.monotonic() - self._state_started_at
            red, green, blue = _STATE_RING_COLORS.get(state, (90, 205, 255))
            cx, cy = SPHERE_CENTER
            painter.save()
            painter.translate(
                self._offset_x + cx * self._scale,
                self._offset_y + cy * self._scale,
            )
            painter.scale(self._scale, self._scale)
            if state is OrbVisualState.PROCESSING:
                self._draw_thinking_ring(painter, elapsed, red, green, blue)
            elif state is OrbVisualState.SPEAKING:
                self._draw_speaking_halo(painter, elapsed, red, green, blue)
            else:
                self._draw_pulse_ring(painter, elapsed, red, green, blue, state)
            painter.restore()

        def _draw_pulse_ring(self, painter, elapsed, red, green, blue, state) -> None:
            period = {
                OrbVisualState.LISTENING: 1.6,
                OrbVisualState.STARTING: 1.2,
                OrbVisualState.RECOVERING: 0.6,
                OrbVisualState.STOPPING: 1.5,
                OrbVisualState.AUTHORIZATION: 2.4,
                OrbVisualState.AUTOMATION: 1.35,
            }.get(state, 1.6)
            phase = (elapsed % period) / period
            wave = math.sin(2 * math.pi * phase)
            alpha = 70 + int(60 * (wave * 0.5 + 0.5))
            radius = SPHERE_RADIUS * (1.04 + 0.05 * (wave * 0.5 + 0.5))
            pen = QPen(QColor(red, green, blue, alpha), 9)
            pen.setCapStyle(Qt.PenCapStyle.FlatCap)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawEllipse(QPointF(0, 0), radius, radius)

        def _draw_thinking_ring(self, painter, elapsed, red, green, blue) -> None:
            angle = (elapsed * 70.0) % 360.0
            pen = QPen(QColor(red, green, blue, 90), 6)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            radius = SPHERE_RADIUS * 1.06
            rect = QRectF(-radius, -radius, radius * 2, radius * 2)
            painter.drawArc(rect, int(angle * 16), 90 * 16)
            painter.setPen(QPen(QColor(red, green, blue, 45), 5))
            painter.drawArc(rect, int((angle + 180) * 16), 60 * 16)

        def _draw_speaking_halo(self, painter, elapsed, red, green, blue) -> None:
            period = 0.9
            phase = (elapsed % period) / period
            wave = math.sin(2 * math.pi * phase) * 0.5 + 0.5
            gradient = QRadialGradient(QPointF(0, 0), SPHERE_RADIUS * 1.22)
            gradient.setColorAt(0.82, QColor(red, green, blue, 0))
            gradient.setColorAt(0.94, QColor(red, green, blue, int(50 + 70 * wave)))
            gradient.setColorAt(1.0, QColor(red, green, blue, 0))
            painter.setPen(QPen(Qt.PenStyle.NoPen))
            painter.setBrush(gradient)
            radius = SPHERE_RADIUS * 1.22
            painter.drawEllipse(QPointF(0, 0), radius, radius)

        def _draw_hover_glow(self, painter: QPainter) -> None:
            capability_id = self._hovered
            if capability_id is None:
                return
            cx, cy = CAPABILITY_CIRCLES[capability_id]
            strength = self._hover_strength
            if time.monotonic() < self._press_until:
                strength = min(1.35, strength + 0.35)
            painter.save()
            painter.translate(
                self._offset_x + cx * self._scale,
                self._offset_y + cy * self._scale,
            )
            painter.scale(self._scale, self._scale)
            gradient = QRadialGradient(QPointF(0, 0), 92)
            gradient.setColorAt(0.0, QColor(130, 200, 255, int(80 * strength)))
            gradient.setColorAt(0.55, QColor(90, 170, 255, int(46 * strength)))
            gradient.setColorAt(1.0, QColor(80, 160, 255, 0))
            painter.setPen(QPen(Qt.PenStyle.NoPen))
            painter.setBrush(gradient)
            painter.drawEllipse(QPointF(0, 0), 92, 92)
            pen = QPen(QColor(170, 225, 255, int(120 * strength)), 3)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawEllipse(QPointF(0, 0), 46, 46)
            painter.restore()

    return AtlasMasterHud


_hud_class = None


def AtlasMasterHud():  # noqa: N801 (Qt-style factory name)
    global _hud_class
    if _hud_class is None:
        _hud_class = _build_hud_class()
    return _hud_class()
