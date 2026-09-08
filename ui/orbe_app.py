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
from ui import orb_assets


ORB_SIZE = 360
CORE_RADIUS_FACTOR = 0.29
# Hero asset (APPROVED single composition): sphere diameter is ~0.80 of the
# image width and the frame fits inside the window with a small margin.
# Idle window 560 -> sphere ~439 px (target band 380-480 at 1080p).
HERO_ASSET_FIT = 0.98
HERO_SPHERE_FRACTION = 0.80
HERO_ORB_SIZE = 560
# Fraction of the window occupied by the asset-built holo sphere (the device
# body is now a layered composition; the procedural core radius stays for the
# orbit module occlusion test). Sized so the sphere clears the platform base.
DEVICE_RADIUS_FACTOR = 0.335
# Platform base line, lowered to welcome the larger asset sphere.
PLATFORM_BASE_FACTOR = 0.88
# Clickable zone of the orb window: a central disc covering the sphere and its
# orbits. Everything outside this disc is transparent and must pass clicks to
# the desktop (selective hit-testing).
CLICKABLE_RADIUS_FACTOR = 0.50
# Protagonism of the Atlas chevron: the path geometry is scaled around the
# centre by this factor (1.0 = the old compact emblem).
EMBLEM_SCALE = 1.42
# Holographic HUD menu geometry: two lateral fan panels with a transparent
# central gap where the sphere keeps breathing (and stays clickable).
MENU_PANEL_WIDTH = 178
MENU_BUTTON_HEIGHT = 27
MENU_CENTRE_FACTOR = 0.62  # transparent gap width as a fraction of the orb size
MENU_WEDGE_EXTENT = 24     # connector wedge drawn from each panel toward the orb
_MENU_OPEN_MS = 200        # HUD open: fade + lateral slide from the núcleo
_MENU_CLOSE_MS = 150       # HUD close: fade + slide back into the núcleo
_HIT_POLL_MS = 50
_GWL_EXSTYLE = -20
_WS_EX_TRANSPARENT = 0x00000020
_WS_EX_LAYERED = 0x00080000

# Holographic capability menu: every option reuses existing Atlas routing.
# Selecting one opens the chat with a routing prefix; the real router
# (core.operational_request_router / agents) resolves the domain.
_CAPABILITY_OPTIONS: tuple[tuple[str, str], ...] = (
    ("coding", "Coding"),
    ("proyectos", "Proyectos"),
    ("entrenamiento", "Entrenamiento"),
    ("nutricion", "Nutrición"),
    ("salud", "Salud"),
    ("calendario", "Calendario"),
    ("control_pc", "Control PC"),
    ("automatizacion", "Automatización"),
    ("investigacion", "Investigación"),
    ("legal", "Legal"),
    ("finanzas", "Finanzas"),
    ("agentes", "Agentes"),
    ("mas_herramientas", "Más herramientas"),
)
_ACTIVE_ORB_SIZES: dict[OrbVisualState, int] = {
    OrbVisualState.LISTENING: 460,
    OrbVisualState.PROCESSING: 480,
    OrbVisualState.SPEAKING: 490,
    OrbVisualState.AUTHORIZATION: 490,
    OrbVisualState.AUTOMATION: 490,
}

# With the hero asset the sphere already fills the window band (380-480 px),
# so active states only grow subtly; the sphere must never leave that band.
_HERO_ORB_SIZES: dict[OrbVisualState, int] = {
    OrbVisualState.LISTENING: 590,
    OrbVisualState.PROCESSING: 600,
    OrbVisualState.SPEAKING: 600,
    OrbVisualState.AUTHORIZATION: 600,
    OrbVisualState.AUTOMATION: 600,
}

_STATE_COLORS: dict[OrbVisualState, tuple[int, int, int, int]] = {
    OrbVisualState.IDLE: (56, 185, 255, 230),
    OrbVisualState.STARTING: (105, 225, 255, 235),
    OrbVisualState.LISTENING: (80, 225, 255, 245),
    OrbVisualState.PROCESSING: (0, 132, 255, 248),
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
        "core_intensity": 0.85, "core_pulse": 0.54, "particle_intensity": 0.62,
        "segment_activity": 0.45, "base_intensity": 0.60,
        "wave_activity": 0.0, "ping_intensity": 0.0, "node_energy": 0.35, "spark_intensity": 0.0,
    },
    OrbVisualState.LISTENING: {
        "ring_activity": 0.30, "ring_speed": 0.55, "ring_angle": 0.90, "ring_amplitude": 0.95,
        "pulse_strength": 0.55, "halo_intensity": 1.05, "halo_strength": 1.05,
        "core_intensity": 1.00, "core_pulse": 0.70, "particle_intensity": 0.75,
        "segment_activity": 0.60, "base_intensity": 0.75,
        "wave_activity": 0.0, "ping_intensity": 0.0, "node_energy": 0.55, "spark_intensity": 0.45,
    },
    OrbVisualState.PROCESSING: {
        "ring_activity": 1.00, "ring_speed": 1.00, "ring_angle": 1.20, "ring_amplitude": 1.08,
        "pulse_strength": 0.66, "halo_intensity": 1.12, "halo_strength": 1.12,
        "core_intensity": 1.18, "core_pulse": 0.86, "particle_intensity": 0.82,
        "segment_activity": 1.00, "base_intensity": 0.88,
        "wave_activity": 0.0, "ping_intensity": 0.15, "node_energy": 0.85, "spark_intensity": 0.35,
    },
    OrbVisualState.SPEAKING: {
        "ring_activity": 0.30, "ring_speed": 0.64, "ring_angle": 0.98, "ring_amplitude": 1.00,
        "pulse_strength": 1.00, "halo_intensity": 1.28, "halo_strength": 1.28,
        "core_intensity": 1.12, "core_pulse": 1.18, "particle_intensity": 0.86,
        "segment_activity": 0.72, "base_intensity": 1.04,
        "wave_activity": 0.0, "ping_intensity": 0.0, "node_energy": 0.55, "spark_intensity": 0.0,
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


def atlas_emblem_path(size: float, scale: float = EMBLEM_SCALE) -> QPainterPath:
    """Build the compact Atlas chevron plus its detached lower triangle.

    The geometry is scaled around the centre so the emblem keeps clear
    protagonism while staying perfectly centred. Shared with the asset
    generator so the logo layer matches this exact silhouette.
    """
    from PySide6.QtGui import QPainterPath

    def px(fraction: float) -> float:
        return size * (0.5 + (fraction - 0.5) * scale)

    path = QPainterPath()
    # Two diagonal arms form an open chevron; no horizontal A crossbar is used.
    path.moveTo(px(0.5000), px(0.4418))
    path.lineTo(px(0.4269), px(0.5531))
    path.lineTo(px(0.4567), px(0.5625))
    path.lineTo(px(0.5000), px(0.4868))
    path.closeSubpath()
    path.moveTo(px(0.5000), px(0.4418))
    path.lineTo(px(0.5731), px(0.5531))
    path.lineTo(px(0.5433), px(0.5625))
    path.lineTo(px(0.5000), px(0.4868))
    path.closeSubpath()
    # The isolated lower triangle keeps generous negative space inside the core.
    path.moveTo(px(0.5000), px(0.5701))
    path.lineTo(px(0.4745), px(0.6092))
    path.lineTo(px(0.5255), px(0.6092))
    path.closeSubpath()
    return path


def animation_period(state: OrbVisualState) -> float | None:
    """Animation period in seconds, or None when the state is static."""
    return _ANIMATION_PERIODS[state]


def visual_profile(state: OrbVisualState) -> dict[str, float]:
    """Return lightweight per-state render controls for the shared orb."""
    return dict(_VISUAL_PROFILES.get(OrbVisualState(state), _VISUAL_PROFILES[OrbVisualState.IDLE]))


def size_for_state(state: OrbVisualState) -> int:
    """Return the compact idle size or the deliberately larger active size.

    When the approved hero asset is available the window grows so the
    rendered sphere lands in the 380-480 px band; the compact legacy sizes
    remain as the fallback path.
    """
    state = OrbVisualState(state)
    if orb_assets.hero_available():
        return _HERO_ORB_SIZES.get(state, HERO_ORB_SIZE)
    return _ACTIVE_ORB_SIZES.get(state, ORB_SIZE)


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
    from PySide6.QtCore import QPointF, QRect, QRectF, QSize, Qt, Signal, QTimer
    from PySide6.QtGui import (
        QConicalGradient,
        QColor,
        QCursor,
        QIcon,
        QLinearGradient,
        QPainter,
        QPainterPath,
        QPen,
        QPixmap,
        QRegion,
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
        """HUD popup that grows around the sphere instead of beside it.

        Two translucent lateral panels (núcleo / sistemas) flank a fully
        transparent central gap so the orb stays visible and clickable; the
        window mask keeps click-through alive everywhere else.
        """

        chat_requested = Signal()
        voice_requested = Signal()
        quit_requested = Signal()
        capability_selected = Signal(str)

        _LEFT_TITLE = "ATLAS · NÚCLEO"
        _RIGHT_TITLE = "ATLAS · SISTEMAS"

        def __init__(self, parent=None) -> None:
            super().__init__(parent)
            self.setWindowFlags(Qt.WindowType.Popup | Qt.WindowType.FramelessWindowHint)
            self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
            self._voice_active = False
            self._capability_buttons: dict[str, QPushButton] = {}
            self._menu_animating = False
            self._menu_progress = 1.0

            self._left_panel = QWidget(self)
            self._right_panel = QWidget(self)
            self._left_panel.setFixedWidth(MENU_PANEL_WIDTH)
            self._right_panel.setFixedWidth(MENU_PANEL_WIDTH)

            left_layout = QVBoxLayout(self._left_panel)
            left_layout.setContentsMargins(14, 12, 10, 12)
            left_layout.setSpacing(4)
            self._title = QLabel(self._LEFT_TITLE, self)
            self._title.setStyleSheet(
                "color: #bdeeff; font-size: 10px; font-weight: 700; letter-spacing: 1.4px;"
            )
            left_layout.addWidget(self._title)
            self._add_separator(left_layout)
            self._chat_button = self._add_action(left_layout, "Abrir Chat", self.chat_requested, "chat")
            left_layout.addSpacing(2)
            self._capability_title = QLabel("CAPACIDADES", self)
            self._capability_title.setStyleSheet(
                "color: #bdeeff; font-size: 9px; font-weight: 700; letter-spacing: 1.2px;"
            )
            left_layout.addWidget(self._capability_title)
            for capability_id, label in _CAPABILITY_OPTIONS[:6]:
                left_layout.addWidget(self._add_capability_button(label, capability_id))
            left_layout.addStretch(1)
            self._voice_button = self._add_action(
                left_layout, "Modo Voz", self.voice_requested, "voice", voice_status=True
            )

            right_layout = QVBoxLayout(self._right_panel)
            right_layout.setContentsMargins(10, 12, 14, 12)
            right_layout.setSpacing(4)
            self._systems_title = QLabel(self._RIGHT_TITLE, self)
            self._systems_title.setStyleSheet(
                "color: #bdeeff; font-size: 10px; font-weight: 700; letter-spacing: 1.4px;"
            )
            right_layout.addWidget(self._systems_title)
            self._add_separator(right_layout)
            for capability_id, label in _CAPABILITY_OPTIONS[6:]:
                right_layout.addWidget(self._add_capability_button(label, capability_id))
            right_layout.addStretch(1)
            self._quit_button = self._add_action(right_layout, "Salir", self.quit_requested, "quit", danger=True)

            # Manual HUD layout: both modules slide out from the núcleo.
            self._left_panel.adjustSize()
            self._right_panel.adjustSize()
            panel_height = max(self._left_panel.height(), self._right_panel.height(), 120)
            self._panel_height = panel_height
            self._left_panel.setFixedHeight(panel_height)
            self._right_panel.setFixedHeight(panel_height)
            self.setFixedSize(self._menu_window_width(int(HERO_ORB_SIZE * MENU_CENTRE_FACTOR)), panel_height)
            self._layout_panels(1.0)

        def _menu_window_width(self, gap: int) -> int:
            return MENU_PANEL_WIDTH * 2 + max(24, gap)

        def _layout_panels(self, progress: float) -> None:
            """Place both panels: progress 0 = collapsed at the núcleo, 1 = open."""
            progress = max(0.0, min(1.0, progress))
            self._menu_progress = progress
            mid = self.width() / 2.0
            left_start = mid - MENU_PANEL_WIDTH
            left_x = left_start * (1.0 - progress)
            right_final = self.width() - MENU_PANEL_WIDTH
            right_x = mid + (right_final - mid) * progress
            self._left_panel.move(int(round(left_x)), 0)
            self._right_panel.move(int(round(right_x)), 0)
            self._apply_hud_mask()

        def _set_menu_progress(self, progress: float) -> None:
            progress = max(0.0, min(1.0, progress))
            self._layout_panels(progress)
            self.setWindowOpacity(progress)

        def _run_menu_animation(self, *, opening: bool) -> None:
            """Short 150-250 ms fade + lateral slide driven synchronously.

            The loop processes Qt events so the animation really paints, while
            staying transparent to callers/tests (the widget ends hidden when
            closing, exactly as the previous instant ``hide()`` did).
            """
            if self._menu_animating:
                return
            self._menu_animating = True
            try:
                from PySide6.QtCore import QElapsedTimer
                from PySide6.QtWidgets import QApplication

                duration = _MENU_OPEN_MS if opening else _MENU_CLOSE_MS
                app = QApplication.instance()
                clock = QElapsedTimer()
                clock.start()
                while True:
                    t = clock.elapsed() / duration
                    if t >= 1.0:
                        break
                    eased = 1.0 - (1.0 - t) * (1.0 - t) if opening else (1.0 - t) * (1.0 - t)
                    self._set_menu_progress(eased)
                    if app is not None:
                        app.processEvents()
                if opening:
                    self._set_menu_progress(1.0)
                else:
                    self._set_menu_progress(0.0)
                    self.hide()
            finally:
                self._menu_animating = False

        def close_animated(self) -> None:
            """Fade the HUD back into the núcleo and hide it."""
            if not self.isVisible():
                return
            self._run_menu_animation(opening=False)

        def hide(self) -> None:  # noqa: N802 (Qt API)
            super().hide()
            self._menu_animating = False
            self.setWindowOpacity(1.0)
            self._menu_progress = 1.0

        def _add_capability_button(self, text: str, capability_id: str) -> QPushButton:
            button = QPushButton(text, self)
            button.setFixedHeight(MENU_BUTTON_HEIGHT)
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            button.setStyleSheet(
                "QPushButton {"
                "color: #cdeeff; background: rgba(8, 30, 60, 46);"
                "border: 1px solid rgba(96, 202, 255, 64); border-left: 2px solid rgba(110, 220, 255, 150);"
                "border-radius: 4px; padding: 4px 8px; text-align: left; font-size: 11px;"
                "}"
                "QPushButton:hover { background: rgba(52, 173, 255, 46); border-color: rgba(130, 225, 255, 180);"
                "border-left: 2px solid rgba(190, 245, 255, 230); }"
            )
            button.clicked.connect(lambda checked=False, cid=capability_id: self._select_capability(cid))
            self._capability_buttons[capability_id] = button
            return button

        def _select_capability(self, capability_id: str) -> None:
            self.close_animated()
            self.capability_selected.emit(capability_id)

        def _add_separator(self, layout) -> None:
            separator = QFrame(self)
            separator.setFrameShape(QFrame.Shape.HLine)
            separator.setStyleSheet("color: rgba(85, 185, 255, 70);")
            layout.addWidget(separator)

        def _add_action(self, layout, text: str, signal, icon_name: str, *, danger: bool = False, voice_status: bool = False):
            button = QPushButton(text, self)
            button.setFixedHeight(MENU_BUTTON_HEIGHT)
            color = "#ffb6b6" if danger else "#e7f8ff"
            accent = "rgba(255, 96, 96, 200)" if danger else "rgba(110, 220, 255, 150)"
            hover = "rgba(255, 78, 78, 40)" if danger else "rgba(52, 173, 255, 42)"
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            button.setIcon(QIcon(self._action_icon(icon_name, "#ff9c9c" if danger else "#84d8ff")))
            button.setIconSize(QSize(18, 18))
            button.setStyleSheet(
                "QPushButton {"
                f"color: {color}; background: rgba(8, 30, 60, 40);"
                "border: 1px solid rgba(96, 202, 255, 52); border-left: 2px solid " + accent + ";"
                "border-radius: 4px; padding: 4px 8px; text-align: left; font-size: 11px;"
                "}"
                f"QPushButton:hover {{ background: {hover}; border-color: rgba(130, 225, 255, 160); }}"
            )
            button.clicked.connect(signal.emit)
            button.clicked.connect(self.close_animated)
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
            """Grow the HUD panels around the orb centre and clamp to screen."""
            self.setFixedSize(
                self._menu_window_width(int(MENU_CENTRE_FACTOR * orb.width())),
                self._panel_height,
            )
            screen = orb.screen()
            if screen is None:
                self.show()
                return
            bounds = screen.availableGeometry()
            centre = orb.frameGeometry().center()
            x = centre.x() - self.width() // 2
            y = centre.y() - self.height() // 2
            x = max(bounds.left(), min(x, bounds.right() - self.width() + 1))
            y = max(bounds.top(), min(y, bounds.bottom() - self.height() + 1))
            self.move(x, y)
            self._set_menu_progress(0.0)
            self.show()
            self.raise_()
            self._run_menu_animation(opening=True)

        def _apply_hud_mask(self) -> None:
            """Only the two panels (plus wedges) receive clicks; the gap does not."""
            w, h = self.width(), self.height()
            left_x = int(self._left_panel.x())
            right_x = int(self._right_panel.x()) - MENU_WEDGE_EXTENT
            left = QRect(left_x, 0, MENU_PANEL_WIDTH + MENU_WEDGE_EXTENT, h)
            right = QRect(right_x, 0, MENU_PANEL_WIDTH + MENU_WEDGE_EXTENT, h)
            self.setMask(QRegion(left) | QRegion(right))

        def paintEvent(self, event) -> None:  # noqa: N802 (Qt API)
            painter = QPainter(self)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            for panel, side in ((self._left_panel, "left"), (self._right_panel, "right")):
                if not panel.isVisible():
                    continue
                self._draw_hud_panel(painter, panel.geometry(), side)
            painter.end()

        def _draw_hud_panel(self, painter, rect, side: str) -> None:
            """One translucent fan panel with cian HUD border and corner ticks."""
            inner = rect.adjusted(5, 5, -5, -5)
            fill = QLinearGradient(inner.topLeft(), inner.bottomLeft())
            fill.setColorAt(0.0, QColor(5, 24, 48, 52))
            fill.setColorAt(0.55, QColor(3, 14, 32, 40))
            fill.setColorAt(1.0, QColor(2, 9, 22, 30))
            path = QPainterPath()
            path.addRoundedRect(inner, 8, 8)
            # Inner edge angles toward the sphere: the HUD is born from the orb.
            wedge = QPainterPath()
            wedge_extent = MENU_WEDGE_EXTENT - 6
            if side == "left":
                wedge.moveTo(inner.right() - 1, inner.top() + inner.height() * 0.16)
                wedge.lineTo(inner.right() + wedge_extent, inner.center().y())
                wedge.lineTo(inner.right() - 1, inner.top() + inner.height() * 0.84)
            else:
                wedge.moveTo(inner.left() + 1, inner.top() + inner.height() * 0.16)
                wedge.lineTo(inner.left() - wedge_extent, inner.center().y())
                wedge.lineTo(inner.left() + 1, inner.top() + inner.height() * 0.84)
            wedge.closeSubpath()
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(fill)
            painter.drawPath(path)
            painter.setBrush(QColor(4, 18, 40, 36))
            painter.drawPath(wedge)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            # Soft outer glow, then the crisp cian border of the HUD module.
            painter.setPen(QPen(QColor(96, 202, 255, 34), 5.0))
            painter.drawPath(path)
            painter.setPen(QPen(QColor(96, 202, 255, 150), 1.3))
            painter.drawPath(path)
            painter.setPen(QPen(QColor(120, 220, 255, 100), 1.0))
            painter.drawPath(wedge)
            # Corner ticks: small tech brackets on the outer corners.
            tick = 9
            painter.setPen(QPen(QColor(190, 245, 255, 200), 2.0))
            if side == "left":
                painter.drawLine(inner.topLeft() + QPointF(0, tick), inner.topLeft() + QPointF(0, 0) + QPointF(tick, 0))
                painter.drawLine(inner.bottomLeft() + QPointF(0, -tick), inner.bottomLeft() + QPointF(tick, 0))
            else:
                painter.drawLine(inner.topRight() + QPointF(0, tick), inner.topRight() + QPointF(-tick, 0))
                painter.drawLine(inner.bottomRight() + QPointF(0, -tick), inner.bottomRight() + QPointF(-tick, 0))

    class OrbWindow(QWidget):
        """Frameless translucent always-on-top circular state indicator."""

        stop_requested = Signal()
        quit_requested = Signal()
        chat_requested = Signal()
        voice_requested = Signal()

        def __init__(self) -> None:
            super().__init__()
            self.setWindowTitle("Atlas")
            initial_size = self._bounded_size(size_for_state(OrbVisualState.IDLE))
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

            # Selective hit-testing: while the cursor stays outside the
            # clickable disc the whole window forwards clicks to the desktop
            # (WS_EX_TRANSPARENT); inside the disc the window receives them.
            self._hit_timer = QTimer(self)
            self._hit_timer.setInterval(_HIT_POLL_MS)
            self._hit_timer.timeout.connect(self._update_click_through)

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
            menu = self._context_menu
            if menu._menu_animating:
                return
            if menu.isVisible():
                menu.close_animated()
            else:
                menu.set_voice_active(self._voice_active)
                menu.show_beside(self)

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

        def showEvent(self, event) -> None:  # noqa: N802 (Qt API)
            super().showEvent(event)
            if self._hit_timer is not None and not self._hit_timer.isActive():
                self._hit_timer.start()

        def hideEvent(self, event) -> None:  # noqa: N802 (Qt API)
            if self._hit_timer is not None:
                self._hit_timer.stop()
            super().hideEvent(event)

        # -- selective hit-testing --------------------------------------

        def clickable_at(self, global_point) -> bool:
            """True when a global point falls inside the clickable orb disc."""
            centre = self.frameGeometry().center()
            dx = global_point.x() - centre.x()
            dy = global_point.y() - centre.y()
            return math.hypot(dx, dy) <= self.width() * CLICKABLE_RADIUS_FACTOR

        def _update_click_through(self) -> None:
            """Toggle WS_EX_TRANSPARENT so only the orb disc catches clicks."""
            if not self.isVisible() or sys.platform != "win32":
                return
            try:
                from PySide6.QtGui import QGuiApplication

                if QGuiApplication.platformName() == "offscreen":
                    return  # tests/headless: never touch the real window styles
                import ctypes
                from ctypes import wintypes

                user32 = ctypes.windll.user32
                get_style = getattr(user32, "GetWindowLongPtrW", None) or user32.GetWindowLongW
                get_style.argtypes = [wintypes.HWND, ctypes.c_int]
                get_style.restype = ctypes.c_ssize_t
                set_style = getattr(user32, "SetWindowLongPtrW", None) or user32.SetWindowLongW
                set_style.argtypes = [wintypes.HWND, ctypes.c_int, ctypes.c_ssize_t]
                set_style.restype = ctypes.c_ssize_t
                hwnd = wintypes.HWND(int(self.winId()))
                if not hwnd:
                    return
                current = get_style(hwnd, _GWL_EXSTYLE)
                if current in (0, None):
                    return
                if self.clickable_at(QCursor.pos()):
                    updated = current & ~_WS_EX_TRANSPARENT
                else:
                    updated = current | _WS_EX_TRANSPARENT | _WS_EX_LAYERED
                if updated != current:
                    set_style(hwnd, _GWL_EXSTYLE, updated)
            except Exception:
                # A failed hit-test toggle degrades to Qt default behavior
                # (whole window clickable); the orb must never crash over it.
                pass

        def closeEvent(self, event) -> None:  # noqa: N802 (Qt API)
            self.save_position()
            super().closeEvent(event)

        # -- painting ---------------------------------------------------

        def _build_atlas_emblem(self) -> QPainterPath:
            """Compact Atlas chevron; the shared geometry lives at module level."""
            return atlas_emblem_path(self.width())

        # -- painting ---------------------------------------------------

        _ORBIT_SPECS = (
            # (tilt_deg, squash, radius_factor, speed, front segments (start_deg, span_deg))
            # Only three orbital rings now: the sphere is the protagonist and each
            # ring keeps a different inclination; the first passes in front of the
            # sphere, the other two read behind it through the back segments.
            (-26.0, 0.42, 0.404, 1.00, ((196.0, 70.0), (300.0, 44.0))),
            (38.0, 0.30, 0.436, -0.55, ((186.0, 86.0),)),
            (-55.0, 0.54, 0.386, 0.72, ((248.0, 66.0), (322.0, 26.0))),
        )
        _ORBIT_BACK_SEGMENTS = ((22.0, 66.0), (112.0, 58.0))
        _ORBIT_MODULES = ((226.0,), (306.0,), (204.0,))
        _STATE_ORBIT_SPEEDS = {
            OrbVisualState.AUTOMATION: (1.34, 2.10, -1.15, 1.72),
            OrbVisualState.PROCESSING: (1.18, 1.82, -0.95, 1.42),
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
            bundle = orb_assets.state_bundle(self._state)
            hero = bundle.get("hero")
            # Hero mode: the sphere is the asset itself; activity overlays
            # (waves, pings, sparks) hug the rendered sphere edge instead of
            # the legacy procedural core.
            core_radius = max(20.0, size * CORE_RADIUS_FACTOR * core_scale)
            if hero is not None:
                core_radius = size * HERO_ASSET_FIT * HERO_SPHERE_FRACTION / 2.0
            palette = {
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
            painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            # Order: ambient -> orbit backs -> beam/base -> asset device ->
            # asset particles -> orbit fronts -> activity. When any asset is
            # missing the procedural device renders instead (never crash).
            # With the approved hero asset present it IS the whole visual:
            # sphere, logo, glow, orbits, particles and platform in one
            # composition; the provisional layers stay as fallback only.
            if hero is not None:
                self._draw_hero_halo(painter, c)
                self._draw_hero_device(painter, c, hero)
                self._draw_hero_overlays(painter, c)
                self._draw_state_activity(painter, c)
                painter.end()
                return
            self._draw_ambient(painter, c)
            self._draw_orbits(painter, c, front=False)
            self._draw_beam_and_base(painter, c)
            if orb_assets.missing_assets():
                self._draw_device(painter, c)
                self._draw_core_particles(painter, c)
                self._draw_emblem(painter, c)
            else:
                self._draw_asset_device(painter, c, bundle)
                self._draw_asset_particles(painter, c, bundle)
            self._draw_orbits(painter, c, front=True)
            self._draw_state_activity(painter, c)
            self._draw_starfield(painter, c)
            painter.end()

        # -- asset composition -------------------------------------------

        def _draw_hero_device(self, painter, c, hero) -> None:
            """Render the approved asset as the single orb composition.

            No rotation is applied: the image already contains the orbits,
            particles and platform, and tilting the whole frame would make
            the pedestal wobble. Only the shared breathing pulse and the
            state alpha factor modulate it.
            """
            size = c["size"]
            frame = c["frame"]
            pulse = 1.0 + (frame["scale"] - 1.0) * c["profile"]["core_pulse"]
            painter.save()
            painter.translate(c["center"], c["center"])
            painter.scale(pulse, pulse)
            painter.setOpacity(min(1.0, frame["alpha_factor"]))
            width = size * HERO_ASSET_FIT
            height = width * hero.height() / hero.width()
            if height > size * HERO_ASSET_FIT:
                height = size * HERO_ASSET_FIT
                width = height * hero.width() / hero.height()
            painter.drawPixmap(
                QRectF(-width / 2.0, -height / 2.0, width, height),
                hero,
                QRectF(0.0, 0.0, hero.width(), hero.height()),
            )
            painter.restore()

        # -- hero holographic overlays (approved asset stays untouched) ------

        # (phase0_deg, squash, tilt_deg, speed_multiplier) for 3 orbital nodes.
        _HERO_NODES = ((10.0, 0.46, -24.0, 1.00), (62.0, 0.32, 36.0, -0.74), (161.0, 0.56, 10.0, 0.55))
        # (phase0_rad, radius_factor, speed_rad_s, dot_px, blink_speed, blink_phase).
        _HERO_PARTICLES = (
            (0.00, 1.06, 0.42, 1.5, 1.3, 0.0), (0.55, 1.13, -0.31, 1.1, 1.7, 1.1),
            (1.05, 1.09, 0.27, 1.4, 1.1, 2.4), (1.60, 1.17, -0.38, 1.2, 1.5, 3.2),
            (2.10, 1.05, 0.33, 1.0, 1.9, 0.7), (2.60, 1.21, -0.24, 1.6, 1.2, 1.9),
            (3.15, 1.10, 0.36, 1.1, 1.6, 2.9), (3.60, 1.16, -0.29, 1.3, 1.4, 4.1),
            (4.10, 1.07, 0.30, 1.0, 1.8, 0.4), (4.65, 1.19, -0.34, 1.5, 1.2, 2.2),
            (5.15, 1.12, 0.26, 1.2, 1.5, 3.6), (5.70, 1.08, -0.27, 1.1, 1.7, 1.6),
        )
        # Orbital speed gain per state (deg/s base factor); DEGRADED stays static.
        _HERO_NODE_SPEEDS = {
            OrbVisualState.IDLE: 0.35, OrbVisualState.STARTING: 0.70,
            OrbVisualState.LISTENING: 0.80, OrbVisualState.PROCESSING: 1.80,
            OrbVisualState.SPEAKING: 0.90, OrbVisualState.AUTHORIZATION: 0.50,
            OrbVisualState.AUTOMATION: 2.20, OrbVisualState.RECOVERING: 1.10,
            OrbVisualState.DEGRADED: 0.0, OrbVisualState.STOPPING: 0.45,
        }
        # Seconds between energy sweeps per state (shorter = more frequent).
        _HERO_SWEEP_PERIODS = {
            OrbVisualState.IDLE: 9.0, OrbVisualState.STARTING: 5.0,
            OrbVisualState.LISTENING: 6.0, OrbVisualState.PROCESSING: 3.0,
            OrbVisualState.SPEAKING: 5.0, OrbVisualState.AUTHORIZATION: 7.0,
            OrbVisualState.AUTOMATION: 2.4, OrbVisualState.RECOVERING: 4.0,
            OrbVisualState.DEGRADED: 12.0, OrbVisualState.STOPPING: 8.0,
        }

        def _hero_sphere_radius(self, size: int) -> float:
            return size * HERO_ASSET_FIT * HERO_SPHERE_FRACTION / 2.0

        def _draw_hero_halo(self, painter, c) -> None:
            """Pulsating outer glow behind the approved sphere (never whitens it)."""
            profile, period = c["profile"], c["period"] or 3.0
            breath = 0.5 + 0.5 * math.sin(c["elapsed"] * math.tau / max(0.4, period))
            intensity = profile["halo_strength"] * (0.55 + 0.45 * breath)
            halo = c["halo"]
            r = self._hero_sphere_radius(c["size"])
            gradient = QRadialGradient(0.0, 0.0, r * 1.24)
            gradient.setColorAt(0.70, QColor(*halo, 0))
            gradient.setColorAt(0.80, QColor(*halo, int(26 * intensity)))
            gradient.setColorAt(0.86, QColor(*halo, int(58 * intensity)))
            gradient.setColorAt(0.95, QColor(*halo, int(12 * intensity)))
            gradient.setColorAt(1.0, QColor(*halo, 0))
            painter.save()
            painter.translate(c["center"], c["center"])
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(gradient)
            painter.drawEllipse(self._arc_rect(r * 1.24))
            painter.restore()

        def _draw_hero_overlays(self, painter, c) -> None:
            """Living holographic layers over the approved composition.

            Cierre provisional: solo overlays mínimos que no duplican el
            asset. Los nodos orbitales y partículas procedurales quedan
            desactivados (el asset aprobado ya trae órbitas y partículas).
            """
            painter.save()
            painter.translate(c["center"], c["center"])
            r = self._hero_sphere_radius(c["size"])
            self._draw_energy_sweep(painter, c, r)
            self._draw_hero_sparks(painter, c, r)
            painter.restore()

        def _draw_hero_nodes(self, painter, c, r: float) -> None:
            """2-3 luminous nodes on elliptical tracks around the sphere."""
            state_gain = self._HERO_NODE_SPEEDS.get(c["state"], 0.5)
            if state_gain <= 0.0:
                return
            alpha, peak, light = c["alpha"], c["peak"], c["light"]
            energy = c["profile"]["node_energy"]
            painter.setPen(Qt.PenStyle.NoPen)
            for phase0, squash, tilt, speed in self._HERO_NODES:
                angle_deg = phase0 + c["elapsed"] * 57.6 * state_gain * speed
                x, y = self._orbit_point(r * 1.10, squash, tilt, angle_deg)
                flicker = 0.70 + 0.30 * math.sin(c["elapsed"] * 6.0 + phase0 * 4.0)
                a = alpha * (0.45 + 0.40 * energy) * flicker
                painter.setBrush(QColor(*peak, int(a * 0.30)))
                painter.drawEllipse(QPointF(x, y), 5.2, 5.2)
                painter.setBrush(QColor(*light, int(a)))
                painter.drawEllipse(QPointF(x, y), 2.3, 2.3)
                painter.setBrush(QColor(255, 255, 255, int(a * 0.50)))
                painter.drawEllipse(QPointF(x - 0.6, y - 0.6), 0.9, 0.9)

        def _draw_hero_particles(self, painter, c, r: float) -> None:
            """Sparse data motes fading in/out along the sphere perimeter."""
            state_gain = self._HERO_NODE_SPEEDS.get(c["state"], 0.5)
            alpha, halo = c["alpha"], c["halo"]
            intensity = c["profile"]["particle_intensity"]
            painter.setPen(Qt.PenStyle.NoPen)
            for phase0, rf, speed, dot, blink, bphase in self._HERO_PARTICLES:
                angle = phase0 + c["elapsed"] * speed * (0.45 + 0.90 * state_gain)
                fade = 0.5 + 0.5 * math.sin(c["elapsed"] * blink + bphase)
                fade *= fade
                a = alpha * 0.60 * intensity * (0.25 + 0.75 * fade)
                x = math.cos(angle) * r * rf
                y = math.sin(angle) * r * rf * 0.94
                painter.setBrush(QColor(*halo, int(a)))
                painter.drawEllipse(QPointF(x, y), dot, dot)
                painter.setBrush(QColor(255, 255, 255, int(a * 0.45)))
                painter.drawEllipse(QPointF(x - dot * 0.3, y - dot * 0.3), dot * 0.45, dot * 0.45)

        def _draw_energy_sweep(self, painter, c, r: float) -> None:
            """Thin light reflection crossing the sphere occasionally."""
            period = self._HERO_SWEEP_PERIODS.get(c["state"], 8.0)
            if period <= 0:
                return
            window = 0.32
            cycle = (c["elapsed"] % period) / period
            if cycle >= window:
                return
            t = cycle / window
            env = math.sin(math.pi * t)
            direction = -1.0 if c["state"] is OrbVisualState.AUTOMATION else 1.0
            angle = t * 360.0 * direction
            alpha = c["alpha"] * 0.14 * env * (1.0 + c["profile"]["spark_intensity"])
            peak = c["peak"]
            band = 0.07
            soft = band * 0.35
            gradient = QConicalGradient(-r * 0.06, -r * 0.06, angle)
            gradient.setColorAt(0.0, QColor(*peak, int(alpha * 0.90)))
            gradient.setColorAt(soft, QColor(*peak, int(alpha * 0.35)))
            gradient.setColorAt(band, QColor(*peak, 0))
            gradient.setColorAt(1.0 - band, QColor(*peak, 0))
            gradient.setColorAt(1.0 - soft, QColor(*peak, int(alpha * 0.35)))
            gradient.setColorAt(1.0, QColor(*peak, int(alpha * 0.90)))
            painter.save()
            clip = QPainterPath()
            clip.addEllipse(self._arc_rect(r * 0.985))
            painter.setClipPath(clip)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(gradient)
            painter.drawEllipse(self._arc_rect(r * 0.985))
            painter.restore()

        def _draw_hero_sparks(self, painter, c, r: float) -> None:
            """Internal flashes: thinking/listening sparkle inside the sphere."""
            spark = c["profile"]["spark_intensity"]
            if spark <= 0.03:
                return
            alpha, peak = c["alpha"], c["peak"]
            painter.save()
            clip = QPainterPath()
            clip.addEllipse(self._arc_rect(r * 0.96))
            painter.setClipPath(clip)
            painter.setPen(Qt.PenStyle.NoPen)
            for index in range(5):
                angle = index * 1.256 + c["elapsed"] * (0.5 + 0.3 * index)
                radius = r * (0.32 + 0.14 * (index % 3))
                x = math.cos(angle) * radius
                y = math.sin(angle) * radius * 0.90
                twinkle = max(0.0, math.sin(c["elapsed"] * 5.2 + index * 2.1))
                a = alpha * 0.55 * spark * twinkle
                painter.setBrush(QColor(*peak, int(a)))
                painter.drawEllipse(QPointF(x, y), 1.4, 1.4)
                painter.setBrush(QColor(255, 255, 255, int(a * 0.60)))
                painter.drawEllipse(QPointF(x, y), 0.6, 0.6)
            painter.restore()

        def _draw_asset_layer(self, painter, pixmap, diameter: float) -> None:
            """Centered square layer of the given diameter (SmoothPixmap)."""
            if pixmap is None:
                return
            half = diameter / 2.0
            painter.drawPixmap(
                QRectF(-half, -half, diameter, diameter),
                pixmap,
                QRectF(0.0, 0.0, pixmap.width(), pixmap.height()),
            )

        def _draw_asset_device(self, painter, c, bundle) -> None:
            """Layered asset sphere: body -> rotating grid -> glow -> logo."""
            size = c["size"]
            frame, profile, alpha = c["frame"], c["profile"], c["alpha"]
            device_r = size * DEVICE_RADIUS_FACTOR
            pulse = 1.0 + (frame["scale"] - 1.0) * profile["core_pulse"]
            painter.save()
            painter.translate(c["center"], c["center"])
            painter.scale(pulse, pulse)
            # 1. Core / orb base.
            self._draw_asset_layer(painter, bundle.get("globe_body"), device_r * 2.0)
            # 2. Digital globe texture rotating slowly (state-aware speed).
            painter.save()
            painter.rotate(frame["rotation_deg"] * 0.55 * (0.4 + profile["ring_speed"]))
            painter.setOpacity(min(1.0, 0.90 * frame["alpha_factor"]))
            self._draw_asset_layer(painter, bundle.get("globe_grid"), device_r * 2.0)
            painter.restore()
            # 3. Inner glow breathing with the core energy.
            painter.setOpacity(min(1.0, (0.34 + 0.42 * profile["core_intensity"]) * frame["alpha_factor"]))
            glow_diameter = device_r * (1.10 + 0.06 * profile["core_pulse"])
            self._draw_asset_layer(painter, bundle.get("inner_glow"), glow_diameter)
            painter.setOpacity(min(1.0, frame["alpha_factor"]))
            # 4. Atlas logo: big, integrated, gently pulsing.
            painter.save()
            logo_scale = 1.0 + 0.05 * profile["core_pulse"] * (frame["scale"] - 1.0) * 12.0
            painter.scale(logo_scale, logo_scale)
            self._draw_asset_layer(painter, bundle.get("atlas_logo"), device_r * 1.34)
            painter.restore()
            painter.restore()

        def _draw_asset_particles(self, painter, c, bundle) -> None:
            """Asset particle field drifting/rotating around the device."""
            pixmap = bundle.get("particles")
            if pixmap is None:
                return self._draw_core_particles(painter, c)
            profile = c["profile"]
            painter.save()
            painter.translate(c["center"], c["center"])
            painter.rotate(c["frame"]["rotation_deg"] * 0.32 * (0.5 + profile["particle_intensity"]))
            painter.setOpacity(min(1.0, (0.42 + 0.58 * profile["particle_intensity"]) * c["frame"]["alpha_factor"]))
            self._draw_asset_layer(painter, pixmap, c["size"] * DEVICE_RADIUS_FACTOR * 2.5)
            painter.restore()

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
            device_r = size * DEVICE_RADIUS_FACTOR
            speeds = self._STATE_ORBIT_SPEEDS.get(c["state"], (0.16, -0.10, 0.07, 0.12))
            gain = profile["ring_activity"] * profile["ring_speed"]
            widths = (5.6, 4.4, 3.4)
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
                        painter.setPen(QPen(QColor(*c["depth"], int(alpha * 0.60)), width + 2.6))
                        painter.drawArc(orbit_rect, int(start * 16), int(span * 16))
                        painter.setPen(QPen(QColor(*halo, int(alpha * 0.12)), width * 2.4))
                        painter.drawArc(orbit_rect, int(start * 16), int(span * 16))
                        painter.setPen(QPen(QColor(*light, int(alpha * 0.82)), width))
                        painter.drawArc(orbit_rect, int(start * 16), int(span * 16))
                        # Dark zone: one stretch of the tube stays in shadow.
                        painter.setPen(QPen(QColor(*c["depth"], int(alpha * 0.62)), width * 0.9))
                        painter.drawArc(orbit_rect, int((start + span * 0.48) * 16), int(span * 0.24 * 16))
                        painter.setPen(QPen(QColor(*bright, int(alpha)), max(1.2, width * 0.50)))
                        painter.drawArc(orbit_rect, int((start + 2) * 16), int((span - 4) * 16))
                        painter.setPen(QPen(QColor(255, 255, 255, int(alpha * 0.60)), max(1.0, width * 0.35)))
                        painter.drawArc(orbit_rect, int(start * 16), int(min(9.0, span * 0.35) * 16))
                else:
                    for start, span in self._ORBIT_BACK_SEGMENTS:
                        # The far half is the shadowed underside of the same tube,
                        # not a thin radar line crossing the composition.
                        painter.setPen(QPen(QColor(*c["depth"], int(alpha * 0.72)), 4.6))
                        painter.drawArc(orbit_rect, int(start * 16), int(span * 16))
                        painter.setPen(QPen(QColor(*dim, int(alpha * 0.34)), 1.6))
                        painter.drawArc(orbit_rect, int((start + span * 0.12) * 16), int(span * 0.5 * 16))
                        painter.setPen(QPen(QColor(*halo, int(alpha * 0.22)), 1.1))
                        painter.drawArc(orbit_rect, int((start + span * 0.62) * 16), int(span * 0.22 * 16))
                painter.restore()
                if not front:
                    continue
                # Small technology modules ride the front arc of each orbit.
                for module_angle in self._ORBIT_MODULES[index]:
                    mx, my = self._orbit_point(radius, squash, tilt + rotation, module_angle)
                    if my < -device_r * 0.15 or math.hypot(mx, my) < device_r * 1.12:
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
            device_r = size * DEVICE_RADIUS_FACTOR
            base_y = size * PLATFORM_BASE_FACTOR
            beam_top = center + device_r * 0.96
            half_top = max(6.0, device_r * 0.30)
            half_bottom = max(12.0, device_r * 0.74)
            beam = QPainterPath()
            beam.moveTo(center - half_top, beam_top)
            beam.lineTo(center + half_top, beam_top)
            beam.lineTo(center + half_bottom, base_y)
            beam.lineTo(center - half_bottom, base_y)
            beam.closeSubpath()
            projection_gradient = QLinearGradient(center, beam_top, center, base_y)
            projection_gradient.setColorAt(0.0, QColor(*projection, int(alpha * 0.10)))
            projection_gradient.setColorAt(0.45, QColor(*projection, int(alpha * 0.24)))
            projection_gradient.setColorAt(0.88, QColor(*projection, int(alpha * 0.38)))
            projection_gradient.setColorAt(1.0, QColor(*projection, int(alpha * 0.30)))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(projection_gradient)
            painter.drawPath(beam)
            painter.setPen(QPen(QColor(*peak, int(alpha * 0.36)), 2.2))
            painter.drawLine(QPointF(center, beam_top), QPointF(center, base_y))

            # Reflection pool below the platform.
            painter.save()
            painter.translate(center, base_y + size * 0.02)
            painter.scale(1.0, 0.20)
            pool = QRadialGradient(0, 0, size * 0.24)
            pool.setColorAt(0.0, QColor(*projection, int(alpha * 0.34)))
            pool.setColorAt(0.55, QColor(*projection, int(alpha * 0.13)))
            pool.setColorAt(1.0, QColor(*projection, 0))
            painter.setBrush(pool)
            painter.drawEllipse(self._arc_rect(size * 0.24))
            painter.restore()

            # Layered platform sized to the asset sphere, not to the core:
            # segmented outer ring, sparse outer segments, mid rings, luminous centre.
            painter.save()
            painter.translate(center, base_y)
            painter.scale(1.0, 0.30)
            outer = device_r * 0.94
            sparse_pen = QPen(QColor(*projection, int(alpha * 0.50)), 2.0, Qt.PenStyle.CustomDashLine)
            sparse_pen.setDashPattern((0.020, 0.055))
            painter.setPen(sparse_pen)
            painter.drawEllipse(self._arc_rect(device_r * 1.04))
            dash_pen = QPen(QColor(*projection, int(alpha * 0.78)), 3.4, Qt.PenStyle.CustomDashLine)
            dash_pen.setDashPattern((0.40, 0.10))
            painter.setPen(dash_pen)
            painter.drawEllipse(self._arc_rect(outer))
            mid = size * 0.185
            painter.setPen(QPen(QColor(*projection, int(alpha * 0.66)), 2.2))
            painter.drawEllipse(self._arc_rect(mid))
            hot = size * 0.078
            hot_gradient = QRadialGradient(0, 0, hot)
            hot_gradient.setColorAt(0.0, QColor(240, 252, 255, min(255, int(alpha * 0.98))))
            hot_gradient.setColorAt(0.45, QColor(*peak, int(alpha * 0.62)))
            hot_gradient.setColorAt(1.0, QColor(*projection, 0))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(hot_gradient)
            painter.drawEllipse(self._arc_rect(hot))
            painter.restore()

        def _draw_device(self, painter, c) -> None:
            """Volumetric holographic sphere: body, digital grid, light pockets, glow."""
            center, size = c["center"], c["size"]
            r = c["r"]
            alpha = c["alpha"]
            profile = c["profile"]
            halo, dim, light, bright, peak, depth = (c[key] for key in ("halo", "dim", "light", "bright", "peak", "depth"))
            painter.save()
            painter.translate(center, center)

            # A. Volumetric sphere body: dark translucent core with offset light
            # and a brighter rim, so the window reads as a 3D holographic
            # sphere instead of a stack of flat concentric circles.
            sphere_gradient = QRadialGradient(-r * 0.26, -r * 0.32, r * 1.30)
            state = c["state"]
            state_rgb = c["rgb"]
            if state is OrbVisualState.AUTHORIZATION:
                core, mid, rim = (24, 12, 2), (58, 28, 4), (140, 78, 10)
            elif state is OrbVisualState.AUTOMATION:
                core, mid, rim = (16, 2, 7), (42, 6, 13), (124, 20, 26)
            elif state is OrbVisualState.PROCESSING:
                core, mid, rim = (2, 10, 34), (4, 36, 96), (9, 98, 182)
            elif state is OrbVisualState.SPEAKING:
                core, mid, rim = (2, 22, 16), (4, 52, 32), (15, 108, 62)
            else:
                core, mid, rim = (2, 8, 24), (5, 26, 64), (12, 76, 138)
            sphere_gradient.setColorAt(0.0, QColor(*core, min(255, alpha + 16)))
            sphere_gradient.setColorAt(0.42, QColor(*mid, int(alpha * 0.94)))
            sphere_gradient.setColorAt(0.80, QColor(*rim, int(alpha * 0.66)))
            sphere_gradient.setColorAt(0.94, QColor(*halo, int(alpha * 0.30)))
            sphere_gradient.setColorAt(1.0, QColor(*state_rgb, 0))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(sphere_gradient)
            painter.drawEllipse(self._arc_rect(r))

            # B. Digital surface grid clipped to the sphere: latitude parallels
            # plus longitude meridians replace the old flat ring stack.
            painter.save()
            clip = QPainterPath()
            clip.addEllipse(self._arc_rect(r * 0.995))
            painter.setClipPath(clip)
            grid_pen = QPen(QColor(*light, int(alpha * 0.20)), 1.0)
            for latitude in (-0.66, -0.33, 0.0, 0.33, 0.66):
                span = math.sqrt(max(0.05, 1.0 - latitude * latitude))
                lat_rect = QRectF(
                    -span * r,
                    latitude * r - span * r * 0.24,
                    span * r * 2.0,
                    span * r * 0.48,
                )
                if latitude == 0.0:
                    equator_pen = QPen(QColor(*peak, int(alpha * 0.34)), 1.2, Qt.PenStyle.CustomDashLine)
                    equator_pen.setDashPattern((0.05, 0.035))
                    painter.setPen(equator_pen)
                else:
                    painter.setPen(grid_pen)
                painter.drawEllipse(lat_rect)
            painter.setPen(QPen(QColor(*light, int(alpha * 0.13)), 1.0))
            for meridian in (0.28, 0.60, 0.88):
                painter.drawEllipse(QRectF(-meridian * r, -r, meridian * r * 2.0, r * 2.0))
            # Inner illumination along the upper-left limb (light inside the volume).
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(QPen(QColor(*peak, int(alpha * 0.34)), 1.8))
            painter.drawArc(self._arc_rect(r * 0.97), int(115 * 16), int(58 * 16))
            painter.restore()

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

            # C. Specular highlight: one compact light pocket near the offset
            # light source sells the glass curvature.
            highlight = QRadialGradient(-r * 0.42, -r * 0.44, r * 0.30)
            highlight.setColorAt(0.0, QColor(255, 255, 255, int(alpha * 0.16 * profile["halo_strength"])))
            highlight.setColorAt(0.45, QColor(255, 255, 255, int(alpha * 0.05)))
            highlight.setColorAt(1.0, QColor(255, 255, 255, 0))
            painter.setBrush(highlight)
            painter.drawEllipse(self._arc_rect(r * 0.30).translated(QPointF(-r * 0.42, -r * 0.44)))

            # D. Rim treatment: one luminous border plus a single inner echo
            # ring (the two thin inner circles and arc fragments are gone).
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(QPen(QColor(*peak, int(alpha * 0.42)), 1.2))
            painter.drawEllipse(QRectF(-r + 1, -r + 1, (r - 1) * 2, (r - 1) * 2))
            painter.setPen(QPen(QColor(*bright, int(alpha * 0.22)), 1.0))
            painter.drawEllipse(self._arc_rect(r * 0.84))

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

            # Dark depth pocket behind the emblem keeps it readable over the plasma.
            pocket = QRadialGradient(0, r * 0.10, r * 0.44)
            pocket.setColorAt(0.0, QColor(*depth, int(alpha * 0.44)))
            pocket.setColorAt(0.62, QColor(*depth, int(alpha * 0.26)))
            pocket.setColorAt(1.0, QColor(*depth, 0))
            painter.setBrush(pocket)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(self._arc_rect(r * 0.44).translated(QPointF(0, r * 0.10)))
            # Localised rim reflection: a short bright glint low-right inside the sphere.
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(QPen(QColor(*peak, int(alpha * 0.34)), 1.6))
            painter.drawArc(self._arc_rect(r * 0.90), int(-64 * 16), int(26 * 16))
            painter.setPen(QPen(QColor(255, 255, 255, int(alpha * 0.26)), 0.9))
            painter.drawArc(self._arc_rect(r * 0.90), int(-60 * 16), int(10 * 16))

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
                (0.22, 2.1, 1.1, 1.1, 0.70), (0.34, 4.7, -0.85, 0.9, 0.58),
                (0.47, 0.4, 0.75, 0.8, 0.74), (0.59, 5.9, -0.5, 0.8, 0.68),
                (0.66, 3.1, 0.6, 1.0, 0.50), (0.81, 1.8, -0.4, 1.2, 0.46),
                (0.26, 3.9, -1.0, 0.8, 0.66), (0.90, 4.4, 0.28, 0.9, 0.38),
            ):
                angle = (
                    phase_offset
                    + c["elapsed"] * speed * (0.4 + 0.9 * intensity)
                    + c["frame"]["rotation_deg"] * 0.02
                )
                px = center + math.cos(angle) * r * orbit_factor
                py = center + math.sin(angle) * r * orbit_factor * 0.92
                painter.setBrush(QColor(*c["peak"], int(alpha * 0.22 * base_opacity * intensity)))
                painter.drawEllipse(QPointF(px, py), dot * 2.4, dot * 2.4)
                painter.setBrush(QColor(*c["peak"], int(alpha * base_opacity * intensity)))
                painter.drawEllipse(QPointF(px, py), dot, dot)
                painter.setBrush(QColor(255, 255, 255, int(alpha * 0.55 * base_opacity * intensity)))
                painter.drawEllipse(QPointF(px - dot * 0.3, py - dot * 0.3), dot * 0.42, dot * 0.42)

        def _draw_emblem(self, painter, c) -> None:
            """The Atlas chevron shares the light of the nucleus."""
            center, size, alpha = c["center"], c["size"], c["alpha"]
            emblem = self._emblem_path
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(*c["depth"], int(alpha * 0.78)))
            painter.save()
            painter.translate(2.0, 2.6)
            painter.drawPath(emblem)
            painter.restore()
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(QPen(QColor(*c["halo"], int(alpha * 0.55)), 5.2))
            painter.drawPath(emblem)
            painter.setPen(QPen(QColor(*c["peak"], min(255, alpha + 6)), 1.1))
            painter.drawPath(emblem)
            emblem_gradient = QRadialGradient(center, center - size * 0.02, size * 0.24)
            emblem_gradient.setColorAt(0.0, QColor(250, 253, 255, min(255, alpha + 18)))
            emblem_gradient.setColorAt(0.52, QColor(*c["peak"], min(255, alpha + 12)))
            emblem_gradient.setColorAt(1.0, QColor(*c["light"], min(255, int(alpha * 0.92))))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(emblem_gradient)
            painter.drawPath(emblem)

        def _draw_state_activity(self, painter, c) -> None:
            """Per-state behaviour: nucleus waves, pulse rings, attention pings, sparks."""
            center, r = c["center"], c["r"]
            alpha, profile, state = c["alpha"], c["profile"], c["state"]
            period, elapsed = c["period"], c["elapsed"]
            painter.setBrush(Qt.BrushStyle.NoBrush)

            if state is OrbVisualState.PROCESSING and period:
                # Intense blue-cyan energy waves circling the nucleus (no lateral bars).
                for index in range(4):
                    angle = (elapsed * 84.0 * (1.0 if index % 2 == 0 else -0.8) + index * 87.0) % 360.0
                    radius = r * (1.02 + 0.055 * (index % 3))
                    painter.save()
                    painter.translate(center, center)
                    painter.rotate(angle)
                    painter.setPen(QPen(QColor(*c["peak"], int(alpha * 0.55)), 2.0))
                    painter.drawArc(self._arc_rect(radius), 0, 34 * 16)
                    painter.setPen(QPen(QColor(255, 255, 255, int(alpha * 0.45)), 0.9))
                    painter.drawArc(self._arc_rect(radius), 0, 10 * 16)
                    painter.restore()
                # One thin ring breathing outward from the nucleus.
                p = (elapsed / period) % 1.0
                radius = r * (1.00 + 0.40 * p)
                painter.setPen(QPen(QColor(*c["peak"], int(alpha * 0.30 * (1.0 - p))), 1.6))
                painter.drawEllipse(self._arc_rect(radius).translated(QPointF(center, center)))

            if state is OrbVisualState.SPEAKING and period:
                # Voice pulse: one thin ring born in the nucleus expands while speaking.
                for offset in (0.0, 0.45):
                    p = ((elapsed / period) + offset) % 1.0
                    radius = r * (1.04 + 0.52 * p)
                    painter.setPen(QPen(QColor(*c["peak"], int(alpha * 0.40 * (1.0 - p))), 1.6))
                    painter.drawEllipse(self._arc_rect(radius).translated(QPointF(center, center)))

            ping = profile["ping_intensity"]
            if ping > 0.02 and period:
                # Authorization: concentric attention circles asking for approval.
                for offset in (0.0, 0.5):
                    p = ((elapsed / period) + offset) % 1.0
                    radius = r * (1.10 + 0.48 * p)
                    painter.setPen(QPen(QColor(*c["peak"], int(alpha * 0.34 * ping * (1.0 - p))), 1.8))
                    painter.drawEllipse(self._arc_rect(radius).translated(QPointF(center, center)))
                # Small amber attention dots waiting around the device.
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.setPen(Qt.PenStyle.NoPen)
                for index in range(6):
                    angle = math.radians(index * 60.0 + elapsed * 22.0)
                    radius = r * (1.18 + 0.08 * math.sin(elapsed * 2.1 + index * 1.7))
                    twinkle = 0.45 + 0.55 * (math.sin(elapsed * 4.6 + index * 2.3) * 0.5 + 0.5)
                    painter.setBrush(QColor(*c["peak"], int(alpha * 0.60 * ping * twinkle)))
                    painter.drawEllipse(
                        QPointF(center + math.cos(angle) * radius, center + math.sin(angle) * radius * 0.92),
                        1.8, 1.8,
                    )

            if state is OrbVisualState.LISTENING:
                # Short rotating arcs instead of full-circle radar rings.
                for offset, opacity in ((0.00, 0.42), (0.42, 0.26)):
                    wave_p = (math.sin(c["frame"]["rotation_deg"] * 0.105 - offset * math.tau) + 1.0) * 0.5
                    radius = r * (1.34 + 0.08 * wave_p)
                    start = (c["frame"]["rotation_deg"] * 0.9 + offset * 180.0) % 360.0
                    painter.setPen(QPen(QColor(*c["halo"], int(alpha * opacity * wave_p)), 2.4))
                    painter.drawArc(
                        self._arc_rect(radius).translated(QPointF(center, center)),
                        int(start * 16), int(64 * 16),
                    )

            # Automation: quick bright arclets orbiting the running device.
            spark = profile["spark_intensity"]
            if spark > 0.02:
                radius = c["size"] * 0.414
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
                (0.0, 0.37, 0.44, 0.0, 1.6), (1.6, 0.42, 0.26, 201.0, 1.1),
                (2.9, 0.40, 0.56, 57.0, 1.6), (4.1, 0.35, 0.20, -41.0, 1.1),
                (5.3, 0.44, 0.40, 96.0, 1.1), (0.9, 0.38, 0.48, 152.0, 1.1),
                (3.6, 0.46, 0.30, 224.0, 1.6),
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

        def prefill_input(self, text: str) -> None:
            """Place routing text in the input without submitting it.

            Capability menu options reuse the existing chat routing: the user
            completes the prompt and the real router resolves the domain.
            """
            self._input.setText(str(text))
            self._input.setFocus()

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
