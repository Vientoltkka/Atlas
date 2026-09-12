"""Atlas Orbe V1 stage: fullscreen holographic shell around the existing orb.

Pure VISUAL layer that never blocks Windows: the stage is permanently
click-through (``WA_TransparentForMouseEvents`` plus the Win32
``WS_EX_TRANSPARENT | WS_EX_LAYERED`` styles), so the desktop, the
taskbar and every other window stay fully usable while Atlas is visible.
All interaction lives in :class:`OrbeStageControls`, a masked overlay
window whose only clickable regions are the real buttons (X, capability
menu). The stage owns NO conversation logic: it renders a cinematic
composition (background depth, lateral holo panels, clock) and Atlas
state arrives through ``apply_state(OrbVisualState)`` — the same real
state the orb already renders — so no second state machine exists.
"""
from __future__ import annotations

import ctypes
import math
import sys
import time
from datetime import datetime

from PySide6.QtCore import QPointF, QRect, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import (
    QColor,
    QFont,
    QLinearGradient,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
    QRadialGradient,
    QRegion,
)
from PySide6.QtWidgets import QPushButton, QWidget

from use_cases.ui_state_mapper import OrbVisualState

# Stage FPS: mostly static gradients plus a handful of animated accents.
# FPS is proportional to the REAL state (activity-aware): busy states get
# 15 fps, reposo baja a 6 fps para no robar CPU a STT/TTS/agentes.
STAGE_ANIMATION_FPS = 15
STAGE_IDLE_FPS = 6
_METRICS_PERIOD_MS = 2000
_CLOCK_PERIOD_MS = 1000
_MAX_PARTICLES = 30

# SAFETY: the stage is ALWAYS click-through at the Win32 level. There is
# no polling and no giant hitbox: only OrbeStageControls (masked buttons)
# ever receives input.
_GWL_EXSTYLE = -20
_WS_EX_TRANSPARENT = 0x00000020
_WS_EX_LAYERED = 0x00080000
_WS_EX_NOACTIVATE = 0x08000000

# Real system metrics (no external deps): GlobalMemoryStatusEx + GetSystemTimes.


class _MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_uint32),
        ("dwMemoryLoad", ctypes.c_uint32),
        ("ullTotalPhys", ctypes.c_uint64),
        ("ullAvailPhys", ctypes.c_uint64),
        ("ullTotalPageFile", ctypes.c_uint64),
        ("ullAvailPageFile", ctypes.c_uint64),
        ("ullTotalVirtual", ctypes.c_uint64),
        ("ullAvailVirtual", ctypes.c_uint64),
        ("ullAvailExtendedVirtual", ctypes.c_uint64),
    ]


def system_memory_percent() -> float | None:
    """Real memory load percent via GlobalMemoryStatusEx; None off-Windows."""
    if sys.platform != "win32":
        return None
    try:
        status = _MEMORYSTATUSEX()
        status.dwLength = ctypes.sizeof(_MEMORYSTATUSEX)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return None
        return float(status.dwMemoryLoad)
    except Exception:
        return None


def system_cpu_percent() -> float | None:
    """Real CPU busy percent from GetSystemTimes deltas; None on first call."""
    if sys.platform != "win32":
        return None
    try:
        idle = ctypes.c_ulonglong()
        kernel = ctypes.c_ulonglong()
        user = ctypes.c_ulonglong()
        if not ctypes.windll.kernel32.GetSystemTimes(
            ctypes.byref(idle), ctypes.byref(kernel), ctypes.byref(user)
        ):
            return None
        now = (idle.value, kernel.value, user.value)
        last = getattr(system_cpu_percent, "_last", None)
        system_cpu_percent._last = now  # type: ignore[attr-defined]
        if last is None:
            return None
        idle_delta = now[0] - last[0]
        total_delta = (now[1] - last[1]) + (now[2] - last[2])
        if total_delta <= 0:
            return None
        return max(0.0, min(100.0, 100.0 * (1.0 - idle_delta / total_delta)))
    except Exception:
        return None


def _agent_names(atlas) -> list[str]:
    """Best-effort real agent names from the live Atlas core (never fake)."""
    for attr in ("list_agent_names", "agent_names"):
        getter = getattr(atlas, attr, None)
        if callable(getter):
            try:
                return [str(name) for name in getter()][:8]
            except Exception:
                pass
    agents = getattr(atlas, "agents", None)
    if isinstance(agents, dict):
        return [str(name) for name in agents][:8]
    if isinstance(agents, (list, tuple)):
        return [str(getattr(item, "name", item)) for item in agents][:8]
    return []


_STATE_COGNITIVE = {
    OrbVisualState.IDLE: ("EN REPOSO", (70, 205, 255)),
    OrbVisualState.STARTING: ("INICIANDO", (105, 225, 255)),
    OrbVisualState.LISTENING: ("ESCUCHANDO", (80, 225, 255)),
    OrbVisualState.PROCESSING: ("ANALIZANDO · PROCESANDO", (64, 170, 255)),
    OrbVisualState.SPEAKING: ("RESPONDIENDO", (72, 238, 148)),
    OrbVisualState.AUTHORIZATION: ("ESPERANDO AUTORIZACIÓN", (255, 174, 52)),
    OrbVisualState.AUTOMATION: ("EJECUTANDO AUTOMATIZACIÓN", (255, 82, 82)),
    OrbVisualState.RECOVERING: ("RECUPERANDO", (60, 175, 235)),
    OrbVisualState.DEGRADED: ("RENDIMIENTO DEGRADADO", (140, 150, 165)),
    OrbVisualState.STOPPING: ("DETENIENDO", (80, 145, 180)),
}

_STATE_CORE_LABEL = {
    OrbVisualState.IDLE: "ACTIVE",
    OrbVisualState.LISTENING: "ACTIVE · VOZ",
    OrbVisualState.PROCESSING: "ACTIVE · RAZONANDO",
    OrbVisualState.SPEAKING: "ACTIVE · VOZ",
    OrbVisualState.AUTOMATION: "ACTIVE · EJECUCIÓN",
    OrbVisualState.AUTHORIZATION: "ACTIVE · AUTORIZAR",
}

_INTERFACE_ROWS = (
    ("CHAT", True),
    ("VOICE", True),
    ("FILES", True),
    ("APPS", True),
    ("CALENDAR", True),
    ("WEB", True),
    ("SYSTEM", True),
)

_CAPABILITY_OPTIONS: tuple[tuple[str, str], ...] = (
    ("chat", "CHAT"),
    ("voz", "VOZ"),
    ("control_pc", "PC"),
    ("entrenamiento", "ENTRENAMIENTO"),
    ("nutricion", "NUTRICIÓN"),
    ("salud", "SALUD"),
    ("coding", "CÓDIGO"),
    ("proyectos", "PROYECTOS"),
    ("mas_herramientas", "HERRAMIENTAS"),
)


class _Particle:
    __slots__ = (
        "angle", "radius_factor", "speed", "size", "phase",
        "blink_speed", "orbit_squash", "orbit_tilt",
    )

    def __init__(self, angle: float, radius_factor: float, speed: float, size: float,
                 phase: float, blink_speed: float, squash: float, tilt: float) -> None:
        self.angle = angle
        self.radius_factor = radius_factor
        self.speed = speed
        self.size = size
        self.phase = phase
        self.blink_speed = blink_speed
        self.orbit_squash = squash
        self.orbit_tilt = tilt


_PARTICLE_SEEDS = tuple(
    (
        i * 0.61,                     # angle
        1.12 + (i % 5) * 0.09,        # radius factor over sphere radius
        0.10 + (i % 4) * 0.045,       # angular speed rad/s
        1.0 + (i % 3) * 0.5,          # dot radius
        (i * 1.29) % 6.283,           # blink phase
        0.9 + (i % 3) * 0.5,          # blink speed
        0.30 + (i % 4) * 0.10,        # orbit squash
        (i % 6 - 2) * 14.0,           # orbit tilt deg
    )
    for i in range(_MAX_PARTICLES)
)


def _draw_holo_panel(painter, rect: QRectF, *, accent: QColor,
                     fill_top=(6, 26, 52, 66), fill_mid=(3, 14, 32, 46),
                     fill_bot=(2, 9, 22, 34)) -> None:
    """One translucent holo panel: glow border, gradient fill, corner ticks."""
    inner = rect.adjusted(1, 1, -1, -1)
    path = QPainterPath()
    path.addRoundedRect(inner, 10, 10)
    gradient = QLinearGradient(inner.topLeft(), inner.bottomLeft())
    gradient.setColorAt(0.0, QColor(*fill_top))
    gradient.setColorAt(0.55, QColor(*fill_mid))
    gradient.setColorAt(1.0, QColor(*fill_bot))
    painter.setBrush(gradient)
    painter.drawPath(path)
    painter.setBrush(Qt.BrushStyle.NoBrush)
    # Soft outer glow pass, then the crisp accent border.
    glow = QColor(accent.red(), accent.green(), accent.blue(), 40)
    painter.setPen(QPen(glow, 4.5))
    painter.drawPath(path)
    painter.setPen(QPen(QColor(accent.red(), accent.green(), accent.blue(), 150), 1.2))
    painter.drawPath(path)
    line_y = inner.top() + 26
    painter.setPen(QPen(QColor(accent.red(), accent.green(), accent.blue(), 60), 1.0))
    painter.drawLine(QPointF(inner.left() + 10, line_y), QPointF(inner.right() - 10, line_y))
    tick = 10
    painter.setPen(QPen(QColor(190, 245, 255, 190), 2.0))
    painter.drawLine(inner.topLeft() + QPointF(0, tick), inner.topLeft() + QPointF(tick, 0))
    painter.drawLine(inner.topLeft() + QPointF(0, -tick), inner.topLeft() + QPointF(tick, 0))
    painter.drawLine(inner.bottomRight() + QPointF(0, -tick), inner.bottomRight() + QPointF(-tick, 0))
    painter.drawLine(inner.bottomRight() + QPointF(0, tick), inner.bottomRight() + QPointF(-tick, 0))


class OrbeStage(QWidget):
    """Fullscreen holographic stage hosting the existing orb window.

    SAFETY CONTRACT: this window is pure visual and permanently
    click-through. It never receives mouse input and never blocks the
    desktop, the taskbar or other windows; interaction is delegated to
    :class:`OrbeStageControls`.
    """

    chat_requested = Signal()
    voice_requested = Signal()
    quit_requested = Signal()
    capability_selected = Signal(str)
    # Failsafe: ESC / X button / Alt+F4 must always be able to hide the UI.
    hide_requested = Signal()

    def __init__(self, orb, atlas=None, parent=None) -> None:
        super().__init__(parent)
        self._orb = orb
        self._atlas = atlas
        self._state = OrbVisualState.IDLE
        self._voice_active = False
        self._automation_task = ""
        self._last_error = ""
        self._started_at = time.monotonic()

        self.setWindowTitle("Atlas Stage")
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        # Fullscreen must be the window STATE, not a post-show() call:
        # calling showFullScreen() inside showEvent (or before exec())
        # leaves the native window unmapped and Atlas never appears.
        self.setWindowState(Qt.WindowState.WindowFullScreen)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        # SAFETY: Qt-level click-through, independent of any Win32 style.
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)

        self._particles = [_Particle(*seed) for seed in _PARTICLE_SEEDS]
        self._panel_rects: tuple[QRectF, ...] = ()
        self._cpu_percent: float | None = None
        self._mem_percent: float | None = None
        self._agents: list[str] = []
        self._clock_text = ""
        self._date_text = ""
        self._static_cache = None
        self._static_cache_key_value: tuple | None = None

        self._timer = QTimer(self)
        self._timer.setInterval(int(1000 / STAGE_ANIMATION_FPS))
        self._timer.timeout.connect(self.update)
        self._metrics_timer = QTimer(self)
        self._metrics_timer.setInterval(_METRICS_PERIOD_MS)
        self._metrics_timer.timeout.connect(self._refresh_metrics)
        self._clock_timer = QTimer(self)
        self._clock_timer.setInterval(_CLOCK_PERIOD_MS)
        self._clock_timer.timeout.connect(self._refresh_clock)
        self._refresh_clock()
        self._refresh_metrics()

    def _orb_center(self):
        """Orb centre in stage-local coordinates: the composition follows the
        REAL sphere position, so the compact orb stays draggable and
        position-persistent without any re-anchoring timer."""
        from PySide6.QtCore import QPointF

        local = self.mapFromGlobal(self._orb.frameGeometry().center())
        return QPointF(float(local.x()), float(local.y()))

    # ------------------------------------------------------------------ state

    @property
    def state(self) -> OrbVisualState:
        return self._state

    @property
    def orb(self):
        return self._orb

    def apply_state(self, state) -> None:
        """Adopt the same real OrbVisualState the orb window shows."""
        try:
            self._state = OrbVisualState(state)
        except ValueError:
            return
        # Activity-aware FPS: reposo consumes much less than execution.
        busy = self._state in (
            OrbVisualState.PROCESSING, OrbVisualState.SPEAKING,
            OrbVisualState.AUTOMATION, OrbVisualState.RECOVERING,
            OrbVisualState.LISTENING,
        )
        self._timer.setInterval(int(1000 / (STAGE_ANIMATION_FPS if busy else STAGE_IDLE_FPS)))
        self.update()

    def set_voice_active(self, active: bool) -> None:
        self._voice_active = bool(active)
        self.update()

    def set_automation_task(self, description: str) -> None:
        """Real automation task label shown by the red panel (empty = none)."""
        self._automation_task = str(description or "").strip()
        self.update()

    def set_last_error(self, message: str) -> None:
        """Real error text for the discrete error signal (empty = clear)."""
        self._last_error = str(message or "").strip()
        self.update()

    # ------------------------------------------------------------------ show/hide

    def showEvent(self, event) -> None:  # noqa: N802 (Qt API)
        super().showEvent(event)
        self._apply_click_through_style()
        if not self._timer.isActive():
            self._timer.start()
        if not self._metrics_timer.isActive():
            self._metrics_timer.start()
        if not self._clock_timer.isActive():
            self._clock_timer.start()

    def hideEvent(self, event) -> None:  # noqa: N802 (Qt API)
        """Hide must stop every stage timer immediately (no zombie CPU)."""
        if self._timer.isActive():
            self._timer.stop()
        self._metrics_timer.stop()
        self._clock_timer.stop()
        super().hideEvent(event)

    def closeEvent(self, event) -> None:  # noqa: N802 (Qt API)
        """Alt+F4 must never leave a zombie overlay: hide everything instead."""
        event.ignore()
        self.hide_requested.emit()
        # Belt and braces: even without a controller listener, this overlay
        # must hide itself instead of staying fullscreen on top of Windows.
        self.hide()

    def keyPressEvent(self, event) -> None:  # noqa: N802 (Qt API)
        if event.key() == Qt.Key.Key_Escape:
            self.hide_requested.emit()
            event.accept()
            return
        super().keyPressEvent(event)

    # ------------------------------------------------------------------ click-through

    def _apply_click_through_style(self) -> None:
        """Permanently forward clicks to Windows (WS_EX_TRANSPARENT).

        No polling, no toggling: the stage is visual-only for its whole
        lifetime, so the desktop and the taskbar stay usable even while
        the cursor moves across it.
        """
        if sys.platform != "win32":
            return
        try:
            from PySide6.QtGui import QGuiApplication

            if QGuiApplication.platformName() == "offscreen":
                return
            import ctypes.wintypes as wintypes

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
            updated = current | _WS_EX_TRANSPARENT | _WS_EX_LAYERED | _WS_EX_NOACTIVATE
            if updated != current:
                set_style(hwnd, _GWL_EXSTYLE, updated)
        except Exception:
            # A failed style toggle degrades to the Qt-level
            # WA_TransparentForMouseEvents; the stage must never crash.
            pass

    def _refresh_metrics(self) -> None:
        self._cpu_percent = system_cpu_percent()
        self._mem_percent = system_memory_percent()
        if self._atlas is not None and not self._agents:
            self._agents = _agent_names(self._atlas)
        if self.isVisible():
            self.update()

    def _refresh_clock(self) -> None:
        now = datetime.now()
        self._clock_text = now.strftime("%H:%M")
        self._date_text = now.strftime("%a, %d %b %Y")
        if self.isVisible():
            self.update()

    # ------------------------------------------------------------------ painting

    def _static_cache_key(self):
        return (
            self._state, self._voice_active, self._cpu_percent, self._mem_percent,
            tuple(self._agents), self._automation_task, self._last_error,
            self.width(), self.height(),
        )

    def _static_layer(self):
        """Cached fullscreen layer: fondo + paneles + HUD estático.

        Sólo se repinta cuando cambia el estado real o los datos del panel;
        el coste por frame queda en copiar el pixmap y pintar las pocas
        capas dinámicas (halo, partículas, pulsos, menú, reloj).
        """
        key = self._static_cache_key()
        cache = self._static_cache
        if cache is not None and self._static_cache_key_value == key:
            return cache
        if cache is None or cache.width() != self.width() or cache.height() != self.height():
            # One reusable fullscreen buffer: reallocating per state change
            # churns ~6 MB and stresses the heap under heavy test cycles.
            cache = QPixmap(self.size())
            cache.setDevicePixelRatio(1.0)
            self._static_cache = cache
        cache.fill(Qt.GlobalColor.transparent)
        cache_painter = QPainter(cache)
        cache_painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        self._draw_background(cache_painter, 0.0)
        self._draw_panels(cache_painter)
        self._draw_corner_hud(cache_painter, with_clock=False)
        cache_painter.end()
        self._static_cache = cache
        self._static_cache_key_value = key
        return cache

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt API)
        painter = QPainter(self)
        painter.drawPixmap(0, 0, self._static_layer())
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        elapsed = time.monotonic() - self._started_at
        self._draw_orb_halo(painter, elapsed)
        self._draw_particles(painter, elapsed)
        self._draw_energy_pulses(painter, elapsed)
        self._draw_clock_hud(painter)
        painter.end()

    def _state_accent(self) -> tuple[int, int, int]:
        return _STATE_COGNITIVE.get(self._state, ((70, 205, 255)))[1]

    def _draw_background(self, painter, elapsed: float) -> None:
        """Dark technological depth: vertical gradient + vignette + horizon glow.

        The static layers (base gradient, depth glow, floor pool, vignette)
        are rendered ONCE into a cached pixmap; per-frame cost is only the
        cheap twinkling stars. This keeps 15 fps alive without soaking CPU.
        """
        rect = self.rect()
        cache = getattr(self, "_background_cache", None)
        if cache is None or cache.width() != rect.width() or cache.height() != rect.height():
            from PySide6.QtGui import QPixmap

            cache = QPixmap(rect.size())
            cache.fill(Qt.GlobalColor.transparent)
            cache_painter = QPainter(cache)
            cache_painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            # SAFETY: translucent tint, never an opaque wall. The desktop,
            # the taskbar and other windows must stay visible behind Atlas.
            base = QLinearGradient(0.0, 0.0, 0.0, float(rect.height()))
            base.setColorAt(0.0, QColor(2, 6, 16, 148))
            base.setColorAt(0.45, QColor(3, 10, 26, 138))
            base.setColorAt(1.0, QColor(1, 4, 12, 152))
            cache_painter.fillRect(rect, base)

            # Faint radial depth behind the orb.
            centre = QPointF(rect.width() / 2.0, rect.height() * 0.46)
            depth = QRadialGradient(centre, min(rect.width(), rect.height()) * 0.55)
            depth.setColorAt(0.0, QColor(10, 40, 84, 60))
            depth.setColorAt(0.6, QColor(6, 22, 52, 34))
            depth.setColorAt(1.0, QColor(0, 0, 0, 0))
            cache_painter.setPen(Qt.PenStyle.NoPen)
            cache_painter.setBrush(depth)
            cache_painter.drawEllipse(centre, min(rect.width(), rect.height()) * 0.55,
                                      min(rect.width(), rect.height()) * 0.55)

            # Horizon floor glow under the orb (reflection pool feeling).
            floor_y = rect.height() * 0.86
            floor = QRadialGradient(QPointF(rect.width() / 2.0, floor_y), rect.width() * 0.42)
            floor.setColorAt(0.0, QColor(20, 120, 210, 46))
            floor.setColorAt(0.5, QColor(10, 60, 130, 22))
            floor.setColorAt(1.0, QColor(0, 0, 0, 0))
            cache_painter.setBrush(floor)
            cache_painter.drawEllipse(QPointF(rect.width() / 2.0, floor_y), rect.width() * 0.42, rect.height() * 0.10)

            # Vignette: darker corners for cinematic depth.
            vignette = QRadialGradient(QPointF(rect.width() / 2.0, rect.height() / 2.0),
                                       max(rect.width(), rect.height()) * 0.75)
            vignette.setColorAt(0.0, QColor(0, 0, 0, 0))
            vignette.setColorAt(0.72, QColor(0, 0, 0, 0))
            vignette.setColorAt(1.0, QColor(0, 0, 0, 80))
            cache_painter.setBrush(vignette)
            cache_painter.drawRect(rect)
            cache_painter.end()
            self._background_cache = cache
        painter.drawPixmap(0, 0, cache)

        # Sparse twinkling stars (deterministic, cheap dots per frame).
        painter.setPen(Qt.PenStyle.NoPen)
        for index in range(46):
            sx = (index * 8171) % max(1, rect.width())
            sy = (index * 3607) % max(1, int(rect.height() * 0.9))
            twinkle = 0.5 + 0.5 * math.sin(elapsed * 1.4 + index * 1.7)
            alpha = int(10 + 22 * twinkle)
            painter.setBrush(QColor(160, 220, 255, alpha))
            size = 1.0 + (index % 3) * 0.4
            painter.drawEllipse(QPointF(float(sx), float(sy)), size, size)

    def _orb_radius(self) -> float:
        return min(self.width(), self.height()) * 0.21

    def _draw_orb_halo(self, painter, elapsed: float) -> None:
        """Pulsating glow ring behind the orb centre; red accents on automation."""
        red, green, blue = self._state_accent()
        breath = 0.5 + 0.5 * math.sin(elapsed * 1.1)
        intensity = 0.55 + 0.45 * breath
        radius = self._orb_radius()
        centre = self._orb_center()
        gradient = QRadialGradient(centre, radius * 1.45)
        gradient.setColorAt(0.62, QColor(red, green, blue, 0))
        gradient.setColorAt(0.78, QColor(red, green, blue, int(34 * intensity)))
        gradient.setColorAt(0.88, QColor(red, green, blue, int(66 * intensity)))
        gradient.setColorAt(0.97, QColor(red, green, blue, int(14 * intensity)))
        gradient.setColorAt(1.0, QColor(red, green, blue, 0))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(gradient)
        painter.drawEllipse(centre, radius * 1.45, radius * 1.45)

        # Two slow rotating dashed reference circles around the orb zone.
        painter.setBrush(Qt.BrushStyle.NoBrush)
        for factor, speed, width in ((1.24, 3.2, 1.1), (1.38, -2.1, 1.0)):
            start = (elapsed * 57.3 * speed) % 360
            pen = QPen(QColor(red, green, blue, 60), width, Qt.PenStyle.CustomDashLine)
            pen.setDashPattern((0.06, 0.10))
            painter.setPen(pen)
            painter.save()
            painter.translate(centre)
            painter.rotate(start)
            painter.drawEllipse(QPointF(0, 0), radius * factor, radius * factor)
            painter.restore()

    def _draw_particles(self, painter, elapsed: float) -> None:
        """Deterministic data motes orbiting the orb; density grows with state."""
        centre = self._orb_center()
        red, green, blue = self._state_accent()
        activity = {
            OrbVisualState.IDLE: 0.5, OrbVisualState.STARTING: 0.6,
            OrbVisualState.LISTENING: 0.7, OrbVisualState.PROCESSING: 1.0,
            OrbVisualState.SPEAKING: 0.8, OrbVisualState.AUTHORIZATION: 0.55,
            OrbVisualState.AUTOMATION: 1.0, OrbVisualState.RECOVERING: 0.6,
            OrbVisualState.DEGRADED: 0.3, OrbVisualState.STOPPING: 0.4,
        }.get(self._state, 0.5)
        radius = self._orb_radius()
        painter.setPen(Qt.PenStyle.NoPen)
        visible = int(len(self._particles) * (0.55 + 0.45 * activity))
        for index, particle in enumerate(self._particles):
            if index >= visible:
                break
            angle = particle.angle + elapsed * particle.speed * (0.6 + 0.8 * activity)
            rf = particle.radius_factor + 0.03 * math.sin(elapsed * 0.9 + particle.phase)
            x0 = math.cos(angle) * radius * rf
            y0 = math.sin(angle) * radius * rf * particle.orbit_squash
            tilt = math.radians(particle.orbit_tilt)
            x = centre.x() + x0 * math.cos(tilt) - y0 * math.sin(tilt)
            y = centre.y() + x0 * math.sin(tilt) + y0 * math.cos(tilt)
            fade = 0.5 + 0.5 * math.sin(elapsed * particle.blink_speed + particle.phase)
            alpha = int(90 * activity * (0.3 + 0.7 * fade * fade))
            painter.setBrush(QColor(red, green, blue, alpha))
            painter.drawEllipse(QPointF(x, y), particle.size, particle.size)

    def _draw_energy_pulses(self, painter, elapsed: float) -> None:
        """State-driven expanding rings from the orb (rare when idle)."""
        state = self._state
        period = {
            OrbVisualState.IDLE: 8.0, OrbVisualState.STARTING: 3.0,
            OrbVisualState.LISTENING: 2.6, OrbVisualState.PROCESSING: 1.8,
            OrbVisualState.SPEAKING: 1.4, OrbVisualState.AUTHORIZATION: 3.4,
            OrbVisualState.AUTOMATION: 1.6, OrbVisualState.RECOVERING: 2.2,
            OrbVisualState.DEGRADED: 0.0, OrbVisualState.STOPPING: 4.0,
        }.get(state, 8.0)
        if period <= 0:
            return
        red, green, blue = self._state_accent()
        centre = self._orb_center()
        radius = self._orb_radius()
        painter.setBrush(Qt.BrushStyle.NoBrush)
        offsets = (0.0, 0.5) if state in (OrbVisualState.PROCESSING, OrbVisualState.AUTOMATION,
                                          OrbVisualState.SPEAKING) else (0.0,)
        for offset in offsets:
            p = ((elapsed / period) + offset) % 1.0
            ring = radius * (1.05 + 0.55 * p)
            alpha = int(90 * (1.0 - p))
            pen = QPen(QColor(red, green, blue, alpha), 1.6)
            painter.setPen(pen)
            painter.save()
            painter.translate(centre)
            painter.scale(1.0, 0.92)
            painter.drawEllipse(QPointF(0, 0), ring, ring)
            painter.restore()

    def _draw_panels(self, painter) -> None:
        w, h = self.width(), self.height()
        margin = max(18.0, w * 0.022)
        panel_w = min(340.0, max(240.0, w * 0.19))
        panel_h = min(230.0, max(160.0, h * 0.24))
        top = h * 0.14
        gap = max(16.0, h * 0.03)
        red, green, blue = self._state_accent()
        accent = QColor(red, green, blue)

        left_top = QRectF(margin, top, panel_w, panel_h)
        left_bottom = QRectF(margin, top + panel_h + gap, panel_w, panel_h * 0.82)
        right_top = QRectF(w - margin - panel_w, top, panel_w, panel_h)
        right_bottom = QRectF(w - margin - panel_w, top + panel_h + gap, panel_w, panel_h * 0.86)
        self._panel_rects = (left_top, left_bottom, right_top, right_bottom)

        self._draw_system_core_panel(painter, left_top)
        self._draw_cognitive_panel(painter, left_bottom)
        self._draw_data_node_panel(painter, right_top)
        self._draw_automation_panel(painter, right_bottom)

    def _panel_title(self, painter, rect: QRectF, text: str, accent: QColor) -> None:
        painter.setPen(QColor(190, 245, 255, 235))
        font = painter.font()
        font.setPixelSize(13)
        font.setBold(True)
        font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 1.2)
        painter.setFont(font)
        painter.drawText(rect.adjusted(12, 5, -12, 0), Qt.AlignmentFlag.AlignLeft, text)

    def _draw_system_core_panel(self, painter, rect: QRectF) -> None:
        _draw_holo_panel(painter, rect, accent=QColor(96, 202, 255))
        self._panel_title(painter, rect, "SYSTEM CORE STATUS", QColor(96, 202, 255))
        inner = rect.adjusted(14, 32, -14, -10)
        cpu = 0 if self._cpu_percent is None else int(round(self._cpu_percent))
        mem = 0 if self._mem_percent is None else int(round(self._mem_percent))
        # CPU ring gauge.
        gauge_rect = QRectF(inner.left() + 4, inner.top() + 6, 74, 74)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        base_pen = QPen(QColor(30, 70, 110, 160), 6.0)
        painter.setPen(base_pen)
        painter.drawArc(gauge_rect, 45 * 16, 270 * 16)
        value_pen = QPen(QColor(96, 220, 255, 220), 6.0)
        painter.setPen(value_pen)
        span = int(270 * 16 * min(1.0, cpu / 100.0))
        painter.drawArc(gauge_rect, 45 * 16, -span)
        painter.setPen(QColor(225, 250, 255, 240))
        font = painter.font()
        font.setPixelSize(15)
        font.setBold(True)
        painter.setFont(font)
        painter.drawText(gauge_rect, Qt.AlignmentFlag.AlignCenter, f"{cpu}%")
        painter.setPen(QColor(140, 200, 240, 190))
        font.setPixelSize(9)
        font.setBold(False)
        painter.setFont(font)
        painter.drawText(QRectF(gauge_rect.left(), gauge_rect.bottom() + 2, gauge_rect.width(), 14),
                         Qt.AlignmentFlag.AlignCenter, "CPU")
        # Memory bar + real facts.
        bar_x = gauge_rect.right() + 18
        bar_rect = QRectF(bar_x, inner.top() + 12, inner.right() - bar_x, 8)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(24, 60, 96, 180))
        painter.drawRoundedRect(bar_rect, 4, 4)
        fill_w = bar_rect.width() * min(1.0, mem / 100.0)
        painter.setBrush(QColor(96, 220, 255, 210))
        painter.drawRoundedRect(QRectF(bar_rect.left(), bar_rect.top(), fill_w, 8), 4, 4)
        painter.setPen(QColor(140, 200, 240, 200))
        font.setPixelSize(10)
        painter.setFont(font)
        label = f"MEMORIA {mem}%"
        painter.drawText(QRectF(bar_x, bar_rect.bottom() + 3, bar_rect.width(), 14),
                         Qt.AlignmentFlag.AlignLeft, label)
        state_label = _STATE_CORE_LABEL.get(self._state, "ACTIVE")
        lines = [f"ESTADO ATLAS: {state_label}"]
        if self._agents:
            lines.append(f"AGENTES: {len(self._agents)} ACTIVOS")
            lines.append(", ".join(self._agents[:3]).upper())
        else:
            lines.append("AGENTES: SISTEMA NÚCLEO")
        lines.append("MEMORIA OPTIMIZADA · SISTEMA OPERATIVO")
        y = bar_rect.bottom() + 24
        for line in lines:
            painter.drawText(QRectF(bar_x, y, inner.right() - bar_x, 15),
                             Qt.AlignmentFlag.AlignLeft, line)
            y += 16

    def _draw_cognitive_panel(self, painter, rect: QRectF) -> None:
        label, rgb = _STATE_COGNITIVE.get(self._state, ("EN REPOSO", (70, 205, 255)))
        accent = QColor(*rgb)
        _draw_holo_panel(painter, rect, accent=accent)
        title = f"PENSANDO: {label}" if self._state in (OrbVisualState.PROCESSING, OrbVisualState.AUTOMATION) else label
        self._panel_title(painter, rect, title, accent)
        inner = rect.adjusted(14, 32, -14, -8)
        # Animated cognitive wave bars (pseudo waveform driven by elapsed).
        painter.setPen(Qt.PenStyle.NoPen)
        bar_count = 18
        bar_w = inner.width() / (bar_count * 2)
        for index in range(bar_count):
            h_frac = abs(math.sin(index * 0.7)) * (0.35 + 0.4 * abs(math.sin(index * 0.23)))
            height = inner.height() * 0.45 * h_frac
            x = inner.left() + index * bar_w * 2
            painter.setBrush(QColor(*rgb, 150))
            painter.drawRoundedRect(QRectF(x, inner.center().y() - height, bar_w, height), 1.5, 1.5)
        # Steps checklist on the left column.
        steps = ("ANALIZANDO", "RAZONANDO", "PLANIFICANDO", "GENERANDO RESPUESTA")
        font = painter.font()
        font.setPixelSize(10)
        painter.setFont(font)
        y = inner.top() + 4
        for step in steps:
            active = self._state in (OrbVisualState.PROCESSING, OrbVisualState.AUTOMATION)
            color = QColor(*rgb, 230) if active else QColor(90, 130, 165, 150)
            painter.setPen(color)
            painter.drawText(QRectF(inner.left(), y, inner.width(), 14),
                             Qt.AlignmentFlag.AlignLeft, step)
            y += 16

    def _draw_data_node_panel(self, painter, rect: QRectF) -> None:
        _draw_holo_panel(painter, rect, accent=QColor(96, 202, 255))
        self._panel_title(painter, rect, "DATA NODE & INTERFACES", QColor(96, 202, 255))
        inner = rect.adjusted(14, 32, -14, -8)
        font = painter.font()
        font.setPixelSize(10)
        painter.setFont(font)
        voice = self._voice_active or self._state in (OrbVisualState.LISTENING, OrbVisualState.SPEAKING)
        chat = True
        statuses = {
            "CHAT": chat,
            "VOICE": voice,
            "FILES": chat,
            "APPS": chat,
            "CALENDAR": chat,
            "WEB": chat,
            "SYSTEM": True,
        }
        y = inner.top() + 2
        for name, _default in _INTERFACE_ROWS:
            active = statuses.get(name, False)
            color = QColor(120, 230, 255, 235) if active else QColor(90, 110, 135, 160)
            painter.setPen(color)
            painter.drawText(QRectF(inner.left(), y, inner.width() * 0.4, 14),
                             Qt.AlignmentFlag.AlignLeft, name)
            # connector line to a node column
            painter.setPen(QPen(color, 1.0))
            line_y = y + 7
            painter.drawLine(QPointF(inner.left() + inner.width() * 0.38, line_y),
                             QPointF(inner.left() + inner.width() * 0.55, line_y))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(color)
            painter.drawEllipse(QPointF(inner.left() + inner.width() * 0.58, line_y), 2.4, 2.4)
            y += 17

    def _draw_automation_panel(self, painter, rect: QRectF) -> None:
        executing = self._state is OrbVisualState.AUTOMATION
        waiting = self._state is OrbVisualState.AUTHORIZATION
        accent = QColor(255, 82, 82) if executing else QColor(120, 60, 66) if waiting else QColor(90, 40, 46)
        fill_top = (52, 8, 14, 80) if executing else (26, 8, 12, 56)
        _draw_holo_panel(painter, rect, accent=accent, fill_top=fill_top,
                         fill_mid=(24, 5, 10, 46), fill_bot=(14, 3, 7, 36))
        title = "AUTOMATIZACIÓN: " + ("RED PROCESS CHAIN" if executing else "EN ESPERA")
        self._panel_title(painter, rect, title, accent)
        inner = rect.adjusted(14, 32, -14, -8)
        if executing:
            red = (255, 82, 82)
            # Task gauge: indeterminate energy ring while the real task runs.
            gauge_rect = QRectF(inner.left() + 4, inner.top() + 6, 64, 64)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(QPen(QColor(70, 16, 22, 200), 6.0))
            painter.drawArc(gauge_rect, 45 * 16, 270 * 16)
            elapsed = time.monotonic() - self._started_at
            sweep = int((elapsed * 240) % 360)
            painter.setPen(QPen(QColor(255, 90, 90, 235), 6.0))
            painter.drawArc(gauge_rect, sweep * 16, -100 * 16)
            painter.setPen(QColor(255, 200, 200, 240))
            font = painter.font()
            font.setPixelSize(13)
            font.setBold(True)
            painter.setFont(font)
            painter.drawText(gauge_rect, Qt.AlignmentFlag.AlignCenter, "RUN")
            label = self._automation_task or "EJECUTANDO TAREA SUPERVISADA"
            painter.setPen(QColor(255, 170, 170, 235))
            font.setPixelSize(10)
            font.setBold(False)
            painter.setFont(font)
            painter.drawText(inner.adjusted(gauge_rect.width() + 16, 8, 0, 0),
                             Qt.TextFlag.TextWordWrap, label)
        else:
            painter.setPen(QColor(180, 110, 120, 170))
            font = painter.font()
            font.setPixelSize(10)
            painter.setFont(font)
            text = "SIN TAREA ACTIVA" if not waiting else "ESPERANDO AUTORIZACIÓN DEL USUARIO"
            if self._last_error:
                text = f"ÚLTIMA SEÑAL: {self._last_error[:60]}"
            painter.drawText(inner, Qt.TextFlag.TextWordWrap, text)

    def _draw_corner_hud(self, painter, with_clock: bool = True) -> None:
        """Atlas brand, clock, status chip and menu hint."""
        w, h = self.width(), self.height()
        margin = max(18.0, w * 0.022)
        # Top-left brand.
        painter.setPen(QColor(190, 245, 255, 245))
        font = painter.font()
        font.setPixelSize(26)
        font.setBold(True)
        font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 6.0)
        painter.setFont(font)
        painter.drawText(QRectF(margin, margin * 0.7, 300, 34), Qt.AlignmentFlag.AlignLeft, "ATLAS")
        font.setPixelSize(10)
        font.setBold(False)
        font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 3.0)
        painter.setFont(font)
        painter.setPen(QColor(120, 200, 240, 200))
        painter.drawText(QRectF(margin, margin * 0.7 + 34, 300, 16), Qt.AlignmentFlag.AlignLeft,
                         "PERSONAL AI OS")
        # Top-right motto column.
        painter.setPen(QColor(120, 200, 240, 190))
        for index, line in enumerate(("APRENDER", "AUTOMATIZAR", "CREAR", "EVOLUCIONAR", "CONTIGO")):
            painter.drawText(QRectF(w - margin - 240, margin * 0.7 + index * 15, 240, 14),
                             Qt.AlignmentFlag.AlignRight, line + "  —")
        if with_clock:
            self._draw_clock_hud(painter)
        # Bottom-right status chip (ESCUCHANDO / state label).
        label, rgb = _STATE_COGNITIVE.get(self._state, ("EN REPOSO", (70, 205, 255)))
        chip_w = 210.0
        chip_rect = QRectF(w - margin - chip_w, h - margin - 34, chip_w, 30)
        painter.setPen(QPen(QColor(*rgb, 160), 1.2))
        painter.setBrush(QColor(4, 16, 34, 160))
        path = QPainterPath()
        path.addRoundedRect(chip_rect, 6, 6)
        painter.drawPath(path)
        painter.setPen(QColor(210, 245, 255, 235))
        font.setPixelSize(11)
        font.setBold(True)
        painter.setFont(font)
        painter.drawText(chip_rect.adjusted(12, 0, -12, 0), Qt.AlignmentFlag.AlignVCenter,
                         label if label != "EN REPOSO" else "LISTO")
        # Bottom-center slogan + menu hint.
        painter.setPen(QColor(120, 200, 240, 150))
        font.setPixelSize(10)
        font.setBold(False)
        font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 2.5)
        painter.setFont(font)
        painter.drawText(QRectF(0, h - margin - 6, w, 14), Qt.AlignmentFlag.AlignCenter,
                         "IDEAS  >  ACCIONES  >  RESULTADOS")

    def _draw_clock_hud(self, painter) -> None:
        """Clock + date (dynamic: updated once per second, drawn per frame)."""
        w, h = self.width(), self.height()
        margin = max(18.0, w * 0.022)
        painter.setPen(QColor(190, 245, 255, 235))
        font = painter.font()
        font.setPixelSize(22)
        font.setBold(True)
        painter.setFont(font)
        painter.drawText(QRectF(margin, h - margin - 52, 220, 28), Qt.AlignmentFlag.AlignLeft,
                         self._clock_text)
        font.setPixelSize(10)
        font.setBold(False)
        painter.setFont(font)
        painter.setPen(QColor(120, 200, 240, 190))
        painter.drawText(QRectF(margin, h - margin - 22, 260, 14), Qt.AlignmentFlag.AlignLeft,
                         self._date_text)


class OrbeStageControls(QWidget):
    """Interactive layer of the fullscreen stage: the ONLY clickable surface.

    SAFETY CONTRACT: the window covers the stage area but a hard
    ``setMask`` keeps every pixel outside the real buttons transparent to
    input, so clicks pass to the desktop/taskbar/other windows. It hosts
    the visible X (top-right) and the bottom capability menu as real
    buttons with hover/pressed feedback. ESC also emits ``hide_requested``.
    """

    chat_requested = Signal()
    voice_requested = Signal()
    quit_requested = Signal()
    capability_selected = Signal(str)
    hide_requested = Signal()

    _CLOSE_BUTTON_SIZE = 36
    _MASK_PADDING = 4

    _BUTTON_STYLE = (
        "QPushButton {"
        "color: rgb(190, 245, 255); background: rgba(8, 30, 60, 150);"
        "border: 1px solid rgba(120, 220, 255, 130); border-radius: 8px;"
        "font-size: 11px; font-weight: 600;"
        "}"
        "QPushButton:hover {"
        "color: white; background: rgba(52, 173, 255, 90);"
        "border: 2px solid rgba(190, 245, 255, 235);"
        "}"
        "QPushButton:pressed { background: rgba(52, 173, 255, 140); }"
    )

    _CLOSE_STYLE = (
        "QPushButton {"
        "color: rgb(190, 245, 255); background: rgba(8, 30, 60, 170);"
        "border: 1px solid rgba(120, 220, 255, 150); border-radius: 6px;"
        "font-size: 15px; font-weight: 700;"
        "}"
        "QPushButton:hover {"
        "color: white; background: rgba(255, 96, 96, 170);"
        "border: 2px solid rgba(255, 170, 170, 240);"
        "}"
        "QPushButton:pressed { background: rgba(255, 96, 96, 230); }"
    )

    def __init__(self, stage=None, parent=None) -> None:
        super().__init__(parent)
        self._stage = stage
        self.setWindowTitle("Atlas Controles")
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)

        self._close_button = QPushButton("✕", self)
        self._close_button.setToolTip("Ocultar Atlas (ESC)")
        self._close_button.setFixedSize(self._CLOSE_BUTTON_SIZE, 30)
        self._close_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self._close_button.setStyleSheet(self._CLOSE_STYLE)
        self._close_button.clicked.connect(self.hide_requested.emit)

        self._menu_buttons: list[QPushButton] = []
        for capability_id, label in _CAPABILITY_OPTIONS:
            button = QPushButton(label, self)
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            button.setStyleSheet(self._BUTTON_STYLE)
            if capability_id == "chat":
                button.clicked.connect(self.chat_requested.emit)
            elif capability_id == "voz":
                button.clicked.connect(self.voice_requested.emit)
            else:
                button.clicked.connect(  # pragma: no branch - closure binding
                    lambda checked=False, cid=capability_id: self.capability_selected.emit(cid)
                )
            self._menu_buttons.append(button)

    # ------------------------------------------------------------------ layout

    def menu_button_rects(self) -> list[QRect]:
        """Same bottom-band geometry the stage composition always used."""
        rect = self.rect()
        count = len(self._menu_buttons)
        height = min(52.0, max(38.0, rect.height() * 0.055))
        width = min(128.0, max(74.0, rect.width() * 0.062))
        gap = 10.0
        total = count * width + (count - 1) * gap
        x = (rect.width() - total) / 2.0
        y = rect.height() - height - max(18.0, rect.height() * 0.03)
        rects: list[QRect] = []
        for _index in range(count):
            rects.append(QRect(int(round(x)), int(round(y)), int(round(width)), int(round(height))))
            x += width + gap
        return rects

    def close_button_rect(self) -> QRect:
        margin = max(18, int(self.width() * 0.022))
        size = self._CLOSE_BUTTON_SIZE
        return QRect(self.width() - margin - size, max(12, margin - 8), size, 30)

    def _relayout(self) -> None:
        for button, rect in zip(self._menu_buttons, self.menu_button_rects()):
            button.setGeometry(rect)
        self._close_button.setGeometry(self.close_button_rect())
        self._apply_input_mask()

    def _apply_input_mask(self) -> None:
        """Only the real buttons receive input; everything else passes to Windows."""
        region = QRegion()
        for button in (self._close_button, *self._menu_buttons):
            region += QRegion(button.geometry().adjusted(
                -self._MASK_PADDING, -self._MASK_PADDING,
                self._MASK_PADDING, self._MASK_PADDING,
            ))
        self.setMask(region)

    # ------------------------------------------------------------------ events

    def mousePressEvent(self, event) -> None:  # noqa: N802 (Qt API)
        super().mousePressEvent(event)

    def showEvent(self, event) -> None:  # noqa: N802 (Qt API)
        stage = self._stage
        if stage is not None:
            # NEVER trust stage.frameGeometry() here: during showEvent the
            # fullscreen geometry may not be final yet, which used to leave
            # the buttons in the wrong place (X / CHAT dead clicks). The
            # screen geometry is the stable fullscreen target.
            screen = stage.screen()
            if screen is None:
                from PySide6.QtGui import QGuiApplication

                screen = QGuiApplication.screenAt(stage.frameGeometry().center())
            if screen is not None:
                self.setGeometry(screen.geometry())
            else:
                self.setGeometry(stage.frameGeometry())
        self._relayout()
        self._apply_noactivate_style()
        super().showEvent(event)
        self.raise_()

    def _apply_noactivate_style(self) -> None:
        """The control layer must never steal the user's foreground."""
        if sys.platform != "win32":
            return
        try:
            from PySide6.QtGui import QGuiApplication

            if QGuiApplication.platformName() == "offscreen":
                return
            import ctypes.wintypes as wintypes

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
            if not current & _WS_EX_NOACTIVATE:
                set_style(hwnd, _GWL_EXSTYLE, current | _WS_EX_NOACTIVATE)
        except Exception:
            pass

    def keyPressEvent(self, event) -> None:  # noqa: N802 (Qt API)
        if event.key() == Qt.Key.Key_Escape:
            self.hide_requested.emit()
            event.accept()
            return
        super().keyPressEvent(event)

    def closeEvent(self, event) -> None:  # noqa: N802 (Qt API)
        # Alt+F4 safety: never leave a zombie overlay; hide instead of close.
        event.ignore()
        self.hide()
