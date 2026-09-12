# -*- coding: utf-8 -*-
"""STAGE V2 AISLADA - renderer por capas (asset-driven).

Copia adaptada de demo/atlas_layered/demo_layered.py (renderer aprobado).
NO toca el Orbe estable (ui/orbe_stage.py / orbe_controller.py /
orbe_app.py / windows_hotkey.py). Sin datos reales. El boton CHAT emite
chat_requested; el controlador decide la accion (mismo flujo que el
overlay legacy). Sin hotkeys propios. Ventana independiente de prueba visual.
Ejecutar:  python -m ui.stage_v2
"""
import os
import sys
import math
import random
from datetime import datetime

from PySide6.QtCore import Qt, QTimer, QElapsedTimer, QPointF, QRectF, Signal
from PySide6.QtGui import (QPainter, QPixmap, QImage, QColor, QPen, QBrush,
                           QFont, QRadialGradient, QLinearGradient,
                           QPainterPath, QTransform)
from PySide6.QtWidgets import QApplication
from PySide6.QtOpenGLWidgets import QOpenGLWidget

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(HERE, "assets")

W, H = 1280, 800
CORE_CX, CORE_CY = W * 0.5, H * 0.40
CORE_W = 680.0  # ancho del nucleo en pantalla

random.seed(7)

CYAN = (90, 200, 255)
CYAN_HI = (160, 235, 255)

# ------------------------------------------------------- estado (integracion) ---
# Campos de UI preparados para la integracion: los valores "--" / "UNKNOWN"
# significan explicitamente dato aun NO conectado. Nada inventado.
system_state = {
    "CPU": "--",
    "RAM": "--",
    "SISTEMA": "--",
    "ATLAS": "DEMO",
    "MODELO": "--",
    "ESTADO": "IDLE",
}
atlas_state = {
    "ACTIVE TASK": "--",
    "ACTIVE AGENT": "--",
    "MODEL": "--",
    "AUTHORIZATION": "--",
}
CAP_NAMES = ["CHAT", "VOICE", "FILES", "CALENDAR", "WEB", "SYSTEM",
             "AGENT", "WORKER"]
capability_state = {name: "UNKNOWN" for name in CAP_NAMES}
MENU_ITEMS = ["CHAT", "VOZ", "PC", "ENTRENAMIENTO", "NUTRICIÓN",
              "SALUD", "CÓDIGO", "PROYECTOS", "HERRAMIENTAS"]

# ---------------------------------------------------------------- assets ---
def _pixmap_from_image(img):
    return QPixmap.fromImage(img)


def build_assets():
    """Genera una sola vez todos los assets derivados (cache en disco/mem)."""
    os.makedirs(ASSETS, exist_ok=True)
    global _ICON_GLOW
    _ICON_GLOW = radial_sprite(90, QColor(110, 215, 255, 130), 2)
    A = {}

    # --- nucleo nuevo: volumen construido por luz (sin globo solido) ---
    CS = 640
    body = core_body_sprite(CS)
    A["core"] = _pixmap_from_image(body)
    A["core_energy"] = core_energy_sprite(320)
    A["fragments"] = fragments_sprite(CS, seed=11)
    A["fragments2"] = fragments_sprite(int(CS * 0.92), seed=29)
    A["mesh"] = mesh_sprite(int(CS * 0.98), seed=8)
    A["rim_arcs"] = rim_arcs_sprite(int(CS * 1.04), seed=5)
    A["micro_back"] = micro_back_sprite(CS, seed=23)
    A["inner_arcs"] = inner_arcs_sprite(int(CS * 0.80), seed=31)

    # scanlines holograficas sobre el nucleo (una vez)
    sl = QImage(CS, CS, QImage.Format_ARGB32_Premultiplied)
    sl.fill(Qt.transparent)
    p = QPainter(sl)
    p.setPen(QPen(QColor(150, 230, 255, 16), 1))
    for y in range(0, sl.height(), 4):
        p.drawLine(0, y, sl.width(), y)
    p.end()
    feather(sl, 0.10)
    A["scanlines"] = _pixmap_from_image(sl)

    # --- tintes por estado (cacheados, SourceAtop conserva alpha/sombreado) --
    tints = {
        "idle":       None,
        "listening":  QColor(80, 200, 255, 46),
        "thinking":   QColor(120, 230, 255, 34),
        "speaking":   QColor(60, 190, 255, 30),
        "automation": QColor(6, 8, 26, 130),
    }
    for name, col in tints.items():
        img = body.copy()
        if col is not None:
            p = QPainter(img)
            p.setCompositionMode(QPainter.CompositionMode_SourceAtop)
            p.fillRect(img.rect(), col)
            p.end()
        if name == "automation":  # acento rojo localizado
            p = QPainter(img)
            p.setCompositionMode(QPainter.CompositionMode_SourceAtop)
            g = QRadialGradient(img.width() * 0.5, img.height() * 0.5,
                                img.width() * 0.55)
            g.setColorAt(0.55, QColor(0, 0, 0, 0))
            g.setColorAt(1.0, QColor(255, 40, 60, 70))
            p.fillRect(img.rect(), QBrush(g))
            p.end()
        A["core_" + name] = _pixmap_from_image(img)

    # --- simbolo Atlas/A integrado en el volumen (procedural, por capas) ---
    G = 512
    A["glyph_halo"] = glyph_halo_sprite(G)
    A["glyph_fill"] = glyph_fill_sprite(G)
    A["glyph_edge"] = glyph_edge_sprite(G)
    A["glyph_back"] = glyph_edge_sprite(G, dim=True)
    A["glyph_energy"] = glyph_energy_sprite(int(G * 1.4))
    # pozo de contraste suave detras del simbolo (mucho mas leve: sin look
    # pegatina, solo profundidad local)
    A["glyph_backdrop"] = radial_sprite(360, QColor(2, 8, 20, 140), 2)

    # --- glow ambiental grande (sprite radial, tamano final 1:1) ---
    A["glow_big"] = radial_sprite(1180, QColor(0, 120, 255, 60), 3)
    A["glow_small"] = radial_sprite(300, QColor(120, 220, 255, 110), 2)
    A["glow_deep"] = radial_sprite(900, QColor(0, 60, 160, 90), 2)

    # --- anillos traseros (finos, con ticks, planos distintos) ---
    # los arcos exteriores gigantes: muy fragmentados, gaps grandes y tenues
    A["ring_back_a"] = dash_ring_sprite(600, 214, 4, segs=8, gap=26, seed=2)
    A["ring_back_b"] = ring_sprite(724, 118, 4, ticks=40, seed=7)
    A["ring_back_c"] = dash_ring_sprite(520, 300, 3, segs=9, gap=10, seed=13)
    # --- anillos delanteros (discontinuos + fino continuo) ---
    A["ring_front_a"] = dash_ring_sprite(668, 246, 4, segs=9, gap=28, seed=3)
    A["ring_front_b"] = dash_ring_sprite(560, 130, 3, segs=7, gap=22, seed=17)
    A["ring_front_c"] = ring_sprite(700, 196, 3, ticks=32, seed=21)

    # --- haz de barrido ---
    A["beam"] = beam_sprite(140, 760)

    # --- haz vertical detras del nucleo ---
    A["vbeam"] = vbeam_sprite(260, 820)

    # --- particula (bokeh dot) ---
    A["dot"] = radial_sprite(18, QColor(150, 230, 255, 255), 2, hard=True)

    # --- halos superior / inferior ---
    A["halo_top"] = halo_ellipse_sprite(560, 110, QColor(110, 215, 255))
    A["halo_bottom"] = halo_ellipse_sprite(860, 180, QColor(60, 170, 255))

    # --- pedestal holografico + anillos base ---
    A["pedestal"] = pedestal_sprite(720, 300)
    A["base_ring_a"] = halo_ellipse_sprite(900, 210, QColor(70, 190, 255), ring=True)
    A["base_ring_b"] = halo_ellipse_sprite(700, 150, QColor(90, 205, 255), ring=True)

    # --- microdatos / lineas finas de fondo ---
    A["bg_data"] = bg_data_sprite(W, H)
    A["chip"] = chip_sprite()

    # --- corona superior: anillos holograficos horizontales ---
    A["crown_halo"] = halo_ellipse_sprite(760, 170, QColor(60, 170, 255))
    A["crown_a"] = dash_ring_sprite(600, 66, 3, segs=10, gap=13, seed=41)
    A["crown_b"] = dash_ring_sprite(440, 44, 2, segs=8, gap=9, seed=43)
    A["crown_c"] = ring_sprite(680, 88, 2, ticks=36, seed=47)
    A["crown_beam"] = vbeam_sprite(90, 300)

    # --- marcos HUD inferiores (reloj + voz) ---
    A["clock_frame"] = frame_sprite(198, 58, QColor(60, 200, 255))
    A["voice_frame"] = frame_sprite(292, 54, QColor(60, 200, 255))
    A["voice_frame_red"] = frame_sprite(292, 54, QColor(255, 60, 80))

    # --- identidad (cabeceras) ---
    A["hdr_left"] = hdr_left_sprite()
    A["hdr_right"] = hdr_right_sprite()

    # --- HUD reservado: paneles + tira de iconos (cacheados) ---
    # Los paneles se construyen desde los dicts de estado (integracion
    # posterior: basta regenerar el sprite al cambiar el dict).
    A["panel_left"] = system_panel_sprite(
        312, 196, "SYSTEM CORE STATUS", QColor(60, 200, 255),
        footer="// TELEMETRY: WAITING BACKEND")
    A["panel_right"] = capability_panel_sprite(
        312, 238, "DATA NODE & INTERFACES", QColor(60, 200, 255),
        footer="// CAPABILITY STATES")
    A["panel_activity"] = fields_panel_sprite(
        300, 150, "ACTIVITY", atlas_state, QColor(60, 200, 255),
        footer="// ATLAS CORE")
    A["panel_activity_red"] = fields_panel_sprite(
        300, 150, "ACTIVITY", atlas_state,
        QColor(255, 60, 80), footer="// ATLAS CORE")
    A["panel_auto"] = fields_panel_sprite(
        312, 150, "AUTOMATION: PROCESS CHAIN", atlas_state,
        QColor(255, 60, 80), footer="// LOCAL SIMULATION")
    icons = [icon_pixmap(fn, 34, bright=False) for fn in ICON_DRAWERS]
    icons_hi = [icon_pixmap(fn, 34, bright=True) for fn in ICON_DRAWERS]
    A["menu_icons"] = icons
    A["menu_icons_hi"] = icons_hi
    A["hud_bottom"] = bottom_hud_sprite(W, 130, icons)

    # --- viñeta de fondo ---
    A["vignette"] = vignette_sprite(W, H)
    return A


def feather(img, frac):
    """Fundido suave de alpha hacia los 4 bordes (evita costuras del recorte)."""
    w, h = img.width(), img.height()
    p = QPainter(img)
    p.setCompositionMode(QPainter.CompositionMode_DestinationIn)
    g = QLinearGradient(0, 0, w, 0)
    a = int(frac * 255 * 0.5)
    g.setColorAt(0.0, QColor(0, 0, 0, 0))
    g.setColorAt(frac, QColor(0, 0, 0, 255))
    g.setColorAt(1.0 - frac, QColor(0, 0, 0, 255))
    g.setColorAt(1.0, QColor(0, 0, 0, 0))
    p.fillRect(img.rect(), QBrush(g))
    g2 = QLinearGradient(0, 0, 0, h)
    g2.setColorAt(0.0, QColor(0, 0, 0, 0))
    g2.setColorAt(frac, QColor(0, 0, 0, 255))
    g2.setColorAt(1.0 - frac, QColor(0, 0, 0, 255))
    g2.setColorAt(1.0, QColor(0, 0, 0, 0))
    p.fillRect(img.rect(), QBrush(g2))
    p.end()


def radial_sprite(size, color, steps, hard=False):
    img = QImage(size, size, QImage.Format_ARGB32)
    img.fill(Qt.transparent)
    p = QPainter(img)
    g = QRadialGradient(size / 2, size / 2, size / 2)
    c = QColor(color)
    c.setAlpha(color.alpha())
    g.setColorAt(0.0, QColor(c.red(), c.green(), c.blue(), c.alpha()))
    g.setColorAt(0.35 if not hard else 0.5,
                 QColor(c.red(), c.green(), c.blue(), int(c.alpha() * 0.45)))
    g.setColorAt(1.0, QColor(c.red(), c.green(), c.blue(), 0))
    p.fillRect(img.rect(), QBrush(g))
    p.end()
    return _pixmap_from_image(img)


def core_body_sprite(size):
    """Masa oscura luminosa: centro casi negro con respiracion azul tenue,
    bordes translucidos (sin borde circular solido)."""
    img = QImage(size, size, QImage.Format_ARGB32)
    img.fill(Qt.transparent)
    p = QPainter(img)
    g = QRadialGradient(size / 2, size / 2, size * 0.5)
    g.setColorAt(0.0, QColor(16, 36, 72, 250))
    g.setColorAt(0.45, QColor(20, 48, 96, 218))
    g.setColorAt(0.70, QColor(24, 64, 126, 135))
    g.setColorAt(0.88, QColor(28, 80, 156, 52))
    g.setColorAt(1.0, QColor(24, 72, 140, 0))
    p.setBrush(QBrush(g))
    p.setPen(Qt.NoPen)
    p.drawEllipse(img.rect())
    # luz interior difusa arriba-izquierda (volumen por luz, no por borde)
    g2 = QRadialGradient(size * 0.44, size * 0.36, size * 0.36)
    g2.setColorAt(0.0, QColor(90, 175, 240, 82))
    g2.setColorAt(0.55, QColor(55, 115, 195, 34))
    g2.setColorAt(1.0, QColor(0, 40, 90, 0))
    p.setBrush(QBrush(g2))
    p.drawEllipse(img.rect())
    # nube de bruma irregular (rompe la lectura de esfera perfecta)
    rnd = random.Random(3)
    p.setCompositionMode(QPainter.CompositionMode_Plus)
    for i in range(15):
        a = rnd.uniform(0, 6.28)
        r = rnd.uniform(0.10, 0.40) * size
        bx = size / 2 + math.cos(a) * r * 0.6
        by = size / 2 + math.sin(a) * r * 0.55
        s = rnd.uniform(0.16, 0.34) * size
        ga = QRadialGradient(bx, by, s / 2)
        c = QColor(70, 155, 235, rnd.randint(24, 50))
        ga.setColorAt(0.0, c)
        ga.setColorAt(1.0, QColor(0, 0, 0, 0))
        p.setBrush(QBrush(ga))
        p.drawEllipse(QRectF(bx - s / 2, by - s / 2, s, s))
    p.end()
    return img


def core_energy_sprite(size):
    """Corazon de energia: centro mas oscuro (la A debe leerse) y energia
    cian concentrada en anillo alrededor/periferia (uso aditivo)."""
    img = QImage(size, size, QImage.Format_ARGB32)
    img.fill(Qt.transparent)
    p = QPainter(img)
    g = QRadialGradient(size / 2, size / 2, size / 2)
    g.setColorAt(0.0, QColor(120, 195, 235, 128))
    g.setColorAt(0.18, QColor(100, 195, 250, 148))
    g.setColorAt(0.36, QColor(85, 190, 255, 168))
    g.setColorAt(0.58, QColor(65, 170, 255, 92))
    g.setColorAt(1.0, QColor(30, 120, 220, 0))
    p.setBrush(QBrush(g))
    p.setPen(Qt.NoPen)
    p.drawEllipse(img.rect())
    p.end()
    return _pixmap_from_image(img)


def fragments_sprite(size, seed=1):
    """Fragmentos digitales: lineas, esquirlas y puntos en una envolvente
    irregular (sugieren volumen por datos, no por superficie)."""
    img = QImage(size, size, QImage.Format_ARGB32)
    img.fill(Qt.transparent)
    p = QPainter(img)
    p.setRenderHint(QPainter.Antialiasing)
    rnd = random.Random(seed)
    c = size / 2
    for i in range(310):
        th = rnd.uniform(0, 6.28)
        rr = min(0.52, max(0.24, rnd.gauss(0.40, 0.075)))
        depth = 0.5 + 0.5 * math.sin(th)  # frente mas brillante
        x = c + math.cos(th) * rr * c
        y = c + math.sin(th) * rr * c * 0.92
        base = int(34 + 96 * depth + rnd.uniform(0, 30))
        col = QColor(int(120 + 65 * depth), int(200 + 40 * depth), 255, base)
        kind = rnd.random()
        if kind < 0.42:  # linea / esquirla
            ln = rnd.uniform(2.0, 9.0) * (0.5 + depth)
            ang = th + rnd.uniform(-0.5, 0.5)
            p.setPen(QPen(col, rnd.choice((1.0, 1.0, 1.4))))
            p.drawLine(QPointF(x, y),
                       QPointF(x + math.cos(ang) * ln,
                               y + math.sin(ang) * ln * 0.7))
        elif kind < 0.72:  # punto
            s = rnd.uniform(1.4, 3.4)
            p.setPen(Qt.NoPen)
            p.setBrush(col)
            p.drawEllipse(QRectF(x - s / 2, y - s / 2, s, s))
        else:  # micro-rectangulo (dato)
            w2 = rnd.uniform(3.0, 9.0)
            h2 = rnd.uniform(1.6, 4.0)
            p.setPen(QPen(col, 1.0))
            p.setBrush(Qt.NoBrush)
            p.drawRect(QRectF(x - w2 / 2, y - h2 / 2, w2, h2))
    p.end()
    return _pixmap_from_image(img)


def mesh_sprite(size, seed=8):
    """Estructura interna: arcos geodesicos parciales + nodos conectados."""
    img = QImage(size, size, QImage.Format_ARGB32)
    img.fill(Qt.transparent)
    p = QPainter(img)
    p.setRenderHint(QPainter.Antialiasing)
    rnd = random.Random(seed)
    c = size / 2
    R = size * 0.44
    # arcos geodesicos parciales (elipses con distinta inclinacion)
    for i in range(12):
        rx = R * rnd.uniform(0.45, 1.0)
        ry = R * rnd.uniform(0.14, 0.62)
        start = rnd.uniform(0, 360)
        span = rnd.uniform(60, 170)
        al = int(45 + 70 * rnd.random())
        pen = QPen(QColor(110, 210, 255, al), rnd.choice((1.0, 1.2, 1.6)))
        pen.setCapStyle(Qt.RoundCap)
        p.setPen(pen)
        p.setBrush(Qt.NoBrush)
        p.drawArc(QRectF(c - rx, c - ry, 2 * rx, 2 * ry),
                  int(start * 16), int(span * 16))
    # nodos + conexiones
    nodes = []
    for i in range(34):
        th = rnd.uniform(0, 6.28)
        rr = min(0.50, abs(rnd.gauss(0.34, 0.10)))
        nodes.append((c + math.cos(th) * rr * c,
                      c + math.sin(th) * rr * c * 0.92))
    p.setPen(QPen(QColor(100, 205, 255, 0), 1))
    for i in range(30):
        a, b = rnd.choice(nodes), rnd.choice(nodes)
        d = math.hypot(a[0] - b[0], a[1] - b[1])
        if d > size * 0.34:
            continue
        pen = QPen(QColor(105, 210, 255, rnd.randint(35, 80)), 1)
        p.setPen(pen)
        p.drawLine(QPointF(a[0], a[1]), QPointF(b[0], b[1]))
    p.setPen(Qt.NoPen)
    for (x, y) in nodes:
        s = rnd.uniform(1.6, 3.2)
        p.setBrush(QColor(160, 235, 255, rnd.randint(110, 200)))
        p.drawEllipse(QRectF(x - s / 2, y - s / 2, s, s))
    p.end()
    return _pixmap_from_image(img)


def rim_arcs_sprite(size, seed=5):
    """Arcos de borde discontinuos: sugieren envolvente sin cerrar el circulo."""
    img = QImage(size, size, QImage.Format_ARGB32)
    img.fill(Qt.transparent)
    p = QPainter(img)
    p.setRenderHint(QPainter.Antialiasing)
    rnd = random.Random(seed)
    c = size / 2
    R = size * 0.455
    arcs = []
    for i in range(14):
        start = rnd.uniform(0, 6.28)
        ln = rnd.uniform(0.18, 0.72)
        if i < 4:
            arcs.append((start, ln, 7.0, 34))    # glow difuso
        elif i < 10:
            arcs.append((start, ln, 1.8, 150 + rnd.randint(0, 50)))
        else:
            arcs.append((start, ln, 3.0, 80))
    for (start, ln, wd, al) in arcs:
        rect = QRectF(c - R, c - R, 2 * R, 2 * R)
        pen = QPen(QColor(110, 215, 255, al), wd)
        pen.setCapStyle(Qt.RoundCap)
        p.setPen(pen)
        p.setBrush(Qt.NoBrush)
        p.drawArc(rect, int(start * 180 / math.pi * 16),
                  int(ln * 180 / math.pi * 16))
    # nodos brillantes en extremos de algunos arcos
    p.setPen(Qt.NoPen)
    for (start, ln, wd, al) in arcs[4:10]:
        for a in (start, start + ln):
            x = c + math.cos(a) * R
            y = c + math.sin(a) * R
            p.setBrush(QColor(190, 240, 255, 180))
            p.drawEllipse(QRectF(x - 2.2, y - 2.2, 4.4, 4.4))
    p.end()
    return _pixmap_from_image(img)


def micro_back_sprite(size, seed=23):
    """Plano TRASERO del nucleo: microdatos, nodos pequenos y lineas tenues."""
    img = QImage(size, size, QImage.Format_ARGB32)
    img.fill(Qt.transparent)
    p = QPainter(img)
    p.setRenderHint(QPainter.Antialiasing)
    rnd = random.Random(seed)
    c = size / 2
    p.setFont(QFont("Consolas", 6))
    for i in range(90):
        th = rnd.uniform(0, 6.28)
        rr = rnd.uniform(0.10, 0.46)
        x = c + math.cos(th) * rr * c
        y = c + math.sin(th) * rr * c * 0.92
        al = int(16 + 44 * (1.0 - rr / 0.5) * rnd.random())
        col = QColor(110, 205, 255, al)
        kind = rnd.random()
        if kind < 0.30:
            ln = rnd.uniform(3.0, 12.0)
            p.setPen(QPen(col, 1))
            p.drawLine(QPointF(x, y), QPointF(x + ln, y))
            p.drawText(QPointF(x + ln + 2, y + 3),
                       "%02X" % rnd.randint(0, 0xFF))
        elif kind < 0.75:
            s = rnd.uniform(1.0, 2.2)
            p.setPen(Qt.NoPen)
            p.setBrush(col)
            p.drawEllipse(QRectF(x - s / 2, y - s / 2, s, s))
        else:
            w2 = rnd.uniform(2.5, 6.0)
            p.setPen(QPen(col, 1))
            p.setBrush(Qt.NoBrush)
            p.drawRect(QRectF(x - w2 / 2, y - 1, w2, 2))
    p.end()
    return _pixmap_from_image(img)


def inner_arcs_sprite(size, seed=31):
    """Arcos de energia internos parciales (plano medio, uso aditivo)."""
    img = QImage(size, size, QImage.Format_ARGB32)
    img.fill(Qt.transparent)
    p = QPainter(img)
    p.setRenderHint(QPainter.Antialiasing)
    rnd = random.Random(seed)
    c = size / 2
    for i in range(9):
        rx = c * rnd.uniform(0.18, 0.46)
        ry = rx * rnd.uniform(0.30, 0.85)
        start = rnd.uniform(0, 360)
        span = rnd.uniform(40, 130)
        al = int(50 + 70 * rnd.random())
        pen = QPen(QColor(130, 220, 255, al), rnd.choice((1.4, 1.8, 2.4)))
        pen.setCapStyle(Qt.RoundCap)
        p.setPen(pen)
        p.setBrush(Qt.NoBrush)
        p.drawArc(QRectF(c - rx, c - ry, 2 * rx, 2 * ry),
                  int(start * 16), int(span * 16))
        # nodo brillante en un extremo del arco
        a0 = math.radians(start)
        x = c + rx * math.cos(a0)
        y = c + ry * math.sin(a0)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(200, 245, 255, 190))
        p.drawEllipse(QRectF(x - 2.0, y - 2.0, 4.0, 4.0))
    p.end()
    return _pixmap_from_image(img)


def ring_sprite(rw, rh, thick=6, ticks=48, seed=2):
    """Elipse cacheada en segmentos con zonas intensas y gaps + ticks."""
    m = 60
    w, h = rw + 2 * m, rh + 2 * m
    img = QImage(w, h, QImage.Format_ARGB32)
    img.fill(Qt.transparent)
    p = QPainter(img)
    p.setRenderHint(QPainter.Antialiasing)
    rect = QRectF(m, m, rw, rh)
    cy = QColor(90, 200, 255)
    rnd = random.Random(seed)
    nseg = 20
    seg = 360 * 16 / nseg
    for i in range(nseg):
        inten = max(0.0, math.sin(i * 1.1 + seed * 1.7)) * \
            rnd.uniform(0.45, 1.0)
        if inten < 0.10:          # gaps: tramos casi invisibles
            continue
        al = int(18 + 175 * inten)
        for wid, mul in ((thick * 3.2, 0.16), (thick * 1.9, 0.4),
                         (thick * 0.6, 1.0)):
            pen = QPen(QColor(cy.red(), cy.green(), cy.blue(),
                              int(al * mul)), wid)
            pen.setCapStyle(Qt.RoundCap)
            p.setPen(pen)
            p.setBrush(Qt.NoBrush)
            p.drawArc(rect, int(i * seg + 90), int(seg - 180))
    # ticks tecnicos
    p.setPen(QPen(QColor(140, 230, 255, 110), 1.5))
    for i in range(ticks):
        t = i / ticks * 2 * math.pi
        x1 = w / 2 + rw / 2 * math.cos(t)
        y1 = h / 2 + rh / 2 * math.sin(t)
        x2 = w / 2 + (rw / 2 + 9) * math.cos(t)
        y2 = h / 2 + (rh / 2 + 9) * math.sin(t)
        if i % 4 == 0:
            p.drawLine(QPointF(x1, y1), QPointF(x2, y2))
    # micro-nodos
    p.setPen(Qt.NoPen)
    p.setBrush(QColor(170, 235, 255, 160))
    for i in range(0, ticks, max(4, ticks // 6)):
        t = i / ticks * 2 * math.pi
        x = w / 2 + rw / 2 * math.cos(t)
        y = h / 2 + rh / 2 * math.sin(t)
        p.drawEllipse(QRectF(x - 2, y - 2, 4, 4))
    p.end()
    return _pixmap_from_image(img)


def dash_ring_sprite(rw, rh, thick=4, segs=10, gap=12, seed=1):
    """Elipse discontinua cacheada: segmentos irregulares con gaps y
    zonas de intensidad variable + glow fino."""
    m = 60
    w, h = rw + 2 * m, rh + 2 * m
    img = QImage(w, h, QImage.Format_ARGB32)
    img.fill(Qt.transparent)
    p = QPainter(img)
    p.setRenderHint(QPainter.Antialiasing)
    rect = QRectF(m, m, rw, rh)
    total = 360 * 16
    seg = total / segs
    gapi = gap * 16
    rnd = random.Random(seed)
    for i in range(segs):
        inten = max(0.0, math.sin(i * 1.35 + seed * 2.3)) * \
            rnd.uniform(0.4, 1.0)
        if inten < 0.12:          # zona casi invisible
            continue
        span = int((seg - 2 * gapi) * rnd.uniform(0.45, 1.0))
        off = int(i * seg + gapi + rnd.uniform(0, 0.25) * (seg - 2 * gapi))
        # glow externo difuso
        pen = QPen(QColor(60, 170, 255, int(26 * inten)), thick * 3.4)
        pen.setCapStyle(Qt.FlatCap)
        p.setPen(pen)
        p.drawArc(rect, off, span)
        # trazo principal fino
        pen = QPen(QColor(120, 215, 255, int(70 + 110 * inten)), thick * 0.8)
        pen.setCapStyle(Qt.RoundCap)
        p.setPen(pen)
        p.drawArc(rect, off, span)
        # trazo interior muy fino brillante
        pen = QPen(QColor(190, 240, 255, int(40 + 60 * inten)),
                   max(1, thick * 0.3))
        pen.setCapStyle(Qt.RoundCap)
        p.setPen(pen)
        p.drawArc(rect, off, span)
    p.end()
    return _pixmap_from_image(img)


def halo_ellipse_sprite(rw, rh, color, ring=False):
    """Halo eliptico difuso (o anillo base) para composicion luminosa."""
    m = 60
    w, h = rw + 2 * m, rh + 2 * m
    img = QImage(w, h, QImage.Format_ARGB32)
    img.fill(Qt.transparent)
    p = QPainter(img)
    p.setRenderHint(QPainter.Antialiasing)
    rect = QRectF(m, m, rw, rh)
    if ring:
        for wid, al in ((10, 18), (5, 40), (1.6, 130)):
            p.setPen(QPen(QColor(color.red(), color.green(), color.blue(), al), wid))
            p.setBrush(Qt.NoBrush)
            p.drawEllipse(rect)
    else:
        g = QRadialGradient(w / 2, h / 2, max(rw, rh) / 2)
        g.setColorAt(0.0, QColor(color.red(), color.green(), color.blue(), 0))
        g.setColorAt(0.62, QColor(color.red(), color.green(), color.blue(),
                                  int(color.alpha() * 0.55)))
        g.setColorAt(0.78, QColor(color.red(), color.green(), color.blue(),
                                  int(color.alpha() * 0.8)))
        g.setColorAt(1.0, QColor(color.red(), color.green(), color.blue(), 0))
        p.setBrush(QBrush(g))
        p.setPen(Qt.NoPen)
        p.drawEllipse(rect)
    p.end()
    return _pixmap_from_image(img)


def pedestal_sprite(w, h):
    """Plataforma tecnologica de la referencia: base eliptica con varios
    anillos concéntricos segmentados, profundidad, glow y haz de proyeccion
    hacia el nucleo (el nucleo parece proyectado desde la base)."""
    img = QImage(w, h, QImage.Format_ARGB32)
    img.fill(Qt.transparent)
    p = QPainter(img)
    p.setRenderHint(QPainter.Antialiasing)
    cx = w / 2
    yc = h * 0.74  # centro del disco superior
    p.setPen(Qt.NoPen)

    # 1) haz de proyeccion hacia arriba (base -> nucleo)
    cone = [QPointF(cx - w * 0.16, yc - h * 0.04),
            QPointF(cx + w * 0.16, yc - h * 0.04),
            QPointF(cx + w * 0.36, h * 0.02),
            QPointF(cx - w * 0.36, h * 0.02)]
    gc = QLinearGradient(0, yc, 0, h * 0.02)
    gc.setColorAt(0.0, QColor(70, 185, 255, 52))
    gc.setColorAt(0.5, QColor(60, 175, 255, 22))
    gc.setColorAt(1.0, QColor(50, 160, 255, 0))
    p.setBrush(QBrush(gc))
    p.drawPolygon(cone)

    # 2) glow ambiental bajo la base
    gg = QRadialGradient(cx, yc + h * 0.10, w * 0.44)
    gg.setColorAt(0.0, QColor(45, 155, 255, 70))
    gg.setColorAt(0.55, QColor(35, 125, 240, 30))
    gg.setColorAt(1.0, QColor(20, 80, 180, 0))
    p.setBrush(QBrush(gg))
    p.drawEllipse(QRectF(cx - w * 0.44, yc - h * 0.16,
                         w * 0.88, h * 0.52))

    # 3) reflejo luminoso ascendente bajo el nucleo
    gr = QLinearGradient(0, yc - h * 0.18, 0, yc)
    gr.setColorAt(0.0, QColor(20, 80, 180, 0))
    gr.setColorAt(1.0, QColor(60, 180, 255, 60))
    p.setBrush(QBrush(gr))
    p.drawPolygon([QPointF(cx - w * 0.30, yc - h * 0.18),
                   QPointF(cx + w * 0.30, yc - h * 0.18),
                   QPointF(cx + w * 0.19, yc - h * 0.02),
                   QPointF(cx - w * 0.19, yc - h * 0.02)])

    # 4) discos apilados (profundidad fisica)
    discs = ((0.34, 0.075, 0.02, 0.62, 190),   # disco superior (alpha alto)
             (0.44, 0.105, 0.055, 0.40, 120),
             (0.56, 0.145, 0.10, 0.26, 70),
             (0.70, 0.19, 0.155, 0.16, 42))
    for ew, eh, dy, fa, ra in discs:
        ey = yc + h * dy
        # cuerpo del disco
        gd = QLinearGradient(0, ey - h * eh / 2, 0, ey + h * eh / 2)
        gd.setColorAt(0.0, QColor(14, 46, 92, int(120 * fa * 2.2)))
        gd.setColorAt(1.0, QColor(6, 22, 48, int(70 * fa * 2.2)))
        p.setBrush(QBrush(gd))
        p.drawEllipse(QRectF(cx - w * ew / 2, ey - h * eh / 2,
                             w * ew, h * eh))
        # borde brillante superior (semiarco frontal)
        p.setBrush(Qt.NoBrush)
        p.setPen(QPen(QColor(150, 228, 255, ra), 1.8))
        p.drawArc(QRectF(cx - w * ew / 2, ey - h * eh / 2,
                         w * ew, h * eh), 200 * 16, 140 * 16)
        p.setPen(QPen(QColor(90, 200, 255, int(ra * 0.55)), 0.9))
        p.drawEllipse(QRectF(cx - w * ew / 2, ey - h * eh / 2,
                             w * ew, h * eh))

    # 5) anillos concéntricos segmentados sobre el disco superior
    ry0 = yc - h * 0.02
    for rw, rh, seed, segs, gap, al in ((w * 0.30, h * 0.062, 61, 9, 7, 165),
                                        (w * 0.38, h * 0.085, 62, 11, 9, 110),
                                        (w * 0.48, h * 0.115, 63, 8, 12, 70),
                                        (w * 0.60, h * 0.15, 64, 10, 16, 44)):
        rnd = random.Random(seed)
        rect = QRectF(cx - rw / 2, ry0 - rh / 2, rw, rh)
        for i in range(segs):
            inten = max(0.0, math.sin(i * 1.6 + seed)) * rnd.uniform(0.5, 1.0)
            if inten < 0.12:
                continue
            span = int(360 * 16 / segs * rnd.uniform(0.4, 0.85))
            off = int(i * 360 * 16 / segs + rnd.uniform(0, 40) * 16)
            for wd, mul in ((3.4, 0.18), (1.0, 0.55), (0.4, 1.0)):
                p.setPen(QPen(QColor(120, 218, 255, int(al * inten * mul)), wd))
                p.drawArc(rect, off + 90 * 16, span)
    # anillo continuo exterior muy fino
    p.setPen(QPen(QColor(80, 190, 255, 34), 1))
    p.drawEllipse(QRectF(cx - w * 0.66 / 2, ry0 - h * 0.17 / 2,
                         w * 0.66, h * 0.17))

    # 6) marcas radiales + micro nodos sobre el borde del disco superior
    p.setPen(QPen(QColor(150, 228, 255, 100), 1.2))
    for i in range(28):
        a = i / 28 * 2 * math.pi
        r1, r2 = w * 0.175, w * 0.205
        x1 = cx + r1 * math.cos(a)
        y1 = yc + h * 0.062 * math.sin(a)
        x2 = cx + r2 * math.cos(a)
        y2 = yc + h * 0.075 * math.sin(a)
        if i % 2 == 0:
            p.drawLine(QPointF(x1, y1), QPointF(x2, y2))
    p.setPen(Qt.NoPen)
    for i in range(6):
        a = i / 6 * 2 * math.pi + 0.3
        x = cx + w * 0.205 * math.cos(a)
        y = yc + h * 0.075 * math.sin(a)
        p.setBrush(QColor(180, 240, 255, 170))
        p.drawEllipse(QRectF(x - 1.6, y - 1.6, 3.2, 3.2))

    # 7) segmentos luminosos independientes (parpadeo estatico en sprite)
    for k in range(10):
        a0 = k * 36 + 8
        rect = QRectF(cx - w * 0.335 / 2, ry0 - h * 0.07 / 2,
                      w * 0.335, h * 0.07)
        al = 70 + int(90 * abs(math.sin(k * 1.9 + 1.2)))
        p.setPen(QPen(QColor(160, 235, 255, al), 2.2))
        p.drawArc(rect, int(a0 * 16), int(13 * 16))
    p.end()
    return _pixmap_from_image(img)


def vbeam_sprite(w, h):
    """Haz vertical suave detras del nucleo."""
    img = QImage(w, h, QImage.Format_ARGB32)
    img.fill(Qt.transparent)
    p = QPainter(img)
    gh = QLinearGradient(0, 0, 0, h)
    gh.setColorAt(0.0, QColor(80, 190, 255, 0))
    gh.setColorAt(0.35, QColor(90, 200, 255, 40))
    gh.setColorAt(1.0, QColor(60, 160, 255, 0))
    p.fillRect(img.rect(), QBrush(gh))
    gw = QLinearGradient(0, 0, w, 0)
    gw.setColorAt(0.0, QColor(0, 0, 0, 0))
    gw.setColorAt(0.5, QColor(255, 255, 255, 255))
    gw.setColorAt(1.0, QColor(0, 0, 0, 0))
    p.setCompositionMode(QPainter.CompositionMode_DestinationIn)
    p.fillRect(img.rect(), QBrush(gw))
    p.end()
    return _pixmap_from_image(img)


def bg_data_sprite(w, h):
    """Microdatos de fondo: lineas finas, ticks y fragmentos (estaticos)."""
    img = QImage(w, h, QImage.Format_ARGB32)
    img.fill(Qt.transparent)
    p = QPainter(img)
    p.setRenderHint(QPainter.Antialiasing)
    f = QFont("Consolas", 7)
    p.setFont(f)
    rnd = random.Random(42)
    for i in range(26):
        x, y = rnd.uniform(20, w - 120), rnd.uniform(20, h - 40)
        al = rnd.randint(14, 40)
        p.setPen(QPen(QColor(110, 205, 255, al), 1))
        ln = rnd.randint(14, 90)
        horiz = rnd.random() < 0.6
        if horiz:
            p.drawLine(QPointF(x, y), QPointF(x + ln, y))
            for k in range(3):
                tx = x + rnd.uniform(0, ln)
                p.drawLine(QPointF(tx, y - 3), QPointF(tx, y + 3))
        else:
            p.drawLine(QPointF(x, y), QPointF(x, y + ln))
        # microdato textual
        if rnd.random() < 0.55:
            p.setPen(QColor(120, 215, 255, al + 12))
            p.drawText(QPointF(x + ln + 5, y + 3),
                       "%04X" % rnd.randint(0, 0xFFFF))
    # fragmentos diagonales dispersos
    for i in range(14):
        x, y = rnd.uniform(30, w - 60), rnd.uniform(30, h - 60)
        s = rnd.uniform(3, 8)
        p.setPen(QPen(QColor(130, 220, 255, rnd.randint(12, 32)), 1))
        p.drawLine(QPointF(x, y), QPointF(x + s, y + s * rnd.uniform(-1, 1)))
    p.end()
    return _pixmap_from_image(img)


def beam_sprite(w, h):
    img = QImage(w, h, QImage.Format_ARGB32)
    img.fill(Qt.transparent)
    p = QPainter(img)
    g = QLinearGradient(0, 0, w, 0)
    g.setColorAt(0.0, QColor(0, 170, 255, 0))
    g.setColorAt(0.5, QColor(120, 220, 255, 70))
    g.setColorAt(1.0, QColor(0, 170, 255, 0))
    p.fillRect(img.rect(), QBrush(g))
    p.end()
    return _pixmap_from_image(img)


def vignette_sprite(w, h):
    img = QImage(w, h, QImage.Format_ARGB32)
    img.fill(Qt.transparent)
    p = QPainter(img)
    g = QRadialGradient(w / 2, h * 0.42, w * 0.75)
    g.setColorAt(0.0, QColor(2, 8, 18, 0))
    g.setColorAt(0.6, QColor(2, 8, 18, 40))
    g.setColorAt(1.0, QColor(1, 4, 10, 170))
    p.fillRect(img.rect(), QBrush(g))
    p.end()
    return _pixmap_from_image(img)


def frame_sprite(w, h, color, c=9):
    """Mini-marco tecnologico con esquinas recortadas (modulos HUD)."""
    m = 10
    img = QImage(w + 2 * m, h + 2 * m, QImage.Format_ARGB32)
    img.fill(Qt.transparent)
    p = QPainter(img)
    p.setRenderHint(QPainter.Antialiasing)
    r = QRectF(m, m, w, h)
    shape = _chamfer_path(r, c)
    g = QLinearGradient(0, r.top(), 0, r.bottom())
    g.setColorAt(0.0, QColor(color.red(), color.green(), color.blue(), 34))
    g.setColorAt(1.0, QColor(4, 12, 26, 52))
    p.setPen(Qt.NoPen)
    p.setBrush(QBrush(g))
    p.drawPath(shape)
    p.setBrush(Qt.NoBrush)
    p.setPen(QPen(QColor(color.red(), color.green(), color.blue(), 130), 1))
    p.drawPath(shape)
    p.setPen(QPen(QColor(170, 240, 255, 200), 1.4))
    for (px, py, dx, dy) in ((r.left(), r.top() + c, 1, -1),
                             (r.right(), r.top() + c, -1, -1),
                             (r.left(), r.bottom() - c, 1, 1),
                             (r.right(), r.bottom() - c, -1, 1)):
        p.drawLine(QPointF(px, py), QPointF(px + dx * 6, py))
        p.drawLine(QPointF(px, py), QPointF(px, py + dy * 6))
    p.end()
    return _pixmap_from_image(img)


def hdr_left_sprite():
    """Cabecera superior izquierda: ATLAS / PERSONAL AI OS + 4 lineas."""
    w, h = 250, 128
    img = QImage(w, h, QImage.Format_ARGB32)
    img.fill(Qt.transparent)
    p = QPainter(img)
    p.setRenderHint(QPainter.TextAntialiasing)
    f = QFont("Consolas", 15)
    f.setBold(True)
    f.setLetterSpacing(QFont.AbsoluteSpacing, 6)
    p.setFont(f)
    p.setPen(QColor(200, 242, 255, 235))
    p.drawText(QRectF(0, 0, w, 26), Qt.AlignLeft, "ATLAS")
    f2 = QFont("Consolas", 8)
    f2.setLetterSpacing(QFont.AbsoluteSpacing, 4)
    p.setFont(f2)
    p.setPen(QColor(140, 220, 255, 175))
    p.drawText(QRectF(2, 26, w, 16), Qt.AlignLeft, "PERSONAL AI OS")
    p.setPen(QPen(QColor(90, 200, 255, 90), 1))
    p.drawLine(QPointF(0, 46), QPointF(120, 46))
    f3 = QFont("Consolas", 8)
    f3.setLetterSpacing(QFont.AbsoluteSpacing, 1)
    p.setFont(f3)
    for i, s in enumerate(("TU INTELIGENCIA", "TU ENTORNO",
                           "TU RENDIMIENTO", "TU LIBERTAD")):
        y = 54 + i * 17
        p.setPen(QPen(QColor(110, 210, 255, 120), 1.2))
        p.drawLine(QPointF(4, y + 7), QPointF(12, y + 7))
        p.setPen(QColor(160, 228, 255, 190))
        p.drawText(QRectF(18, y, w - 18, 14), Qt.AlignLeft, s)
    p.end()
    return _pixmap_from_image(img)


def hdr_right_sprite():
    """Cabecera superior derecha: verbos de identidad (referencia)."""
    w, h = 220, 96
    img = QImage(w, h, QImage.Format_ARGB32)
    img.fill(Qt.transparent)
    p = QPainter(img)
    p.setRenderHint(QPainter.TextAntialiasing)
    f = QFont("Consolas", 9)
    f.setBold(True)
    f.setLetterSpacing(QFont.AbsoluteSpacing, 2)
    p.setFont(f)
    for i, s in enumerate(("APRENDER", "AUTOMATIZAR", "CREAR",
                           "EVOLUCIONAR", "CONTIGO")):
        y = i * 18
        p.setPen(QColor(170, 232, 255, 205 - i * 8))
        p.drawText(QRectF(0, y, w - 16, 15),
                   Qt.AlignRight, s)
        p.setPen(QPen(QColor(110, 210, 255, 130), 1.2))
        p.drawLine(QPointF(w - 12, y + 7), QPointF(w - 2, y + 7))
    p.end()
    return _pixmap_from_image(img)


def chip_sprite():
    """Micro-cuadrado de dato flotante (ambiente)."""
    img = QImage(14, 12, QImage.Format_ARGB32)
    img.fill(Qt.transparent)
    p = QPainter(img)
    p.setPen(QPen(QColor(140, 225, 255, 210), 1))
    p.setBrush(Qt.NoBrush)
    p.drawRect(QRectF(1.5, 1.5, 11, 9))
    p.setPen(Qt.NoPen)
    p.setBrush(QColor(170, 235, 255, 190))
    p.drawEllipse(QRectF(6, 5, 2.4, 2.4))
    p.end()
    return _pixmap_from_image(img)


def _corner_brackets(p, r, color, L=18, wd=1.6):
    pen = QPen(color, wd)
    p.setPen(pen)
    p.setBrush(Qt.NoBrush)
    for (cx, cy, dx, dy) in ((r.left(), r.top(), 1, 1), (r.right(), r.top(), -1, 1),
                             (r.left(), r.bottom(), 1, -1), (r.right(), r.bottom(), -1, -1)):
        p.drawLine(QPointF(cx + dx * L, cy), QPointF(cx, cy))
        p.drawLine(QPointF(cx, cy + dy * L), QPointF(cx, cy))


def _value_alpha(v):
    """Alfa del valor segun fiabilidad: '--'/'UNKNOWN' = no conectado."""
    if v in ("--", "UNKNOWN"):
        return 105
    if v in ("DEMO", "IDLE"):
        return 170
    return 215


def _chamfer_path(r, c):
    """Rectangulo con esquinas recortadas (estetica de la referencia)."""
    path = QPainterPath()
    path.moveTo(r.left() + c, r.top())
    path.lineTo(r.right() - c, r.top())
    path.lineTo(r.right(), r.top() + c)
    path.lineTo(r.right(), r.bottom() - c)
    path.lineTo(r.right() - c, r.bottom())
    path.lineTo(r.left() + c, r.bottom())
    path.lineTo(r.left(), r.bottom() - c)
    path.lineTo(r.left(), r.top() + c)
    path.closeSubpath()
    return path


def panel_sprite(w, h, title, rows, color, footer="// DEMO LAYER",
                 row_h=21, marker=False):
    """Panel holografico generico por campos (marco con esquinas recortadas,
    marcas tecnicas, lineas internas y glow controlado).

    rows: lista de (izquierda, derecha). Con marker=True la columna izquierda
    es (nombre, estado) y se dibuja un anillo de estado:
    UNKNOWN (discreto) / AVAILABLE (anillo+punto) / ACTIVE (brillante).
    """
    m = 30
    img = QImage(w + 2 * m, h + 2 * m, QImage.Format_ARGB32)
    img.fill(Qt.transparent)
    p = QPainter(img)
    p.setRenderHint(QPainter.Antialiasing)
    r = QRectF(m, m, w, h)
    ch = 16
    shape = _chamfer_path(r, ch)
    inner = _chamfer_path(r.adjusted(5, 5, -5, -5), max(4, ch - 6))
    # glow controlado detras del marco
    p.setCompositionMode(QPainter.CompositionMode_Plus)
    gg = QRadialGradient(r.center().x(), r.center().y(), w * 0.62)
    gg.setColorAt(0.0, QColor(color.red(), color.green(), color.blue(), 0))
    gg.setColorAt(0.72, QColor(color.red(), color.green(), color.blue(), 26))
    gg.setColorAt(1.0, QColor(color.red(), color.green(), color.blue(), 0))
    p.setPen(Qt.NoPen)
    p.setBrush(QBrush(gg))
    p.drawRect(r.adjusted(-m, -m, m, m))
    p.setCompositionMode(QPainter.CompositionMode_SourceOver)
    # relleno muy tenue + gradiente superior
    g = QLinearGradient(0, r.top(), 0, r.bottom())
    g.setColorAt(0.0, QColor(color.red(), color.green(), color.blue(), 30))
    g.setColorAt(0.28, QColor(6, 16, 32, 52))
    g.setColorAt(1.0, QColor(4, 12, 26, 46))
    p.setPen(Qt.NoPen)
    p.setBrush(QBrush(g))
    p.drawPath(shape)
    # marco exterior fino
    p.setBrush(Qt.NoBrush)
    p.setPen(QPen(QColor(color.red(), color.green(), color.blue(), 110), 1))
    p.drawPath(shape)
    # linea interior (profundidad)
    p.setPen(QPen(QColor(color.red(), color.green(), color.blue(), 42), 1))
    p.drawPath(inner)
    # acentos en los recortes de esquina (marcas tecnicas)
    p.setPen(QPen(QColor(170, 240, 255, 200), 1.6))
    for (px, py, dx, dy) in ((r.left(), r.top() + ch, 1, -1),
                             (r.right(), r.top() + ch, -1, -1),
                             (r.left(), r.bottom() - ch, 1, 1),
                             (r.right(), r.bottom() - ch, -1, 1)):
        p.drawLine(QPointF(px, py), QPointF(px + dx * 8, py))
        p.drawLine(QPointF(px, py), QPointF(px, py + dy * 8))
    # micro-marcas en el centro de cada lado
    p.setPen(QPen(QColor(color.red(), color.green(), color.blue(), 150), 1))
    p.drawLine(QPointF(r.center().x() - 8, r.top()),
               QPointF(r.center().x() + 8, r.top()))
    p.drawLine(QPointF(r.center().x() - 8, r.bottom()),
               QPointF(r.center().x() + 8, r.bottom()))
    # ticks superiores (regla)
    p.setPen(QPen(QColor(color.red(), color.green(), color.blue(), 90), 1))
    for i in range(0, int(w), 14):
        x = r.left() + i
        ln = 6 if i % 70 else 11
        p.drawLine(QPointF(x, r.top()), QPointF(x, r.top() + ln))
    # barra de titulo fina + linea vertical interna derecha
    p.setPen(QPen(QColor(color.red(), color.green(), color.blue(), 130), 1))
    p.drawLine(QPointF(r.left() + 12, r.top() + 30),
               QPointF(r.right() - 12, r.top() + 30))
    p.setPen(QPen(QColor(color.red(), color.green(), color.blue(), 36), 1))
    p.drawLine(QPointF(r.right() - 14, r.top() + 36),
               QPointF(r.right() - 14, r.bottom() - 18))
    # titulo
    f = QFont("Consolas", 11)
    f.setBold(True)
    f.setLetterSpacing(QFont.AbsoluteSpacing, 1.5)
    p.setFont(f)
    p.setPen(QColor(min(255, color.red() + 60), min(255, color.green() + 50),
                    min(255, color.blue() + 40), 235))
    p.drawText(QRectF(r.left() + 14, r.top() + 8, w - 40, 22), Qt.AlignLeft,
               title)
    # marcador titulo (chip)
    p.setPen(QPen(QColor(color.red(), color.green(), color.blue(), 160), 1))
    p.setBrush(QColor(color.red(), color.green(), color.blue(), 50))
    p.drawRect(QRectF(r.right() - 22, r.top() + 9, 12, 12))
    # filas de campos (label izq / valor der)
    f2 = QFont("Consolas", 8.5)
    p.setFont(f2)
    y = r.top() + 44
    for row in rows:
        if marker:
            name, st = row
            cxm, cym = r.left() + 20, y + 8
            if st == "ACTIVE":
                p.setPen(QPen(QColor(160, 240, 255, 230), 1.4))
                p.setBrush(QColor(150, 235, 255, 200))
            elif st == "AVAILABLE":
                p.setPen(QPen(QColor(120, 220, 255, 190), 1.3))
                p.setBrush(QColor(120, 220, 255, 110))
            else:  # UNKNOWN / no conectado
                p.setPen(QPen(QColor(110, 205, 255, 85), 1.2))
                p.setBrush(Qt.NoBrush)
            p.drawEllipse(QRectF(cxm - 4, cym - 4, 8, 8))
            p.setPen(QColor(150, 220, 255, 200))
            p.drawText(QRectF(r.left() + 34, y, w - 120, 16), Qt.AlignLeft,
                       name)
        else:
            p.setPen(QPen(QColor(110, 205, 255, 110), 1))
            p.drawLine(QPointF(r.left() + 16, y + 8),
                       QPointF(r.left() + 24, y + 8))
            p.setPen(QColor(150, 220, 255, 195))
            p.drawText(QRectF(r.left() + 30, y, w - 110, 16), Qt.AlignLeft,
                       row[0])
        val = row[1]
        p.setPen(QColor(150, 225, 255, _value_alpha(val)))
        p.drawText(QRectF(r.left() + 30, y, w - 44, 16), Qt.AlignRight, val)
        y += row_h
    # linea inferior de estado
    p.setPen(QPen(QColor(color.red(), color.green(), color.blue(), 70), 1))
    p.drawLine(QPointF(r.left() + 12, r.bottom() - 12),
               QPointF(r.right() - 12, r.bottom() - 12))
    p.setFont(f2)
    p.setPen(QColor(120, 215, 255, 140))
    p.drawText(QRectF(r.left() + 14, r.bottom() - 26, w - 28, 14),
               Qt.AlignLeft, footer)
    p.end()
    return _pixmap_from_image(img)


def system_panel_sprite(w, h, title, color, footer):
    """Panel izquierdo: telemetria desde system_state (sin datos inventados)."""
    return panel_sprite(w, h, title,
                        [(k, v) for k, v in system_state.items()],
                        color, footer=footer, row_h=21)


def capability_panel_sprite(w, h, title, color, footer):
    """Panel derecho: capacidades desde capability_state (UNKNOWN de serie)."""
    return panel_sprite(w, h, title,
                        [(name, capability_state[name]) for name in CAP_NAMES],
                        color, footer=footer, row_h=20, marker=True)


def fields_panel_sprite(w, h, title, fields, color, footer):
    """Panel generico desde un dict de campos (actividad / automatizacion)."""
    return panel_sprite(w, h, title, [(k, v) for k, v in fields.items()],
                        color, footer=footer, row_h=20)


def menu_button_rects(w):
    """Rectangulos (en coords del sprite) de los 9 botones del menu.

    Botones compactos y con separacion uniforme, apoyados sobre la
    plataforma (sub-rayo inferior en bottom_hud_sprite).
    """
    n = len(MENU_ITEMS)
    cw, ch = 84, 60
    x0 = 150
    gap = (w - 2 * x0 - n * cw) / (n - 1) if n > 1 else 0
    return [QRectF(x0 + i * (cw + gap), 12, cw, ch) for i in range(n)]


def draw_menu_button(p, r, icon, label, mode="normal"):
    """Boton de menu con icono vectorial. Estados: normal/hover/active."""
    hi = mode != "normal"
    act = mode == "active"
    fa = 40 if act else (30 if hi else 22)
    ba = 255 if act else (200 if hi else 130)
    # relleno tenue
    g = QLinearGradient(0, r.top(), 0, r.bottom())
    g.setColorAt(0.0, QColor(60, 200, 255, fa))
    g.setColorAt(1.0, QColor(4, 14, 28, 60))
    p.setPen(Qt.NoPen)
    p.setBrush(QBrush(g))
    p.drawRect(r)
    # borde fino
    p.setPen(QPen(QColor(80, 210, 255, ba), 1.2 if hi else 1))
    p.setBrush(Qt.NoBrush)
    p.drawRect(r)
    # esquinas L
    _corner_brackets(p, r, QColor(170, 240, 255, 255 if act else
                                  (220 if hi else 170)), L=7, wd=1.2)
    # glow minimo detras del icono en hover/active
    ic_s = 24
    ix = r.center().x() - ic_s / 2
    iy = r.top() + 5
    if hi:
        p.setCompositionMode(QPainter.CompositionMode_Plus)
        p.setOpacity(0.28 if act else 0.16)
        gs = ic_s * 2.1
        p.drawPixmap(QRectF(ix + ic_s / 2 - gs / 2, iy + ic_s / 2 - gs / 2,
                            gs, gs), _ICON_GLOW, QRectF(_ICON_GLOW.rect()))
        p.setCompositionMode(QPainter.CompositionMode_SourceOver)
        p.setOpacity(1.0)
    # icono (linea fina cian)
    if icon is not None:
        p.setOpacity(1.0 if hi else 0.82)
        p.drawPixmap(QRectF(ix, iy, ic_s, ic_s), icon, QRectF(icon.rect()))
        p.setOpacity(1.0)
    # linea inferior de acento
    p.setPen(QPen(QColor(100, 215, 255, 255 if act else (150 if hi else 90)),
                  2.6 if act else 2))
    p.drawLine(QPointF(r.left() + 10, r.bottom() - 4),
               QPointF(r.right() - 10, r.bottom() - 4))
    # etiqueta
    f = QFont("Consolas", 7)
    f.setBold(True)
    f.setLetterSpacing(QFont.AbsoluteSpacing, 0.6)
    p.setFont(f)
    p.setPen(QColor(200, 245, 255, 245 if hi else 215))
    p.drawText(QRectF(r.left(), r.bottom() - 17, r.width(), 13),
               Qt.AlignCenter, label)


def bottom_hud_sprite(w, h, icons):
    """Tira inferior: botones compactos apoyados sobre la plataforma
    (sub-rayo/reflejo inferior por boton) + lema."""
    img = QImage(w, h, QImage.Format_ARGB32)
    img.fill(Qt.transparent)
    p = QPainter(img)
    p.setRenderHint(QPainter.Antialiasing)
    rects = menu_button_rects(w)
    # halo/reflejo inferior: los botones se apoyan en la plataforma
    p.setCompositionMode(QPainter.CompositionMode_Plus)
    for r in rects:
        gcx, gcy = r.center().x(), r.bottom() + 6
        gw2, gh2 = r.width() * 1.15, 26
        g = QRadialGradient(gcx, gcy, gw2 / 2)
        g.setColorAt(0.0, QColor(80, 200, 255, 64))
        g.setColorAt(0.5, QColor(60, 175, 255, 26))
        g.setColorAt(1.0, QColor(40, 150, 255, 0))
        p.setBrush(QBrush(g))
        p.drawEllipse(QRectF(gcx - gw2 / 2, gcy - gh2 / 2, gw2, gh2))
        # micro-reflejo espejado justo bajo el boton
        g2 = QLinearGradient(0, r.bottom() + 2, 0, r.bottom() + 12)
        g2.setColorAt(0.0, QColor(90, 210, 255, 44))
        g2.setColorAt(1.0, QColor(70, 190, 255, 0))
        p.setBrush(QBrush(g2))
        p.drawRect(QRectF(r.left() + 8, r.bottom() + 2,
                          r.width() - 16, 10))
    p.setCompositionMode(QPainter.CompositionMode_SourceOver)
    for i, r in enumerate(rects):
        draw_menu_button(p, r, icons[i], MENU_ITEMS[i], mode="normal")
    # lema
    f2 = QFont("Consolas", 10)
    f2.setLetterSpacing(QFont.AbsoluteSpacing, 6)
    p.setFont(f2)
    p.setPen(QColor(150, 225, 255, 190))
    p.drawText(QRectF(0, 100, w, 24), Qt.AlignCenter,
               "IDEAS  >  ACCIONES  >  RESULTADOS")
    p.end()
    return _pixmap_from_image(img)


# --------------------------------------------------------- glifo central ---
def _glyph_path(size):
    """Logotipo ATLAS de la referencia: simbolo angular entrelazado.

    Geometria (caja 0..1, y hacia abajo):
    - chevron principal con vertice alto y piernas anchas;
    - pierna derecha SEGMENTADA por una abertura diagonal (corte A-B);
    - barra diagonal que cruza la abertura hacia arriba-derecha
      (efecto entrelazado);
    - acento diagonal inferior paralelo (diagonales interiores).
    """
    path = QPainterPath()
    path.setFillRule(Qt.WindingFill)
    s = float(size)

    def P(x, y):
        return QPointF(x * s, y * s)

    # pierna izquierda + vertice (cinta unica)
    path.moveTo(P(0.50, 0.03))
    path.lineTo(P(0.50, 0.40))
    path.lineTo(P(0.225, 0.97))
    path.lineTo(P(0.04, 0.97))
    path.closeSubpath()
    # pierna derecha, tramo superior (hasta la abertura diagonal)
    path.moveTo(P(0.50, 0.03))
    path.lineTo(P(0.754, 0.55))
    path.lineTo(P(0.597, 0.60))
    path.lineTo(P(0.50, 0.40))
    path.closeSubpath()
    # pierna derecha, tramo inferior (tras la abertura)
    path.moveTo(P(0.789, 0.62))
    path.lineTo(P(0.96, 0.97))
    path.lineTo(P(0.775, 0.97))
    path.lineTo(P(0.630, 0.67))
    path.closeSubpath()
    # barra diagonal entrelazada (puentea la abertura)
    path.moveTo(P(0.335, 0.68))
    path.lineTo(P(0.74, 0.545))
    path.lineTo(P(0.74, 0.665))
    path.lineTo(P(0.375, 0.80))
    path.closeSubpath()
    # acento diagonal inferior paralelo (segmento corto)
    path.moveTo(P(0.40, 0.865))
    path.lineTo(P(0.585, 0.795))
    path.lineTo(P(0.615, 0.87))
    path.lineTo(P(0.425, 0.945))
    path.closeSubpath()
    return path


def _glyph_scaled(path, size, kx, ky):
    t = QTransform()
    c = size / 2.0
    t.translate(c, c)
    t.scale(kx, ky)
    t.translate(-c, -c)
    return t.map(path)


def glyph_halo_sprite(size):
    """Halo volumetrico del simbolo: siluetas apiladas + luz radial amplia."""
    img = QImage(size, size, QImage.Format_ARGB32)
    img.fill(Qt.transparent)
    p = QPainter(img)
    p.setRenderHint(QPainter.Antialiasing)
    # luz volumetrica amplia
    g = QRadialGradient(size / 2, size * 0.52, size * 0.55)
    g.setColorAt(0.0, QColor(70, 185, 255, 92))
    g.setColorAt(0.45, QColor(45, 150, 245, 48))
    g.setColorAt(1.0, QColor(20, 110, 220, 0))
    p.setPen(Qt.NoPen)
    p.setBrush(QBrush(g))
    p.drawEllipse(img.rect())
    # siluetas apiladas (halo con cuerpo, no glow plano)
    path = _glyph_path(size)
    for grow, ky, al in ((1.30, 1.26, 22), (1.18, 1.15, 32),
                         (1.08, 1.06, 44), (1.02, 1.01, 58)):
        p.setBrush(QColor(70, 185, 255, al))
        p.drawPath(_glyph_scaled(path, size, grow, ky))
    p.end()
    return _pixmap_from_image(img)


def glyph_fill_sprite(size):
    """Cuerpo translucido del simbolo + circuitos integrados recortados."""
    img = QImage(size, size, QImage.Format_ARGB32)
    img.fill(Qt.transparent)
    p = QPainter(img)
    p.setRenderHint(QPainter.Antialiasing)
    path = _glyph_path(size)
    g = QLinearGradient(0, 0, 0, size)
    g.setColorAt(0.0, QColor(130, 220, 255, 96))
    g.setColorAt(0.5, QColor(70, 175, 250, 66))
    g.setColorAt(1.0, QColor(35, 130, 230, 44))
    p.setPen(Qt.NoPen)
    p.setBrush(QBrush(g))
    p.drawPath(path)
    # circuitos dentro del simbolo (textura tech, integrada al volumen)
    p.save()
    p.setClipPath(path, Qt.IntersectClip)
    rnd = random.Random(51)
    for i in range(46):
        x = rnd.uniform(0.08, 0.92) * size
        y = rnd.uniform(0.10, 0.95) * size
        kind = rnd.random()
        al = rnd.randint(20, 58)
        col = QColor(150, 230, 255, al)
        if kind < 0.45:
            ln = rnd.uniform(0.03, 0.11) * size
            horiz = rnd.random() < 0.6
            p.setPen(QPen(col, 1))
            if horiz:
                p.drawLine(QPointF(x, y), QPointF(x + ln, y))
            else:
                p.drawLine(QPointF(x, y), QPointF(x, y + ln))
        elif kind < 0.75:
            s = rnd.uniform(1.4, 2.6)
            p.setPen(Qt.NoPen)
            p.setBrush(col)
            p.drawEllipse(QRectF(x - s / 2, y - s / 2, s, s))
        else:
            w2 = rnd.uniform(0.03, 0.08) * size
            p.setPen(QPen(col, 1))
            p.setBrush(Qt.NoBrush)
            p.drawRect(QRectF(x, y - 1, w2, 2))
    p.restore()
    p.end()
    return _pixmap_from_image(img)


def glyph_edge_sprite(size, dim=False):
    """Doble contorno cian estilizado (filo ancho tenue + linea fina viva)."""
    img = QImage(size, size, QImage.Format_ARGB32)
    img.fill(Qt.transparent)
    p = QPainter(img)
    p.setRenderHint(QPainter.Antialiasing)
    path = _glyph_path(size)
    a = 0.32 if dim else 1.0
    # filo exterior ancho tenue
    p.setBrush(Qt.NoBrush)
    p.setPen(QPen(QColor(60, 170, 255, int(120 * a)), 6.0))
    p.drawPath(path)
    # trazo principal fino cian
    p.setPen(QPen(QColor(140, 228, 255, int(245 * a)), 2.0))
    p.drawPath(path)
    # doble filo interior (linea desplazada, estilo estilizado)
    inner = _glyph_scaled(path, size, 0.88, 0.84)
    p.setPen(QPen(QColor(190, 242, 255, int(150 * a)), 1.1))
    p.drawPath(inner)
    p.end()
    return _pixmap_from_image(img)


def glyph_energy_sprite(size):
    """Energia atravesando el simbolo: haces horizontales suaves (aditivo)."""
    img = QImage(size, size, QImage.Format_ARGB32)
    img.fill(Qt.transparent)
    p = QPainter(img)
    rnd = random.Random(63)
    for i in range(5):
        y = size * rnd.uniform(0.34, 0.68)
        hband = size * rnd.uniform(0.05, 0.09)
        # banda ancha difusa
        g = QLinearGradient(0, 0, size, 0)
        g.setColorAt(0.0, QColor(60, 180, 255, 0))
        g.setColorAt(0.5, QColor(110, 215, 255, rnd.randint(34, 58)))
        g.setColorAt(1.0, QColor(60, 180, 255, 0))
        p.setPen(Qt.NoPen)
        p.setBrush(QBrush(g))
        p.drawRect(QRectF(0, y - hband / 2, size, hband))
        # linea fina viva
        g2 = QLinearGradient(0, 0, size, 0)
        g2.setColorAt(0.0, QColor(160, 235, 255, 0))
        g2.setColorAt(0.5, QColor(180, 240, 255, rnd.randint(120, 175)))
        g2.setColorAt(1.0, QColor(160, 235, 255, 0))
        p.setBrush(QBrush(g2))
        p.drawRect(QRectF(0, y - 0.8, size, 1.6))
    p.end()
    return _pixmap_from_image(img)


# ------------------------------------------------------------ iconos HUD ---
def icon_pixmap(draw_fn, size, bright=False):
    """Icono vectorial de linea fina: pasada glow + pasada principal cian."""
    img = QImage(size, size, QImage.Format_ARGB32)
    img.fill(Qt.transparent)
    p = QPainter(img)
    p.setRenderHint(QPainter.Antialiasing)
    col = QColor(150, 235, 255) if bright else QColor(125, 218, 255)
    for wd, al in ((4.6, 40 if bright else 26), (2.0, 240 if bright else 195)):
        pen = QPen(QColor(col.red(), col.green(), col.blue(), al), wd)
        pen.setCapStyle(Qt.RoundCap)
        pen.setJoinStyle(Qt.RoundJoin)
        p.setPen(pen)
        p.setBrush(Qt.NoBrush)
        draw_fn(p, size)
    p.end()
    return _pixmap_from_image(img)


def _ic_chat(p, s):
    p.drawRoundedRect(QRectF(0.10 * s, 0.14 * s, 0.80 * s, 0.56 * s),
                      0.16 * s, 0.16 * s)
    path = QPainterPath()
    path.moveTo(0.32 * s, 0.70 * s)
    path.lineTo(0.26 * s, 0.88 * s)
    path.lineTo(0.46 * s, 0.70 * s)
    p.drawPath(path)
    p.setPen(Qt.NoPen)
    for k in range(3):
        p.drawEllipse(QRectF((0.32 + 0.18 * k) * s - 0.045 * s,
                             0.40 * s - 0.045 * s, 0.09 * s, 0.09 * s))


def _ic_voz(p, s):
    p.drawRoundedRect(QRectF(0.40 * s, 0.10 * s, 0.20 * s, 0.42 * s),
                      0.10 * s, 0.10 * s)
    p.drawArc(QRectF(0.22 * s, 0.16 * s, 0.56 * s, 0.56 * s),
              200 * 16, 140 * 16)
    p.drawLine(QPointF(0.50 * s, 0.64 * s), QPointF(0.50 * s, 0.80 * s))
    p.drawLine(QPointF(0.32 * s, 0.84 * s), QPointF(0.68 * s, 0.84 * s))
    # ondas laterales
    p.drawArc(QRectF(0.04 * s, 0.30 * s, 0.14 * s, 0.30 * s), 60 * 16, 120 * 16)
    p.drawArc(QRectF(0.82 * s, 0.30 * s, 0.14 * s, 0.30 * s),
              -80 * 16, 120 * 16)


def _ic_pc(p, s):
    p.drawRect(QRectF(0.10 * s, 0.18 * s, 0.80 * s, 0.50 * s))
    p.drawLine(QPointF(0.22 * s, 0.30 * s), QPointF(0.52 * s, 0.22 * s))
    p.drawLine(QPointF(0.50 * s, 0.68 * s), QPointF(0.50 * s, 0.80 * s))
    p.drawLine(QPointF(0.30 * s, 0.83 * s), QPointF(0.70 * s, 0.83 * s))


def _ic_entrenamiento(p, s):
    p.drawLine(QPointF(0.14 * s, 0.50 * s), QPointF(0.86 * s, 0.50 * s))
    p.drawRect(QRectF(0.14 * s, 0.26 * s, 0.11 * s, 0.48 * s))
    p.drawRect(QRectF(0.28 * s, 0.34 * s, 0.07 * s, 0.32 * s))
    p.drawRect(QRectF(0.75 * s, 0.26 * s, 0.11 * s, 0.48 * s))
    p.drawRect(QRectF(0.65 * s, 0.34 * s, 0.07 * s, 0.32 * s))


def _ic_nutricion(p, s):
    path = QPainterPath()
    path.moveTo(0.50 * s, 0.34 * s)
    path.cubicTo(0.24 * s, 0.20 * s, 0.10 * s, 0.42 * s, 0.16 * s, 0.62 * s)
    path.cubicTo(0.22 * s, 0.84 * s, 0.34 * s, 0.92 * s, 0.50 * s, 0.88 * s)
    path.cubicTo(0.66 * s, 0.92 * s, 0.78 * s, 0.84 * s, 0.84 * s, 0.62 * s)
    path.cubicTo(0.90 * s, 0.42 * s, 0.76 * s, 0.20 * s, 0.50 * s, 0.34 * s)
    p.drawPath(path)
    p.drawLine(QPointF(0.50 * s, 0.32 * s), QPointF(0.55 * s, 0.14 * s))
    leaf = QPainterPath()
    leaf.moveTo(0.55 * s, 0.16 * s)
    leaf.cubicTo(0.66 * s, 0.04 * s, 0.82 * s, 0.10 * s, 0.80 * s, 0.20 * s)
    leaf.cubicTo(0.72 * s, 0.26 * s, 0.60 * s, 0.24 * s, 0.55 * s, 0.16 * s)
    p.drawPath(leaf)


def _ic_salud(p, s):
    path = QPainterPath()
    path.moveTo(0.50 * s, 0.86 * s)
    path.cubicTo(0.06 * s, 0.54 * s, 0.14 * s, 0.14 * s, 0.50 * s, 0.34 * s)
    path.cubicTo(0.86 * s, 0.14 * s, 0.94 * s, 0.54 * s, 0.50 * s, 0.86 * s)
    p.drawPath(path)
    pulse = QPainterPath()
    pulse.moveTo(0.14 * s, 0.52 * s)
    pulse.lineTo(0.34 * s, 0.52 * s)
    pulse.lineTo(0.40 * s, 0.40 * s)
    pulse.lineTo(0.48 * s, 0.64 * s)
    pulse.lineTo(0.56 * s, 0.44 * s)
    pulse.lineTo(0.60 * s, 0.52 * s)
    pulse.lineTo(0.86 * s, 0.52 * s)
    p.drawPath(pulse)


def _ic_codigo(p, s):
    p.drawLine(QPointF(0.30 * s, 0.24 * s), QPointF(0.10 * s, 0.50 * s))
    p.drawLine(QPointF(0.10 * s, 0.50 * s), QPointF(0.30 * s, 0.76 * s))
    p.drawLine(QPointF(0.70 * s, 0.24 * s), QPointF(0.90 * s, 0.50 * s))
    p.drawLine(QPointF(0.90 * s, 0.50 * s), QPointF(0.70 * s, 0.76 * s))
    p.drawLine(QPointF(0.57 * s, 0.18 * s), QPointF(0.43 * s, 0.82 * s))


def _ic_proyectos(p, s):
    path = QPainterPath()
    path.moveTo(0.08 * s, 0.28 * s)
    path.lineTo(0.36 * s, 0.28 * s)
    path.lineTo(0.44 * s, 0.38 * s)
    path.lineTo(0.92 * s, 0.38 * s)
    path.lineTo(0.92 * s, 0.80 * s)
    path.lineTo(0.08 * s, 0.80 * s)
    path.closeSubpath()
    p.drawPath(path)
    p.drawLine(QPointF(0.30 * s, 0.60 * s), QPointF(0.70 * s, 0.60 * s))
    p.setPen(Qt.NoPen)
    p.drawEllipse(QRectF(0.26 * s, 0.56 * s, 0.08 * s, 0.08 * s))
    p.drawEllipse(QRectF(0.66 * s, 0.56 * s, 0.08 * s, 0.08 * s))


def _ic_herramientas(p, s):
    cx, cy = 0.50 * s, 0.50 * s
    p.drawEllipse(QRectF(cx - 0.24 * s, cy - 0.24 * s,
                         0.48 * s, 0.48 * s))
    p.drawEllipse(QRectF(cx - 0.10 * s, cy - 0.10 * s,
                         0.20 * s, 0.20 * s))
    for k in range(8):
        a = k * math.pi / 4
        x1 = cx + 0.24 * s * math.cos(a)
        y1 = cy + 0.24 * s * math.sin(a)
        x2 = cx + 0.36 * s * math.cos(a)
        y2 = cy + 0.36 * s * math.sin(a)
        p.drawLine(QPointF(x1, y1), QPointF(x2, y2))


ICON_DRAWERS = [_ic_chat, _ic_voz, _ic_pc, _ic_entrenamiento, _ic_nutricion,
                _ic_salud, _ic_codigo, _ic_proyectos, _ic_herramientas]
_ICON_GLOW = None  # glow para hover/active (se inicializa en build_assets)


# ------------------------------------------------------------------ demo ---
STATE_NAMES = ["IDLE", "LISTENING", "THINKING", "SPEAKING", "AUTOMATION"]
STATE_KEYS = ["idle", "listening", "thinking", "speaking", "automation"]

# Estado de voz mostrado en el modulo inferior derecho (segun STATE real)
VOICE_STATUS = {
    "IDLE": "ESPERANDO...",
    "LISTENING": "ESCUCHANDO...",
    "THINKING": "PENSANDO...",
    "SPEAKING": "HABLANDO...",
    "AUTOMATION": "EJECUTANDO...",
}
DIAS = ["Lun", "Mar", "Mié", "Jue", "Vie", "Sáb", "Dom"]
MESES = ["Ene", "Feb", "Mar", "Abr", "May", "Jun",
         "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]

PARAMS = {
    "idle":       dict(glow=1.00, ring=1.0, data=1.0, beam=0.55, breathe=0.012,
                       tint_alpha=0.0),
    "listening":  dict(glow=1.35, ring=1.3, data=1.2, beam=0.8, breathe=0.016,
                       tint_alpha=0.5),
    "thinking":   dict(glow=1.30, ring=2.2, data=2.6, beam=1.0, breathe=0.014,
                       tint_alpha=0.5),
    "speaking":   dict(glow=1.40, ring=1.1, data=1.0, beam=0.7, breathe=0.030,
                       tint_alpha=0.5),
    "automation": dict(glow=0.9,  ring=1.6, data=1.8, beam=0.9, breathe=0.012,
                       tint_alpha=1.0),
}


class AtlasStageV2(QOpenGLWidget):
    chat_requested = Signal()

    def __init__(self):
        super().__init__()
        self.status_message = None
        self.status_until = 0.0
        self.setWindowTitle("ATLAS Stage V2 - [1..5] estados  ESC salir")
        self.setFixedSize(W, H)
        self.A = build_assets()

        self.state = 0
        self.t = 0.0
        self.fps = 0.0
        self.mouse = QPointF(W / 2, H / 2)
        self.par = [QPointF(0, 0)] * 5
        # menu inferior: rects en coords de widget + estado visual
        hud_y = H - 150
        self.menu_rects = [r.translated(QPointF(0, hud_y))
                           for r in menu_button_rects(W)]
        self.active_menu = None
        # chips de datos flotantes (ambiente, pocos y baratos)
        rnd2 = random.Random(77)
        self.chips = [dict(x=rnd2.uniform(40, W - 60),
                           y=rnd2.uniform(60, H - 200),
                           spd=rnd2.uniform(4, 12),
                           ph=rnd2.uniform(0, 6.28),
                           drift=rnd2.uniform(-8, 8))
                      for _ in range(12)]

        # fondo lejano
        self.far = [dict(x=random.uniform(0, W), y=random.uniform(0, H),
                         s=random.uniform(0.25, 0.7),
                         ph=random.uniform(0, 6.28)) for _ in range(60)]
        # nodos de datos en orbitas
        self.orbits = []
        for i in range(3):
            rx = CORE_W * random.uniform(0.62, 0.95)
            ry = rx * random.uniform(0.16, 0.30)
            self.orbits.append(dict(rx=rx, ry=ry, tilt=random.uniform(-0.2, 0.2),
                                    speed=random.uniform(0.25, 0.5) *
                                    (1 if i % 2 else -1),
                                    off=random.uniform(0, 6.28),
                                    n=random.randint(5, 8)))
        # barridos
        self.sweep_t = -2.0
        # nodos brillantes del plano delantero (pocas, sobre el nucleo)
        rnd = random.Random(19)
        self.front_nodes = [dict(th=rnd.uniform(0, 6.28),
                                 rr=rnd.uniform(0.18, 0.40),
                                 sp=rnd.uniform(1.2, 2.2),
                                 ph=rnd.uniform(0, 6.28))
                            for _ in range(5)]

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.tick)
        self.timer.start(15)
        self.clock = QElapsedTimer()
        self.clock.start()
        self.last = 0
        self.fps_acc = 0.0
        self.fps_n = 0
        self.fps_t = 0.0
        self.setMouseTracking(True)

    # ------------------------------------------------------------- frame --
    def tick(self):
        now = self.clock.elapsed() / 1000.0
        dt = min(now - self.last, 0.1)
        self.last = now
        if dt > 0:
            self.fps_acc += 1.0 / dt
            self.fps_n += 1
            if now - self.fps_t > 0.5:
                self.fps = self.fps_acc / self.fps_n
                self.fps_acc = 0.0
                self.fps_n = 0
                self.fps_t = now
                self.setWindowTitle("ATLAS Stage V2 - ESTADO: %s - %.0f FPS"
                                    % (STATE_NAMES[self.state], self.fps))
        self.t = now
        p = PARAMS[STATE_KEYS[self.state]]
        # parallax con deriva idle + raton
        dx = math.sin(now * 0.23) * 6 + \
            (self.mouse.x() - W / 2) / W * 26
        dy = math.cos(now * 0.19) * 4 + \
            (self.mouse.y() - H / 2) / H * 16
        fac = [0.15, 0.35, 0.6, 0.85, 1.15]
        for i in range(5):
            self.par[i] = QPointF(dx * fac[i], dy * fac[i] * 0.6)
        # barrido periodico
        self.sweep_t += dt
        if self.sweep_t > 6.0 / p["beam"]:
            self.sweep_t = -0.9
        self.update()

    # ------------------------------------------------------------ painter --
    def paintEvent(self, ev):
        t = self.t
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setRenderHint(QPainter.SmoothPixmapTransform)
        p.fillRect(self.rect(), QColor(2, 6, 12))
        P = PARAMS[STATE_KEYS[self.state]]

        # 1 FONDO: glow profundo + microdatos + particulas + viñeta
        gb = self.A["glow_big"]
        gw = gb.width()
        p.setOpacity(0.9 * P["glow"])
        p.drawPixmap(QPointF(CORE_CX - gw / 2 + self.par[0].x(),
                             CORE_CY - gw / 2 + self.par[0].y() - 40), gb)
        p.setOpacity(0.55)
        p.drawPixmap(QPointF(CORE_CX - 450 + self.par[0].x() * 0.5,
                             CORE_CY - 450 + self.par[0].y() * 0.5),
                     self.A["glow_deep"], QRectF(self.A["glow_deep"].rect()))
        p.setOpacity(0.5 + 0.08 * math.sin(t * 0.7))
        p.drawPixmap(0, 0, self.A["bg_data"])
        p.setOpacity(1.0)
        p.drawPixmap(0, 0, self.A["vignette"])
        for d in self.far:
            a = 0.25 + 0.2 * math.sin(t * 0.9 + d["ph"])
            p.setOpacity(max(0.05, a) * 0.8)
            s = 2 + d["s"] * 3
            p.drawPixmap(QRectF(d["x"] + self.par[0].x() * 0.6,
                                d["y"] + self.par[0].y() * 0.6, s, s),
                         self.A["dot"], QRectF(self.A["dot"].rect()))
        p.setOpacity(1.0)

        # 2 COMPOSICION LUMINOSA: halo superior + pedestal + anillos base
        ht = self.A["halo_top"]
        ht_s = ht.width() * 0.88
        p.setOpacity(0.28 * P["glow"] * (0.9 + 0.1 * math.sin(t * 0.8)))
        p.drawPixmap(QRectF(CORE_CX - ht_s / 2 + self.par[1].x() * 0.5,
                            CORE_CY - 310 + self.par[1].y() * 0.3,
                            ht_s, ht_s * ht.height() / ht.width()),
                     ht, QRectF(ht.rect()))
        pd = self.A["pedestal"]
        p.setOpacity(0.95)
        p.drawPixmap(QPointF(CORE_CX - pd.width() / 2 + self.par[1].x() * 0.25,
                             H - pd.height() + 40 + self.par[1].y() * 0.1), pd)
        for name, oy, sp, al in (("base_ring_a", 60, 0.06, 0.7),
                                 ("base_ring_b", 26, -0.09, 0.6)):
            br = self.A[name]
            p.setOpacity(al * (0.85 + 0.15 * math.sin(t * 1.1 + oy)))
            p.drawPixmap(QPointF(CORE_CX - br.width() / 2 +
                                 self.par[1].x() * 0.15,
                                 H - 190 + oy + self.par[1].y() * 0.08), br)
        p.setOpacity(1.0)

        # 2b CORONA SUPERIOR: campo holografico suspendiendo el nucleo
        # halo amplio sobre el nucleo
        chalo = self.A["crown_halo"]
        chs = chalo.width() * 0.92
        p.setOpacity(0.30 * P["glow"] * (0.85 + 0.15 * math.sin(t * 0.9)))
        p.drawPixmap(QRectF(CORE_CX - chs / 2 + self.par[1].x() * 0.4,
                            CORE_CY - 330 + self.par[1].y() * 0.3,
                            chs, chs * chalo.height() / chalo.width()),
                     chalo, QRectF(chalo.rect()))
        p.setOpacity(1.0)
        # anillos de corona (detras del nucleo, rotaciones independientes)
        self.draw_ring(p, self.A["crown_a"],
                       CORE_CX + self.par[1].x() * 0.5,
                       CORE_CY - 238 + self.par[1].y() * 0.3,
                       t * 0.30, 0.62)
        self.draw_ring(p, self.A["crown_c"],
                       CORE_CX + self.par[1].x() * 0.6,
                       CORE_CY - 196 + self.par[1].y() * 0.35,
                       -t * 0.18, 0.46)
        self.draw_ring(p, self.A["crown_b"],
                       CORE_CX + self.par[1].x() * 0.45,
                       CORE_CY - 292 + self.par[1].y() * 0.25,
                       t * 0.44, 0.52)
        # haces verticales muy tenues corona -> nucleo
        p.setCompositionMode(QPainter.CompositionMode_Plus)
        for bx, bw, ba in ((-150, 46, 0.35), (0, 70, 0.5), (150, 46, 0.35)):
            cb = self.A["crown_beam"]
            p.setOpacity(ba * P["glow"] *
                         (0.8 + 0.2 * math.sin(t * 1.5 + bx)))
            p.drawPixmap(QRectF(CORE_CX + bx - bw / 2 + self.par[1].x() * 0.4,
                                CORE_CY - 320 + self.par[1].y() * 0.3,
                                bw, 250), cb, QRectF(cb.rect()))
        p.setCompositionMode(QPainter.CompositionMode_SourceOver)
        p.setOpacity(1.0)

        # haz vertical detras del nucleo
        vb = self.A["vbeam"]
        p.setCompositionMode(QPainter.CompositionMode_Plus)
        p.setOpacity(0.7 * P["glow"] * (0.85 + 0.15 * math.sin(t * 1.7)))
        p.drawPixmap(QPointF(CORE_CX - vb.width() / 2 + self.par[1].x() * 0.3,
                             CORE_CY - vb.height() * 0.52 + self.par[1].y() * 0.2),
                     vb)
        p.setCompositionMode(QPainter.CompositionMode_SourceOver)

        # 3 ANILLOS TRASEROS (detras del nucleo, planos distintos)
        self.draw_ring(p, self.A["ring_back_a"], CORE_CX + self.par[1].x(),
                       CORE_CY + self.par[1].y() + 30,
                       t * 0.35 * P["ring"], 0.20)
        self.draw_ring(p, self.A["ring_back_b"], CORE_CX + self.par[1].x() * 0.8,
                       CORE_CY + self.par[1].y() - 60,
                       -t * 0.22 * P["ring"], 0.26)
        self.draw_ring(p, self.A["ring_back_c"], CORE_CX + self.par[1].x() * 1.2,
                       CORE_CY + self.par[1].y() - 150,
                       t * 0.14 * P["ring"], 0.28)

        # 3b ORBITA TRASERA: sus nodos pasan por detras del nucleo
        dp = P["data"]
        self.draw_orbit(p, 0, self.orbits[0], dp, t, 0.50)

        # 4 NUCLEO: volumen construido por luz (sin globo solido)
        breathe = 1 + P["breathe"] * math.sin(t * 0.9) \
            + (0.012 * math.sin(t * 2.1) if self.state == 3 else 0)
        flicker = 0.94 + 0.06 * math.sin(t * 3.7) * math.sin(t * 1.3)
        cw = CORE_W * breathe
        chh = cw * (self.A["core"].height() / self.A["core"].width())
        key = "core_" + STATE_KEYS[self.state]
        # masa oscura luminosa, translucida
        p.setOpacity(1.0 * flicker)
        p.drawPixmap(QRectF(CORE_CX - cw / 2 + self.par[2].x(),
                            CORE_CY - chh / 2 + self.par[2].y(), cw, chh),
                     self.A[key], QRectF(self.A[key].rect()))
        # PLANO TRASERO: microdatos y nodos diminutos (muy tenues)
        msz_b = cw * 0.96
        p.setOpacity(0.85)
        p.drawPixmap(QRectF(CORE_CX - msz_b / 2 + self.par[2].x() * 1.05,
                            CORE_CY - msz_b / 2 + self.par[2].y() * 1.05,
                            msz_b, msz_b),
                     self.A["micro_back"], QRectF(self.A["micro_back"].rect()))
        # estructura interna: nodos conectados + arcos geodesicos
        p.save()
        p.translate(CORE_CX + self.par[2].x() * 0.9,
                    CORE_CY + self.par[2].y() * 0.9)
        p.rotate(math.degrees(-t * 0.04))
        msz = cw * 0.98
        p.setOpacity(1.0)
        p.drawPixmap(QRectF(-msz / 2, -msz / 2, msz, msz),
                     self.A["mesh"], QRectF(self.A["mesh"].rect()))
        p.restore()
        # PLANO MEDIO extra: arcos de energia internos (aditivos)
        p.save()
        p.translate(CORE_CX + self.par[2].x() * 0.85,
                    CORE_CY + self.par[2].y() * 0.85)
        p.rotate(math.degrees(t * 0.07))
        isz = cw * 0.80
        p.setCompositionMode(QPainter.CompositionMode_Plus)
        p.setOpacity(0.9 * flicker)
        p.drawPixmap(QRectF(-isz / 2, -isz / 2, isz, isz),
                     self.A["inner_arcs"], QRectF(self.A["inner_arcs"].rect()))
        p.setCompositionMode(QPainter.CompositionMode_SourceOver)
        p.restore()
        # fragmentos digitales en 2 planos (paralaje distinto, rotacion lenta)
        p.save()
        p.translate(CORE_CX + self.par[2].x() * 0.8,
                    CORE_CY + self.par[2].y() * 0.8)
        p.rotate(math.degrees(t * 0.05))
        fsz = cw * 1.02
        p.setOpacity(1.0)
        p.drawPixmap(QRectF(-fsz / 2, -fsz / 2, fsz, fsz),
                     self.A["fragments"], QRectF(self.A["fragments"].rect()))
        p.rotate(math.degrees(-t * 0.09))
        fsz2 = cw * 0.94
        p.setOpacity(0.7)
        p.drawPixmap(QRectF(-fsz2 / 2, -fsz2 / 2, fsz2, fsz2),
                     self.A["fragments2"], QRectF(self.A["fragments2"].rect()))
        p.restore()
        # energia central aditiva (el corazon luminoso)
        en = self.A["core_energy"]
        esz = cw * (0.68 + 0.06 * math.sin(t * 1.6)) * P["glow"]
        p.setCompositionMode(QPainter.CompositionMode_Plus)
        p.setOpacity((0.52 + 0.12 * math.sin(t * 2.1)) * flicker)
        p.drawPixmap(QRectF(CORE_CX - esz / 2 + self.par[2].x(),
                            CORE_CY - esz / 2 + self.par[2].y(), esz, esz),
                     en, QRectF(en.rect()))
        # arcos de borde discontinuos (sugieren envolvente, no la cierran)
        ra = self.A["rim_arcs"]
        rsz = cw * 1.00
        p.setOpacity(0.78 * flicker)
        p.drawPixmap(QRectF(CORE_CX - rsz / 2 + self.par[2].x() * 1.1,
                            CORE_CY - rsz / 2 + self.par[2].y() * 1.1,
                            rsz, rsz), ra, QRectF(ra.rect()))
        # microdescargas / pulsos cerca del corazon
        for i in range(6):
            ph = t * (1.3 + 0.37 * i) + i * 2.4
            a = max(0.0, math.sin(ph)) ** 3
            if a < 0.05:
                continue
            ang = i * 1.047 + t * 0.6
            rr = cw * (0.10 + 0.15 * (0.5 + 0.5 * math.sin(ph * 0.7)))
            x = CORE_CX + math.cos(ang) * rr + self.par[2].x()
            y = CORE_CY + math.sin(ang) * rr * 0.9 + self.par[2].y()
            s = 10 + 30 * a
            p.setOpacity(0.55 * a * flicker)
            p.drawPixmap(QRectF(x - s / 2, y - s / 2, s, s),
                         self.A["glow_small"], QRectF(self.A["glow_small"].rect()))
        p.setCompositionMode(QPainter.CompositionMode_SourceOver)
        # scanlines holograficas
        p.setOpacity(0.5 * flicker)
        p.drawPixmap(QRectF(CORE_CX - cw / 2 + self.par[2].x(),
                            CORE_CY - chh / 2 + self.par[2].y(), cw, chh),
                     self.A["scanlines"], QRectF(self.A["scanlines"].rect()))

        # brillo extra por estado (additivo sobre el nucleo)
        if P["tint_alpha"] > 0 and self.state in (1, 2, 3):
            gs = self.A["glow_small"]
            gsz = cw * 0.8
            p.setCompositionMode(QPainter.CompositionMode_Plus)
            p.setOpacity(0.22 * P["tint_alpha"] * P["glow"] *
                         (0.8 + 0.2 * math.sin(t * 2.4)))
            p.drawPixmap(QRectF(CORE_CX - gsz / 2 + self.par[2].x(),
                                CORE_CY - gsz / 2 + self.par[2].y(),
                                gsz, gsz), gs, QRectF(gs.rect()))
            p.setCompositionMode(QPainter.CompositionMode_SourceOver)

        # 5 SIMBOLO ATLAS: A integrada en el nucleo (capas + energia)
        if self.A.get("glyph_fill") is not None:
            glyph_w = CORE_W * 0.41 * breathe
            px, py = self.par[3].x() * 0.55, self.par[3].y() * 0.55
            cx0 = CORE_CX + px
            cy0 = CORE_CY - glyph_w * 0.01 + py
            gr = QRectF(cx0 - glyph_w / 2, cy0 - glyph_w / 2,
                        glyph_w, glyph_w)
            # pozo suave de contraste local (leve: parte del volumen)
            bk = self.A["glyph_backdrop"]
            bsz = glyph_w * 1.55
            p.setOpacity(0.50 * flicker)
            p.drawPixmap(QRectF(cx0 - bsz / 2, cy0 - bsz / 2, bsz, bsz),
                         bk, QRectF(bk.rect()))
            # copia trasera desplazada (geometria visible detras del simbolo)
            p.setOpacity(0.32)
            p.drawPixmap(QRectF(gr.x() + glyph_w * 0.024,
                                gr.y() + glyph_w * 0.012,
                                glyph_w, glyph_w),
                         self.A["glyph_back"], QRectF(self.A["glyph_back"].rect()))
            # energia atravesando el simbolo (aditiva, respirando)
            p.setCompositionMode(QPainter.CompositionMode_Plus)
            en = self.A["glyph_energy"]
            esz = glyph_w * 1.30
            p.setOpacity((0.46 + 0.14 * math.sin(t * 2.2)) * P["glow"] * flicker)
            p.drawPixmap(QRectF(cx0 - esz / 2, cy0 - esz / 2, esz, esz),
                         en, QRectF(en.rect()))
            # halo volumetrico controlado
            hsz = glyph_w * 1.16
            p.setOpacity(0.42 + 0.10 * math.sin(t * 1.9))
            p.drawPixmap(QRectF(cx0 - hsz / 2, cy0 - hsz / 2, hsz, hsz),
                         self.A["glyph_halo"], QRectF(self.A["glyph_halo"].rect()))
            p.setCompositionMode(QPainter.CompositionMode_SourceOver)
            # cuerpo translucido flotando en el nucleo
            pl = 0.60 + 0.10 * math.sin(t * 2.3)
            if self.state == 3:
                pl += 0.14 * max(0.0, math.sin(t * 2.3)) ** 2
            p.setOpacity(min(1.0, pl) * flicker)
            p.drawPixmap(gr, self.A["glyph_fill"],
                         QRectF(self.A["glyph_fill"].rect()))
            # doble contorno cian nitido (alpha ligeramente reducido)
            p.setOpacity(0.85)
            p.drawPixmap(gr, self.A["glyph_edge"],
                         QRectF(self.A["glyph_edge"].rect()))
            # contorno frontal adicional (profundidad, geometria delante)
            p.setOpacity(0.15)
            p.drawPixmap(QRectF(gr.x() - glyph_w * 0.020,
                                gr.y() - glyph_w * 0.010,
                                glyph_w, glyph_w),
                         self.A["glyph_back"], QRectF(self.A["glyph_back"].rect()))
            p.setOpacity(1.0)
        # fragmentos por DELANTE del simbolo (volumen delante de la A)
        p.setOpacity(0.55)
        fszf = cw * 1.02
        p.drawPixmap(QRectF(CORE_CX - fszf / 2 + self.par[3].x() * 0.9,
                            CORE_CY - fszf / 2 + self.par[3].y() * 0.9,
                            fszf, fszf),
                     self.A["fragments"], QRectF(self.A["fragments"].rect()))
        # arcos internos cruzando por delante de la A (aditivos, otro plano)
        p.setCompositionMode(QPainter.CompositionMode_Plus)
        p.save()
        p.translate(CORE_CX + self.par[3].x() * 0.7,
                    CORE_CY + self.par[3].y() * 0.7)
        p.rotate(math.degrees(t * 0.11))
        iszf = cw * 0.80
        p.setOpacity(0.42)
        p.drawPixmap(QRectF(-iszf / 2, -iszf / 2, iszf, iszf),
                     self.A["inner_arcs"], QRectF(self.A["inner_arcs"].rect()))
        p.restore()
        p.setCompositionMode(QPainter.CompositionMode_SourceOver)
        p.setOpacity(1.0)

        # PLANO DELANTERO: pocos nodos brillantes sobre el nucleo
        p.setCompositionMode(QPainter.CompositionMode_Plus)
        for fn in self.front_nodes:
            tw = 0.5 + 0.5 * math.sin(t * fn["sp"] + fn["ph"])
            if tw < 0.25:
                continue
            x = CORE_CX + math.cos(fn["th"] + t * 0.15) * fn["rr"] * cw * 0.5 \
                + self.par[3].x() * 0.7
            y = CORE_CY + math.sin(fn["th"] + t * 0.15) * fn["rr"] * cw * 0.44 \
                + self.par[3].y() * 0.7
            s = 10 + 22 * tw
            p.setOpacity(0.75 * tw)
            p.drawPixmap(QRectF(x - s / 2, y - s / 2, s, s),
                         self.A["glow_small"], QRectF(self.A["glow_small"].rect()))
        p.setCompositionMode(QPainter.CompositionMode_SourceOver)

        # 6 BARRIDO DE ENERGIA sobre el nucleo
        if -0.8 < self.sweep_t < 1.0:
            beam = self.A["beam"]
            prog = (self.sweep_t + 0.8) / 1.8
            bx = CORE_CX - CORE_W / 2 + prog * CORE_W - \
                beam.width() * 0.5 * breathe
            fade = math.sin(prog * math.pi)
            p.setCompositionMode(QPainter.CompositionMode_Plus)
            p.setOpacity(0.55 * fade)
            p.drawPixmap(QRectF(bx + self.par[2].x(),
                                CORE_CY - chh * 0.44 + self.par[2].y(),
                                beam.width() * breathe, chh * 0.9),
                         beam, QRectF(beam.rect()))
            p.setCompositionMode(QPainter.CompositionMode_SourceOver)
            p.setOpacity(1.0)

        # 7 PARTICULAS / DATOS EN ORBITA (delante del nucleo)
        dp = P["data"]
        for oi, o in enumerate(self.orbits):
            if oi == 0:
                continue  # ya dibujada por detras del nucleo
            self.draw_orbit(p, oi, o, dp, t, 0.60 if oi == 1 else 0.45)

        # 8 ANILLOS DELANTEROS (planos distintos, mezcla discontinuo/fino)
        self.draw_ring(p, self.A["ring_front_a"], CORE_CX + self.par[3].x(),
                       CORE_CY + self.par[3].y() + 10,
                       -t * 0.42 * P["ring"], 0.22)
        self.draw_ring(p, self.A["ring_front_b"], CORE_CX + self.par[3].x() * 1.1,
                       CORE_CY + self.par[3].y() - 90,
                       t * 0.6 * P["ring"], 0.28)
        self.draw_ring(p, self.A["ring_front_c"], CORE_CX + self.par[3].x() * 0.9,
                       CORE_CY + self.par[3].y() + 60,
                       -t * 0.20 * P["ring"], 0.15)
        # anillo de corona que pasa por DELANTE del nucleo (profundidad)
        self.draw_ring(p, self.A["crown_b"], CORE_CX + self.par[3].x() * 0.45,
                       CORE_CY - 288 + self.par[3].y() * 0.25,
                       t * 0.44 + 2.4, 0.16)

        # 8b IDENTIDAD BAJO EL NUCLEO (integrada sobre el pedestal)
        idy = 576
        p.setCompositionMode(QPainter.CompositionMode_Plus)
        p.setOpacity(0.45 * P["glow"] * (0.8 + 0.2 * math.sin(t * 1.1)))
        p.drawPixmap(QRectF(CORE_CX - 260 + self.par[1].x() * 0.2,
                            idy - 26, 520, 88),
                     self.A["glow_small"], QRectF(self.A["glow_small"].rect()))
        p.setCompositionMode(QPainter.CompositionMode_SourceOver)
        f_id = QFont("Consolas", 23)
        f_id.setBold(True)
        f_id.setLetterSpacing(QFont.AbsoluteSpacing, 12)
        p.setFont(f_id)
        p.setPen(QColor(205, 244, 255, 235))
        p.drawText(QRectF(0, idy, W, 34), Qt.AlignHCenter, "ATLAS")
        f_tag = QFont("Consolas", 9.5)
        f_tag.setLetterSpacing(QFont.AbsoluteSpacing, 3)
        p.setFont(f_tag)
        p.setPen(QColor(150, 226, 255, 205))
        p.drawText(QRectF(0, idy + 38, W, 18), Qt.AlignHCenter,
                   "TU ASISTENTE.  TU SISTEMA.  SIN LÍMITES.")
        # reglas laterales del tagline
        p.setPen(QPen(QColor(90, 200, 255, 110), 1))
        y_tag = idy + 47
        p.drawLine(QPointF(W / 2 - 260, y_tag), QPointF(W / 2 - 200, y_tag))
        p.drawLine(QPointF(W / 2 + 200, y_tag), QPointF(W / 2 + 260, y_tag))
        p.setOpacity(1.0)

        # acentos rojos AUTOMATION (localizados)
        if self.state == 4:
            p.setCompositionMode(QPainter.CompositionMode_Plus)
            gs = self.A["glow_small"]
            for i in range(3):
                ph = t * 1.4 + i * 2.1
                a = max(0.0, math.sin(ph)) ** 2
                p.setOpacity(0.35 * a)
                s = 220 + 60 * math.sin(t + i)
                x = CORE_CX + math.sin(t * 0.5 + i * 2.1) * CORE_W * 0.42
                y = CORE_CY + math.cos(t * 0.4 + i) * 150
                p.drawPixmap(QRectF(x - s / 2, y - s / 2, s, s), gs,
                             QRectF(gs.rect()))
            p.setCompositionMode(QPainter.CompositionMode_SourceOver)

        # 9 HUD RESERVADO
        # cabeceras de identidad (discretas, no compiten con el nucleo)
        p.setOpacity(0.92)
        p.drawPixmap(36, 18, self.A["hdr_left"])
        p.drawPixmap(W - 268, 26, self.A["hdr_right"])
        # paneles (acompanan al nucleo, no compiten)
        p.setOpacity(0.92)
        p.drawPixmap(40, 148, self.A["panel_left"])
        p.drawPixmap(W - 420, 168, self.A["panel_right"])
        p.drawPixmap(40, 414, self.A["panel_activity_red" if self.state == 4
                                          else "panel_activity"])
        if self.state == 4:
            p.setOpacity(0.95)
            p.drawPixmap(W - 420, 480, self.A["panel_auto"])
        p.setOpacity(0.94)
        p.drawPixmap(0, H - 150, self.A["hud_bottom"])
        p.setOpacity(1.0)
        if self.status_message is not None:
            if self.t < self.status_until:
                f_err = QFont("Consolas", 10)
                f_err.setBold(True)
                f_err.setLetterSpacing(QFont.AbsoluteSpacing, 1)
                p.setFont(f_err)
                p.setPen(QColor(255, 120, 140, 235))
                p.drawText(QRectF(0, H - 186, W, 20), Qt.AlignHCenter,
                           self.status_message)
            else:
                self.status_message = None
        # overlays del menu: hover + active (solo el boton afectado)
        hover_i = self._menu_index_at(self.mouse)
        if self.active_menu is not None:
            i = self.active_menu
            draw_menu_button(p, self.menu_rects[i], self.A["menu_icons_hi"][i],
                             MENU_ITEMS[i], mode="active")
        if hover_i is not None and hover_i != self.active_menu:
            draw_menu_button(p, self.menu_rects[hover_i],
                             self.A["menu_icons_hi"][hover_i],
                             MENU_ITEMS[hover_i], mode="hover")

        # 9b MODULO RELOJ (inferior izquierda, hora/fecha reales del PC)
        p.setOpacity(0.95)
        p.drawPixmap(26, H - 82, self.A["clock_frame"])
        p.setOpacity(1.0)
        now_dt = datetime.now()
        f_clk = QFont("Consolas", 16)
        f_clk.setBold(True)
        f_clk.setLetterSpacing(QFont.AbsoluteSpacing, 2)
        p.setFont(f_clk)
        p.setPen(QColor(210, 245, 255, 240))
        p.drawText(QRectF(26 + 14, H - 82 + 12, 190, 22), Qt.AlignLeft,
                   now_dt.strftime("%H:%M"))
        f_dat = QFont("Consolas", 8)
        f_dat.setLetterSpacing(QFont.AbsoluteSpacing, 1)
        p.setFont(f_dat)
        p.setPen(QColor(150, 226, 255, 190))
        fecha = "%s, %d %s %d" % (DIAS[now_dt.weekday()], now_dt.day,
                                  MESES[now_dt.month - 1], now_dt.year)
        p.drawText(QRectF(26 + 14, H - 82 + 36, 190, 14), Qt.AlignLeft, fecha)

        # 9c MODULO ESTADO DE VOZ (inferior derecha, segun STATE real)
        red = self.state == 4
        p.setOpacity(0.95)
        p.drawPixmap(W - 326, H - 78,
                     self.A["voice_frame_red" if red else "voice_frame"])
        p.setOpacity(1.0)
        # mini waveform animada
        bars = 15
        amp = 11 + 4 * math.sin(t * 5)
        wcol = QColor(255, 90, 110) if red else QColor(120, 220, 255)
        for bi in range(bars):
            ph = t * (5.2 + 0.4 * (bi % 3)) + bi * 0.9
            hh = amp * (0.25 + 0.75 * abs(math.sin(ph))) * \
                (0.6 + 0.4 * math.sin(bi * 0.7 + t * 1.3))
            bx = W - 326 + 14 + bi * 4.4
            byc = H - 78 + 27
            p.setPen(QPen(QColor(wcol.red(), wcol.green(), wcol.blue(),
                                 200), 2.0))
            p.drawLine(QPointF(bx, byc - hh / 2), QPointF(bx, byc + hh / 2))
        # etiqueta de estado
        f_voz = QFont("Consolas", 9.5)
        f_voz.setBold(True)
        f_voz.setLetterSpacing(QFont.AbsoluteSpacing, 2)
        p.setFont(f_voz)
        if red:
            p.setPen(QColor(255, 150, 165, 240))
        else:
            p.setPen(QColor(200, 244, 255, 235))
        p.drawText(QRectF(W - 326 + 92, H - 78 + 18, 195, 18), Qt.AlignLeft,
                   VOICE_STATUS[STATE_NAMES[self.state]])

        # chips de datos flotantes (ambiente)
        p.setOpacity(1.0)
        for c in self.chips:
            yy = (c["y"] - t * c["spd"]) % (H + 60) - 30
            xx = c["x"] + math.sin(t * 0.4 + c["ph"]) * c["drift"] + \
                self.par[0].x() * 0.4
            a = 0.18 + 0.22 * (0.5 + 0.5 * math.sin(t * 0.8 + c["ph"] * 2))
            p.setOpacity(a)
            p.drawPixmap(QRectF(xx, yy, 14, 12), self.A["chip"],
                         QRectF(self.A["chip"].rect()))
        p.setOpacity(1.0)

        # titulo estado
        f = QFont("Consolas", 11)
        f.setBold(True)
        f.setLetterSpacing(QFont.AbsoluteSpacing, 3)
        p.setFont(f)
        p.setPen(QColor(170, 235, 255, 220))
        p.setOpacity(0.9)
        p.drawText(QRectF(0, 12, W, 22), Qt.AlignHCenter,
                   "ESTADO: " + STATE_NAMES[self.state])
        p.setOpacity(1.0)
        p.end()

    def draw_orbit(self, p, oi, o, dp, t, alpha):
        """Orbita con nodos moviles y linea segmentada (tramos ocultos)."""
        pts = []
        for k in range(o["n"]):
            th = o["off"] + t * o["speed"] * dp + k * 6.283 / o["n"]
            x = CORE_CX + self.par[2].x() * 0.7 + o["rx"] * math.cos(th)
            y = CORE_CY + self.par[2].y() * 0.7 + o["ry"] * math.sin(th) \
                + o["rx"] * o["tilt"] * math.sin(th) * 0.25
            pts.append(QPointF(x, y))
        p.setPen(QPen(QColor(90, 210, 255, 40), 1))
        if oi != 1:
            for k in range(len(pts) - 1):
                if k % 3 == 2:
                    continue  # pequenos tramos ocultos
                p.setOpacity(alpha * (0.55 + 0.30 * math.sin(t * 1.1 + k)))
                p.drawLine(pts[k], pts[k + 1])
        for k, pt in enumerate(pts):
            tw = 0.5 + 0.5 * math.sin(t * 2.2 + k * 1.7 + oi)
            s = 4 + 5 * tw
            p.setOpacity(alpha * (0.35 + 0.5 * tw))
            p.drawPixmap(QRectF(pt.x() - s / 2, pt.y() - s / 2, s, s),
                         self.A["dot"], QRectF(self.A["dot"].rect()))
        p.setOpacity(1.0)

    def draw_ring(self, p, spr, cx, cy, phase, alpha):
        """Anillo cacheado 1:1 + destello cometa recorriendo la elipse."""
        w, h = spr.width(), spr.height()
        rw, rh = w - 120, h - 120
        p.setOpacity(alpha)
        p.drawPixmap(QPointF(cx - w / 2, cy - h / 2), spr)
        a = phase % (2 * math.pi)
        ex = rw / 2 * math.cos(a)
        ey = rh / 2 * math.sin(a)
        p.setCompositionMode(QPainter.CompositionMode_Plus)
        p.setOpacity(alpha * 1.4)
        s = 26 + 10 * math.sin(self.t * 6)
        p.drawPixmap(QRectF(cx + ex - s, cy + ey - s, s * 2, s * 2),
                     self.A["glow_small"], QRectF(self.A["glow_small"].rect()))
        p.setCompositionMode(QPainter.CompositionMode_SourceOver)
        p.setOpacity(1.0)

    def _menu_index_at(self, pos):
        for i, r in enumerate(self.menu_rects):
            if r.contains(pos):
                return i
        return None

    def mouseMoveEvent(self, e):
        self.mouse = e.position()

    def mousePressEvent(self, e):
        i = self._menu_index_at(e.position())
        if i is None:
            return
        self.active_menu = None if self.active_menu == i else i
        if i == 0:
            self.chat_requested.emit()

    def set_status_error(self, message: str) -> None:
        self.status_message = str(message)
        self.status_until = self.clock.elapsed() / 1000.0 + 4.0
        self.update()

    def _set_state(self, idx):
        self.state = idx
        system_state["ESTADO"] = STATE_NAMES[idx]
        self.A["panel_left"] = system_panel_sprite(
            312, 196, "SYSTEM CORE STATUS", QColor(60, 200, 255),
            footer="// TELEMETRY: WAITING BACKEND")

    def keyPressEvent(self, e):
        k = e.key()
        if k == Qt.Key_Escape:
            self.close()
        elif Qt.Key_1 <= k <= Qt.Key_5:
            self._set_state(int(k) - int(Qt.Key_1))
            self.setWindowTitle("ATLAS Stage V2 - ESTADO: " +
                                STATE_NAMES[self.state])


def main():
    QApplication.setStyle("Fusion")
    app = QApplication(sys.argv)
    w = LayeredDemo()
    # utilidad de prueba: estado inicial via CLI (1..5), no afecta la demo
    for a in sys.argv[1:]:
        if a in ("1", "2", "3", "4", "5"):
            w._set_state(int(a) - 1)
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
