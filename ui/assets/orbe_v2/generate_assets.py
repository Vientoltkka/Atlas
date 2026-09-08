"""Generador offline de assets del Orbe V2 (capas transparentes).

Ejecutar manualmente SOLO cuando se regeneren assets:
    set QT_QPA_PLATFORM=offscreen && python ui/assets/orbe_v2/generate_assets.py

Cada PNG sale con alpha real y sin fondo; el runtime (ui.orb_assets) solo
los carga y compone. No se usa en producción.
"""
from __future__ import annotations

import math
import os
import random
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import (
    QConicalGradient,
    QColor,
    QImage,
    QPainter,
    QPainterPath,
    QPen,
    QRadialGradient,
)

OUT_DIR = Path(__file__).resolve().parent
GLOBE = 1024
LOGO = 512
GLOW = 512
PARTICLES = 512

# Atlas chevron (same geometry contract as ui.orbe_app.atlas_emblem_path).
EMBLEM_ARM_TOP = 0.4418
EMBLEM_ARM_OUT = 0.5531
EMBLEM_ARM_WIDE = 0.5625
EMBLEM_ARM_IN = 0.4868
EMBLEM_TRI_TOP = 0.5701
EMBLEM_TRI_BASE = 0.6092
EMBLEM_HALF = 0.0731


def atlas_emblem_path(size: float) -> QPainterPath:
    def px(fraction: float) -> float:
        return size * fraction

    path = QPainterPath()
    path.moveTo(px(0.5), px(EMBLEM_ARM_TOP))
    path.lineTo(px(0.5 - EMBLEM_HALF), px(EMBLEM_ARM_OUT))
    path.lineTo(px(0.5 - 0.0433), px(EMBLEM_ARM_WIDE))
    path.lineTo(px(0.5), px(EMBLEM_ARM_IN))
    path.closeSubpath()
    path.moveTo(px(0.5), px(EMBLEM_ARM_TOP))
    path.lineTo(px(0.5 + EMBLEM_HALF), px(EMBLEM_ARM_OUT))
    path.lineTo(px(0.5 + 0.0433), px(EMBLEM_ARM_WIDE))
    path.lineTo(px(0.5), px(EMBLEM_ARM_IN))
    path.closeSubpath()
    path.moveTo(px(0.5), px(EMBLEM_TRI_TOP))
    path.lineTo(px(0.5 - 0.0255), px(EMBLEM_TRI_BASE))
    path.lineTo(px(0.5 + 0.0255), px(EMBLEM_TRI_BASE))
    path.closeSubpath()
    return path


def new_image(size: int) -> QImage:
    image = QImage(size, size, QImage.Format.Format_ARGB32_Premultiplied)
    image.fill(Qt.GlobalColor.transparent)
    return image


def draw_globe_body() -> QImage:
    image = new_image(GLOBE)
    painter = QPainter(image)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    center, r = GLOBE / 2.0, GLOBE / 2.0 - 4
    # A. Volumetric dark glass sphere with offset light and luminous rim.
    sphere = QRadialGradient(center - r * 0.26, center - r * 0.32, r * 1.30)
    sphere.setColorAt(0.00, QColor(3, 10, 30, 252))
    sphere.setColorAt(0.34, QColor(4, 22, 62, 250))
    sphere.setColorAt(0.62, QColor(7, 44, 104, 240))
    sphere.setColorAt(0.82, QColor(12, 82, 152, 210))
    sphere.setColorAt(0.94, QColor(64, 190, 255, 140))
    sphere.setColorAt(1.00, QColor(140, 235, 255, 0))
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(sphere)
    painter.drawEllipse(QRectF(center - r, center - r, r * 2, r * 2))

    painter.save()
    clip = QPainterPath()
    clip.addEllipse(QRectF(center - r, center - r, r * 2, r * 2))
    painter.setClipPath(clip)
    # B. Vignette: darker lower limb sells 3D volume.
    vignette = QRadialGradient(center, center + r * 0.08, r)
    vignette.setColorAt(0.00, QColor(1, 4, 14, 0))
    vignette.setColorAt(0.58, QColor(1, 4, 14, 0))
    vignette.setColorAt(0.88, QColor(1, 4, 14, 110))
    vignette.setColorAt(1.00, QColor(1, 4, 14, 185))
    painter.setBrush(vignette)
    painter.drawEllipse(QRectF(center - r, center - r, r * 2, r * 2))
    # C. Sheen: soft angular bands over the glass.
    sheen = QConicalGradient(center, center, -48.0)
    sheen.setColorAt(0.00, QColor(255, 255, 255, 16))
    sheen.setColorAt(0.16, QColor(120, 225, 255, 0))
    sheen.setColorAt(0.52, QColor(2, 12, 40, 30))
    sheen.setColorAt(0.74, QColor(120, 225, 255, 0))
    sheen.setColorAt(1.00, QColor(255, 255, 255, 16))
    painter.setBrush(sheen)
    painter.drawEllipse(QRectF(center - r, center - r, r * 2, r * 2))
    # D. Inner illumination along the upper-left limb.
    painter.setBrush(Qt.BrushStyle.NoBrush)
    painter.setPen(QPen(QColor(190, 245, 255, 85), 5.0))
    painter.drawArc(QRectF(center - r * 0.965, center - r * 0.965, r * 1.93, r * 1.93), 115 * 16, 58 * 16)
    painter.restore()

    # E. Specular pocket near the light source.
    highlight = QRadialGradient(center - r * 0.42, center - r * 0.44, r * 0.30)
    highlight.setColorAt(0.00, QColor(255, 255, 255, 40))
    highlight.setColorAt(0.45, QColor(255, 255, 255, 12))
    highlight.setColorAt(1.00, QColor(255, 255, 255, 0))
    painter.setBrush(highlight)
    painter.drawEllipse(QRectF(center - r * 0.42 - r * 0.30, center - r * 0.44 - r * 0.30, r * 0.60, r * 0.60))
    painter.end()
    return image


def draw_globe_grid() -> QImage:
    image = new_image(GLOBE)
    painter = QPainter(image)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    center, r = GLOBE / 2.0, GLOBE / 2.0 - 6
    painter.save()
    clip = QPainterPath()
    clip.addEllipse(QRectF(center - r, center - r, r * 2, r * 2))
    painter.setClipPath(clip)

    rng = random.Random(20260907)
    nodes: list[tuple[float, float]] = []
    # Digital dot sphere: latitude rings of projected dots.
    row = 0
    latitude = -78.0
    while latitude <= 78.0:
        radians = math.radians(latitude)
        span = math.cos(radians) * r
        y = center - math.sin(radians) * r * 0.94
        squash = 0.22
        step = max(9.0, 26.0 * span / r * 1.4)
        count = int(2.0 * span / step) + 1
        for index in range(count):
            t = (index / max(1, count - 1)) * 2.0 - 1.0
            x = center + t * span
            dot = 2.6 + 2.2 * (1.0 - abs(t) * 0.4)
            edge = 1.0 - abs(math.hypot(t * span, (y - center) / squash)) / (r * 1.25)
            alpha = max(0, min(255, int(120 * max(0.18, edge))))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(120, 222, 255, alpha))
            painter.drawEllipse(QPointF(x, y + (row % 2) * 1.5), dot, dot)
            if rng.random() < 0.05:
                nodes.append((x, y))
        row += 1
        latitude += 6.0

    # Bright network nodes with halo.
    painter.setPen(Qt.PenStyle.NoPen)
    for x, y in nodes:
        halo = QRadialGradient(x, y, 14)
        halo.setColorAt(0.0, QColor(210, 250, 255, 220))
        halo.setColorAt(0.35, QColor(120, 222, 255, 120))
        halo.setColorAt(1.0, QColor(120, 222, 255, 0))
        painter.setBrush(halo)
        painter.drawEllipse(QPointF(x, y), 14, 14)
        painter.setBrush(QColor(235, 252, 255, 235))
        painter.drawEllipse(QPointF(x, y), 3.2, 3.2)
    # Connection lines between nearby nodes.
    painter.setBrush(Qt.BrushStyle.NoBrush)
    for i, (x0, y0) in enumerate(nodes):
        for x1, y1 in nodes[i + 1:]:
            dist = math.hypot(x1 - x0, y1 - y0)
            if 40 < dist < 150 and rng.random() < 0.5:
                painter.setPen(QPen(QColor(140, 232, 255, 60), 1.2))
                painter.drawLine(QPointF(x0, y0), QPointF(x1, y1))
    # Faint meridians.
    for meridian in (0.30, 0.62, 0.88):
        painter.setPen(QPen(QColor(120, 222, 255, 34), 2.0))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawEllipse(QRectF(center - meridian * r, center - r, meridian * r * 2.0, r * 2.0))
    painter.restore()
    painter.end()
    return image


def draw_atlas_logo() -> QImage:
    image = new_image(LOGO)
    painter = QPainter(image)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    emblem = atlas_emblem_path(LOGO)
    center = LOGO / 2.0
    # Wide outer glow: stacked translucent strokes.
    painter.setBrush(Qt.BrushStyle.NoBrush)
    for width, alpha in ((26.0, 26), (16.0, 46), (9.0, 76)):
        painter.setPen(QPen(QColor(80, 205, 255, alpha), width, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
        painter.drawPath(emblem)
    # Luminous fill.
    fill = QRadialGradient(center, center - LOGO * 0.03, LOGO * 0.42)
    fill.setColorAt(0.00, QColor(248, 253, 255, 255))
    fill.setColorAt(0.45, QColor(140, 230, 255, 250))
    fill.setColorAt(1.00, QColor(60, 180, 255, 235))
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(fill)
    painter.drawPath(emblem)
    # Bright inner border.
    painter.setBrush(Qt.BrushStyle.NoBrush)
    painter.setPen(QPen(QColor(255, 255, 255, 235), 3.0))
    painter.drawPath(emblem)
    painter.end()
    return image


def draw_inner_glow() -> QImage:
    image = new_image(GLOW)
    painter = QPainter(image)
    center, radius = GLOW / 2.0, GLOW / 2.0 - 2
    glow = QRadialGradient(center, center, radius)
    glow.setColorAt(0.00, QColor(226, 250, 255, 205))
    glow.setColorAt(0.30, QColor(140, 232, 255, 120))
    glow.setColorAt(0.62, QColor(60, 180, 255, 45))
    glow.setColorAt(1.00, QColor(60, 180, 255, 0))
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(glow)
    painter.drawEllipse(QRectF(0, 0, GLOW, GLOW))
    painter.end()
    return image


def draw_particles() -> QImage:
    image = new_image(PARTICLES)
    painter = QPainter(image)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    center, radius = PARTICLES / 2.0, PARTICLES / 2.0 - 6
    rng = random.Random(424242)
    painter.setPen(Qt.PenStyle.NoPen)
    for _ in range(120):
        angle = rng.uniform(0, math.tau)
        dist = radius * math.sqrt(rng.uniform(0.02, 1.0))
        x = center + math.cos(angle) * dist
        y = center + math.sin(angle) * dist * 0.92
        dot = rng.uniform(1.2, 3.4)
        alpha = int(rng.uniform(60, 190) * (1.0 - dist / (radius * 1.6)))
        painter.setBrush(QColor(120, 224, 255, max(24, alpha)))
        painter.drawEllipse(QPointF(x, y), dot * 2.2, dot * 2.2)
        painter.setBrush(QColor(235, 252, 255, min(255, alpha + 60)))
        painter.drawEllipse(QPointF(x, y), dot, dot)
    # A few tiny glints.
    for _ in range(10):
        angle = rng.uniform(0, math.tau)
        dist = radius * math.sqrt(rng.uniform(0.1, 0.9))
        x = center + math.cos(angle) * dist
        y = center + math.sin(angle) * dist
        painter.setPen(QPen(QColor(255, 255, 255, 170), 1.4))
        painter.drawLine(QPointF(x - 5, y), QPointF(x + 5, y))
        painter.drawLine(QPointF(x, y - 5), QPointF(x, y + 5))
    painter.end()
    return image


def main() -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    assets = {
        "globe_body.png": draw_globe_body(),
        "globe_grid.png": draw_globe_grid(),
        "atlas_logo.png": draw_atlas_logo(),
        "inner_glow.png": draw_inner_glow(),
        "particles.png": draw_particles(),
    }
    for name, image in assets.items():
        target = OUT_DIR / name
        if not image.save(str(target), "PNG"):
            raise SystemExit(f"no se pudo guardar {target}")
        print(f"OK  {target.name}  {image.width()}x{image.height()} RGBA")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
