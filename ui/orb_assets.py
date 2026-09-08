"""Carga y composición de assets del Orbe V2 (capas transparentes).

Infraestructura de la capa gráfica por assets: los PNG viven en
``ui/assets/orbe_v2`` con alpha real. El renderer del orbe solo pide
``state_bundle`` y compone las capas con transformes Qt. Si un asset falta,
se reporta en :func:`missing_assets` y el orbe cae al renderer procedural
sin romper nada. Los imports de Qt son perezosos para no acoplar el resto
de Atlas a PySide6.
"""
from __future__ import annotations

from pathlib import Path

ASSET_DIR = Path(__file__).resolve().parent / "assets" / "orbe_v2"

# Capa -> (nombre de archivo, fuerza de tinte del estado 0..1).
# El tinte reutiliza el color semántico del estado (_STATE_COLORS) sin
# aplanar la capa: solo SourceAtop conserva alpha y sombreado.
#
# La capa "hero" es el asset APROBADO completo (esfera + logo + glow +
# órbitas + partículas + pedestal en una sola imagen). Cuando está presente
# el renderer la usa como composición única y no dibuja las capas
# provisionales (que quedan como fallback si el hero falta).
_HERO_ASSET = "atlas_orbe_approved.png"
_ASSET_SPECS: dict[str, tuple[str, float]] = {
    "hero": (_HERO_ASSET, 0.45),
    "globe_body": ("globe_body.png", 0.30),
    "globe_grid": ("globe_grid.png", 0.55),
    "atlas_logo": ("atlas_logo.png", 0.22),
    "inner_glow": ("inner_glow.png", 0.45),
    "particles": ("particles.png", 0.55),
}

# Estados con identidad propia frente al cian base del asset.
_TINT_STATES = frozenset({"SPEAKING", "AUTHORIZATION", "AUTOMATION", "DEGRADED"})

_PIXMAP_CACHE: dict[str, object] = {}
_BUNDLE_CACHE: dict[tuple[str, str], object] = {}


def missing_assets() -> list[str]:
    """Asset names that are absent on disk (empty list = carga completa)."""
    return sorted(name for spec in _ASSET_SPECS.values() for name in (spec[0],) if not (ASSET_DIR / name).is_file())


def hero_available() -> bool:
    """True cuando el asset aprobado (composición única) está en disco."""
    return load_asset(_HERO_ASSET) is not None


def load_asset(name: str):
    """Return the raw QPixmap for one layer, or None when missing."""
    from PySide6.QtGui import QPixmap

    if name in _PIXMAP_CACHE:
        return _PIXMAP_CACHE[name]
    path = ASSET_DIR / name
    if not path.is_file():
        _PIXMAP_CACHE[name] = None
        return None
    pixmap = QPixmap(str(path))
    if pixmap.isNull():
        _PIXMAP_CACHE[name] = None
        return None
    pixmap.setDevicePixelRatio(1.0)
    _PIXMAP_CACHE[name] = pixmap
    return pixmap


def _tinted(pixmap, rgb: tuple[int, int, int], strength: float, darken=None):
    """Colorize a copy preserving alpha (SourceAtop); optional dark pass first."""
    from PySide6.QtGui import QColor, QImage, QPainter, QPixmap

    image = QImage(pixmap.toImage())
    painter = QPainter(image)
    painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceAtop)
    if darken is not None:
        painter.fillRect(image.rect(), QColor(*darken, 150))
    painter.fillRect(image.rect(), QColor(*rgb, int(255 * strength)))
    painter.end()
    result = QPixmap.fromImage(image)
    result.setDevicePixelRatio(1.0)
    return result


def state_bundle(state) -> dict:
    """Layers tinted for one visual state; values may be None if an asset is missing.

    IDLE/LISTENING/... usan el cian base del asset; SPEAKING/AUTHORIZATION/
    AUTOMATION/DEGRADED reciben el color semántico y AUTOMATION oscurece el
    cuerpo (núcleo oscuro + rojo luminoso, sin capa roja plana).
    """
    from PySide6.QtGui import QPixmap  # noqa: F401 ( mantiene Qt cargado )
    from ui.orbe_app import color_for_state

    state_name = str(getattr(state, "name", state)).upper()
    cache_key = (state_name, "v2")
    if cache_key in _BUNDLE_CACHE:
        return _BUNDLE_CACHE[cache_key]

    red, green, blue, _alpha = color_for_state(state)
    needs_tint = state_name in _TINT_STATES
    bundle: dict[str, object] = {}
    for layer, (filename, strength) in _ASSET_SPECS.items():
        raw = load_asset(filename)
        if raw is None:
            bundle[layer] = None
            continue
        if needs_tint:
            darken = (12, 3, 8) if (state_name == "AUTOMATION" and layer in ("globe_body", "inner_glow", "hero")) else None
            bundle[layer] = _tinted(raw, (red, green, blue), strength, darken=darken)
        else:
            bundle[layer] = raw
    _BUNDLE_CACHE[cache_key] = bundle
    return bundle
