"""Tests de la capa de assets del Orbe V2 (composición por capas)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

pytest.importorskip("PySide6")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from use_cases.ui_state_mapper import OrbVisualState
from ui import orb_assets, orbe_app


@pytest.fixture(scope="module")
def qapp():
    app = orbe_app.create_application([])
    yield app


@pytest.fixture()
def orb(qapp):
    window = orbe_app.create_orb_window()
    yield window
    window.close()


def _expected_asset_names() -> list[str]:
    return sorted(filename for filename, _strength in orb_assets._ASSET_SPECS.values())


def test_assets_are_present() -> None:
    assert orb_assets.missing_assets() == []


def test_every_asset_is_rgba_with_real_alpha_and_no_background(qapp) -> None:
    from PySide6.QtGui import QImage

    for layer, (filename, _strength) in orb_assets._ASSET_SPECS.items():
        image = QImage(str(orb_assets.ASSET_DIR / filename))
        assert not image.isNull(), layer
        assert image.hasAlphaChannel(), layer
        # The corners stay fully transparent: never a rectangular background.
        assert image.pixelColor(2, 2).alpha() == 0, layer
        assert image.pixelColor(image.width() - 3, 2).alpha() == 0, layer
        # The body/logo layers keep content near the centre.
        if layer in ("globe_body", "atlas_logo", "inner_glow"):
            assert image.pixelColor(image.width() // 2, image.height() // 2).alpha() > 0, layer
        else:  # grid/particles are sparse: some sampled pixel must carry alpha
            small = image.scaled(24, 24)
            assert any(small.pixelColor(x, y).alpha() > 0 for x in range(24) for y in range(24)), layer


def test_load_asset_caches_pixmaps(qapp) -> None:
    first = orb_assets.load_asset("globe_body.png")
    second = orb_assets.load_asset("globe_body.png")
    assert first is not None and first is second
    assert first.hasAlphaChannel()


def test_missing_assets_are_reported_by_name(qapp, monkeypatch) -> None:
    monkeypatch.setattr(orb_assets, "ASSET_DIR", Path("Z:/atlas/assets/inexistentes"))
    monkeypatch.setattr(orb_assets, "_PIXMAP_CACHE", {})
    monkeypatch.setattr(orb_assets, "_BUNDLE_CACHE", {})
    assert orb_assets.missing_assets() == _expected_asset_names()
    assert orb_assets.load_asset("globe_body.png") is None


def test_state_bundle_covers_every_state_and_every_layer(qapp) -> None:
    for state in OrbVisualState:
        bundle = orb_assets.state_bundle(state)
        assert set(bundle) == set(orb_assets._ASSET_SPECS)
        assert all(pixmap is not None for pixmap in bundle.values())


def test_state_bundle_is_cached_per_state(qapp) -> None:
    first = orb_assets.state_bundle(OrbVisualState.IDLE)
    second = orb_assets.state_bundle(OrbVisualState.IDLE)
    assert first is second
    # Tinted states rebuild their layers; the neutral states share the raw asset.
    assert first["globe_body"] is orb_assets.state_bundle(OrbVisualState.PROCESSING)["globe_body"]
    assert first["globe_body"] is not orb_assets.state_bundle(OrbVisualState.SPEAKING)["globe_body"]


def test_speaking_tints_layers_green_and_automation_tints_red_with_dark_core(qapp) -> None:
    from PySide6.QtGui import QImage

    center = lambda image: image.pixelColor(image.width() // 2, image.height() // 2)  # noqa: E731
    idle_body = QImage(orb_assets.state_bundle(OrbVisualState.IDLE)["globe_body"].toImage())
    speaking_body = QImage(orb_assets.state_bundle(OrbVisualState.SPEAKING)["globe_body"].toImage())
    automation_body = QImage(orb_assets.state_bundle(OrbVisualState.AUTOMATION)["globe_body"].toImage())

    assert center(speaking_body).green() > center(idle_body).green()
    assert center(automation_body).red() > center(idle_body).red()
    # The red state keeps a dark core: not a flat red overlay.
    core = center(automation_body)
    assert core.red() < 120 and core.value() < 140


def test_hero_asset_is_present_with_alpha_and_drives_state_variants(qapp) -> None:
    from PySide6.QtGui import QImage

    assert orb_assets.hero_available()
    image = QImage(str(orb_assets.ASSET_DIR / "atlas_orbe_approved.png"))
    assert not image.isNull() and image.hasAlphaChannel()
    # No rectangular background: exterior corners stay fully transparent.
    assert image.pixelColor(2, 2).alpha() == 0
    assert image.pixelColor(image.width() - 3, image.height() - 3).alpha() == 0
    # The sphere carries the content near the centre.
    assert image.pixelColor(image.width() // 2, image.height() // 2).alpha() > 0
    # Neutral states share the raw hero; tinted states rebuild it (tint/composición).
    idle_hero = orb_assets.state_bundle(OrbVisualState.IDLE)["hero"]
    assert idle_hero is orb_assets.state_bundle(OrbVisualState.LISTENING)["hero"]
    assert idle_hero is not orb_assets.state_bundle(OrbVisualState.AUTOMATION)["hero"]


def test_orb_renders_every_state_through_the_asset_layers(orb) -> None:
    assert orb_assets.missing_assets() == []
    for state in OrbVisualState:
        orb.apply_state(state)
        orb.repaint()  # asset composition must not raise for any state
        assert orb.state is state


def test_atlas_logo_asset_matches_the_emblem_geometry(orb) -> None:
    from PySide6.QtCore import QPointF

    emblem = orbe_app.atlas_emblem_path(512)
    # Deterministic geometry shared with the asset generator.
    assert emblem == orbe_app.atlas_emblem_path(512)
    assert emblem.contains(QPointF(512 * 0.5, 512 * 0.45))
    # The class still builds the exact same path.
    assert orb._emblem_path == orbe_app.atlas_emblem_path(orb.width())


def test_device_geometry_keeps_sphere_clear_of_the_platform_and_clickable_disc() -> None:
    assert orbe_app.DEVICE_RADIUS_FACTOR < orbe_app.PLATFORM_BASE_FACTOR - 0.5
    # Sphere stays inside the clickable central disc.
    assert orbe_app.DEVICE_RADIUS_FACTOR < orbe_app.CLICKABLE_RADIUS_FACTOR
