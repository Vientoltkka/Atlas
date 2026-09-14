# -*- coding: utf-8 -*-
"""Test focal del routing Ctrl+Espacio ON/OFF de ATLAS_STAGE_V2.

Ejecutar:  python -m pytest ui/stage_v2/test_routing_v2.py -q

Contrato (V5):
- Ctrl+Espacio SIEMPRE muestra/oculta el orbe flotante V5, con o sin
  ATLAS_STAGE_V2; Stage V2 ya no secuestra el hotkey.
- ATLAS_STAGE=1 restaura el overlay legacy fullscreen opt-in.
"""
from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

pytest.importorskip("PySide6")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from ui import orbe_app, orbe_controller


@pytest.fixture(scope="module")
def qapp():
    return orbe_app.create_application([])


class FakeStageV2:
    """Doble de AtlasStageV2: solo contrato de visibilidad."""

    def __init__(self):
        self.visible = False

    def show(self):
        self.visible = True

    def hide(self):
        self.visible = False

    def isVisible(self):
        return self.visible

    def raise_(self):
        pass


def _controller(qapp, stage_v2_factory=None):
    orb = orbe_app.create_orb_window()
    panel = orbe_app.create_transcript_panel()
    controller = orbe_controller.OrbeController(
        atlas=SimpleNamespace(),
        application=qapp,
        orb=orb,
        transcript_panel=panel,
        stage_v2_factory=stage_v2_factory,
    )
    return controller, orb, panel


def _atlas_visible(controller) -> bool:
    """Atlas interface visible: MASTER HUD (default) or legacy floating orb."""
    hud = controller.master_hud
    if hud is not None and hud.isVisible():
        return True
    return controller._orb.isVisible()


def test_flag_on_keeps_ctrl_space_on_v5_orb(qapp, monkeypatch):
    """V5 regression guard: ATLAS_STAGE_V2=1 must NOT hijack Ctrl+Espacio.

    The V2 demo window has no stay-on-top hint; when it kept the hotkey
    routing it opened BEHIND the foreground app and Atlas seemed to not
    appear at all. The V5 orb is always the hotkey's target.
    """
    monkeypatch.setenv("ATLAS_STAGE_V2", "1")
    controller, orb, panel = _controller(qapp, stage_v2_factory=FakeStageV2)
    try:
        assert controller.stage_v2 is not None
        controller.toggle_interface()  # Ctrl+Espacio #1
        assert _atlas_visible(controller)
        assert not controller.stage_v2.isVisible()
        controller.toggle_interface()  # Ctrl+Espacio #2
        assert not _atlas_visible(controller)
    finally:
        controller.hide_interface()
        orb.close()
        panel.close()


def test_flag_off_keeps_legacy_routing(qapp, monkeypatch):
    monkeypatch.delenv("ATLAS_STAGE_V2", raising=False)
    monkeypatch.delenv("ATLAS_STAGE", raising=False)
    controller, orb, panel = _controller(qapp)
    try:
        assert controller.stage_v2 is None
        controller.toggle_interface()  # Ctrl+Espacio #1
        assert _atlas_visible(controller)
        # V5 identity default: MASTER HUD (or floating orb fallback); the
        # legacy fullscreen stage stays dormant unless ATLAS_STAGE is enabled.
        assert controller._stage is None or not controller._stage.isVisible()
        controller.toggle_interface()  # Ctrl+Espacio #2
        assert not _atlas_visible(controller)
    finally:
        controller.hide_interface()
        orb.close()
        panel.close()


def test_atlas_stage_flag_restores_legacy_fullscreen_overlay(qapp, monkeypatch):
    monkeypatch.delenv("ATLAS_STAGE_V2", raising=False)
    monkeypatch.setenv("ATLAS_STAGE", "1")
    controller, orb, panel = _controller(qapp)
    try:
        controller.toggle_interface()  # Ctrl+Espacio #1
        assert orb.isVisible()
        assert controller._stage is not None and controller._stage.isVisible()
        controller.toggle_interface()  # Ctrl+Espacio #2
        assert not orb.isVisible()
        assert controller._stage is None or not controller._stage.isVisible()
    finally:
        controller.hide_interface()
        orb.close()
        panel.close()


def test_flag_zero_disables_stage_v2(qapp, monkeypatch):
    monkeypatch.setenv("ATLAS_STAGE_V2", "0")
    controller, orb, panel = _controller(qapp, stage_v2_factory=FakeStageV2)
    try:
        assert controller.stage_v2 is None
    finally:
        controller.hide_interface()
        orb.close()
        panel.close()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
