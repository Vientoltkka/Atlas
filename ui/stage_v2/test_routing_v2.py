# -*- coding: utf-8 -*-
"""Test focal del routing Ctrl+Espacio ON/OFF de ATLAS_STAGE_V2.

Ejecutar:  python -m pytest ui/stage_v2/test_routing_v2.py -q

Contrato:
- ATLAS_STAGE_V2=1  -> Ctrl+Espacio muestra SOLO AtlasStageV2;
  segundo Ctrl+Espacio la oculta; el orbe legacy NO aparece.
- ATLAS_STAGE_V2=0/ausente -> comportamiento legacy intacto.
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


def test_flag_on_routes_ctrl_space_to_stage_v2_only(qapp, monkeypatch):
    monkeypatch.setenv("ATLAS_STAGE_V2", "1")
    controller, orb, panel = _controller(qapp, stage_v2_factory=FakeStageV2)
    try:
        assert controller.stage_v2 is not None
        controller.toggle_interface()  # Ctrl+Espacio #1
        assert controller.stage_v2.isVisible()
        assert not orb.isVisible()
        assert not panel.isVisible()
        assert controller._stage is None or not controller._stage.isVisible()
        controller.toggle_interface()  # Ctrl+Espacio #2
        assert not controller.stage_v2.isVisible()
    finally:
        controller.hide_interface()
        orb.close()
        panel.close()


def test_flag_off_keeps_legacy_routing(qapp, monkeypatch):
    monkeypatch.delenv("ATLAS_STAGE_V2", raising=False)
    controller, orb, panel = _controller(qapp)
    try:
        assert controller.stage_v2 is None
        controller.toggle_interface()  # Ctrl+Espacio #1
        assert orb.isVisible()
        assert controller._stage is None or controller._stage.isVisible()
        controller.toggle_interface()  # Ctrl+Espacio #2
        assert not orb.isVisible()
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
