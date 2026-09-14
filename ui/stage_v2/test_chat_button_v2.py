# -*- coding: utf-8 -*-
"""Test focal: click en CHAT dentro de Stage V2 reutiliza el chat real.

Ejecutar:  python -m pytest ui/stage_v2/test_chat_button_v2.py -q

Contrato:
- Click en el boton CHAT de AtlasStageV2 emite chat_requested.
- Con Stage V2 VISIBLE a pantalla completa, el click lo OCULTA y el
  panel de chat queda visible y al frente (no tapado por Stage V2).
- Si la accion falla, Stage V2 se restaura y recibe un error breve.
- El modo legacy sin ATLAS_STAGE_V2 no cambia.
- El resto de botones no emiten chat_requested.
"""
from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

pytest.importorskip("PySide6")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QEvent, QObject, QPointF, Qt, Signal
from PySide6.QtGui import QMouseEvent

import ui.stage_v2.renderer as renderer
from ui import orbe_app, orbe_controller


@pytest.fixture(scope="module")
def qapp():
    return orbe_app.create_application([])


def _click(stage, index):
    rect = stage.menu_rects[index]
    pos = QPointF(rect.center().x(), rect.center().y())
    event = QMouseEvent(
        QEvent.Type.MouseButtonPress,
        pos,
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    stage.mousePressEvent(event)


def test_chat_click_emits_chat_requested(qapp):
    stage = renderer.AtlasStageV2()
    received = []
    stage.chat_requested.connect(lambda: received.append(True))
    _click(stage, 0)
    assert received == [True]


def test_other_buttons_do_not_emit_chat_requested(qapp):
    stage = renderer.AtlasStageV2()
    received = []
    stage.chat_requested.connect(lambda: received.append(True))
    for index in range(1, len(stage.menu_rects)):
        _click(stage, index)
    assert received == []


def _controller_with(qapp, stage_v2):
    orb = orbe_app.create_orb_window()
    panel = orbe_app.create_transcript_panel()
    controller = orbe_controller.OrbeController(
        atlas=SimpleNamespace(),
        application=qapp,
        orb=orb,
        transcript_panel=panel,
        stage_v2_factory=lambda: stage_v2,
    )
    return controller, orb, panel


def _atlas_visible(controller) -> bool:
    """Atlas interface visible: MASTER HUD (default) or legacy floating orb."""
    hud = controller.master_hud
    if hud is not None and hud.isVisible():
        return True
    return controller._orb.isVisible()


def test_controller_routes_stage_v2_chat_to_show_chat(qapp, monkeypatch):
    monkeypatch.setenv("ATLAS_STAGE_V2", "1")
    stage = renderer.AtlasStageV2()
    controller, orb, panel = _controller_with(qapp, stage)
    try:
        assert controller.stage_v2 is stage
        _click(stage, 0)
        assert panel.isVisible()
        assert _atlas_visible(controller)
    finally:
        controller.hide_interface()
        orb.close()
        panel.close()


def test_chat_click_hides_stage_v2_and_shows_chat(qapp, monkeypatch):
    monkeypatch.setenv("ATLAS_STAGE_V2", "1")
    stage = renderer.AtlasStageV2()
    controller, orb, panel = _controller_with(qapp, stage)
    try:
        # Contrato V5: Ctrl+Espacio/orbe nunca muestran Stage V2; la
        # ventana demo V2 abre por su propio flujo y su boton CHAT sigue
        # trayendo el chat real por delante.
        controller.show_orb()
        assert not stage.isVisible()
        stage.show()
        _click(stage, 0)
        # El chat debe quedar VISIBLE, no tapado por Stage V2.
        assert not stage.isVisible()
        assert panel.isVisible()
        assert _atlas_visible(controller)
    finally:
        controller.hide_interface()
        stage.close()
        orb.close()
        panel.close()


def test_chat_failure_restores_visible_stage_v2(qapp, monkeypatch):
    monkeypatch.setenv("ATLAS_STAGE_V2", "1")
    stage = FakeStageV2()
    stage.show()
    controller, orb, panel = _controller_with(qapp, stage)
    original_panel = controller._transcript_panel
    try:
        controller._transcript_panel = None
        stage.chat_requested.emit()
        # Si show_chat falla, Stage V2 vuelve a la vista y avisa el error.
        assert stage.isVisible()
        assert stage.errors
    finally:
        controller._transcript_panel = original_panel
        controller.hide_interface()
        orb.close()
        panel.close()


def test_legacy_mode_without_flag_unchanged(qapp, monkeypatch):
    monkeypatch.delenv("ATLAS_STAGE_V2", raising=False)
    orb = orbe_app.create_orb_window()
    panel = orbe_app.create_transcript_panel()
    controller = orbe_controller.OrbeController(
        atlas=SimpleNamespace(),
        application=qapp,
        orb=orb,
        transcript_panel=panel,
    )
    try:
        assert controller.stage_v2 is None
        controller.show_orb()
        orb.chat_requested.emit()
        # Flujo legacy intacto: chat visible junto a la interfaz, sin Stage V2.
        assert panel.isVisible()
        assert _atlas_visible(controller)
    finally:
        controller.hide_interface()
        orb.close()
        panel.close()


class FakeStageV2(QObject):
    chat_requested = Signal()

    def __init__(self):
        super().__init__()
        self.visible = False
        self.errors = []

    def show(self):
        self.visible = True

    def hide(self):
        self.visible = False

    def isVisible(self):
        return self.visible

    def raise_(self):
        pass

    def set_status_error(self, message):
        self.errors.append(message)


def test_chat_action_failure_reports_brief_error_in_stage(qapp, monkeypatch):
    monkeypatch.setenv("ATLAS_STAGE_V2", "1")
    stage = FakeStageV2()
    controller, orb, panel = _controller_with(qapp, stage)
    try:
        original_panel = controller._transcript_panel
        controller._transcript_panel = None
        stage.chat_requested.emit()
        assert stage.errors
        stage.chat_requested.emit()
        assert len(stage.errors) == 2
    finally:
        controller._transcript_panel = original_panel
        controller.hide_interface()
        orb.close()
        panel.close()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
