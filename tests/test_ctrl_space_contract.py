"""Contrato de Ctrl+Espacio (proteccion anti-regresion V5).

Offscreen-only. Contract under test:

1. Atlas arrancado + Ctrl+Espacio  -> Atlas aparece (orbe V5 visible),
   tanto por defecto como con ATLAS_STAGE_V2 o ATLAS_STAGE activos.
2. Atlas visible + Ctrl+Espacio    -> se oculta todo (toggle).
3. ESC global queda registrado mientras Atlas es visible y se
   desregistra al ocultar.
4. La esfera V5 es la interfaz principal; el stage legacy solo aparece
   con ATLAS_STAGE=1 y Stage V2 jamas secuestra el hotkey.

Ejecutar:  python -m pytest tests/test_ctrl_space_contract.py -q
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


class RecordingHotkey:
    """Doble del hotkey nativo: evita RegistrarHotKeys reales en tests."""

    def __init__(self, callback, logger=None, **kwargs):
        self.callback = callback
        self.started = False
        self.stopped = False

    def start(self) -> bool:
        self.started = True
        return True

    def stop(self) -> None:
        self.stopped = True


class FakeStageV2:
    def __init__(self):
        self.shown = False

    def show(self):
        self.shown = True

    def hide(self):
        self.shown = False

    def isVisible(self):
        return self.shown


def _controller(qapp, stage_v2_factory=None):
    orb = orbe_app.create_orb_window()
    panel = orbe_app.create_transcript_panel()
    controller = orbe_controller.OrbeController(
        atlas=SimpleNamespace(),
        application=qapp,
        orb=orb,
        transcript_panel=panel,
        hotkey_factory=lambda callback, logger=None, **kwargs: RecordingHotkey(
            callback, logger=logger, **kwargs
        ),
        stage_v2_factory=stage_v2_factory,
    )
    return controller, orb, panel


def _press_ctrl_space(controller) -> None:
    """Simulate the native hotkey callback exactly as windows_hotkey does."""
    controller._request_orb_visible()


def _atlas_visible(controller) -> bool:
    """Atlas is visible through the MASTER HUD or the legacy floating orb."""
    hud = controller.master_hud
    if hud is not None and hud.isVisible():
        return True
    return controller._orb.isVisible()


def _drain(qapp) -> None:
    qapp.processEvents()


def test_default_env_hotkey_shows_atlas_and_toggles(qapp, monkeypatch):
    monkeypatch.delenv("ATLAS_STAGE_V2", raising=False)
    monkeypatch.delenv("ATLAS_STAGE", raising=False)
    controller, orb, panel = _controller(qapp)
    try:
        controller.start(start_voice=False, show_on_start=False, start_hidden=True)
        assert not controller._overlay_visible()

        _press_ctrl_space(controller)  # Ctrl+Espacio #1: Atlas aparece
        _drain(qapp)
        assert _atlas_visible(controller), "Ctrl+Espacio debe hacer aparecer Atlas"
        assert not panel.isVisible()

        _press_ctrl_space(controller)  # Ctrl+Espacio #2: se oculta
        _drain(qapp)
        assert not _atlas_visible(controller)
        assert not controller._overlay_visible()
    finally:
        controller.hide_interface()
        controller._stop_chat_hotkey()
        orb.close()
        panel.close()


def test_stage_v2_flag_never_hijacks_ctrl_space(qapp, monkeypatch):
    monkeypatch.setenv("ATLAS_STAGE_V2", "1")
    stage_v2 = FakeStageV2()
    controller, orb, panel = _controller(
        qapp, stage_v2_factory=lambda: stage_v2
    )
    try:
        controller.start(start_voice=False, show_on_start=False, start_hidden=True)
        _press_ctrl_space(controller)  # Ctrl+Espacio #1
        _drain(qapp)
        assert _atlas_visible(controller), "Ctrl+Espacio debe traer la interfaz Atlas"
        assert not stage_v2.isVisible(), "Stage V2 no debe secuestrar el hotkey"

        _press_ctrl_space(controller)  # Ctrl+Espacio #2
        _drain(qapp)
        assert not _atlas_visible(controller)
        assert not stage_v2.isVisible(), "Stage V2 no debe secuestrar el hotkey"
    finally:
        controller.hide_interface()
        controller._stop_chat_hotkey()
        orb.close()
        panel.close()


def test_atlas_stage_flag_keeps_opt_in_fullscreen_overlay(qapp, monkeypatch):
    monkeypatch.delenv("ATLAS_STAGE_V2", raising=False)
    monkeypatch.setenv("ATLAS_STAGE", "1")
    controller, orb, panel = _controller(qapp)
    try:
        controller.start(start_voice=False, show_on_start=False, start_hidden=True)
        _press_ctrl_space(controller)
        _drain(qapp)
        assert orb.isVisible()
        assert controller.stage is not None and controller.stage.isVisible()

        _press_ctrl_space(controller)
        _drain(qapp)
        assert not orb.isVisible()
        assert controller.stage is None or not controller.stage.isVisible()
    finally:
        controller.hide_interface()
        controller._stop_chat_hotkey()
        orb.close()
        panel.close()


def test_esc_hotkey_registered_while_visible_and_cleared_when_hidden(
    qapp, monkeypatch
):
    monkeypatch.delenv("ATLAS_STAGE_V2", raising=False)
    monkeypatch.delenv("ATLAS_STAGE", raising=False)
    controller, orb, panel = _controller(qapp)
    try:
        controller.start(start_voice=False, show_on_start=False, start_hidden=True)
        _press_ctrl_space(controller)
        _drain(qapp)
        assert _atlas_visible(controller)
        assert controller._escape_hotkey is not None

        controller._request_escape_hide()  # ESC global
        _drain(qapp)
        assert not _atlas_visible(controller)
        assert controller._escape_hotkey is None
    finally:
        controller.hide_interface()
        controller._stop_chat_hotkey()
        orb.close()
        panel.close()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))

