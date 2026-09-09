"""Focal test for the controlled Computer Worker fixture."""

from __future__ import annotations

import importlib.util
from pathlib import Path

_CONTROL_PATH = (
    Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "dev_worker_control.py"
)

EXPECTED_CONTROL_VALUE = "valor-correcto"


def _control_value() -> str:
    spec = importlib.util.spec_from_file_location("dev_worker_control", _CONTROL_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.CONTROL_VALUE


def test_control_value_is_correct() -> None:
    assert _control_value() == EXPECTED_CONTROL_VALUE


def test_control_value_declaration_is_the_edit_target() -> None:
    source = _CONTROL_PATH.read_text(encoding="utf-8")
    assert source.count("CONTROL_VALUE = ") == 1
