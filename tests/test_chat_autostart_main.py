"""CLI routing tests for silent chat auto-start."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

import main
from core.startup import StartupReport


@pytest.fixture(autouse=True)
def successful_preflight(monkeypatch):
    class FakePreflight:
        def __init__(self, project_root: Path) -> None:
            self.project_root = project_root

        def run(self, mode: str) -> StartupReport:
            return StartupReport(self.project_root, mode, (), ("texto",))

    monkeypatch.setattr(main, "WindowsStartupPreflight", FakePreflight)


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        (["main.py", "--chat"], (False, False, False)),
        (["main.py", "--chat", "--start-hidden"], (False, False, True)),
        (["main.py", "--ui"], (True, True, False)),
    ],
)
def test_desktop_modes_preserve_voice_and_visibility(monkeypatch, argv, expected) -> None:
    calls: list[tuple[bool, bool, bool]] = []
    monkeypatch.setattr(
        main,
        "_run_desktop_ui",
        lambda _logger, *, start_voice=True, show_on_start=True, start_hidden=False: calls.append(
            (start_voice, show_on_start, start_hidden)
        ) or 0,
    )
    monkeypatch.setattr(sys, "argv", argv)

    assert main.main() == 0
    assert calls == [expected]
