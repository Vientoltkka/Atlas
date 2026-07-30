from __future__ import annotations

from pathlib import Path
import sys

import main
import pytest

from core.startup import CheckStatus, StartupCheck, StartupReport


@pytest.fixture(autouse=True)
def successful_preflight(monkeypatch):
    """Keep CLI routing tests independent from optional workstation packages."""

    class FakePreflight:
        def __init__(self, project_root: Path) -> None:
            self.project_root = project_root

        def run(self, mode: str) -> StartupReport:
            return StartupReport(
                project_root=self.project_root,
                mode=mode,
                checks=(),
                capabilities=("texto",),
            )

    monkeypatch.setattr(main, "WindowsStartupPreflight", FakePreflight)


def test_main_starts_text_mode_by_default(monkeypatch) -> None:
    calls: list[str] = []

    class FakeAtlas:
        def start(self) -> None:
            calls.append("text")

        def start_voice(self) -> None:
            calls.append("voice")

        def start_assistant(self) -> None:
            calls.append("assistant")

    monkeypatch.setattr(main, "Atlas", FakeAtlas)
    monkeypatch.setattr(sys, "argv", ["main.py"])

    main.main()

    assert calls == ["text"]


def test_main_starts_voice_mode_with_flag(monkeypatch) -> None:
    calls: list[str] = []

    class FakeAtlas:
        def start(self) -> None:
            calls.append("text")

        def start_voice(self) -> None:
            calls.append("voice")

        def start_assistant(self) -> None:
            calls.append("assistant")

    monkeypatch.setattr(main, "Atlas", FakeAtlas)
    monkeypatch.setattr(sys, "argv", ["main.py", "--voice"])

    main.main()

    assert calls == ["voice"]


def test_main_starts_assistant_mode_with_flag(monkeypatch) -> None:
    calls: list[str] = []

    class FakeAtlas:
        def start(self) -> None:
            calls.append("text")

        def start_voice(self) -> None:
            calls.append("voice")

        def start_assistant(self) -> None:
            calls.append("assistant")

    monkeypatch.setattr(main, "Atlas", FakeAtlas)
    monkeypatch.setattr(sys, "argv", ["main.py", "--assistant"])

    main.main()

    assert calls == ["assistant"]


def test_main_lists_microphones(monkeypatch, capsys) -> None:
    calls: list[str] = []

    class FakeAtlas:
        def __init__(self) -> None:
            raise AssertionError("Atlas must not be built for microphone listing")

        def start(self) -> None:
            calls.append("text")

        def start_voice(self) -> None:
            calls.append("voice")

        def start_assistant(self) -> None:
            calls.append("assistant")

        def list_microphones(self) -> str:
            calls.append("list")
            return "Microfonos disponibles:\n0. Microfono fisico"

    class FakeCapture:
        def list_microphones(self, include_open_status=False):
            calls.append(f"list:{include_open_status}")
            return [
                type(
                    "Mic",
                    (),
                    {
                        "index": 1,
                        "name": "Microfono fisico",
                        "is_default": False,
                        "host_api": "MME",
                        "channels": 2,
                        "default_samplerate": 44100.0,
                        "can_open": True,
                        "open_error": "",
                    },
                )()
            ]

    monkeypatch.setattr(main, "Atlas", FakeAtlas)
    monkeypatch.setattr(main, "SoundDeviceAudioCapture", FakeCapture)
    monkeypatch.setattr(sys, "argv", ["main.py", "--list-microphones"])

    main.main()

    output = capsys.readouterr().out
    assert calls == ["list:True"]
    assert "Microfonos disponibles:" in output
    assert "Microfono fisico" in output
    assert "host API: MME" in output


def test_main_tests_microphone_without_building_atlas(monkeypatch, capsys) -> None:
    calls: list[str] = []

    class FakeAtlas:
        def __init__(self) -> None:
            raise AssertionError("Atlas must not be built for microphone test")

    class FakeCapture:
        def test_microphone(self, index: int, duration_seconds: float = 3.0):
            calls.append(f"test:{index}:{duration_seconds}")
            microphone = type(
                "Mic",
                (),
                {
                    "index": index,
                    "name": "Microfono fisico",
                    "host_api": "MME",
                    "channels": 2,
                },
            )()
            return type(
                "Result",
                (),
                {
                    "microphone": microphone,
                    "error": "",
                    "duration_seconds": 3.0,
                    "rms": 0.01,
                    "voice_detected": True,
                },
            )()

    monkeypatch.setattr(main, "Atlas", FakeAtlas)
    monkeypatch.setattr(main, "SoundDeviceAudioCapture", FakeCapture)
    monkeypatch.setattr(sys, "argv", ["main.py", "--test-microphone", "1"])

    main.main()

    output = capsys.readouterr().out
    assert calls == ["test:1:3.0"]
    assert "RMS: 0.0100" in output
    assert "Voz detectada: si" in output


def test_main_stops_before_bootstrap_when_preflight_fails(
    monkeypatch,
    capsys,
) -> None:
    class FailingPreflight:
        def __init__(self, project_root: Path) -> None:
            self.project_root = project_root

        def run(self, mode: str) -> StartupReport:
            return StartupReport(
                project_root=self.project_root,
                mode=mode,
                checks=(
                    StartupCheck(
                        "Dependencia ollama",
                        CheckStatus.ERROR,
                        "No esta instalada.",
                        "Instala requirements.txt.",
                    ),
                ),
                capabilities=("texto",),
            )

    class ForbiddenAtlas:
        def __init__(self) -> None:
            raise AssertionError("Bootstrap must not run after a failed preflight")

    monkeypatch.setattr(main, "WindowsStartupPreflight", FailingPreflight)
    monkeypatch.setattr(main, "Atlas", ForbiddenAtlas)
    monkeypatch.setattr(sys, "argv", ["main.py"])

    exit_code = main.main()

    output = capsys.readouterr().out
    assert exit_code == 1
    assert "Atlas no puede arrancar" in output
    assert "Accion recomendada" in output
    assert "Traceback" not in output


def test_main_converts_internal_failure_and_closes_atlas(
    monkeypatch,
    capsys,
) -> None:
    calls: list[str] = []

    class FailingAtlas:
        def start(self) -> None:
            raise RuntimeError("token=do-not-print")

        def close(self) -> None:
            calls.append("close")

    monkeypatch.setattr(main, "Atlas", FailingAtlas)
    monkeypatch.setattr(sys, "argv", ["main.py"])

    exit_code = main.main()

    output = capsys.readouterr().out
    assert exit_code == 1
    assert calls == ["close"]
    assert "error interno" in output
    assert "do-not-print" not in output
    assert "Traceback" not in output


def test_main_treats_keyboard_interrupt_as_voluntary_close(
    monkeypatch,
    capsys,
) -> None:
    calls: list[str] = []

    class InterruptedAtlas:
        def start(self) -> None:
            raise KeyboardInterrupt

        def close(self) -> None:
            calls.append("close")

    monkeypatch.setattr(main, "Atlas", InterruptedAtlas)
    monkeypatch.setattr(sys, "argv", ["main.py"])

    exit_code = main.main()

    output = capsys.readouterr().out
    assert exit_code == 0
    assert calls == ["close"]
    assert "cerrado correctamente" in output


def test_main_prints_one_startup_block(monkeypatch, capsys) -> None:
    calls: list[str] = []

    class QuietAtlas:
        def start(self) -> None:
            calls.append("start")

        def close(self) -> None:
            calls.append("close")

    monkeypatch.setattr(main, "Atlas", QuietAtlas)
    monkeypatch.setattr(sys, "argv", ["main.py"])

    assert main.main() == 0

    output = capsys.readouterr().out
    assert calls == ["start", "close"]
    assert output.count("Estado: preparado") == 1
    assert "Atlas iniciado correctamente" not in output


def test_main_uses_degraded_logging_without_traceback(
    monkeypatch,
    capsys,
) -> None:
    calls: list[str] = []

    class QuietAtlas:
        def start(self) -> None:
            calls.append("start")

        def close(self) -> None:
            calls.append("close")

    def fail_logging(_project_root: Path):
        raise OSError("token=must-not-be-visible")

    monkeypatch.setattr(main, "configure_operational_logging", fail_logging)
    monkeypatch.setattr(main, "Atlas", QuietAtlas)
    monkeypatch.setattr(sys, "argv", ["main.py"])

    assert main.main() == 0

    output = capsys.readouterr().out
    assert calls == ["start", "close"]
    assert "modo degradado sin log de archivo" in output
    assert "must-not-be-visible" not in output
    assert "Traceback" not in output
