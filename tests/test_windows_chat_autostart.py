"""Tests for the current-user silent Atlas chat startup launcher."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from core import windows_chat_autostart


class _Sandbox:
    def __enter__(self) -> Path:
        artifacts = Path(__file__).resolve().parents[1] / "artifacts"
        self._directory = tempfile.TemporaryDirectory(dir=artifacts)
        return Path(self._directory.name)

    def __exit__(self, *_args) -> None:
        self._directory.cleanup()


def _python_pair(root: Path) -> Path:
    python = root / "Scripts" / "python.exe"
    python.parent.mkdir()
    python.write_text("", encoding="utf-8")
    python.with_name("pythonw.exe").write_text("", encoding="utf-8")
    return python


def test_launcher_uses_absolute_pythonw_main_and_hidden_chat() -> None:
    root = Path(r"C:\AI\Atlas")
    launcher = windows_chat_autostart.build_launcher_contents(
        python_executable=str(root / ".venv" / "Scripts" / "pythonw.exe"),
        project_root=root,
    )

    assert "WScript.Shell" in launcher
    assert "pythonw.exe" in launcher
    assert "main.py" in launcher
    assert "--chat --start-hidden" in launcher
    assert ", 0, False" in launcher


def test_install_and_status_use_only_the_given_current_user_launcher(capsys) -> None:
    with _Sandbox() as root:
        python = _python_pair(root)
        launcher = root / "Startup" / windows_chat_autostart.LAUNCHER_NAME

        assert windows_chat_autostart.install(
            root, python_executable=str(python), startup_path=launcher
        ) == 0
        assert launcher.is_file()
        assert windows_chat_autostart.status(
            root, python_executable=str(python), startup_path=launcher
        ) == 0
        assert "instalado" in capsys.readouterr().out


def test_status_and_start_distinguish_missing_launcher() -> None:
    with _Sandbox() as root:
        launcher = root / "Startup" / windows_chat_autostart.LAUNCHER_NAME
        python = _python_pair(root)

        assert windows_chat_autostart.status(
            root, python_executable=str(python), startup_path=launcher
        ) == 1
        with pytest.raises(windows_chat_autostart.TaskError, match="no instalado"):
            windows_chat_autostart.start(startup_path=launcher)


def test_start_and_uninstall_manage_the_existing_launcher(monkeypatch) -> None:
    with _Sandbox() as root:
        python = _python_pair(root)
        launcher = root / "Startup" / windows_chat_autostart.LAUNCHER_NAME
        windows_chat_autostart.install(
            root, python_executable=str(python), startup_path=launcher
        )
        calls: list[list[str]] = []
        monkeypatch.setattr(
            windows_chat_autostart.subprocess,
            "Popen",
            lambda args: calls.append(args),
        )

        assert windows_chat_autostart.start(startup_path=launcher) == 0
        assert calls == [["wscript.exe", str(launcher)]]
        assert windows_chat_autostart.uninstall(startup_path=launcher) == 0
        assert not launcher.exists()
