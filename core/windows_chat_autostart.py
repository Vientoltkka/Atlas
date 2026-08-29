"""Current-user Windows Startup Folder management for silent Atlas chat."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


LAUNCHER_NAME = "AtlasChatAutostart.vbs"
CHAT_ARGUMENTS = "--chat --start-hidden"


class TaskError(RuntimeError):
    """Controlled failure of a chat auto-start management operation."""


def startup_folder(appdata: str | None = None) -> Path:
    """Return the current user's Windows Startup folder."""
    base = appdata or os.environ.get("APPDATA", "")
    if not base:
        raise TaskError("No se pudo resolver APPDATA para el usuario actual.")
    return Path(base) / "Microsoft" / "Windows" / "Start Menu" / "Programs" / "Startup"


def launcher_path(*, appdata: str | None = None) -> Path:
    return startup_folder(appdata) / LAUNCHER_NAME


def build_launcher_contents(*, python_executable: str, project_root: Path) -> str:
    """Render a hidden VBS launcher using absolute paths only."""
    main_script = project_root / "main.py"
    command = f'{_quote(python_executable)} -B {_quote(str(main_script))} {CHAT_ARGUMENTS}'
    return 'CreateObject("WScript.Shell").Run "' + command.replace('"', '""') + '", 0, False\n'


def install(
    project_root: Path,
    *,
    python_executable: str | None = None,
    startup_path: Path | None = None,
) -> int:
    """Write the user's silent launcher without requiring elevation."""
    executable = _pythonw_executable(python_executable or sys.executable)
    path = startup_path or launcher_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        build_launcher_contents(python_executable=executable, project_root=project_root),
        encoding="utf-8",
        newline="",
    )
    return 0


def status(
    project_root: Path,
    *,
    python_executable: str | None = None,
    startup_path: Path | None = None,
) -> int:
    """Report whether the exact current-user launcher is installed."""
    executable = _pythonw_executable(python_executable or sys.executable)
    path = startup_path or launcher_path()
    expected = build_launcher_contents(
        python_executable=executable,
        project_root=project_root,
    )
    if path.is_file() and path.read_text(encoding="utf-8") == expected:
        print(f"Autoarranque instalado: {path}")
        return 0
    print(f"Autoarranque no instalado: {path}")
    return 1


def start(*, startup_path: Path | None = None) -> int:
    """Run the installed launcher now without opening a console."""
    path = startup_path or launcher_path()
    if not path.is_file():
        raise TaskError("Autoarranque no instalado.")
    subprocess.Popen(["wscript.exe", str(path)])
    return 0


def uninstall(*, startup_path: Path | None = None) -> int:
    """Remove only Atlas's current-user startup launcher."""
    path = startup_path or launcher_path()
    path.unlink(missing_ok=True)
    return 0


def dispatch(action: str, project_root: Path) -> int:
    """Run one explicit startup-management action."""
    handlers = {
        "install": lambda: install(project_root),
        "uninstall": uninstall,
        "start": start,
        "status": lambda: status(project_root),
    }
    handler = handlers.get(action)
    if handler is None:
        raise TaskError(f"Accion no reconocida: {action}")
    return handler()


def main(argv: list[str] | None = None) -> int:
    """Manage the current user's Atlas chat auto-start launcher."""
    parser = argparse.ArgumentParser(
        description="Gestiona el autoarranque silencioso del chat de Atlas."
    )
    parser.add_argument("action", help="install | uninstall | start | status")
    args = parser.parse_args(argv)
    project_root = Path(__file__).resolve().parents[1]
    try:
        return dispatch(args.action, project_root)
    except TaskError as error:
        print(f"Error: {error}")
        return 1


def _pythonw_executable(python_executable: str) -> str:
    executable = Path(python_executable)
    pythonw = executable.with_name("pythonw.exe")
    if not pythonw.is_file():
        raise TaskError(f"No se encontro pythonw.exe junto a {executable}.")
    return str(pythonw)


def _quote(value: str) -> str:
    return f'"{value}"'


if __name__ == "__main__":
    raise SystemExit(main())
