"""Windows Task Scheduler management for the Atlas WhatsApp webhook (W2).

Registers a scheduled task that starts ``python -B main.py
--whatsapp-webhook`` at logon and restarts it automatically when the
process fails. Chosen over a pywin32 service to avoid new dependencies,
admin requirements and custom service plumbing; Task Scheduler provides
native logon triggers, failure restart and single-instance policy.

No secret value is ever written to the task XML or printed: credentials
are read at runtime by the process itself through ``main.py`` (.env /
environment, loaded since W1).
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from core.startup import WHATSAPP_REQUIRED_ENV_VARS


TASK_NAME = "AtlasWhatsAppWebhook"
TASK_ARGUMENTS = "-B main.py --whatsapp-webhook"
RESTART_INTERVAL_MINUTES = 1
RESTART_ATTEMPTS = 3


class TaskError(RuntimeError):
    """Controlled failure of a task-management operation."""


def build_task_xml(*, python_executable: str, project_root: Path) -> str:
    """Render the deterministic task definition used by installation."""
    return f"""<?xml version="1.0" encoding="UTF-16"?>
<Task version="1.4" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">
  <Triggers>
    <LogonTrigger>
      <Enabled>true</Enabled>
    </LogonTrigger>
  </Triggers>
  <Principals>
    <Principal id="Author">
      <LogonType>InteractiveToken</LogonType>
      <RunLevel>LeastPrivilege</RunLevel>
    </Principal>
  </Principals>
  <Settings>
    <MultipleInstancesPolicy>IgnoreNew</MultipleInstancesPolicy>
    <DisallowStartIfOnBatteries>false</DisallowStartIfOnBatteries>
    <StopIfGoingOnBatteries>false</StopIfGoingOnBatteries>
    <AllowHardTerminate>true</AllowHardTerminate>
    <StartWhenAvailable>true</StartWhenAvailable>
    <RunOnlyIfNetworkAvailable>false</RunOnlyIfNetworkAvailable>
    <AllowStartOnDemand>true</AllowStartOnDemand>
    <Enabled>true</Enabled>
    <Hidden>false</Hidden>
    <RunOnlyIfIdle>false</RunOnlyIfIdle>
    <WakeToRun>false</WakeToRun>
    <ExecutionTimeLimit>PT0S</ExecutionTimeLimit>
    <Priority>7</Priority>
    <RestartOnFailure>
      <Interval>PT{RESTART_INTERVAL_MINUTES}M</Interval>
      <Count>{RESTART_ATTEMPTS}</Count>
    </RestartOnFailure>
  </Settings>
  <Actions Context="Author">
    <Exec>
      <Command>{_escape(python_executable)}</Command>
      <Arguments>{_escape(TASK_ARGUMENTS)}</Arguments>
      <WorkingDirectory>{_escape(str(project_root))}</WorkingDirectory>
    </Exec>
  </Actions>
</Task>
"""


def _escape(value: str) -> str:
    return value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def build_install_args(xml_path: Path) -> list[str]:
    return ["schtasks", "/Create", "/F", "/TN", TASK_NAME, "/XML", str(xml_path)]


def build_uninstall_args() -> list[str]:
    return ["schtasks", "/Delete", "/F", "/TN", TASK_NAME]


def build_start_args() -> list[str]:
    return ["schtasks", "/Run", "/TN", TASK_NAME]


def build_stop_args() -> list[str]:
    return ["schtasks", "/End", "/TN", TASK_NAME]


def build_status_args() -> list[str]:
    return ["schtasks", "/Query", "/TN", TASK_NAME]


def _require_credentials() -> None:
    missing = [
        name
        for name in WHATSAPP_REQUIRED_ENV_VARS
        if not os.environ.get(name, "").strip()
    ]
    if missing:
        raise TaskError(
            "Faltan variables de entorno antes de instalar la tarea: "
            + ", ".join(missing)
            + " (ver .env.example)."
        )


def _run_schtasks(args: list[str]) -> int:
    result = subprocess.run(
        args,
        capture_output=True,
        text=True,
        check=False,
    )
    output = "\n".join(
        part.strip()
        for part in (result.stdout or "", result.stderr or "")
        if part.strip()
    )
    if result.returncode != 0:
        raise TaskError(
            f"schtasks fallo con codigo {result.returncode}: {output}"
        )
    if output:
        print(output)
    return result.returncode


def install(project_root: Path) -> int:
    """Register (or replace) the scheduled task."""
    _require_credentials()
    xml = build_task_xml(python_executable=sys.executable, project_root=project_root)
    handle = tempfile.NamedTemporaryFile(
        mode="w", suffix=".xml", encoding="utf-8", delete=False
    )
    try:
        # UTF-8 with BOM is accepted by schtasks /XML.
        handle.write(xml)
        handle.close()
        return _run_schtasks(build_install_args(Path(handle.name)))
    finally:
        Path(handle.name).unlink(missing_ok=True)


def dispatch(action: str, project_root: Path) -> int:
    handlers = {
        "install": lambda: install(project_root),
        "uninstall": lambda: _run_schtasks(build_uninstall_args()),
        "start": lambda: _run_schtasks(build_start_args()),
        "stop": lambda: _run_schtasks(build_stop_args()),
        "status": lambda: _run_schtasks(build_status_args()),
    }
    handler = handlers.get(action)
    if handler is None:
        raise TaskError(f"Accion no reconocida: {action}")
    return handler()


def main(argv: list[str] | None = None) -> int:
    """Manage the Atlas WhatsApp scheduled task (no secrets are printed)."""
    parser = argparse.ArgumentParser(
        description=(
            "Gestiona la tarea programada del webhook de WhatsApp "
            "(acciones: install | uninstall | start | stop | status)."
        )
    )
    parser.add_argument("action", help="install | uninstall | start | stop | status")
    args = parser.parse_args(argv)
    project_root = Path(__file__).resolve().parents[1]

    from dotenv import load_dotenv

    load_dotenv(project_root / ".env")

    try:
        return dispatch(args.action, project_root)
    except TaskError as error:
        print(f"Error: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
