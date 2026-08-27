"""Tests for V4.1-W2: Windows Task Scheduler management of the webhook."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from core import windows_task
from core.startup import WHATSAPP_REQUIRED_ENV_VARS
from tools.effect_permissions import ToolEffectPermissionPolicy


def _whatsapp_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ATLAS_WHATSAPP_VERIFY_TOKEN", "verify-secret")
    monkeypatch.setenv("ATLAS_WHATSAPP_ACCESS_TOKEN", "access-secret")
    monkeypatch.setenv("ATLAS_WHATSAPP_PHONE_NUMBER_ID", "pnid-123")


# ---------------------------------------------------------------------------
# Task definition
# ---------------------------------------------------------------------------


def test_task_xml_runs_exact_official_whatsapp_command(tmp_path: Path) -> None:
    python_exe = r"C:\Python312\python.exe"
    xml = windows_task.build_task_xml(
        python_executable=python_exe, project_root=tmp_path
    )
    assert "<Command>C:\\Python312\\python.exe</Command>" in xml
    assert "<Arguments>-B main.py --whatsapp-webhook</Arguments>" in xml
    assert f"<WorkingDirectory>{tmp_path}</WorkingDirectory>" in xml


def test_task_xml_restarts_on_failure() -> None:
    xml = windows_task.build_task_xml(
        python_executable="python.exe",
        project_root=Path("."),
    )
    assert "<RestartOnFailure>" in xml
    assert f"<Interval>PT{windows_task.RESTART_INTERVAL_MINUTES}M</Interval>" in xml
    assert f"<Count>{windows_task.RESTART_ATTEMPTS}</Count>" in xml


def test_task_xml_has_unlimited_execution_time_and_single_instance() -> None:
    xml = windows_task.build_task_xml(
        python_executable="python.exe",
        project_root=Path("."),
    )
    # A long-running webhook must not be killed by the default 72h limit.
    assert "<ExecutionTimeLimit>PT0S</ExecutionTimeLimit>" in xml
    assert "<MultipleInstancesPolicy>IgnoreNew</MultipleInstancesPolicy>" in xml


def test_task_xml_starts_at_logon() -> None:
    xml = windows_task.build_task_xml(
        python_executable="python.exe",
        project_root=Path("."),
    )
    assert "<LogonTrigger>" in xml
    assert "<Enabled>true</Enabled>" in xml


def test_task_xml_never_contains_secret_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _whatsapp_env(monkeypatch)
    xml = windows_task.build_task_xml(
        python_executable="python.exe", project_root=tmp_path
    )
    assert "verify-secret" not in xml
    assert "access-secret" not in xml
    assert "pnid-123" not in xml


# ---------------------------------------------------------------------------
# schtasks command construction
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("builder", "expected"),
    (
        (windows_task.build_uninstall_args, ["schtasks", "/Delete", "/F", "/TN", "AtlasWhatsAppWebhook"]),
        (windows_task.build_start_args, ["schtasks", "/Run", "/TN", "AtlasWhatsAppWebhook"]),
        (windows_task.build_stop_args, ["schtasks", "/End", "/TN", "AtlasWhatsAppWebhook"]),
        (windows_task.build_status_args, ["schtasks", "/Query", "/TN", "AtlasWhatsAppWebhook"]),
    ),
)
def test_schtasks_argument_builders(builder, expected) -> None:
    assert builder() == expected


def test_install_invokes_schtasks_with_xml_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    _whatsapp_env(monkeypatch)
    captured: dict = {}

    def fake_run(args, **kwargs):
        captured["args"] = args
        return subprocess.CompletedProcess(args, 0)

    monkeypatch.setattr(windows_task.subprocess, "run", fake_run)

    policy = ToolEffectPermissionPolicy()
    exit_code = windows_task.install(
        tmp_path,
        permission_policy=policy,
        authorization=policy.authorize("windows_task.install", ("windows.task",)),
    )

    assert exit_code == 0
    args = captured["args"]
    assert args[:2] == ["schtasks", "/Create"]
    xml_path = Path(args[-1])
    assert not xml_path.exists()  # temp file cleaned up


def test_install_failure_reports_error_without_secrets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
) -> None:
    _whatsapp_env(monkeypatch)

    def fake_run(args, **kwargs):
        return subprocess.CompletedProcess(args, 1, stderr="access denied")

    monkeypatch.setattr(windows_task.subprocess, "run", fake_run)

    policy = ToolEffectPermissionPolicy()
    with pytest.raises(windows_task.TaskError) as excinfo:
        windows_task.install(
            tmp_path,
            permission_policy=policy,
            authorization=policy.authorize("windows_task.install", ("windows.task",)),
        )
    assert "access denied" in str(excinfo.value)
    assert "secret" not in str(excinfo.value)


# ---------------------------------------------------------------------------
# CLI dispatch
# ---------------------------------------------------------------------------


def test_install_refuses_without_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in WHATSAPP_REQUIRED_ENV_VARS:
        monkeypatch.delenv(name, raising=False)

    with pytest.raises(windows_task.TaskError) as excinfo:
        windows_task.dispatch("install", Path("."))

    message = str(excinfo.value)
    assert "ATLAS_WHATSAPP_VERIFY_TOKEN" in message
    assert "ATLAS_WHATSAPP_ACCESS_TOKEN" in message
    assert "=" not in message  # no values are echoed


def test_dispatch_unknown_action_is_controlled() -> None:
    with pytest.raises(windows_task.TaskError):
        windows_task.dispatch("explode", Path("."))


def test_main_loads_env_and_dispatches(monkeypatch: pytest.MonkeyPatch) -> None:
    dispatched: dict = {}

    def fake_dispatch(action: str, project_root: Path) -> int:
        dispatched["action"] = action
        return 0

    monkeypatch.setattr(windows_task, "dispatch", fake_dispatch)

    exit_code = windows_task.main(["status"])

    assert exit_code == 0
    assert dispatched["action"] == "status"


def test_main_returns_1_on_task_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def failing_dispatch(action: str, project_root: Path) -> int:
        raise windows_task.TaskError("algo fallo")

    monkeypatch.setattr(windows_task, "dispatch", failing_dispatch)

    assert windows_task.main(["start"]) == 1
