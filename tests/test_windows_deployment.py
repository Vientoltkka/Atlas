"""Tests for V4.1-W1: Windows deployment baseline (deps, .env, preflight)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from core.startup import (
    CheckStatus,
    WindowsStartupPreflight,
    render_startup_failure,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _project(tmp_path: Path) -> Path:
    for relative in (
        "main.py",
        "core/atlas.py",
        "bootstrap/bootstrap.py",
        "requirements.txt",
    ):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# test\n", encoding="utf-8")
    return tmp_path


def _all_modules(_name: str) -> object:
    return object()


def _whatsapp_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ATLAS_WHATSAPP_VERIFY_TOKEN", "verify-secret")
    monkeypatch.setenv("ATLAS_WHATSAPP_ACCESS_TOKEN", "access-secret")
    monkeypatch.setenv("ATLAS_WHATSAPP_PHONE_NUMBER_ID", "pnid-123")


def test_requirements_txt_contains_httpx() -> None:
    content = (PROJECT_ROOT / "requirements.txt").read_text(encoding="utf-8-sig")
    assert "httpx" in content


def test_load_environment_file_reads_env_before_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from main import _load_environment_file

    (tmp_path / ".env").write_text("ATLAS_W1_TEST_VAR=from-file\n", encoding="utf-8")
    monkeypatch.delenv("ATLAS_W1_TEST_VAR", raising=False)

    _load_environment_file(tmp_path)
    try:
        assert os.environ["ATLAS_W1_TEST_VAR"] == "from-file"
    finally:
        os.environ.pop("ATLAS_W1_TEST_VAR", None)


def test_existing_environment_variable_keeps_precedence_over_dotenv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from main import _load_environment_file

    (tmp_path / ".env").write_text("ATLAS_W1_TEST_VAR=from-file\n", encoding="utf-8")
    monkeypatch.setenv("ATLAS_W1_TEST_VAR", "from-env")

    _load_environment_file(tmp_path)
    assert os.environ["ATLAS_W1_TEST_VAR"] == "from-env"


def test_missing_dotenv_module_is_tolerated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import builtins
    import importlib

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "dotenv":
            raise ImportError("no dotenv")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    importlib.invalidate_caches()

    from main import _load_environment_file

    _load_environment_file(tmp_path)


def test_whatsapp_mode_with_valid_deps_and_credentials_passes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _whatsapp_env(monkeypatch)
    report = WindowsStartupPreflight(
        _project(tmp_path),
        python_version=(3, 12),
        module_finder=_all_modules,
        socket_probe=lambda *_args: False,
    ).run("whatsapp")

    assert report.ready
    credential_checks = [
        check for check in report.checks if check.name == "Credenciales WhatsApp"
    ]
    assert len(credential_checks) == 1
    assert credential_checks[0].status is CheckStatus.OK


def test_whatsapp_mode_missing_required_dependency_fails_controlled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _whatsapp_env(monkeypatch)

    def find_module(name: str) -> object | None:
        if name == "uvicorn":
            return None
        return object()

    report = WindowsStartupPreflight(
        _project(tmp_path),
        python_version=(3, 12),
        module_finder=find_module,
        socket_probe=lambda *_args: False,
    ).run("whatsapp")

    assert not report.ready
    failed = render_startup_failure(report)
    assert "uvicorn" in failed
    assert "python -m pip install -r requirements.txt" in failed


def test_whatsapp_mode_missing_dependency_does_not_break_text_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _whatsapp_env(monkeypatch)

    def find_module(name: str) -> object | None:
        if name in ("fastapi", "uvicorn", "httpx"):
            return None
        return object()

    finder = find_module
    text_report = WindowsStartupPreflight(
        _project(tmp_path),
        python_version=(3, 12),
        module_finder=finder,
        socket_probe=lambda *_args: False,
    ).run("text")
    whatsapp_report = WindowsStartupPreflight(
        _project(tmp_path),
        python_version=(3, 12),
        module_finder=finder,
        socket_probe=lambda *_args: False,
    ).run("whatsapp")

    assert text_report.ready
    assert not whatsapp_report.ready


@pytest.mark.parametrize(
    "variable",
    (
        "ATLAS_WHATSAPP_VERIFY_TOKEN",
        "ATLAS_WHATSAPP_ACCESS_TOKEN",
        "ATLAS_WHATSAPP_PHONE_NUMBER_ID",
    ),
)
def test_whatsapp_mode_missing_single_credential_fails_controlled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, variable: str
) -> None:
    _whatsapp_env(monkeypatch)
    monkeypatch.delenv(variable)

    report = WindowsStartupPreflight(
        _project(tmp_path),
        python_version=(3, 12),
        module_finder=_all_modules,
        socket_probe=lambda *_args: False,
    ).run("whatsapp")

    assert not report.ready
    failure = render_startup_failure(report)
    assert variable in failure


def test_whatsapp_mode_blank_credentials_are_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ATLAS_WHATSAPP_VERIFY_TOKEN", "   ")
    monkeypatch.setenv("ATLAS_WHATSAPP_ACCESS_TOKEN", "")
    monkeypatch.setenv("ATLAS_WHATSAPP_PHONE_NUMBER_ID", "pnid-123")

    report = WindowsStartupPreflight(
        _project(tmp_path),
        python_version=(3, 12),
        module_finder=_all_modules,
        socket_probe=lambda *_args: False,
    ).run("whatsapp")

    assert not report.ready


def test_whatsapp_mode_error_messages_never_contain_secret_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    secret_token = "super-secret-token-value-xyz"
    secret_access = "super-secret-access-value-xyz"
    monkeypatch.setenv("ATLAS_WHATSAPP_ACCESS_TOKEN", secret_access)
    monkeypatch.delenv("ATLAS_WHATSAPP_VERIFY_TOKEN", raising=False)
    monkeypatch.setenv("ATLAS_WHATSAPP_PHONE_NUMBER_ID", "pnid-123")
    assert secret_token not in os.environ.get("ATLAS_WHATSAPP_VERIFY_TOKEN", "")

    report = WindowsStartupPreflight(
        _project(tmp_path),
        python_version=(3, 12),
        module_finder=_all_modules,
        socket_probe=lambda *_args: False,
    ).run("whatsapp")

    rendered = render_startup_failure(report) + capsys.readouterr().out
    assert secret_token not in rendered
    assert secret_access not in rendered
    assert "ATLAS_WHATSAPP_VERIFY_TOKEN" in rendered


def test_other_modes_do_not_require_whatsapp_credentials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for variable in (
        "ATLAS_WHATSAPP_VERIFY_TOKEN",
        "ATLAS_WHATSAPP_ACCESS_TOKEN",
        "ATLAS_WHATSAPP_PHONE_NUMBER_ID",
    ):
        monkeypatch.delenv(variable, raising=False)

    report = WindowsStartupPreflight(
        _project(tmp_path),
        python_version=(3, 12),
        module_finder=_all_modules,
        socket_probe=lambda *_args: False,
    ).run("text")

    assert report.ready
    assert all(check.name != "Credenciales WhatsApp" for check in report.checks)
