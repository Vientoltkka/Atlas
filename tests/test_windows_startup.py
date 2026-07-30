from __future__ import annotations

import logging
from pathlib import Path

from core.startup import (
    CheckStatus,
    StartupCheck,
    StartupReport,
    WindowsStartupPreflight,
    configure_operational_logging,
    render_startup_banner,
    render_startup_failure,
    sanitize_log_message,
)


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


def test_preflight_creates_operational_directories_idempotently(tmp_path) -> None:
    root = _project(tmp_path)
    preflight = WindowsStartupPreflight(
        root,
        python_version=(3, 14),
        module_finder=_all_modules,
        socket_probe=lambda *_args: False,
    )

    first = preflight.run("text")
    second = preflight.run("text")

    assert first.ready
    assert second.ready
    assert (root / "logs").is_dir()
    assert (root / ".atlas" / "execution_sessions").is_dir()


def test_preflight_blocks_an_unsupported_python(tmp_path) -> None:
    report = WindowsStartupPreflight(
        _project(tmp_path),
        python_version=(3, 15),
        module_finder=_all_modules,
        socket_probe=lambda *_args: True,
    ).run("text")

    assert not report.ready
    assert any(check.name == "Python" for check in report.errors)


def test_preflight_blocks_missing_text_dependency(tmp_path) -> None:
    def find_module(name: str) -> object | None:
        return None if name == "ollama" else object()

    report = WindowsStartupPreflight(
        _project(tmp_path),
        module_finder=find_module,
        socket_probe=lambda *_args: False,
    ).run("text")

    assert not report.ready
    assert any("ollama" in check.name for check in report.errors)


def test_preflight_only_warns_for_optional_voice_dependency(tmp_path) -> None:
    def find_module(name: str) -> object | None:
        return None if name == "faster_whisper" else object()

    report = WindowsStartupPreflight(
        _project(tmp_path),
        module_finder=find_module,
        socket_probe=lambda *_args: False,
    ).run("text")

    assert report.ready
    assert any("faster_whisper" in check.name for check in report.warnings)


def test_preflight_blocks_invalid_environment_without_exposing_values(
    tmp_path,
) -> None:
    root = _project(tmp_path)
    (root / ".env").write_text(
        "OPENAI_API_KEY=super-secret\nINVALID_LINE\n",
        encoding="utf-8",
    )

    report = WindowsStartupPreflight(
        root,
        module_finder=_all_modules,
        socket_probe=lambda *_args: True,
    ).run("text")
    rendered = render_startup_failure(report)

    assert not report.ready
    assert "lineas 2" in rendered
    assert "super-secret" not in rendered


def test_preflight_reports_unwritable_operational_directory(tmp_path) -> None:
    def fail_write(_directory: Path) -> None:
        raise PermissionError("denied")

    report = WindowsStartupPreflight(
        _project(tmp_path),
        module_finder=_all_modules,
        socket_probe=lambda *_args: True,
        write_probe=fail_write,
    ).run("text")

    assert not report.ready
    assert any(check.name.startswith("Directorio") for check in report.errors)


def test_banner_identifies_mode_root_capabilities_and_exit_commands(
    tmp_path,
) -> None:
    report = StartupReport(
        project_root=tmp_path,
        mode="text",
        checks=(
            StartupCheck("Ollama", CheckStatus.WARNING, "No disponible."),
        ),
        capabilities=("texto",),
    )

    banner = render_startup_banner(report)

    assert "Atlas - Base operativa v1.0" in banner
    assert "Estado: preparado" in banner
    assert "Modo: texto" in banner
    assert "Texto: disponible" in banner
    assert "Voz: no configurada" in banner
    assert "Modelo local: no disponible" in banner
    assert "Modo operativo basico: disponible" in banner
    assert f"Directorio de trabajo: {tmp_path}" in banner
    assert 'usa "salir" para cerrar' in banner


def test_operational_log_is_bounded_idempotent_and_redacted(tmp_path) -> None:
    (tmp_path / "logs").mkdir()
    logger = configure_operational_logging(tmp_path)
    same_logger = configure_operational_logging(tmp_path)
    initial_count = len(logger.handlers)
    logger.error(sanitize_log_message("token=visible password:visible"))
    for handler in logger.handlers:
        handler.flush()

    content = (tmp_path / "logs" / "atlas.log").read_text(encoding="utf-8")
    matching_handlers = [
        handler
        for handler in logger.handlers
        if getattr(handler, "baseFilename", None)
        == str((tmp_path / "logs" / "atlas.log").resolve())
    ]

    assert same_logger is logger
    assert len(logger.handlers) == initial_count
    assert len(matching_handlers) == 1
    assert "visible" not in content
    assert content.count("[REDACTED]") == 2
    assert matching_handlers[0].maxBytes == 1_000_000
    assert matching_handlers[0].backupCount == 3

    for handler in list(logger.handlers):
        if getattr(handler, "baseFilename", None) == str(
            (tmp_path / "logs" / "atlas.log").resolve()
        ):
            handler.close()
            logger.removeHandler(handler)


def test_operation_guide_documents_one_official_start_command() -> None:
    guide = (
        Path(__file__).resolve().parents[1]
        / "docs"
        / "manual"
        / "operation.md"
    ).read_text(encoding="utf-8-sig")

    assert "## Comando oficial" in guide
    assert "python -B main.py" in guide
    assert "python main.py" not in guide
    assert "variante de desarrollo" not in guide
