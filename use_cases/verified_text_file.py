"""Verified text-file desktop automation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from tools.executor import ToolExecutor
from tools.tool_context import ToolContext
from use_cases.action_engine import (
    ActionEngineUseCase,
    ActionStep,
    ActionStepResult,
)
from use_cases.wait_engine import WaitEngine


@dataclass(frozen=True)
class VerifiedTextFileAutomationResult:
    """Structured result for create_verified_text_file."""

    workflow_name: str
    target_file: str
    confirmed: bool
    total_steps: int
    executed_steps: int
    successful_steps: int
    failed_steps: int
    completed: bool
    stopped_early: bool
    retries_used: int
    verification_passed: bool
    file_preexisted: bool
    file_created: bool
    rolled_back: bool
    rollback_failed: bool
    step_results: tuple[ActionStepResult, ...] = field(default_factory=tuple)
    warnings: tuple[str, ...] = field(default_factory=tuple)
    summary: str = ""


@dataclass(frozen=True)
class _OriginalFileState:
    existed: bool
    content: bytes | None
    mtime_ns: int | None


class CreateVerifiedTextFileUseCase:
    """Create or replace one text file through the desktop action engine."""

    _WORKFLOW_NAME = "create_verified_text_file"
    _ALLOWED_SUFFIXES = {".txt", ".md"}
    _BLOCKED_SUFFIXES = {".py", ".bat", ".cmd", ".ps1", ".exe", ".reg"}
    _DEFAULT_MAX_CONTENT_CHARS = 100_000

    def __init__(
        self,
        executor: ToolExecutor,
        action_engine: ActionEngineUseCase,
        wait_engine: WaitEngine,
        max_content_chars: int = _DEFAULT_MAX_CONTENT_CHARS,
        max_retries: int = 2,
        timeout: float = 20,
        poll_interval: float = 0.2,
    ) -> None:
        self._executor = executor
        self._action_engine = action_engine
        self._wait_engine = wait_engine
        self._max_content_chars = max_content_chars
        self._max_retries = max_retries
        self._timeout = timeout
        self._poll_interval = poll_interval
        self._retries_used = 0
        self._warnings: list[str] = []
        self._modification_started = False
        self._target_mtime_ns: int | None = None

    def execute(
        self,
        workspace_root: Path,
        target_file: str | Path,
        content: str,
        confirm: Callable[[str], str] | None,
    ) -> VerifiedTextFileAutomationResult:
        """Run the verified text-file workflow."""
        self._retries_used = 0
        self._warnings = []
        self._modification_started = False

        try:
            root, target = self._validate_inputs(
                workspace_root,
                target_file,
                content,
            )
            original_state = self._read_original_state(target)
        except Exception as error:
            return self._failure_result(
                target_file=str(target_file),
                confirmed=False,
                file_preexisted=False,
                file_created=False,
                warnings=(str(error),),
                summary=f"Validacion fallida: {error}",
            )

        if not self._confirmed(confirm, target):
            return self._failure_result(
                target_file=str(target),
                confirmed=False,
                file_preexisted=original_state.existed,
                file_created=False,
                summary="Automatizacion cancelada. No se ejecuto ninguna accion.",
            )

        file_created = False
        parent_created = False
        steps = self._build_steps(root, target, content, original_state)
        automation = self._action_engine.execute(self._WORKFLOW_NAME, steps)
        verification_passed = False
        rolled_back = False
        rollback_failed = False

        try:
            file_created = not original_state.existed and target.exists()
            parent_created = target.parent.exists() and not original_state.existed
            if automation.completed:
                verification_passed = self._verify_content(target, content)

                if not verification_passed:
                    raise RuntimeError("La verificacion final no coincide.")
        except Exception as error:
            self._warnings.append(str(error))
            automation = self._append_failure(automation, "Verificacion final", error)

        completed = automation.completed and verification_passed

        if not completed and self._modification_started:
            rolled_back, rollback_failed = self._rollback(
                target=target,
                original_state=original_state,
            )

        if (
            not completed
            and rolled_back
            and parent_created
            and target.parent.exists()
            and not any(target.parent.iterdir())
        ):
            try:
                target.parent.rmdir()
            except OSError:
                self._warnings.append(
                    f"No se pudo eliminar el directorio creado: {target.parent}"
                )

        return VerifiedTextFileAutomationResult(
            workflow_name=self._WORKFLOW_NAME,
            target_file=str(target),
            confirmed=True,
            total_steps=automation.total_steps,
            executed_steps=automation.executed_steps,
            successful_steps=automation.successful_steps,
            failed_steps=automation.failed_steps,
            completed=completed,
            stopped_early=automation.stopped_early,
            retries_used=self._retries_used,
            verification_passed=verification_passed,
            file_preexisted=original_state.existed,
            file_created=file_created,
            rolled_back=rolled_back,
            rollback_failed=rollback_failed,
            step_results=automation.step_results,
            warnings=tuple(self._warnings),
            summary=self._summary(completed, rolled_back, rollback_failed),
        )

    def _build_steps(
        self,
        workspace_root: Path,
        target_file: Path,
        content: str,
        original_state: _OriginalFileState,
    ) -> list[ActionStep]:
        relative_file = target_file.relative_to(workspace_root)

        return [
            ActionStep(
                "Preparar directorio destino",
                lambda: self._prepare_parent(target_file),
            ),
            ActionStep(
                "Abrir Visual Studio Code",
                lambda: self._retry_transient(
                    "abrir Visual Studio Code",
                    lambda: self._run_tool(
                        "desktop.open_application",
                        {"application": "Visual Studio Code"},
                    ),
                ),
            ),
            ActionStep(
                "Esperar proceso Code.exe",
                lambda: self._wait_engine.wait_process(
                    "Code",
                    timeout=self._timeout,
                    poll_interval=self._poll_interval,
                ),
            ),
            ActionStep(
                "Esperar ventana Visual Studio Code",
                lambda: self._retry_transient(
                    "esperar ventana Visual Studio Code",
                    lambda: self._wait_engine.wait_window(
                        "Visual Studio Code",
                        timeout=self._timeout,
                        poll_interval=self._poll_interval,
                    ),
                ),
            ),
            ActionStep(
                "Abrir workspace",
                lambda: self._run_tool(
                    "desktop.open_folder",
                    {"path": str(workspace_root)},
                ),
            ),
            ActionStep(
                "Preparar archivo destino",
                lambda: self._prepare_target_file(target_file, original_state),
            ),
            ActionStep(
                f"Abrir archivo {relative_file.as_posix()}",
                lambda: self._run_tool(
                    "desktop.open_file",
                    {
                        "path": str(target_file),
                        "application": "Visual Studio Code",
                    },
                ),
            ),
            ActionStep(
                "Esperar pestaña de archivo",
                lambda: self._retry_transient(
                    "esperar pestaña de archivo",
                    lambda: self._wait_file_or_editor_window(target_file),
                ),
            ),
            ActionStep(
                "Copiar contenido al portapapeles",
                lambda: self._begin_modification_and_copy(content),
            ),
            ActionStep(
                "Activar Visual Studio Code",
                lambda: self._retry_transient(
                    "activar Visual Studio Code",
                    lambda: self._activate_editor_window(
                        workspace_root,
                        target_file,
                    ),
                ),
            ),
            ActionStep(
                "Esperar Visual Studio Code activo",
                lambda: self._retry_transient(
                    "esperar Visual Studio Code activo",
                    lambda: self._wait_engine.wait_active_window(
                        "Visual Studio Code",
                        timeout=self._timeout,
                        poll_interval=self._poll_interval,
                    ),
                ),
            ),
            ActionStep(
                "Seleccionar contenido",
                lambda: self._run_tool(
                    "desktop.press_hotkey",
                    {
                        "window_title": target_file.name,
                        "keys": ["ctrl", "a"],
                    },
                ),
            ),
            ActionStep(
                "Pegar contenido",
                lambda: self._run_tool(
                    "desktop.press_hotkey",
                    {
                        "window_title": target_file.name,
                        "keys": ["ctrl", "v"],
                    },
                ),
            ),
            ActionStep(
                "Guardar archivo",
                lambda: self._run_tool(
                    "desktop.press_hotkey",
                    {
                        "window_title": target_file.name,
                        "keys": ["ctrl", "s"],
                    },
                ),
            ),
            ActionStep(
                "Esperar archivo actualizado",
                lambda: self._retry_transient(
                    "esperar archivo actualizado",
                    lambda: self._wait_engine.wait_file_updated(
                        target_file,
                        previous_mtime_ns=self._target_mtime_ns,
                        timeout=self._timeout,
                        poll_interval=self._poll_interval,
                    ),
                ),
            ),
            ActionStep(
                "Verificar contenido exacto",
                lambda: self._verify_step(target_file, content),
            ),
        ]

    def _validate_inputs(
        self,
        workspace_root: Path,
        target_file: str | Path,
        content: str,
    ) -> tuple[Path, Path]:
        root = workspace_root.resolve()

        if not root.exists():
            raise ValueError(f"workspace_root no existe: {root}")

        if not root.is_dir():
            raise ValueError(f"workspace_root no es una carpeta: {root}")

        if not isinstance(content, str):
            raise TypeError("content debe ser str.")

        if content == "":
            raise ValueError("content no puede estar vacio.")

        if len(content) > self._max_content_chars:
            raise ValueError("content excede el limite configurado.")

        raw_target = Path(target_file)
        candidate = raw_target if raw_target.is_absolute() else root / raw_target
        resolved = candidate.resolve()

        if not self._is_relative_to(resolved, root):
            raise ValueError("target_file debe estar dentro de workspace_root.")

        if resolved.suffix.lower() in self._BLOCKED_SUFFIXES:
            raise ValueError(f"extension no permitida: {resolved.suffix}")

        if resolved.suffix.lower() not in self._ALLOWED_SUFFIXES:
            raise ValueError("target_file debe tener extension .txt o .md.")

        if resolved.exists() and not resolved.is_file():
            raise ValueError("target_file no es un archivo.")

        parent = resolved.parent

        if parent.exists() and not parent.is_dir():
            raise ValueError("el directorio padre no es una carpeta.")

        if not parent.exists() and not self._is_relative_to(parent, root):
            raise ValueError("directorio padre fuera de workspace_root.")

        if not parent.exists() and not parent.parent.exists():
            raise ValueError("el directorio padre no puede crearse de forma controlada.")

        self._check_reasonable_permissions(root, parent, resolved)

        return root, resolved

    def _check_reasonable_permissions(
        self,
        root: Path,
        parent: Path,
        target: Path,
    ) -> None:
        probe_dir = parent if parent.exists() else parent.parent

        if not probe_dir.exists():
            raise ValueError(f"directorio no disponible: {probe_dir}")

        if target.exists():
            with target.open("rb"):
                pass

        probe = probe_dir / ".atlas_write_probe.tmp"
        try:
            probe.write_text("ok", encoding="utf-8")
            probe.unlink()
        except OSError as error:
            raise PermissionError(f"sin permisos razonables en {root}") from error

    def _read_original_state(
        self,
        target: Path,
    ) -> _OriginalFileState:
        if not target.exists():
            return _OriginalFileState(False, None, None)

        return _OriginalFileState(
            existed=True,
            content=target.read_bytes(),
            mtime_ns=target.stat().st_mtime_ns,
        )

    def _confirmed(
        self,
        confirm: Callable[[str], str] | None,
        target: Path,
    ) -> bool:
        if confirm is None:
            return False

        answer = confirm(
            "\n".join(
                [
                    "Se modificara:",
                    "",
                    str(target),
                    "",
                    "¿Confirmas ejecutar la automatizacion? [s/N]: ",
                ]
            )
        )

        return answer.strip().lower() in {"s", "si", "sí", "y", "yes"}

    def _prepare_parent(
        self,
        target_file: Path,
    ) -> str:
        target_file.parent.mkdir(parents=False, exist_ok=True)
        return f"Directorio preparado: {target_file.parent}"

    def _prepare_target_file(
        self,
        target_file: Path,
        original_state: _OriginalFileState,
    ) -> str:
        if original_state.existed:
            self._target_mtime_ns = target_file.stat().st_mtime_ns
            return "Archivo destino existente conservado."

        self._modification_started = True
        target_file.write_text("", encoding="utf-8")
        self._target_mtime_ns = target_file.stat().st_mtime_ns
        return f"Archivo destino creado: {target_file}"

    def _begin_modification_and_copy(
        self,
        content: str,
    ) -> str:
        self._modification_started = True
        self._warnings.append(
            "El portapapeles puede quedar modificado; no existe restauracion segura configurada."
        )
        return self._run_tool(
            "desktop.copy_clipboard_text",
            {"text": content},
        )

    def _wait_file_or_editor_window(
        self,
        target_file: Path,
    ):
        file_result = self._wait_engine.wait_window(
            target_file.name,
            timeout=self._timeout,
            poll_interval=self._poll_interval,
        )

        if file_result.completed:
            return file_result

        self._warnings.append(
            f"No se detecto la pestaña {target_file.name}; se valida ventana de VS Code."
        )
        return self._wait_engine.wait_window(
            "Visual Studio Code",
            timeout=self._timeout,
            poll_interval=self._poll_interval,
        )

    def _activate_editor_window(
        self,
        workspace_root: Path,
        target_file: Path,
    ) -> str:
        windows = self._list_editor_windows(target_file.name)

        if not windows:
            raise RuntimeError("No se encontraron ventanas de Visual Studio Code.")

        root_name = workspace_root.name.lower()
        file_name = target_file.name.lower()
        scored = []

        for window in windows:
            title = str(window.get("title", "")).lower()
            score = 0

            if file_name in title:
                score += 8

            if root_name and root_name in title:
                score += 4

            if "visual studio code" in title:
                score += 2

            scored.append((score, int(window.get("handle", 0)), window))

        scored.sort(
            key=lambda item: (
                -item[0],
                item[1],
                str(item[2].get("title", "")).lower(),
            )
        )
        best = scored[0][2]
        handle = int(best["handle"])
        self._run_tool("desktop.bring_window_to_front", {"handle": handle})

        return f"Ventana activada: {best['title']}"

    def _list_editor_windows(
        self,
        file_name: str,
    ) -> list[dict[str, object]]:
        windows_by_handle: dict[int, dict[str, object]] = {}

        for title in (file_name, "Visual Studio Code", "Code"):
            result = self._executor.execute(
                "desktop.list_windows",
                ToolContext(parameters={"title": title}),
            )

            if not isinstance(result, list):
                raise RuntimeError("Respuesta de ventanas invalida.")

            for window in result:
                if not isinstance(window, dict):
                    continue

                window_title = str(window.get("title", ""))
                normalized = window_title.lower()

                if (
                    "visual studio code" not in normalized
                    and not normalized.endswith(" - code")
                    and file_name.lower() not in normalized
                ):
                    continue

                handle = window.get("handle")

                if isinstance(handle, int):
                    windows_by_handle[handle] = window

        return sorted(
            windows_by_handle.values(),
            key=lambda window: (
                str(window.get("title", "")).lower(),
                int(window.get("handle", 0)),
            ),
        )

    def _retry_transient(
        self,
        label: str,
        operation: Callable[[], object],
    ) -> object:
        attempts = self._max_retries + 1
        last_error: Exception | None = None

        for attempt in range(1, attempts + 1):
            self._warnings.append(f"{label}: intento {attempt}/{attempts}")

            try:
                result = operation()
            except Exception as error:
                last_error = error

                if attempt < attempts:
                    self._retries_used += 1
                    continue

                raise

            if hasattr(result, "completed") and not result.completed:
                if attempt < attempts:
                    self._retries_used += 1
                    continue

            return result

        assert last_error is not None
        raise last_error

    def _run_tool(
        self,
        tool_name: str,
        parameters: dict[str, object],
    ) -> str:
        result = self._executor.execute(
            tool_name,
            ToolContext(parameters=parameters),
        )

        return str(result)

    def _verify_step(
        self,
        target_file: Path,
        expected_content: str,
    ) -> str:
        if not self._verify_content(target_file, expected_content):
            raise RuntimeError("El contenido escrito no coincide exactamente.")

        return "Contenido verificado exactamente."

    def _verify_content(
        self,
        target_file: Path,
        expected_content: str,
    ) -> bool:
        if not target_file.exists() or not target_file.is_file():
            return False

        return target_file.read_text(encoding="utf-8") == expected_content

    def _append_failure(
        self,
        automation,
        step_name: str,
        error: Exception,
    ):
        results = list(automation.step_results)
        results.append(
            ActionStepResult(
                step_name=step_name,
                success=False,
                message="",
                error=str(error),
            )
        )

        successful = sum(1 for result in results if result.success)
        failed = sum(1 for result in results if not result.success)

        return type(automation)(
            workflow_name=automation.workflow_name,
            total_steps=automation.total_steps,
            executed_steps=len(results),
            successful_steps=successful,
            failed_steps=failed,
            completed=False,
            stopped_early=True,
            step_results=tuple(results),
            summary=automation.summary,
        )

    def _rollback(
        self,
        target: Path,
        original_state: _OriginalFileState,
    ) -> tuple[bool, bool]:
        try:
            if original_state.existed:
                assert original_state.content is not None
                target.write_bytes(original_state.content)

                if target.read_bytes() != original_state.content:
                    raise RuntimeError("restauracion no coincide exactamente")
            else:
                if target.exists():
                    target.unlink()

                if target.exists():
                    raise RuntimeError("archivo nuevo no eliminado")
        except Exception as error:
            self._warnings.append(f"ADVERTENCIA CRITICA: rollback fallido: {error}")
            return (True, True)

        return (True, False)

    def _failure_result(
        self,
        target_file: str,
        confirmed: bool,
        file_preexisted: bool,
        file_created: bool,
        warnings: tuple[str, ...] = tuple(),
        summary: str = "",
    ) -> VerifiedTextFileAutomationResult:
        return VerifiedTextFileAutomationResult(
            workflow_name=self._WORKFLOW_NAME,
            target_file=target_file,
            confirmed=confirmed,
            total_steps=0,
            executed_steps=0,
            successful_steps=0,
            failed_steps=1 if warnings else 0,
            completed=False,
            stopped_early=True,
            retries_used=0,
            verification_passed=False,
            file_preexisted=file_preexisted,
            file_created=file_created,
            rolled_back=False,
            rollback_failed=False,
            step_results=tuple(),
            warnings=warnings,
            summary=summary or "Automatizacion no ejecutada.",
        )

    def _summary(
        self,
        completed: bool,
        rolled_back: bool,
        rollback_failed: bool,
    ) -> str:
        if completed:
            return "Archivo de trabajo verificado creado correctamente."

        if rollback_failed:
            return "Automatizacion fallida; rollback fallido."

        if rolled_back:
            return "Automatizacion fallida; rollback exacto verificado."

        return "Automatizacion fallida."

    def _is_relative_to(
        self,
        path: Path,
        parent: Path,
    ) -> bool:
        try:
            path.relative_to(parent)
        except ValueError:
            return False

        return True
