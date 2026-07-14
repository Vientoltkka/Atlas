"""Sequential desktop action execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from tools.executor import ToolExecutor
from tools.tool_context import ToolContext
from use_cases.wait_engine import WaitEngine, WaitResult


@dataclass(frozen=True)
class ActionStep:
    """Explicit action step."""

    name: str
    operation: Callable[[], str | WaitResult]


@dataclass(frozen=True)
class ActionStepResult:
    """Result of one action step."""

    step_name: str
    success: bool
    message: str
    error: str | None = None


@dataclass(frozen=True)
class AutomationResult:
    """Result of a sequential automation."""

    workflow_name: str
    total_steps: int
    executed_steps: int
    successful_steps: int
    failed_steps: int
    completed: bool
    stopped_early: bool
    step_results: tuple[ActionStepResult, ...] = field(default_factory=tuple)
    summary: str = ""


class ActionEngineUseCase:
    """Execute explicit action steps in order."""

    def execute(
        self,
        workflow_name: str,
        steps: list[ActionStep],
    ) -> AutomationResult:
        """Execute a workflow until success or first blocking failure."""
        results: list[ActionStepResult] = []

        for step in steps:
            try:
                operation_result = step.operation()
            except Exception as exc:
                results.append(
                    ActionStepResult(
                        step_name=step.name,
                        success=False,
                        message="",
                        error=str(exc),
                    )
                )
                break

            if isinstance(operation_result, WaitResult):
                results.append(
                    ActionStepResult(
                        step_name=step.name,
                        success=operation_result.completed,
                        message=self._format_wait_result(operation_result),
                        error=None
                        if operation_result.completed
                        else operation_result.error,
                    )
                )

                if not operation_result.completed:
                    break

                continue

            results.append(
                ActionStepResult(
                    step_name=step.name,
                    success=True,
                    message=operation_result,
                )
            )

        successful_steps = sum(1 for result in results if result.success)
        failed_steps = sum(1 for result in results if not result.success)
        completed = len(results) == len(steps) and failed_steps == 0
        stopped_early = len(results) < len(steps)

        return AutomationResult(
            workflow_name=workflow_name,
            total_steps=len(steps),
            executed_steps=len(results),
            successful_steps=successful_steps,
            failed_steps=failed_steps,
            completed=completed,
            stopped_early=stopped_early,
            step_results=tuple(results),
            summary=self._summary(
                workflow_name,
                completed,
                len(results),
                failed_steps,
            ),
        )

    def _summary(
        self,
        workflow_name: str,
        completed: bool,
        executed_steps: int,
        failed_steps: int,
    ) -> str:
        """Return a deterministic automation summary."""
        state = "completada" if completed else "incompleta"

        return (
            f"Automatizacion {state}: {workflow_name}. "
            f"Pasos ejecutados: {executed_steps}. "
            f"Pasos fallidos: {failed_steps}."
        )

    def _format_wait_result(
        self,
        result: WaitResult,
    ) -> str:
        """Return a compact wait result message."""
        state = "completada" if result.completed else "agotada"

        return (
            f"Espera {state}: {result.description}. "
            f"Condicion: {result.condition}. "
            f"Tiempo: {result.elapsed_time:.2f}/{result.timeout:.2f}s."
        )


class PrepareAtlasWorkspaceUseCase:
    """Prepare Atlas workspace using existing desktop tools."""

    def __init__(
        self,
        executor: ToolExecutor,
        action_engine: ActionEngineUseCase,
        wait_engine: WaitEngine | None = None,
    ) -> None:
        self._executor = executor
        self._action_engine = action_engine
        self._wait_engine = wait_engine or WaitEngine(executor)

    def execute(
        self,
        project_root: Path,
        editor_file: str | Path = Path("core") / "orchestrator.py",
    ) -> AutomationResult:
        """Prepare Atlas workspace."""
        root = project_root.resolve()
        file_path = self._validate_input(root, Path(editor_file))
        relative_file = file_path.relative_to(root)

        steps = [
            ActionStep(
                name="Abrir Visual Studio Code",
                operation=lambda: self._run_tool(
                    "desktop.open_application",
                    {"application": "Visual Studio Code"},
                ),
            ),
            ActionStep(
                name="Esperar proceso Code.exe",
                operation=lambda: self._wait_engine.wait_process(
                    "Code",
                    timeout=20,
                    poll_interval=0.2,
                ),
            ),
            ActionStep(
                name="Esperar ventana Visual Studio Code",
                operation=lambda: self._wait_engine.wait_window(
                    "Visual Studio Code",
                    timeout=20,
                    poll_interval=0.2,
                ),
            ),
            ActionStep(
                name="Activar Visual Studio Code",
                operation=lambda: self._activate_editor_window(
                    root,
                    relative_file,
                    file_path.name,
                ),
            ),
            ActionStep(
                name="Esperar ventana activa Visual Studio Code",
                operation=lambda: self._wait_engine.wait_active_window(
                    "Visual Studio Code",
                    timeout=20,
                    poll_interval=0.2,
                ),
            ),
            ActionStep(
                name="Abrir carpeta Atlas",
                operation=lambda: self._run_tool(
                    "desktop.open_folder",
                    {"path": str(root)},
                ),
            ),
            ActionStep(
                name="Esperar carpeta Atlas disponible",
                operation=lambda: self._wait_engine.wait_file(
                    root,
                    timeout=20,
                    poll_interval=0.2,
                ),
            ),
            ActionStep(
                name=f"Abrir archivo {relative_file.as_posix()}",
                operation=lambda: self._run_tool(
                    "desktop.open_file",
                    {
                        "path": str(file_path),
                        "application": "Visual Studio Code",
                    },
                ),
            ),
            ActionStep(
                name=f"Esperar archivo {relative_file.as_posix()} disponible",
                operation=lambda: self._wait_engine.wait_file(
                    file_path,
                    timeout=20,
                    poll_interval=0.2,
                ),
            ),
            ActionStep(
                name="Confirmar Visual Studio Code activo",
                operation=lambda: self._activate_editor_window(
                    root,
                    relative_file,
                    file_path.name,
                ),
            ),
        ]

        return self._action_engine.execute(
            "open_vscode_workspace",
            steps,
        )

    def _run_tool(
        self,
        tool_name: str,
        parameters: dict[str, object],
    ) -> str:
        """Run a registered tool."""
        result = self._executor.execute(
            tool_name,
            ToolContext(parameters=parameters),
        )

        return str(result)

    def _activate_editor_window(
        self,
        project_root: Path,
        relative_file: Path,
        expected_file_name: str,
    ) -> str:
        """Resolve and activate the intended Visual Studio Code window."""
        window = self._resolve_editor_window(
            project_root,
            relative_file,
            expected_file_name,
        )
        handle = int(window["handle"])

        self._run_tool(
            "desktop.bring_window_to_front",
            {"handle": handle},
        )

        return f"Ventana activada: {window['title']}"

    def _resolve_editor_window(
        self,
        project_root: Path,
        relative_file: Path,
        expected_file_name: str,
    ) -> dict[str, object]:
        """Resolve one compatible Visual Studio Code window."""
        windows = self._list_editor_windows()

        if not windows:
            raise RuntimeError("No se encontraron ventanas de Visual Studio Code.")

        scored = [
            (
                self._editor_window_score(
                    window,
                    project_root,
                    relative_file,
                    expected_file_name,
                ),
                self._window_order(window),
                window,
            )
            for window in windows
        ]
        scored.sort(
            key=lambda item: (
                -item[0],
                item[1],
                str(item[2].get("title", "")).lower(),
                int(item[2].get("handle", 0)),
            )
        )
        best_score = scored[0][0]
        best_order = scored[0][1]
        best_matches = [
            window
            for score, order, window in scored
            if score == best_score and order == best_order
        ]

        if len(best_matches) != 1:
            titles = ", ".join(str(window.get("title", "")) for window in best_matches)
            raise RuntimeError(
                f"Varias ventanas de Visual Studio Code siguen siendo ambiguas: {titles}"
            )

        return best_matches[0]

    def _list_editor_windows(self) -> list[dict[str, object]]:
        """List visible windows compatible with Visual Studio Code."""
        windows_by_handle: dict[int, dict[str, object]] = {}

        for query in ("Visual Studio Code", "Code"):
            result = self._executor.execute(
                "desktop.list_windows",
                ToolContext(parameters={"title": query}),
            )

            if not isinstance(result, list):
                raise RuntimeError("Respuesta de ventanas invalida.")

            for window in result:
                if not isinstance(window, dict):
                    continue

                title = str(window.get("title", ""))

                if not self._is_visual_studio_code_title(title):
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

    def _is_visual_studio_code_title(
        self,
        title: str,
    ) -> bool:
        """Return whether a title is compatible with Visual Studio Code."""
        normalized = title.lower()

        return (
            "visual studio code" in normalized
            or normalized.endswith(" - code")
            or " - visual studio code" in normalized
        )

    def _editor_window_score(
        self,
        window: dict[str, object],
        project_root: Path,
        relative_file: Path,
        expected_file_name: str,
    ) -> int:
        """Score a VS Code window for deterministic workflow activation."""
        title = str(window.get("title", "")).lower()
        root_name = project_root.name.lower()
        relative_posix = relative_file.as_posix().lower()
        relative_windows = str(relative_file).lower()
        file_name = expected_file_name.lower()
        score = 0

        if root_name and root_name in title:
            score += 8

        if relative_posix in title or relative_windows in title:
            score += 6

        if file_name in title:
            score += 4

        if "visual studio code" in title:
            score += 2

        return score

    def _window_order(
        self,
        window: dict[str, object],
    ) -> int:
        """Return the preserved Windows Z-order for deterministic tie breaks."""
        order = window.get("order")

        if isinstance(order, int) and order >= 0:
            return order

        return 999999

    def _validate_input(
        self,
        project_root: Path,
        editor_file: Path,
    ) -> Path:
        """Validate workflow input before executing actions."""
        if not project_root.exists():
            raise ValueError(f"project_root no existe: {project_root}")

        if not project_root.is_dir():
            raise ValueError(f"project_root no es una carpeta: {project_root}")

        candidate = (
            editor_file
            if editor_file.is_absolute()
            else project_root / editor_file
        ).resolve()

        if not candidate.exists():
            raise ValueError(f"editor_file no existe: {candidate}")

        if not candidate.is_file():
            raise ValueError(f"editor_file no es un archivo: {candidate}")

        if candidate.suffix != ".py":
            raise ValueError("editor_file debe tener extension .py.")

        try:
            candidate.relative_to(project_root)
        except ValueError as exc:
            raise ValueError(
                "editor_file debe estar dentro de project_root."
            ) from exc

        return candidate


class CopyAndPasteTextUseCase:
    """Copy text to the clipboard and paste it into one target window."""

    def __init__(
        self,
        executor: ToolExecutor,
        action_engine: ActionEngineUseCase,
    ) -> None:
        self._executor = executor
        self._action_engine = action_engine

    def execute(
        self,
        text: str,
        target_window_title: str,
    ) -> AutomationResult:
        """Execute the explicit copy-and-paste workflow."""
        self._validate_input(text, target_window_title)

        steps = [
            ActionStep(
                name="Copiar texto al portapapeles",
                operation=lambda: self._run_tool(
                    "desktop.copy_clipboard_text",
                    {"text": text},
                ),
            ),
            ActionStep(
                name="Activar ventana destino",
                operation=lambda: self._run_tool(
                    "desktop.activate_window",
                    {"window_title": target_window_title},
                ),
            ),
            ActionStep(
                name="Pegar contenido del portapapeles",
                operation=lambda: self._run_tool(
                    "desktop.press_hotkey",
                    {
                        "window_title": target_window_title,
                        "keys": ["ctrl", "v"],
                    },
                ),
            ),
        ]

        return self._action_engine.execute("copy_and_paste_text", steps)

    def _run_tool(
        self,
        tool_name: str,
        parameters: dict[str, object],
    ) -> str:
        """Run a registered tool and return its message."""
        result = self._executor.execute(
            tool_name,
            ToolContext(parameters=parameters),
        )

        return str(result)

    def _validate_input(
        self,
        text: str,
        target_window_title: str,
    ) -> None:
        """Validate workflow inputs."""
        if not isinstance(text, str):
            raise TypeError("El contenido del portapapeles debe ser texto.")

        if text == "":
            raise ValueError("No se puede copiar texto vacio.")

        if not target_window_title.strip():
            raise ValueError("Falta la ventana destino.")


class RestartApplicationUseCase:
    """Restart one application with normal close semantics."""

    def __init__(
        self,
        executor: ToolExecutor,
        action_engine: ActionEngineUseCase,
        wait_engine: WaitEngine | None = None,
    ) -> None:
        self._executor = executor
        self._action_engine = action_engine
        self._wait_engine = wait_engine or WaitEngine(executor)

    def execute(
        self,
        application_name: str,
    ) -> AutomationResult:
        """Execute the explicit restart workflow."""
        if not application_name.strip():
            raise ValueError("Falta la aplicacion.")

        steps = [
            ActionStep(
                name="Comprobar aplicacion en ejecucion",
                operation=lambda: self._check_running(application_name),
            ),
            ActionStep(
                name="Solicitar cierre normal",
                operation=lambda: self._close_if_running(application_name),
            ),
            ActionStep(
                name="Verificar cierre",
                operation=lambda: self._wait_engine.wait_application(
                    application_name,
                    timeout=20,
                    poll_interval=0.2,
                    opened=False,
                ),
            ),
            ActionStep(
                name="Abrir aplicacion",
                operation=lambda: self._run_tool(
                    "desktop.open_application",
                    {"application": application_name},
                ),
            ),
            ActionStep(
                name="Verificar ejecucion",
                operation=lambda: self._wait_engine.wait_application(
                    application_name,
                    timeout=20,
                    poll_interval=0.2,
                ),
            ),
        ]

        return self._action_engine.execute("restart_application", steps)

    def _run_tool(
        self,
        tool_name: str,
        parameters: dict[str, object],
    ) -> str:
        """Run a registered tool and return its message."""
        result = self._executor.execute(
            tool_name,
            ToolContext(parameters=parameters),
        )

        return str(result)

    def _processes(
        self,
        application_name: str,
    ) -> list[dict[str, object]]:
        """Return process matches for an application."""
        result = self._executor.execute(
            "desktop.list_processes",
            ToolContext(parameters={"query": application_name}),
        )

        if not isinstance(result, list):
            raise RuntimeError("Respuesta de procesos invalida.")

        return result

    def _check_running(
        self,
        application_name: str,
    ) -> str:
        """Check whether the application is currently running."""
        processes = self._processes(application_name)

        if not processes:
            return "La aplicacion no estaba abierta."

        return f"Procesos detectados: {len(processes)}"

    def _close_if_running(
        self,
        application_name: str,
    ) -> str:
        """Request normal close for the only matching process."""
        processes = self._processes(application_name)

        if not processes:
            return "No habia procesos que cerrar."

        if len(processes) != 1:
            raise RuntimeError("Varias coincidencias; cierre no automatico.")

        pid = processes[0].get("pid")

        if not isinstance(pid, int):
            raise RuntimeError("PID invalido.")

        return self._run_tool("desktop.close_application", {"pid": pid})

    def _verify_closed(
        self,
        application_name: str,
    ) -> str:
        """Verify that the application has no running processes."""
        if self._processes(application_name):
            raise RuntimeError("La aplicacion continua abierta.")

        return "Aplicacion cerrada."

    def _verify_running(
        self,
        application_name: str,
    ) -> str:
        """Verify that the application is running."""
        processes = self._processes(application_name)

        if not processes:
            raise RuntimeError("La aplicacion no se inicio.")

        return f"Aplicacion en ejecucion. Procesos: {len(processes)}"
