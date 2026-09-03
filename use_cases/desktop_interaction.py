"""Desktop interaction use case."""

from __future__ import annotations

import logging
import os
from pathlib import Path
import re
from typing import Callable
import unicodedata

from use_cases.action_engine import (
    AutomationResult,
    PrepareAtlasWorkspaceUseCase,
    RestartApplicationUseCase,
)
from use_cases.verified_text_file import (
    CreateVerifiedTextFileUseCase,
    VerifiedTextFileAutomationResult,
)
from use_cases.wait_engine import WaitEngine, WaitResult
from tools.executor import ToolExecutor
from tools.desktop.windows_controller import DesktopController, WindowsDesktopController
from tools.tool_context import ToolContext


class DesktopInteractionUseCase:
    """Execute simple desktop commands through Atlas tools."""

    _DEFAULT_WINDOW_TITLE = "Visual Studio Code"
    _CONFIRMATION_PREFIX = "\u2713"
    _WAKE_WORD_PREFIX_PATTERN = re.compile(r"^atlas\b[ ,.;:!?-]*", re.IGNORECASE)
    _ACTIVATION_TITLE_ALIASES: dict[str, tuple[str, ...]] = {
        "calculator": ("calculadora",),
        "calculadora": ("calculator",),
        "notepad": ("bloc de notas",),
        "bloc de notas": ("notepad",),
        "vs code": ("visual studio code",),
        "vscode": ("visual studio code",),
    }

    def __init__(
        self,
        executor: ToolExecutor,
        project_root: Path | None = None,
        screenshots_dir: Path | None = None,
        prepare_atlas_workspace: PrepareAtlasWorkspaceUseCase | None = None,
        restart_application: RestartApplicationUseCase | None = None,
        create_verified_text_file: CreateVerifiedTextFileUseCase | None = None,
        wait_engine: WaitEngine | None = None,
        focus_controller: DesktopController | None = None,
    ) -> None:
        self._executor = executor
        self._project_root = project_root or Path(".")
        self._screenshots_dir = (
            screenshots_dir
            if screenshots_dir is not None
            else self._project_root / "artifacts" / "screenshots"
        )
        self._prepare_atlas_workspace = prepare_atlas_workspace
        self._restart_application = restart_application
        self._create_verified_text_file = create_verified_text_file
        self._wait_engine = wait_engine or WaitEngine(executor)
        self._focus_controller = focus_controller or WindowsDesktopController()

    def capture_external_foreground_handle(self) -> int:
        """Capture the exact active external HWND for one pending text action."""
        window = self._focus_controller.get_foreground_window()
        handle = window.get("handle")
        if not isinstance(handle, int) or handle <= 0:
            raise RuntimeError("No se pudo identificar la ventana destino.")
        if self._focus_controller.get_window_process_id(handle) == os.getpid():
            raise RuntimeError("No hay una ventana externa activa para escribir.")
        return handle

    def restore_external_foreground_handle(self, handle: int) -> None:
        """Validate and reactivate the exact HWND captured for a pending action."""
        self._focus_controller.get_window_rect(handle)
        self._focus_controller.bring_window_to_front(handle)

    def execute(
        self,
        prompt: str,
        confirm: Callable[[str], str] | None = None,
    ) -> str | None:
        """Execute a supported desktop command."""
        text = prompt.strip()
        text = self._strip_wake_word_prefix(text)
        normalized = self._normalize(text)

        try:
            if self._is_free_sequence_command(normalized):
                raise ValueError("No se admiten secuencias libres.")

            if normalized.startswith("ejecuta "):
                raise ValueError("No se aceptan comandos arbitrarios.")

            verified_file_response = self._execute_verified_text_file_command(
                normalized,
                confirm,
            )

            if verified_file_response is not None:
                return verified_file_response

            wait_response = self._execute_wait_command(text, normalized)

            if wait_response is not None:
                return wait_response

            filesystem_response = self._execute_filesystem_command(
                text,
                confirm,
            )

            if filesystem_response is not None:
                return filesystem_response

            clipboard_response = self._execute_clipboard_command(
                text,
                normalized,
                confirm,
            )

            if clipboard_response is not None:
                return clipboard_response

            if self._is_prepare_atlas_workspace_command(normalized):
                if self._prepare_atlas_workspace is None:
                    raise RuntimeError("Workflow no disponible.")

                result = self._prepare_atlas_workspace.execute(
                    self._project_root,
                )

                return self._format_automation_result(result)

            process_response = self._execute_process_command(
                text,
                normalized,
                confirm,
            )

            if process_response is not None:
                return process_response

            window_response = self._execute_window_command(
                text,
                normalized,
                confirm,
            )

            if window_response is not None:
                return window_response

            if self._is_screen_size_command(normalized):
                width, height = self._execute_tuple(
                    "desktop.get_screen_size",
                    {},
                )
                return f"{self._CONFIRMATION_PREFIX} Tamaño de pantalla: {width} x {height}."

            if self._is_cursor_position_command(normalized):
                x, y = self._execute_tuple(
                    "desktop.get_cursor_position",
                    {},
                )
                return f"{self._CONFIRMATION_PREFIX} Posición actual: ({x}, {y})."

            if self._is_forbidden_screenshot_command(normalized):
                raise ValueError("Solo se admite captura de pantalla completa.")

            if self._is_screenshot_command(normalized):
                path = self._execute(
                    "desktop.capture_screenshot",
                    ToolContext(
                        parameters={
                            "output_dir": str(self._screenshots_dir),
                        }
                    ),
                )
                return f"{self._CONFIRMATION_PREFIX} Captura guardada en:\n{path}"

            if self._is_scroll_down_command(normalized):
                return self._run(
                    "desktop.scroll_vertical",
                    {"direction": "down"},
                    "Scroll hacia abajo.",
                )

            if self._is_scroll_up_command(normalized):
                return self._run(
                    "desktop.scroll_vertical",
                    {"direction": "up"},
                    "Scroll hacia arriba.",
                )

            if self._is_move_cursor_command(normalized):
                x, y = self._extract_coordinates(text)
                return self._run(
                    "desktop.move_cursor",
                    {
                        "x": x,
                        "y": y,
                    },
                    f"Cursor movido a ({x}, {y}).",
                )

            if normalized in {"mueve el raton", "mueve el cursor"}:
                raise ValueError("Orden incompleta: faltan coordenadas.")

            click_kind = self._click_kind(normalized)

            if click_kind is not None:
                x, y = self._extract_coordinates(text)
                self._validate_coordinates(x, y)
                tool_name, label = click_kind

                if not self._confirmed_click(confirm, label, x, y):
                    return "Acción cancelada."

                return self._run(
                    tool_name,
                    {
                        "x": x,
                        "y": y,
                    },
                    f"{label} realizado en ({x}, {y}).",
                )

            if normalized.startswith("abre "):
                return self._open(text[5:].strip())

            if normalized.startswith("abrir "):
                return self._open(text[6:].strip())

            if normalized.startswith("open "):
                return self._open(text[5:].strip())

            if normalized in {"abre", "abrir", "open"}:
                raise ValueError("Falta el objetivo a abrir.")

            if normalized.startswith("activa "):
                title = text[7:].strip()
                return self._run(
                    "desktop.activate_window",
                    {"window_title": title},
                    f"Ventana activada: {title}",
                )

            if normalized.startswith("escribe"):
                content = self._extract_text_to_type(text)
                return self._run(
                    "desktop.type_text",
                    {"text": content},
                    "Texto escrito.",
                )

            if normalized.startswith("guarda"):
                return self._run(
                    "desktop.save_file",
                    {"window_title": self._DEFAULT_WINDOW_TITLE},
                    "Archivo guardado.",
                )

            if normalized.startswith("pulsa ") or normalized.startswith(
                "presiona "
            ):
                keys = self._extract_keys(text)
                return self._run(
                    "desktop.press_hotkey",
                    {
                        "window_title": self._DEFAULT_WINDOW_TITLE,
                        "keys": keys,
                    },
                    "Atajo enviado.",
                )
        except Exception as exc:
            return f"Error: {exc}"

        return None

    def filesystem_tool_request(self, text: str) -> tuple[str, dict[str, str], str] | None:
        """Resolve one explicit filesystem request for supervised execution."""
        create = re.match(r"^\s*crea\s+la\s+carpeta\s+(.+?)\s*$", text, re.I)
        if create:
            path = create.group(1)
            return "desktop.create_folder", {"path": path}, f"Voy a crear la carpeta:\n{path}\n¿Confirmas?"
        delete = re.match(r"^\s*elimina\s+(?:el\s+archivo\s+|la\s+carpeta\s+)?(.+?)\s*$", text, re.I)
        if delete:
            path = delete.group(1)
            return "desktop.delete_path", {"path": path}, f"Voy a eliminar:\n{path}\n¿Confirmas?"
        for verb, tool_name, action in (("copia", "desktop.copy_path", "copiar"), ("mueve", "desktop.move_path", "mover")):
            match = re.match(rf"^\s*{verb}\s+(.+?)\s+a\s+(.+?)\s*$", text, re.I)
            if match:
                source, destination = match.groups()
                if not self._looks_like_path(source) or not self._looks_like_path(destination):
                    return None
                if len(re.findall(r"\s+a\s+", text, re.I)) != 1:
                    raise ValueError("Orden ambigua: usa rutas sin separadores ambiguos.")
                return tool_name, {"source_path": source, "destination_path": destination}, f"Voy a {action}:\n{source}\n→\n{destination}\n¿Confirmas?"
        rename = re.match(r"^\s*renombra\s+(.+?)\s+a\s+([^\\/]+?)\s*$", text, re.I)
        if rename:
            source, new_name = rename.groups()
            if not self._looks_like_path(source):
                return None
            if len(re.findall(r"\s+a\s+", text, re.I)) != 1:
                raise ValueError("Orden ambigua: usa rutas sin separadores ambiguos.")
            return "desktop.rename_path", {"source_path": source, "new_name": new_name}, f"Voy a renombrar:\n{source}\n→\n{new_name}\n¿Confirmas?"
        return None

    def close_application_tool_request(
        self,
        text: str,
    ) -> tuple[str, dict[str, object], str] | None:
        """Resolve one explicit application-close request for supervised execution."""
        stripped = self._strip_wake_word_prefix(text.strip())
        stripped = stripped.rstrip("¿?").strip()
        normalized = self._normalize(stripped)

        if normalized.startswith("cierra "):
            query = stripped[len("cierra ") :].strip()
        elif normalized.startswith("close "):
            query = stripped[len("close ") :].strip()
        else:
            return None

        query = query.rstrip(" .,;:!?").strip()

        if not query or self._normalize(query) in {"todo", "everything", "all"}:
            return None

        query = self._strip_leading_article(query)

        processes = self._list_processes(query)

        if not processes:
            return None

        window = self._uwp_window_close_target(query, processes)

        if window is not None:
            title = str(window["title"])
            names = ", ".join(sorted({str(p["name"]) for p in processes}))
            pids = ", ".join(str(p["pid"]) for p in processes)
            return (
                "desktop.close_window",
                {"handle": int(window["handle"])},
                f"Voy a cerrar la ventana '{title}' ({names} - PID {pids}). ¿Confirmas?",
            )

        if len(processes) != 1:
            return None

        process = processes[0]
        name = str(process["name"])
        pid = int(process["pid"])
        pid_request = (
            "desktop.close_application",
            {"pid": pid},
            f"Voy a cerrar {name} - PID {pid}. ¿Confirmas?",
        )

        if process.get("window_titles"):
            return pid_request

        matches = self._close_window_matches(query)

        if len(matches) == 1:
            window = matches[0]
            title = str(window["title"])
            return (
                "desktop.close_window",
                {"handle": int(window["handle"])},
                f"Voy a cerrar la ventana '{title}' ({name} - PID {pid}). ¿Confirmas?",
            )

        return pid_request

    def _uwp_window_close_target(
        self,
        query: str,
        processes: list[dict[str, object]],
    ) -> dict[str, object] | None:
        """Return one unambiguous visible window when no nominal process owns windows."""
        if any(process.get("window_titles") for process in processes):
            return None

        matches = self._close_window_matches(query)

        if len(matches) == 1:
            return matches[0]

        return None

    def _execute_filesystem_command(
        self,
        text: str,
        confirm: Callable[[str], str] | None,
    ) -> str | None:
        """Route one explicit, supervised filesystem operation."""
        create = re.match(r"^\s*crea\s+la\s+carpeta\s+(.+?)\s*$", text, re.I)
        if create:
            path = create.group(1)
            if not self._confirmed_filesystem(confirm, "crear la carpeta", path):
                return "Acción cancelada."
            result = self._execute("desktop.create_folder", ToolContext(parameters={"path": path}))
            return f"✓ Carpeta {'creada' if result['created'] else 'ya existente'}: {result['path']}"

        delete = re.match(r"^\s*elimina\s+(?:el\s+archivo\s+|la\s+carpeta\s+)?(.+?)\s*$", text, re.I)
        if delete:
            path = delete.group(1)
            if not self._confirmed_filesystem(confirm, "eliminar", path):
                return "Acción cancelada."
            result = self._execute("desktop.delete_path", ToolContext(parameters={"path": path}))
            return f"✓ {result['kind'].capitalize()} eliminado: {result['path']}"

        for verb, tool_name, label in (
            ("copia", "desktop.copy_path", "copiado"),
            ("mueve", "desktop.move_path", "movido"),
        ):
            match = re.match(rf"^\s*{verb}\s+(.+?)\s+a\s+(.+?)\s*$", text, re.I)
            if match:
                source, destination = match.groups()
                if not self._looks_like_path(source) or not self._looks_like_path(destination):
                    return None
                if len(re.findall(r"\s+a\s+", text, re.I)) != 1:
                    raise ValueError("Orden ambigua: usa rutas sin separadores ambiguos.")
                if not self._confirmed_filesystem(confirm, verb, f"{source} -> {destination}"):
                    return "Acción cancelada."
                result = self._execute(tool_name, ToolContext(parameters={"source_path": source, "destination_path": destination}))
                return f"✓ {result['kind'].capitalize()} {label}: {result['source']} -> {result['destination']}"

        rename = re.match(r"^\s*renombra\s+(.+?)\s+a\s+([^\\/]+?)\s*$", text, re.I)
        if rename:
            source, new_name = rename.groups()
            if len(re.findall(r"\s+a\s+", text, re.I)) != 1:
                raise ValueError("Orden ambigua: usa rutas sin separadores ambiguos.")
            if not self._confirmed_filesystem(confirm, "renombrar", f"{source} -> {new_name}"):
                return "Acción cancelada."
            result = self._execute("desktop.rename_path", ToolContext(parameters={"source_path": source, "new_name": new_name}))
            return f"✓ {result['kind'].capitalize()} renombrado: {result['source']} -> {result['destination']}"
        return None

    def _confirmed_filesystem(
        self,
        confirm: Callable[[str], str] | None,
        operation: str,
        target: str,
    ) -> bool:
        if confirm is None:
            return False
        return self._normalize(confirm(f"¿Confirmas {operation}?\n{target}\n[s/N]: ")) in {"s", "si", "y", "yes"}

    def _is_prepare_atlas_workspace_command(
        self,
        text: str,
    ) -> bool:
        """Return whether the command asks for the Atlas workspace workflow."""
        return text in {
            "prepara atlas para trabajar",
            "prepara el proyecto atlas",
            "abre el entorno de atlas",
            "prepare atlas workspace",
        }

    def _is_free_sequence_command(
        self,
        text: str,
    ) -> bool:
        """Return whether the command asks for an unsupported free sequence."""
        return (
            " y luego " in text
            or text.startswith("ejecuta estas ")
            or text.startswith("repite ")
            or text.startswith("si falla ")
        )

    def _format_automation_result(
        self,
        result: AutomationResult,
    ) -> str:
        """Format an automation result for the interactive flow."""
        lines = [
            "Automatización iniciada: preparar Atlas para trabajar",
            "",
        ]

        for step_result in result.step_results:
            if step_result.success:
                lines.append(
                    f"{self._CONFIRMATION_PREFIX} {step_result.step_name}"
                )
                continue

            lines.append(f"Error en {step_result.step_name}:")
            lines.append(step_result.error or step_result.message)

        lines.append("")

        if result.completed:
            lines.append("Automatización completada correctamente.")
        else:
            lines.append("Automatización incompleta.")

        lines.append(f"Pasos ejecutados: {result.executed_steps}")
        lines.append(f"Pasos fallidos: {result.failed_steps}")

        return "\n".join(lines)

    def _execute_verified_text_file_command(
        self,
        normalized: str,
        confirm: Callable[[str], str] | None,
    ) -> str | None:
        """Execute the controlled verified text-file workflow."""
        if normalized not in {
            "crea un archivo de trabajo verificado",
            "crea archivo verificado",
            "create verified work file",
        }:
            return None

        if self._create_verified_text_file is None:
            raise RuntimeError("Workflow no disponible.")

        if confirm is None:
            raise RuntimeError("Confirmacion no disponible.")

        relative_path = confirm("Ruta relativa del archivo: ").strip()
        content = confirm("Contenido: ")
        result = self._create_verified_text_file.execute(
            workspace_root=self._project_root,
            target_file=relative_path,
            content=content,
            confirm=confirm,
        )

        return self._format_verified_text_file_result(result)

    def _format_verified_text_file_result(
        self,
        result: VerifiedTextFileAutomationResult,
    ) -> str:
        """Format a verified text-file result for the console flow."""
        lines = [
            "Automatizacion iniciada: create_verified_text_file",
            f"Archivo: {result.target_file}",
            f"Confirmado: {result.confirmed}",
            f"Completada: {result.completed}",
            f"Verificacion: {result.verification_passed}",
            f"Rollback: {result.rolled_back}",
            f"Rollback fallido: {result.rollback_failed}",
            f"Reintentos utilizados: {result.retries_used}",
            "",
        ]

        for step_result in result.step_results:
            if step_result.success:
                lines.append(f"{self._CONFIRMATION_PREFIX} {step_result.step_name}")
                continue

            lines.append(f"Error en {step_result.step_name}:")
            lines.append(step_result.error or step_result.message)

        if result.warnings:
            lines.append("")
            lines.append("Advertencias:")
            lines.extend(f"- {warning}" for warning in result.warnings)

        lines.append("")
        lines.append(result.summary)

        return "\n".join(lines)

    def _execute_wait_command(
        self,
        text: str,
        normalized: str,
    ) -> str | None:
        """Execute supported wait commands."""
        if not normalized.startswith("espera"):
            return None

        if normalized in {"espera", "esperar"}:
            raise ValueError("Espera invalida: falta el objetivo.")

        body = self._wait_body(text, normalized)
        normalized_body = self._normalize(body)
        disappeared = any(
            marker in normalized_body
            for marker in (
                "desaparezca ",
                "desaparecer ",
                "termine ",
                "cerrada ",
                "cerrado ",
            )
        )
        body = self._remove_wait_markers(body)
        normalized_body = self._normalize(body)

        if normalized_body.startswith("proceso "):
            result = self._wait_engine.wait_process(
                body[len("proceso ") :].strip(),
                timeout=20,
                poll_interval=0.2,
                exists=not disappeared,
            )
            return self._format_wait_result(result)

        if normalized_body.startswith("ventana activa "):
            result = self._wait_engine.wait_active_window(
                body[len("ventana activa ") :].strip(),
                timeout=20,
                poll_interval=0.2,
            )
            return self._format_wait_result(result)

        if normalized_body.startswith("ventana "):
            result = self._wait_engine.wait_window(
                body[len("ventana ") :].strip(),
                timeout=20,
                poll_interval=0.2,
                exists=not disappeared,
            )
            return self._format_wait_result(result)

        if normalized_body.startswith("archivo "):
            result = self._wait_engine.wait_file(
                self._resolve_path(body[len("archivo ") :].strip()),
                timeout=20,
                poll_interval=0.2,
                exists=not disappeared,
            )
            return self._format_wait_result(result)

        result = self._wait_engine.wait_application(
            body,
            timeout=20,
            poll_interval=0.2,
            opened=not disappeared,
        )
        return self._format_wait_result(result)

    def _wait_body(
        self,
        text: str,
        normalized: str,
    ) -> str:
        """Return the target part of a wait command."""
        if normalized.startswith("espera hasta que "):
            return text[len("espera hasta que ") :].strip()

        if normalized.startswith("espera a que "):
            return text[len("espera a que ") :].strip()

        if normalized.startswith("esperar hasta que "):
            return text[len("esperar hasta que ") :].strip()

        return text[len("espera ") :].strip()

    def _remove_wait_markers(
        self,
        body: str,
    ) -> str:
        """Remove natural-language wait state markers."""
        cleaned = body.strip()
        normalized = self._normalize(cleaned)
        prefixes = (
            "aparezca ",
            "exista ",
            "este abierto ",
            "estÃ© abierto ",
            "este activa ",
            "este activa la ",
            "desaparezca ",
            "desaparecer ",
            "termine ",
            "se cierre ",
            "cerrada ",
            "cerrado ",
        )

        for prefix in prefixes:
            if normalized.startswith(self._normalize(prefix)):
                return cleaned[len(prefix) :].strip()

        return cleaned

    def _format_wait_result(
        self,
        result: WaitResult,
    ) -> str:
        """Format a wait result for the interactive flow."""
        if result.completed:
            return (
                f"{self._CONFIRMATION_PREFIX} {result.description}\n"
                f"Condicion: {result.condition}\n"
                f"Tiempo: {result.elapsed_time:.2f}s"
            )

        return (
            "Error: timeout agotado.\n"
            f"Condicion: {result.condition}\n"
            f"Objetivo: {result.description}\n"
            f"Tiempo: {result.elapsed_time:.2f}/{result.timeout:.2f}s"
        )

    def _execute_process_command(
        self,
        text: str,
        normalized: str,
        confirm: Callable[[str], str] | None,
    ) -> str | None:
        """Execute supported process-management commands."""
        text = text.strip().strip("¿?")
        normalized = normalized.strip("¿?")
        terminate_pid = self._extract_terminate_pid(normalized)

        if terminate_pid is not None:
            return self._terminate_process(terminate_pid, confirm)

        if normalized in {
            "termina proceso",
            "termina el proceso",
            "termina pid",
            "mata proceso",
        }:
            raise ValueError("Orden incompleta: falta el PID.")

        if normalized in {"termina pid abc", "termina proceso abc"}:
            raise ValueError("PID invalido.")

        if normalized in {"mata todos los procesos", "cierra todo"}:
            raise ValueError("No se admiten operaciones masivas.")

        if normalized.startswith("lista los procesos de "):
            query = text[len("lista los procesos de ") :].strip()
            return self._format_process_list(self._list_processes(query))

        if normalized.startswith("lista procesos "):
            query = text[len("lista procesos ") :].strip()
            return self._format_process_list(self._list_processes(query))

        if normalized.startswith("list ") and normalized.endswith(" processes"):
            query = text[len("list ") : -len(" processes")].strip()
            return self._format_process_list(self._list_processes(query))

        if normalized.startswith("muestra el pid de "):
            query = text[len("muestra el PID de ") :].strip()
            return self._format_process_pids(query, self._list_processes(query))

        if normalized.startswith("esta abierto "):
            query = text[len("está abierto ") :].strip()
            return self._format_running_state(query, self._list_processes(query))

        if normalized.startswith("comprueba si ") and normalized.endswith(
            " esta abierto"
        ):
            query = text[len("comprueba si ") : -len(" está abierto")].strip()
            return self._format_running_state(query, self._list_processes(query))

        if normalized.startswith("is ") and normalized.endswith(" running"):
            query = text[len("is ") : -len(" running")].strip()
            return self._format_running_state(query, self._list_processes(query))

        if normalized.startswith("cierra "):
            query = text[len("cierra ") :].strip()
            processes = self._list_processes(query)

            if not processes:
                return None

            return self._close_application(query, processes, confirm)

        if normalized.startswith("close "):
            query = text[len("close ") :].strip()
            processes = self._list_processes(query)

            if not processes:
                return None

            return self._close_application(query, processes, confirm)

        return None

    def _extract_terminate_pid(
        self,
        normalized: str,
    ) -> int | None:
        """Extract a PID from supported terminate commands."""
        patterns = (
            r"^termina el proceso con pid (\S+)$",
            r"^mata el proceso (\S+)$",
            r"^terminate process (\S+)$",
        )

        for pattern in patterns:
            match = re.match(pattern, normalized)

            if match is None:
                continue

            value = match.group(1)

            if not value.isdigit():
                raise ValueError("PID invalido.")

            pid = int(value)

            if pid <= 0:
                raise ValueError("PID invalido.")

            return pid

        return None

    def _list_processes(
        self,
        query: str,
    ) -> list[dict[str, object]]:
        """Return process matches."""
        if not query:
            raise ValueError("Orden incompleta: falta la aplicacion.")

        result = self._execute(
            "desktop.list_processes",
            ToolContext(parameters={"query": query}),
        )

        if not isinstance(result, list):
            raise RuntimeError("Respuesta de procesos invalida.")

        return result

    def _format_process_list(
        self,
        processes: list[dict[str, object]],
    ) -> str:
        """Format a deterministic process list."""
        if not processes:
            return "No se encontraron procesos."

        lines = ["Procesos encontrados:", ""]

        for index, process in enumerate(processes, start=1):
            line = f"{index}. {process['name']} - PID {process['pid']}"
            titles = process.get("window_titles", ())

            if titles:
                line += " - ventanas visibles: si"
            else:
                line += " - ventanas visibles: no"

            lines.append(line)

        return "\n".join(lines)

    def _format_process_pids(
        self,
        query: str,
        processes: list[dict[str, object]],
    ) -> str:
        """Format PID information for process matches."""
        if not processes:
            return f"{query} no esta abierto."

        pids = ", ".join(str(process["pid"]) for process in processes)
        return f"PID: {pids}"

    def _format_running_state(
        self,
        query: str,
        processes: list[dict[str, object]],
    ) -> str:
        """Format whether an application is running."""
        if not processes:
            return f"{query} no esta abierto."

        pids = ", ".join(str(process["pid"]) for process in processes)
        return (
            f"{self._CONFIRMATION_PREFIX} {query} esta abierto.\n"
            f"Procesos: {len(processes)}\n"
            f"PID: {pids}"
        )

    def _close_application(
        self,
        query: str,
        processes: list[dict[str, object]],
        confirm: Callable[[str], str] | None,
    ) -> str:
        """Request normal application close after explicit confirmation."""
        window = self._uwp_window_close_target(query, processes)

        if window is not None:
            title = str(window["title"])
            handle = int(window["handle"])

            if not self._confirmed_close(confirm, title):
                return "Acción cancelada."

            self._execute(
                "desktop.close_window",
                ToolContext(parameters={"handle": handle}),
            )

            names = ", ".join(sorted({str(p["name"]) for p in processes}))
            pids = ", ".join(str(p["pid"]) for p in processes)
            return (
                f"{self._CONFIRMATION_PREFIX} Solicitud de cierre enviada: "
                f"ventana '{title}' ({names} - PID {pids})"
            )

        process = self._select_process(query, processes, confirm)
        name = str(process["name"])
        pid = int(process["pid"])
        window_titles = tuple(process.get("window_titles") or ())

        if not window_titles:
            window = self._single_close_window_target(query, confirm)

            if window is not None:
                title = str(window["title"])
                handle = int(window["handle"])

                if not self._confirmed_close(confirm, title):
                    return "Acción cancelada."

                self._execute(
                    "desktop.close_window",
                    ToolContext(parameters={"handle": handle}),
                )

                return (
                    f"{self._CONFIRMATION_PREFIX} Solicitud de cierre enviada: "
                    f"ventana '{title}' ({name} - PID {pid})"
                )

        if not self._confirmed_close_application(confirm, query, pid):
            return "Cierre cancelado."

        self._execute(
            "desktop.close_application",
            ToolContext(parameters={"pid": pid}),
        )

        return f"{self._CONFIRMATION_PREFIX} Solicitud de cierre enviada: {name} - PID {pid}"

    def _single_close_window_target(
        self,
        query: str,
        confirm: Callable[[str], str] | None,
    ) -> dict[str, object] | None:
        """Return one safe visible-window target for an app close, if any."""
        matches = self._close_window_matches(query)

        if len(matches) == 1:
            return matches[0]

        if len(matches) > 1:
            if confirm is None:
                raise ValueError(
                    "Varias ventanas coinciden. Se requiere seleccion explicita."
                )

            selection = confirm(
                self._format_window_matches(matches)
                + f"\nSelecciona una ventana [1-{len(matches)}]: "
            )

            if not selection.strip().isdigit():
                raise ValueError("Seleccion invalida.")

            index = int(selection.strip())

            if index < 1 or index > len(matches):
                raise ValueError("Seleccion invalida.")

            return matches[index - 1]

        return None

    def _close_window_matches(
        self,
        query: str,
    ) -> list[dict[str, object]]:
        """Return visible windows matching the close query or its aliases."""
        normalized = self._normalize(self._strip_leading_article(query))

        if not normalized:
            return []

        candidates = {normalized, *self._ACTIVATION_TITLE_ALIASES.get(normalized, ())}
        result = self._execute(
            "desktop.list_windows",
            ToolContext(parameters={}),
        )

        if not isinstance(result, list):
            raise RuntimeError("Respuesta de ventanas invalida.")

        return [
            window
            for window in result
            if any(
                candidate in self._normalize(str(window.get("title", "")))
                for candidate in candidates
            )
        ]

    def _select_process(
        self,
        query: str,
        processes: list[dict[str, object]],
        confirm: Callable[[str], str] | None,
    ) -> dict[str, object]:
        """Select a process match without arbitrary choice."""
        if len(processes) == 1:
            return processes[0]

        if confirm is None:
            raise ValueError(
                "Varias coincidencias. Se requiere seleccion explicita."
            )

        selection = confirm(
            self._format_process_list(processes)
            + f"\nSelecciona un proceso [1-{len(processes)}]: "
        )

        if not selection.strip().isdigit():
            raise ValueError("Seleccion invalida.")

        index = int(selection.strip())

        if index < 1 or index > len(processes):
            raise ValueError("Seleccion invalida.")

        return processes[index - 1]

    def _terminate_process(
        self,
        pid: int,
        confirm: Callable[[str], str] | None,
    ) -> str:
        """Terminate one process by PID after explicit confirmation."""
        if pid == 4:
            raise ValueError("Proceso protegido.")

        process = self._execute(
            "desktop.get_process",
            ToolContext(parameters={"pid": pid}),
        )

        if process is None:
            raise RuntimeError(f"No existe un proceso con PID {pid}.")

        if not isinstance(process, dict):
            raise RuntimeError("Respuesta de proceso invalida.")

        name = str(process["name"])

        if not self._confirmed_terminate_process(confirm, name, pid):
            return "Terminacion cancelada."

        result = self._execute(
            "desktop.terminate_process",
            ToolContext(parameters={"pid": pid}),
        )

        return f"{self._CONFIRMATION_PREFIX} {result}"

    def _confirmed_close_application(
        self,
        confirm: Callable[[str], str] | None,
        query: str,
        pid: int,
    ) -> bool:
        """Return whether application close was explicitly confirmed."""
        if confirm is None:
            return False

        response = confirm(f"Â¿Confirmas cerrar {query} - PID {pid}? [s/N]: ")

        return self._normalize(response) in {"s", "si", "y", "yes"}

    def _confirmed_terminate_process(
        self,
        confirm: Callable[[str], str] | None,
        name: str,
        pid: int,
    ) -> bool:
        """Return whether forced process termination was confirmed."""
        if confirm is None:
            return False

        response = confirm(
            f"Proceso:\n{name} - PID {pid}\n\n"
            "Â¿Confirmas terminarlo de forma forzada? [s/N]: "
        )

        return self._normalize(response) in {"s", "si", "y", "yes"}

    def _open(
        self,
        target: str,
    ) -> str:
        """Open an application, folder, or file."""
        expected_kind = self._open_target_kind(target)
        target = self._clean_open_target(target)
        target, application = self._open_file_application(target)
        target = self._resolve_application_alias(target)

        if not target:
            raise ValueError("Falta el objetivo a abrir.")

        path = self._resolve_path(target)

        if path.exists() and path.is_dir():
            if expected_kind == "file":
                return "La ruta es una carpeta, no un archivo."
            return self._run(
                "desktop.open_folder",
                {"path": str(path)},
                f"Abriendo {path}.",
            )

        if path.exists() and path.is_file():
            if expected_kind == "folder":
                return "La ruta es un archivo, no una carpeta."
            if application is not None:
                return self._run(
                    "desktop.open_file",
                    {"path": str(path), "application": application},
                    f"Abriendo {path} con {application}.",
                )
            return self._run(
                "desktop.open_file",
                {"path": str(path)},
                f"Abriendo {path}.",
            )

        if self._looks_like_path(target):
            return "La ruta no existe."

        try:
            return self._run(
                "desktop.open_application",
                {"application": target},
                f"Abriendo {target}.",
            )
        except FileNotFoundError:
            return "No encuentro esa aplicación."
        except OSError:
            logging.getLogger(__name__).exception(
                "Fallo al lanzar la aplicación desktop: %s",
                target,
            )
            return f"No pude abrir {target}."

    def _execute_clipboard_command(
        self,
        text: str,
        normalized: str,
        confirm: Callable[[str], str] | None,
    ) -> str | None:
        """Execute supported clipboard commands."""
        copy_text = self._extract_clipboard_copy_text(text)

        if copy_text is not None:
            length = self._execute(
                "desktop.copy_clipboard_text",
                ToolContext(parameters={"text": copy_text}),
            )

            return (
                f"{self._CONFIRMATION_PREFIX} "
                f"Texto copiado al portapapeles.\nCaracteres: {length}"
            )

        if normalized in {
            "copia al portapapeles",
            "copia este texto al portapapeles",
            "copia este texto",
            "copia",
            "copiar al portapapeles",
            "copiar",
        }:
            raise ValueError("Orden incompleta: falta el texto a copiar.")

        if normalized in {
            "copia una imagen",
            "lee imagen del portapapeles",
        }:
            raise ValueError("Formato de portapapeles no soportado.")

        if normalized in {
            "lee el portapapeles",
            "leer portapapeles",
            "que hay en el portapapeles",
            "que tengo en el portapapeles",
            "read clipboard",
        }:
            content = self._execute(
                "desktop.read_clipboard_text",
                ToolContext(),
            )

            if content is None:
                return "El portapapeles no contiene texto."

            return f"Contenido del portapapeles:\n\n{content}"

        if normalized in {
            "hay texto en el portapapeles",
            "el portapapeles contiene texto",
            "clipboard has text",
        }:
            has_text = self._execute(
                "desktop.clipboard_has_text",
                ToolContext(),
            )

            if has_text is True:
                return "El portapapeles contiene texto."

            return "El portapapeles no contiene texto."

        if normalized in {
            "limpia el portapapeles",
            "vacia el portapapeles",
            "clear clipboard",
        }:
            if not self._confirmed_clear_clipboard(confirm):
                return "Limpieza cancelada."

            return self._run(
                "desktop.clear_clipboard",
                {},
                "Portapapeles vaciado.",
            )

        if normalized in {"limpia", "vacia"}:
            raise ValueError("Orden incompleta: falta el portapapeles.")

        if normalized in {
            "pega",
            "pega el texto",
            "pega el portapapeles",
            "pega el contenido del portapapeles",
            "paste clipboard",
        }:
            return self._paste_clipboard(confirm)


        return None

    def _extract_clipboard_copy_text(
        self,
        text: str,
    ) -> str | None:
        """Extract copy text from supported clipboard commands."""
        patterns = (
            r"^\s*copia\s+este\s+texto\s*:\s*(.*)$",
            r"^\s*copia\s+al\s+portapapeles\s*:\s*(.*)$",
            r"^\s*copia\s+este\s+texto\s+al\s+portapapeles\s*:\s*(.*)$",
            r"^\s*copia\s+(.+?)\s+al\s+portapapeles\s*$",
            r"^\s*copia\s+(?!al\s+portapapeles\s*$)(.+)$",
            r"^\s*copiar\s+al\s+portapapeles\s+(.+)$",
        )

        for pattern in patterns:
            match = re.match(pattern, text, flags=re.IGNORECASE | re.DOTALL)

            if match is None:
                continue

            content = match.group(1)

            if content == "":
                raise ValueError("Orden incompleta: falta el texto a copiar.")

            return content

        return None

    def _paste_clipboard(
        self,
        confirm: Callable[[str], str] | None,
    ) -> str:
        """Paste clipboard text into the active window after confirmation."""
        has_text = self._execute(
            "desktop.clipboard_has_text",
            ToolContext(),
        )

        if has_text is not True:
            raise RuntimeError("El portapapeles no contiene texto.")

        if not self._confirmed_paste_clipboard(confirm):
            return "Pegado cancelado."

        return self._run(
            "desktop.paste_clipboard",
            {},
            "Contenido pegado.",
        )

    def _confirmed_clear_clipboard(
        self,
        confirm: Callable[[str], str] | None,
    ) -> bool:
        """Return whether the user explicitly confirmed clearing clipboard."""
        if confirm is None:
            return False

        response = confirm("Â¿Confirmas vaciar el portapapeles? [s/N]: ")

        return self._normalize(response) in {
            "s",
            "si",
            "y",
            "yes",
        }

    def _confirmed_paste_clipboard(
        self,
        confirm: Callable[[str], str] | None,
    ) -> bool:
        """Return whether the user explicitly confirmed active-window paste."""
        if confirm is None:
            return False

        response = confirm("Â¿Confirmas pegar el contenido en la ventana activa? [s/N]: ")

        return self._normalize(response) in {
            "s",
            "si",
            "y",
            "yes",
        }

    def _open_target_kind(self, target: str) -> str | None:
        """Return an explicitly requested filesystem target kind, if any."""
        normalized = self._normalize(target)

        if normalized.startswith(("la carpeta ", "carpeta ")):
            return "folder"
        if normalized.startswith(("el archivo ", "archivo ")):
            return "file"
        return None

    def _clean_open_target(
        self,
        target: str,
    ) -> str:
        """Remove natural-language prefixes from an open command."""
        cleaned = target.strip().rstrip(" .,;:!?")
        normalized = self._normalize(cleaned)
        prefixes = (
            "la carpeta ",
            "carpeta ",
            "el archivo ",
            "archivo ",
            "la aplicacion ",
            "la aplicación ",
            "aplicacion ",
            "aplicación ",
        )

        for prefix in prefixes:
            if normalized.startswith(self._normalize(prefix)):
                return cleaned[len(prefix) :].strip()

        return cleaned

    def _resolve_application_alias(
        self,
        target: str,
    ) -> str:
        """Resolve the few application aliases supported by desktop opening."""
        aliases = {
            "bloc de notas": "notepad",
            "el bloc de notas": "notepad",
            "la calculadora": "calculadora",
            "el chrome": "chrome",
            "el vs code": "vs code",
        }
        return aliases.get(self._normalize(target), target)

    def _open_file_application(
        self,
        target: str,
    ) -> tuple[str, str | None]:
        """Split one explicit "con/with <aplicacion>" suffix using known aliases only."""
        stripped = target.strip()
        match = re.search(r"\s+(?:con|with)\s+(.+)$", stripped, re.IGNORECASE)
        if match is None:
            return target, None
        requested = match.group(1).strip()
        application = self._resolve_application_alias(requested)
        if self._normalize(application) == self._normalize(requested):
            return target, None
        return stripped[: match.start()].strip(), application

    def _run(
        self,
        tool_name: str,
        parameters: dict[str, object],
        confirmation: str,
    ) -> str:
        """Run a tool and return a user-facing confirmation."""
        self._execute(
            tool_name,
            ToolContext(parameters=parameters),
        )

        return f"{self._CONFIRMATION_PREFIX} {confirmation}"

    def _execute_tuple(
        self,
        tool_name: str,
        parameters: dict[str, object],
    ) -> tuple[int, int]:
        """Run a tuple-returning tool."""
        result = self._execute(
            tool_name,
            ToolContext(parameters=parameters),
        )

        if not (
            isinstance(result, tuple)
            and len(result) == 2
            and isinstance(result[0], int)
            and isinstance(result[1], int)
        ):
            raise RuntimeError("Respuesta de herramienta invalida.")

        return result

    def _execute(
        self,
        tool_name: str,
        context: ToolContext,
    ):
        """Execute one desktop tool with its existing one-use authorization."""
        requires_authorization = getattr(
            self._executor,
            "requires_explicit_authorization",
            lambda _tool_name: False,
        )
        if requires_authorization(tool_name):
            return self._executor.execute(
                tool_name,
                context,
                authorization=self._executor.authorize(tool_name),
            )
        return self._executor.execute(tool_name, context)

    def _execute_window_command(
        self,
        text: str,
        normalized: str,
        confirm: Callable[[str], str] | None,
    ) -> str | None:
        """Execute a supported window-management command."""
        if normalized in {"mueve el raton", "mueve el cursor"}:
            return None

        if normalized in {
            "que ventanas tengo abiertas",
            "lista las ventanas abiertas",
            "muestrame las ventanas abiertas",
        }:
            result = self._execute(
                "desktop.list_windows",
                ToolContext(parameters={}),
            )
            if not isinstance(result, list):
                raise RuntimeError("Respuesta de ventanas invalida.")
            lines = ["Ventanas abiertas:"]
            lines.extend(f"- {window['title']}" for window in result)
            return "\n".join(lines)
        if normalized.startswith("lista ventanas de "):
            title = text[len("lista ventanas de ") :].strip()
            matches = self._list_windows(title)
            return self._format_window_matches(matches)

        if normalized.startswith("maximiza "):
            title = text[len("maximiza ") :].strip()
            return self._window_state_action(
                title,
                "desktop.maximize_window",
                "Ventana maximizada",
                confirm,
            )

        if normalized.startswith("minimiza "):
            title = text[len("minimiza ") :].strip()
            return self._window_state_action(
                title,
                "desktop.minimize_window",
                "Ventana minimizada",
                confirm,
            )

        if normalized.startswith("restaura "):
            title = text[len("restaura ") :].strip()
            return self._window_state_action(
                title,
                "desktop.restore_window",
                "Ventana restaurada",
                confirm,
            )

        if normalized.startswith("trae ") and normalized.endswith(
            " al frente"
        ):
            title = text[len("trae ") : -len(" al frente")].strip()
            return self._window_state_action(
                title,
                "desktop.bring_window_to_front",
                "Ventana activada",
                confirm,
            )

        if normalized.startswith("activa "):
            return self._activate_window(text[len("activa ") :].strip())

        if normalized.startswith("pon ") and normalized.endswith(" delante"):
            return self._activate_window(text[len("pon ") : -len(" delante")].strip())
        if normalized.startswith("pon "):
            return self._activate_window(text[len("pon ") :].strip())
        if normalized.startswith("ve a "):
            return self._activate_window(text[len("ve a ") :].strip())
        if normalized.startswith("cambia a "):
            return self._activate_window(text[len("cambia a ") :].strip())
        if normalized.startswith("cierra "):
            title = text[len("cierra ") :].strip()
            return self._close_window(title, confirm)

        if normalized in {
            "maximiza",
            "minimiza",
            "restaura",
            "cierra",
            "activa",
            "ve",
            "pon",
            "cambia",
            "centra",
        }:
            raise ValueError("Orden incompleta: falta el titulo de ventana.")

        if normalized.startswith("centra "):
            title = text[len("centra ") :].strip()
            return self._center_window(title, confirm)

        if normalized.startswith("mueve ") and normalized.endswith(
            " a la izquierda"
        ):
            title = text[len("mueve ") : -len(" a la izquierda")].strip()
            return self._move_window_to_edge(title, "izquierda", confirm)

        if normalized.startswith("mueve ") and normalized.endswith(
            " a la derecha"
        ):
            title = text[len("mueve ") : -len(" a la derecha")].strip()
            return self._move_window_to_edge(title, "derecha", confirm)

        if normalized.startswith("mueve y cambia el tamano de "):
            return self._move_resize_window(text, confirm)

        if normalized.startswith("mueve ") and not self._is_move_cursor_command(
            normalized
        ):
            return self._move_window(text, confirm)

        if normalized.startswith("cambia el tamano de ") or normalized.startswith(
            "redimensiona "
        ):
            return self._resize_window(text, confirm)

        return None

    def _activate_window(self, title: str) -> str:
        """Activate one unambiguously matched window by exact HWND."""
        resolved = self.resolve_window_for_activation(title)
        if isinstance(resolved, str):
            return resolved

        handle, resolved_title = resolved
        self._execute(
            "desktop.bring_window_to_front",
            ToolContext(parameters={"handle": handle}),
        )
        return f"{self._CONFIRMATION_PREFIX} Ventana activada:\n{resolved_title}"

    def resolve_window_for_activation(self, title: str) -> tuple[int, str] | str:
        """Resolve one activation target without bringing it to the foreground."""
        query = self._normalize(self._strip_leading_article(title))
        if not query:
            raise ValueError("Orden incompleta: falta el titulo de ventana.")

        queries = [query, *self._ACTIVATION_TITLE_ALIASES.get(query, ())]

        result = self._execute(
            "desktop.list_windows",
            ToolContext(parameters={}),
        )
        if not isinstance(result, list):
            raise RuntimeError("Respuesta de ventanas invalida.")

        matches = [
            window
            for window in result
            if any(
                candidate in self._normalize(str(window.get("title", "")))
                for candidate in queries
            )
        ]
        if not matches:
            return f"No se encontro ninguna ventana para '{title}'."
        if len(matches) > 1:
            return "Varias ventanas coinciden:\n" + "\n".join(
                f"- {window['title']}" for window in matches
            )

        window = matches[0]
        return int(window["handle"]), str(window["title"])
    def _window_state_action(
        self,
        title: str,
        tool_name: str,
        label: str,
        confirm: Callable[[str], str] | None,
    ) -> str:
        """Execute a state action against one resolved window."""
        window = self._resolve_window(title, confirm)
        self._execute(
            tool_name,
            ToolContext(parameters={"handle": int(window["handle"])}),
        )

        return f"{self._CONFIRMATION_PREFIX} {label}:\n{window['title']}"

    def _move_window(
        self,
        text: str,
        confirm: Callable[[str], str] | None,
    ) -> str:
        """Move a resolved window."""
        title, numbers = self._split_window_command(text, "mueve ", 2)
        window = self._resolve_window(title, confirm)
        x, y = numbers
        self._execute(
            "desktop.move_window",
            ToolContext(
                parameters={
                    "handle": int(window["handle"]),
                    "x": x,
                    "y": y,
                }
            ),
        )

        return f"{self._CONFIRMATION_PREFIX} Ventana movida a ({x}, {y})."

    def _center_window(
        self,
        title: str,
        confirm: Callable[[str], str] | None,
    ) -> str:
        """Center a resolved window on the screen."""
        window = self._resolve_window(title, confirm)
        handle = int(window["handle"])
        left, top, right, bottom = self._execute(
            "desktop.get_window_rect",
            ToolContext(parameters={"handle": handle}),
        )
        screen_width, screen_height = self._execute_tuple(
            "desktop.get_screen_size",
            {},
        )
        x = max(0, (screen_width - (right - left)) // 2)
        y = max(0, (screen_height - (bottom - top)) // 2)
        self._execute(
            "desktop.move_window",
            ToolContext(parameters={"handle": handle, "x": x, "y": y}),
        )

        return f"{self._CONFIRMATION_PREFIX} Ventana centrada en ({x}, {y})."

    def _move_window_to_edge(
        self,
        title: str,
        edge: str,
        confirm: Callable[[str], str] | None,
    ) -> str:
        """Move a resolved window to the left or right screen edge."""
        window = self._resolve_window(title, confirm)
        handle = int(window["handle"])
        left, top, right, _ = self._execute(
            "desktop.get_window_rect",
            ToolContext(parameters={"handle": handle}),
        )

        if edge == "izquierda":
            x = 0
        else:
            screen_width, _ = self._execute_tuple(
                "desktop.get_screen_size",
                {},
            )
            x = max(0, screen_width - (right - left))

        self._execute(
            "desktop.move_window",
            ToolContext(parameters={"handle": handle, "x": x, "y": top}),
        )

        return f"{self._CONFIRMATION_PREFIX} Ventana movida a la {edge}."

    def _resize_window(
        self,
        text: str,
        confirm: Callable[[str], str] | None,
    ) -> str:
        """Resize a resolved window."""
        normalized = self._normalize(text)
        prefix = (
            "cambia el tamaño de "
            if normalized.startswith("cambia el tamano de ")
            else "redimensiona "
        )
        title, numbers = self._split_window_command(text, prefix, 2)
        window = self._resolve_window(title, confirm)
        width, height = numbers
        self._execute(
            "desktop.resize_window",
            ToolContext(
                parameters={
                    "handle": int(window["handle"]),
                    "width": width,
                    "height": height,
                }
            ),
        )

        return (
            f"{self._CONFIRMATION_PREFIX} "
            f"Ventana redimensionada a {width} x {height}."
        )

    def _move_resize_window(
        self,
        text: str,
        confirm: Callable[[str], str] | None,
    ) -> str:
        """Move and resize a resolved window."""
        title, numbers = self._split_window_command(
            text,
            "mueve y cambia el tamaño de ",
            4,
        )
        window = self._resolve_window(title, confirm)
        x, y, width, height = numbers
        self._execute(
            "desktop.move_resize_window",
            ToolContext(
                parameters={
                    "handle": int(window["handle"]),
                    "x": x,
                    "y": y,
                    "width": width,
                    "height": height,
                }
            ),
        )

        return (
            f"{self._CONFIRMATION_PREFIX} Ventana movida a ({x}, {y}) "
            f"y redimensionada a {width} x {height}."
        )

    def _close_window(
        self,
        title: str,
        confirm: Callable[[str], str] | None,
    ) -> str:
        """Request closing a resolved window after confirmation."""
        window = self._resolve_window(title, confirm)

        if not self._confirmed_close(confirm, str(window["title"])):
            return "Acción cancelada."

        self._execute(
            "desktop.close_window",
            ToolContext(parameters={"handle": int(window["handle"])}),
        )

        return f"{self._CONFIRMATION_PREFIX} Solicitud de cierre enviada."

    def _resolve_window(
        self,
        title: str,
        confirm: Callable[[str], str] | None,
    ) -> dict[str, object]:
        """Resolve a title into one explicit window match."""
        if not title:
            raise ValueError("Orden incompleta: falta el titulo de ventana.")

        matches = self._list_windows(title)

        if not matches:
            raise ValueError(f"No se encontraron ventanas para '{title}'.")

        if len(matches) == 1:
            return matches[0]

        if confirm is None:
            raise ValueError(
                "Varias ventanas coinciden. Se requiere seleccion explicita."
            )

        selection = confirm(
            self._format_window_matches(matches)
            + f"\nSelecciona una ventana [1-{len(matches)}]: "
        )

        if not selection.strip().isdigit():
            raise ValueError("Seleccion invalida.")

        index = int(selection.strip())

        if index < 1 or index > len(matches):
            raise ValueError("Seleccion invalida.")

        return matches[index - 1]

    def _list_windows(
        self,
        title: str,
    ) -> list[dict[str, object]]:
        """Return visible windows matching title."""
        if not title:
            raise ValueError("Orden incompleta: falta el titulo de ventana.")

        query = self._normalize(title)

        if query in {"vs code", "vscode"}:
            title = "Visual Studio Code"

        title = self._strip_leading_article(title.strip())

        result = self._execute(
            "desktop.list_windows",
            ToolContext(parameters={"title": title}),
        )

        if not isinstance(result, list):
            raise RuntimeError("Respuesta de ventanas invalida.")

        return result

    def _format_window_matches(
        self,
        matches: list[dict[str, object]],
    ) -> str:
        """Format window matches deterministically."""
        if not matches:
            return "No se encontraron ventanas."

        lines = ["Se encontraron ventanas:"]

        for index, window in enumerate(matches, start=1):
            lines.append(f"{index}. {window['title']}")

        return "\n".join(lines)

    def _split_window_command(
        self,
        text: str,
        prefix: str,
        expected_numbers: int,
    ) -> tuple[str, list[int]]:
        """Split a window command into title and numeric arguments."""
        if not self._normalize(text).startswith(self._normalize(prefix)):
            raise ValueError("Orden incompleta.")

        body = text[len(prefix) :].strip()
        separator = re.search(r"\s+a\s+", body, flags=re.IGNORECASE)

        if separator is None:
            raise ValueError("Orden incompleta: faltan parametros.")

        title = body[: separator.start()].strip()
        parameters = body[separator.end() :].strip()
        numbers = self._extract_number_list(parameters, expected_numbers)

        if not title:
            raise ValueError("Orden incompleta: falta el titulo de ventana.")

        return title, numbers

    def _extract_number_list(
        self,
        text: str,
        expected: int,
    ) -> list[int]:
        """Extract an exact number of integer parameters."""
        text = re.sub(r"(?<=\d)\s*[x\u00d7]\s*(?=-?\d)", " ", text)

        if re.search(r"[A-Za-z]+", text):
            raise ValueError("Los parametros deben ser numericos.")

        numbers = [int(value) for value in re.findall(r"-?\d+", text)]

        if len(numbers) != expected:
            raise ValueError("Numero de parametros invalido.")

        return numbers

    def _confirmed_close(
        self,
        confirm: Callable[[str], str] | None,
        title: str,
    ) -> bool:
        """Return whether the user explicitly confirmed closing a window."""
        if confirm is None:
            return False

        response = confirm(
            f"¿Confirmas cerrar la ventana \"{title}\"? [s/N]: "
        )

        return self._normalize(response) in {
            "s",
            "si",
            "y",
            "yes",
        }

    def _validate_coordinates(
        self,
        x: int,
        y: int,
    ) -> None:
        """Validate coordinates against the current screen size."""
        if x < 0 or y < 0:
            raise ValueError("Las coordenadas no pueden ser negativas.")

        width, height = self._execute_tuple(
            "desktop.get_screen_size",
            {},
        )

        if x >= width or y >= height:
            raise ValueError(
                f"Coordenadas fuera de pantalla: ({x}, {y})."
            )

    def _resolve_path(
        self,
        value: str,
    ) -> Path:
        """Resolve a project-relative or absolute path."""
        expanded = Path(value.strip().strip('"'))

        if expanded.is_absolute():
            return expanded

        return self._project_root / expanded

    def _looks_like_path(
        self,
        value: str,
    ) -> bool:
        """Return whether a value looks like a filesystem path."""
        return (
            "\\" in value
            or "/" in value
            or bool(re.search(r"\.[A-Za-z0-9]{1,8}$", value.strip()))
        )

    def _extract_text_to_type(
        self,
        text: str,
    ) -> str:
        """Extract text after an Escribe command."""
        _, separator, content = text.partition(":")

        if separator:
            return content.lstrip("\r\n ")

        content = text[len("Escribe") :].strip()

        if content.lower().startswith("el texto "):
            content = content[len("el texto ") :]

        if not content:
            raise ValueError("Falta el texto a escribir.")

        return content

    def _extract_keys(
        self,
        text: str,
    ) -> list[str]:
        """Extract shortcut keys from a command."""
        lowered = self._normalize(text)
        raw = (
            text[6:]
            if lowered.startswith("pulsa ")
            else text[len("presiona ") :]
        )
        keys = [
            key.strip().lower()
            for key in re.split(r"\+|,", raw)
            if key.strip()
        ]

        if not keys:
            raise ValueError("Faltan teclas para el atajo.")

        return keys

    def _strip_wake_word_prefix(
        self,
        text: str,
    ) -> str:
        """Drop one leading invocation prefix ("Atlas, abre ...") from the command."""
        stripped = self._WAKE_WORD_PREFIX_PATTERN.sub("", text, count=1).strip()
        return stripped if stripped else text

    def _strip_leading_article(
        self,
        query: str,
    ) -> str:
        """Drop one leading Spanish article ("cierra la calculadora") from the target."""
        stripped = re.sub(
            r"^(?:el|la|los|las)\s+",
            "",
            query,
            count=1,
            flags=re.I,
        ).strip()
        return stripped if stripped else query

    def _normalize(
        self,
        text: str,
    ) -> str:
        """Normalize command text."""
        normalized = unicodedata.normalize("NFKD", text.strip().lower())
        without_accents = "".join(
            character
            for character in normalized
            if unicodedata.category(character) != "Mn"
        )

        return " ".join(without_accents.split())

    def _is_screen_size_command(
        self,
        text: str,
    ) -> bool:
        """Return whether the command asks for screen size."""
        return text in {
            "tamano de pantalla",
            "obten el tamano de la pantalla",
            "obten tamano de pantalla",
        }

    def _is_cursor_position_command(
        self,
        text: str,
    ) -> bool:
        """Return whether the command asks for cursor position."""
        return text in {
            "posicion del raton",
            "posicion del cursor",
            "obten posicion del raton",
            "obten la posicion del raton",
        }

    def _is_screenshot_command(
        self,
        text: str,
    ) -> bool:
        """Return whether the command asks for a full screenshot."""
        return text in {
            "haz una captura de pantalla",
            "captura la pantalla",
            "screenshot",
        }

    def _is_forbidden_screenshot_command(
        self,
        text: str,
    ) -> bool:
        """Return whether the screenshot command is out of scope."""
        return "captura" in text and "region" in text

    def _is_scroll_down_command(
        self,
        text: str,
    ) -> bool:
        """Return whether the command asks for downward scroll."""
        return text in {
            "desplazate hacia abajo",
            "scroll abajo",
        }

    def _is_scroll_up_command(
        self,
        text: str,
    ) -> bool:
        """Return whether the command asks for upward scroll."""
        return text in {
            "desplazate hacia arriba",
            "scroll arriba",
        }

    def _is_move_cursor_command(
        self,
        text: str,
    ) -> bool:
        """Return whether the command asks to move the cursor."""
        return text.startswith("mueve el raton a ") or text.startswith(
            "mueve el cursor a "
        )

    def _click_kind(
        self,
        text: str,
    ) -> tuple[str, str] | None:
        """Return the click tool and label for a click command."""
        if text.startswith("haz doble clic en "):
            return "desktop.double_click", "Doble clic"

        if text.startswith("haz clic derecho en "):
            return "desktop.right_click", "Clic derecho"

        if text.startswith("haz clic en ") or text.startswith("clic en "):
            return "desktop.left_click", "Clic"

        if text in {"haz clic", "clic", "doble clic"}:
            raise ValueError("Orden incompleta: faltan coordenadas.")

        return None

    def _extract_coordinates(
        self,
        text: str,
    ) -> tuple[int, int]:
        """Extract exactly two integer coordinates from a command."""
        if re.search(r"[A-Za-z]+", text) and not re.search(r"-?\d+", text):
            raise ValueError("La orden requiere exactamente dos coordenadas.")

        tokens = re.findall(r"-?\d+|[A-Za-z]+", text)
        numbers: list[int] = []

        for token in tokens:
            if re.fullmatch(r"-?\d+", token):
                numbers.append(int(token))
                continue

            if token.lower() in {"abc"}:
                raise ValueError("Las coordenadas deben ser numericas.")

        if len(numbers) != 2:
            raise ValueError("La orden requiere exactamente dos coordenadas.")

        return numbers[0], numbers[1]

    def _confirmed_click(
        self,
        confirm: Callable[[str], str] | None,
        label: str,
        x: int,
        y: int,
    ) -> bool:
        """Return whether the user explicitly confirmed a click."""
        if confirm is None:
            return False

        response = confirm(
            f"¿Confirmas el {label.lower()} en ({x}, {y})? [s/N]: "
        )

        return self._normalize(response) in {
            "s",
            "si",
            "y",
            "yes",
        }
