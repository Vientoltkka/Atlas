"""Desktop interaction use case."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Callable
import unicodedata

from use_cases.action_engine import (
    AutomationResult,
    PrepareAtlasWorkspaceUseCase,
)
from tools.executor import ToolExecutor
from tools.tool_context import ToolContext


class DesktopInteractionUseCase:
    """Execute simple desktop commands through Atlas tools."""

    _DEFAULT_EDITOR = "Visual Studio Code"
    _DEFAULT_WINDOW_TITLE = "Visual Studio Code"
    _CONFIRMATION_PREFIX = "\u2713"

    def __init__(
        self,
        executor: ToolExecutor,
        project_root: Path | None = None,
        screenshots_dir: Path | None = None,
        prepare_atlas_workspace: PrepareAtlasWorkspaceUseCase | None = None,
    ) -> None:
        self._executor = executor
        self._project_root = project_root or Path(".")
        self._screenshots_dir = (
            screenshots_dir
            if screenshots_dir is not None
            else self._project_root / "artifacts" / "screenshots"
        )
        self._prepare_atlas_workspace = prepare_atlas_workspace

    def execute(
        self,
        prompt: str,
        confirm: Callable[[str], str] | None = None,
    ) -> str | None:
        """Execute a supported desktop command."""
        text = prompt.strip()
        normalized = self._normalize(text)

        try:
            if self._is_free_sequence_command(normalized):
                raise ValueError("No se admiten secuencias libres.")

            if self._is_prepare_atlas_workspace_command(normalized):
                if self._prepare_atlas_workspace is None:
                    raise RuntimeError("Workflow no disponible.")

                result = self._prepare_atlas_workspace.execute(
                    self._project_root,
                )

                return self._format_automation_result(result)

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
                path = self._executor.execute(
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
                    {
                        "window_title": self._DEFAULT_WINDOW_TITLE,
                        "text": content,
                    },
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

    def _open(
        self,
        target: str,
    ) -> str:
        """Open an application, folder, or file."""
        target = self._clean_open_target(target)

        if not target:
            raise ValueError("Falta el objetivo a abrir.")

        path = self._resolve_path(target)

        if path.exists() and path.is_dir():
            return self._run(
                "desktop.open_folder",
                {"path": str(path)},
                f"Carpeta abierta: {path}",
            )

        if path.exists() and path.is_file():
            return self._run(
                "desktop.open_file",
                {
                    "path": str(path),
                    "application": self._DEFAULT_EDITOR,
                },
                f"Archivo abierto en {self._DEFAULT_EDITOR}.",
            )

        if self._looks_like_path(target):
            raise FileNotFoundError(str(path))

        return self._run(
            "desktop.open_application",
            {"application": target},
            f"{target} abierto.",
        )

    def _clean_open_target(
        self,
        target: str,
    ) -> str:
        """Remove natural-language prefixes from an open command."""
        cleaned = target.strip()
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

    def _run(
        self,
        tool_name: str,
        parameters: dict[str, object],
        confirmation: str,
    ) -> str:
        """Run a tool and return a user-facing confirmation."""
        self._executor.execute(
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
        result = self._executor.execute(
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

    def _execute_window_command(
        self,
        text: str,
        normalized: str,
        confirm: Callable[[str], str] | None,
    ) -> str | None:
        """Execute a supported window-management command."""
        if normalized in {"mueve el raton", "mueve el cursor"}:
            return None

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
            title = text[len("activa ") :].strip()
            return self._window_state_action(
                title,
                "desktop.bring_window_to_front",
                "Ventana activada",
                confirm,
            )

        if normalized.startswith("cierra "):
            title = text[len("cierra ") :].strip()
            return self._close_window(title, confirm)

        if normalized in {
            "maximiza",
            "minimiza",
            "restaura",
            "cierra",
            "activa",
        }:
            raise ValueError("Orden incompleta: falta el titulo de ventana.")

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

    def _window_state_action(
        self,
        title: str,
        tool_name: str,
        label: str,
        confirm: Callable[[str], str] | None,
    ) -> str:
        """Execute a state action against one resolved window."""
        window = self._resolve_window(title, confirm)
        self._executor.execute(
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
        self._executor.execute(
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
        self._executor.execute(
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
        self._executor.execute(
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

        self._executor.execute(
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

        result = self._executor.execute(
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
