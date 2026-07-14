"""Desktop interaction use case."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Callable
import unicodedata

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
    ) -> None:
        self._executor = executor
        self._project_root = project_root or Path(".")
        self._screenshots_dir = (
            screenshots_dir
            if screenshots_dir is not None
            else self._project_root / "artifacts" / "screenshots"
        )

    def execute(
        self,
        prompt: str,
        confirm: Callable[[str], str] | None = None,
    ) -> str | None:
        """Execute a supported desktop command."""
        text = prompt.strip()
        normalized = self._normalize(text)

        try:
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
