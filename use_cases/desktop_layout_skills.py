"""Reusable window-layout skills built on existing desktop tools.

Each skill handler chains already-registered desktop tools (list, activate,
move/resize) following the same curated self-authorization pattern used by
DesktopInteractionUseCase: deterministic reviewed code, validated inputs,
and explicit one-use authorizations for every effectful tool call.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from tools.executor import ToolExecutor
from tools.tool_context import ToolContext
from use_cases.desktop_interaction import DesktopInteractionUseCase


_MODO_TRABAJO_HANDLER_ID = "handler.modo-trabajo"
_MODO_ESCRITURA_HANDLER_ID = "handler.modo-escritura"
_MODO_INVESTIGACION_HANDLER_ID = "handler.modo-investigacion"
_PREPARAR_VENTANA_HANDLER_ID = "handler.preparar-ventana"
_LAYOUT_POSITIONS = ("izquierda", "derecha", "centro")
_WRITING_WIDTH = 900
_WRITING_HEIGHT = 700


class WindowLayoutSkills:
    """Deterministic window layouts exposed as Atlas skill handlers."""

    def __init__(self, tool_executor: ToolExecutor | None) -> None:
        self._executor = tool_executor

    def modo_trabajo(
        self,
        inputs: Mapping[str, object],
        *,
        execution_context: Any = None,
    ) -> Mapping[str, object]:
        title = _optional_title(inputs.get("window_title"))
        window = (
            self._resolve_window(title)
            if title is not None
            else self._foreground_window()
        )
        rect = self._placement_rect("mitad-izquierda")
        self._snap(window, rect)
        label = str(window.get("title", "")).strip()
        return {"result": f"Modo trabajo listo: '{label}' anclada a la mitad izquierda."}

    def modo_escritura(
        self,
        inputs: Mapping[str, object],
        *,
        execution_context: Any = None,
    ) -> Mapping[str, object]:
        title = _optional_title(inputs.get("window_title"))
        window = (
            self._resolve_window(title)
            if title is not None
            else self._foreground_window()
        )
        rect = self._centered_rect(_WRITING_WIDTH, _WRITING_HEIGHT)
        self._snap(window, rect)
        label = str(window.get("title", "")).strip()
        return {
            "result": (
                f"Modo escritura listo: '{label}' centrada en "
                f"{rect[2]} x {rect[3]}."
            )
        }

    def modo_investigacion(
        self,
        inputs: Mapping[str, object],
        *,
        execution_context: Any = None,
    ) -> Mapping[str, object]:
        title = _optional_title(inputs.get("window_title"))
        primary = (
            self._resolve_window(title)
            if title is not None
            else self._foreground_window()
        )
        secondary_title = _optional_title(inputs.get("secondary_title"))
        secondary = (
            self._resolve_window(secondary_title)
            if secondary_title is not None
            else None
        )

        self._snap(primary, self._placement_rect("mitad-izquierda"))
        labels = [f"'{str(primary.get('title', '')).strip()}' a la izquierda"]
        if secondary is not None:
            self._snap(secondary, self._placement_rect("mitad-derecha"))
            labels.append(
                f"'{str(secondary.get('title', '')).strip()}' a la derecha"
            )

        return {"result": "Modo investigacion listo: " + " y ".join(labels) + "."}

    def preparar_ventana(
        self,
        inputs: Mapping[str, object],
        *,
        execution_context: Any = None,
    ) -> Mapping[str, object]:
        title = _optional_title(inputs.get("window_title"))
        if title is None:
            raise ValueError("Falta el titulo de la ventana a preparar.")
        width = inputs.get("width")
        height = inputs.get("height")
        if not isinstance(width, int) or not isinstance(height, int):
            raise ValueError("El tamano debe ser numerico: ancho y alto enteros.")
        if width <= 0 or height <= 0:
            raise ValueError("El ancho y el alto deben ser mayores que cero.")

        position = inputs.get("position")
        position = "derecha" if position is None else str(position).strip().casefold()
        if position not in _LAYOUT_POSITIONS:
            raise ValueError(
                "Posicion no soportada: usa izquierda, derecha o centro."
            )

        window = self._resolve_window(title)
        rect = self._positioned_rect(position, width, height)
        self._snap(window, rect)
        label = str(window.get("title", "")).strip()
        position_phrase = {
            "izquierda": "a la izquierda",
            "derecha": "a la derecha",
            "centro": "en el centro",
        }[position]
        return {
            "result": (
                f"Ventana '{label}' preparada {position_phrase} en "
                f"{rect[2]} x {rect[3]}."
            )
        }

    # ------------------------------------------------------------------
    # Reused desktop tool plumbing
    # ------------------------------------------------------------------

    def _execute_tool(
        self,
        tool_name: str,
        parameters: Mapping[str, object],
    ):
        if self._executor is None:
            raise RuntimeError("tool executor is unavailable")
        authorization = None
        if self._executor.requires_explicit_authorization(tool_name):
            authorization = self._executor.authorize(tool_name)
        return self._executor.execute(
            tool_name,
            ToolContext(parameters=dict(parameters)),
            authorization=authorization,
        )

    def _resolve_window(self, title: str) -> dict[str, object]:
        matches = self._execute_tool("desktop.list_windows", {"title": title})
        if not matches:
            raise ValueError(f"No se encontraron ventanas para '{title}'.")
        if len(matches) > 1:
            raise ValueError(
                f"Varias ventanas ({len(matches)}) coinciden con '{title}'. "
                "Se requiere seleccion explicita."
            )
        return dict(matches[0])

    def _foreground_window(self) -> dict[str, object]:
        window = self._execute_tool("desktop.get_foreground_window", {})
        if not isinstance(window, Mapping) or "handle" not in window:
            raise RuntimeError("No hay una ventana en primer plano.")
        return dict(window)

    def _screen_size(self) -> tuple[int, int]:
        return self._execute_tool("desktop.get_screen_size", {})

    def _placement_rect(
        self,
        placement: str,
    ) -> tuple[int, int, int, int]:
        width, height = self._screen_size()
        return DesktopInteractionUseCase._window_placement_rect(
            placement,
            width,
            height,
        )

    def _centered_rect(
        self,
        width: int,
        height: int,
    ) -> tuple[int, int, int, int]:
        screen_width, screen_height = self._screen_size()
        width = min(width, screen_width)
        height = min(height, screen_height)
        x = (screen_width - width) // 2
        y = (screen_height - height) // 2
        return (x, y, width, height)

    def _positioned_rect(
        self,
        position: str,
        width: int,
        height: int,
    ) -> tuple[int, int, int, int]:
        screen_width, screen_height = self._screen_size()
        width = min(width, screen_width)
        height = min(height, screen_height)
        y = (screen_height - height) // 2
        if position == "izquierda":
            return (0, y, width, height)
        if position == "derecha":
            return (screen_width - width, y, width, height)
        return ((screen_width - width) // 2, y, width, height)

    def _snap(
        self,
        window: Mapping[str, object],
        rect: tuple[int, int, int, int],
    ) -> None:
        handle = int(window["handle"])
        x, y, width, height = rect
        self._execute_tool(
            "desktop.bring_window_to_front",
            {"handle": handle},
        )
        self._execute_tool(
            "desktop.move_resize_window",
            {
                "handle": handle,
                "x": x,
                "y": y,
                "width": width,
                "height": height,
            },
        )


def _optional_title(value: object) -> str | None:
    if value is None:
        return None
    normalized = " ".join(str(value).strip().split())
    return normalized or None
