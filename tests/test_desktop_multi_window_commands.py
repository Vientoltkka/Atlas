from __future__ import annotations

import pytest

from tools.tool_context import ToolContext
from use_cases.desktop_interaction import DesktopInteractionUseCase


class FakeToolExecutor:
    def __init__(self) -> None:
        self.calls: list[tuple[str, ToolContext]] = []
        self.windows: list[dict[str, object]] = []

    def execute(
        self,
        tool_name: str,
        context: ToolContext,
    ):
        self.calls.append((tool_name, context))

        if tool_name == "desktop.list_processes":
            return []

        if tool_name == "desktop.get_screen_size":
            return (1920, 1080)

        if tool_name == "desktop.list_windows":
            title = str(context.parameters.get("title", "")).casefold()
            return [
                window
                for window in self.windows
                if title in str(window["title"]).casefold()
            ]

        if tool_name == "desktop.get_window_rect":
            handle = context.parameters["handle"]

            for window in self.windows:
                if window["handle"] == handle:
                    return window["rect"]

            raise RuntimeError("missing")

        return "ok"


def _build_use_case_with_two_notepads() -> tuple[FakeToolExecutor, DesktopInteractionUseCase]:
    executor = FakeToolExecutor()
    executor.windows.extend(
        [
            {
                "handle": 11,
                "title": "Sin título - Bloc de notas",
                "rect": (0, 0, 640, 480),
            },
            {
                "handle": 22,
                "title": "Atlas - Bloc de notas",
                "rect": (10, 10, 640, 480),
            },
        ]
    )
    return executor, DesktopInteractionUseCase(executor)


def test_activate_second_window_by_ordinal() -> None:
    executor, use_case = _build_use_case_with_two_notepads()

    result = use_case.execute("cambia al segundo bloc de notas")

    assert result == "\u2713 Ventana activada:\nAtlas - Bloc de notas"
    assert executor.calls[0][0] == "desktop.list_windows"
    assert executor.calls[1][0] == "desktop.bring_window_to_front"
    assert executor.calls[1][1].parameters == {"handle": 22}


def test_activate_first_window_by_ordinal() -> None:
    executor, use_case = _build_use_case_with_two_notepads()

    result = use_case.execute("ve a la primera ventana de bloc de notas")

    assert result == "\u2713 Ventana activada:\nSin título - Bloc de notas"
    assert executor.calls[1][1].parameters == {"handle": 11}


def test_activate_ordinal_beyond_available_reports_count() -> None:
    executor, use_case = _build_use_case_with_two_notepads()

    result = use_case.execute("cambia al tercer bloc de notas")

    assert result == "Solo hay 2 ventana(s) para 'tercer bloc de notas'."
    assert all(name != "desktop.bring_window_to_front" for name, _ in executor.calls)


def test_activate_ordinal_without_matches_reports_missing() -> None:
    executor, use_case = _build_use_case_with_two_notepads()

    result = use_case.execute("cambia a la quinta calculadora")

    assert result == "No se encontro ninguna ventana para 'la quinta calculadora'."


def test_close_window_by_name_within_app_description() -> None:
    executor, use_case = _build_use_case_with_two_notepads()

    result = use_case.execute(
        "cierra la ventana de bloc de notas llamada Atlas",
        confirm=lambda _prompt: "s",
    )

    assert result == "\u2713 Solicitud de cierre enviada."
    assert executor.calls[-1][0] == "desktop.close_window"
    assert executor.calls[-1][1].parameters == {"handle": 22}


def test_close_window_by_name_without_confirmation_cancels() -> None:
    executor, use_case = _build_use_case_with_two_notepads()

    result = use_case.execute(
        "cierra la ventana de bloc de notas llamada Atlas",
        confirm=lambda _prompt: "no",
    )

    assert result == "Acción cancelada."
    assert all(name != "desktop.close_window" for name, _ in executor.calls)


def test_state_action_over_second_window_by_ordinal() -> None:
    executor, use_case = _build_use_case_with_two_notepads()

    result = use_case.execute("maximiza el segundo bloc de notas")

    assert result == "\u2713 Ventana maximizada:\nAtlas - Bloc de notas"
    assert executor.calls[-1][1].parameters == {"handle": 22}


def test_ambiguous_without_ordinal_still_asks_selection() -> None:
    executor, use_case = _build_use_case_with_two_notepads()

    result = use_case.execute("maximiza bloc de notas")

    assert "Varias ventanas coinciden" in result or "Selecciona" in result


def test_single_window_activation_without_ordinal_unaffected() -> None:
    executor = FakeToolExecutor()
    executor.windows.append(
        {"handle": 30, "title": "Calculadora", "rect": (0, 0, 320, 500)}
    )
    use_case = DesktopInteractionUseCase(executor)

    result = use_case.execute("ve a calculadora")

    assert result == "\u2713 Ventana activada:\nCalculadora"
    assert executor.calls[1][1].parameters == {"handle": 30}


@pytest.mark.parametrize(
    ("command", "expected_rect"),
    [
        ("pon la calculadora en la mitad izquierda", (0, 0, 960, 1080)),
        ("pon la calculadora en la mitad derecha", (960, 0, 960, 1080)),
        ("pon la calculadora arriba a la izquierda", (0, 0, 960, 540)),
        ("pon la calculadora arriba a la derecha", (960, 0, 960, 540)),
        ("pon la calculadora abajo a la izquierda", (0, 540, 960, 540)),
        ("pon la calculadora abajo a la derecha", (960, 540, 960, 540)),
    ],
)
def test_snap_commands_move_and_resize_to_screen_placements(
    command: str,
    expected_rect: tuple[int, int, int, int],
) -> None:
    executor = FakeToolExecutor()
    executor.windows.append(
        {"handle": 30, "title": "Calculadora", "rect": (0, 0, 320, 500)}
    )
    use_case = DesktopInteractionUseCase(executor)

    result = use_case.execute(command)

    assert result.startswith("\u2713 Ventana anclada")
    assert executor.calls[-1][0] == "desktop.move_resize_window"
    assert executor.calls[-1][1].parameters == {
        "handle": 30,
        "x": expected_rect[0],
        "y": expected_rect[1],
        "width": expected_rect[2],
        "height": expected_rect[3],
    }


def test_snap_uses_resolved_window_when_several_match_with_confirm() -> None:
    executor, use_case = _build_use_case_with_two_notepads()

    result = use_case.execute(
        "pon el segundo bloc de notas en la mitad derecha",
        confirm=lambda _prompt: "2",
    )

    assert result == "\u2713 Ventana anclada a la mitad derecha."
    assert executor.calls[-1][0] == "desktop.move_resize_window"
    assert executor.calls[-1][1].parameters == {
        "handle": 22,
        "x": 960,
        "y": 0,
        "width": 960,
        "height": 1080,
    }


def test_bare_pon_still_activates_instead_of_snapping() -> None:
    executor = FakeToolExecutor()
    executor.windows.append(
        {"handle": 30, "title": "Calculadora", "rect": (0, 0, 320, 500)}
    )
    use_case = DesktopInteractionUseCase(executor)

    result = use_case.execute("pon la calculadora")

    assert result == "\u2713 Ventana activada:\nCalculadora"
    assert all(
        name != "desktop.move_resize_window" for name, _ in executor.calls
    )


@pytest.mark.parametrize(
    "command",
    [
        "tráeme calculadora al frente",
        "traeme calculadora al frente",
        "pon calculadora delante",
        "cambia a calculadora",
        "ve a calculadora",
        "trae calculadora al frente",
    ],
)
def test_natural_focus_variants_activate_the_window(command: str) -> None:
    executor = FakeToolExecutor()
    executor.windows.append(
        {"handle": 30, "title": "Calculadora", "rect": (0, 0, 320, 500)}
    )
    use_case = DesktopInteractionUseCase(executor)

    result = use_case.execute(command)

    assert result == "\u2713 Ventana activada:\nCalculadora"
    assert executor.calls[-1][0] == "desktop.bring_window_to_front"
    assert executor.calls[-1][1].parameters == {"handle": 30}
