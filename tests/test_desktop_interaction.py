from pathlib import Path

from tools.tool_context import ToolContext
from use_cases.desktop_interaction import DesktopInteractionUseCase


class FakeToolExecutor:
    def __init__(self) -> None:
        self.calls: list[tuple[str, ToolContext]] = []
        self.fail_on_execute = False

    def execute(
        self,
        tool_name: str,
        context: ToolContext,
    ):
        if self.fail_on_execute:
            raise RuntimeError("tool failed")

        self.calls.append((tool_name, context))

        if tool_name == "desktop.get_screen_size":
            return (1920, 1080)

        if tool_name == "desktop.get_cursor_position":
            return (812, 430)

        if tool_name == "desktop.capture_screenshot":
            return str(Path(context.parameters["output_dir"]) / "shot.png")

        return "ok"


def test_desktop_interaction_opens_application() -> None:
    executor = FakeToolExecutor()
    use_case = DesktopInteractionUseCase(executor)

    result = use_case.execute("Abre Visual Studio Code")

    assert result == "\u2713 Visual Studio Code abierto."
    assert executor.calls[0][0] == "desktop.open_application"
    assert executor.calls[0][1].parameters == {
        "application": "Visual Studio Code"
    }


def test_desktop_interaction_opens_existing_folder(
    tmp_path: Path,
) -> None:
    executor = FakeToolExecutor()
    use_case = DesktopInteractionUseCase(executor, project_root=tmp_path)
    folder = tmp_path / "core"
    folder.mkdir()

    result = use_case.execute("Abre core")

    assert result == f"\u2713 Carpeta abierta: {folder}"
    assert executor.calls[0][0] == "desktop.open_folder"
    assert executor.calls[0][1].parameters == {"path": str(folder)}


def test_desktop_interaction_opens_folder_with_natural_prefix(
    tmp_path: Path,
) -> None:
    executor = FakeToolExecutor()
    use_case = DesktopInteractionUseCase(executor, project_root=tmp_path)
    folder = tmp_path / "Atlas"
    folder.mkdir()

    result = use_case.execute(f"Abre la carpeta {folder}")

    assert result == f"\u2713 Carpeta abierta: {folder}"
    assert executor.calls[0][0] == "desktop.open_folder"


def test_desktop_interaction_opens_existing_file(
    tmp_path: Path,
) -> None:
    executor = FakeToolExecutor()
    use_case = DesktopInteractionUseCase(executor, project_root=tmp_path)
    file = tmp_path / "core" / "router.py"
    file.parent.mkdir()
    file.write_text("print('demo')", encoding="utf-8")

    result = use_case.execute("Abre core/router.py")

    assert result == "\u2713 Archivo abierto en Visual Studio Code."
    assert executor.calls[0][0] == "desktop.open_file"
    assert executor.calls[0][1].parameters == {
        "path": str(file),
        "application": "Visual Studio Code",
    }


def test_desktop_interaction_opens_file_with_natural_prefix(
    tmp_path: Path,
) -> None:
    executor = FakeToolExecutor()
    use_case = DesktopInteractionUseCase(executor, project_root=tmp_path)
    file = tmp_path / "core" / "router.py"
    file.parent.mkdir()
    file.write_text("print('demo')", encoding="utf-8")

    result = use_case.execute("Abre el archivo core/router.py")

    assert result == "\u2713 Archivo abierto en Visual Studio Code."
    assert executor.calls[0][0] == "desktop.open_file"


def test_desktop_interaction_reports_missing_file(
    tmp_path: Path,
) -> None:
    executor = FakeToolExecutor()
    use_case = DesktopInteractionUseCase(executor, project_root=tmp_path)

    result = use_case.execute("Abre missing.py")

    assert result == f"Error: {tmp_path / 'missing.py'}"
    assert executor.calls == []


def test_desktop_interaction_types_text() -> None:
    executor = FakeToolExecutor()
    use_case = DesktopInteractionUseCase(executor)

    result = use_case.execute('Escribe: print("Hola")')

    assert result == "\u2713 Texto escrito."
    assert executor.calls[0][0] == "desktop.type_text"
    assert executor.calls[0][1].parameters == {
        "window_title": "Visual Studio Code",
        "text": 'print("Hola")',
    }


def test_desktop_interaction_saves_file() -> None:
    executor = FakeToolExecutor()
    use_case = DesktopInteractionUseCase(executor)

    result = use_case.execute("Guarda el archivo")

    assert result == "\u2713 Archivo guardado."
    assert executor.calls[0][0] == "desktop.save_file"
    assert executor.calls[0][1].parameters == {
        "window_title": "Visual Studio Code"
    }


def test_desktop_interaction_ignores_unknown_command() -> None:
    executor = FakeToolExecutor()
    use_case = DesktopInteractionUseCase(executor)

    result = use_case.execute("Analiza router.py")

    assert result is None
    assert executor.calls == []


def test_desktop_interaction_gets_screen_size() -> None:
    executor = FakeToolExecutor()
    use_case = DesktopInteractionUseCase(executor)

    result = use_case.execute("tamaño de pantalla")

    assert result == "\u2713 Tamaño de pantalla: 1920 x 1080."
    assert executor.calls[0][0] == "desktop.get_screen_size"


def test_desktop_interaction_gets_cursor_position() -> None:
    executor = FakeToolExecutor()
    use_case = DesktopInteractionUseCase(executor)

    result = use_case.execute("posición del ratón")

    assert result == "\u2713 Posición actual: (812, 430)."
    assert executor.calls[0][0] == "desktop.get_cursor_position"


def test_desktop_interaction_moves_cursor_with_comma_coordinates() -> None:
    executor = FakeToolExecutor()
    use_case = DesktopInteractionUseCase(executor)

    result = use_case.execute("mueve el ratón a 500, 300")

    assert result == "\u2713 Cursor movido a (500, 300)."
    assert executor.calls[0][0] == "desktop.move_cursor"
    assert executor.calls[0][1].parameters == {"x": 500, "y": 300}


def test_desktop_interaction_moves_cursor_with_space_coordinates() -> None:
    executor = FakeToolExecutor()
    use_case = DesktopInteractionUseCase(executor)

    result = use_case.execute("mueve el cursor a 500 300")

    assert result == "\u2713 Cursor movido a (500, 300)."
    assert executor.calls[0][0] == "desktop.move_cursor"


def test_desktop_interaction_rejects_incomplete_move_command() -> None:
    executor = FakeToolExecutor()
    use_case = DesktopInteractionUseCase(executor)

    result = use_case.execute("mueve el ratón")

    assert result == "Error: Orden incompleta: faltan coordenadas."
    assert executor.calls == []


def test_desktop_interaction_rejects_non_numeric_coordinates() -> None:
    executor = FakeToolExecutor()
    use_case = DesktopInteractionUseCase(executor)

    result = use_case.execute("haz clic en abc, 300", confirm=lambda _: "s")

    assert result == "Error: Las coordenadas deben ser numericas."
    assert executor.calls == []


def test_desktop_interaction_rejects_negative_click_coordinates() -> None:
    executor = FakeToolExecutor()
    use_case = DesktopInteractionUseCase(executor)

    result = use_case.execute("haz clic en -1, 300", confirm=lambda _: "s")

    assert result == "Error: Las coordenadas no pueden ser negativas."
    assert executor.calls == []


def test_desktop_interaction_rejects_out_of_screen_click_coordinates() -> None:
    executor = FakeToolExecutor()
    use_case = DesktopInteractionUseCase(executor)

    result = use_case.execute("haz clic en 99999, 99999", confirm=lambda _: "s")

    assert result == "Error: Coordenadas fuera de pantalla: (99999, 99999)."
    assert [call[0] for call in executor.calls] == ["desktop.get_screen_size"]


def test_desktop_interaction_left_click_after_confirmation() -> None:
    executor = FakeToolExecutor()
    use_case = DesktopInteractionUseCase(executor)

    result = use_case.execute("haz clic en 500, 300", confirm=lambda _: "s")

    assert result == "\u2713 Clic realizado en (500, 300)."
    assert executor.calls[0][0] == "desktop.get_screen_size"
    assert executor.calls[1][0] == "desktop.left_click"
    assert executor.calls[1][1].parameters == {"x": 500, "y": 300}


def test_desktop_interaction_double_click_after_confirmation() -> None:
    executor = FakeToolExecutor()
    use_case = DesktopInteractionUseCase(executor)

    result = use_case.execute(
        "haz doble clic en 500, 300",
        confirm=lambda _: "si",
    )

    assert result == "\u2713 Doble clic realizado en (500, 300)."
    assert executor.calls[0][0] == "desktop.get_screen_size"
    assert executor.calls[1][0] == "desktop.double_click"


def test_desktop_interaction_right_click_after_confirmation() -> None:
    executor = FakeToolExecutor()
    use_case = DesktopInteractionUseCase(executor)

    result = use_case.execute(
        "haz clic derecho en 500, 300",
        confirm=lambda _: "yes",
    )

    assert result == "\u2713 Clic derecho realizado en (500, 300)."
    assert executor.calls[0][0] == "desktop.get_screen_size"
    assert executor.calls[1][0] == "desktop.right_click"


def test_desktop_interaction_cancels_click_with_empty_response() -> None:
    executor = FakeToolExecutor()
    use_case = DesktopInteractionUseCase(executor)

    result = use_case.execute("clic en 500 300", confirm=lambda _: "")

    assert result == "Acción cancelada."
    assert [call[0] for call in executor.calls] == ["desktop.get_screen_size"]


def test_desktop_interaction_cancels_click_with_no_response() -> None:
    executor = FakeToolExecutor()
    use_case = DesktopInteractionUseCase(executor)

    result = use_case.execute("clic en 500 300", confirm=lambda _: "n")

    assert result == "Acción cancelada."
    assert [call[0] for call in executor.calls] == ["desktop.get_screen_size"]


def test_desktop_interaction_click_requires_confirmation_callback() -> None:
    executor = FakeToolExecutor()
    use_case = DesktopInteractionUseCase(executor)

    result = use_case.execute("clic en 500 300")

    assert result == "Acción cancelada."
    assert [call[0] for call in executor.calls] == ["desktop.get_screen_size"]


def test_desktop_interaction_scrolls_down() -> None:
    executor = FakeToolExecutor()
    use_case = DesktopInteractionUseCase(executor)

    result = use_case.execute("desplázate hacia abajo")

    assert result == "\u2713 Scroll hacia abajo."
    assert executor.calls[0][0] == "desktop.scroll_vertical"
    assert executor.calls[0][1].parameters == {"direction": "down"}


def test_desktop_interaction_scrolls_up() -> None:
    executor = FakeToolExecutor()
    use_case = DesktopInteractionUseCase(executor)

    result = use_case.execute("scroll arriba")

    assert result == "\u2713 Scroll hacia arriba."
    assert executor.calls[0][0] == "desktop.scroll_vertical"
    assert executor.calls[0][1].parameters == {"direction": "up"}


def test_desktop_interaction_rejects_incomplete_click() -> None:
    executor = FakeToolExecutor()
    use_case = DesktopInteractionUseCase(executor)

    result = use_case.execute("haz clic")

    assert result == "Error: Orden incompleta: faltan coordenadas."
    assert executor.calls == []


def test_desktop_interaction_captures_screenshot(
    tmp_path: Path,
) -> None:
    executor = FakeToolExecutor()
    screenshots = tmp_path / "screenshots"
    use_case = DesktopInteractionUseCase(
        executor,
        screenshots_dir=screenshots,
    )

    result = use_case.execute("haz una captura de pantalla")

    assert result == f"\u2713 Captura guardada en:\n{screenshots / 'shot.png'}"
    assert executor.calls[0][0] == "desktop.capture_screenshot"
    assert executor.calls[0][1].parameters == {
        "output_dir": str(screenshots)
    }


def test_desktop_interaction_rejects_region_capture() -> None:
    executor = FakeToolExecutor()
    use_case = DesktopInteractionUseCase(executor)

    result = use_case.execute("captura una región")

    assert result == "Error: Solo se admite captura de pantalla completa."
    assert executor.calls == []


def test_desktop_interaction_reports_screenshot_failure() -> None:
    executor = FakeToolExecutor()
    executor.fail_on_execute = True
    use_case = DesktopInteractionUseCase(executor)

    result = use_case.execute("screenshot")

    assert result == "Error: tool failed"
