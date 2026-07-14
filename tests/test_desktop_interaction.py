from pathlib import Path

from tools.tool_context import ToolContext
from use_cases.desktop_interaction import DesktopInteractionUseCase


class FakeToolExecutor:
    def __init__(self) -> None:
        self.calls: list[tuple[str, ToolContext]] = []

    def execute(
        self,
        tool_name: str,
        context: ToolContext,
    ) -> str:
        self.calls.append((tool_name, context))
        return "ok"


def test_desktop_interaction_opens_application() -> None:
    executor = FakeToolExecutor()
    use_case = DesktopInteractionUseCase(executor)

    result = use_case.execute("Abre Visual Studio Code")

    assert result == "✓ Visual Studio Code abierto."
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

    assert result == f"✓ Carpeta abierta: {folder}"
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

    assert result == f"✓ Carpeta abierta: {folder}"
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

    assert result == "✓ Archivo abierto en Visual Studio Code."
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

    assert result == "✓ Archivo abierto en Visual Studio Code."
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

    assert result == "✓ Texto escrito."
    assert executor.calls[0][0] == "desktop.type_text"
    assert executor.calls[0][1].parameters == {
        "window_title": "Visual Studio Code",
        "text": 'print("Hola")',
    }


def test_desktop_interaction_saves_file() -> None:
    executor = FakeToolExecutor()
    use_case = DesktopInteractionUseCase(executor)

    result = use_case.execute("Guarda el archivo")

    assert result == "✓ Archivo guardado."
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
