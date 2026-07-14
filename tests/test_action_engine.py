from pathlib import Path

import pytest

from tools.tool_context import ToolContext
from use_cases.action_engine import (
    ActionEngineUseCase,
    ActionStep,
    CopyAndPasteTextUseCase,
    PrepareAtlasWorkspaceUseCase,
)
from use_cases.desktop_interaction import DesktopInteractionUseCase


class FakeToolExecutor:
    def __init__(self) -> None:
        self.calls: list[tuple[str, ToolContext]] = []
        self.fail_tool: str | None = None
        self.windows: list[dict[str, object]] = [
            {
                "handle": 20,
                "title": "Atlas - Explorador de archivos",
                "rect": (0, 0, 100, 100),
            },
            {
                "handle": 10,
                "title": "orchestrator.py - Atlas - Visual Studio Code",
                "rect": (0, 0, 100, 100),
            },
        ]

    def execute(
        self,
        tool_name: str,
        context: ToolContext,
    ) -> str:
        self.calls.append((tool_name, context))

        if tool_name == self.fail_tool:
            raise RuntimeError(f"{tool_name} failed")

        if tool_name == "desktop.get_foreground_window":
            return {
                "handle": 20,
                "title": "Atlas - Explorador de archivos",
                "rect": (0, 0, 100, 100),
            }

        if tool_name == "desktop.list_windows":
            title = str(context.parameters["title"]).lower()
            return [
                window
                for window in self.windows
                if title in str(window["title"]).lower()
            ]

        return f"{tool_name} ok"


def test_action_engine_executes_actions_in_order() -> None:
    calls: list[str] = []
    engine = ActionEngineUseCase()
    steps = [
        ActionStep("one", lambda: calls.append("one") or "one ok"),
        ActionStep("two", lambda: calls.append("two") or "two ok"),
    ]

    result = engine.execute("demo", steps)

    assert calls == ["one", "two"]
    assert [step.step_name for step in result.step_results] == ["one", "two"]


def test_action_engine_completes_all_steps_successfully() -> None:
    engine = ActionEngineUseCase()
    steps = [
        ActionStep("one", lambda: "one ok"),
        ActionStep("two", lambda: "two ok"),
    ]

    result = engine.execute("demo", steps)

    assert result.completed is True
    assert result.stopped_early is False
    assert result.total_steps == 2
    assert result.executed_steps == 2
    assert result.successful_steps == 2
    assert result.failed_steps == 0
    assert "Pasos ejecutados: 2" in result.summary


def test_action_engine_stops_on_first_failure() -> None:
    calls: list[str] = []

    def fail() -> str:
        calls.append("fail")
        raise RuntimeError("boom")

    engine = ActionEngineUseCase()
    steps = [
        ActionStep("one", lambda: calls.append("one") or "one ok"),
        ActionStep("fail", fail),
        ActionStep("three", lambda: calls.append("three") or "three ok"),
    ]

    result = engine.execute("demo", steps)

    assert calls == ["one", "fail"]
    assert result.completed is False
    assert result.stopped_early is True
    assert result.executed_steps == 2
    assert result.successful_steps == 1
    assert result.failed_steps == 1
    assert result.step_results[1].error == "boom"


def test_prepare_atlas_workspace_validates_project_root_exists(
    tmp_path: Path,
) -> None:
    executor = FakeToolExecutor()
    use_case = PrepareAtlasWorkspaceUseCase(executor, ActionEngineUseCase())

    with pytest.raises(ValueError):
        use_case.execute(tmp_path / "missing")

    assert executor.calls == []


def test_prepare_atlas_workspace_rejects_project_root_file(
    tmp_path: Path,
) -> None:
    root_file = tmp_path / "Atlas"
    root_file.write_text("", encoding="utf-8")
    executor = FakeToolExecutor()
    use_case = PrepareAtlasWorkspaceUseCase(executor, ActionEngineUseCase())

    with pytest.raises(ValueError):
        use_case.execute(root_file)

    assert executor.calls == []


def test_prepare_atlas_workspace_validates_editor_file_exists(
    tmp_path: Path,
) -> None:
    executor = FakeToolExecutor()
    use_case = PrepareAtlasWorkspaceUseCase(executor, ActionEngineUseCase())

    with pytest.raises(ValueError):
        use_case.execute(tmp_path, "core/orchestrator.py")

    assert executor.calls == []


def test_prepare_atlas_workspace_rejects_file_outside_project(
    tmp_path: Path,
) -> None:
    outside = tmp_path.parent / "outside.py"
    outside.write_text("", encoding="utf-8")
    executor = FakeToolExecutor()
    use_case = PrepareAtlasWorkspaceUseCase(executor, ActionEngineUseCase())

    with pytest.raises(ValueError):
        use_case.execute(tmp_path, outside)

    assert executor.calls == []


def test_prepare_atlas_workspace_rejects_non_python_file(
    tmp_path: Path,
) -> None:
    file = tmp_path / "README.md"
    file.write_text("", encoding="utf-8")
    executor = FakeToolExecutor()
    use_case = PrepareAtlasWorkspaceUseCase(executor, ActionEngineUseCase())

    with pytest.raises(ValueError):
        use_case.execute(tmp_path, "README.md")

    assert executor.calls == []


def test_prepare_atlas_workspace_uses_default_editor_file(
    tmp_path: Path,
) -> None:
    file = tmp_path / "core" / "orchestrator.py"
    file.parent.mkdir()
    file.write_text("", encoding="utf-8")
    executor = FakeToolExecutor()
    use_case = PrepareAtlasWorkspaceUseCase(executor, ActionEngineUseCase())

    result = use_case.execute(tmp_path)

    assert result.completed is True
    assert [call[0] for call in executor.calls] == [
        "desktop.open_application",
        "desktop.open_folder",
        "desktop.open_file",
        "desktop.list_windows",
        "desktop.list_windows",
        "desktop.bring_window_to_front",
    ]
    assert executor.calls[2][1].parameters["path"] == str(file.resolve())
    assert executor.calls[5][1].parameters == {"handle": 10}


def test_prepare_atlas_workspace_accepts_configured_python_file(
    tmp_path: Path,
) -> None:
    file = tmp_path / "main.py"
    file.write_text("", encoding="utf-8")
    executor = FakeToolExecutor()
    use_case = PrepareAtlasWorkspaceUseCase(executor, ActionEngineUseCase())

    result = use_case.execute(tmp_path, "main.py")

    assert result.completed is True
    assert executor.calls[2][1].parameters["path"] == str(file.resolve())


def test_prepare_atlas_workspace_stops_when_application_fails(
    tmp_path: Path,
) -> None:
    file = tmp_path / "core" / "orchestrator.py"
    file.parent.mkdir()
    file.write_text("", encoding="utf-8")
    executor = FakeToolExecutor()
    executor.fail_tool = "desktop.open_application"
    use_case = PrepareAtlasWorkspaceUseCase(executor, ActionEngineUseCase())

    result = use_case.execute(tmp_path)

    assert result.completed is False
    assert result.stopped_early is True
    assert result.executed_steps == 1
    assert result.failed_steps == 1
    assert [call[0] for call in executor.calls] == ["desktop.open_application"]


def test_prepare_atlas_workspace_stops_when_folder_fails(
    tmp_path: Path,
) -> None:
    file = tmp_path / "core" / "orchestrator.py"
    file.parent.mkdir()
    file.write_text("", encoding="utf-8")
    executor = FakeToolExecutor()
    executor.fail_tool = "desktop.open_folder"
    use_case = PrepareAtlasWorkspaceUseCase(executor, ActionEngineUseCase())

    result = use_case.execute(tmp_path)

    assert result.completed is False
    assert [call[0] for call in executor.calls] == [
        "desktop.open_application",
        "desktop.open_folder",
    ]


def test_prepare_atlas_workspace_stops_when_file_fails(
    tmp_path: Path,
) -> None:
    file = tmp_path / "core" / "orchestrator.py"
    file.parent.mkdir()
    file.write_text("", encoding="utf-8")
    executor = FakeToolExecutor()
    executor.fail_tool = "desktop.open_file"
    use_case = PrepareAtlasWorkspaceUseCase(executor, ActionEngineUseCase())

    result = use_case.execute(tmp_path)

    assert result.completed is False
    assert [call[0] for call in executor.calls] == [
        "desktop.open_application",
        "desktop.open_folder",
        "desktop.open_file",
    ]


def test_prepare_atlas_workspace_reports_activation_failure(
    tmp_path: Path,
) -> None:
    file = tmp_path / "core" / "orchestrator.py"
    file.parent.mkdir()
    file.write_text("", encoding="utf-8")
    executor = FakeToolExecutor()
    executor.fail_tool = "desktop.list_windows"
    use_case = PrepareAtlasWorkspaceUseCase(executor, ActionEngineUseCase())

    result = use_case.execute(tmp_path)

    assert result.completed is False
    assert result.executed_steps == 4
    assert result.successful_steps == 3
    assert result.failed_steps == 1
    assert result.step_results[-1].error == "desktop.list_windows failed"


def test_prepare_atlas_workspace_ignores_foreground_explorer_and_uses_vscode(
    tmp_path: Path,
) -> None:
    file = tmp_path / "core" / "orchestrator.py"
    file.parent.mkdir()
    file.write_text("", encoding="utf-8")
    executor = FakeToolExecutor()
    use_case = PrepareAtlasWorkspaceUseCase(executor, ActionEngineUseCase())

    result = use_case.execute(tmp_path)

    assert result.completed is True
    assert ("desktop.get_foreground_window", ToolContext()) not in executor.calls
    assert executor.calls[-1][0] == "desktop.bring_window_to_front"
    assert executor.calls[-1][1].parameters == {"handle": 10}


def test_prepare_atlas_workspace_prioritizes_atlas_window_over_other_vscode(
    tmp_path: Path,
) -> None:
    file = tmp_path / "core" / "orchestrator.py"
    file.parent.mkdir()
    file.write_text("", encoding="utf-8")
    executor = FakeToolExecutor()
    executor.windows.append(
        {
            "handle": 11,
            "title": "README.md - Other Project - Visual Studio Code",
            "rect": (0, 0, 100, 100),
        }
    )
    use_case = PrepareAtlasWorkspaceUseCase(executor, ActionEngineUseCase())

    result = use_case.execute(tmp_path)

    assert result.completed is True
    assert executor.calls[-1][1].parameters == {"handle": 10}


def test_prepare_atlas_workspace_fails_when_vscode_matches_remain_ambiguous(
    tmp_path: Path,
) -> None:
    file = tmp_path / "core" / "orchestrator.py"
    file.parent.mkdir()
    file.write_text("", encoding="utf-8")
    executor = FakeToolExecutor()
    executor.windows = [
        {
            "handle": 10,
            "title": "Visual Studio Code",
            "rect": (0, 0, 100, 100),
        },
        {
            "handle": 11,
            "title": "Other - Visual Studio Code",
            "rect": (0, 0, 100, 100),
        },
    ]
    use_case = PrepareAtlasWorkspaceUseCase(executor, ActionEngineUseCase())

    result = use_case.execute(tmp_path)

    assert result.completed is False
    assert result.step_results[-1].success is False
    assert "ambiguas" in str(result.step_results[-1].error)
    assert all(call[0] != "desktop.bring_window_to_front" for call in executor.calls)


def test_prepare_atlas_workspace_uses_z_order_to_break_identical_vscode_titles(
    tmp_path: Path,
) -> None:
    file = tmp_path / "core" / "orchestrator.py"
    file.parent.mkdir()
    file.write_text("", encoding="utf-8")
    executor = FakeToolExecutor()
    executor.windows = [
        {
            "handle": 12,
            "title": "orchestrator.py - Visual Studio Code",
            "rect": (0, 0, 100, 100),
            "order": 2,
        },
        {
            "handle": 10,
            "title": "orchestrator.py - Visual Studio Code",
            "rect": (0, 0, 100, 100),
            "order": 0,
        },
    ]
    use_case = PrepareAtlasWorkspaceUseCase(executor, ActionEngineUseCase())

    result = use_case.execute(tmp_path)

    assert result.completed is True
    assert executor.calls[-1][1].parameters == {"handle": 10}


def test_copy_and_paste_text_executes_steps_in_order() -> None:
    executor = FakeToolExecutor()
    use_case = CopyAndPasteTextUseCase(executor, ActionEngineUseCase())

    result = use_case.execute("Hola Atlas", "Visual Studio Code")

    assert result.completed is True
    assert result.workflow_name == "copy_and_paste_text"
    assert [call[0] for call in executor.calls] == [
        "desktop.copy_clipboard_text",
        "desktop.activate_window",
        "desktop.press_hotkey",
    ]
    assert executor.calls[0][1].parameters == {"text": "Hola Atlas"}
    assert executor.calls[1][1].parameters == {
        "window_title": "Visual Studio Code"
    }
    assert executor.calls[2][1].parameters == {
        "window_title": "Visual Studio Code",
        "keys": ["ctrl", "v"],
    }


def test_copy_and_paste_text_stops_when_copy_fails() -> None:
    executor = FakeToolExecutor()
    executor.fail_tool = "desktop.copy_clipboard_text"
    use_case = CopyAndPasteTextUseCase(executor, ActionEngineUseCase())

    result = use_case.execute("Hola Atlas", "Visual Studio Code")

    assert result.completed is False
    assert result.stopped_early is True
    assert [call[0] for call in executor.calls] == [
        "desktop.copy_clipboard_text"
    ]


def test_copy_and_paste_text_stops_when_activation_fails() -> None:
    executor = FakeToolExecutor()
    executor.fail_tool = "desktop.activate_window"
    use_case = CopyAndPasteTextUseCase(executor, ActionEngineUseCase())

    result = use_case.execute("Hola Atlas", "Visual Studio Code")

    assert result.completed is False
    assert result.stopped_early is True
    assert [call[0] for call in executor.calls] == [
        "desktop.copy_clipboard_text",
        "desktop.activate_window",
    ]
    assert result.step_results[-1].success is False


def test_copy_and_paste_text_returns_structured_result() -> None:
    executor = FakeToolExecutor()
    use_case = CopyAndPasteTextUseCase(executor, ActionEngineUseCase())

    result = use_case.execute("Hola Atlas", "Visual Studio Code")

    assert result.total_steps == 3
    assert result.executed_steps == 3
    assert result.successful_steps == 3
    assert result.failed_steps == 0
    assert [step.step_name for step in result.step_results] == [
        "Copiar texto al portapapeles",
        "Activar ventana destino",
        "Pegar contenido del portapapeles",
    ]


def test_desktop_interaction_interprets_prepare_commands(
    tmp_path: Path,
) -> None:
    file = tmp_path / "core" / "orchestrator.py"
    file.parent.mkdir()
    file.write_text("", encoding="utf-8")
    executor = FakeToolExecutor()
    workflow = PrepareAtlasWorkspaceUseCase(executor, ActionEngineUseCase())
    use_case = DesktopInteractionUseCase(
        executor,
        project_root=tmp_path,
        prepare_atlas_workspace=workflow,
    )

    for command in (
        "prepara Atlas para trabajar",
        "prepara el proyecto Atlas",
        "abre el entorno de Atlas",
        "prepare Atlas workspace",
    ):
        executor.calls.clear()
        result = use_case.execute(command)

        assert "Automatización iniciada: preparar Atlas para trabajar" in result
        assert "Pasos ejecutados: 4" in result
        assert [call[0] for call in executor.calls] == [
            "desktop.open_application",
            "desktop.open_folder",
            "desktop.open_file",
            "desktop.list_windows",
            "desktop.list_windows",
            "desktop.bring_window_to_front",
        ]


def test_desktop_interaction_reports_partial_workflow_failure(
    tmp_path: Path,
) -> None:
    file = tmp_path / "core" / "orchestrator.py"
    file.parent.mkdir()
    file.write_text("", encoding="utf-8")
    executor = FakeToolExecutor()
    executor.fail_tool = "desktop.open_file"
    workflow = PrepareAtlasWorkspaceUseCase(executor, ActionEngineUseCase())
    use_case = DesktopInteractionUseCase(
        executor,
        project_root=tmp_path,
        prepare_atlas_workspace=workflow,
    )

    result = use_case.execute("prepara Atlas para trabajar")

    assert "Automatización incompleta." in result
    assert "Error en Abrir archivo core/orchestrator.py:" in result
    assert "desktop.open_file failed" in result
    assert "Pasos ejecutados: 3" in result


def test_desktop_interaction_does_not_handle_free_sequences() -> None:
    executor = FakeToolExecutor()
    use_case = DesktopInteractionUseCase(executor)

    result = use_case.execute("abre VS Code y luego Chrome")

    assert result == "Error: No se admiten secuencias libres."
    assert executor.calls == []
