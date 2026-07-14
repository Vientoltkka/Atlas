from pathlib import Path

import pytest

from tools.tool_context import ToolContext
from use_cases.action_engine import (
    ActionEngineUseCase,
    ActionStep,
    CopyAndPasteTextUseCase,
    PrepareAtlasWorkspaceUseCase,
    RestartApplicationUseCase,
)
from use_cases.desktop_interaction import DesktopInteractionUseCase
from use_cases.wait_engine import WaitEngine


class ManualClock:
    def __init__(self) -> None:
        self.now = 0.0

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


class FakeToolExecutor:
    def __init__(self) -> None:
        self.calls: list[tuple[str, ToolContext]] = []
        self.fail_tool: str | None = None
        self.foreground_window: dict[str, object] = {
            "handle": 20,
            "title": "Atlas - Explorador de archivos",
            "rect": (0, 0, 100, 100),
        }
        self.processes: list[dict[str, object]] = [
            {
                "pid": 20,
                "name": "chrome.exe",
                "executable_path": None,
                "window_titles": ("Chrome",),
                "is_running": True,
            }
        ]
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
            return self.foreground_window

        if tool_name == "desktop.list_windows":
            title = str(context.parameters["title"]).lower()
            return [
                window
                for window in self.windows
                if title in str(window["title"]).lower()
            ]

        if tool_name == "desktop.list_processes":
            query = str(context.parameters["query"]).lower()
            return [
                process
                for process in self.processes
                if query in str(process["name"]).lower()
                or query in str(process["name"]).lower().replace(".exe", "")
            ]

        if tool_name == "desktop.bring_window_to_front":
            handle = context.parameters["handle"]
            self.foreground_window = next(
                window for window in self.windows if window["handle"] == handle
            )
            return "activated"

        if tool_name == "desktop.close_application":
            self.processes = []
            return "closed"

        if tool_name == "desktop.open_application":
            application = str(context.parameters["application"]).lower()
            if "visual studio code" in application:
                self.processes = [
                    {
                        "pid": 21,
                        "name": "Code.exe",
                        "executable_path": None,
                        "window_titles": ("Visual Studio Code",),
                        "is_running": True,
                    }
                ]
            else:
                self.processes = [
                    {
                        "pid": 21,
                        "name": "chrome.exe",
                        "executable_path": None,
                        "window_titles": ("Chrome",),
                        "is_running": True,
                    }
                ]
            return "opened"

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


def test_action_engine_stops_on_failed_wait_result() -> None:
    executor = FakeToolExecutor()
    executor.processes = []
    clock = ManualClock()
    wait_engine = WaitEngine(executor, clock.monotonic, clock.sleep)
    calls: list[str] = []
    engine = ActionEngineUseCase()

    result = engine.execute(
        "wait_demo",
        [
            ActionStep(
                "Esperar proceso",
                lambda: wait_engine.wait_process(
                    "missing",
                    timeout=1,
                    poll_interval=0.5,
                ),
            ),
            ActionStep("No ejecutar", lambda: calls.append("late") or "ok"),
        ],
    )

    assert result.completed is False
    assert result.stopped_early is True
    assert result.executed_steps == 1
    assert result.step_results[0].error == "Timeout agotado."
    assert calls == []


def test_wait_engine_process_existing() -> None:
    executor = FakeToolExecutor()
    wait_engine = WaitEngine(executor, sleeper=lambda _: None)

    result = wait_engine.wait_process("chrome", timeout=1, poll_interval=0.1)

    assert result.completed is True
    assert result.condition == "process_exists"


def test_wait_engine_process_missing_times_out() -> None:
    executor = FakeToolExecutor()
    executor.processes = []
    clock = ManualClock()
    wait_engine = WaitEngine(executor, clock.monotonic, clock.sleep)

    result = wait_engine.wait_process("chrome", timeout=1, poll_interval=0.5)

    assert result.completed is False
    assert result.timed_out is True
    assert result.error == "Timeout agotado."


def test_wait_engine_window_existing() -> None:
    executor = FakeToolExecutor()
    wait_engine = WaitEngine(executor, sleeper=lambda _: None)

    result = wait_engine.wait_window(
        "Visual Studio Code",
        timeout=1,
        poll_interval=0.1,
    )

    assert result.completed is True
    assert result.condition == "window_exists"


def test_wait_engine_window_timeout() -> None:
    executor = FakeToolExecutor()
    executor.windows = []
    clock = ManualClock()
    wait_engine = WaitEngine(executor, clock.monotonic, clock.sleep)

    result = wait_engine.wait_window("Guardar como", timeout=1, poll_interval=0.5)

    assert result.completed is False
    assert result.condition == "window_exists"


def test_wait_engine_file_existing(tmp_path: Path) -> None:
    file = tmp_path / "README.md"
    file.write_text("ok", encoding="utf-8")
    executor = FakeToolExecutor()
    wait_engine = WaitEngine(executor, sleeper=lambda _: None)

    result = wait_engine.wait_file(file, timeout=1, poll_interval=0.1)

    assert result.completed is True
    assert result.condition == "file_exists"


def test_wait_engine_file_timeout(tmp_path: Path) -> None:
    executor = FakeToolExecutor()
    clock = ManualClock()
    wait_engine = WaitEngine(executor, clock.monotonic, clock.sleep)

    result = wait_engine.wait_file(
        tmp_path / "missing.txt",
        timeout=1,
        poll_interval=0.5,
    )

    assert result.completed is False
    assert result.condition == "file_exists"


def test_wait_engine_application_disappears() -> None:
    executor = FakeToolExecutor()
    executor.processes = []
    wait_engine = WaitEngine(executor, sleeper=lambda _: None)

    result = wait_engine.wait_application(
        "chrome",
        timeout=1,
        poll_interval=0.1,
        opened=False,
    )

    assert result.completed is True
    assert result.condition == "process_absent"


def test_wait_engine_active_window() -> None:
    executor = FakeToolExecutor()
    executor.foreground_window = {
        "handle": 10,
        "title": "orchestrator.py - Atlas - Visual Studio Code",
        "rect": (0, 0, 100, 100),
    }
    wait_engine = WaitEngine(executor, sleeper=lambda _: None)

    result = wait_engine.wait_active_window(
        "Visual Studio Code",
        timeout=1,
        poll_interval=0.1,
    )

    assert result.completed is True
    assert result.condition == "active_window"


def test_wait_engine_cancellation() -> None:
    executor = FakeToolExecutor()
    clock = ManualClock()
    wait_engine = WaitEngine(executor, clock.monotonic, clock.sleep)

    result = wait_engine.wait_process(
        "chrome",
        timeout=1,
        poll_interval=0.5,
        cancelled=lambda: True,
    )

    assert result.completed is False
    assert result.timed_out is False
    assert result.error == "Espera cancelada."


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
    assert result.workflow_name == "open_vscode_workspace"
    assert [call[0] for call in executor.calls] == [
        "desktop.open_application",
        "desktop.list_processes",
        "desktop.list_windows",
        "desktop.list_windows",
        "desktop.list_windows",
        "desktop.bring_window_to_front",
        "desktop.get_foreground_window",
        "desktop.open_folder",
        "desktop.open_file",
        "desktop.list_windows",
        "desktop.list_windows",
        "desktop.bring_window_to_front",
    ]
    assert executor.calls[8][1].parameters["path"] == str(file.resolve())
    assert executor.calls[-1][1].parameters == {"handle": 10}


def test_prepare_atlas_workspace_accepts_configured_python_file(
    tmp_path: Path,
) -> None:
    file = tmp_path / "main.py"
    file.write_text("", encoding="utf-8")
    executor = FakeToolExecutor()
    use_case = PrepareAtlasWorkspaceUseCase(executor, ActionEngineUseCase())

    result = use_case.execute(tmp_path, "main.py")

    assert result.completed is True
    assert executor.calls[8][1].parameters["path"] == str(file.resolve())


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
    assert result.step_results[-1].step_name == "Abrir carpeta Atlas"
    assert executor.calls[-1][0] == "desktop.open_folder"


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
    assert result.step_results[-1].step_name == "Abrir archivo core/orchestrator.py"
    assert executor.calls[-1][0] == "desktop.open_file"


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
    assert result.executed_steps == 3
    assert result.successful_steps == 2
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
    assert any(call[0] == "desktop.get_foreground_window" for call in executor.calls)
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


def test_restart_application_executes_steps_in_order() -> None:
    executor = FakeToolExecutor()
    use_case = RestartApplicationUseCase(executor, ActionEngineUseCase())

    result = use_case.execute("chrome")

    assert result.completed is True
    assert result.workflow_name == "restart_application"
    assert [call[0] for call in executor.calls] == [
        "desktop.list_processes",
        "desktop.list_processes",
        "desktop.close_application",
        "desktop.list_processes",
        "desktop.open_application",
        "desktop.list_processes",
    ]
    assert all(call[0] != "desktop.terminate_process" for call in executor.calls)


def test_restart_application_stops_on_first_failure() -> None:
    executor = FakeToolExecutor()
    executor.fail_tool = "desktop.close_application"
    use_case = RestartApplicationUseCase(executor, ActionEngineUseCase())

    result = use_case.execute("chrome")

    assert result.completed is False
    assert result.stopped_early is True
    assert [call[0] for call in executor.calls] == [
        "desktop.list_processes",
        "desktop.list_processes",
        "desktop.close_application",
    ]


def test_restart_application_does_not_force_close_with_multiple_matches() -> None:
    executor = FakeToolExecutor()
    executor.processes.append(
        {
            "pid": 21,
            "name": "chrome.exe",
            "executable_path": None,
            "window_titles": (),
            "is_running": True,
        }
    )
    use_case = RestartApplicationUseCase(executor, ActionEngineUseCase())

    result = use_case.execute("chrome")

    assert result.completed is False
    assert result.step_results[-1].success is False
    assert "Varias coincidencias" in str(result.step_results[-1].error)
    assert all(call[0] != "desktop.terminate_process" for call in executor.calls)


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
        assert "Pasos ejecutados: 10" in result
        assert [call[0] for call in executor.calls] == [
            "desktop.open_application",
            "desktop.list_processes",
            "desktop.list_windows",
            "desktop.list_windows",
            "desktop.list_windows",
            "desktop.bring_window_to_front",
            "desktop.get_foreground_window",
            "desktop.open_folder",
            "desktop.open_file",
            "desktop.list_windows",
            "desktop.list_windows",
            "desktop.bring_window_to_front",
        ]


def test_desktop_interaction_interprets_wait_commands(tmp_path: Path) -> None:
    file = tmp_path / "README.md"
    file.write_text("Atlas", encoding="utf-8")
    executor = FakeToolExecutor()
    executor.processes = [
        {
            "pid": 21,
            "name": "Code.exe",
            "executable_path": None,
            "window_titles": ("Visual Studio Code",),
            "is_running": True,
        }
    ]
    executor.foreground_window = {
        "handle": 10,
        "title": "orchestrator.py - Atlas - Visual Studio Code",
        "rect": (0, 0, 100, 100),
    }
    use_case = DesktopInteractionUseCase(executor, project_root=tmp_path)

    assert "process_exists" in str(use_case.execute("espera VS Code"))
    assert "process_exists" in str(use_case.execute("espera proceso Code"))
    assert "window_exists" in str(
        use_case.execute("espera ventana Visual Studio Code")
    )
    assert "active_window" in str(
        use_case.execute("espera ventana activa Visual Studio Code")
    )
    assert "file_exists" in str(use_case.execute("espera archivo README.md"))


def test_desktop_interaction_interprets_disappearance_wait() -> None:
    executor = FakeToolExecutor()
    executor.processes = []
    use_case = DesktopInteractionUseCase(executor)

    result = use_case.execute("espera hasta que desaparezca Chrome")

    assert "process_absent" in str(result)


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
    assert "Pasos ejecutados: 8" in result


def test_desktop_interaction_does_not_handle_free_sequences() -> None:
    executor = FakeToolExecutor()
    use_case = DesktopInteractionUseCase(executor)

    result = use_case.execute("abre VS Code y luego Chrome")

    assert result == "Error: No se admiten secuencias libres."
    assert executor.calls == []
