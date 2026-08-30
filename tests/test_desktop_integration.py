from bootstrap.bootstrap import Bootstrap
from tools.tool_context import ToolContext
from use_cases.desktop_interaction import DesktopInteractionUseCase


class FakeExecutor:
    def __init__(self) -> None:
        self.calls: list[tuple[str, ToolContext]] = []

    def execute(
        self,
        tool_name: str,
        context: ToolContext,
    ) -> str | tuple[int, int] | list[dict[str, object]]:
        self.calls.append((tool_name, context))

        if tool_name == "desktop.get_screen_size":
            return (1920, 1080)

        if tool_name == "desktop.list_windows":
            return [
                {
                    "handle": 10,
                    "title": "Visual Studio Code - Atlas",
                    "rect": (20, 30, 820, 630),
                }
            ]

        if tool_name == "desktop.list_processes":
            return []

        return "ok"


def test_bootstrap_registers_desktop_tools() -> None:
    orchestrator = Bootstrap.build()
    executor = orchestrator._desktop_interaction._executor

    assert executor._registry.exists("desktop.open_application")
    assert executor._registry.exists("desktop.list_processes")
    assert executor._registry.exists("desktop.is_process_running")
    assert executor._registry.exists("desktop.get_process")
    assert executor._registry.exists("desktop.close_application")
    assert executor._registry.exists("desktop.terminate_process")
    assert executor._registry.exists("desktop.open_folder")
    assert executor._registry.exists("desktop.open_file")
    assert executor._registry.exists("desktop.type_text")
    assert executor._registry.exists("desktop.copy_clipboard_text")
    assert executor._registry.exists("desktop.read_clipboard_text")
    assert executor._registry.exists("desktop.clear_clipboard")
    assert executor._registry.exists("desktop.clipboard_has_text")
    assert executor._registry.exists("desktop.paste_clipboard")
    assert executor._registry.exists("desktop.press_hotkey")
    assert executor._registry.exists("desktop.save_file")
    assert executor._registry.exists("desktop.activate_window")
    assert executor._registry.exists("desktop.get_screen_size")
    assert executor._registry.exists("desktop.get_cursor_position")
    assert executor._registry.exists("desktop.move_cursor")
    assert executor._registry.exists("desktop.left_click")
    assert executor._registry.exists("desktop.double_click")
    assert executor._registry.exists("desktop.right_click")
    assert executor._registry.exists("desktop.scroll_vertical")
    assert executor._registry.exists("desktop.capture_screenshot")
    assert executor._registry.exists("desktop.list_windows")
    assert executor._registry.exists("desktop.get_window_rect")
    assert executor._registry.exists("desktop.get_foreground_window")
    assert executor._registry.exists("desktop.bring_window_to_front")
    assert executor._registry.exists("desktop.maximize_window")
    assert executor._registry.exists("desktop.minimize_window")
    assert executor._registry.exists("desktop.restore_window")
    assert executor._registry.exists("desktop.move_window")
    assert executor._registry.exists("desktop.resize_window")
    assert executor._registry.exists("desktop.move_resize_window")
    assert executor._registry.exists("desktop.close_window")


def test_desktop_interaction_runs_before_agent_routing() -> None:
    executor = FakeExecutor()
    use_case = DesktopInteractionUseCase(executor)

    response = use_case.execute("Abre Visual Studio Code")

    assert response == "\u2713 Abriendo Visual Studio Code."
    assert executor.calls[0][0] == "desktop.open_application"


def test_desktop_click_requires_confirmation_in_integration() -> None:
    executor = FakeExecutor()
    use_case = DesktopInteractionUseCase(executor)

    response = use_case.execute("haz clic en 500, 300", confirm=lambda _: "n")

    assert response == "Acción cancelada."
    assert [call[0] for call in executor.calls] == ["desktop.get_screen_size"]


def test_window_close_requires_confirmation_in_integration() -> None:
    executor = FakeExecutor()
    use_case = DesktopInteractionUseCase(executor)

    response = use_case.execute("cierra Visual Studio Code", confirm=lambda _: "n")

    assert response == "Acción cancelada."
    assert [call[0] for call in executor.calls] == [
        "desktop.list_processes",
        "desktop.list_windows",
    ]


def test_bootstrap_injects_restart_application_workflow() -> None:
    orchestrator = Bootstrap.build()

    assert orchestrator._desktop_interaction._restart_application is not None
