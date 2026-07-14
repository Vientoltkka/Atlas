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
    ) -> str:
        self.calls.append((tool_name, context))

        if tool_name == "desktop.get_screen_size":
            return (1920, 1080)

        return "ok"


def test_bootstrap_registers_desktop_tools() -> None:
    orchestrator = Bootstrap.build()
    executor = orchestrator._desktop_interaction._executor

    assert executor._registry.exists("desktop.open_application")
    assert executor._registry.exists("desktop.open_folder")
    assert executor._registry.exists("desktop.open_file")
    assert executor._registry.exists("desktop.type_text")
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


def test_desktop_interaction_runs_before_agent_routing() -> None:
    executor = FakeExecutor()
    use_case = DesktopInteractionUseCase(executor)

    response = use_case.execute("Abre Visual Studio Code")

    assert response == "\u2713 Visual Studio Code abierto."
    assert executor.calls[0][0] == "desktop.open_application"


def test_desktop_click_requires_confirmation_in_integration() -> None:
    executor = FakeExecutor()
    use_case = DesktopInteractionUseCase(executor)

    response = use_case.execute("haz clic en 500, 300", confirm=lambda _: "n")

    assert response == "Acción cancelada."
    assert [call[0] for call in executor.calls] == ["desktop.get_screen_size"]
