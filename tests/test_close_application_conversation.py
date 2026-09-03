from pathlib import Path

from bootstrap.bootstrap import Bootstrap
from core.orchestrator import AtlasOrchestrator
from tools.desktop.desktop_tools import (
    CloseApplicationTool,
    CloseWindowTool,
    ListProcessesTool,
    ListWindowsTool,
)
from tools.desktop.windows_controller import ProcessInfo
from tools.executor import ToolExecutor
from tools.registry import ToolRegistry
from use_cases.desktop_interaction import DesktopInteractionUseCase
from use_cases.execution_conversation import ExecutionConversationController


class FakeProcessController:
    def __init__(self, processes: list[ProcessInfo]) -> None:
        self._processes = processes
        self.close_requests: list[int] = []

    def list_processes(self, query: str) -> list[ProcessInfo]:
        normalized = query.strip().lower()
        candidates = {normalized}
        if normalized in {"calculadora", "calculator"}:
            candidates.add("calculatorapp")
        return [
            process
            for process in self._processes
            if any(
                candidate in process.name.lower()
                or any(
                    candidate in title.lower()
                    for title in process.window_titles
                )
                for candidate in candidates
            )
        ]

    def close_process_windows(self, pid: int) -> int:
        self.close_requests.append(pid)
        return 1


class FakeUwpController(FakeProcessController):
    def __init__(
        self,
        processes: list[ProcessInfo],
        windows: list[dict[str, object]],
    ) -> None:
        super().__init__(processes)
        self._windows = windows
        self.close_window_requests: list[int] = []

    def list_windows(self) -> list[dict[str, object]]:
        return self._windows

    def close_window(self, handle: int) -> None:
        self.close_window_requests.append(handle)


def _process(pid: int, name: str, *titles: str) -> ProcessInfo:
    return ProcessInfo(
        pid=pid,
        name=name,
        executable_path=None,
        window_titles=tuple(titles),
        is_running=True,
    )


def _close_orchestrator(
    controller: FakeProcessController,
) -> tuple[AtlasOrchestrator, ExecutionConversationController]:
    registry = ToolRegistry()
    registry.register(ListProcessesTool(controller))
    registry.register(CloseApplicationTool(controller))
    executor = ToolExecutor(registry)
    conversation = ExecutionConversationController(
        Bootstrap.build_execution_coordinator(tool_registry=registry, executor=executor)
    )
    orchestrator = AtlasOrchestrator(
        planner=None,
        router=None,
        model_manager=None,
        memory=None,
        registry=None,
        write_file=None,
        desktop_interaction=DesktopInteractionUseCase(executor),
        execution_conversation=conversation,
    )
    return orchestrator, conversation


def test_close_application_asks_confirmation_then_closes_on_yes() -> None:
    controller = FakeProcessController(
        [_process(4321, "Calculator.exe", "Calculadora")]
    )
    orchestrator, conversation = _close_orchestrator(controller)

    pending = orchestrator.process_prompt(
        "cierra calculadora",
        confirm=lambda _: "",
    )

    assert "Voy a cerrar Calculator.exe - PID 4321" in pending
    assert controller.close_requests == []
    assert conversation.pending_confirmation_id is not None
    assert conversation.pending_confirmation_context is not None

    confirmation = orchestrator.process_prompt("sí", confirm=lambda _: "")

    assert controller.close_requests == [4321]
    assert "cierre" in confirmation.lower()
    assert conversation.pending_confirmation_id is None


def test_close_application_cancels_on_no_without_closing() -> None:
    controller = FakeProcessController(
        [_process(4321, "Calculator.exe", "Calculadora")]
    )
    orchestrator, conversation = _close_orchestrator(controller)

    orchestrator.process_prompt("cierra calculadora", confirm=lambda _: "")

    rejection = orchestrator.process_prompt("no", confirm=lambda _: "")

    assert controller.close_requests == []
    assert "cancel" in rejection.lower()
    assert conversation.pending_confirmation_id is None


def test_close_application_with_atlas_prefix_asks_confirmation_then_closes_on_yes() -> None:
    controller = FakeProcessController(
        [_process(4321, "Calculator.exe", "Calculadora")]
    )
    orchestrator, conversation = _close_orchestrator(controller)

    pending = orchestrator.process_prompt(
        "atlas cierra calculadora",
        confirm=lambda _: "",
    )

    assert "Voy a cerrar Calculator.exe - PID 4321" in pending
    assert controller.close_requests == []
    assert conversation.pending_confirmation_id is not None

    confirmation = orchestrator.process_prompt("sí", confirm=lambda _: "")

    assert controller.close_requests == [4321]
    assert "cierre" in confirmation.lower()


def test_close_application_with_atlas_prefix_cancels_on_no() -> None:
    controller = FakeProcessController(
        [_process(4321, "Calculator.exe", "Calculadora")]
    )
    orchestrator, conversation = _close_orchestrator(controller)

    orchestrator.process_prompt("atlas cierra calculadora", confirm=lambda _: "")

    rejection = orchestrator.process_prompt("no", confirm=lambda _: "")

    assert controller.close_requests == []
    assert "cancel" in rejection.lower()
    assert conversation.pending_confirmation_id is None


def test_close_application_with_article_asks_confirmation_then_closes_on_yes() -> None:
    controller = FakeProcessController(
        [_process(4321, "Calculator.exe", "Calculadora")]
    )
    orchestrator, conversation = _close_orchestrator(controller)

    pending = orchestrator.process_prompt(
        "atlas cierra la calculadora",
        confirm=lambda _: "",
    )

    assert "Voy a cerrar Calculator.exe - PID 4321" in pending
    assert "No se encontraron ventanas" not in pending
    assert controller.close_requests == []
    assert conversation.pending_confirmation_id is not None

    confirmation = orchestrator.process_prompt("sí", confirm=lambda _: "")

    assert controller.close_requests == [4321]
    assert "cierre" in confirmation.lower()


def test_close_application_with_article_cancels_on_no() -> None:
    controller = FakeProcessController(
        [_process(4321, "Calculator.exe", "Calculadora")]
    )
    orchestrator, conversation = _close_orchestrator(controller)

    orchestrator.process_prompt("atlas cierra la calculadora", confirm=lambda _: "")

    rejection = orchestrator.process_prompt("no", confirm=lambda _: "")

    assert controller.close_requests == []
    assert "cancel" in rejection.lower()
    assert conversation.pending_confirmation_id is None


def test_close_application_without_matches_falls_back_to_legacy_routing() -> None:
    controller = FakeProcessController([])
    registry = ToolRegistry()
    registry.register(ListProcessesTool(controller))
    use_case = DesktopInteractionUseCase(ToolExecutor(registry))

    assert use_case.close_application_tool_request("cierra calculadora") is None


def test_close_application_ignores_mass_close_orders() -> None:
    controller = FakeProcessController(
        [_process(4321, "Calculator.exe", "Calculadora")]
    )
    registry = ToolRegistry()
    registry.register(ListProcessesTool(controller))
    use_case = DesktopInteractionUseCase(ToolExecutor(registry))

    assert use_case.close_application_tool_request("cierra todo") is None
    assert use_case.close_application_tool_request("abre calculadora") is None


def _uwp_orchestrator(
    controller: FakeUwpController,
) -> tuple[AtlasOrchestrator, ExecutionConversationController]:
    registry = ToolRegistry()
    registry.register(ListProcessesTool(controller))
    registry.register(CloseApplicationTool(controller))
    registry.register(ListWindowsTool(controller))
    registry.register(CloseWindowTool(controller))
    executor = ToolExecutor(registry)
    conversation = ExecutionConversationController(
        Bootstrap.build_execution_coordinator(tool_registry=registry, executor=executor)
    )
    orchestrator = AtlasOrchestrator(
        planner=None,
        router=None,
        model_manager=None,
        memory=None,
        registry=None,
        write_file=None,
        desktop_interaction=DesktopInteractionUseCase(executor),
        execution_conversation=conversation,
    )
    return orchestrator, conversation


def test_close_uwp_application_by_visible_window_on_yes() -> None:
    controller = FakeUwpController(
        [_process(4321, "CalculatorApp.exe")],
        [
            {
                "handle": 555,
                "title": "Calculadora",
                "process_id": 999,
            }
        ],
    )
    orchestrator, conversation = _uwp_orchestrator(controller)

    pending = orchestrator.process_prompt(
        "cierra calculadora",
        confirm=lambda _: "",
    )

    assert "Voy a cerrar la ventana 'Calculadora'" in pending
    assert "CalculatorApp.exe - PID 4321" in pending
    assert controller.close_requests == []
    assert controller.close_window_requests == []
    assert conversation.pending_confirmation_id is not None

    confirmation = orchestrator.process_prompt("sí", confirm=lambda _: "")

    assert controller.close_window_requests == [555]
    assert controller.close_requests == []
    assert "cierre" in confirmation.lower()
    assert conversation.pending_confirmation_id is None


def test_close_uwp_application_cancels_on_no_without_closing() -> None:
    controller = FakeUwpController(
        [_process(4321, "CalculatorApp.exe")],
        [
            {
                "handle": 555,
                "title": "Calculadora",
                "process_id": 999,
            }
        ],
    )
    orchestrator, _ = _uwp_orchestrator(controller)

    orchestrator.process_prompt("cierra calculadora", confirm=lambda _: "")

    rejection = orchestrator.process_prompt("no", confirm=lambda _: "")

    assert controller.close_window_requests == []
    assert controller.close_requests == []
    assert "cancel" in rejection.lower()


def test_close_uwp_multiple_nominal_processes_uses_single_visible_window() -> None:
    controller = FakeUwpController(
        [_process(4321, "CalculatorApp.exe"), _process(4322, "CalculatorApp.exe")],
        [
            {
                "handle": 555,
                "title": "Calculadora",
                "process_id": 999,
            }
        ],
    )
    orchestrator, conversation = _uwp_orchestrator(controller)

    pending = orchestrator.process_prompt(
        "cierra calculadora",
        confirm=lambda _: "",
    )

    assert "Voy a cerrar la ventana 'Calculadora'" in pending
    assert conversation.pending_confirmation_id is not None

    confirmation = orchestrator.process_prompt("sí", confirm=lambda _: "")

    assert controller.close_window_requests == [555]
    assert controller.close_requests == []
    assert "cierre" in confirmation.lower()


def test_close_uwp_process_without_matching_window_keeps_pid_request() -> None:
    controller = FakeUwpController(
        [_process(4321, "CalculatorApp.exe")],
        [
            {
                "handle": 555,
                "title": "Otra ventana",
                "process_id": 999,
            }
        ],
    )
    orchestrator, _ = _uwp_orchestrator(controller)

    pending = orchestrator.process_prompt(
        "cierra calculadora",
        confirm=lambda _: "",
    )

    assert "Voy a cerrar CalculatorApp.exe - PID 4321" in pending
    assert "Calculadora" not in pending
