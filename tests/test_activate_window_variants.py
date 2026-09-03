from __future__ import annotations

import pytest

from bootstrap.bootstrap import Bootstrap
from core.orchestrator import AtlasOrchestrator
from tools.desktop.desktop_tools import BringWindowToFrontTool, ListWindowsTool
from tools.executor import ToolExecutor
from tools.registry import ToolRegistry
from use_cases.desktop_interaction import DesktopInteractionUseCase
from use_cases.execution_conversation import ExecutionConversationController


class _DesktopController:
    def __init__(self) -> None:
        self.brought_to_front: list[int] = []

    def list_windows(self) -> list[dict[str, object]]:
        return [
            {
                "handle": 614,
                "title": "Sin título: Bloc de notas",
                "process_id": 10,
            },
        ]

    def bring_window_to_front(self, handle: int) -> None:
        self.brought_to_front.append(handle)


class _Planner:
    def create_plan(self, _prompt: str) -> dict[str, str]:
        return {}


class _Router:
    def route(self, _plan: dict[str, str]) -> str:
        return "chat"


class _ModelManager:
    def choose_model(self, _agent_name: str) -> str:
        return "test"


class _Memory:
    def add_user(self, _text: str) -> None:
        return None

    def add_assistant(self, _text: str) -> None:
        return None

    def history(self) -> list[dict[str, str]]:
        return []


class _Registry:
    def get(self, _name: str):
        return None


class _WriteFile:
    def execute(self, _path: str, _content: str) -> str:
        return "ok"


def _build_orchestrator() -> tuple[AtlasOrchestrator, _DesktopController]:
    desktop = _DesktopController()
    registry = ToolRegistry()
    registry.register(ListWindowsTool(desktop))
    registry.register(BringWindowToFrontTool(desktop))
    executor = ToolExecutor(registry)
    controller = ExecutionConversationController(
        Bootstrap.build_execution_coordinator(
            tool_registry=registry,
            executor=executor,
        ),
    )
    orchestrator = AtlasOrchestrator(
        planner=_Planner(),
        router=_Router(),
        model_manager=_ModelManager(),
        memory=_Memory(),
        registry=_Registry(),
        write_file=_WriteFile(),
        desktop_interaction=DesktopInteractionUseCase(executor),
        execution_conversation=controller,
    )
    orchestrator._handle_validated_capability_closure = lambda _prompt: None  # noqa: SLF001
    return orchestrator, desktop


@pytest.mark.parametrize(
    "prompt",
    ["ve a Bloc de notas", "cambia a Bloc de notas", "pon Bloc de notas"],
)
def test_activation_variants_confirm_and_activate_the_resolved_handle_once(
    prompt: str,
) -> None:
    orchestrator, desktop = _build_orchestrator()

    pending = orchestrator.process_prompt(
        prompt,
        confirm=lambda _prompt: "",
    )

    assert pending == "Voy a activar 'Sin título: Bloc de notas'. ¿Confirmas?"
    assert desktop.brought_to_front == []

    confirmed = orchestrator.process_prompt("sí", confirm=lambda _prompt: "")

    assert "Ventana activada." in confirmed
    assert desktop.brought_to_front == [614]
