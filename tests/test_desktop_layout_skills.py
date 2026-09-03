from __future__ import annotations

from pathlib import Path

import pytest

from bootstrap.skill_system import (
    DESKTOP_SKILLS_ROOT,
    build_builtin_skill_handler_registry,
    build_core_skill_system,
    register_desktop_skills,
)
from core.skill_registry import SkillExecutionTargetType
from core.skill_resolver import SkillResolutionRequest, SkillResolutionStatus
from tools.desktop.desktop_tools import (
    BringWindowToFrontTool,
    GetForegroundWindowTool,
    GetScreenSizeTool,
    ListWindowsTool,
    MoveResizeWindowTool,
)
from tools.executor import ToolExecutor
from tools.registry import ToolRegistry
from use_cases.desktop_layout_skills import WindowLayoutSkills


SCREEN = (1920, 1080)


class FakeLayoutController:
    def __init__(self) -> None:
        self.windows: list[dict[str, object]] = []
        self.front: list[int] = []
        self.resized: list[tuple[int, int, int, int, int]] = []

    def list_windows(self) -> list[dict[str, object]]:
        return list(self.windows)

    def get_foreground_window(self) -> dict[str, object]:
        return dict(self.windows[0])

    def get_screen_size(self) -> tuple[int, int]:
        return SCREEN

    def get_virtual_desktop_rect(self) -> tuple[int, int, int, int]:
        return (0, 0, SCREEN[0], SCREEN[1])

    def bring_window_to_front(self, handle: int) -> None:
        self.front.append(handle)

    def move_resize_window(self, handle: int, x: int, y: int, width: int, height: int) -> None:
        self.resized.append((handle, x, y, width, height))


def _executor(controller: FakeLayoutController) -> ToolExecutor:
    registry = ToolRegistry()
    for tool in (
        ListWindowsTool(controller),
        GetForegroundWindowTool(controller),
        GetScreenSizeTool(controller),
        BringWindowToFrontTool(controller),
        MoveResizeWindowTool(controller),
    ):
        registry.register(tool)
    return ToolExecutor(registry)


def _controller() -> FakeLayoutController:
    controller = FakeLayoutController()
    controller.windows.extend(
        [
            {"handle": 11, "title": "Documento - Bloc de notas"},
            {"handle": 22, "title": "Atlas - Navegador"},
        ]
    )
    return controller


def _handler(executor: ToolExecutor) -> WindowLayoutSkills:
    return WindowLayoutSkills(executor)


def test_modo_trabajo_snaps_titled_window_to_left_half() -> None:
    controller = _controller()

    output = _handler(_executor(controller)).modo_trabajo(
        {"window_title": "bloc de notas"}
    )

    assert output == {
        "result": "Modo trabajo listo: 'Documento - Bloc de notas' anclada a la mitad izquierda."
    }
    assert controller.front == [11]
    assert controller.resized == [(11, 0, 0, 960, 1080)]


def test_modo_trabajo_without_title_uses_foreground_window() -> None:
    controller = _controller()

    output = _handler(_executor(controller)).modo_trabajo({})

    assert controller.resized == [(11, 0, 0, 960, 1080)]
    assert "Modo trabajo listo" in output["result"]


def test_modo_trabajo_with_unknown_window_raises_and_never_moves() -> None:
    controller = _controller()

    with pytest.raises(ValueError, match="No se encontraron ventanas"):
        _handler(_executor(controller)).modo_trabajo(
            {"window_title": "inexistente"}
        )

    assert controller.resized == []
    assert controller.front == []


def test_modo_trabajo_with_ambiguous_window_raises_and_never_moves() -> None:
    controller = _controller()
    controller.windows.append({"handle": 33, "title": "Otro - Bloc de notas"})

    with pytest.raises(ValueError, match="Varias ventanas"):
        _handler(_executor(controller)).modo_trabajo(
            {"window_title": "bloc"}
        )

    assert controller.resized == []


def test_layout_handler_without_tool_executor_fails_safe() -> None:
    with pytest.raises(RuntimeError, match="tool executor is unavailable"):
        WindowLayoutSkills(None).modo_trabajo({})


def test_modo_escritura_centers_window_with_focus_size() -> None:
    controller = _controller()

    output = _handler(_executor(controller)).modo_escritura(
        {"window_title": "navegador"}
    )

    assert output == {
        "result": "Modo escritura listo: 'Atlas - Navegador' centrada en 900 x 700."
    }
    assert controller.front == [22]
    assert controller.resized == [(22, 510, 190, 900, 700)]


def test_desktop_manifests_are_discovered_and_registered() -> None:
    system = build_core_skill_system(
        skill_handler_registry=build_builtin_skill_handler_registry()
    )

    registration = register_desktop_skills(system)

    assert registration.status.value == "COMPLETED"
    assert registration.registered_skill_ids == (
        "skill.modo-escritura",
        "skill.modo-trabajo",
    )
    skill = system.skill_registry.get("skill.modo-trabajo")
    assert skill.execution_target_type is SkillExecutionTargetType.HANDLER
    assert skill.handler_id == "handler.modo-trabajo"
    resolution = system.skill_resolver.resolve(
        SkillResolutionRequest(required_skill_ids=("skill.modo-trabajo",))
    )
    assert resolution.status is SkillResolutionStatus.RESOLVED


def test_desktop_skill_root_contains_the_manifest() -> None:
    manifest = Path(DESKTOP_SKILLS_ROOT) / "modo_trabajo" / "skill.json"

    assert manifest.is_file()
