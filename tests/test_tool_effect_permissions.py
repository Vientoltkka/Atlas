from __future__ import annotations

import subprocess

import pytest

from core import windows_task
from tools.base_tool import BaseTool
from tools.effect_permissions import (
    ToolEffectPermissionPolicy,
    ToolPermissionDeniedError,
)
from tools.executor import ToolExecutor
from tools.registry import ToolRegistry
from tools.tool_context import ToolContext


class EffectTool(BaseTool):
    def __init__(self, name: str, permission: str) -> None:
        self._name = name
        self._permission = permission
        self.calls = 0

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return "Effectful fake tool."

    @property
    def required_permissions(self) -> tuple[str, ...]:
        return (self._permission,)

    def execute(self, context: ToolContext) -> str:
        self.calls += 1
        return "executed"


@pytest.mark.parametrize(
    "permission",
    ("filesystem.write", "subprocess.cli", "desktop.control"),
)
def test_effectful_tool_is_denied_until_explicitly_authorized(permission: str) -> None:
    tool = EffectTool("effect.tool", permission)
    registry = ToolRegistry()
    registry.register(tool)
    policy = ToolEffectPermissionPolicy()
    executor = ToolExecutor(registry, policy)

    with pytest.raises(ToolPermissionDeniedError):
        executor.execute(tool.name)
    assert tool.calls == 0

    authorization = executor.authorize(tool.name)
    assert executor.execute(tool.name, authorization=authorization) == "executed"
    assert tool.calls == 1

    with pytest.raises(ToolPermissionDeniedError):
        executor.execute(tool.name, authorization=authorization)


def test_windows_task_dispatch_requires_explicit_authorization(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []

    def fake_run(args, **kwargs):
        calls.append(args)
        return subprocess.CompletedProcess(args, 0)

    monkeypatch.setattr(windows_task.subprocess, "run", fake_run)
    policy = ToolEffectPermissionPolicy()

    with pytest.raises(ToolPermissionDeniedError):
        windows_task.dispatch("status", __import__("pathlib").Path("."), permission_policy=policy)
    assert calls == []

    authorization = policy.authorize("windows_task.status", ("windows.task",))
    assert windows_task.dispatch(
        "status",
        __import__("pathlib").Path("."),
        permission_policy=policy,
        authorization=authorization,
    ) == 0
    assert calls == [["schtasks", "/Query", "/TN", windows_task.TASK_NAME]]
