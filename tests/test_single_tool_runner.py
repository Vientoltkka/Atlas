from __future__ import annotations

from typing import Any

import pytest

from bootstrap.bootstrap import Bootstrap
from tools.argument_schema import (
    ArgumentField,
    ArgumentSchema,
    ArgumentSchemaNotRegisteredError,
    ArgumentSchemaRegistry,
    ArgumentValidationError,
    ArgumentValidator,
)
from tools.base_tool import BaseTool
from tools.executor import ToolExecutor
from tools.intent_selector import (
    ToolIntent,
    ToolIntentNotSupportedError,
    ToolIntentRegistry,
    ToolSelector,
)
from tools.registry import ToolNotRegisteredError, ToolRegistry
from tools.single_tool_runner import SingleToolRunner, ValidatedToolRequest
from tools.tool_context import ToolContext


class CountingTool(BaseTool):
    def __init__(
        self,
        name: str = "demo.tool",
        result: Any = "raw-result",
    ) -> None:
        self._name = name
        self._result = result
        self.calls = 0
        self.contexts: list[ToolContext] = []

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return "Counting tool."

    def execute(
        self,
        context: ToolContext,
    ) -> Any:
        self.calls += 1
        self.contexts.append(context)
        return self._result


class FailingTool(CountingTool):
    def execute(
        self,
        context: ToolContext,
    ) -> Any:
        self.calls += 1
        self.contexts.append(context)
        raise RuntimeError("tool failed")


class CountingExecutor:
    def __init__(self, result: Any = "executor-result") -> None:
        self.result = result
        self.calls = 0
        self.tool_names: list[str] = []
        self.contexts: list[ToolContext] = []

    def execute(
        self,
        tool_name: str,
        context: ToolContext,
    ) -> Any:
        self.calls += 1
        self.tool_names.append(tool_name)
        self.contexts.append(context)
        return self.result


def _build_runner(
    tool: BaseTool | None = None,
    executor: Any | None = None,
    register_schema: bool = True,
    register_mapping: bool = True,
) -> tuple[SingleToolRunner, BaseTool | None, Any]:
    tool_registry = ToolRegistry()
    active_tool = tool or CountingTool()
    if active_tool is not None:
        tool_registry.register(active_tool)

    intent_registry = ToolIntentRegistry()
    if register_mapping:
        intent_registry.register("demo.run", active_tool.name if active_tool else "demo.tool")

    schema_registry = ArgumentSchemaRegistry()
    if register_schema:
        schema_registry.register(
            ArgumentSchema(
                "demo.run",
                (
                    ArgumentField("path", str, required=True),
                    ArgumentField("mode", str, default="safe"),
                ),
            )
        )

    active_executor = executor or ToolExecutor(tool_registry)

    return (
        SingleToolRunner(
            ToolSelector(tool_registry, intent_registry),
            ArgumentValidator(schema_registry),
            active_executor,
        ),
        active_tool,
        active_executor,
    )


def test_runner_executes_one_valid_tool_once_and_returns_raw_result() -> None:
    runner, tool, _executor = _build_runner()

    result = runner.run(ToolIntent("demo.run", {"path": "README.md"}))

    assert result == "raw-result"
    assert isinstance(tool, CountingTool)
    assert tool.calls == 1
    assert dict(tool.contexts[0].parameters) == {"path": "README.md", "mode": "safe"}
    assert runner.execution_count == 1


def test_runner_uses_real_selector_validator_and_executor_with_validated_arguments() -> None:
    runner, tool, _executor = _build_runner()

    request = runner.build_request(ToolIntent("demo.run", {"path": "README.md"}))
    result = runner.run(ToolIntent("demo.run", {"path": "README.md"}))

    assert isinstance(request, ValidatedToolRequest)
    assert request.tool_name == "demo.tool"
    assert request.descriptor.name == "demo.tool"
    assert dict(request.original_arguments) == {"path": "README.md"}
    assert dict(request.validated_arguments) == {"path": "README.md", "mode": "safe"}
    assert request.validated is True
    assert request.executed is False
    assert result == "raw-result"
    assert isinstance(tool, CountingTool)
    assert dict(tool.contexts[-1].parameters) == {"path": "README.md", "mode": "safe"}


def test_runner_calls_tool_executor_once() -> None:
    executor = CountingExecutor()
    runner, _tool, active_executor = _build_runner(executor=executor)

    result = runner.run(ToolIntent("demo.run", {"path": "README.md"}))

    assert result == "executor-result"
    assert active_executor.calls == 1
    assert active_executor.tool_names == ["demo.tool"]
    assert dict(active_executor.contexts[0].parameters) == {"path": "README.md", "mode": "safe"}
    assert runner.execution_count == 1


def test_runner_does_not_execute_unknown_intent() -> None:
    runner, tool, _executor = _build_runner(register_mapping=False)

    with pytest.raises(ToolIntentNotSupportedError):
        runner.run(ToolIntent("demo.unknown", {"path": "README.md"}))

    assert isinstance(tool, CountingTool)
    assert tool.calls == 0
    assert runner.execution_count == 0


def test_runner_does_not_execute_when_mapped_tool_is_missing() -> None:
    tool_registry = ToolRegistry()
    intent_registry = ToolIntentRegistry()
    intent_registry.register("demo.run", "missing.tool")
    schema_registry = ArgumentSchemaRegistry()
    schema_registry.register(
        ArgumentSchema(
            "demo.run",
            (ArgumentField("path", str, required=True),),
        )
    )
    executor = CountingExecutor()
    runner = SingleToolRunner(
        ToolSelector(tool_registry, intent_registry),
        ArgumentValidator(schema_registry),
        executor,
    )

    with pytest.raises(ToolNotRegisteredError):
        runner.run(ToolIntent("demo.run", {"path": "README.md"}))

    assert executor.calls == 0
    assert runner.execution_count == 0


def test_runner_does_not_execute_when_schema_is_missing() -> None:
    runner, tool, _executor = _build_runner(register_schema=False)

    with pytest.raises(ArgumentSchemaNotRegisteredError):
        runner.run(ToolIntent("demo.run", {"path": "README.md"}))

    assert isinstance(tool, CountingTool)
    assert tool.calls == 0
    assert runner.execution_count == 0


@pytest.mark.parametrize(
    "arguments",
    (
        {},
        {"path": 123},
        {"path": "README.md", "unknown": "value"},
    ),
)
def test_runner_does_not_execute_when_validation_fails(arguments: dict) -> None:
    runner, tool, _executor = _build_runner()

    with pytest.raises(ArgumentValidationError):
        runner.run(ToolIntent("demo.run", arguments))

    assert isinstance(tool, CountingTool)
    assert tool.calls == 0
    assert runner.execution_count == 0


def test_runner_does_not_retry_or_fallback_when_tool_raises() -> None:
    failing_tool = FailingTool()
    runner, tool, _executor = _build_runner(tool=failing_tool)

    with pytest.raises(RuntimeError, match="tool failed"):
        runner.run(ToolIntent("demo.run", {"path": "README.md"}))

    assert tool is failing_tool
    assert failing_tool.calls == 1
    assert runner.execution_count == 0


def test_bootstrap_builds_single_tool_runner() -> None:
    runner = Bootstrap.build_single_tool_runner()

    assert isinstance(runner, SingleToolRunner)


def test_bootstrap_runner_executes_safe_file_read_once() -> None:
    runner = Bootstrap.build_single_tool_runner()

    result = runner.run(ToolIntent("file.read", {"path": "README.md"}))

    assert isinstance(result, str)
    assert result
    assert runner.execution_count == 1
    assert runner.last_request is not None
    assert runner.last_request.tool_name == "read_file"
    assert dict(runner.last_request.validated_arguments) == {"path": "README.md"}
