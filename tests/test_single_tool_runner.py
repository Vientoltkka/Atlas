from __future__ import annotations

from typing import Any

import pytest

from bootstrap.bootstrap import Bootstrap
from tools.argument_schema import (
    ArgumentField,
    ArgumentSchema,
    ArgumentSchemaRegistry,
    ArgumentValidator,
)
from tools.base_tool import BaseTool
from tools.executor import ToolExecutor
from tools.intent_selector import (
    ToolIntent,
    ToolIntentRegistry,
    ToolSelector,
)
from tools.registry import ToolRegistry
from tools.single_tool_runner import SingleToolRunner, ToolRunResult, ValidatedToolRequest
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
    def __init__(
        self,
        result: Any = "executor-result",
        fail: bool = False,
    ) -> None:
        self.result = result
        self.fail = fail
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

        if self.fail:
            raise RuntimeError("executor failed")

        return self.result


class CountingSelector:
    def __init__(self, selector: ToolSelector) -> None:
        self._selector = selector
        self.calls = 0

    def select(self, intent: ToolIntent):
        self.calls += 1
        return self._selector.select(intent)


class CountingValidator:
    def __init__(self, validator: ArgumentValidator) -> None:
        self._validator = validator
        self.calls = 0

    def validate(self, selection):
        self.calls += 1
        return self._validator.validate(selection)


class ExplodingSelector:
    def select(self, intent: ToolIntent):
        raise AssertionError("selector broke")


def _build_runner(
    tool: BaseTool | None = None,
    executor: Any | None = None,
    register_tool: bool = True,
    register_schema: bool = True,
    register_mapping: bool = True,
    schema_allows_none: bool = False,
) -> tuple[SingleToolRunner, BaseTool | None, Any, CountingSelector, CountingValidator]:
    tool_registry = ToolRegistry()
    active_tool = tool or CountingTool()
    if register_tool:
        tool_registry.register(active_tool)

    intent_registry = ToolIntentRegistry()
    if register_mapping:
        intent_registry.register("demo.run", active_tool.name)

    schema_registry = ArgumentSchemaRegistry()
    if register_schema:
        schema_registry.register(
            ArgumentSchema(
                "demo.run",
                (
                    ArgumentField("path", str, required=True, allow_none=schema_allows_none),
                    ArgumentField("mode", str, default="safe"),
                ),
            )
        )

    selector = CountingSelector(ToolSelector(tool_registry, intent_registry))
    validator = CountingValidator(ArgumentValidator(schema_registry))
    active_executor = executor or ToolExecutor(tool_registry)

    return (
        SingleToolRunner(selector, validator, active_executor),
        active_tool,
        active_executor,
        selector,
        validator,
    )


def test_runner_returns_uniform_success_and_preserves_raw_result() -> None:
    raw_result = {"ok": True, "items": [1, 2]}
    runner, tool, _executor, selector, validator = _build_runner(
        tool=CountingTool(result=raw_result)
    )

    outcome = runner.run(ToolIntent("demo.run", {"path": "README.md"}))

    assert isinstance(outcome, ToolRunResult)
    assert outcome.success is True
    assert outcome.status == "success"
    assert outcome.error_code is None
    assert outcome.tool_name == "demo.tool"
    assert dict(outcome.original_arguments) == {"path": "README.md"}
    assert dict(outcome.validated_arguments) == {"path": "README.md", "mode": "safe"}
    assert outcome.executed is True
    assert outcome.execution_count == 1
    assert outcome.result is raw_result
    assert isinstance(tool, CountingTool)
    assert tool.calls == 1
    assert selector.calls == 1
    assert validator.calls == 1
    assert runner.execution_count == 1


def test_runner_build_request_still_returns_validated_request_without_execution() -> None:
    runner, tool, _executor, _selector, _validator = _build_runner()

    request = runner.build_request(ToolIntent("demo.run", {"path": "README.md"}))

    assert isinstance(request, ValidatedToolRequest)
    assert request.tool_name == "demo.tool"
    assert dict(request.original_arguments) == {"path": "README.md"}
    assert dict(request.validated_arguments) == {"path": "README.md", "mode": "safe"}
    assert request.validated is True
    assert request.executed is False
    assert isinstance(tool, CountingTool)
    assert tool.calls == 0


def test_runner_calls_tool_executor_once_with_validated_arguments() -> None:
    executor = CountingExecutor()
    runner, _tool, active_executor, _selector, _validator = _build_runner(executor=executor)

    outcome = runner.run(ToolIntent("demo.run", {"path": "README.md"}))

    assert outcome.success is True
    assert outcome.result == "executor-result"
    assert active_executor.calls == 1
    assert active_executor.tool_names == ["demo.tool"]
    assert dict(active_executor.contexts[0].parameters) == {"path": "README.md", "mode": "safe"}


@pytest.mark.parametrize(
    ("arguments", "status", "field", "message"),
    (
        ({}, "missing_argument", "path", "required argument is missing"),
        ({"path": 123}, "invalid_argument_type", "path", "expected str, got int"),
        ({"path": "README.md", "unknown": "value"}, "unexpected_argument", "unknown", "unexpected argument"),
        ({"path": None}, "none_not_allowed", "path", "None is not allowed"),
    ),
)
def test_runner_returns_uniform_validation_errors(
    arguments: dict,
    status: str,
    field: str,
    message: str,
) -> None:
    runner, tool, _executor, selector, validator = _build_runner()

    outcome = runner.run(ToolIntent("demo.run", arguments))

    assert outcome.success is False
    assert outcome.status == status
    assert outcome.error_code == status
    assert outcome.error_field == field
    assert outcome.error_message == message
    assert outcome.tool_name == "demo.tool"
    assert outcome.executed is False
    assert outcome.execution_count == 0
    assert outcome.result is None
    assert dict(outcome.original_arguments) == arguments
    assert outcome.validated_arguments is None
    assert isinstance(tool, CountingTool)
    assert tool.calls == 0
    assert selector.calls == 1
    assert validator.calls == 1


def test_runner_returns_unknown_intent_without_execution() -> None:
    runner, tool, _executor, selector, validator = _build_runner(register_mapping=False)

    outcome = runner.run(ToolIntent("demo.unknown", {"path": "README.md"}))

    assert outcome.success is False
    assert outcome.status == "unknown_intent"
    assert outcome.executed is False
    assert outcome.execution_count == 0
    assert outcome.result is None
    assert dict(outcome.original_arguments) == {"path": "README.md"}
    assert isinstance(tool, CountingTool)
    assert tool.calls == 0
    assert selector.calls == 1
    assert validator.calls == 0


def test_runner_returns_tool_not_registered_without_execution() -> None:
    runner, _tool, executor, selector, validator = _build_runner(
        executor=CountingExecutor(),
        register_tool=False,
    )

    outcome = runner.run(ToolIntent("demo.run", {"path": "README.md"}))

    assert outcome.success is False
    assert outcome.status == "tool_not_registered"
    assert outcome.executed is False
    assert outcome.execution_count == 0
    assert outcome.result is None
    assert executor.calls == 0
    assert selector.calls == 1
    assert validator.calls == 0


def test_runner_returns_schema_not_registered_without_execution() -> None:
    runner, tool, _executor, selector, validator = _build_runner(register_schema=False)

    outcome = runner.run(ToolIntent("demo.run", {"path": "README.md"}))

    assert outcome.success is False
    assert outcome.status == "schema_not_registered"
    assert outcome.tool_name == "demo.tool"
    assert outcome.executed is False
    assert outcome.execution_count == 0
    assert outcome.result is None
    assert isinstance(tool, CountingTool)
    assert tool.calls == 0
    assert selector.calls == 1
    assert validator.calls == 1


def test_runner_returns_tool_execution_error_when_executor_fails() -> None:
    executor = CountingExecutor(fail=True)
    runner, _tool, active_executor, _selector, _validator = _build_runner(executor=executor)

    outcome = runner.run(ToolIntent("demo.run", {"path": "README.md"}))

    assert outcome.success is False
    assert outcome.status == "tool_execution_error"
    assert outcome.error_code == "tool_execution_error"
    assert outcome.error_message == "executor failed"
    assert outcome.exception_type == "RuntimeError"
    assert outcome.executed is True
    assert outcome.execution_count == 1
    assert outcome.result is None
    assert active_executor.calls == 1
    assert runner.execution_count == 1


def test_runner_returns_tool_execution_error_when_tool_execute_fails() -> None:
    failing_tool = FailingTool()
    runner, tool, _executor, _selector, _validator = _build_runner(tool=failing_tool)

    outcome = runner.run(ToolIntent("demo.run", {"path": "README.md"}))

    assert outcome.success is False
    assert outcome.status == "tool_execution_error"
    assert outcome.error_message == "tool failed"
    assert outcome.executed is True
    assert outcome.execution_count == 1
    assert outcome.result is None
    assert tool is failing_tool
    assert failing_tool.calls == 1
    assert runner.execution_count == 1


def test_runner_returns_internal_error_for_unexpected_pre_execution_error() -> None:
    _runner, _tool, executor, _selector, validator = _build_runner(
        executor=CountingExecutor()
    )
    runner = SingleToolRunner(ExplodingSelector(), validator, executor)

    outcome = runner.run(ToolIntent("demo.run", {"path": "README.md"}))

    assert outcome.success is False
    assert outcome.status == "internal_error"
    assert outcome.error_code == "internal_error"
    assert outcome.exception_type == "AssertionError"
    assert outcome.executed is False
    assert outcome.execution_count == 0
    assert outcome.result is None
    assert executor.calls == 0


def test_runner_does_not_retry_or_fallback() -> None:
    failing_tool = FailingTool()
    runner, tool, _executor, selector, validator = _build_runner(tool=failing_tool)

    first = runner.run(ToolIntent("demo.run", {"path": "README.md"}))

    assert first.status == "tool_execution_error"
    assert tool is failing_tool
    assert failing_tool.calls == 1
    assert selector.calls == 1
    assert validator.calls == 1
    assert runner.execution_count == 1


def test_result_mappings_are_read_only() -> None:
    runner, _tool, _executor, _selector, _validator = _build_runner()

    outcome = runner.run(ToolIntent("demo.run", {"path": "README.md"}))

    with pytest.raises(TypeError):
        outcome.validated_arguments["other"] = "value"  # type: ignore[index]


def test_bootstrap_builds_single_tool_runner() -> None:
    runner = Bootstrap.build_single_tool_runner()

    assert isinstance(runner, SingleToolRunner)


def test_bootstrap_runner_executes_safe_file_read_once() -> None:
    runner = Bootstrap.build_single_tool_runner()

    outcome = runner.run(ToolIntent("file.read", {"path": "README.md"}))

    assert outcome.success is True
    assert outcome.status == "success"
    assert isinstance(outcome.result, str)
    assert outcome.result
    assert outcome.execution_count == 1
    assert runner.last_request is not None
    assert runner.last_request.tool_name == "read_file"
    assert dict(runner.last_request.validated_arguments) == {"path": "README.md"}
