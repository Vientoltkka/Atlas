from __future__ import annotations

from typing import Any

from bootstrap.bootstrap import Bootstrap
from tools.argument_schema import (
    ArgumentField,
    ArgumentSchema,
    ArgumentSchemaRegistry,
    ArgumentValidator,
)
from tools.base_tool import BaseTool
from tools.executor import ToolExecutor
from tools.intent_selector import ToolIntentRegistry, ToolSelector
from tools.registry import ToolRegistry
from tools.single_tool_runner import SingleToolRunner
from tools.tool_chain_runner import ToolChainRunner, ToolChainStep
from tools.tool_context import ToolContext


class ChainTool(BaseTool):
    def __init__(
        self,
        name: str,
        calls: list[str],
        result: Any,
        *,
        dangerous: bool = False,
        fail: bool = False,
    ) -> None:
        self._name = name
        self._calls = calls
        self._result = result
        self._dangerous = dangerous
        self._fail = fail
        self.contexts: list[ToolContext] = []

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return f"Tool {self._name}."

    @property
    def requires_confirmation(self) -> bool:
        return self._dangerous

    def execute(
        self,
        context: ToolContext,
    ) -> Any:
        self._calls.append(self._name)
        self.contexts.append(context)

        if self._fail:
            raise RuntimeError(f"{self._name} failed")

        if callable(self._result):
            return self._result(context)

        return self._result


def _chain_runner(
    *,
    include_second: bool = True,
    fail_second: bool = False,
) -> tuple[ToolChainRunner, dict[str, ChainTool], list[str]]:
    calls: list[str] = []
    tools = {
        "first_tool": ChainTool(
            "first_tool",
            calls,
            {"content": "alpha", "nested": {"value": "deep"}},
        ),
        "second_tool": ChainTool(
            "second_tool",
            calls,
            lambda context: f"second:{context.parameters.get('value')}",
            fail=fail_second,
        ),
        "third_tool": ChainTool(
            "third_tool",
            calls,
            lambda context: f"third:{context.parameters.get('value')}",
        ),
        "danger_tool": ChainTool(
            "danger_tool",
            calls,
            lambda context: f"danger:{context.parameters.get('value')}",
            dangerous=True,
        ),
    }
    registry = ToolRegistry()
    registry.register(tools["first_tool"])
    if include_second:
        registry.register(tools["second_tool"])
    registry.register(tools["third_tool"])
    registry.register(tools["danger_tool"])

    intent_registry = ToolIntentRegistry()
    intent_registry.register("demo.first", "first_tool")
    intent_registry.register("demo.second", "second_tool")
    intent_registry.register("demo.third", "third_tool")
    intent_registry.register("demo.danger", "danger_tool")

    schema_registry = ArgumentSchemaRegistry()
    schema_registry.register(
        ArgumentSchema(
            "demo.first",
            (ArgumentField("value", str, default=""),),
        )
    )
    schema_registry.register(
        ArgumentSchema(
            "demo.second",
            (ArgumentField("value", str, required=True),),
        )
    )
    schema_registry.register(
        ArgumentSchema(
            "demo.third",
            (ArgumentField("value", str, required=True),),
        )
    )
    schema_registry.register(
        ArgumentSchema(
            "demo.danger",
            (ArgumentField("value", str, required=True),),
        )
    )

    single = SingleToolRunner(
        ToolSelector(registry, intent_registry),
        ArgumentValidator(schema_registry),
        ToolExecutor(registry),
    )

    return ToolChainRunner(single), tools, calls


def test_chain_runs_two_safe_tools_in_order() -> None:
    runner, _tools, calls = _chain_runner()

    outcome = runner.run(
        (
            ToolChainStep("first", "demo.first"),
            ToolChainStep("second", "demo.second", {"value": "manual"}),
        )
    )

    assert outcome.success is True
    assert outcome.status == "success"
    assert calls == ["first_tool", "second_tool"]
    assert outcome.execution_count == 2
    assert [step.step_id for step in outcome.steps] == ["first", "second"]


def test_chain_runs_three_safe_tools() -> None:
    runner, _tools, calls = _chain_runner()

    outcome = runner.run(
        (
            ToolChainStep("first", "demo.first"),
            ToolChainStep("second", "demo.second", {"value": "b"}),
            ToolChainStep("third", "demo.third", {"value": "c"}),
        )
    )

    assert outcome.success is True
    assert calls == ["first_tool", "second_tool", "third_tool"]
    assert outcome.execution_count == 3


def test_chain_reuses_previous_output_as_argument() -> None:
    runner, tools, _calls = _chain_runner()

    outcome = runner.run(
        (
            ToolChainStep("read", "demo.first"),
            ToolChainStep(
                "use",
                "demo.second",
                {"value": "${steps.read.output.content}"},
            ),
        )
    )

    assert outcome.success is True
    assert tools["second_tool"].contexts[0].parameters["value"] == "alpha"
    assert outcome.steps[1].result.result == "second:alpha"


def test_chain_rejects_missing_step_reference() -> None:
    runner, _tools, calls = _chain_runner()

    outcome = runner.run(
        (
            ToolChainStep(
                "use",
                "demo.second",
                {"value": "${steps.missing.output.content}"},
            ),
        )
    )

    assert outcome.success is False
    assert outcome.status == "reference_not_found"
    assert outcome.failed_step_id == "use"
    assert outcome.execution_count == 0
    assert calls == []


def test_chain_rejects_missing_reference_field() -> None:
    runner, _tools, calls = _chain_runner()

    outcome = runner.run(
        (
            ToolChainStep("read", "demo.first"),
            ToolChainStep(
                "use",
                "demo.second",
                {"value": "${steps.read.output.missing}"},
            ),
        )
    )

    assert outcome.success is False
    assert outcome.status == "reference_field_not_found"
    assert outcome.failed_step_id == "use"
    assert outcome.execution_count == 1
    assert calls == ["first_tool"]


def test_chain_stops_on_missing_tool() -> None:
    runner, _tools, calls = _chain_runner(include_second=False)

    outcome = runner.run(
        (
            ToolChainStep("first", "demo.first"),
            ToolChainStep("second", "demo.second", {"value": "b"}),
            ToolChainStep("third", "demo.third", {"value": "c"}),
        )
    )

    assert outcome.success is False
    assert outcome.status == "tool_not_registered"
    assert outcome.failed_step_id == "second"
    assert calls == ["first_tool"]


def test_chain_stops_on_invalid_arguments() -> None:
    runner, _tools, calls = _chain_runner()

    outcome = runner.run(
        (
            ToolChainStep("first", "demo.first"),
            ToolChainStep("second", "demo.second", {}),
            ToolChainStep("third", "demo.third", {"value": "c"}),
        )
    )

    assert outcome.success is False
    assert outcome.status == "missing_argument"
    assert outcome.failed_step_id == "second"
    assert calls == ["first_tool"]


def test_chain_stops_on_execution_failure() -> None:
    runner, _tools, calls = _chain_runner(fail_second=True)

    outcome = runner.run(
        (
            ToolChainStep("first", "demo.first"),
            ToolChainStep("second", "demo.second", {"value": "b"}),
            ToolChainStep("third", "demo.third", {"value": "c"}),
        )
    )

    assert outcome.success is False
    assert outcome.status == "tool_execution_error"
    assert outcome.failed_step_id == "second"
    assert calls == ["first_tool", "second_tool"]
    assert outcome.execution_count == 2


def test_dangerous_step_pauses_before_execution() -> None:
    runner, _tools, calls = _chain_runner()

    outcome = runner.run(
        (
            ToolChainStep("first", "demo.first"),
            ToolChainStep("danger", "demo.danger", {"value": "x"}),
            ToolChainStep("third", "demo.third", {"value": "c"}),
        )
    )

    assert outcome.success is False
    assert outcome.status == "confirmation_required"
    assert outcome.failed_step_id == "danger"
    assert outcome.confirmation_id
    assert calls == ["first_tool"]
    assert outcome.execution_count == 1


def test_confirmation_resumes_without_repeating_previous_steps() -> None:
    runner, _tools, calls = _chain_runner()
    pending = runner.run(
        (
            ToolChainStep("first", "demo.first"),
            ToolChainStep("danger", "demo.danger", {"value": "x"}),
            ToolChainStep("third", "demo.third", {"value": "c"}),
        )
    )

    outcome = runner.confirm(str(pending.confirmation_id), "s")

    assert outcome.success is True
    assert outcome.status == "success"
    assert calls == ["first_tool", "danger_tool", "third_tool"]
    assert outcome.execution_count == 3


def test_rejecting_confirmation_cancels_chain() -> None:
    runner, _tools, calls = _chain_runner()
    pending = runner.run(
        (
            ToolChainStep("first", "demo.first"),
            ToolChainStep("danger", "demo.danger", {"value": "x"}),
            ToolChainStep("third", "demo.third", {"value": "c"}),
        )
    )

    outcome = runner.confirm(str(pending.confirmation_id), "n")

    assert outcome.success is False
    assert outcome.status == "cancelled"
    assert outcome.failed_step_id == "danger"
    assert calls == ["first_tool"]


def test_ambiguous_confirmation_keeps_chain_pending() -> None:
    runner, _tools, calls = _chain_runner()
    pending = runner.run(
        (
            ToolChainStep("first", "demo.first"),
            ToolChainStep("danger", "demo.danger", {"value": "x"}),
            ToolChainStep("third", "demo.third", {"value": "c"}),
        )
    )

    ambiguous = runner.confirm(str(pending.confirmation_id), "maybe")

    assert ambiguous.status == "invalid_confirmation"
    assert len(runner.pending_chains) == 1

    confirmed = runner.confirm(str(pending.confirmation_id), "yes")

    assert len(runner.pending_chains) == 0
    assert confirmed.success is True
    assert calls == ["first_tool", "danger_tool", "third_tool"]


def test_wrong_confirmation_id_does_not_resume_chain() -> None:
    runner, _tools, calls = _chain_runner()
    runner.run(
        (
            ToolChainStep("first", "demo.first"),
            ToolChainStep("danger", "demo.danger", {"value": "x"}),
        )
    )

    outcome = runner.confirm("wrong-id", "s")

    assert outcome.status == "confirmation_not_found"
    assert outcome.execution_count == 0
    assert calls == ["first_tool"]


def test_finished_chain_cannot_be_resumed_with_old_confirmation() -> None:
    runner, _tools, calls = _chain_runner()
    pending = runner.run(
        (
            ToolChainStep("first", "demo.first"),
            ToolChainStep("danger", "demo.danger", {"value": "x"}),
        )
    )
    done = runner.confirm(str(pending.confirmation_id), "s")

    old = runner.confirm(str(pending.confirmation_id), "s")

    assert done.success is True
    assert old.status == "confirmation_not_found"
    assert calls == ["first_tool", "danger_tool"]


def test_bootstrap_builds_tool_chain_runner() -> None:
    runner = Bootstrap.build_tool_chain_runner()

    assert isinstance(runner, ToolChainRunner)
