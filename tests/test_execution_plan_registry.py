from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from typing import Any

import pytest

from core.execution_condition import ExecutionCondition, ExecutionConditionOperator
from core.execution_context import ExecutionContext
from core.execution_plan_executor import (
    ExecutionErrorCode,
    ExecutionProgress,
    ExecutionPlanExecutor,
    ResumableExecutionState,
)
from core.execution_plan_output import ExecutionPlanOutput
from core.execution_plan_registry import (
    ExecutionPlanAlreadyRegisteredError,
    ExecutionPlanNotFoundError,
    ExecutionPlanReference,
    ExecutionPlanRegistry,
    InvalidExecutionPlanIdError,
    InvalidExecutionPlanVersionError,
    RegisteredExecutionPlan,
)
from core.execution_plan_validator import ExecutionPlanValidator, plan_signature
from core.execution_retry import RetryPolicy
from core.execution_variable_binding import ExecutionVariableBinding
from core.execution_variable_reference import ExecutionVariableReference
from core.planner import ExecutionPlan, ExecutionStep
from core.step_output_reference import StepOutputReference
from tools.base_tool import BaseTool
from tools.registry import ToolRegistry
from tools.tool_context import ToolContext


class SpyTool(BaseTool):
    def __init__(
        self,
        name: str,
        calls: list[dict[str, Any]],
        output: Any = "ok",
    ) -> None:
        self._name = name
        self._calls = calls
        self._output = output

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return f"Spy tool {self._name}."

    def execute(self, context: ToolContext) -> Any:
        self._calls.append(
            {
                "tool": self._name,
                "parameters": dict(context.parameters),
                "execution_id": context.metadata.get("execution_id"),
            }
        )
        return self._output


def _step(
    step_id: str,
    tool: str | None = "safe.tool",
    *,
    dependencies: tuple[str, ...] = (),
    arguments: dict[str, Any] | None = None,
    subplan: ExecutionPlan | None = None,
    subplan_ref: ExecutionPlanReference | None = None,
    output_binding: ExecutionVariableBinding | None = None,
    condition: object | None = None,
) -> ExecutionStep:
    return ExecutionStep(
        id=step_id,
        description=f"Run {step_id}.",
        tool=tool,
        dependencies=dependencies,
        arguments={} if arguments is None else arguments,
        subplan=subplan,
        subplan_ref=subplan_ref,
        output_binding=output_binding,
        condition=condition,
    )


def _plan(
    steps: tuple[ExecutionStep, ...],
    *,
    required_tools: tuple[str, ...] | None = None,
    output: object | None = None,
) -> ExecutionPlan:
    tools = required_tools
    if tools is None:
        tools = tuple(
            step.tool
            for step in steps
            if step.tool is not None and step.tool != "direct_response"
        )
    return ExecutionPlan(
        goal="Run plan.",
        ordered_steps=steps,
        estimated_steps=len(steps),
        required_tools=tools,
        detected_risks=(),
        requires_confirmation=False,
        output=output,
    )


def _tool_registry(*tools: BaseTool) -> ToolRegistry:
    registry = ToolRegistry()
    for tool in tools:
        registry.register(tool)
    return registry


def test_registry_crud_versioning_order_and_immutability() -> None:
    registry = ExecutionPlanRegistry()
    plan_a = _plan((_step("a"),))
    plan_b = _plan((_step("b"),))

    assert len(registry) == 0
    entry_a = registry.register("project.analysis", plan_a)
    entry_b = registry.register("project.analysis", plan_b, version="1.0")

    assert registry.contains("project.analysis") is True
    assert registry.contains("project.analysis", version="1.0") is True
    assert registry.get("project.analysis") is plan_a
    assert registry.resolve(ExecutionPlanReference("project.analysis", "1.0")) is plan_b
    assert registry.list_entries() == (entry_a, entry_b)
    assert tuple(registry) == (entry_a, entry_b)

    with pytest.raises(ExecutionPlanAlreadyRegisteredError):
        registry.register("project.analysis", plan_b)
    replacement = registry.register("project.analysis", plan_b, replace=True)
    assert registry.list_entries() == (replacement, entry_b)
    assert registry.unregister("missing") is False
    assert registry.unregister("project.analysis", version="1.0") is True
    registry.clear()
    assert len(registry) == 0

    with pytest.raises(FrozenInstanceError):
        entry_a.reference = ExecutionPlanReference("other")  # type: ignore[misc]


def test_reference_validation_policy() -> None:
    for plan_id in ("project.analysis", "coding-run_tests", "workflow_1"):
        assert ExecutionPlanReference(plan_id).plan_id == plan_id
    assert ExecutionPlanReference("project.analysis", "beta-1").version == "beta-1"

    for plan_id in ("", " project.analysis ", "../project", "C:\\plan", "project/analysis", "$secret", "__class__"):
        with pytest.raises(InvalidExecutionPlanIdError):
            ExecutionPlanReference(plan_id)
    for version in ("", " 1.0 ", "../1", "1/0", "v$"):
        with pytest.raises(InvalidExecutionPlanVersionError):
            ExecutionPlanReference("project.analysis", version)


def test_registry_resolve_missing_fails() -> None:
    with pytest.raises(ExecutionPlanNotFoundError):
        ExecutionPlanRegistry().resolve(ExecutionPlanReference("project.analysis"))


def test_validator_accepts_registered_subplan_and_rejects_bad_configurations() -> None:
    child = _plan((_step("child"),))
    registry = ExecutionPlanRegistry()
    registry.register("project.analysis", child, version="1.0")
    parent = _plan(
        (_step("run_child", tool=None, subplan_ref=ExecutionPlanReference("project.analysis", "1.0")),),
        required_tools=(),
    )

    assert ExecutionPlanValidator(plan_registry=registry).validate(parent).is_valid is True
    assert ExecutionPlanValidator().validate(parent).is_valid is False

    missing = _plan(
        (_step("run_child", tool=None, subplan_ref=ExecutionPlanReference("missing")),),
        required_tools=(),
    )
    result = ExecutionPlanValidator(plan_registry=registry).validate(missing)
    assert result.is_valid is False
    assert any("ExecutionPlanNotFoundError" in error for error in result.errors)

    invalid_child = replace(child, estimated_steps=99)
    registry.register("invalid.child", invalid_child)
    invalid_parent = _plan(
        (_step("run_child", tool=None, subplan_ref=ExecutionPlanReference("invalid.child")),),
        required_tools=(),
    )
    invalid_result = ExecutionPlanValidator(plan_registry=registry).validate(invalid_parent)
    assert invalid_result.is_valid is False
    assert any("estimated_steps" in error for error in invalid_result.errors)


def test_step_action_xor_includes_subplan_ref() -> None:
    child = _plan((_step("child"),))
    reference = ExecutionPlanReference("project.analysis")

    valid_tool = _plan((_step("tool", "safe.tool"),))
    valid_embedded = _plan((_step("embedded", None, subplan=child),), required_tools=())
    valid_ref = _plan((_step("ref", None, subplan_ref=reference),), required_tools=())

    registry = ExecutionPlanRegistry()
    registry.register("project.analysis", child)
    assert ExecutionPlanValidator().validate(valid_tool).is_valid is True
    assert ExecutionPlanValidator().validate(valid_embedded).is_valid is True
    assert ExecutionPlanValidator(plan_registry=registry).validate(valid_ref).is_valid is True

    invalid_plans = (
        _plan((_step("bad", "safe.tool", subplan_ref=reference),)),
        _plan((_step("bad", None, subplan=child, subplan_ref=reference),), required_tools=()),
        _plan((_step("bad", None),), required_tools=()),
        _plan((_step("bad", "safe.tool", subplan=child, subplan_ref=reference),)),
    )
    for plan in invalid_plans:
        result = ExecutionPlanValidator(plan_registry=registry).validate(plan)
        assert result.is_valid is False
        assert any("exactly one of tool, subplan, or subplan_ref" in error for error in result.errors)


def test_registered_subplan_executes_with_inputs_output_binding_and_trace() -> None:
    calls: list[dict[str, Any]] = []
    child = _plan(
        (_step("child", "echo.tool", arguments={"value": ExecutionVariableReference("input")}),),
        required_tools=("echo.tool",),
        output=ExecutionPlanOutput({"answer": StepOutputReference("child")}),
    )
    registry = ExecutionPlanRegistry()
    registry.register("project.analysis", child, version="1.0")
    parent = _plan(
        (
            _step(
                "run_child",
                None,
                arguments={"input": "hello"},
                subplan_ref=ExecutionPlanReference("project.analysis", "1.0"),
                output_binding=ExecutionVariableBinding("child_answer", ("answer",)),
            ),
            _step(
                "consume",
                "echo.tool",
                dependencies=("run_child",),
                arguments={"value": StepOutputReference("run_child", ("answer",))},
            ),
        ),
        required_tools=("echo.tool",),
    )
    validation = ExecutionPlanValidator(_tool_registry(), plan_registry=registry).validate(parent)

    result = ExecutionPlanExecutor(
        _tool_registry(SpyTool("echo.tool", calls, output={"seen": True})),
        plan_registry=registry,
    ).execute(parent, validation)

    assert result.success is True
    assert calls[0]["parameters"] == {"value": "hello"}
    assert calls[1]["parameters"] == {"value": {"seen": True}}
    assert result.step_results[0].output == {"answer": {"seen": True}}
    assert result.step_results[0].metadata["plan_id"] == "project.analysis"
    assert result.step_results[0].metadata["resolved_plan_signature"] == plan_signature(child)
    actions = [event.action for event in result.trace.events]
    assert "execution_plan_reference_resolution_started" in actions
    assert "execution_plan_reference_resolution_succeeded" in actions


def test_false_condition_does_not_resolve_registered_subplan() -> None:
    parent = _plan(
        (
            _step(
                "run_child",
                None,
                subplan_ref=ExecutionPlanReference("missing.plan"),
                condition=ExecutionCondition(False, ExecutionConditionOperator.TRUTHY),
            ),
        ),
        required_tools=(),
    )
    validation = replace(
        ExecutionPlanValidator().validate(_plan((_step("noop", "direct_response"),), required_tools=())),
        plan_signature=plan_signature(parent),
    )

    result = ExecutionPlanExecutor(_tool_registry()).execute(parent, validation)

    assert result.success is True
    assert result.skipped_steps == ["run_child"]
    assert not any(
        event.action.startswith("execution_plan_reference_resolution")
        for event in result.trace.events
    )


def test_registered_reference_recursion_and_depth_are_validated() -> None:
    registry = ExecutionPlanRegistry()
    plan_a = _plan((_step("a", None, subplan_ref=ExecutionPlanReference("plan.a")),), required_tools=())
    registry.register("plan.a", plan_a)
    root = _plan((_step("root", None, subplan_ref=ExecutionPlanReference("plan.a")),), required_tools=())
    result = ExecutionPlanValidator(plan_registry=registry).validate(root)
    assert result.is_valid is False
    assert any("RecursiveRegisteredExecutionPlanError" in error for error in result.errors)

    registry.clear()
    plan_c = _plan((_step("c", "safe.tool"),))
    plan_b = _plan((_step("b", None, subplan_ref=ExecutionPlanReference("plan.c")),), required_tools=())
    plan_a = _plan((_step("a", None, subplan_ref=ExecutionPlanReference("plan.b")),), required_tools=())
    registry.register("plan.a", plan_a)
    registry.register("plan.b", plan_b)
    registry.register("plan.c", plan_c)
    assert ExecutionPlanValidator(plan_registry=registry).validate(plan_a).is_valid is True


def test_parent_signature_uses_reference_not_registered_content() -> None:
    reference = ExecutionPlanReference("project.analysis", "1.0")
    parent = _plan((_step("run_child", None, subplan_ref=reference),), required_tools=())
    changed_parent = replace(
        parent,
        ordered_steps=(
            _step("run_child", None, subplan_ref=ExecutionPlanReference("project.analysis", "2.0")),
        ),
    )

    assert plan_signature(parent) != plan_signature(changed_parent)
    assert "project.analysis" not in plan_signature(parent)


def test_resume_rejects_registered_plan_signature_mismatch() -> None:
    registry = ExecutionPlanRegistry()
    original_child = _plan((_step("child", "safe.tool"),))
    changed_child = _plan((_step("changed", "safe.tool"),))
    registry.register("project.analysis", original_child, replace=True)
    parent = _plan(
        (_step("run_child", None, subplan_ref=ExecutionPlanReference("project.analysis")),),
        required_tools=(),
    )
    validation = ExecutionPlanValidator(plan_registry=registry).validate(parent)
    context = ExecutionContext("exec-registered-resume")
    context.mark_step_started("run_child", 1)
    context.mark_step_succeeded("run_child", "old")
    context.set_metadata(
        "registered_plan_signatures",
        {
            "run_child": {
                "plan_id": "project.analysis",
                "version": None,
                "resolved_plan_signature": plan_signature(original_child),
            }
        },
    )
    state = ResumableExecutionState(
        objective="resume",
        original_plan=parent,
        validation_result=validation,
        validated_plan_signature=validation.plan_signature,
        completed_step_ids=("run_child",),
        pending_step_ids=("next",),
        failed_step_ids=(),
        interrupted_step_id="next",
        previous_results={"run_child": "old"},
        resumable=True,
        execution_context_snapshot=context.snapshot(),
    )
    registry.register("project.analysis", changed_child, replace=True)

    result = ExecutionPlanExecutor(_tool_registry(), plan_registry=registry).resume(state)

    assert result.success is False
    assert result.error_code == ExecutionErrorCode.VALIDATION_MISMATCH.value
    assert "signature mismatch" in (result.error or "")


def test_retry_resolves_registered_reference_again_after_explicit_register() -> None:
    calls: list[dict[str, Any]] = []
    registry = ExecutionPlanRegistry()
    child = _plan((_step("child", "echo.tool"),), required_tools=("echo.tool",))
    parent = _plan(
        (_step("run_child", None, subplan_ref=ExecutionPlanReference("project.analysis")),),
        required_tools=(),
    )
    validation = replace(
        ExecutionPlanValidator().validate(_plan((_step("noop", "direct_response"),), required_tools=())),
        plan_signature=plan_signature(parent),
    )

    def on_progress(progress: ExecutionProgress) -> None:
        if progress.phase == "step_retry_scheduled":
            registry.register("project.analysis", child)

    result = ExecutionPlanExecutor(
        _tool_registry(SpyTool("echo.tool", calls, output="registered")),
        retry_policy=RetryPolicy(max_attempts=2),
        plan_registry=registry,
    ).execute(parent, validation, on_progress=on_progress)

    assert result.success is True
    assert calls
    assert result.step_results[0].metadata["completed_after_retry"] is True
    assert result.step_results[0].metadata["retry_history"][0]["error_code"] == (
        ExecutionErrorCode.EXECUTION_PLAN_REFERENCE_NOT_FOUND.value
    )
