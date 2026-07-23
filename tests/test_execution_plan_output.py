from __future__ import annotations

import math
import types

import pytest

from core.execution_context import ExecutionContext
from core.execution_plan_output import (
    ExecutionPlanOutput,
    ExecutionPlanOutputResolutionError,
    ExecutionPlanOutputTooDeepError,
    ExecutionPlanOutputTooLargeError,
    InvalidExecutionPlanOutputError,
    MAX_EXECUTION_OUTPUT_DEPTH,
    MAX_EXECUTION_OUTPUT_NODES,
)
from core.execution_variable_reference import ExecutionVariableReference
from core.step_output_reference import StepOutputReference


def test_execution_plan_output_accepts_safe_static_values() -> None:
    assert ExecutionPlanOutput(None).as_definition() is None
    assert ExecutionPlanOutput(True).as_definition() is True
    assert ExecutionPlanOutput(3).as_definition() == 3
    assert ExecutionPlanOutput(1.5).as_definition() == 1.5
    assert ExecutionPlanOutput("ok").as_definition() == "ok"
    assert ExecutionPlanOutput(["a", 1]).as_definition() == ["a", 1]
    assert ExecutionPlanOutput(("a", 1)).as_definition() == ["a", 1]
    assert ExecutionPlanOutput({"status": "completed"}).as_definition() == {
        "status": "completed"
    }


def test_execution_plan_output_accepts_nested_references_and_paths() -> None:
    output = ExecutionPlanOutput(
        {
            "summary": StepOutputReference("analyze", ("summary",)),
            "score": ExecutionVariableReference("quality", ("value",)),
            "items": [StepOutputReference("scan", ("items", 0))],
        }
    )

    definition = output.as_definition()

    assert definition["summary"] == StepOutputReference("analyze", ("summary",))
    assert output.stats().reference_count == 3
    assert output.stats().step_reference_count == 2
    assert output.stats().variable_reference_count == 1


def test_execution_plan_output_definition_is_defensively_copied() -> None:
    source = {"items": ["a"]}
    output = ExecutionPlanOutput(source)

    source["items"].append("b")
    first = output.as_definition()
    first["items"].append("c")

    assert output.as_definition() == {"items": ["a"]}


def test_execution_plan_output_resolved_value_is_defensively_copied() -> None:
    output = ExecutionPlanOutput({"value": StepOutputReference("step_1")})
    context = ExecutionContext("exec-output-copy")
    context.mark_step_started("step_1", 1)
    context.mark_step_succeeded("step_1", {"items": ["a"]})

    resolved = output.resolve(context)
    resolved["value"]["items"].append("b")

    assert output.resolve(context) == {"value": {"items": ["a"]}}


def test_execution_plan_output_resolves_step_and_variable_paths() -> None:
    output = ExecutionPlanOutput(
        {
            "mapping": StepOutputReference("step_1", ("nested", "value")),
            "list": StepOutputReference("step_1", ("items", 1)),
            "tuple": ExecutionVariableReference("pair", (0,)),
        }
    )
    context = ExecutionContext(
        "exec-output-resolve",
        initial_variables={"pair": ("left", "right")},
    )
    context.mark_step_started("step_1", 1)
    context.mark_step_succeeded(
        "step_1",
        {"nested": {"value": "ok"}, "items": ["a", "b"]},
    )

    assert output.resolve(context) == {
        "mapping": "ok",
        "list": "b",
        "tuple": "left",
    }


def test_execution_plan_output_rejects_unsafe_values() -> None:
    unsupported = [
        math.nan,
        math.inf,
        lambda: None,
        object,
        types,
        object(),
        {"bad": {1, 2}},
        {1: "bad"},
    ]

    for value in unsupported:
        with pytest.raises(InvalidExecutionPlanOutputError):
            ExecutionPlanOutput(value)


def test_execution_plan_output_limits_depth_and_nodes() -> None:
    value: object = "leaf"
    for _ in range(MAX_EXECUTION_OUTPUT_DEPTH + 1):
        value = [value]
    with pytest.raises(ExecutionPlanOutputTooDeepError):
        ExecutionPlanOutput(value)

    too_many = ["x" for _ in range(MAX_EXECUTION_OUTPUT_NODES)]
    with pytest.raises(ExecutionPlanOutputTooLargeError):
        ExecutionPlanOutput(too_many)


def test_execution_plan_output_resolution_fails_for_unavailable_references() -> None:
    context = ExecutionContext("exec-output-missing")
    output = ExecutionPlanOutput(
        {
            "step": StepOutputReference("missing"),
            "variable": ExecutionVariableReference("missing"),
        }
    )

    with pytest.raises(ExecutionPlanOutputResolutionError):
        output.resolve(context)


def test_execution_plan_output_resolution_fails_for_missing_paths() -> None:
    context = ExecutionContext("exec-output-path")
    context.mark_step_started("step_1", 1)
    context.mark_step_succeeded("step_1", {"items": []})
    output = ExecutionPlanOutput(StepOutputReference("step_1", ("items", 0)))

    with pytest.raises(ExecutionPlanOutputResolutionError):
        output.resolve(context)
