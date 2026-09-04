"""Bridge tests: validated ExecutionPlan → AsyncTaskScheduler task specs."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from core.async_task_scheduler import AsyncTaskScheduler, TaskOutcome
from core.execution_plan_task_adapter import (
    ExecutionPlanTaskBridgeError,
    execution_plan_to_task_specs,
)
from core.execution_plan_validator import ExecutionPlanValidator
from core.execution_retry import RetryPolicy, RetryStrategy
from core.planner import ExecutionPlan, ExecutionStep
from core.step_output_reference import StepOutputReference


def _canonical_plan() -> ExecutionPlan:
    read_first = ExecutionStep(
        "read_first",
        "Leer nota1.txt",
        "read_file",
        arguments={"path": "nota1.txt"},
    )
    read_second = ExecutionStep(
        "read_second",
        "Leer nota2.txt",
        "read_file",
        arguments={"path": "nota2.txt"},
    )
    compare = ExecutionStep(
        "compare",
        "Comparar los contenidos",
        "direct_response",
        dependencies=("read_first", "read_second"),
        arguments={"instruction": "Compara los siguientes contenidos: {input}"},
    )
    write_result = ExecutionStep(
        "write_result",
        "Guardar el resultado",
        "write_file",
        dependencies=("compare",),
        arguments={
            "path": "resultado.txt",
            "content": StepOutputReference("compare"),
        },
    )
    return ExecutionPlan(
        goal="comparar las dos notas",
        ordered_steps=(read_first, read_second, compare, write_result),
        estimated_steps=4,
        required_tools=("read_file", "write_file"),
        detected_risks=("write_file",),
        requires_confirmation=True,
        status="planned",
    )


def _plan(*steps: ExecutionStep, **plan_kwargs) -> ExecutionPlan:
    kwargs = {
        "goal": "objetivo de prueba",
        "estimated_steps": len(steps),
        "required_tools": (),
        "detected_risks": (),
        "requires_confirmation": False,
        "status": "planned",
    }
    kwargs.update(plan_kwargs)
    return ExecutionPlan(ordered_steps=tuple(steps), **kwargs)


def test_canonical_plan_maps_to_chained_scheduler_specs() -> None:
    plan = _canonical_plan()

    assert ExecutionPlanValidator().validate(plan).is_valid is True
    specs = execution_plan_to_task_specs(plan, validator=ExecutionPlanValidator())

    assert [spec["task_id"] for spec in specs] == [
        "read_first",
        "read_second",
        "compare",
        "write_result",
    ]
    assert all(spec["requires_approval"] is False for spec in specs)

    first, second, compare_spec, write = specs
    assert first["dependencies"] == []
    assert first["payload"] == {
        "kind": "tool",
        "tool": "read_file",
        "arguments": {"path": "nota1.txt"},
    }
    assert second["payload"]["arguments"] == {"path": "nota2.txt"}

    # Reasoning step: unambiguous direct_response + instruction → transform.
    assert compare_spec["dependencies"] == ["read_first", "read_second"]
    assert compare_spec["payload"]["kind"] == "transform"
    assert compare_spec["payload"]["input_tasks"] == ["read_first", "read_second"]
    assert "{input}" in compare_spec["payload"]["instruction"]

    # Output chaining: single top-level content reference → content_task.
    assert write["dependencies"] == ["compare"]
    assert write["payload"]["kind"] == "tool"
    assert write["payload"]["tool"] == "write_file"
    assert write["payload"]["arguments"] == {"path": "resultado.txt"}
    assert write["payload"]["content_task"] == "compare"


def test_specs_are_accepted_by_the_existing_scheduler() -> None:
    specs = execution_plan_to_task_specs(_canonical_plan())
    scheduler = AsyncTaskScheduler(lambda task, payload: TaskOutcome.succeed("ok"))

    goal_id = scheduler.submit_goal("comparar las dos notas", list(specs))

    state = scheduler.goal(goal_id)
    assert set(state.tasks) == {
        "read_first",
        "read_second",
        "compare",
        "write_result",
    }
    assert state.tasks["compare"].dependencies == ("read_first", "read_second")
    assert state.tasks["write_result"].dependencies == ("compare",)


def test_retry_policy_maps_attempts_to_bounded_retries() -> None:
    steps = (
        ExecutionStep(
            "flaky",
            "Paso con reintentos",
            "read_file",
            arguments={"path": "a.txt"},
            retry_policy=RetryPolicy(max_attempts=3),
        ),
        ExecutionStep(
            "strict",
            "Paso sin reintentos",
            "read_file",
            arguments={"path": "b.txt"},
            retry_policy=RetryPolicy(max_attempts=1, strategy=RetryStrategy.NO_RETRY),
        ),
        ExecutionStep(
            "default",
            "Paso con política por defecto",
            "read_file",
            arguments={"path": "c.txt"},
        ),
    )
    specs = execution_plan_to_task_specs(_plan(*steps))

    # 3 attempts == 2 retries (attempts semantics).
    assert specs[0]["max_retries"] == 2
    assert specs[1]["max_retries"] == 0
    assert "max_retries" not in specs[2]


def test_plan_metadata_is_preserved_without_extending_the_scheduler() -> None:
    step = ExecutionStep(
        "meta",
        "Paso con metadatos",
        "read_file",
        arguments={"path": "a.txt"},
        parallel_safe=True,
        priority=3,
        urgency=2,
        criticality=1,
        deadline=datetime(2026, 9, 4, 12, 0, tzinfo=timezone.utc),
    )
    specs = execution_plan_to_task_specs(_plan(step))

    assert specs[0]["payload"]["plan_metadata"] == {
        "parallel_safe": True,
        "priority": 3,
        "urgency": 2,
        "criticality": 1,
        "deadline": "2026-09-04T12:00:00+00:00",
    }


def test_invalid_plan_is_rejected_without_partial_conversion() -> None:
    broken = _plan(
        ExecutionStep("A", "Leer", "read_file", arguments={"path": "a.txt"}),
        ExecutionStep("A", "Leer otra vez", "read_file", arguments={"path": "b.txt"}),
    )
    validator = ExecutionPlanValidator()
    assert validator.validate(broken).is_valid is False

    with pytest.raises(ExecutionPlanTaskBridgeError) as error:
        execution_plan_to_task_specs(broken, validator=validator)

    assert error.value.code == "INVALID_PLAN"
    assert error.value.errors


def test_unknown_dependency_is_rejected_without_a_validator() -> None:
    broken = _plan(
        ExecutionStep("A", "Leer", "read_file", arguments={"path": "a.txt"}),
        ExecutionStep(
            "B",
            "Comparar",
            "direct_response",
            dependencies=("missing",),
            arguments={"instruction": "Compara {input}"},
        ),
    )

    with pytest.raises(ExecutionPlanTaskBridgeError) as error:
        execution_plan_to_task_specs(broken)

    assert error.value.code == "UNKNOWN_DEPENDENCY"


@pytest.mark.parametrize(
    "step",
    [
        ExecutionStep("x", "Paso lógico", None),
        ExecutionStep("x", "Respuesta ambigua", "direct_response"),
        ExecutionStep(
            "x",
            "Instrucción ambigua",
            "direct_response",
            arguments={"instruction": "", "otro": "valor"},
        ),
    ],
)
def test_ambiguous_reasoning_steps_are_rejected(step: ExecutionStep) -> None:
    with pytest.raises(ExecutionPlanTaskBridgeError) as error:
        execution_plan_to_task_specs(_plan(step))

    assert error.value.code in {"AMBIGUOUS_LOGICAL_STEP", "AMBIGUOUS_REASONING_STEP"}


def test_subplan_steps_are_rejected_without_silent_flattening() -> None:
    inner = _plan(
        ExecutionStep("inner", "Paso interno", "read_file", arguments={"path": "i.txt"})
    )
    step = ExecutionStep("outer", "Paso con subplan", None, subplan=inner)
    with pytest.raises(ExecutionPlanTaskBridgeError) as error:
        execution_plan_to_task_specs(_plan(step))
    assert error.value.code == "UNSUPPORTED_STEP_KIND"


def test_unsafe_result_references_are_rejected() -> None:
    referenced = ExecutionStep(
        "src",
        "Leer",
        "read_file",
        arguments={"path": "a.txt"},
    )
    # Reference outside the supported 'content' argument.
    bad_key = ExecutionStep(
        "bad_key",
        "Escribir",
        "write_file",
        dependencies=("src",),
        arguments={
            "path": StepOutputReference("src"),
            "content": "texto fijo",
        },
    )
    with pytest.raises(ExecutionPlanTaskBridgeError) as error:
        execution_plan_to_task_specs(_plan(referenced, bad_key))
    assert error.value.code == "UNSUPPORTED_RESULT_REFERENCE"

    # Reference to a step that is not declared as a dependency.
    undeclared = ExecutionStep(
        "undeclared",
        "Escribir",
        "write_file",
        arguments={"content": StepOutputReference("src")},
    )
    with pytest.raises(ExecutionPlanTaskBridgeError) as error:
        execution_plan_to_task_specs(_plan(referenced, undeclared))
    assert error.value.code == "UNSUPPORTED_RESULT_REFERENCE"

    # Nested reference inside a mapping value.
    nested = ExecutionStep(
        "nested",
        "Escribir",
        "write_file",
        dependencies=("src",),
        arguments={"data": {"content": StepOutputReference("src")}},
    )
    with pytest.raises(ExecutionPlanTaskBridgeError) as error:
        execution_plan_to_task_specs(_plan(referenced, nested))
    assert error.value.code == "UNSUPPORTED_RESULT_REFERENCE"

    # Declarative $ref dict form is not resolvable by the scheduler executor.
    template = ExecutionStep(
        "template",
        "Escribir",
        "write_file",
        dependencies=("src",),
        arguments={"content": {"$ref": "src.result"}},
    )
    with pytest.raises(ExecutionPlanTaskBridgeError) as error:
        execution_plan_to_task_specs(_plan(referenced, template))
    assert error.value.code == "UNSUPPORTED_RESULT_REFERENCE"


def test_single_content_reference_maps_to_content_task() -> None:
    source = ExecutionStep(
        "src",
        "Leer",
        "read_file",
        arguments={"path": "a.txt"},
    )
    writer = ExecutionStep(
        "writer",
        "Escribir",
        "write_file",
        dependencies=("src",),
        arguments={"path": "out.txt", "content": StepOutputReference("src")},
    )
    specs = execution_plan_to_task_specs(_plan(source, writer))

    assert specs[1]["payload"]["content_task"] == "src"
    assert specs[1]["payload"]["arguments"] == {"path": "out.txt"}
