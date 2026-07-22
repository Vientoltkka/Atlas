from __future__ import annotations

import pytest

from core.execution_condition import (
    ConditionOperandTypeError,
    ExecutionCondition,
    ExecutionConditionEvaluator,
    ExecutionConditionOperator,
    InvalidExecutionConditionError,
    ReferencedStepSkippedError,
)
from core.execution_context import ExecutionContext
from core.execution_variable_reference import ExecutionVariableReference
from core.step_output_reference import StepOutputReference


@pytest.mark.parametrize(
    ("condition", "expected"),
    [
        (ExecutionCondition("alpha", ExecutionConditionOperator.EQUALS, "alpha"), True),
        (ExecutionCondition("alpha", ExecutionConditionOperator.NOT_EQUALS, "beta"), True),
        (ExecutionCondition(None, ExecutionConditionOperator.IS_NONE), True),
        (ExecutionCondition("x", ExecutionConditionOperator.IS_NOT_NONE), True),
        (ExecutionCondition(3, ExecutionConditionOperator.GREATER_THAN, 2), True),
        (ExecutionCondition(3, ExecutionConditionOperator.GREATER_THAN_OR_EQUAL, 3), True),
        (ExecutionCondition(2, ExecutionConditionOperator.LESS_THAN, 3), True),
        (ExecutionCondition(2, ExecutionConditionOperator.LESS_THAN_OR_EQUAL, 2), True),
        (ExecutionCondition("atlas", ExecutionConditionOperator.CONTAINS, "las"), True),
        (ExecutionCondition([1, 2], ExecutionConditionOperator.NOT_CONTAINS, 3), True),
        (ExecutionCondition([], ExecutionConditionOperator.IS_EMPTY), True),
        (ExecutionCondition({"a": 1}, ExecutionConditionOperator.IS_NOT_EMPTY), True),
        (ExecutionCondition(True, ExecutionConditionOperator.TRUTHY), True),
        (ExecutionCondition(False, ExecutionConditionOperator.FALSY), True),
    ],
)
def test_condition_evaluator_supports_declared_operators(
    condition: ExecutionCondition,
    expected: bool,
) -> None:
    result = ExecutionConditionEvaluator().evaluate(condition, ExecutionContext("exec-1"))

    assert result.matched is expected
    assert result.operator == condition.operator.value


def test_condition_resolves_step_outputs_and_variables() -> None:
    context = ExecutionContext("exec-1", initial_variables={"threshold": 10})
    context.mark_step_started("step_1", 1)
    context.mark_step_succeeded("step_1", {"count": 12})

    condition = ExecutionCondition(
        StepOutputReference("step_1", path=("count",)),
        ExecutionConditionOperator.GREATER_THAN,
        ExecutionVariableReference("threshold"),
    )

    assert ExecutionConditionEvaluator().evaluate(condition, context).matched is True


def test_exists_distinguishes_missing_from_present_references() -> None:
    context = ExecutionContext("exec-1", initial_variables={"flag": None})

    assert ExecutionConditionEvaluator().evaluate(
        ExecutionCondition(ExecutionVariableReference("flag"), ExecutionConditionOperator.EXISTS),
        context,
    ).matched is True
    assert ExecutionConditionEvaluator().evaluate(
        ExecutionCondition(ExecutionVariableReference("missing"), ExecutionConditionOperator.NOT_EXISTS),
        context,
    ).matched is True


def test_condition_rejects_wrong_arity_and_incompatible_operands() -> None:
    with pytest.raises(InvalidExecutionConditionError):
        ExecutionCondition("x", ExecutionConditionOperator.TRUTHY, True)

    with pytest.raises(InvalidExecutionConditionError):
        ExecutionCondition("x", ExecutionConditionOperator.EQUALS)

    with pytest.raises(ConditionOperandTypeError):
        ExecutionConditionEvaluator().evaluate(
            ExecutionCondition(True, ExecutionConditionOperator.GREATER_THAN, 1),
            ExecutionContext("exec-1"),
        )


def test_condition_reference_to_skipped_step_fails_clearly() -> None:
    context = ExecutionContext("exec-1")
    context.mark_step_skipped("step_1")

    with pytest.raises(ReferencedStepSkippedError):
        ExecutionConditionEvaluator().evaluate(
            ExecutionCondition(
                StepOutputReference("step_1"),
                ExecutionConditionOperator.EXISTS,
            ),
            context,
        )
