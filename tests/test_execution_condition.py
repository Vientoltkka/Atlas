from __future__ import annotations

import pytest

from core.execution_condition import (
    AllOfCondition,
    AnyOfCondition,
    ConditionOperandTypeError,
    ConditionTreeTooDeepError,
    ConditionTreeTooLargeError,
    EmptyConditionGroupError,
    ExecutionCondition,
    ExecutionConditionEvaluator,
    ExecutionConditionOperator,
    InvalidConditionNodeError,
    InvalidExecutionConditionError,
    MAX_CONDITION_DEPTH,
    MAX_CONDITION_NODES,
    NotCondition,
    ReferencedStepSkippedError,
    condition_tree_stats,
)
from core.execution_context import ExecutionContext
from core.execution_variable_reference import ExecutionVariableReference
from core.step_output_reference import StepOutputReference


def _truthy(value: object) -> ExecutionCondition:
    return ExecutionCondition(value, ExecutionConditionOperator.TRUTHY)


def _equals(left: object, right: object) -> ExecutionCondition:
    return ExecutionCondition(left, ExecutionConditionOperator.EQUALS, right)


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


def test_composite_condition_nodes_are_valid_and_immutable() -> None:
    source = [_truthy(True)]
    all_condition = AllOfCondition(source)
    any_condition = AnyOfCondition(tuple(source))
    not_condition = NotCondition(_truthy(False))
    source.append(_truthy(False))

    assert len(all_condition.conditions) == 1
    assert len(any_condition.conditions) == 1
    assert not_condition.condition == _truthy(False)
    assert condition_tree_stats(all_condition) == {"node_count": 2, "max_depth": 2}


def test_composite_condition_rejects_empty_and_invalid_nodes() -> None:
    with pytest.raises(EmptyConditionGroupError):
        AllOfCondition(())
    with pytest.raises(EmptyConditionGroupError):
        AnyOfCondition([])
    with pytest.raises(InvalidConditionNodeError):
        AllOfCondition((_truthy(True), "bad"))  # type: ignore[arg-type]
    with pytest.raises(InvalidConditionNodeError):
        NotCondition(None)  # type: ignore[arg-type]
    with pytest.raises(InvalidConditionNodeError):
        NotCondition([_truthy(True)])  # type: ignore[arg-type]


def test_all_any_not_semantics_and_nested_tree() -> None:
    context = ExecutionContext("exec-1")

    assert ExecutionConditionEvaluator().evaluate(
        AllOfCondition((_truthy(True), _equals("a", "a"))),
        context,
    ).matched is True
    assert ExecutionConditionEvaluator().evaluate(
        AllOfCondition((_truthy(True), _truthy(False))),
        context,
    ).matched is False
    assert ExecutionConditionEvaluator().evaluate(
        AnyOfCondition((_truthy(False), _equals("a", "a"))),
        context,
    ).matched is True
    assert ExecutionConditionEvaluator().evaluate(
        AnyOfCondition((_truthy(False), _truthy(False))),
        context,
    ).matched is False
    assert ExecutionConditionEvaluator().evaluate(NotCondition(_truthy(True)), context).matched is False
    assert ExecutionConditionEvaluator().evaluate(NotCondition(_truthy(False)), context).matched is True

    nested = AllOfCondition(
        (
            _truthy(True),
            AnyOfCondition((_truthy(False), NotCondition(_truthy(False)))),
        )
    )
    result = ExecutionConditionEvaluator().evaluate(nested, context)

    assert result.matched is True
    assert result.condition_kind == "all"
    assert result.evaluated_nodes == 6


def test_condition_tree_depth_and_node_limits() -> None:
    node = _truthy(True)
    for _ in range(MAX_CONDITION_DEPTH - 1):
        node = NotCondition(node)
    assert condition_tree_stats(node)["max_depth"] == MAX_CONDITION_DEPTH

    with pytest.raises(ConditionTreeTooDeepError):
        NotCondition(node)

    allowed = AllOfCondition(tuple(_truthy(True) for _ in range(MAX_CONDITION_NODES - 1)))
    assert condition_tree_stats(allowed)["node_count"] == MAX_CONDITION_NODES

    with pytest.raises(ConditionTreeTooLargeError):
        AllOfCondition(tuple(_truthy(True) for _ in range(MAX_CONDITION_NODES)))


def test_all_short_circuits_without_evaluating_later_missing_variable() -> None:
    condition = AllOfCondition(
        (
            _truthy(False),
            ExecutionCondition(
                ExecutionVariableReference("missing"),
                ExecutionConditionOperator.EQUALS,
                True,
            ),
        )
    )

    result = ExecutionConditionEvaluator().evaluate(condition, ExecutionContext("exec-1"))

    assert result.matched is False
    assert result.evaluated_nodes == 2
    assert result.skipped_nodes_due_to_short_circuit == 1


def test_any_short_circuits_without_evaluating_later_missing_variable() -> None:
    condition = AnyOfCondition(
        (
            _truthy(True),
            ExecutionCondition(
                ExecutionVariableReference("missing"),
                ExecutionConditionOperator.EQUALS,
                True,
            ),
        )
    )

    result = ExecutionConditionEvaluator().evaluate(condition, ExecutionContext("exec-1"))

    assert result.matched is True
    assert result.evaluated_nodes == 2
    assert result.skipped_nodes_due_to_short_circuit == 1


def test_required_composite_node_errors_are_not_silenced() -> None:
    condition = AnyOfCondition(
        (
            _truthy(False),
            ExecutionCondition(
                ExecutionVariableReference("missing"),
                ExecutionConditionOperator.EQUALS,
                True,
            ),
        )
    )

    with pytest.raises(Exception, match="missing"):
        ExecutionConditionEvaluator().evaluate(condition, ExecutionContext("exec-1"))
