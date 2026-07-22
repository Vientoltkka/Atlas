"""Safe structured execution conditions."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
import math

from core.execution_arguments import ExecutionArguments, InvalidExecutionArgumentError
from core.execution_context import ExecutionContext, ExecutionStepState
from core.execution_variable_reference import ExecutionVariableReference
from core.parameter_resolver import ParameterResolver, ParameterResolutionError
from core.step_output_reference import StepOutputReference


_MISSING_RIGHT = object()
_UNARY_OPERATORS = frozenset(
    {
        "is_none",
        "is_not_none",
        "exists",
        "not_exists",
        "truthy",
        "falsy",
        "is_empty",
        "is_not_empty",
    }
)


class ExecutionConditionOperator(str, Enum):
    """Supported structured condition operators."""

    EQUALS = "equals"
    NOT_EQUALS = "not_equals"
    IS_NONE = "is_none"
    IS_NOT_NONE = "is_not_none"
    EXISTS = "exists"
    NOT_EXISTS = "not_exists"
    TRUTHY = "truthy"
    FALSY = "falsy"
    GREATER_THAN = "greater_than"
    GREATER_THAN_OR_EQUAL = "greater_than_or_equal"
    LESS_THAN = "less_than"
    LESS_THAN_OR_EQUAL = "less_than_or_equal"
    CONTAINS = "contains"
    NOT_CONTAINS = "not_contains"
    IS_EMPTY = "is_empty"
    IS_NOT_EMPTY = "is_not_empty"


class ExecutionConditionError(ValueError):
    """Base error for execution condition failures."""


class InvalidExecutionConditionError(ExecutionConditionError):
    """Raised when a condition is structurally invalid."""


class ExecutionConditionEvaluationError(ExecutionConditionError):
    """Raised when a condition cannot be evaluated safely."""


class UnsupportedConditionOperatorError(ExecutionConditionError):
    """Raised when a condition operator is unknown."""


class ConditionOperandTypeError(ExecutionConditionEvaluationError):
    """Raised when operands are incompatible with an operator."""


class ReferencedStepSkippedError(ExecutionConditionEvaluationError):
    """Raised when a condition references a skipped step result."""


@dataclass(frozen=True, slots=True)
class ExecutionCondition:
    """A simple structured condition evaluated against an execution context."""

    left: object
    operator: ExecutionConditionOperator
    right: object = field(default_factory=lambda: _MISSING_RIGHT)

    def __post_init__(self) -> None:
        try:
            operator = (
                self.operator
                if isinstance(self.operator, ExecutionConditionOperator)
                else ExecutionConditionOperator(self.operator)
            )
        except ValueError as error:
            raise UnsupportedConditionOperatorError(
                f"Unsupported execution condition operator: {self.operator}."
            ) from error

        object.__setattr__(self, "operator", operator)
        _validate_operand(self.left, "condition.left")
        object.__setattr__(self, "left", ExecutionArguments({"value": self.left}).require("value"))
        if operator.value in _UNARY_OPERATORS:
            if self.right is not _MISSING_RIGHT:
                raise InvalidExecutionConditionError(
                    f"Operator {operator.value} does not accept a right operand."
                )
            object.__setattr__(self, "right", None)
            return

        if self.right is _MISSING_RIGHT:
            raise InvalidExecutionConditionError(
                f"Operator {operator.value} requires a right operand."
            )
        _validate_operand(self.right, "condition.right")
        object.__setattr__(self, "right", ExecutionArguments({"value": self.right}).require("value"))


@dataclass(frozen=True, slots=True)
class ExecutionConditionResult:
    """Safe condition evaluation result without operand values."""

    matched: bool
    operator: str
    left_type: str | None = None
    right_type: str | None = None
    error_code: str | None = None
    references: tuple[str, ...] = ()


class ExecutionConditionEvaluator:
    """Evaluate structured conditions without mutating execution state."""

    def __init__(
        self,
        resolver: ParameterResolver | None = None,
    ) -> None:
        self._resolver = resolver or ParameterResolver()

    def evaluate(
        self,
        condition: ExecutionCondition,
        context: ExecutionContext,
    ) -> ExecutionConditionResult:
        operator = condition.operator
        references = tuple(_condition_references(condition))
        if operator in {
            ExecutionConditionOperator.EXISTS,
            ExecutionConditionOperator.NOT_EXISTS,
        }:
            exists = self._operand_exists(condition.left, context)
            return ExecutionConditionResult(
                matched=exists
                if operator is ExecutionConditionOperator.EXISTS
                else not exists,
                operator=operator.value,
                left_type="reference",
                references=references,
            )

        left = self._resolve_operand(condition.left, context)
        right = (
            None
            if operator.value in _UNARY_OPERATORS
            else self._resolve_operand(condition.right, context)
        )
        matched = self._compare(operator, left, right)
        return ExecutionConditionResult(
            matched=matched,
            operator=operator.value,
            left_type=type(left).__name__,
            right_type=None if operator.value in _UNARY_OPERATORS else type(right).__name__,
            references=references,
        )

    def _resolve_operand(
        self,
        value: object,
        context: ExecutionContext,
    ) -> object:
        try:
            return self._resolver.resolve_value(value, context)
        except ParameterResolutionError as error:
            raise ExecutionConditionEvaluationError(str(error)) from error

    def _operand_exists(
        self,
        value: object,
        context: ExecutionContext,
    ) -> bool:
        if isinstance(value, ExecutionVariableReference):
            if not context.has_variable(value.name):
                return False
            self._resolve_operand(value, context)
            return True
        if isinstance(value, StepOutputReference):
            state = context.state_for_step(value.step_id)
            if state == ExecutionStepState.SKIPPED.value:
                raise ReferencedStepSkippedError(
                    f"Referenced step '{value.step_id}' was skipped."
                )
            if not context.has_result(value.step_id):
                return False
            self._resolve_operand(value, context)
            return True
        return self._resolve_operand(value, context) is not None

    def _compare(
        self,
        operator: ExecutionConditionOperator,
        left: object,
        right: object,
    ) -> bool:
        if operator is ExecutionConditionOperator.EQUALS:
            return left == right
        if operator is ExecutionConditionOperator.NOT_EQUALS:
            return left != right
        if operator is ExecutionConditionOperator.IS_NONE:
            return left is None
        if operator is ExecutionConditionOperator.IS_NOT_NONE:
            return left is not None
        if operator is ExecutionConditionOperator.TRUTHY:
            return bool(left)
        if operator is ExecutionConditionOperator.FALSY:
            return not bool(left)
        if operator is ExecutionConditionOperator.GREATER_THAN:
            return _number(left, "left") > _number(right, "right")
        if operator is ExecutionConditionOperator.GREATER_THAN_OR_EQUAL:
            return _number(left, "left") >= _number(right, "right")
        if operator is ExecutionConditionOperator.LESS_THAN:
            return _number(left, "left") < _number(right, "right")
        if operator is ExecutionConditionOperator.LESS_THAN_OR_EQUAL:
            return _number(left, "left") <= _number(right, "right")
        if operator is ExecutionConditionOperator.CONTAINS:
            return _contains(left, right)
        if operator is ExecutionConditionOperator.NOT_CONTAINS:
            return not _contains(left, right)
        if operator is ExecutionConditionOperator.IS_EMPTY:
            return _length(left) == 0
        if operator is ExecutionConditionOperator.IS_NOT_EMPTY:
            return _length(left) > 0
        raise UnsupportedConditionOperatorError(
            f"Unsupported execution condition operator: {operator}."
        )


def copy_execution_condition(
    condition: ExecutionCondition | None,
) -> ExecutionCondition | None:
    """Return a defensive copy of an optional condition."""
    if condition is None:
        return None
    if condition.operator.value in _UNARY_OPERATORS:
        return ExecutionCondition(condition.left, condition.operator)
    return ExecutionCondition(condition.left, condition.operator, condition.right)


def _validate_operand(
    value: object,
    path: str,
) -> None:
    try:
        ExecutionArguments({"value": value})
    except InvalidExecutionArgumentError as error:
        raise InvalidExecutionConditionError(f"{path}: {error}") from error


def _number(
    value: object,
    label: str,
) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConditionOperandTypeError(f"{label} operand must be a finite number.")
    if not math.isfinite(float(value)):
        raise ConditionOperandTypeError(f"{label} operand must be finite.")
    return value


def _contains(
    left: object,
    right: object,
) -> bool:
    if isinstance(left, str):
        if not isinstance(right, str):
            raise ConditionOperandTypeError("String contains requires string right operand.")
        return right in left
    if isinstance(left, (list, tuple)):
        return right in left
    if isinstance(left, Mapping):
        return right in left
    raise ConditionOperandTypeError("Contains requires str, list, tuple or mapping.")


def _length(
    value: object,
) -> int:
    if not isinstance(value, (str, list, tuple, Mapping)):
        raise ConditionOperandTypeError("Empty check requires str, list, tuple or mapping.")
    return len(value)


def _condition_references(
    condition: ExecutionCondition,
) -> list[str]:
    references: list[str] = []

    def visit(value: object) -> None:
        if isinstance(value, ExecutionVariableReference):
            references.append(_variable_label(value))
            return
        if isinstance(value, StepOutputReference):
            references.append(_step_label(value))
            return
        if isinstance(value, Mapping):
            for item in value.values():
                visit(item)
            return
        if isinstance(value, (list, tuple)):
            for item in value:
                visit(item)

    visit(condition.left)
    if condition.operator.value not in _UNARY_OPERATORS:
        visit(condition.right)
    return sorted(references)


def _variable_label(
    reference: ExecutionVariableReference,
) -> str:
    if not reference.path:
        return f"variables.{reference.name}"
    return f"variables.{reference.name}:" + ".".join(str(part) for part in reference.path)


def _step_label(
    reference: StepOutputReference,
) -> str:
    if not reference.path:
        return f"steps.{reference.step_id}.output"
    return f"steps.{reference.step_id}.output:" + ".".join(str(part) for part in reference.path)
