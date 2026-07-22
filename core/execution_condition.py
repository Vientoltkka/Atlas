"""Safe structured execution conditions."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
import math
from typing import TypeAlias

from core.execution_arguments import ExecutionArguments, InvalidExecutionArgumentError
from core.execution_context import ExecutionContext, ExecutionStepState
from core.execution_variable_reference import ExecutionVariableReference
from core.parameter_resolver import ParameterResolver, ParameterResolutionError
from core.step_output_reference import StepOutputReference


_MISSING_RIGHT = object()
MAX_CONDITION_DEPTH = 16
MAX_CONDITION_NODES = 128
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


class CompositeExecutionConditionError(ExecutionConditionError):
    """Base error for composite execution condition failures."""


class InvalidConditionTreeError(CompositeExecutionConditionError):
    """Raised when a condition tree is structurally invalid."""


class ConditionTreeTooDeepError(InvalidConditionTreeError):
    """Raised when a condition tree exceeds the maximum depth."""


class ConditionTreeTooLargeError(InvalidConditionTreeError):
    """Raised when a condition tree exceeds the maximum node count."""


class EmptyConditionGroupError(InvalidConditionTreeError):
    """Raised when a composite group has no children."""


class InvalidConditionNodeError(InvalidConditionTreeError):
    """Raised when a composite contains a non-condition node."""


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


ExecutionConditionNode: TypeAlias = (
    "ExecutionCondition | AllOfCondition | AnyOfCondition | NotCondition"
)


@dataclass(frozen=True, slots=True)
class AllOfCondition:
    """Composite condition that matches only when every child matches."""

    conditions: tuple[ExecutionConditionNode, ...]

    def __post_init__(self) -> None:
        normalized = _normalize_condition_group(
            self.conditions,
            node_kind="all_of_condition",
        )
        object.__setattr__(self, "conditions", normalized)
        validate_condition_tree(self)


@dataclass(frozen=True, slots=True)
class AnyOfCondition:
    """Composite condition that matches when at least one child matches."""

    conditions: tuple[ExecutionConditionNode, ...]

    def __post_init__(self) -> None:
        normalized = _normalize_condition_group(
            self.conditions,
            node_kind="any_of_condition",
        )
        object.__setattr__(self, "conditions", normalized)
        validate_condition_tree(self)


@dataclass(frozen=True, slots=True)
class NotCondition:
    """Composite condition that negates one child condition."""

    condition: ExecutionConditionNode

    def __post_init__(self) -> None:
        if not is_execution_condition_node(self.condition):
            raise InvalidConditionNodeError(
                "node_kind=not_condition path=condition depth=1 "
                "reason=condition must be a valid condition node"
            )
        object.__setattr__(self, "condition", copy_execution_condition(self.condition))
        validate_condition_tree(self)


@dataclass(frozen=True, slots=True)
class ExecutionConditionResult:
    """Safe condition evaluation result without operand values."""

    matched: bool
    operator: str
    left_type: str | None = None
    right_type: str | None = None
    error_code: str | None = None
    references: tuple[str, ...] = ()
    evaluated_nodes: int = 1
    skipped_nodes_due_to_short_circuit: int = 0
    condition_kind: str = "simple"


class ExecutionConditionEvaluator:
    """Evaluate structured conditions without mutating execution state."""

    def __init__(
        self,
        resolver: ParameterResolver | None = None,
    ) -> None:
        self._resolver = resolver or ParameterResolver()

    def evaluate(
        self,
        condition: ExecutionConditionNode,
        context: ExecutionContext,
    ) -> ExecutionConditionResult:
        validate_condition_tree(condition)
        return self._evaluate_node(condition, context)

    def _evaluate_node(
        self,
        condition: ExecutionConditionNode,
        context: ExecutionContext,
    ) -> ExecutionConditionResult:
        if isinstance(condition, AllOfCondition):
            return self._evaluate_all(condition, context)
        if isinstance(condition, AnyOfCondition):
            return self._evaluate_any(condition, context)
        if isinstance(condition, NotCondition):
            return self._evaluate_not(condition, context)
        return self._evaluate_simple(condition, context)

    def _evaluate_simple(
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
                condition_kind="simple",
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
            condition_kind="simple",
        )

    def _evaluate_all(
        self,
        condition: AllOfCondition,
        context: ExecutionContext,
    ) -> ExecutionConditionResult:
        evaluated = 1
        skipped = 0
        references: list[str] = []
        for index, child in enumerate(condition.conditions):
            child_result = self._evaluate_node(child, context)
            evaluated += child_result.evaluated_nodes
            skipped += child_result.skipped_nodes_due_to_short_circuit
            references.extend(child_result.references)
            if not child_result.matched:
                skipped += sum(
                    condition_node_count(item)
                    for item in condition.conditions[index + 1 :]
                )
                return ExecutionConditionResult(
                    matched=False,
                    operator="all",
                    references=tuple(sorted(set(references))),
                    evaluated_nodes=evaluated,
                    skipped_nodes_due_to_short_circuit=skipped,
                    condition_kind="all",
                )
        return ExecutionConditionResult(
            matched=True,
            operator="all",
            references=tuple(sorted(set(references))),
            evaluated_nodes=evaluated,
            skipped_nodes_due_to_short_circuit=skipped,
            condition_kind="all",
        )

    def _evaluate_any(
        self,
        condition: AnyOfCondition,
        context: ExecutionContext,
    ) -> ExecutionConditionResult:
        evaluated = 1
        skipped = 0
        references: list[str] = []
        for index, child in enumerate(condition.conditions):
            child_result = self._evaluate_node(child, context)
            evaluated += child_result.evaluated_nodes
            skipped += child_result.skipped_nodes_due_to_short_circuit
            references.extend(child_result.references)
            if child_result.matched:
                skipped += sum(
                    condition_node_count(item)
                    for item in condition.conditions[index + 1 :]
                )
                return ExecutionConditionResult(
                    matched=True,
                    operator="any",
                    references=tuple(sorted(set(references))),
                    evaluated_nodes=evaluated,
                    skipped_nodes_due_to_short_circuit=skipped,
                    condition_kind="any",
                )
        return ExecutionConditionResult(
            matched=False,
            operator="any",
            references=tuple(sorted(set(references))),
            evaluated_nodes=evaluated,
            skipped_nodes_due_to_short_circuit=skipped,
            condition_kind="any",
        )

    def _evaluate_not(
        self,
        condition: NotCondition,
        context: ExecutionContext,
    ) -> ExecutionConditionResult:
        child_result = self._evaluate_node(condition.condition, context)
        return ExecutionConditionResult(
            matched=not child_result.matched,
            operator="not",
            references=child_result.references,
            evaluated_nodes=1 + child_result.evaluated_nodes,
            skipped_nodes_due_to_short_circuit=child_result.skipped_nodes_due_to_short_circuit,
            condition_kind="not",
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
    condition: ExecutionConditionNode | None,
) -> ExecutionConditionNode | None:
    """Return a defensive copy of an optional condition."""
    if condition is None:
        return None
    if isinstance(condition, AllOfCondition):
        return AllOfCondition(tuple(copy_execution_condition(item) for item in condition.conditions))
    if isinstance(condition, AnyOfCondition):
        return AnyOfCondition(tuple(copy_execution_condition(item) for item in condition.conditions))
    if isinstance(condition, NotCondition):
        return NotCondition(copy_execution_condition(condition.condition))
    if condition.operator.value in _UNARY_OPERATORS:
        return ExecutionCondition(condition.left, condition.operator)
    return ExecutionCondition(condition.left, condition.operator, condition.right)


def is_execution_condition_node(
    value: object,
) -> bool:
    return isinstance(value, (ExecutionCondition, AllOfCondition, AnyOfCondition, NotCondition))


def validate_condition_tree(
    condition: ExecutionConditionNode,
    *,
    step_id: str | None = None,
) -> None:
    if not is_execution_condition_node(condition):
        raise InvalidConditionNodeError(
            _tree_error_message(
                step_id=step_id,
                node_kind=type(condition).__name__,
                path="condition",
                depth=1,
                reason="invalid condition node",
            )
        )
    stats = condition_tree_stats(condition, step_id=step_id)
    if stats["max_depth"] > MAX_CONDITION_DEPTH:
        raise ConditionTreeTooDeepError(
            _tree_error_message(
                step_id=step_id,
                node_kind=condition_kind(condition),
                path="condition",
                depth=stats["max_depth"],
                reason=f"condition tree depth exceeds {MAX_CONDITION_DEPTH}",
            )
        )
    if stats["node_count"] > MAX_CONDITION_NODES:
        raise ConditionTreeTooLargeError(
            _tree_error_message(
                step_id=step_id,
                node_kind=condition_kind(condition),
                path="condition",
                depth=stats["max_depth"],
                reason=f"condition tree node count exceeds {MAX_CONDITION_NODES}",
            )
        )


def condition_tree_stats(
    condition: ExecutionConditionNode,
    *,
    step_id: str | None = None,
) -> dict[str, int]:
    node_count = 0
    max_depth = 0

    def visit(node: object, path: str, depth: int) -> None:
        nonlocal node_count, max_depth
        if not is_execution_condition_node(node):
            raise InvalidConditionNodeError(
                _tree_error_message(
                    step_id=step_id,
                    node_kind=type(node).__name__,
                    path=path,
                    depth=depth,
                    reason="invalid condition node",
                )
            )
        node_count += 1
        max_depth = max(max_depth, depth)
        if node_count > MAX_CONDITION_NODES:
            return
        if isinstance(node, (AllOfCondition, AnyOfCondition)):
            for index, child in enumerate(node.conditions):
                visit(child, f"{path}.conditions[{index}]", depth + 1)
            return
        if isinstance(node, NotCondition):
            visit(node.condition, f"{path}.condition", depth + 1)

    visit(condition, "condition", 1)
    return {"node_count": node_count, "max_depth": max_depth}


def condition_node_count(
    condition: ExecutionConditionNode,
) -> int:
    return condition_tree_stats(condition)["node_count"]


def condition_kind(
    condition: ExecutionConditionNode,
) -> str:
    if isinstance(condition, AllOfCondition):
        return "all"
    if isinstance(condition, AnyOfCondition):
        return "any"
    if isinstance(condition, NotCondition):
        return "not"
    return "simple"


def iter_condition_operands(
    condition: ExecutionConditionNode,
) -> tuple[object, ...]:
    operands: list[object] = []

    def visit(node: ExecutionConditionNode) -> None:
        if isinstance(node, ExecutionCondition):
            operands.append(node.left)
            if node.operator.value not in _UNARY_OPERATORS:
                operands.append(node.right)
            return
        if isinstance(node, (AllOfCondition, AnyOfCondition)):
            for child in node.conditions:
                visit(child)
            return
        visit(node.condition)

    visit(condition)
    return tuple(operands)


def _normalize_condition_group(
    conditions: object,
    *,
    node_kind: str,
) -> tuple[ExecutionConditionNode, ...]:
    if isinstance(conditions, (str, bytes)) or not isinstance(conditions, (list, tuple)):
        raise InvalidConditionNodeError(
            f"node_kind={node_kind} path=conditions depth=1 "
            "reason=conditions must be a non-empty list or tuple"
        )
    if not conditions:
        raise EmptyConditionGroupError(
            f"node_kind={node_kind} path=conditions depth=1 "
            "reason=condition group cannot be empty"
        )
    normalized: list[ExecutionConditionNode] = []
    for index, condition in enumerate(conditions):
        if not is_execution_condition_node(condition):
            raise InvalidConditionNodeError(
                f"node_kind={node_kind} path=conditions[{index}] depth=2 "
                "reason=invalid condition node"
            )
        copied = copy_execution_condition(condition)
        assert copied is not None
        normalized.append(copied)
    return tuple(normalized)


def _tree_error_message(
    *,
    step_id: str | None,
    node_kind: str,
    path: str,
    depth: int,
    reason: str,
) -> str:
    prefix = f"step_id={step_id} " if step_id is not None else ""
    return f"{prefix}node_kind={node_kind} path={path} depth={depth} reason={reason}"


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
    condition: ExecutionConditionNode,
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

    for operand in iter_condition_operands(condition):
        visit(operand)
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
