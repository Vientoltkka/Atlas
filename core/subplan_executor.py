"""Hierarchical subplan execution for Atlas execution plans."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable, TYPE_CHECKING

from core.execution_arguments import ExecutionArguments, InvalidExecutionArgumentError
from core.execution_context import ExecutionContext
from core.execution_plan_validator import ExecutionPlanValidator
from core.planner import ExecutionPlan

if TYPE_CHECKING:
    from core.execution_plan_executor import PlanExecutionResult


MAX_SUBPLAN_DEPTH = ExecutionPlanValidator.MAX_SUBPLAN_DEPTH


class SubplanExecutionError(RuntimeError):
    """Base error for contextual subplan execution failures."""


class InvalidSubplanStepError(SubplanExecutionError):
    """Raised when a parent step is not a valid subplan step."""


class SubplanDepthExceededError(SubplanExecutionError):
    """Raised when hierarchical execution exceeds the configured limit."""


class RecursiveSubplanError(SubplanExecutionError):
    """Raised when a subplan object repeats in the active execution branch."""


class SubplanValidationError(SubplanExecutionError):
    """Raised when a subplan fails static validation at execution time."""


class SubplanFailedError(SubplanExecutionError):
    """Raised when a subplan fails its execution contract."""


class SubplanCancelledError(SubplanExecutionError):
    """Raised when a subplan is cancelled."""


class SubplanCheckpointUnsupportedError(SubplanExecutionError):
    """Documents that active nested checkpoints are not supported in this phase."""


@dataclass(frozen=True, slots=True)
class SubplanExecutionResult:
    """Structured result returned to the parent plan step."""

    parent_execution_id: str
    parent_step_id: str
    child_execution_id: str
    status: str
    output: object | None
    child_result: "PlanExecutionResult"
    depth: int


class SubplanExecutor:
    """Execute one child ExecutionPlan without routing it through ToolExecutor."""

    def __init__(
        self,
        *,
        validator: ExecutionPlanValidator | None = None,
        executor_factory: Callable[[], Any],
    ) -> None:
        self._validator = validator or ExecutionPlanValidator()
        self._executor_factory = executor_factory

    def execute(
        self,
        *,
        parent_execution_id: str,
        parent_step_id: str,
        subplan: ExecutionPlan,
        parent_context: ExecutionContext,
        resolved_inputs: dict[str, object],
        depth: int,
        plan_stack: tuple[int, ...],
    ) -> SubplanExecutionResult:
        """Validate and execute a child plan with an isolated context."""
        if depth > MAX_SUBPLAN_DEPTH:
            raise SubplanDepthExceededError(
                self._error_context(
                    parent_execution_id=parent_execution_id,
                    parent_step_id=parent_step_id,
                    child_execution_id=None,
                    depth=depth,
                    operation="validate_depth",
                    reason=f"maximum depth {MAX_SUBPLAN_DEPTH} exceeded",
                )
            )

        if id(subplan) in plan_stack:
            raise RecursiveSubplanError(
                self._error_context(
                    parent_execution_id=parent_execution_id,
                    parent_step_id=parent_step_id,
                    child_execution_id=None,
                    depth=depth,
                    operation="detect_recursion",
                    reason="recursive subplan reference detected",
                )
            )

        try:
            inputs = ExecutionArguments(resolved_inputs).as_dict()
        except InvalidExecutionArgumentError as error:
            raise InvalidSubplanStepError(
                self._error_context(
                    parent_execution_id=parent_execution_id,
                    parent_step_id=parent_step_id,
                    child_execution_id=None,
                    depth=depth,
                    operation="prepare_inputs",
                    reason=str(error),
                )
            ) from error

        child_context = ExecutionContext(
            initial_variables=inputs,
            metadata={
                "parent_execution_id": parent_execution_id,
                "parent_step_id": parent_step_id,
                "depth": depth,
            },
        )
        child_execution_id = child_context.execution_id

        validation = self._validator.validate(
            subplan,
            depth=depth,
            plan_stack=plan_stack,
        )
        if not validation.is_valid:
            raise SubplanValidationError(
                self._error_context(
                    parent_execution_id=parent_execution_id,
                    parent_step_id=parent_step_id,
                    child_execution_id=child_execution_id,
                    depth=depth,
                    operation="validate_subplan",
                    reason="; ".join(validation.errors[:3]),
                )
            )

        child_executor = self._executor_factory()
        child_result = child_executor.execute(
            subplan,
            validation,
            confirmation_granted=True,
            execution_context=child_context,
            subplan_depth=depth,
            plan_stack=plan_stack,
        )
        output = self._functional_output(child_result)

        return SubplanExecutionResult(
            parent_execution_id=parent_execution_id,
            parent_step_id=parent_step_id,
            child_execution_id=child_execution_id,
            status=child_result.plan_status,
            output=deepcopy(output),
            child_result=child_result,
            depth=depth,
        )

    def _functional_output(
        self,
        child_result: "PlanExecutionResult",
    ) -> object | None:
        for step_result in reversed(child_result.step_results):
            if step_result.success and step_result.status == "completed":
                return deepcopy(step_result.output)
        return None

    def _error_context(
        self,
        *,
        parent_execution_id: str,
        parent_step_id: str,
        child_execution_id: str | None,
        depth: int,
        operation: str,
        reason: str,
    ) -> str:
        child = child_execution_id or "<not_created>"
        return (
            f"parent_execution_id={parent_execution_id} parent_step_id={parent_step_id} "
            f"child_execution_id={child} depth={depth} operation={operation} "
            f"reason={reason}"
        )
