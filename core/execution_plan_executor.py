"""Controlled execution for validated Atlas execution plans."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

from core.execution_plan_validator import PlanValidationResult, plan_signature
from core.parameter_resolver import ParameterResolver
from core.planner import ExecutionPlan, ExecutionStep
from tools.executor import ToolExecutor
from tools.registry import ToolNotRegisteredError, ToolRegistry
from tools.tool_context import ToolContext


class PlanExecutionStatus(str, Enum):
    """Global statuses for controlled plan execution."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    INTERRUPTED = "interrupted"
    CANCELLED = "cancelled"
    BLOCKED_CONFIRMATION = "blocked_confirmation"
    REJECTED = "rejected"
    PARTIALLY_COMPLETED = "partially_completed"


class StepExecutionStatus(str, Enum):
    """Step statuses for controlled plan execution."""

    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    INTERRUPTED = "interrupted"
    CANCELLED = "cancelled"
    BLOCKED = "blocked"
    NOT_STARTED = "not_started"


class ExecutionErrorCode(str, Enum):
    """Stable error codes for plan execution outcomes."""

    INVALID_PLAN = "INVALID_PLAN"
    VALIDATION_MISMATCH = "VALIDATION_MISMATCH"
    CONFIRMATION_REQUIRED = "CONFIRMATION_REQUIRED"
    TOOL_NOT_FOUND = "TOOL_NOT_FOUND"
    TOOL_EXECUTION_FAILED = "TOOL_EXECUTION_FAILED"
    TOOL_EXCEPTION = "TOOL_EXCEPTION"
    DEPENDENCY_NOT_COMPLETED = "DEPENDENCY_NOT_COMPLETED"
    EXECUTION_INTERRUPTED = "EXECUTION_INTERRUPTED"
    EXECUTION_CANCELLED = "EXECUTION_CANCELLED"
    PARAMETER_RESOLUTION_FAILED = "PARAMETER_RESOLUTION_FAILED"
    INTERNAL_EXECUTOR_ERROR = "INTERNAL_EXECUTOR_ERROR"


@dataclass(frozen=True, slots=True)
class ExecutionControl:
    """Small control surface for interruption or cancellation."""

    should_stop: Callable[[], bool] | None = None
    should_cancel: Callable[[], bool] | None = None
    interruption_reason: str = "Execution interrupted by control signal."
    cancellation_reason: str = "Execution cancelled by control signal."
    interruption_resumable: bool = True


@dataclass(frozen=True, slots=True)
class StepExecutionResult:
    """Structured execution outcome for one plan step."""

    step_id: str
    status: str
    success: bool
    tool_name: str | None
    output: object | None = None
    error: str | None = None
    error_code: str | None = None
    interruption_reason: str | None = None
    started_at: str | None = None
    finished_at: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class PlanExecutionResult:
    """Structured execution outcome for an execution plan."""

    plan_status: str
    success: bool
    completed_steps: list[str] = field(default_factory=list)
    failed_step: str | None = None
    skipped_steps: list[str] = field(default_factory=list)
    step_results: list[StepExecutionResult] = field(default_factory=list)
    error: str | None = None
    requires_confirmation: bool = False
    interrupted: bool = False
    completed: bool = False
    cancelled: bool = False
    failed: bool = False
    blocked: bool = False
    resumable: bool = False
    failed_steps: list[str] = field(default_factory=list)
    pending_steps: list[str] = field(default_factory=list)
    current_step: str | None = None
    interruption_reason: str | None = None
    failure_reason: str | None = None
    error_code: str | None = None
    started_at: str | None = None
    finished_at: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def status(self) -> str:
        """Return the global execution status."""
        return self.plan_status


class ExecutionPlanExecutor:
    """Execute validated plans without planning or changing their structure."""

    _EXECUTABLE_PLAN_STATUSES = {"planned"}
    _COMPLETED_STEP_STATUS = "completed"
    _LOGICAL_TOOLS = {None, "direct_response"}

    def __init__(
        self,
        tool_registry: ToolRegistry,
        tool_executor: ToolExecutor | None = None,
        parameter_resolver: ParameterResolver | None = None,
    ) -> None:
        self._tool_registry = tool_registry
        self._tool_executor = tool_executor or ToolExecutor(tool_registry)
        self._parameter_resolver = parameter_resolver or ParameterResolver()

    def execute(
        self,
        plan: ExecutionPlan,
        validation_result: PlanValidationResult | None,
        *,
        confirmation_granted: bool = False,
        control: ExecutionControl | None = None,
    ) -> PlanExecutionResult:
        """Execute a previously validated plan in dependency order."""
        precondition_error = self._precondition_error(plan, validation_result)
        if precondition_error is not None:
            return PlanExecutionResult(
                plan_status=PlanExecutionStatus.REJECTED.value,
                success=False,
                error=precondition_error,
                requires_confirmation=(
                    bool(validation_result.requires_confirmation)
                    if validation_result is not None
                    else plan.requires_confirmation
                ),
                interrupted=False,
                failed=True,
                error_code=self._precondition_error_code(precondition_error),
                pending_steps=self._pending_step_ids(plan),
                failure_reason=precondition_error,
                metadata={"plan_signature": self._safe_plan_signature(plan)},
            )

        assert validation_result is not None

        if validation_result.requires_confirmation and not confirmation_granted:
            return PlanExecutionResult(
                plan_status=PlanExecutionStatus.BLOCKED_CONFIRMATION.value,
                success=False,
                error="Plan execution requires explicit confirmation.",
                requires_confirmation=True,
                interrupted=False,
                blocked=True,
                resumable=True,
                pending_steps=self._pending_step_ids(plan),
                current_step=self._first_pending_step_id(plan),
                error_code=ExecutionErrorCode.CONFIRMATION_REQUIRED.value,
                metadata={"plan_signature": self._safe_plan_signature(plan)},
            )

        completed_steps = [
            step.id
            for step in plan.ordered_steps
            if step.status == self._COMPLETED_STEP_STATUS
        ]
        completed: set[str] = set(completed_steps)
        previous_results: dict[str, object] = {}
        step_results: list[StepExecutionResult] = []

        for index, step in enumerate(plan.ordered_steps):
            if step.id in completed:
                continue

            control_result = self._control_result(
                plan=plan,
                validation_result=validation_result,
                control=control,
                completed_steps=completed_steps,
                step_results=step_results,
                current_index=index,
            )
            if control_result is not None:
                return control_result

            missing_dependency = self._missing_dependency(step, completed)
            if missing_dependency is not None:
                skipped = self._remaining_step_ids(plan.ordered_steps, index + 1)
                error = (
                    f"Dependency '{missing_dependency}' is not completed "
                    f"for step '{step.id}'."
                )
                return PlanExecutionResult(
                    plan_status=self._failure_status(completed_steps),
                    success=False,
                    completed_steps=completed_steps,
                    failed_step=step.id,
                    failed_steps=[step.id],
                    skipped_steps=skipped,
                    pending_steps=[],
                    step_results=step_results
                    + [
                        StepExecutionResult(
                            step_id=step.id,
                            status=StepExecutionStatus.FAILED.value,
                            success=False,
                            tool_name=step.tool,
                            error=error,
                            error_code=ExecutionErrorCode.DEPENDENCY_NOT_COMPLETED.value,
                        )
                    ]
                    + self._not_executed_results(
                        plan.ordered_steps,
                        index + 1,
                        StepExecutionStatus.SKIPPED.value,
                    ),
                    error=error,
                    requires_confirmation=validation_result.requires_confirmation,
                    interrupted=False,
                    failed=True,
                    resumable=False,
                    current_step=step.id,
                    failure_reason=error,
                    error_code=ExecutionErrorCode.DEPENDENCY_NOT_COMPLETED.value,
                    metadata={"plan_signature": validation_result.plan_signature},
                )

            outcome = self._execute_step(
                step,
                plan_signature=validation_result.plan_signature,
                previous_results=previous_results,
            )
            step_results.append(outcome)

            if outcome.success:
                completed.add(step.id)
                completed_steps.append(step.id)
                previous_results[step.id] = outcome.output
                continue

            return PlanExecutionResult(
                plan_status=self._failure_status(completed_steps),
                success=False,
                completed_steps=completed_steps,
                failed_step=step.id,
                failed_steps=[step.id],
                skipped_steps=self._remaining_step_ids(plan.ordered_steps, index + 1),
                pending_steps=[],
                step_results=step_results
                + self._not_executed_results(
                    plan.ordered_steps,
                    index + 1,
                    StepExecutionStatus.SKIPPED.value,
                ),
                error=outcome.error,
                requires_confirmation=validation_result.requires_confirmation,
                interrupted=False,
                failed=True,
                resumable=False,
                current_step=step.id,
                failure_reason=outcome.error,
                error_code=outcome.error_code,
                metadata={"plan_signature": validation_result.plan_signature},
            )

        return PlanExecutionResult(
            plan_status=PlanExecutionStatus.COMPLETED.value,
            success=True,
            completed_steps=completed_steps,
            failed_step=None,
            skipped_steps=[],
            step_results=step_results,
            error=None,
            requires_confirmation=validation_result.requires_confirmation,
            interrupted=False,
            completed=True,
            failed=False,
            blocked=False,
            resumable=False,
            pending_steps=[],
            metadata={"plan_signature": validation_result.plan_signature},
        )

    def _precondition_error(
        self,
        plan: ExecutionPlan,
        validation_result: PlanValidationResult | None,
    ) -> str | None:
        if validation_result is None:
            return "Plan execution requires an explicit PlanValidationResult."

        if not validation_result.is_valid:
            return "Cannot execute an invalid execution plan."

        if (
            validation_result.plan_signature is not None
        ):
            current_signature = self._safe_plan_signature(plan)
            if current_signature is None:
                return "Execution plan is not deterministically serializable."

            if validation_result.plan_signature != current_signature:
                return "PlanValidationResult does not match the execution plan."

        if plan.status not in self._EXECUTABLE_PLAN_STATUSES:
            return f"Plan status '{plan.status}' is not executable."

        return None

    def _safe_plan_signature(
        self,
        plan: ExecutionPlan,
    ) -> str | None:
        try:
            return plan_signature(plan)
        except TypeError:
            return None

    def _precondition_error_code(
        self,
        message: str,
    ) -> str:
        if "does not match" in message:
            return ExecutionErrorCode.VALIDATION_MISMATCH.value

        if "serializable" in message:
            return ExecutionErrorCode.INVALID_PLAN.value

        if "status" in message:
            return ExecutionErrorCode.INVALID_PLAN.value

        return ExecutionErrorCode.INVALID_PLAN.value

    def _control_result(
        self,
        *,
        plan: ExecutionPlan,
        validation_result: PlanValidationResult,
        control: ExecutionControl | None,
        completed_steps: list[str],
        step_results: list[StepExecutionResult],
        current_index: int,
    ) -> PlanExecutionResult | None:
        if control is None:
            return None

        try:
            should_cancel = (
                control.should_cancel() if control.should_cancel is not None else False
            )
            should_stop = (
                control.should_stop() if control.should_stop is not None else False
            )
        except Exception as error:
            message = f"Internal executor control error: {error}"
            return PlanExecutionResult(
                plan_status=self._failure_status(completed_steps),
                success=False,
                completed_steps=completed_steps,
                failed_step=None,
                skipped_steps=self._remaining_step_ids(
                    plan.ordered_steps,
                    current_index,
                ),
                step_results=step_results
                + self._not_executed_results(
                    plan.ordered_steps,
                    current_index,
                    StepExecutionStatus.SKIPPED.value,
                ),
                error=message,
                requires_confirmation=validation_result.requires_confirmation,
                failed=True,
                resumable=False,
                current_step=plan.ordered_steps[current_index].id,
                failure_reason=message,
                error_code=ExecutionErrorCode.INTERNAL_EXECUTOR_ERROR.value,
                metadata={"exception_type": type(error).__name__},
            )

        if should_cancel:
            return self._controlled_stop_result(
                plan=plan,
                validation_result=validation_result,
                completed_steps=completed_steps,
                step_results=step_results,
                current_index=current_index,
                status=PlanExecutionStatus.CANCELLED.value,
                step_status=StepExecutionStatus.CANCELLED.value,
                reason=control.cancellation_reason,
                error_code=ExecutionErrorCode.EXECUTION_CANCELLED.value,
                resumable=False,
                cancelled=True,
            )

        if should_stop:
            return self._controlled_stop_result(
                plan=plan,
                validation_result=validation_result,
                completed_steps=completed_steps,
                step_results=step_results,
                current_index=current_index,
                status=PlanExecutionStatus.INTERRUPTED.value,
                step_status=StepExecutionStatus.INTERRUPTED.value,
                reason=control.interruption_reason,
                error_code=ExecutionErrorCode.EXECUTION_INTERRUPTED.value,
                resumable=control.interruption_resumable,
                cancelled=False,
            )

        return None

    def _controlled_stop_result(
        self,
        *,
        plan: ExecutionPlan,
        validation_result: PlanValidationResult,
        completed_steps: list[str],
        step_results: list[StepExecutionResult],
        current_index: int,
        status: str,
        step_status: str,
        reason: str,
        error_code: str,
        resumable: bool,
        cancelled: bool,
    ) -> PlanExecutionResult:
        current_step = plan.ordered_steps[current_index]
        pending_steps = self._remaining_step_ids(plan.ordered_steps, current_index)
        current_result = StepExecutionResult(
            step_id=current_step.id,
            status=step_status,
            success=False,
            tool_name=current_step.tool,
            output=None,
            error=None if cancelled else reason,
            error_code=error_code,
            interruption_reason=reason,
        )
        remaining_status = (
            StepExecutionStatus.NOT_STARTED.value
            if step_status == StepExecutionStatus.INTERRUPTED.value
            else StepExecutionStatus.CANCELLED.value
        )

        return PlanExecutionResult(
            plan_status=status,
            success=False,
            completed_steps=completed_steps,
            failed_step=None,
            skipped_steps=[],
            pending_steps=pending_steps,
            step_results=step_results
            + [current_result]
            + self._not_executed_results(
                plan.ordered_steps,
                current_index + 1,
                remaining_status,
            ),
            error=None if cancelled else reason,
            requires_confirmation=validation_result.requires_confirmation,
            interrupted=not cancelled,
            cancelled=cancelled,
            resumable=resumable,
            current_step=current_step.id,
            interruption_reason=reason,
            error_code=error_code,
            metadata={"plan_signature": validation_result.plan_signature},
        )

    def _execute_step(
        self,
        step: ExecutionStep,
        *,
        plan_signature: str | None,
        previous_results: dict[str, object],
    ) -> StepExecutionResult:
        if step.tool in self._LOGICAL_TOOLS:
            return StepExecutionResult(
                step_id=step.id,
                status=StepExecutionStatus.COMPLETED.value,
                success=True,
                tool_name=step.tool,
                output=None,
                error=None,
                metadata={"logical_step": True},
            )

        assert step.tool is not None

        resolution = self._parameter_resolver.resolve(
            step.arguments,
            previous_results,
        )
        if not resolution.success:
            error = "; ".join(resolution.errors)
            return StepExecutionResult(
                step_id=step.id,
                status=StepExecutionStatus.FAILED.value,
                success=False,
                tool_name=step.tool,
                output=None,
                error=error,
                error_code=ExecutionErrorCode.PARAMETER_RESOLUTION_FAILED.value,
                metadata={
                    "parameter_resolution_error_code": resolution.error_code,
                    "unresolved_references": resolution.unresolved_references,
                    "used_step_ids": resolution.used_step_ids,
                },
            )

        if not self._tool_registry.exists(step.tool):
            return StepExecutionResult(
                step_id=step.id,
                status=StepExecutionStatus.FAILED.value,
                success=False,
                tool_name=step.tool,
                output=None,
                error=f"Tool '{step.tool}' is not registered.",
                error_code=ExecutionErrorCode.TOOL_NOT_FOUND.value,
            )

        try:
            output = self._tool_executor.execute(
                step.tool,
                ToolContext(
                    parameters=deepcopy(resolution.resolved_arguments),
                    step_id=step.id,
                    plan_signature=plan_signature,
                    previous_results=dict(previous_results),
                    metadata={"executor": "ExecutionPlanExecutor"},
                ),
            )
        except ToolNotRegisteredError as error:
            return StepExecutionResult(
                step_id=step.id,
                status=StepExecutionStatus.FAILED.value,
                success=False,
                tool_name=step.tool,
                output=None,
                error=str(error),
                error_code=ExecutionErrorCode.TOOL_NOT_FOUND.value,
            )
        except Exception as error:
            return StepExecutionResult(
                step_id=step.id,
                status=StepExecutionStatus.FAILED.value,
                success=False,
                tool_name=step.tool,
                output=None,
                error=str(error),
                error_code=ExecutionErrorCode.TOOL_EXCEPTION.value,
                metadata={"exception_type": type(error).__name__},
            )

        failed_error = self._failed_output_error(output)
        if failed_error is not None:
            return StepExecutionResult(
                step_id=step.id,
                status=StepExecutionStatus.FAILED.value,
                success=False,
                tool_name=step.tool,
                output=output,
                error=failed_error,
                error_code=ExecutionErrorCode.TOOL_EXECUTION_FAILED.value,
            )

        return StepExecutionResult(
            step_id=step.id,
            status=StepExecutionStatus.COMPLETED.value,
            success=True,
            tool_name=step.tool,
            output=output,
            error=None,
        )

    def _missing_dependency(
        self,
        step: ExecutionStep,
        completed: set[str],
    ) -> str | None:
        for dependency in step.dependencies:
            if dependency not in completed:
                return dependency

        return None

    def _remaining_step_ids(
        self,
        steps: tuple[ExecutionStep, ...],
        start_index: int,
    ) -> list[str]:
        return [
            step.id
            for step in steps[start_index:]
            if step.status != self._COMPLETED_STEP_STATUS
        ]

    def _pending_step_ids(
        self,
        plan: ExecutionPlan,
    ) -> list[str]:
        return self._remaining_step_ids(plan.ordered_steps, 0)

    def _first_pending_step_id(
        self,
        plan: ExecutionPlan,
    ) -> str | None:
        pending = self._pending_step_ids(plan)
        return pending[0] if pending else None

    def _not_executed_results(
        self,
        steps: tuple[ExecutionStep, ...],
        start_index: int,
        status: str,
    ) -> list[StepExecutionResult]:
        return [
            StepExecutionResult(
                step_id=step.id,
                status=status,
                success=False,
                tool_name=step.tool,
                output=None,
                error=None,
            )
            for step in steps[start_index:]
            if step.status != self._COMPLETED_STEP_STATUS
        ]

    def _failure_status(
        self,
        completed_steps: list[str],
    ) -> str:
        if completed_steps:
            return PlanExecutionStatus.PARTIALLY_COMPLETED.value

        return PlanExecutionStatus.FAILED.value

    def _failed_output_error(
        self,
        output: Any,
    ) -> str | None:
        success = getattr(output, "success", None)
        if success is False:
            return str(
                getattr(output, "error_message", None)
                or getattr(output, "error", None)
                or "Tool returned an unsuccessful result."
            )

        if isinstance(output, dict) and output.get("success") is False:
            return str(
                output.get("error")
                or output.get("error_message")
                or "Tool returned an unsuccessful result."
            )

        return None
