"""Controlled execution for validated Atlas execution plans."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
import time
from typing import Any, Callable

from core.execution_plan_validator import PlanValidationResult, plan_signature
from core.execution_retry import RetryPolicy
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
class ExecutionProgress:
    """Safe progress metadata for plan execution."""

    phase: str
    step_id: str | None = None
    step_index: int | None = None
    total_steps: int | None = None
    tool_name: str | None = None
    elapsed_ms: int = 0
    message: str | None = None
    attempt_number: int | None = None
    max_attempts: int | None = None
    retry_reason: str | None = None


@dataclass(frozen=True, slots=True)
class ResumableExecutionState:
    """In-memory state required to safely resume an interrupted execution."""

    objective: str
    original_plan: ExecutionPlan
    validation_result: PlanValidationResult
    validated_plan_signature: str | None
    completed_step_ids: tuple[str, ...]
    pending_step_ids: tuple[str, ...]
    failed_step_ids: tuple[str, ...]
    interrupted_step_id: str | None
    previous_results: dict[str, object] = field(default_factory=dict)
    resumable: bool = False
    interruption_reason: str | None = None
    confirmation_granted: bool = False
    retry_attempts: dict[str, int] = field(default_factory=dict)
    retry_history: dict[str, tuple[dict[str, object], ...]] = field(default_factory=dict)
    metadata: dict[str, object] = field(default_factory=dict)


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
        retry_policy: RetryPolicy | None = None,
    ) -> None:
        self._tool_registry = tool_registry
        self._tool_executor = tool_executor or ToolExecutor(tool_registry)
        self._parameter_resolver = parameter_resolver or ParameterResolver()
        self._retry_policy = retry_policy or RetryPolicy()

    def execute(
        self,
        plan: ExecutionPlan,
        validation_result: PlanValidationResult | None,
        *,
        confirmation_granted: bool = False,
        control: ExecutionControl | None = None,
        on_progress: Callable[[ExecutionProgress], None] | None = None,
    ) -> PlanExecutionResult:
        """Execute a previously validated plan in dependency order."""
        return self._execute_from_checkpoint(
            plan,
            validation_result,
            confirmation_granted=confirmation_granted,
            control=control,
            on_progress=on_progress,
            initial_completed_step_ids=(),
            initial_previous_results={},
            initial_retry_attempts={},
            initial_retry_history={},
        )

    def resume(
        self,
        state: ResumableExecutionState,
        *,
        confirmation_granted: bool = False,
        control: ExecutionControl | None = None,
        on_progress: Callable[[ExecutionProgress], None] | None = None,
    ) -> PlanExecutionResult:
        """Resume an interrupted execution from the first pending step."""
        resume_error = self._resume_precondition_error(state)
        if resume_error is not None:
            return PlanExecutionResult(
                plan_status=PlanExecutionStatus.REJECTED.value,
                success=False,
                error=resume_error,
                requires_confirmation=state.validation_result.requires_confirmation,
                failed=True,
                resumable=False,
                pending_steps=list(state.pending_step_ids),
                current_step=state.interrupted_step_id,
                failure_reason=resume_error,
                error_code=self._precondition_error_code(resume_error),
                metadata={"plan_signature": state.validated_plan_signature},
            )

        return self._execute_from_checkpoint(
            state.original_plan,
            state.validation_result,
            confirmation_granted=confirmation_granted or state.confirmation_granted,
            control=control,
            on_progress=on_progress,
            initial_completed_step_ids=state.completed_step_ids,
            initial_previous_results=state.previous_results,
            initial_retry_attempts=state.retry_attempts,
            initial_retry_history=state.retry_history,
        )

    def _execute_from_checkpoint(
        self,
        plan: ExecutionPlan,
        validation_result: PlanValidationResult | None,
        *,
        confirmation_granted: bool,
        control: ExecutionControl | None,
        on_progress: Callable[[ExecutionProgress], None] | None,
        initial_completed_step_ids: tuple[str, ...],
        initial_previous_results: dict[str, object],
        initial_retry_attempts: dict[str, int],
        initial_retry_history: dict[str, tuple[dict[str, object], ...]],
    ) -> PlanExecutionResult:
        """Execute a validated plan from a known in-memory checkpoint."""
        started = time.perf_counter()
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
        total_steps = len(
            [
                step
                for step in plan.ordered_steps
                if step.status != self._COMPLETED_STEP_STATUS
            ]
        )

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
        for step_id in initial_completed_step_ids:
            if step_id not in completed_steps:
                completed_steps.append(step_id)
        completed: set[str] = set(completed_steps)
        previous_results: dict[str, object] = dict(initial_previous_results)
        retry_attempts: dict[str, int] = dict(initial_retry_attempts)
        retry_history: dict[str, list[dict[str, object]]] = {
            step_id: list(history)
            for step_id, history in initial_retry_history.items()
        }
        step_results: list[StepExecutionResult] = []
        self._emit_progress(
            on_progress,
            "preparing",
            started,
            total_steps=total_steps,
        )

        for index, step in enumerate(plan.ordered_steps):
            if step.id in completed:
                continue

            progress_index = len(completed_steps) + 1
            control_result = self._control_result(
                plan=plan,
                validation_result=validation_result,
                control=control,
                completed_steps=completed_steps,
                step_results=step_results,
                current_index=index,
                started=started,
                on_progress=on_progress,
                step_index=progress_index,
                total_steps=total_steps,
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

            self._emit_progress(
                on_progress,
                "step_started",
                started,
                step=step,
                step_index=progress_index,
                total_steps=total_steps,
            )
            outcome = self._execute_step(
                step,
                plan_signature=validation_result.plan_signature,
                previous_results=previous_results,
                control=control,
                on_progress=on_progress,
                started=started,
                step_index=progress_index,
                total_steps=total_steps,
                retry_attempts=retry_attempts,
                retry_history=retry_history,
            )
            if isinstance(outcome, PlanExecutionResult):
                return outcome
            if outcome.error_code in {
                ExecutionErrorCode.EXECUTION_CANCELLED.value,
                ExecutionErrorCode.EXECUTION_INTERRUPTED.value,
            }:
                return self._retry_stop_result(
                    plan=plan,
                    validation_result=validation_result,
                    completed_steps=completed_steps,
                    step_results=step_results,
                    outcome=outcome,
                    current_index=index,
                )
            step_results.append(outcome)

            if outcome.success:
                completed.add(step.id)
                completed_steps.append(step.id)
                previous_results[step.id] = outcome.output
                self._emit_progress(
                    on_progress,
                    "step_completed",
                    started,
                    step=step,
                    step_index=progress_index,
                    total_steps=total_steps,
                )
                continue

            self._emit_progress(
                on_progress,
                "step_failed",
                started,
                step=step,
                step_index=progress_index,
                total_steps=total_steps,
            )
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

        self._emit_progress(
            on_progress,
            "completed",
            started,
            total_steps=total_steps,
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

        if "signature" in message:
            return ExecutionErrorCode.VALIDATION_MISMATCH.value

        if "serializable" in message:
            return ExecutionErrorCode.INVALID_PLAN.value

        if "status" in message:
            return ExecutionErrorCode.INVALID_PLAN.value

        return ExecutionErrorCode.INVALID_PLAN.value

    def _resume_precondition_error(
        self,
        state: ResumableExecutionState,
    ) -> str | None:
        if not state.resumable:
            return "Execution state is not resumable."

        current_signature = self._safe_plan_signature(state.original_plan)
        if current_signature is None:
            return "Execution plan is not deterministically serializable."

        if state.validated_plan_signature != current_signature:
            return "Resumable execution signature does not match the execution plan."

        if state.validation_result.plan_signature != state.validated_plan_signature:
            return "PlanValidationResult does not match resumable execution signature."

        if not state.validation_result.is_valid:
            return "Cannot resume an invalid execution plan."

        all_step_ids = tuple(step.id for step in state.original_plan.ordered_steps)
        all_step_id_set = set(all_step_ids)
        completed = set(state.completed_step_ids)
        pending = set(state.pending_step_ids)
        failed = set(state.failed_step_ids)

        if not completed.issubset(all_step_id_set):
            return "Resumable execution contains unknown completed steps."

        if not pending.issubset(all_step_id_set):
            return "Resumable execution contains unknown pending steps."

        if failed:
            return "Failed executions are not resumable."

        if not pending:
            return "Completed executions are not resumable."

        if completed & pending:
            return "Resumable execution has inconsistent completed and pending steps."

        expected_pending = tuple(step_id for step_id in all_step_ids if step_id not in completed)
        if tuple(state.pending_step_ids) != expected_pending:
            return "Resumable execution pending steps are inconsistent."

        if state.interrupted_step_id not in state.pending_step_ids:
            return "Interrupted step is not pending."

        for step in state.original_plan.ordered_steps:
            if step.id not in completed:
                continue
            if step.tool in self._LOGICAL_TOOLS:
                continue
            if step.id not in state.previous_results:
                return "Completed step result is missing."

        return None

    def _control_result(
        self,
        *,
        plan: ExecutionPlan,
        validation_result: PlanValidationResult,
        control: ExecutionControl | None,
        completed_steps: list[str],
        step_results: list[StepExecutionResult],
        current_index: int,
        started: float,
        on_progress: Callable[[ExecutionProgress], None] | None,
        step_index: int,
        total_steps: int,
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
                started=started,
                on_progress=on_progress,
                step_index=step_index,
                total_steps=total_steps,
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
                started=started,
                on_progress=on_progress,
                step_index=step_index,
                total_steps=total_steps,
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
        started: float,
        on_progress: Callable[[ExecutionProgress], None] | None,
        step_index: int,
        total_steps: int,
    ) -> PlanExecutionResult:
        current_step = plan.ordered_steps[current_index]
        self._emit_progress(
            on_progress,
            "cancelled" if cancelled else "interrupted",
            started,
            step=current_step,
            step_index=step_index,
            total_steps=total_steps,
        )
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
        control: ExecutionControl | None,
        on_progress: Callable[[ExecutionProgress], None] | None,
        started: float,
        step_index: int,
        total_steps: int,
        retry_attempts: dict[str, int],
        retry_history: dict[str, list[dict[str, object]]],
    ) -> StepExecutionResult | PlanExecutionResult:
        if step.tool in self._LOGICAL_TOOLS:
            return StepExecutionResult(
                step_id=step.id,
                status=StepExecutionStatus.COMPLETED.value,
                success=True,
                tool_name=step.tool,
                output=None,
                error=None,
                metadata={
                    "logical_step": True,
                    "attempt_number": 1,
                    "max_attempts": 1,
                    "retry_history": [],
                },
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
                metadata={
                    "attempt_number": 1,
                    "max_attempts": self._retry_policy.max_attempts,
                    "retry_history": [],
                    "retry_scheduled": False,
                    "retry_reason": "non_retryable_error",
                },
            )

        return self._execute_resolved_step_with_retries(
            step,
            plan_signature=plan_signature,
            previous_results=previous_results,
            resolved_arguments=resolution.resolved_arguments,
            control=control,
            on_progress=on_progress,
            started=started,
            step_index=step_index,
            total_steps=total_steps,
            retry_attempts=retry_attempts,
            retry_history=retry_history,
        )

    def _execute_resolved_step_with_retries(
        self,
        step: ExecutionStep,
        *,
        plan_signature: str | None,
        previous_results: dict[str, object],
        resolved_arguments: dict[str, object],
        control: ExecutionControl | None,
        on_progress: Callable[[ExecutionProgress], None] | None,
        started: float,
        step_index: int,
        total_steps: int,
        retry_attempts: dict[str, int],
        retry_history: dict[str, list[dict[str, object]]],
    ) -> StepExecutionResult | PlanExecutionResult:
        attempt_number = retry_attempts.get(step.id, 0) + 1
        history = retry_history.setdefault(step.id, [])

        while True:
            retry_attempts[step.id] = attempt_number
            outcome = self._execute_resolved_step_once(
                step,
                plan_signature=plan_signature,
                previous_results=previous_results,
                resolved_arguments=resolved_arguments,
                attempt_number=attempt_number,
                history=history,
            )

            if outcome.success:
                metadata = dict(outcome.metadata)
                metadata["attempt_number"] = attempt_number
                metadata["max_attempts"] = self._retry_policy.max_attempts
                metadata["retry_history"] = list(history)
                metadata["completed_after_retry"] = attempt_number > 1
                if attempt_number > 1:
                    self._emit_progress(
                        on_progress,
                        "step_completed_after_retry",
                        started,
                        step=step,
                        step_index=step_index,
                        total_steps=total_steps,
                        attempt_number=attempt_number,
                        max_attempts=self._retry_policy.max_attempts,
                    )
                return StepExecutionResult(
                    step_id=outcome.step_id,
                    status=outcome.status,
                    success=outcome.success,
                    tool_name=outcome.tool_name,
                    output=outcome.output,
                    error=outcome.error,
                    error_code=outcome.error_code,
                    interruption_reason=outcome.interruption_reason,
                    started_at=outcome.started_at,
                    finished_at=outcome.finished_at,
                    metadata=metadata,
                )

            history.append(
                {
                    "attempt_number": attempt_number,
                    "error_code": outcome.error_code,
                    "error": outcome.error,
                }
            )
            decision = self._retry_policy.decide(
                attempt_number=attempt_number,
                error_code=outcome.error_code,
                error=outcome.error,
                metadata=outcome.metadata,
            )
            if not decision.should_retry:
                metadata = dict(outcome.metadata)
                metadata["attempt_number"] = attempt_number
                metadata["max_attempts"] = decision.max_attempts
                metadata["retry_history"] = list(history)
                metadata["retry_scheduled"] = False
                metadata["retry_reason"] = decision.reason
                metadata["retry_exhausted"] = (
                    decision.reason == "max_attempts_reached"
                )
                if metadata["retry_exhausted"]:
                    self._emit_progress(
                        on_progress,
                        "step_retry_exhausted",
                        started,
                        step=step,
                        step_index=step_index,
                        total_steps=total_steps,
                        attempt_number=attempt_number,
                        max_attempts=decision.max_attempts,
                        retry_reason=decision.reason,
                    )
                return StepExecutionResult(
                    step_id=outcome.step_id,
                    status=outcome.status,
                    success=outcome.success,
                    tool_name=outcome.tool_name,
                    output=outcome.output,
                    error=outcome.error,
                    error_code=outcome.error_code,
                    interruption_reason=outcome.interruption_reason,
                    started_at=outcome.started_at,
                    finished_at=outcome.finished_at,
                    metadata=metadata,
                )

            self._emit_progress(
                on_progress,
                "step_retry_scheduled",
                started,
                step=step,
                step_index=step_index,
                total_steps=total_steps,
                attempt_number=decision.attempt_number,
                max_attempts=decision.max_attempts,
                retry_reason=decision.reason,
            )
            stop_result = self._wait_before_retry(
                control=control,
                delay_ms=decision.delay_ms,
            )
            if stop_result is not None:
                return StepExecutionResult(
                    step_id=step.id,
                    status=stop_result,
                    success=False,
                    tool_name=step.tool,
                    output=None,
                    error=None,
                    error_code=(
                        ExecutionErrorCode.EXECUTION_CANCELLED.value
                        if stop_result == StepExecutionStatus.CANCELLED.value
                        else ExecutionErrorCode.EXECUTION_INTERRUPTED.value
                    ),
                    interruption_reason="Execution stopped before retry.",
                    metadata={
                        "attempt_number": attempt_number,
                        "max_attempts": decision.max_attempts,
                        "retry_history": list(history),
                        "retry_scheduled": True,
                        "retry_reason": decision.reason,
                    },
                )
            attempt_number = decision.attempt_number

    def _execute_resolved_step_once(
        self,
        step: ExecutionStep,
        *,
        plan_signature: str | None,
        previous_results: dict[str, object],
        resolved_arguments: dict[str, object],
        attempt_number: int,
        history: list[dict[str, object]],
    ) -> StepExecutionResult:
        assert step.tool is not None

        try:
            output = self._tool_executor.execute(
                step.tool,
                ToolContext(
                    parameters=deepcopy(resolved_arguments),
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
                metadata={"attempt_number": attempt_number},
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
                metadata={
                    "attempt_number": attempt_number,
                    "exception_type": type(error).__name__,
                },
            )

        failed_error = self._failed_output_error(output)
        if failed_error is not None:
            error_code = self._failed_output_error_code(output)
            return StepExecutionResult(
                step_id=step.id,
                status=StepExecutionStatus.FAILED.value,
                success=False,
                tool_name=step.tool,
                output=output,
                error=failed_error,
                error_code=error_code,
                metadata={"attempt_number": attempt_number},
            )

        return StepExecutionResult(
            step_id=step.id,
            status=StepExecutionStatus.COMPLETED.value,
            success=True,
            tool_name=step.tool,
            output=output,
            error=None,
            metadata={
                "attempt_number": attempt_number,
                "retry_history": list(history),
            },
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

    def _retry_stop_result(
        self,
        *,
        plan: ExecutionPlan,
        validation_result: PlanValidationResult,
        completed_steps: list[str],
        step_results: list[StepExecutionResult],
        outcome: StepExecutionResult,
        current_index: int,
    ) -> PlanExecutionResult:
        cancelled = outcome.error_code == ExecutionErrorCode.EXECUTION_CANCELLED.value
        status = (
            PlanExecutionStatus.CANCELLED.value
            if cancelled
            else PlanExecutionStatus.INTERRUPTED.value
        )
        remaining_status = (
            StepExecutionStatus.CANCELLED.value
            if cancelled
            else StepExecutionStatus.NOT_STARTED.value
        )
        return PlanExecutionResult(
            plan_status=status,
            success=False,
            completed_steps=completed_steps,
            failed_step=None,
            skipped_steps=[],
            pending_steps=self._remaining_step_ids(plan.ordered_steps, current_index),
            step_results=step_results
            + [outcome]
            + self._not_executed_results(
                plan.ordered_steps,
                current_index + 1,
                remaining_status,
            ),
            error=None if cancelled else outcome.interruption_reason,
            requires_confirmation=validation_result.requires_confirmation,
            interrupted=not cancelled,
            cancelled=cancelled,
            failed=False,
            resumable=not cancelled,
            current_step=outcome.step_id,
            interruption_reason=outcome.interruption_reason,
            error_code=outcome.error_code,
            metadata={"plan_signature": validation_result.plan_signature},
        )

    def _emit_progress(
        self,
        on_progress: Callable[[ExecutionProgress], None] | None,
        phase: str,
        started: float,
        *,
        step: ExecutionStep | None = None,
        step_index: int | None = None,
        total_steps: int | None = None,
        attempt_number: int | None = None,
        max_attempts: int | None = None,
        retry_reason: str | None = None,
    ) -> None:
        if on_progress is None:
            return

        on_progress(
            ExecutionProgress(
                phase=phase,
                step_id=step.id if step is not None else None,
                step_index=step_index,
                total_steps=total_steps,
                tool_name=step.tool if step is not None else None,
                elapsed_ms=int((time.perf_counter() - started) * 1000),
                attempt_number=attempt_number,
                max_attempts=max_attempts,
                retry_reason=retry_reason,
            )
        )

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

    def _failed_output_error_code(
        self,
        output: Any,
    ) -> str:
        error_code = getattr(output, "error_code", None)
        if isinstance(error_code, str) and error_code:
            return error_code

        if isinstance(output, dict):
            value = output.get("error_code") or output.get("code")
            if isinstance(value, str) and value:
                return value

        return ExecutionErrorCode.TOOL_EXECUTION_FAILED.value

    def _wait_before_retry(
        self,
        *,
        control: ExecutionControl | None,
        delay_ms: int,
    ) -> str | None:
        before = self._retry_control_status(control)
        if before is not None:
            return before

        if delay_ms > 0:
            time.sleep(delay_ms / 1000)

        return self._retry_control_status(control)

    def _retry_control_status(
        self,
        control: ExecutionControl | None,
    ) -> str | None:
        if control is None:
            return None

        try:
            if control.should_cancel is not None and control.should_cancel():
                return StepExecutionStatus.CANCELLED.value
            if control.should_stop is not None and control.should_stop():
                return StepExecutionStatus.INTERRUPTED.value
        except Exception:
            return StepExecutionStatus.INTERRUPTED.value

        return None
