"""Controlled execution for validated Atlas execution plans."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field, replace
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


class PartialStepStatus(str, Enum):
    """Closed statuses for partial execution step state."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    INTERRUPTED = "interrupted"
    SKIPPED = "skipped"
    BLOCKED_CONFIRMATION = "blocked_confirmation"


_PARTIAL_PLAN_STATUSES = frozenset(status.value for status in PlanExecutionStatus)
_PARTIAL_STEP_STATUSES = frozenset(status.value for status in PartialStepStatus)


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
class PartialStepExecutionState:
    """Safe state snapshot for one plan step."""

    step_id: str
    tool_name: str | None
    status: str
    attempt_count: int = 0
    started_at: str | None = None
    finished_at: str | None = None
    result: object | None = None
    error_code: str | None = None
    error_message: str | None = None
    retryable: bool = False
    confirmation_required: bool = False

    def __post_init__(self) -> None:
        if self.status not in _PARTIAL_STEP_STATUSES:
            raise ValueError(f"Invalid partial step status: {self.status}")


@dataclass(frozen=True, slots=True)
class PartialExecutionState:
    """Explicit, safe and verifiable snapshot of partial plan execution."""

    objective: str
    original_plan: ExecutionPlan
    validated_plan_signature: str | None
    overall_status: str
    completed_step_ids: tuple[str, ...]
    failed_step_ids: tuple[str, ...]
    interrupted_step_id: str | None
    pending_step_ids: tuple[str, ...]
    skipped_step_ids: tuple[str, ...]
    step_results: tuple[PartialStepExecutionState, ...]
    failure_reason: str | None
    interruption_reason: str | None
    resumable: bool
    requires_confirmation: bool
    retry_attempts: dict[str, int] = field(default_factory=dict)
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_partial_execution_state(self)


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
    partial_state: PartialExecutionState | None = None

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
            return self._finalize_result(
                state.original_plan,
                state.validation_result,
                PlanExecutionResult(
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
                ),
                objective=state.objective,
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
            return self._finalize_result(
                plan,
                validation_result,
                PlanExecutionResult(
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
                ),
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
            return self._finalize_result(
                plan,
                validation_result,
                PlanExecutionResult(
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
                ),
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
                return self._finalize_result(
                    plan,
                    validation_result,
                    PlanExecutionResult(
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
                    ),
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
            return self._finalize_result(
                plan,
                validation_result,
                PlanExecutionResult(
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
                ),
            )

        self._emit_progress(
            on_progress,
            "completed",
            started,
            total_steps=total_steps,
        )
        return self._finalize_result(
            plan,
            validation_result,
            PlanExecutionResult(
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
            ),
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
            return self._finalize_result(
                plan,
                validation_result,
                PlanExecutionResult(
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
                ),
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

        return self._finalize_result(
            plan,
            validation_result,
            PlanExecutionResult(
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
            ),
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
        return self._finalize_result(
            plan,
            validation_result,
            PlanExecutionResult(
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
            ),
        )

    def _finalize_result(
        self,
        plan: ExecutionPlan,
        validation_result: PlanValidationResult | None,
        result: PlanExecutionResult,
        *,
        objective: str | None = None,
    ) -> PlanExecutionResult:
        partial_state = build_partial_execution_state(
            objective=objective or plan.goal,
            plan=plan,
            validation_result=validation_result,
            execution=result,
        )
        return replace(result, partial_state=partial_state)

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


def build_partial_execution_state(
    *,
    objective: str,
    plan: ExecutionPlan,
    validation_result: PlanValidationResult | None,
    execution: PlanExecutionResult,
) -> PartialExecutionState:
    """Build a safe partial execution state from the executor result."""
    step_by_id = {step.id: step for step in plan.ordered_steps}
    raw_result_by_id = {result.step_id: result for result in execution.step_results}
    ordered_step_ids = tuple(step.id for step in plan.ordered_steps)
    raw_failed_step_ids = tuple(execution.failed_steps)
    if (
        not raw_failed_step_ids
        and execution.plan_status == PlanExecutionStatus.FAILED.value
        and execution.current_step is not None
    ):
        raw_failed_step_ids = (execution.current_step,)
    completed = tuple(step_id for step_id in ordered_step_ids if step_id in execution.completed_steps)
    failed = tuple(step_id for step_id in ordered_step_ids if step_id in raw_failed_step_ids)
    skipped = tuple(
        step_id
        for step_id in ordered_step_ids
        if step_id in execution.skipped_steps and step_id not in failed
    )
    pending = tuple(
        step_id
        for step_id in ordered_step_ids
        if step_id not in completed
        and step_id not in failed
        and step_id not in skipped
        and step_id != execution.current_step
    )
    if execution.plan_status in {
        PlanExecutionStatus.INTERRUPTED.value,
        PlanExecutionStatus.CANCELLED.value,
    }:
        pending = tuple(
            step_id
            for step_id in execution.pending_steps
            if step_id in step_by_id and step_id != execution.current_step
        )
    elif execution.pending_steps:
        pending = tuple(step_id for step_id in execution.pending_steps if step_id in step_by_id)

    partial_steps: list[PartialStepExecutionState] = []
    retry_attempts: dict[str, int] = {}
    for step in plan.ordered_steps:
        raw_result = raw_result_by_id.get(step.id)
        if raw_result is None:
            status = _implicit_partial_step_status(step.id, execution)
            partial_steps.append(
                PartialStepExecutionState(
                    step_id=step.id,
                    tool_name=step.tool,
                    status=status,
                    attempt_count=0,
                    confirmation_required=(
                        execution.plan_status
                        == PlanExecutionStatus.BLOCKED_CONFIRMATION.value
                        and step.id == execution.current_step
                    ),
                )
            )
            continue

        attempt_count = _attempt_count(raw_result)
        if attempt_count:
            retry_attempts[step.id] = attempt_count
        partial_steps.append(
            PartialStepExecutionState(
                step_id=step.id,
                tool_name=raw_result.tool_name,
                status=_partial_step_status(raw_result.status, execution, step.id),
                attempt_count=attempt_count,
                started_at=raw_result.started_at,
                finished_at=raw_result.finished_at,
                result=_safe_result_reference(raw_result),
                error_code=raw_result.error_code,
                error_message=_safe_error_message(raw_result.error),
                retryable=_is_retryable(raw_result),
                confirmation_required=(
                    raw_result.error_code
                    == ExecutionErrorCode.CONFIRMATION_REQUIRED.value
                ),
            )
        )

    return PartialExecutionState(
        objective=objective,
        original_plan=plan,
        validated_plan_signature=(
            validation_result.plan_signature
            if validation_result is not None
            else _metadata_signature(execution)
        ),
        overall_status=execution.plan_status,
        completed_step_ids=completed,
        failed_step_ids=failed,
        interrupted_step_id=(
            execution.current_step
            if execution.plan_status
            in {PlanExecutionStatus.INTERRUPTED.value, PlanExecutionStatus.CANCELLED.value}
            else None
        ),
        pending_step_ids=pending,
        skipped_step_ids=skipped,
        step_results=tuple(partial_steps),
        failure_reason=_safe_error_message(execution.failure_reason or execution.error),
        interruption_reason=_safe_error_message(execution.interruption_reason),
        resumable=execution.resumable,
        requires_confirmation=execution.requires_confirmation,
        retry_attempts=retry_attempts,
        metadata=_partial_metadata(execution),
    )


def partial_execution_state_to_dict(
    state: PartialExecutionState,
) -> dict[str, object]:
    """Serialize a partial state without model raw responses or tool outputs."""
    return {
        "objective": state.objective,
        "validated_plan_signature": state.validated_plan_signature,
        "overall_status": state.overall_status,
        "completed_step_ids": list(state.completed_step_ids),
        "failed_step_ids": list(state.failed_step_ids),
        "interrupted_step_id": state.interrupted_step_id,
        "pending_step_ids": list(state.pending_step_ids),
        "skipped_step_ids": list(state.skipped_step_ids),
        "step_results": [
            {
                "step_id": step.step_id,
                "tool_name": step.tool_name,
                "status": step.status,
                "attempt_count": step.attempt_count,
                "started_at": step.started_at,
                "finished_at": step.finished_at,
                "result": step.result,
                "error_code": step.error_code,
                "error_message": step.error_message,
                "retryable": step.retryable,
                "confirmation_required": step.confirmation_required,
            }
            for step in state.step_results
        ],
        "failure_reason": state.failure_reason,
        "interruption_reason": state.interruption_reason,
        "resumable": state.resumable,
        "requires_confirmation": state.requires_confirmation,
        "retry_attempts": dict(state.retry_attempts),
        "metadata": dict(state.metadata),
    }


def partial_execution_state_from_dict(
    payload: dict[str, object],
    *,
    original_plan: ExecutionPlan,
) -> PartialExecutionState:
    """Load and validate a serialized partial state."""
    step_payloads = payload.get("step_results")
    if not isinstance(step_payloads, list):
        raise ValueError("Partial execution step_results must be a list.")

    return PartialExecutionState(
        objective=_payload_str(payload, "objective"),
        original_plan=original_plan,
        validated_plan_signature=_payload_optional_str(
            payload,
            "validated_plan_signature",
        ),
        overall_status=_payload_str(payload, "overall_status"),
        completed_step_ids=_payload_str_tuple(payload, "completed_step_ids"),
        failed_step_ids=_payload_str_tuple(payload, "failed_step_ids"),
        interrupted_step_id=_payload_optional_str(payload, "interrupted_step_id"),
        pending_step_ids=_payload_str_tuple(payload, "pending_step_ids"),
        skipped_step_ids=_payload_str_tuple(payload, "skipped_step_ids"),
        step_results=tuple(
            PartialStepExecutionState(
                step_id=_payload_str(step_payload, "step_id"),
                tool_name=_payload_optional_str(step_payload, "tool_name"),
                status=_payload_str(step_payload, "status"),
                attempt_count=_payload_int(step_payload, "attempt_count"),
                started_at=_payload_optional_str(step_payload, "started_at"),
                finished_at=_payload_optional_str(step_payload, "finished_at"),
                result=step_payload.get("result"),
                error_code=_payload_optional_str(step_payload, "error_code"),
                error_message=_payload_optional_str(step_payload, "error_message"),
                retryable=_payload_bool(step_payload, "retryable"),
                confirmation_required=_payload_bool(
                    step_payload,
                    "confirmation_required",
                ),
            )
            for step_payload in step_payloads
            if isinstance(step_payload, dict)
        ),
        failure_reason=_payload_optional_str(payload, "failure_reason"),
        interruption_reason=_payload_optional_str(payload, "interruption_reason"),
        resumable=_payload_bool(payload, "resumable"),
        requires_confirmation=_payload_bool(payload, "requires_confirmation"),
        retry_attempts=_payload_int_mapping(payload, "retry_attempts"),
        metadata=_payload_dict(payload, "metadata"),
    )


def _validate_partial_execution_state(
    state: PartialExecutionState,
) -> None:
    if state.overall_status not in _PARTIAL_PLAN_STATUSES:
        raise ValueError(f"Invalid partial execution status: {state.overall_status}")

    plan_step_ids = tuple(step.id for step in state.original_plan.ordered_steps)
    plan_step_set = set(plan_step_ids)
    buckets = (
        state.completed_step_ids,
        state.failed_step_ids,
        state.pending_step_ids,
        state.skipped_step_ids,
    )
    for bucket in buckets:
        _reject_duplicates(bucket, "step state bucket")
        if not set(bucket).issubset(plan_step_set):
            raise ValueError("Partial execution state contains unknown step IDs.")

    listed: set[str] = set()
    for bucket in buckets:
        current = set(bucket)
        if listed & current:
            raise ValueError("Partial execution state has contradictory step IDs.")
        listed.update(current)

    if state.interrupted_step_id is not None:
        if state.interrupted_step_id not in plan_step_set:
            raise ValueError("Interrupted step is unknown.")
        if state.interrupted_step_id in listed:
            raise ValueError("Interrupted step has contradictory status.")

    step_result_ids = tuple(step.step_id for step in state.step_results)
    _reject_duplicates(step_result_ids, "partial step results")
    if step_result_ids != plan_step_ids:
        raise ValueError("Partial step results must preserve the plan order.")

    if state.overall_status == PlanExecutionStatus.COMPLETED.value:
        if state.pending_step_ids or state.failed_step_ids or state.skipped_step_ids:
            raise ValueError("Completed execution cannot contain unfinished steps.")

    if state.overall_status == PlanExecutionStatus.PARTIALLY_COMPLETED.value:
        if not state.completed_step_ids:
            raise ValueError("Partially completed execution requires completed steps.")
        unfinished = (
            state.failed_step_ids
            or state.pending_step_ids
            or state.skipped_step_ids
            or state.interrupted_step_id
        )
        if not unfinished:
            raise ValueError("Partially completed execution requires unfinished steps.")

    if state.overall_status == PlanExecutionStatus.INTERRUPTED.value:
        if state.interrupted_step_id is None and not state.interruption_reason:
            raise ValueError("Interrupted execution requires a step or reason.")

    if state.overall_status == PlanExecutionStatus.FAILED.value:
        if not state.failed_step_ids:
            raise ValueError("Failed execution requires at least one failed step.")

    if state.overall_status == PlanExecutionStatus.BLOCKED_CONFIRMATION.value:
        blocked = [
            step
            for step in state.step_results
            if step.status == PartialStepStatus.BLOCKED_CONFIRMATION.value
        ]
        if not blocked:
            raise ValueError("Blocked confirmation requires a blocked step.")
        if any(step.attempt_count for step in blocked):
            raise ValueError("Blocked confirmation step must not execute.")


def _implicit_partial_step_status(
    step_id: str,
    execution: PlanExecutionResult,
) -> str:
    if step_id in execution.completed_steps:
        return PartialStepStatus.COMPLETED.value
    if step_id in execution.failed_steps:
        return PartialStepStatus.FAILED.value
    if step_id in execution.skipped_steps:
        return PartialStepStatus.SKIPPED.value
    if (
        execution.plan_status == PlanExecutionStatus.BLOCKED_CONFIRMATION.value
        and step_id == execution.current_step
    ):
        return PartialStepStatus.BLOCKED_CONFIRMATION.value
    if (
        execution.plan_status
        in {PlanExecutionStatus.INTERRUPTED.value, PlanExecutionStatus.CANCELLED.value}
        and step_id == execution.current_step
    ):
        return PartialStepStatus.INTERRUPTED.value
    return PartialStepStatus.PENDING.value


def _partial_step_status(
    status: str,
    execution: PlanExecutionResult,
    step_id: str,
) -> str:
    if (
        execution.plan_status == PlanExecutionStatus.FAILED.value
        and execution.current_step == step_id
    ):
        return PartialStepStatus.FAILED.value
    if status == StepExecutionStatus.COMPLETED.value:
        return PartialStepStatus.COMPLETED.value
    if status == StepExecutionStatus.FAILED.value:
        return PartialStepStatus.FAILED.value
    if status == StepExecutionStatus.SKIPPED.value:
        return PartialStepStatus.SKIPPED.value
    if status in {
        StepExecutionStatus.INTERRUPTED.value,
        StepExecutionStatus.CANCELLED.value,
    }:
        return PartialStepStatus.INTERRUPTED.value
    if status == StepExecutionStatus.BLOCKED.value:
        return PartialStepStatus.BLOCKED_CONFIRMATION.value
    if status == StepExecutionStatus.NOT_STARTED.value:
        return _implicit_partial_step_status(step_id, execution)
    return PartialStepStatus.PENDING.value


def _attempt_count(
    result: StepExecutionResult,
) -> int:
    value = result.metadata.get("attempt_number")
    if isinstance(value, int) and value >= 0:
        return value
    return 0


def _safe_result_reference(
    result: StepExecutionResult,
) -> object | None:
    if not result.success:
        return None
    return {"result_ref": result.step_id}


def _safe_error_message(
    value: str | None,
) -> str | None:
    if not value:
        return None
    normalized = " ".join(value.split())
    return normalized[:300]


def _is_retryable(
    result: StepExecutionResult,
) -> bool:
    if result.success:
        return False
    reason = result.metadata.get("retry_reason")
    exhausted = result.metadata.get("retry_exhausted")
    scheduled = result.metadata.get("retry_scheduled")
    return bool(scheduled or exhausted or reason == "max_attempts_reached")


def _metadata_signature(
    execution: PlanExecutionResult,
) -> str | None:
    value = execution.metadata.get("plan_signature")
    return value if isinstance(value, str) else None


def _partial_metadata(
    execution: PlanExecutionResult,
) -> dict[str, object]:
    safe: dict[str, object] = {}
    if execution.error_code:
        safe["error_code"] = execution.error_code
    safe["completed_count"] = len(execution.completed_steps)
    safe["failed_count"] = len(execution.failed_steps)
    safe["pending_count"] = len(execution.pending_steps)
    safe["skipped_count"] = len(execution.skipped_steps)
    return safe


def _reject_duplicates(
    values: tuple[str, ...],
    label: str,
) -> None:
    if len(values) != len(set(values)):
        raise ValueError(f"Partial execution state contains duplicate {label}.")


def _payload_str(
    payload: dict[str, object],
    key: str,
) -> str:
    value = payload.get(key)
    if not isinstance(value, str):
        raise ValueError(f"Partial execution field '{key}' must be a string.")
    return value


def _payload_optional_str(
    payload: dict[str, object],
    key: str,
) -> str | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"Partial execution field '{key}' must be a string or null.")
    return value


def _payload_bool(
    payload: dict[str, object],
    key: str,
) -> bool:
    value = payload.get(key)
    if not isinstance(value, bool):
        raise ValueError(f"Partial execution field '{key}' must be a boolean.")
    return value


def _payload_int(
    payload: dict[str, object],
    key: str,
) -> int:
    value = payload.get(key)
    if not isinstance(value, int):
        raise ValueError(f"Partial execution field '{key}' must be an integer.")
    return value


def _payload_dict(
    payload: dict[str, object],
    key: str,
) -> dict[str, object]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Partial execution field '{key}' must be an object.")
    return dict(value)


def _payload_str_tuple(
    payload: dict[str, object],
    key: str,
) -> tuple[str, ...]:
    value = payload.get(key)
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"Partial execution field '{key}' must be a list of strings.")
    return tuple(value)


def _payload_int_mapping(
    payload: dict[str, object],
    key: str,
) -> dict[str, int]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Partial execution field '{key}' must be an object.")
    result: dict[str, int] = {}
    for item_key, item_value in value.items():
        if not isinstance(item_key, str) or not isinstance(item_value, int):
            raise ValueError(
                f"Partial execution field '{key}' must map strings to integers."
            )
        result[item_key] = item_value
    return result
