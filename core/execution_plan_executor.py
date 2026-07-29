"""Controlled execution for validated Atlas execution plans."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field, replace
from enum import Enum
import time
from typing import TYPE_CHECKING, Any, Callable

from core.execution_context import (
    ExecutionContext,
    ExecutionContextSnapshot,
    ExecutionStepState,
)
from core.execution_dependency_checker import (
    ExecutionDependencyChecker,
    ExecutionDependencyCheckResult,
)
from core.execution_condition import (
    AllOfCondition,
    AnyOfCondition,
    ExecutionConditionEvaluationError,
    ExecutionConditionEvaluator,
    ExecutionConditionResult,
    NotCondition,
    condition_kind,
    condition_tree_stats,
    iter_condition_operands,
)
from core.execution_arguments import ExecutionArguments
from core.execution_metrics import ExecutionMetrics, ExecutionMetricsCalculator
from core.goal_verifier import GoalVerificationResult, GoalVerifier
from core.execution_plan_output import (
    ExecutionPlanOutput,
    ExecutionPlanOutputError,
    ExecutionPlanOutputResolutionError,
)
from core.execution_plan_registry import (
    ExecutionPlanReference,
    ExecutionPlanRegistry,
    ExecutionPlanRegistryError,
)
from core.execution_plan_validator import PlanValidationResult, plan_signature
from core.execution_plan_validator import ExecutionPlanValidator
from core.execution_plan_topology import (
    ExecutionDependencyStateInconsistencyError,
    ExecutionPlanTopologicalSorter,
    ExecutionPlanTopologyError,
    TopologicalExecutionOrder,
)
from core.execution_retry import RetryEngine, RetryPolicy, RetryReason, RetryStrategy
from core.execution_trace import ExecutionTrace, TraceEventStatus, TraceStatus
from core.parameter_resolver import ParameterResolver
from core.planner import ExecutionPlan, ExecutionStep
from core.structured_reference_path import (
    StructuredReferencePathError,
    navigate_structured_path,
)
from core.subplan_executor import (
    RecursiveSubplanError,
    SubplanDepthExceededError,
    SubplanExecutionError,
    SubplanExecutor,
    SubplanValidationError,
)
from tools.executor import ToolExecutor
from tools.registry import ToolNotRegisteredError, ToolRegistry
from tools.tool_context import ToolContext
from tools.tool_schema import ToolSchemaValidationException

if TYPE_CHECKING:
    from core.execution_history import ExecutionHistorySink


class PlanExecutionStatus(str, Enum):
    """Global statuses for controlled plan execution."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    INTERRUPTED = "interrupted"
    CANCELLED = "cancelled"
    BLOCKED = "blocked"
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
    BLOCKED = "blocked"
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
    DEPENDENCY_STATE_INCONSISTENCY = "DEPENDENCY_STATE_INCONSISTENCY"
    EXECUTION_INTERRUPTED = "EXECUTION_INTERRUPTED"
    EXECUTION_CANCELLED = "EXECUTION_CANCELLED"
    PARAMETER_RESOLUTION_FAILED = "PARAMETER_RESOLUTION_FAILED"
    TOOL_SCHEMA_VALIDATION_FAILED = "TOOL_SCHEMA_VALIDATION_FAILED"
    EXECUTION_VARIABLE_BINDING_FAILED = "EXECUTION_VARIABLE_BINDING_FAILED"
    EXECUTION_CONDITION_FAILED = "EXECUTION_CONDITION_FAILED"
    INTERNAL_EXECUTOR_ERROR = "INTERNAL_EXECUTOR_ERROR"
    SUBPLAN_VALIDATION_FAILED = "SUBPLAN_VALIDATION_FAILED"
    SUBPLAN_DEPTH_EXCEEDED = "SUBPLAN_DEPTH_EXCEEDED"
    SUBPLAN_RECURSIVE = "SUBPLAN_RECURSIVE"
    SUBPLAN_FAILED = "SUBPLAN_FAILED"
    SUBPLAN_CANCELLED = "SUBPLAN_CANCELLED"
    EXECUTION_PLAN_OUTPUT_RESOLUTION_FAILED = "EXECUTION_PLAN_OUTPUT_RESOLUTION_FAILED"
    EXECUTION_PLAN_REGISTRY_UNAVAILABLE = "EXECUTION_PLAN_REGISTRY_UNAVAILABLE"
    EXECUTION_PLAN_REFERENCE_NOT_FOUND = "EXECUTION_PLAN_REFERENCE_NOT_FOUND"
    REGISTERED_EXECUTION_PLAN_SIGNATURE_MISMATCH = "REGISTERED_EXECUTION_PLAN_SIGNATURE_MISMATCH"
    LOOP_MAX_ITERATIONS_REACHED = "LOOP_MAX_ITERATIONS_REACHED"
    LOOP_CONDITION_FAILED = "LOOP_CONDITION_FAILED"
    LOOP_BODY_FAILED = "LOOP_BODY_FAILED"
    LOOP_BODY_CANCELLED = "LOOP_BODY_CANCELLED"
    LOOP_BODY_BLOCKED = "LOOP_BODY_BLOCKED"


class LoopTerminationReason(str, Enum):
    """Closed termination reasons for controlled execution loops."""

    CONDITION_FALSE = "CONDITION_FALSE"
    MAX_ITERATIONS_REACHED = "MAX_ITERATIONS_REACHED"
    BODY_FAILED = "BODY_FAILED"
    BODY_CANCELLED = "BODY_CANCELLED"
    BODY_BLOCKED = "BODY_BLOCKED"
    CONDITION_EVALUATION_FAILED = "CONDITION_EVALUATION_FAILED"


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
    retry_decisions: dict[str, dict[str, object]] = field(default_factory=dict)
    replanning_policy: object | None = None
    replan_attempts: int = 0
    replanning_history: tuple[object, ...] = ()
    original_plan_signature: str | None = None
    current_plan_signature: str | None = None
    goal_driven_policy: object | None = None
    goal_driven_cycle: int = 0
    goal_driven_history: tuple[object, ...] = ()
    goal_driven_used_signatures: tuple[str, ...] = ()
    goal_driven_last_decision: str | None = None
    goal_driven_terminal_status: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)
    execution_context_snapshot: ExecutionContextSnapshot | None = None
    goal_verification_result: GoalVerificationResult | None = None


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
class LoopExecutionResult:
    """Structured outcome of one controlled loop step attempt."""

    iterations_completed: int
    termination_reason: str
    last_output: object | None
    child_results: tuple[PlanExecutionResult, ...]
    status: str


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
    blocked_step_ids: tuple[str, ...] = ()
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
    blocked_steps: list[str] = field(default_factory=list)
    pending_steps: list[str] = field(default_factory=list)
    current_step: str | None = None
    interruption_reason: str | None = None
    failure_reason: str | None = None
    error_code: str | None = None
    started_at: str | None = None
    finished_at: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)
    partial_state: PartialExecutionState | None = None
    trace: ExecutionTrace | None = None
    metrics: ExecutionMetrics | None = None
    output: object | None = None
    goal_verification_result: GoalVerificationResult | None = None

    @property
    def status(self) -> str:
        """Return the global execution status."""
        return self.plan_status


class ExecutionPlanExecutor:
    """Execute validated plans without planning or changing their structure."""

    _EXECUTABLE_PLAN_STATUSES = {"planned"}
    _COMPLETED_STEP_STATUS = "completed"
    _LOGICAL_TOOLS = {None, "direct_response"}
    _INCONSISTENT_DEPENDENCY_STATES = frozenset(
        {
            ExecutionStepState.PENDING.value,
            ExecutionStepState.RUNNING.value,
        }
    )
    _TERMINAL_UNSATISFIED_DEPENDENCY_STATES = frozenset(
        {
            ExecutionStepState.FAILED.value,
            ExecutionStepState.SKIPPED.value,
            ExecutionStepState.CANCELLED.value,
            ExecutionStepState.BLOCKED.value,
        }
    )

    def __init__(
        self,
        tool_registry: ToolRegistry,
        tool_executor: ToolExecutor | None = None,
        parameter_resolver: ParameterResolver | None = None,
        condition_evaluator: ExecutionConditionEvaluator | None = None,
        dependency_checker: ExecutionDependencyChecker | None = None,
        topological_sorter: ExecutionPlanTopologicalSorter | None = None,
        retry_policy: RetryPolicy | None = None,
        execution_history: ExecutionHistorySink | None = None,
        plan_registry: ExecutionPlanRegistry | None = None,
    ) -> None:
        self._tool_registry = tool_registry
        self._tool_executor = tool_executor or ToolExecutor(tool_registry)
        self._parameter_resolver = parameter_resolver or ParameterResolver()
        self._condition_evaluator = (
            condition_evaluator
            or ExecutionConditionEvaluator(self._parameter_resolver)
        )
        self._dependency_checker = dependency_checker or ExecutionDependencyChecker()
        self._topological_sorter = topological_sorter or ExecutionPlanTopologicalSorter()
        self._retry_policy = retry_policy or RetryPolicy()
        self._retry_engine = RetryEngine()
        self._execution_history = execution_history
        self._plan_registry = plan_registry

    def execute(
        self,
        plan: ExecutionPlan,
        validation_result: PlanValidationResult | None,
        *,
        confirmation_granted: bool = False,
        control: ExecutionControl | None = None,
        on_progress: Callable[[ExecutionProgress], None] | None = None,
        execution_context: ExecutionContext | None = None,
        subplan_depth: int = 0,
        plan_stack: tuple[int, ...] = (),
        operational_config: object | None = None,
    ) -> PlanExecutionResult:
        """Execute a previously validated plan in dependency order."""
        if operational_config is not None:
            from core.execution_strategy import ExecutionStrategyConfiguration

            if not isinstance(
                operational_config,
                ExecutionStrategyConfiguration,
            ):
                raise TypeError(
                    "operational_config must be ExecutionStrategyConfiguration or None."
                )
            if not operational_config.execution_allowed:
                raise ValueError("Blocking operational configuration cannot execute.")
            if len(plan.ordered_steps) > operational_config.max_steps:
                raise ValueError("Plan exceeds the resolved operational step limit.")
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
            execution_context=execution_context,
            context_restored=False,
            subplan_depth=subplan_depth,
            plan_stack=plan_stack,
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
            execution_context=self._context_from_resumable_state(state),
            context_restored=True,
            subplan_depth=0,
            plan_stack=(),
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
        execution_context: ExecutionContext | None,
        context_restored: bool,
        subplan_depth: int,
        plan_stack: tuple[int, ...],
    ) -> PlanExecutionResult:
        """Execute a validated plan from a known in-memory checkpoint."""
        started = time.perf_counter()
        active_context = execution_context or ExecutionContext()
        active_plan_stack = plan_stack + (id(plan),)
        trace = ExecutionTrace(execution_id=active_context.execution_id)
        if context_restored:
            self._trace_execution_context_restored(trace, active_context)
        else:
            self._trace_execution_context_created(trace, active_context)
        self._trace_context_variable_events(trace, active_context)
        self._hydrate_context_from_legacy_state(
            active_context,
            initial_completed_step_ids=initial_completed_step_ids,
            initial_previous_results=initial_previous_results,
        )
        context_error = self._execution_context_plan_error(plan, active_context)
        if context_error is not None:
            trace.add_event(
                component="ExecutionPlanExecutor",
                action="execution_context_validation_failed",
                status=TraceEventStatus.FAILED.value,
                details={
                    "execution_id": active_context.execution_id,
                    "error": context_error,
                },
            )
            return self._finalize_result(
                plan,
                validation_result,
                PlanExecutionResult(
                    plan_status=PlanExecutionStatus.REJECTED.value,
                    success=False,
                    error=context_error,
                    requires_confirmation=(
                        bool(validation_result.requires_confirmation)
                        if validation_result is not None
                        else plan.requires_confirmation
                    ),
                    failed=True,
                    error_code=ExecutionErrorCode.INVALID_PLAN.value,
                    pending_steps=self._pending_step_ids(plan),
                    failure_reason=context_error,
                    metadata={"plan_signature": self._safe_plan_signature(plan)},
                ),
                trace=trace,
            )
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
                trace=trace,
            )

        assert validation_result is not None
        topology = self._topology_or_rejected(
            plan,
            validation_result,
            trace,
        )
        if isinstance(topology, PlanExecutionResult):
            return topology
        execution_steps = topology.ordered_steps(plan)
        total_steps = len(
            [
                step
                for step in execution_steps
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
                pending_steps=self._pending_step_ids(plan, topology=topology),
                current_step=self._first_pending_step_id(plan, topology=topology),
                error_code=ExecutionErrorCode.CONFIRMATION_REQUIRED.value,
                metadata={"plan_signature": self._safe_plan_signature(plan)},
                ),
                trace=trace,
            )

        completed_steps = [
            step.id
            for step in execution_steps
            if step.status == self._COMPLETED_STEP_STATUS
        ]
        for step_id in initial_completed_step_ids:
            if step_id not in completed_steps:
                completed_steps.append(step_id)
        self._hydrate_context_from_legacy_state(
            active_context,
            initial_completed_step_ids=tuple(completed_steps),
            initial_previous_results={},
        )
        completed: set[str] = set(completed_steps)
        skipped_steps = [
            step.id
            for step in execution_steps
            if active_context.state_for_step(step.id) == ExecutionStepState.SKIPPED.value
        ]
        skipped: set[str] = set(skipped_steps)
        blocked_steps = [
            step.id
            for step in execution_steps
            if active_context.state_for_step(step.id) == ExecutionStepState.BLOCKED.value
        ]
        blocked: set[str] = set(blocked_steps)
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

        for index, step in enumerate(execution_steps):
            if step.id in completed or step.id in skipped or step.id in blocked:
                continue

            progress_index = len(completed_steps) + len(skipped_steps) + len(blocked_steps) + 1
            control_result = self._control_result(
                plan=plan,
                execution_steps=execution_steps,
                validation_result=validation_result,
                control=control,
                execution_context=active_context,
                completed_steps=completed_steps,
                skipped_steps=skipped_steps,
                step_results=step_results,
                current_index=index,
                started=started,
                on_progress=on_progress,
                step_index=progress_index,
                total_steps=total_steps,
                trace=trace,
            )
            if control_result is not None:
                return control_result

            dependency_outcome = self._check_step_dependencies(
                plan=plan,
                execution_steps=execution_steps,
                validation_result=validation_result,
                step=step,
                execution_context=active_context,
                trace=trace,
                completed_steps=completed_steps,
                skipped_steps=skipped_steps,
                blocked_steps=blocked_steps,
                step_results=step_results,
                current_index=index,
                started=started,
                on_progress=on_progress,
                step_index=progress_index,
                total_steps=total_steps,
            )
            if dependency_outcome is not None:
                return dependency_outcome

            condition_outcome = self._evaluate_step_condition(
                step=step,
                execution_context=active_context,
                trace=trace,
                started=started,
                on_progress=on_progress,
                step_index=progress_index,
                total_steps=total_steps,
            )
            if condition_outcome is not None:
                step_results.append(condition_outcome)
                if condition_outcome.status == StepExecutionStatus.SKIPPED.value:
                    skipped.add(step.id)
                    skipped_steps.append(step.id)
                    continue

                (
                    trailing_pending,
                    trailing_results,
                ) = self._propagated_not_executed_results(
                    execution_steps=execution_steps,
                    start_index=index + 1,
                    execution_context=active_context,
                    trace=trace,
                    blocked_steps=blocked_steps,
                    skipped_steps=skipped_steps,
                    default_status=StepExecutionStatus.SKIPPED.value,
                )
                self._trace_execution_context_snapshot_created(trace, active_context)
                return self._finalize_result(
                    plan,
                    validation_result,
                    PlanExecutionResult(
                    plan_status=self._failure_status(completed_steps),
                    success=False,
                    completed_steps=completed_steps,
                    failed_step=step.id,
                    failed_steps=[step.id],
                    skipped_steps=skipped_steps,
                    blocked_steps=blocked_steps,
                    pending_steps=trailing_pending,
                    step_results=step_results + trailing_results,
                    error=condition_outcome.error,
                    requires_confirmation=validation_result.requires_confirmation,
                    interrupted=False,
                    failed=True,
                    resumable=False,
                    current_step=step.id,
                    failure_reason=condition_outcome.error,
                    error_code=condition_outcome.error_code,
                    metadata={"plan_signature": validation_result.plan_signature},
                    ),
                    trace=trace,
                )

            step_started = time.perf_counter()
            self._trace_step_event(
                trace,
                action="STEP_STARTED",
                status=TraceEventStatus.STARTED.value,
                step=step,
                step_index=progress_index,
                total_steps=total_steps,
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
                execution_context=active_context,
                trace=trace,
                control=control,
                on_progress=on_progress,
                started=started,
                step_index=progress_index,
                total_steps=total_steps,
                retry_attempts=retry_attempts,
                retry_history=retry_history,
                subplan_depth=subplan_depth,
                plan_stack=active_plan_stack,
            )
            if isinstance(outcome, PlanExecutionResult):
                return outcome
            if outcome.status == StepExecutionStatus.SKIPPED.value:
                step_results.append(outcome)
                self._trace_step_event(
                    trace,
                    action="STEP_SKIPPED",
                    status=TraceEventStatus.FINISHED.value,
                    step=step,
                    step_index=progress_index,
                    total_steps=total_steps,
                    duration_ms=_elapsed_ms(step_started),
                )
                skipped.add(step.id)
                skipped_steps.append(step.id)
                self._emit_progress(
                    on_progress,
                    "step_skipped",
                    started,
                    step=step,
                    step_index=progress_index,
                    total_steps=total_steps,
                )
                continue
            if outcome.error_code in {
                ExecutionErrorCode.EXECUTION_CANCELLED.value,
                ExecutionErrorCode.EXECUTION_INTERRUPTED.value,
                ExecutionErrorCode.SUBPLAN_CANCELLED.value,
                ExecutionErrorCode.LOOP_BODY_CANCELLED.value,
            }:
                self._trace_step_event(
                    trace,
                    action="STEP_FAILED",
                    status=TraceEventStatus.FAILED.value,
                    step=step,
                    step_index=progress_index,
                    total_steps=total_steps,
                    duration_ms=_elapsed_ms(step_started),
                    details={
                        "error_code": outcome.error_code,
                        "error_message": _safe_error_message(
                            outcome.interruption_reason or outcome.error
                        ),
                    },
                )
                return self._retry_stop_result(
                    plan=plan,
                    execution_steps=execution_steps,
                    validation_result=validation_result,
                    completed_steps=completed_steps,
                    step_results=step_results,
                    outcome=outcome,
                    current_index=index,
                    trace=trace,
                )
            step_results.append(outcome)

            if outcome.success:
                self._trace_step_event(
                    trace,
                    action="STEP_FINISHED",
                    status=TraceEventStatus.FINISHED.value,
                    step=step,
                    step_index=progress_index,
                    total_steps=total_steps,
                    duration_ms=_elapsed_ms(step_started),
                )
                completed.add(step.id)
                completed_steps.append(step.id)
                self._emit_progress(
                    on_progress,
                    "step_completed",
                    started,
                    step=step,
                    step_index=progress_index,
                    total_steps=total_steps,
                )
                continue

            self._trace_step_event(
                trace,
                action="STEP_FAILED",
                status=TraceEventStatus.FAILED.value,
                step=step,
                step_index=progress_index,
                total_steps=total_steps,
                duration_ms=_elapsed_ms(step_started),
                details={
                    "error_code": outcome.error_code,
                    "error_message": _safe_error_message(outcome.error),
                },
            )
            self._emit_progress(
                on_progress,
                "step_failed",
                started,
                step=step,
                step_index=progress_index,
                total_steps=total_steps,
            )
            trailing_pending, trailing_results = self._propagated_not_executed_results(
                execution_steps=execution_steps,
                start_index=index + 1,
                execution_context=active_context,
                trace=trace,
                blocked_steps=blocked_steps,
                skipped_steps=skipped_steps,
                default_status=StepExecutionStatus.SKIPPED.value,
            )
            self._trace_execution_context_snapshot_created(trace, active_context)
            return self._finalize_result(
                plan,
                validation_result,
                PlanExecutionResult(
                plan_status=self._failure_status(completed_steps),
                success=False,
                completed_steps=completed_steps,
                failed_step=step.id,
                failed_steps=[step.id],
                skipped_steps=skipped_steps,
                blocked_steps=blocked_steps,
                pending_steps=trailing_pending,
                step_results=step_results + trailing_results,
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
                trace=trace,
            )

        self._emit_progress(
            on_progress,
            "completed",
            started,
            total_steps=total_steps,
        )
        self._trace_execution_context_snapshot_created(trace, active_context)
        return self._finalize_result(
            plan,
            validation_result,
            PlanExecutionResult(
            plan_status=PlanExecutionStatus.COMPLETED.value,
            success=True,
            completed_steps=completed_steps,
            failed_step=None,
            skipped_steps=skipped_steps,
            blocked_steps=blocked_steps,
            step_results=step_results,
            error=None,
            requires_confirmation=validation_result.requires_confirmation,
            interrupted=False,
            completed=True,
            failed=False,
            blocked=False,
            resumable=False,
            pending_steps=[],
            metadata={
                "plan_signature": validation_result.plan_signature,
                "execution_context_snapshot": active_context.snapshot(),
            },
            ),
            trace=trace,
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

        schema_error = self._current_schema_validation_error(plan)
        if schema_error is not None:
            return schema_error

        return None

    def _safe_plan_signature(
        self,
        plan: ExecutionPlan,
    ) -> str | None:
        try:
            return plan_signature(plan)
        except TypeError:
            return None

    def _topology_or_rejected(
        self,
        plan: ExecutionPlan,
        validation_result: PlanValidationResult,
        trace: ExecutionTrace,
    ) -> TopologicalExecutionOrder | PlanExecutionResult:
        self._trace_topology_started(trace, plan)
        try:
            topology = self._topological_sorter.sort(plan)
        except (ExecutionPlanTopologyError, ValueError) as error:
            self._trace_topology_failed(trace, plan, error)
            return self._finalize_result(
                plan,
                validation_result,
                PlanExecutionResult(
                    plan_status=PlanExecutionStatus.REJECTED.value,
                    success=False,
                    error=str(error),
                    requires_confirmation=validation_result.requires_confirmation,
                    failed=True,
                    error_code=ExecutionErrorCode.INVALID_PLAN.value,
                    pending_steps=[],
                    failure_reason=str(error),
                    metadata={"plan_signature": validation_result.plan_signature},
                ),
                trace=trace,
            )
        self._trace_topology_succeeded(trace, topology)
        if topology.reordered:
            self._trace_plan_reordered(trace, topology)
        return topology

    def _safe_topological_steps(
        self,
        plan: ExecutionPlan,
    ) -> tuple[ExecutionStep, ...]:
        try:
            return self._topological_sorter.sort(plan).ordered_steps(plan)
        except Exception:
            return tuple(plan.ordered_steps)

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

        if "schema" in message:
            return ExecutionErrorCode.TOOL_SCHEMA_VALIDATION_FAILED.value

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

        schema_error = self._current_schema_validation_error(state.original_plan)
        if schema_error is not None:
            return schema_error
        registered_signature_error = self._registered_signature_resume_error(state)
        if registered_signature_error is not None:
            return registered_signature_error

        all_step_ids = tuple(step.id for step in state.original_plan.ordered_steps)
        all_step_id_set = set(all_step_ids)
        completed = set(state.completed_step_ids)
        pending = set(state.pending_step_ids)
        failed = set(state.failed_step_ids)
        skipped: set[str] = set()
        blocked: set[str] = set()
        if state.execution_context_snapshot is not None:
            skipped = {
                step_id
                for step_id, step_state in state.execution_context_snapshot.step_states.items()
                if step_state == ExecutionStepState.SKIPPED.value
            }
            blocked = {
                step_id
                for step_id, step_state in state.execution_context_snapshot.step_states.items()
                if step_state == ExecutionStepState.BLOCKED.value
            }

        if not completed.issubset(all_step_id_set):
            return "Resumable execution contains unknown completed steps."

        if not pending.issubset(all_step_id_set):
            return "Resumable execution contains unknown pending steps."

        if not skipped.issubset(all_step_id_set):
            return "Resumable execution contains unknown skipped steps."

        if not blocked.issubset(all_step_id_set):
            return "Resumable execution contains unknown blocked steps."

        if failed:
            return "Failed executions are not resumable."

        if not pending:
            return "Completed executions are not resumable."

        if (
            completed & pending
            or completed & skipped
            or completed & blocked
            or pending & skipped
            or pending & blocked
            or skipped & blocked
        ):
            return "Resumable execution has inconsistent step states."

        expected_pending = tuple(
            step_id
            for step_id in all_step_ids
            if step_id not in completed and step_id not in skipped and step_id not in blocked
        )
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

    def _current_schema_validation_error(
        self,
        plan: ExecutionPlan,
    ) -> str | None:
        validation = ExecutionPlanValidator(
            self._tool_registry,
            plan_registry=self._plan_registry,
        ).validate(plan)
        schema_errors = [
            error
            for error in validation.errors
            if "schema validation failed" in error
        ]
        if not schema_errors:
            return None
        return "Current tool schema is incompatible with the execution plan: " + "; ".join(
            schema_errors
        )

    def _registered_signature_resume_error(
        self,
        state: ResumableExecutionState,
    ) -> str | None:
        if state.execution_context_snapshot is None:
            return None
        signatures = state.execution_context_snapshot.metadata.get(
            "registered_plan_signatures",
        )
        if not isinstance(signatures, Mapping):
            return None
        if self._plan_registry is None and signatures:
            return "Cannot verify registered execution plans without ExecutionPlanRegistry."
        for step_id, payload in signatures.items():
            if not isinstance(step_id, str) or not isinstance(payload, Mapping):
                continue
            plan_id = payload.get("plan_id")
            version = payload.get("version")
            expected_signature = payload.get("resolved_plan_signature")
            if not isinstance(plan_id, str) or not isinstance(expected_signature, str):
                continue
            if version is not None and not isinstance(version, str):
                continue
            assert self._plan_registry is not None
            try:
                current_plan = self._plan_registry.resolve(
                    ExecutionPlanReference(plan_id, version),
                )
                current_signature = plan_signature(current_plan)
            except (ExecutionPlanRegistryError, TypeError):
                return (
                    "Registered execution plan cannot be resolved during resume: "
                    f"{plan_id}."
                )
            if current_signature != expected_signature:
                return (
                    "Registered execution plan signature mismatch during resume: "
                    f"{plan_id}."
                )
        return None

    def _execution_context_plan_error(
        self,
        plan: ExecutionPlan,
        context: ExecutionContext,
    ) -> str | None:
        plan_step_ids = {step.id for step in plan.ordered_steps}
        result_step_ids = set(context.results_snapshot())
        state_step_ids = set(context.step_states)

        if unknown_results := sorted(result_step_ids - plan_step_ids):
            return (
                "Execution context contains results for unknown steps: "
                + ", ".join(unknown_results)
            )

        if unknown_states := sorted(state_step_ids - plan_step_ids):
            return (
                "Execution context contains states for unknown steps: "
                + ", ".join(unknown_states)
            )

        if (
            context.current_step_id is not None
            and context.current_step_id not in plan_step_ids
        ):
            return "Execution context current step is not part of the plan."

        for step_id, state in context.step_states.items():
            if state == ExecutionStepState.SUCCESS.value and not context.has_result(step_id):
                return f"Execution context successful step '{step_id}' has no result."
            if state == ExecutionStepState.SKIPPED.value and context.has_result(step_id):
                return f"Execution context skipped step '{step_id}' cannot have a result."
            if state == ExecutionStepState.BLOCKED.value and context.has_result(step_id):
                return f"Execution context blocked step '{step_id}' cannot have a result."

        step_by_id = {step.id: step for step in plan.ordered_steps}
        for step_id, state in context.step_states.items():
            step = step_by_id.get(step_id)
            if step is None or step.output_binding is None:
                continue
            if state == ExecutionStepState.SKIPPED.value and context.has_variable(step.output_binding.variable_name):
                return (
                    f"Execution context skipped step '{step_id}' cannot have "
                    f"bound variable '{step.output_binding.variable_name}'."
                )
            if state == ExecutionStepState.BLOCKED.value and context.has_variable(step.output_binding.variable_name):
                return (
                    f"Execution context blocked step '{step_id}' cannot have "
                    f"bound variable '{step.output_binding.variable_name}'."
                )
            if state != ExecutionStepState.SUCCESS.value:
                continue
            if not context.has_variable(step.output_binding.variable_name):
                return (
                    f"Execution context completed step '{step_id}' is missing "
                    f"bound variable '{step.output_binding.variable_name}'."
                )

        for step in plan.ordered_steps:
            if context.state_for_step(step.id) != ExecutionStepState.SUCCESS.value:
                continue
            for dependency_id in step.depends_on:
                if context.state_for_step(dependency_id) != ExecutionStepState.SUCCESS.value:
                    return (
                        f"Execution context successful step '{step.id}' has "
                        f"unsatisfied dependency '{dependency_id}'."
                    )

        return None

    def _context_from_resumable_state(
        self,
        state: ResumableExecutionState,
    ) -> ExecutionContext:
        if state.execution_context_snapshot is not None:
            return ExecutionContext.restore(state.execution_context_snapshot)

        context = ExecutionContext()
        self._hydrate_context_from_legacy_state(
            context,
            initial_completed_step_ids=state.completed_step_ids,
            initial_previous_results=state.previous_results,
        )
        return context

    def _hydrate_context_from_legacy_state(
        self,
        context: ExecutionContext,
        *,
        initial_completed_step_ids: tuple[str, ...],
        initial_previous_results: dict[str, object],
    ) -> None:
        for step_id, result in initial_previous_results.items():
            if not context.has_result(step_id):
                context.mark_step_started(step_id, 1)
                context.mark_step_succeeded(step_id, result)

        for step_id in initial_completed_step_ids:
            if context.state_for_step(step_id) == ExecutionStepState.PENDING.value:
                context.mark_step_started(step_id, 1)
                context.mark_step_succeeded(step_id, None)

    def _mark_context_started(
        self,
        trace: ExecutionTrace,
        context: ExecutionContext,
        step_id: str,
        attempt: int,
    ) -> None:
        previous, current = context.mark_step_started(step_id, attempt)
        self._trace_step_state_changed(trace, context, step_id, previous, current, attempt)

    def _mark_context_succeeded(
        self,
        trace: ExecutionTrace,
        context: ExecutionContext,
        step_id: str,
        result: object,
    ) -> None:
        previous, current = context.mark_step_succeeded(step_id, result)
        self._trace_step_state_changed(trace, context, step_id, previous, current, None)

    def _mark_context_failed(
        self,
        trace: ExecutionTrace,
        context: ExecutionContext,
        step_id: str,
    ) -> None:
        if context.state_for_step(step_id) != ExecutionStepState.RUNNING.value:
            return
        previous, current = context.mark_step_failed(step_id)
        self._trace_step_state_changed(trace, context, step_id, previous, current, None)

    def _mark_context_cancelled(
        self,
        trace: ExecutionTrace,
        context: ExecutionContext,
        step_id: str,
    ) -> None:
        if context.state_for_step(step_id) != ExecutionStepState.RUNNING.value:
            return
        previous, current = context.mark_step_cancelled(step_id)
        self._trace_step_state_changed(trace, context, step_id, previous, current, None)

    def _evaluate_step_condition(
        self,
        *,
        step: ExecutionStep,
        execution_context: ExecutionContext,
        trace: ExecutionTrace,
        started: float,
        on_progress: Callable[[ExecutionProgress], None] | None,
        step_index: int,
        total_steps: int,
    ) -> StepExecutionResult | None:
        if step.condition is None:
            return None

        self._trace_condition_started(trace, execution_context, step)
        self._trace_composite_condition_started(trace, execution_context, step)
        try:
            result = self._condition_evaluator.evaluate(step.condition, execution_context)
        except ExecutionConditionEvaluationError as error:
            self._trace_condition_failed(trace, execution_context, step, type(error).__name__)
            self._trace_composite_condition_failed(trace, execution_context, step, type(error).__name__)
            self._mark_context_started(trace, execution_context, step.id, 1)
            self._mark_context_failed(trace, execution_context, step.id)
            return StepExecutionResult(
                step_id=step.id,
                status=StepExecutionStatus.FAILED.value,
                success=False,
                tool_name=step.tool,
                error=str(error),
                error_code=ExecutionErrorCode.EXECUTION_CONDITION_FAILED.value,
                metadata={
                    "condition_operator": self._condition_operator_label(step),
                    "condition_error": type(error).__name__,
                },
            )

        self._trace_condition_succeeded(trace, execution_context, step, result.matched)
        self._trace_composite_condition_succeeded(trace, execution_context, step, result)
        self._trace_condition_short_circuited(trace, execution_context, step, result)
        if result.matched:
            return None

        previous, current = execution_context.mark_step_skipped(step.id)
        self._trace_step_state_changed(trace, execution_context, step.id, previous, current, None)
        self._trace_step_skipped(trace, execution_context, step)
        self._emit_progress(
            on_progress,
            "step_skipped",
            started,
            step=step,
            step_index=step_index,
            total_steps=total_steps,
        )
        return StepExecutionResult(
            step_id=step.id,
            status=StepExecutionStatus.SKIPPED.value,
            success=False,
            tool_name=step.tool,
            output=None,
            error=None,
            error_code=None,
            metadata={
                "condition_operator": self._condition_operator_label(step),
                "condition_matched": False,
            },
        )

    def _check_step_dependencies(
        self,
        *,
        plan: ExecutionPlan,
        execution_steps: tuple[ExecutionStep, ...],
        validation_result: PlanValidationResult,
        step: ExecutionStep,
        execution_context: ExecutionContext,
        trace: ExecutionTrace,
        completed_steps: list[str],
        skipped_steps: list[str],
        blocked_steps: list[str],
        step_results: list[StepExecutionResult],
        current_index: int,
        started: float,
        on_progress: Callable[[ExecutionProgress], None] | None,
        step_index: int,
        total_steps: int,
    ) -> PlanExecutionResult | None:
        self._trace_dependency_check_started(trace, execution_context, step)
        check = self._dependency_checker.check(step, execution_context)
        if check.satisfied:
            self._trace_dependency_check_succeeded(trace, execution_context, step, check)
            return None

        if self._has_inconsistent_dependency_state(check):
            return self._dependency_state_inconsistency_result(
                plan=plan,
                execution_steps=execution_steps,
                validation_result=validation_result,
                step=step,
                execution_context=execution_context,
                trace=trace,
                completed_steps=completed_steps,
                skipped_steps=skipped_steps,
                blocked_steps=blocked_steps,
                step_results=step_results,
                current_index=current_index,
                check=check,
            )

        previous, current = execution_context.mark_step_blocked(step.id)
        self._trace_step_state_changed(
            trace,
            execution_context,
            step.id,
            previous,
            current,
            None,
        )
        self._trace_dependency_check_failed(trace, execution_context, step, check)
        self._trace_step_blocked(trace, execution_context, step, check)
        self._emit_progress(
            on_progress,
            "step_blocked",
            started,
            step=step,
            step_index=step_index,
            total_steps=total_steps,
        )
        blocked_steps.append(step.id)
        remaining, trailing_results = self._propagated_not_executed_results(
            execution_steps=execution_steps,
            start_index=current_index + 1,
            execution_context=execution_context,
            trace=trace,
            blocked_steps=blocked_steps,
            skipped_steps=skipped_steps,
            default_status=StepExecutionStatus.NOT_STARTED.value,
        )
        error = self._dependency_blocked_error(step, check)
        self._trace_execution_context_snapshot_created(trace, execution_context)
        return self._finalize_result(
            plan,
            validation_result,
            PlanExecutionResult(
                plan_status=PlanExecutionStatus.BLOCKED.value,
                success=False,
                completed_steps=completed_steps,
                failed_step=None,
                failed_steps=[],
                skipped_steps=skipped_steps,
                blocked_steps=blocked_steps,
                pending_steps=remaining,
                step_results=step_results
                + [
                    StepExecutionResult(
                        step_id=step.id,
                        status=StepExecutionStatus.BLOCKED.value,
                        success=False,
                        tool_name=step.tool,
                        output=None,
                        error=error,
                        error_code=ExecutionErrorCode.DEPENDENCY_NOT_COMPLETED.value,
                        metadata={
                            "dependency_ids": list(check.dependency_ids),
                            "blocking_dependency_ids": list(check.blocking_dependency_ids),
                            "blocking_states": dict(check.blocking_states),
                            "checked_count": check.checked_count,
                        },
                    )
                ]
                + trailing_results,
                error=error,
                requires_confirmation=validation_result.requires_confirmation,
                interrupted=False,
                completed=False,
                failed=False,
                blocked=True,
                resumable=False,
                current_step=step.id,
                failure_reason=error,
                error_code=ExecutionErrorCode.DEPENDENCY_NOT_COMPLETED.value,
                metadata={"plan_signature": validation_result.plan_signature},
            ),
            trace=trace,
        )

    def _has_inconsistent_dependency_state(
        self,
        check: ExecutionDependencyCheckResult,
    ) -> bool:
        return any(
            state in self._INCONSISTENT_DEPENDENCY_STATES
            for state in check.blocking_states.values()
        )

    def _dependency_state_inconsistency_result(
        self,
        *,
        plan: ExecutionPlan,
        execution_steps: tuple[ExecutionStep, ...],
        validation_result: PlanValidationResult,
        step: ExecutionStep,
        execution_context: ExecutionContext,
        trace: ExecutionTrace,
        completed_steps: list[str],
        skipped_steps: list[str],
        blocked_steps: list[str],
        step_results: list[StepExecutionResult],
        current_index: int,
        check: ExecutionDependencyCheckResult,
    ) -> PlanExecutionResult:
        error = (
            f"{ExecutionDependencyStateInconsistencyError.__name__}: step "
            f"'{step.id}' has unresolved dependency states in topological order: "
            f"{self._dependency_state_details(check)}."
        )
        self._trace_dependency_check_failed(trace, execution_context, step, check)
        self._trace_execution_context_snapshot_created(trace, execution_context)
        return self._finalize_result(
            plan,
            validation_result,
            PlanExecutionResult(
                plan_status=PlanExecutionStatus.REJECTED.value,
                success=False,
                completed_steps=completed_steps,
                failed_step=step.id,
                failed_steps=[step.id],
                skipped_steps=skipped_steps,
                blocked_steps=blocked_steps,
                pending_steps=self._remaining_step_ids(
                    execution_steps,
                    current_index + 1,
                ),
                step_results=step_results
                + [
                    StepExecutionResult(
                        step_id=step.id,
                        status=StepExecutionStatus.FAILED.value,
                        success=False,
                        tool_name=step.tool,
                        output=None,
                        error=error,
                        error_code=(
                            ExecutionErrorCode
                            .DEPENDENCY_STATE_INCONSISTENCY
                            .value
                        ),
                        metadata={
                            "dependency_ids": list(check.dependency_ids),
                            "blocking_dependency_ids": list(
                                check.blocking_dependency_ids
                            ),
                            "blocking_states": dict(check.blocking_states),
                            "checked_count": check.checked_count,
                        },
                    )
                ]
                + self._not_executed_results(
                    execution_steps,
                    current_index + 1,
                    StepExecutionStatus.NOT_STARTED.value,
                ),
                error=error,
                requires_confirmation=validation_result.requires_confirmation,
                interrupted=False,
                completed=False,
                failed=True,
                blocked=False,
                resumable=False,
                current_step=step.id,
                failure_reason=error,
                error_code=ExecutionErrorCode.DEPENDENCY_STATE_INCONSISTENCY.value,
                metadata={"plan_signature": validation_result.plan_signature},
            ),
            trace=trace,
        )

    def _dependency_state_details(
        self,
        check: ExecutionDependencyCheckResult,
    ) -> str:
        return ", ".join(
            f"{dependency_id}:{check.blocking_states.get(dependency_id)}"
            for dependency_id in check.blocking_dependency_ids
        )

    def _propagated_not_executed_results(
        self,
        *,
        execution_steps: tuple[ExecutionStep, ...],
        start_index: int,
        execution_context: ExecutionContext,
        trace: ExecutionTrace,
        blocked_steps: list[str],
        skipped_steps: list[str],
        default_status: str,
    ) -> tuple[list[str], list[StepExecutionResult]]:
        pending_steps: list[str] = []
        results: list[StepExecutionResult] = []

        for step in execution_steps[start_index:]:
            if step.status == self._COMPLETED_STEP_STATUS:
                continue

            blocking_states = self._terminal_dependency_states(step, execution_context)
            if blocking_states:
                check = ExecutionDependencyCheckResult(
                    satisfied=False,
                    dependency_ids=tuple(step.depends_on),
                    blocking_dependency_ids=tuple(blocking_states),
                    blocking_states=blocking_states,
                    checked_count=len(step.depends_on),
                    error_code="EXECUTION_DEPENDENCY_NOT_SATISFIED",
                )
                if execution_context.state_for_step(step.id) == (
                    ExecutionStepState.PENDING.value
                ):
                    previous, current = execution_context.mark_step_blocked(step.id)
                    self._trace_step_state_changed(
                        trace,
                        execution_context,
                        step.id,
                        previous,
                        current,
                        None,
                    )
                self._trace_step_blocked(trace, execution_context, step, check)
                if step.id not in blocked_steps:
                    blocked_steps.append(step.id)
                results.append(
                    StepExecutionResult(
                        step_id=step.id,
                        status=StepExecutionStatus.BLOCKED.value,
                        success=False,
                        tool_name=step.tool,
                        output=None,
                        error=self._dependency_blocked_error(step, check),
                        error_code=ExecutionErrorCode.DEPENDENCY_NOT_COMPLETED.value,
                        metadata={
                            "dependency_ids": list(check.dependency_ids),
                            "blocking_dependency_ids": list(
                                check.blocking_dependency_ids
                            ),
                            "blocking_states": dict(check.blocking_states),
                            "checked_count": check.checked_count,
                            "propagated": True,
                        },
                    )
                )
                continue

            if default_status == StepExecutionStatus.SKIPPED.value:
                if step.id not in skipped_steps:
                    skipped_steps.append(step.id)
                results.append(
                    StepExecutionResult(
                        step_id=step.id,
                        status=StepExecutionStatus.SKIPPED.value,
                        success=False,
                        tool_name=step.tool,
                        output=None,
                        error=None,
                    )
                )
                continue

            pending_steps.append(step.id)
            results.append(
                StepExecutionResult(
                    step_id=step.id,
                    status=default_status,
                    success=False,
                    tool_name=step.tool,
                    output=None,
                    error=None,
                )
            )

        return pending_steps, results

    def _terminal_dependency_states(
        self,
        step: ExecutionStep,
        execution_context: ExecutionContext,
    ) -> dict[str, str]:
        return {
            dependency_id: state
            for dependency_id in step.depends_on
            if (
                state := execution_context.state_for_step(dependency_id)
            )
            in self._TERMINAL_UNSATISFIED_DEPENDENCY_STATES
        }

    def _dependency_blocked_error(
        self,
        step: ExecutionStep,
        check: ExecutionDependencyCheckResult,
    ) -> str:
        details = ", ".join(
            f"{dependency_id}:{check.blocking_states.get(dependency_id)}"
            for dependency_id in check.blocking_dependency_ids
        )
        return (
            f"Step '{step.id}' is blocked by unsatisfied dependencies: "
            f"{details}."
        )

    def _apply_output_binding(
        self,
        trace: ExecutionTrace,
        context: ExecutionContext,
        step: ExecutionStep,
        output: object,
    ) -> str | None:
        binding = step.output_binding
        if binding is None:
            return None

        self._trace_output_binding_started(trace, context, step)
        try:
            selected_value = navigate_structured_path(
                output,
                binding.path,
                owner_label=f"output binding for step '{step.id}'",
            )
            if not binding.overwrite and context.has_variable(binding.variable_name):
                raise ValueError("variable already exists and overwrite is disabled")
            context.set_variable(binding.variable_name, selected_value)
        except StructuredReferencePathError as error:
            message = self._binding_error_message(step, error.message)
            self._trace_output_binding_failed(
                trace,
                context,
                step,
                error_code=error.error_code,
            )
            return message
        except Exception as error:
            message = self._binding_error_message(step, str(error))
            self._trace_output_binding_failed(
                trace,
                context,
                step,
                error_code=(
                    ExecutionErrorCode.EXECUTION_VARIABLE_BINDING_FAILED.value
                ),
            )
            return message

        self._trace_output_binding_succeeded(trace, context, step)
        return None

    def _binding_error_message(
        self,
        step: ExecutionStep,
        reason: str,
    ) -> str:
        binding = step.output_binding
        variable_name = binding.variable_name if binding is not None else "<missing>"
        path = list(binding.path) if binding is not None else []
        return (
            f"step_id={step.id} variable={variable_name} path={path} "
            f"operation=output_binding reason={reason}"
        )

    def _control_result(
        self,
        *,
        plan: ExecutionPlan,
        execution_steps: tuple[ExecutionStep, ...],
        validation_result: PlanValidationResult,
        control: ExecutionControl | None,
        execution_context: ExecutionContext,
        completed_steps: list[str],
        skipped_steps: list[str],
        step_results: list[StepExecutionResult],
        current_index: int,
        started: float,
        on_progress: Callable[[ExecutionProgress], None] | None,
        step_index: int,
        total_steps: int,
        trace: ExecutionTrace,
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
                        execution_steps,
                        current_index,
                    ) + skipped_steps,
                    step_results=step_results
                    + self._not_executed_results(
                        execution_steps,
                        current_index,
                        StepExecutionStatus.SKIPPED.value,
                    ),
                    error=message,
                    requires_confirmation=validation_result.requires_confirmation,
                    failed=True,
                    resumable=False,
                    current_step=execution_steps[current_index].id,
                    failure_reason=message,
                    error_code=ExecutionErrorCode.INTERNAL_EXECUTOR_ERROR.value,
                metadata={"exception_type": type(error).__name__},
                ),
                trace=trace,
            )

        if should_cancel:
            current_step_id = execution_steps[current_index].id
            self._mark_context_started(trace, execution_context, current_step_id, 1)
            self._mark_context_cancelled(trace, execution_context, current_step_id)
            return self._controlled_stop_result(
                plan=plan,
                execution_steps=execution_steps,
                validation_result=validation_result,
                completed_steps=completed_steps,
                skipped_steps=skipped_steps,
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
                trace=trace,
            )

        if should_stop:
            current_step_id = execution_steps[current_index].id
            self._mark_context_started(trace, execution_context, current_step_id, 1)
            self._mark_context_failed(trace, execution_context, current_step_id)
            return self._controlled_stop_result(
                plan=plan,
                execution_steps=execution_steps,
                validation_result=validation_result,
                completed_steps=completed_steps,
                skipped_steps=skipped_steps,
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
                trace=trace,
            )

        return None

    def _controlled_stop_result(
        self,
        *,
        plan: ExecutionPlan,
        execution_steps: tuple[ExecutionStep, ...],
        validation_result: PlanValidationResult,
        completed_steps: list[str],
        skipped_steps: list[str],
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
        trace: ExecutionTrace,
    ) -> PlanExecutionResult:
        current_step = execution_steps[current_index]
        self._emit_progress(
            on_progress,
            "cancelled" if cancelled else "interrupted",
            started,
            step=current_step,
            step_index=step_index,
            total_steps=total_steps,
        )
        pending_steps = self._remaining_step_ids(execution_steps, current_index)
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
                skipped_steps=skipped_steps,
                pending_steps=pending_steps,
                step_results=step_results
                + [current_result]
                + self._not_executed_results(
                    execution_steps,
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
            trace=trace,
        )

    def _execute_step(
        self,
        step: ExecutionStep,
        *,
        plan_signature: str | None,
        execution_context: ExecutionContext,
        trace: ExecutionTrace,
        control: ExecutionControl | None,
        on_progress: Callable[[ExecutionProgress], None] | None,
        started: float,
        step_index: int,
        total_steps: int,
        retry_attempts: dict[str, int],
        retry_history: dict[str, list[dict[str, object]]],
        subplan_depth: int,
        plan_stack: tuple[int, ...],
    ) -> StepExecutionResult | PlanExecutionResult:
        if (
            step.subplan is not None
            or step.subplan_ref is not None
            or step.branch is not None
            or step.loop is not None
        ):
            return self._execute_resolved_step_with_retries(
                step,
                plan_signature=plan_signature,
                execution_context=execution_context,
                trace=trace,
                control=control,
                on_progress=on_progress,
                started=started,
                step_index=step_index,
                total_steps=total_steps,
                retry_attempts=retry_attempts,
                retry_history=retry_history,
                subplan_depth=subplan_depth,
                plan_stack=plan_stack,
            )

        if step.tool in self._LOGICAL_TOOLS:
            self._mark_context_started(trace, execution_context, step.id, 1)
            self._mark_context_succeeded(trace, execution_context, step.id, None)
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

        if not self._tool_registry.exists(step.tool):
            self._mark_context_started(trace, execution_context, step.id, 1)
            self._mark_context_failed(trace, execution_context, step.id)
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
            execution_context=execution_context,
            trace=trace,
            control=control,
            on_progress=on_progress,
            started=started,
            step_index=step_index,
            total_steps=total_steps,
            retry_attempts=retry_attempts,
            retry_history=retry_history,
            subplan_depth=subplan_depth,
            plan_stack=plan_stack,
        )

    def _execute_resolved_step_with_retries(
        self,
        step: ExecutionStep,
        *,
        plan_signature: str | None,
        execution_context: ExecutionContext,
        trace: ExecutionTrace,
        control: ExecutionControl | None,
        on_progress: Callable[[ExecutionProgress], None] | None,
        started: float,
        step_index: int,
        total_steps: int,
        retry_attempts: dict[str, int],
        retry_history: dict[str, list[dict[str, object]]],
        subplan_depth: int,
        plan_stack: tuple[int, ...],
    ) -> StepExecutionResult | PlanExecutionResult:
        effective_retry_policy = step.retry_policy or self._retry_policy
        attempt_number = retry_attempts.get(step.id, 0) + 1
        history = retry_history.setdefault(step.id, [])
        if effective_retry_policy.max_attempts > 1:
            self._trace_retry_started(trace, step, effective_retry_policy)

        while True:
            retry_attempts[step.id] = attempt_number
            if attempt_number > 1:
                self._trace_retry_attempt(trace, step, attempt_number, effective_retry_policy)
            starts_before_resolution = step.branch is None and step.loop is None
            if starts_before_resolution:
                self._mark_context_started(
                    trace,
                    execution_context,
                    step.id,
                    attempt_number,
                )
            self._trace_parameter_resolution_started(trace, step)
            self._trace_variable_resolution_started(trace, step)
            resolution = self._parameter_resolver.resolve(
                _step_arguments_dict(step),
                execution_context,
            )
            if not resolution.success:
                self._trace_parameter_resolution_failed(trace, step, resolution)
                self._trace_variable_resolution_failed(trace, step, resolution)
                if not starts_before_resolution:
                    self._mark_context_started(trace, execution_context, step.id, attempt_number)
                self._mark_context_failed(trace, execution_context, step.id)
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
                        "used_variable_names": resolution.used_variable_names,
                        "used_references": resolution.used_references,
                        "attempt_number": attempt_number,
                    },
                )
            self._trace_parameter_resolution_succeeded(trace, step, resolution)
            self._trace_variable_resolution_succeeded(trace, step, resolution)
            outcome = self._execute_resolved_step_once(
                step,
                plan_signature=plan_signature,
                execution_context=execution_context,
                resolved_arguments=resolution.resolved_arguments.as_dict(),
                trace=trace,
                control=control,
                attempt_number=attempt_number,
                history=history,
                subplan_depth=subplan_depth,
                plan_stack=plan_stack,
            )
            outcome = replace(
                outcome,
                metadata={
                    **dict(outcome.metadata),
                    "parameter_resolution_status": "resolved",
                    "resolved_argument_keys": tuple(
                        sorted(resolution.resolved_arguments.keys())
                    ),
                    "used_step_ids": tuple(resolution.used_step_ids),
                    "used_variable_names": tuple(
                        resolution.used_variable_names
                    ),
                    "used_references": tuple(resolution.used_references),
                },
            )

            if outcome.status == StepExecutionStatus.SKIPPED.value:
                return outcome

            if outcome.success:
                execution_context.set_result(step.id, outcome.output)
                binding_error = self._apply_output_binding(
                    trace,
                    execution_context,
                    step,
                    outcome.output,
                )
                if binding_error is not None:
                    self._mark_context_failed(trace, execution_context, step.id)
                    outcome = StepExecutionResult(
                        step_id=step.id,
                        status=StepExecutionStatus.FAILED.value,
                        success=False,
                        tool_name=step.tool,
                        output=outcome.output,
                        error=binding_error,
                        error_code=(
                            ExecutionErrorCode.EXECUTION_VARIABLE_BINDING_FAILED.value
                        ),
                        started_at=outcome.started_at,
                        finished_at=outcome.finished_at,
                        metadata={
                            **dict(outcome.metadata),
                            "binding_failed": True,
                            "retry_scheduled": False,
                        },
                    )
                else:
                    self._mark_context_succeeded(
                        trace,
                        execution_context,
                        step.id,
                        outcome.output,
                    )
                    metadata = dict(outcome.metadata)
                    metadata["attempt_number"] = attempt_number
                    metadata["max_attempts"] = effective_retry_policy.max_attempts
                    metadata["retry_history"] = list(history)
                    metadata["completed_after_retry"] = attempt_number > 1
                    if attempt_number > 1:
                        self._trace_retry_succeeded(trace, step, attempt_number, effective_retry_policy)
                        self._emit_progress(
                            on_progress,
                            "step_completed_after_retry",
                            started,
                            step=step,
                            step_index=step_index,
                            total_steps=total_steps,
                            attempt_number=attempt_number,
                            max_attempts=effective_retry_policy.max_attempts,
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

            if outcome.status == StepExecutionStatus.CANCELLED.value:
                self._mark_context_cancelled(trace, execution_context, step.id)
                return outcome

            self._mark_context_failed(trace, execution_context, step.id)
            history.append(
                {
                    "attempt_number": attempt_number,
                    "error_code": outcome.error_code,
                    "error": outcome.error,
                }
            )
            decision = self._retry_engine.decide(
                effective_retry_policy,
                attempt_number=attempt_number,
                error_code=outcome.error_code,
                metadata=outcome.metadata,
            )
            if not decision.should_retry:
                metadata = dict(outcome.metadata)
                metadata["attempt_number"] = attempt_number
                metadata["max_attempts"] = decision.max_attempts
                metadata["retry_history"] = list(history)
                metadata["retry_scheduled"] = False
                metadata["retry_reason"] = decision.reason.value
                metadata["retry_exhausted"] = (
                    decision.reason is RetryReason.MAX_RETRIES_REACHED
                )
                if attempt_number > 1 or effective_retry_policy.max_attempts > 1:
                    self._trace_retry_aborted(
                        trace,
                        step,
                        attempt_number,
                        effective_retry_policy,
                        decision.reason,
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
                        retry_reason=decision.reason.value,
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
                retry_reason=decision.reason.value,
            )
            self._trace_retry_failed(
                trace,
                step,
                attempt_number,
                effective_retry_policy,
                decision.reason,
            )
            stop_result = self._retry_control_status(control)
            if stop_result is not None:
                if execution_context.state_for_step(step.id) == ExecutionStepState.FAILED.value:
                    self._mark_context_started(
                        trace,
                        execution_context,
                        step.id,
                        decision.attempt_number,
                    )
                if stop_result == StepExecutionStatus.CANCELLED.value:
                    self._mark_context_cancelled(trace, execution_context, step.id)
                else:
                    self._mark_context_failed(trace, execution_context, step.id)
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
                        "retry_reason": decision.reason.value,
                    },
                )
            attempt_number = decision.attempt_number

    def _execute_resolved_step_once(
        self,
        step: ExecutionStep,
        *,
        plan_signature: str | None,
        execution_context: ExecutionContext,
        resolved_arguments: dict[str, object],
        trace: ExecutionTrace,
        control: ExecutionControl | None,
        attempt_number: int,
        history: list[dict[str, object]],
        subplan_depth: int,
        plan_stack: tuple[int, ...],
    ) -> StepExecutionResult:
        if step.branch is not None:
            return self._execute_branch_step_once(
                step,
                execution_context=execution_context,
                resolved_arguments=resolved_arguments,
                trace=trace,
                control=control,
                attempt_number=attempt_number,
                history=history,
                subplan_depth=subplan_depth,
                plan_stack=plan_stack,
            )

        if step.loop is not None:
            return self._execute_loop_step_once(
                step,
                execution_context=execution_context,
                resolved_arguments=resolved_arguments,
                trace=trace,
                control=control,
                attempt_number=attempt_number,
                history=history,
                subplan_depth=subplan_depth,
                plan_stack=plan_stack,
            )

        if step.subplan is not None or step.subplan_ref is not None:
            return self._execute_subplan_step_once(
                step,
                execution_context=execution_context,
                resolved_arguments=resolved_arguments,
                trace=trace,
                attempt_number=attempt_number,
                history=history,
                subplan_depth=subplan_depth,
                plan_stack=plan_stack,
            )

        assert step.tool is not None

        try:
            self._trace_schema_validation_started(trace, step, resolved_arguments)
            output = self._tool_executor.execute(
                step.tool,
                ToolContext(
                    parameters=deepcopy(resolved_arguments),
                    step_id=step.id,
                    plan_signature=plan_signature,
                    previous_results=execution_context.results_snapshot(),
                    metadata={"executor": "ExecutionPlanExecutor"},
                ),
            )
            self._trace_schema_validation_succeeded(trace, step, resolved_arguments)
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
        except ToolSchemaValidationException as error:
            self._trace_schema_validation_failed(trace, step, error)
            invalid_parameters = tuple(
                item.parameter_name
                for item in error.result.errors
                if item.parameter_name is not None
            )
            return StepExecutionResult(
                step_id=step.id,
                status=StepExecutionStatus.FAILED.value,
                success=False,
                tool_name=step.tool,
                output=None,
                error=str(error),
                error_code=ExecutionErrorCode.TOOL_SCHEMA_VALIDATION_FAILED.value,
                metadata={
                    "attempt_number": attempt_number,
                    "schema_error_count": len(error.result.errors),
                    "schema_invalid_parameters": invalid_parameters,
                    "retry_scheduled": False,
                    "retry_reason": "non_retryable_error",
                },
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

    def _execute_branch_step_once(
        self,
        step: ExecutionStep,
        *,
        execution_context: ExecutionContext,
        resolved_arguments: dict[str, object],
        trace: ExecutionTrace,
        control: ExecutionControl | None,
        attempt_number: int,
        history: list[dict[str, object]],
        subplan_depth: int,
        plan_stack: tuple[int, ...],
    ) -> StepExecutionResult:
        branch = step.branch
        assert branch is not None
        depth = subplan_depth + 1
        self._trace_branch_event(
            trace,
            "execution_branch_evaluation_started",
            TraceEventStatus.STARTED.value,
            execution_id=execution_context.execution_id,
            step_id=step.id,
            selected_branch=None,
            depth=depth,
            attempt_number=attempt_number,
            child_execution_id=None,
            child_status=None,
            child_step_count=None,
        )
        try:
            condition_result = self._condition_evaluator.evaluate(branch.condition, execution_context)
        except ExecutionConditionEvaluationError as error:
            self._mark_context_started(trace, execution_context, step.id, attempt_number)
            self._mark_context_failed(trace, execution_context, step.id)
            self._trace_branch_event(
                trace,
                "execution_branch_failed",
                TraceEventStatus.FAILED.value,
                execution_id=execution_context.execution_id,
                step_id=step.id,
                selected_branch=None,
                depth=depth,
                attempt_number=attempt_number,
                child_execution_id=None,
                child_status=None,
                child_step_count=None,
                error_code=ExecutionErrorCode.EXECUTION_CONDITION_FAILED.value,
            )
            return StepExecutionResult(
                step_id=step.id,
                status=StepExecutionStatus.FAILED.value,
                success=False,
                tool_name=None,
                output=None,
                error=str(error),
                error_code=ExecutionErrorCode.EXECUTION_CONDITION_FAILED.value,
                metadata={
                    "branch": True,
                    "attempt_number": attempt_number,
                    "branch_condition_error": type(error).__name__,
                },
            )

        selected_name = "then" if condition_result.matched else "else"
        selected_plan = branch.then_plan if condition_result.matched else branch.else_plan
        self._trace_branch_event(
            trace,
            (
                "execution_branch_then_selected"
                if condition_result.matched
                else "execution_branch_else_selected"
            ),
            TraceEventStatus.FINISHED.value,
            execution_id=execution_context.execution_id,
            step_id=step.id,
            selected_branch=selected_name,
            depth=depth,
            attempt_number=attempt_number,
            child_execution_id=None,
            child_status=None,
            child_step_count=len(selected_plan.ordered_steps) if selected_plan is not None else 0,
        )
        if selected_plan is None:
            previous, current = execution_context.mark_step_skipped(step.id)
            self._trace_step_state_changed(trace, execution_context, step.id, previous, current, None)
            self._trace_step_skipped(trace, execution_context, step)
            self._trace_branch_event(
                trace,
                "execution_branch_skipped",
                TraceEventStatus.FINISHED.value,
                execution_id=execution_context.execution_id,
                step_id=step.id,
                selected_branch=selected_name,
                depth=depth,
                attempt_number=attempt_number,
                child_execution_id=None,
                child_status=StepExecutionStatus.SKIPPED.value,
                child_step_count=0,
            )
            return StepExecutionResult(
                step_id=step.id,
                status=StepExecutionStatus.SKIPPED.value,
                success=False,
                tool_name=None,
                output=None,
                error=None,
                error_code=None,
                metadata={
                    "branch": True,
                    "selected_branch": selected_name,
                    "attempt_number": attempt_number,
                    "branch_skipped": True,
                },
            )

        self._mark_context_started(trace, execution_context, step.id, attempt_number)
        resolved_plan_signature = self._safe_plan_signature(selected_plan)
        if resolved_plan_signature is None:
            self._mark_context_failed(trace, execution_context, step.id)
            return StepExecutionResult(
                step_id=step.id,
                status=StepExecutionStatus.FAILED.value,
                success=False,
                tool_name=None,
                output=None,
                error="Selected branch plan is not deterministically serializable.",
                error_code=ExecutionErrorCode.SUBPLAN_VALIDATION_FAILED.value,
                metadata={"branch": True, "selected_branch": selected_name, "attempt_number": attempt_number},
            )
        executor = SubplanExecutor(
            validator=ExecutionPlanValidator(
                self._tool_registry,
                self._topological_sorter,
                plan_registry=self._plan_registry,
            ),
            executor_factory=self._child_executor,
            plan_registry=self._plan_registry,
        )
        try:
            result = executor.execute(
                parent_execution_id=execution_context.execution_id,
                parent_step_id=step.id,
                subplan=selected_plan,
                parent_context=execution_context,
                resolved_inputs=resolved_arguments,
                depth=depth,
                plan_stack=plan_stack,
                subplan_ref=None,
                resolved_plan_signature=resolved_plan_signature,
            )
        except SubplanDepthExceededError as error:
            self._mark_context_failed(trace, execution_context, step.id)
            return self._failed_branch_step_result(
                step,
                selected_branch=selected_name,
                error=str(error),
                error_code=ExecutionErrorCode.SUBPLAN_DEPTH_EXCEEDED.value,
                attempt_number=attempt_number,
            )
        except RecursiveSubplanError as error:
            self._mark_context_failed(trace, execution_context, step.id)
            return self._failed_branch_step_result(
                step,
                selected_branch=selected_name,
                error=str(error),
                error_code=ExecutionErrorCode.SUBPLAN_RECURSIVE.value,
                attempt_number=attempt_number,
            )
        except SubplanValidationError as error:
            self._mark_context_failed(trace, execution_context, step.id)
            return self._failed_branch_step_result(
                step,
                selected_branch=selected_name,
                error=str(error),
                error_code=ExecutionErrorCode.SUBPLAN_VALIDATION_FAILED.value,
                attempt_number=attempt_number,
            )
        except SubplanExecutionError as error:
            self._mark_context_failed(trace, execution_context, step.id)
            return self._failed_branch_step_result(
                step,
                selected_branch=selected_name,
                error=str(error),
                error_code=ExecutionErrorCode.SUBPLAN_FAILED.value,
                attempt_number=attempt_number,
            )

        if result.child_result.success:
            self._trace_branch_event(
                trace,
                "execution_branch_succeeded",
                TraceEventStatus.FINISHED.value,
                execution_id=execution_context.execution_id,
                step_id=step.id,
                selected_branch=selected_name,
                depth=result.depth,
                attempt_number=attempt_number,
                child_execution_id=result.child_execution_id,
                child_status=result.status,
                child_step_count=len(selected_plan.ordered_steps),
            )
            return StepExecutionResult(
                step_id=step.id,
                status=StepExecutionStatus.COMPLETED.value,
                success=True,
                tool_name=None,
                output=result.output,
                error=None,
                metadata={
                    "branch": True,
                    "selected_branch": selected_name,
                    "attempt_number": attempt_number,
                    "retry_history": list(history),
                    "child_execution_id": result.child_execution_id,
                    "child_status": result.status,
                    "depth": result.depth,
                    "resolved_plan_signature": result.resolved_plan_signature,
                },
            )

        cancelled = result.child_result.plan_status == PlanExecutionStatus.CANCELLED.value
        self._trace_branch_event(
            trace,
            "execution_branch_cancelled" if cancelled else "execution_branch_failed",
            TraceEventStatus.FAILED.value,
            execution_id=execution_context.execution_id,
            step_id=step.id,
            selected_branch=selected_name,
            depth=result.depth,
            attempt_number=attempt_number,
            child_execution_id=result.child_execution_id,
            child_status=result.status,
            child_step_count=len(selected_plan.ordered_steps),
            error_code=(
                ExecutionErrorCode.SUBPLAN_CANCELLED.value
                if cancelled
                else ExecutionErrorCode.SUBPLAN_FAILED.value
            ),
        )
        return StepExecutionResult(
            step_id=step.id,
            status=(
                StepExecutionStatus.CANCELLED.value
                if cancelled
                else StepExecutionStatus.FAILED.value
            ),
            success=False,
            tool_name=None,
            output=None,
            error=result.child_result.error or result.child_result.failure_reason,
            error_code=(
                ExecutionErrorCode.SUBPLAN_CANCELLED.value
                if cancelled
                else ExecutionErrorCode.SUBPLAN_FAILED.value
            ),
            metadata={
                "branch": True,
                "selected_branch": selected_name,
                "attempt_number": attempt_number,
                "retry_history": list(history),
                "child_execution_id": result.child_execution_id,
                "child_status": result.status,
                "child_error_code": result.child_result.error_code,
                "depth": result.depth,
                "resolved_plan_signature": result.resolved_plan_signature,
            },
        )

    def _failed_branch_step_result(
        self,
        step: ExecutionStep,
        *,
        selected_branch: str,
        error: str,
        error_code: str,
        attempt_number: int,
    ) -> StepExecutionResult:
        return StepExecutionResult(
            step_id=step.id,
            status=StepExecutionStatus.FAILED.value,
            success=False,
            tool_name=None,
            output=None,
            error=error,
            error_code=error_code,
            metadata={
                "branch": True,
                "selected_branch": selected_branch,
                "attempt_number": attempt_number,
                "retry_scheduled": False,
            },
        )

    def _execute_loop_step_once(
        self,
        step: ExecutionStep,
        *,
        execution_context: ExecutionContext,
        resolved_arguments: dict[str, object],
        trace: ExecutionTrace,
        control: ExecutionControl | None,
        attempt_number: int,
        history: list[dict[str, object]],
        subplan_depth: int,
        plan_stack: tuple[int, ...],
    ) -> StepExecutionResult:
        loop = step.loop
        assert loop is not None
        depth = subplan_depth + 1
        iterations_completed = 0
        child_results: list[PlanExecutionResult] = []
        last_output: object | None = None
        parent_started = False
        body_signature = self._safe_plan_signature(loop.body_plan)
        if body_signature is None:
            self._mark_context_started(trace, execution_context, step.id, attempt_number)
            return self._loop_step_result(
                step,
                status=StepExecutionStatus.FAILED.value,
                success=False,
                termination_reason=LoopTerminationReason.BODY_FAILED,
                iterations_completed=0,
                last_output=None,
                child_results=(),
                attempt_number=attempt_number,
                history=history,
                error="Loop body plan is not deterministically serializable.",
                error_code=ExecutionErrorCode.SUBPLAN_VALIDATION_FAILED.value,
            )

        self._trace_loop_event(
            trace,
            "execution_loop_started",
            TraceEventStatus.STARTED.value,
            execution_id=execution_context.execution_id,
            step_id=step.id,
            iteration_index=None,
            iterations_completed=iterations_completed,
            max_iterations=loop.max_iterations,
            termination_reason=None,
            child_execution_id=None,
            child_status=None,
            error_code=None,
        )

        while True:
            try:
                condition_result = self._condition_evaluator.evaluate(loop.condition, execution_context)
            except ExecutionConditionEvaluationError as error:
                if not parent_started:
                    self._mark_context_started(trace, execution_context, step.id, attempt_number)
                self._trace_loop_event(
                    trace,
                    "execution_loop_condition_evaluated",
                    TraceEventStatus.FAILED.value,
                    execution_id=execution_context.execution_id,
                    step_id=step.id,
                    iteration_index=iterations_completed,
                    iterations_completed=iterations_completed,
                    max_iterations=loop.max_iterations,
                    termination_reason=LoopTerminationReason.CONDITION_EVALUATION_FAILED.value,
                    child_execution_id=None,
                    child_status=None,
                    error_code=ExecutionErrorCode.LOOP_CONDITION_FAILED.value,
                )
                self._trace_loop_event(
                    trace,
                    "execution_loop_terminated",
                    TraceEventStatus.FAILED.value,
                    execution_id=execution_context.execution_id,
                    step_id=step.id,
                    iteration_index=iterations_completed,
                    iterations_completed=iterations_completed,
                    max_iterations=loop.max_iterations,
                    termination_reason=LoopTerminationReason.CONDITION_EVALUATION_FAILED.value,
                    child_execution_id=None,
                    child_status=None,
                    error_code=ExecutionErrorCode.LOOP_CONDITION_FAILED.value,
                )
                return self._loop_step_result(
                    step,
                    status=StepExecutionStatus.FAILED.value,
                    success=False,
                    termination_reason=LoopTerminationReason.CONDITION_EVALUATION_FAILED,
                    iterations_completed=iterations_completed,
                    last_output=last_output,
                    child_results=tuple(child_results),
                    attempt_number=attempt_number,
                    history=history,
                    error=str(error),
                    error_code=ExecutionErrorCode.LOOP_CONDITION_FAILED.value,
                )

            self._trace_loop_event(
                trace,
                "execution_loop_condition_evaluated",
                TraceEventStatus.FINISHED.value,
                execution_id=execution_context.execution_id,
                step_id=step.id,
                iteration_index=iterations_completed,
                iterations_completed=iterations_completed,
                max_iterations=loop.max_iterations,
                termination_reason=(
                    None
                    if condition_result.matched
                    else LoopTerminationReason.CONDITION_FALSE.value
                ),
                child_execution_id=None,
                child_status=None,
                error_code=None,
            )
            if not condition_result.matched:
                if not parent_started:
                    self._mark_context_started(trace, execution_context, step.id, attempt_number)
                self._trace_loop_event(
                    trace,
                    "execution_loop_terminated",
                    TraceEventStatus.FINISHED.value,
                    execution_id=execution_context.execution_id,
                    step_id=step.id,
                    iteration_index=iterations_completed,
                    iterations_completed=iterations_completed,
                    max_iterations=loop.max_iterations,
                    termination_reason=LoopTerminationReason.CONDITION_FALSE.value,
                    child_execution_id=None,
                    child_status=StepExecutionStatus.COMPLETED.value,
                    error_code=None,
                )
                return self._loop_step_result(
                    step,
                    status=StepExecutionStatus.COMPLETED.value,
                    success=True,
                    termination_reason=LoopTerminationReason.CONDITION_FALSE,
                    iterations_completed=iterations_completed,
                    last_output=last_output,
                    child_results=tuple(child_results),
                    attempt_number=attempt_number,
                    history=history,
                    error=None,
                    error_code=None,
                )

            if iterations_completed >= loop.max_iterations:
                if not parent_started:
                    self._mark_context_started(trace, execution_context, step.id, attempt_number)
                self._trace_loop_event(
                    trace,
                    "execution_loop_max_iterations_reached",
                    TraceEventStatus.FAILED.value,
                    execution_id=execution_context.execution_id,
                    step_id=step.id,
                    iteration_index=iterations_completed,
                    iterations_completed=iterations_completed,
                    max_iterations=loop.max_iterations,
                    termination_reason=LoopTerminationReason.MAX_ITERATIONS_REACHED.value,
                    child_execution_id=None,
                    child_status=None,
                    error_code=ExecutionErrorCode.LOOP_MAX_ITERATIONS_REACHED.value,
                )
                self._trace_loop_event(
                    trace,
                    "execution_loop_terminated",
                    TraceEventStatus.FAILED.value,
                    execution_id=execution_context.execution_id,
                    step_id=step.id,
                    iteration_index=iterations_completed,
                    iterations_completed=iterations_completed,
                    max_iterations=loop.max_iterations,
                    termination_reason=LoopTerminationReason.MAX_ITERATIONS_REACHED.value,
                    child_execution_id=None,
                    child_status=None,
                    error_code=ExecutionErrorCode.LOOP_MAX_ITERATIONS_REACHED.value,
                )
                return self._loop_step_result(
                    step,
                    status=StepExecutionStatus.FAILED.value,
                    success=False,
                    termination_reason=LoopTerminationReason.MAX_ITERATIONS_REACHED,
                    iterations_completed=iterations_completed,
                    last_output=last_output,
                    child_results=tuple(child_results),
                    attempt_number=attempt_number,
                    history=history,
                    error="Loop reached max_iterations before condition became false.",
                    error_code=ExecutionErrorCode.LOOP_MAX_ITERATIONS_REACHED.value,
                )

            if not parent_started:
                self._mark_context_started(trace, execution_context, step.id, attempt_number)
                parent_started = True
            iteration_number = iterations_completed + 1
            self._trace_loop_event(
                trace,
                "execution_loop_iteration_started",
                TraceEventStatus.STARTED.value,
                execution_id=execution_context.execution_id,
                step_id=step.id,
                iteration_index=iteration_number,
                iterations_completed=iterations_completed,
                max_iterations=loop.max_iterations,
                termination_reason=None,
                child_execution_id=None,
                child_status=None,
                error_code=None,
            )
            child_context = ExecutionContext(
                initial_variables={
                    **execution_context.variables_snapshot(),
                    **deepcopy(resolved_arguments),
                },
                metadata={
                    "parent_execution_id": execution_context.execution_id,
                    "parent_step_id": step.id,
                    "loop_iteration": iteration_number,
                    "depth": depth,
                },
            )
            validation = ExecutionPlanValidator(
                self._tool_registry,
                self._topological_sorter,
                plan_registry=self._plan_registry,
            ).validate(loop.body_plan, depth=depth, plan_stack=plan_stack)
            if not validation.is_valid:
                self._trace_loop_event(
                    trace,
                    "execution_loop_iteration_failed",
                    TraceEventStatus.FAILED.value,
                    execution_id=execution_context.execution_id,
                    step_id=step.id,
                    iteration_index=iteration_number,
                    iterations_completed=iterations_completed,
                    max_iterations=loop.max_iterations,
                    termination_reason=LoopTerminationReason.BODY_FAILED.value,
                    child_execution_id=child_context.execution_id,
                    child_status=None,
                    error_code=ExecutionErrorCode.SUBPLAN_VALIDATION_FAILED.value,
                )
                return self._loop_step_result(
                    step,
                    status=StepExecutionStatus.FAILED.value,
                    success=False,
                    termination_reason=LoopTerminationReason.BODY_FAILED,
                    iterations_completed=iterations_completed,
                    last_output=last_output,
                    child_results=tuple(child_results),
                    attempt_number=attempt_number,
                    history=history,
                    error="Loop body plan did not pass validation.",
                    error_code=ExecutionErrorCode.SUBPLAN_VALIDATION_FAILED.value,
                )
            child_result = self._child_executor().execute(
                loop.body_plan,
                validation,
                confirmation_granted=True,
                control=control,
                execution_context=child_context,
                subplan_depth=depth,
                plan_stack=plan_stack,
            )
            child_results.append(child_result)
            if child_result.success:
                iterations_completed += 1
                last_output = child_result.output
                self._sync_loop_child_variables(execution_context, child_context)
                self._trace_loop_event(
                    trace,
                    "execution_loop_iteration_succeeded",
                    TraceEventStatus.FINISHED.value,
                    execution_id=execution_context.execution_id,
                    step_id=step.id,
                    iteration_index=iteration_number,
                    iterations_completed=iterations_completed,
                    max_iterations=loop.max_iterations,
                    termination_reason=None,
                    child_execution_id=child_context.execution_id,
                    child_status=child_result.plan_status,
                    error_code=None,
                )
                continue

            if child_result.cancelled:
                termination = LoopTerminationReason.BODY_CANCELLED
                status = StepExecutionStatus.CANCELLED.value
                error_code = ExecutionErrorCode.LOOP_BODY_CANCELLED.value
                trace_status = TraceEventStatus.FAILED.value
            elif child_result.blocked:
                termination = LoopTerminationReason.BODY_BLOCKED
                status = StepExecutionStatus.FAILED.value
                error_code = ExecutionErrorCode.LOOP_BODY_BLOCKED.value
                trace_status = TraceEventStatus.FAILED.value
            else:
                termination = LoopTerminationReason.BODY_FAILED
                status = StepExecutionStatus.FAILED.value
                error_code = ExecutionErrorCode.LOOP_BODY_FAILED.value
                trace_status = TraceEventStatus.FAILED.value
            self._trace_loop_event(
                trace,
                "execution_loop_iteration_failed",
                trace_status,
                execution_id=execution_context.execution_id,
                step_id=step.id,
                iteration_index=iteration_number,
                iterations_completed=iterations_completed,
                max_iterations=loop.max_iterations,
                termination_reason=termination.value,
                child_execution_id=child_context.execution_id,
                child_status=child_result.plan_status,
                error_code=error_code,
            )
            self._trace_loop_event(
                trace,
                "execution_loop_terminated",
                trace_status,
                execution_id=execution_context.execution_id,
                step_id=step.id,
                iteration_index=iteration_number,
                iterations_completed=iterations_completed,
                max_iterations=loop.max_iterations,
                termination_reason=termination.value,
                child_execution_id=child_context.execution_id,
                child_status=child_result.plan_status,
                error_code=error_code,
            )
            return self._loop_step_result(
                step,
                status=status,
                success=False,
                termination_reason=termination,
                iterations_completed=iterations_completed,
                last_output=last_output,
                child_results=tuple(child_results),
                attempt_number=attempt_number,
                history=history,
                error=child_result.error or child_result.failure_reason,
                error_code=error_code,
            )

    def _sync_loop_child_variables(
        self,
        parent_context: ExecutionContext,
        child_context: ExecutionContext,
    ) -> None:
        for name, value in child_context.variables_snapshot().items():
            parent_context.set_variable(name, value)

    def _loop_step_result(
        self,
        step: ExecutionStep,
        *,
        status: str,
        success: bool,
        termination_reason: LoopTerminationReason,
        iterations_completed: int,
        last_output: object | None,
        child_results: tuple[PlanExecutionResult, ...],
        attempt_number: int,
        history: list[dict[str, object]],
        error: str | None,
        error_code: str | None,
    ) -> StepExecutionResult:
        loop_result = LoopExecutionResult(
            iterations_completed=iterations_completed,
            termination_reason=termination_reason.value,
            last_output=deepcopy(last_output),
            child_results=child_results,
            status=status,
        )
        del loop_result
        return StepExecutionResult(
            step_id=step.id,
            status=status,
            success=success,
            tool_name=None,
            output=last_output if success else None,
            error=error,
            error_code=error_code,
            metadata={
                "loop": True,
                "iterations_completed": iterations_completed,
                "termination_reason": termination_reason.value,
                "child_result_count": len(child_results),
                "child_execution_ids": tuple(
                    result.trace.execution_id
                    for result in child_results
                    if result.trace is not None
                ),
                "attempt_number": attempt_number,
                "retry_history": list(history),
                "retry_scheduled": False,
            },
        )

    def _execute_subplan_step_once(
        self,
        step: ExecutionStep,
        *,
        execution_context: ExecutionContext,
        resolved_arguments: dict[str, object],
        trace: ExecutionTrace,
        attempt_number: int,
        history: list[dict[str, object]],
        subplan_depth: int,
        plan_stack: tuple[int, ...],
    ) -> StepExecutionResult:
        depth = subplan_depth + 1
        resolution = self._resolve_subplan_for_attempt(
            step,
            execution_context=execution_context,
            trace=trace,
            depth=depth,
            attempt_number=attempt_number,
        )
        if isinstance(resolution, StepExecutionResult):
            return resolution
        subplan, resolved_plan_signature = resolution
        self._trace_subplan_event(
            trace,
            "subplan_execution_started",
            TraceEventStatus.STARTED.value,
            parent_execution_id=execution_context.execution_id,
            parent_step_id=step.id,
            child_execution_id=None,
            depth=depth,
            attempt_number=attempt_number,
            child_status=None,
            child_step_count=len(subplan.ordered_steps),
            plan_reference=step.subplan_ref,
            resolved_plan_signature=resolved_plan_signature,
        )
        executor = SubplanExecutor(
            validator=ExecutionPlanValidator(
                self._tool_registry,
                self._topological_sorter,
                plan_registry=self._plan_registry,
            ),
            executor_factory=self._child_executor,
            plan_registry=self._plan_registry,
        )
        try:
            result = executor.execute(
                parent_execution_id=execution_context.execution_id,
                parent_step_id=step.id,
                subplan=subplan,
                parent_context=execution_context,
                resolved_inputs=resolved_arguments,
                depth=depth,
                plan_stack=plan_stack,
                subplan_ref=step.subplan_ref,
                resolved_plan_signature=resolved_plan_signature,
            )
        except SubplanDepthExceededError as error:
            return self._failed_subplan_step_result(
                step,
                error=str(error),
                error_code=ExecutionErrorCode.SUBPLAN_DEPTH_EXCEEDED.value,
                attempt_number=attempt_number,
            )
        except RecursiveSubplanError as error:
            return self._failed_subplan_step_result(
                step,
                error=str(error),
                error_code=ExecutionErrorCode.SUBPLAN_RECURSIVE.value,
                attempt_number=attempt_number,
            )
        except SubplanValidationError as error:
            return self._failed_subplan_step_result(
                step,
                error=str(error),
                error_code=ExecutionErrorCode.SUBPLAN_VALIDATION_FAILED.value,
                attempt_number=attempt_number,
            )
        except SubplanExecutionError as error:
            return self._failed_subplan_step_result(
                step,
                error=str(error),
                error_code=ExecutionErrorCode.SUBPLAN_FAILED.value,
                attempt_number=attempt_number,
            )

        self._trace_subplan_event(
            trace,
            "child_execution_created",
            TraceEventStatus.FINISHED.value,
            parent_execution_id=result.parent_execution_id,
            parent_step_id=result.parent_step_id,
            child_execution_id=result.child_execution_id,
            depth=result.depth,
            attempt_number=attempt_number,
            child_status=result.status,
            child_step_count=len(subplan.ordered_steps),
            plan_reference=result.plan_reference,
            resolved_plan_signature=result.resolved_plan_signature,
        )

        if result.child_result.success:
            self._trace_subplan_event(
                trace,
                "subplan_execution_succeeded",
                TraceEventStatus.FINISHED.value,
                parent_execution_id=result.parent_execution_id,
                parent_step_id=result.parent_step_id,
                child_execution_id=result.child_execution_id,
                depth=result.depth,
                attempt_number=attempt_number,
                child_status=result.status,
                child_step_count=len(subplan.ordered_steps),
                plan_reference=result.plan_reference,
                resolved_plan_signature=result.resolved_plan_signature,
            )
            self._remember_registered_plan_signature(
                execution_context,
                step,
                result.plan_reference,
                result.resolved_plan_signature,
            )
            return StepExecutionResult(
                step_id=step.id,
                status=StepExecutionStatus.COMPLETED.value,
                success=True,
                tool_name=None,
                output=result.output,
                error=None,
                metadata={
                    "attempt_number": attempt_number,
                    "retry_history": list(history),
                    "subplan": True,
                    "child_execution_id": result.child_execution_id,
                    "child_status": result.status,
                    "depth": result.depth,
                    "plan_id": (
                        result.plan_reference.plan_id
                        if result.plan_reference is not None
                        else None
                    ),
                    "version": (
                        result.plan_reference.version
                        if result.plan_reference is not None
                        else None
                    ),
                    "resolved_plan_signature": result.resolved_plan_signature,
                },
            )

        cancelled = result.child_result.plan_status == PlanExecutionStatus.CANCELLED.value
        action = "subplan_execution_cancelled" if cancelled else "subplan_execution_failed"
        self._trace_subplan_event(
            trace,
            action,
            TraceEventStatus.FAILED.value,
            parent_execution_id=result.parent_execution_id,
            parent_step_id=result.parent_step_id,
            child_execution_id=result.child_execution_id,
            depth=result.depth,
            attempt_number=attempt_number,
            child_status=result.status,
            child_step_count=len(subplan.ordered_steps),
            plan_reference=result.plan_reference,
            resolved_plan_signature=result.resolved_plan_signature,
        )
        return StepExecutionResult(
            step_id=step.id,
            status=(
                StepExecutionStatus.CANCELLED.value
                if cancelled
                else StepExecutionStatus.FAILED.value
            ),
            success=False,
            tool_name=None,
            output=None,
            error=result.child_result.error or result.child_result.failure_reason,
            error_code=(
                ExecutionErrorCode.SUBPLAN_CANCELLED.value
                if cancelled
                else ExecutionErrorCode.SUBPLAN_FAILED.value
            ),
            metadata={
                "attempt_number": attempt_number,
                "retry_history": list(history),
                "subplan": True,
                "child_execution_id": result.child_execution_id,
                "child_status": result.status,
                "child_error_code": result.child_result.error_code,
                "depth": result.depth,
                "plan_id": (
                    result.plan_reference.plan_id
                    if result.plan_reference is not None
                    else None
                ),
                "version": (
                    result.plan_reference.version
                    if result.plan_reference is not None
                    else None
                ),
                "resolved_plan_signature": result.resolved_plan_signature,
            },
        )

    def _failed_subplan_step_result(
        self,
        step: ExecutionStep,
        *,
        error: str,
        error_code: str,
        attempt_number: int,
    ) -> StepExecutionResult:
        return StepExecutionResult(
            step_id=step.id,
            status=StepExecutionStatus.FAILED.value,
            success=False,
            tool_name=None,
            output=None,
            error=error,
            error_code=error_code,
            metadata={
                "attempt_number": attempt_number,
                "subplan": True,
                "retry_scheduled": False,
            },
        )

    def _resolve_subplan_for_attempt(
        self,
        step: ExecutionStep,
        *,
        execution_context: ExecutionContext,
        trace: ExecutionTrace,
        depth: int,
        attempt_number: int,
    ) -> tuple[ExecutionPlan, str] | StepExecutionResult:
        if step.subplan is not None:
            signature = self._safe_plan_signature(step.subplan)
            if signature is None:
                return self._failed_subplan_step_result(
                    step,
                    error="Embedded subplan is not deterministically serializable.",
                    error_code=ExecutionErrorCode.SUBPLAN_VALIDATION_FAILED.value,
                    attempt_number=attempt_number,
                )
            return step.subplan, signature

        reference = step.subplan_ref
        if reference is None:
            return self._failed_subplan_step_result(
                step,
                error="Subplan step has neither subplan nor subplan_ref.",
                error_code=ExecutionErrorCode.SUBPLAN_VALIDATION_FAILED.value,
                attempt_number=attempt_number,
            )
        if self._plan_registry is None:
            self._trace_plan_reference_resolution_failed(
                trace,
                reference,
                parent_execution_id=execution_context.execution_id,
                parent_step_id=step.id,
                depth=depth,
                attempt_number=attempt_number,
                error_code=ExecutionErrorCode.EXECUTION_PLAN_REGISTRY_UNAVAILABLE.value,
            )
            return self._failed_subplan_step_result(
                step,
                error="ExecutionPlanRegistry is required to resolve subplan_ref.",
                error_code=ExecutionErrorCode.EXECUTION_PLAN_REGISTRY_UNAVAILABLE.value,
                attempt_number=attempt_number,
            )

        self._trace_plan_reference_resolution_started(
            trace,
            reference,
            parent_execution_id=execution_context.execution_id,
            parent_step_id=step.id,
            depth=depth,
            attempt_number=attempt_number,
        )
        try:
            subplan = self._plan_registry.resolve(reference)
            resolved_signature = plan_signature(subplan)
        except ExecutionPlanRegistryError as error:
            self._trace_plan_reference_resolution_failed(
                trace,
                reference,
                parent_execution_id=execution_context.execution_id,
                parent_step_id=step.id,
                depth=depth,
                attempt_number=attempt_number,
                error_code=error.code,
            )
            return self._failed_subplan_step_result(
                step,
                error=str(error),
                error_code=ExecutionErrorCode.EXECUTION_PLAN_REFERENCE_NOT_FOUND.value,
                attempt_number=attempt_number,
            )
        except TypeError:
            self._trace_plan_reference_resolution_failed(
                trace,
                reference,
                parent_execution_id=execution_context.execution_id,
                parent_step_id=step.id,
                depth=depth,
                attempt_number=attempt_number,
                error_code=ExecutionErrorCode.SUBPLAN_VALIDATION_FAILED.value,
            )
            return self._failed_subplan_step_result(
                step,
                error="Registered subplan is not deterministically serializable.",
                error_code=ExecutionErrorCode.SUBPLAN_VALIDATION_FAILED.value,
                attempt_number=attempt_number,
            )
        self._trace_plan_reference_resolution_succeeded(
            trace,
            reference,
            parent_execution_id=execution_context.execution_id,
            parent_step_id=step.id,
            depth=depth,
            attempt_number=attempt_number,
            resolved_plan_signature=resolved_signature,
        )
        return subplan, resolved_signature

    def _remember_registered_plan_signature(
        self,
        execution_context: ExecutionContext,
        step: ExecutionStep,
        reference: ExecutionPlanReference | None,
        resolved_plan_signature: str | None,
    ) -> None:
        if reference is None or resolved_plan_signature is None:
            return
        snapshot = execution_context.metadata_snapshot()
        raw_signatures = snapshot.get("registered_plan_signatures")
        signatures = dict(raw_signatures) if isinstance(raw_signatures, Mapping) else {}
        signatures[step.id] = {
            "plan_id": reference.plan_id,
            "version": reference.version,
            "resolved_plan_signature": resolved_plan_signature,
        }
        execution_context.set_metadata("registered_plan_signatures", signatures)

    def _child_executor(self) -> "ExecutionPlanExecutor":
        return ExecutionPlanExecutor(
            self._tool_registry,
            tool_executor=self._tool_executor,
            parameter_resolver=self._parameter_resolver,
            condition_evaluator=self._condition_evaluator,
            dependency_checker=self._dependency_checker,
            topological_sorter=self._topological_sorter,
            retry_policy=RetryPolicy(max_attempts=1),
            execution_history=self._execution_history,
            plan_registry=self._plan_registry,
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
        *,
        topology: TopologicalExecutionOrder | None = None,
    ) -> list[str]:
        steps = (
            topology.ordered_steps(plan)
            if topology is not None
            else self._safe_topological_steps(plan)
        )
        return self._remaining_step_ids(steps, 0)

    def _first_pending_step_id(
        self,
        plan: ExecutionPlan,
        *,
        topology: TopologicalExecutionOrder | None = None,
    ) -> str | None:
        pending = self._pending_step_ids(plan, topology=topology)
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
        execution_steps: tuple[ExecutionStep, ...],
        validation_result: PlanValidationResult,
        completed_steps: list[str],
        step_results: list[StepExecutionResult],
        outcome: StepExecutionResult,
        current_index: int,
        trace: ExecutionTrace,
    ) -> PlanExecutionResult:
        cancelled = (
            outcome.error_code == ExecutionErrorCode.EXECUTION_CANCELLED.value
            or outcome.error_code == ExecutionErrorCode.SUBPLAN_CANCELLED.value
            or outcome.status == StepExecutionStatus.CANCELLED.value
        )
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
            pending_steps=self._remaining_step_ids(execution_steps, current_index),
            step_results=step_results
            + [outcome]
            + self._not_executed_results(
                execution_steps,
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
            trace=trace,
        )

    def _finalize_result(
        self,
        plan: ExecutionPlan,
        validation_result: PlanValidationResult | None,
        result: PlanExecutionResult,
        *,
        objective: str | None = None,
        trace: ExecutionTrace | None = None,
    ) -> PlanExecutionResult:
        active_trace = trace or result.trace or ExecutionTrace()
        result = replace(result, trace=active_trace)
        result = self._resolve_terminal_plan_output(plan, result, active_trace)
        result = replace(
            result,
            goal_verification_result=GoalVerifier().verify(
                plan,
                result,
                trace=active_trace,
            ),
        )
        if "execution_context_snapshot" in result.metadata:
            result = replace(
                result,
                metadata={
                    key: value
                    for key, value in result.metadata.items()
                    if key != "execution_context_snapshot"
                },
            )
        partial_state = build_partial_execution_state(
            objective=objective or plan.goal,
            plan=plan,
            validation_result=validation_result,
            execution=result,
        )
        if active_trace.finished_at is None:
            active_trace.finish(_trace_status_for_result(result))
        metrics = ExecutionMetricsCalculator().calculate(active_trace)
        final_result = replace(
            result,
            partial_state=partial_state,
            trace=active_trace,
            metrics=metrics,
        )
        if self._execution_history is not None:
            self._execution_history.add(final_result)
        return final_result

    def _resolve_terminal_plan_output(
        self,
        plan: ExecutionPlan,
        result: PlanExecutionResult,
        trace: ExecutionTrace,
    ) -> PlanExecutionResult:
        if not result.success or result.plan_status != PlanExecutionStatus.COMPLETED.value:
            return result
        started = time.perf_counter()
        if isinstance(plan.output, ExecutionPlanOutput):
            stats = plan.output.stats()
            self._trace_plan_output_resolution_started(trace, result, stats)
            context = self._context_from_success_result(plan, result)
            try:
                output = plan.output.resolve(context)
            except ExecutionPlanOutputResolutionError as error:
                self._trace_plan_output_resolution_failed(
                    trace,
                    result,
                    stats,
                    error_code=error.error_code,
                    duration_ms=_elapsed_ms(started),
                )
                return replace(
                    result,
                    plan_status=PlanExecutionStatus.FAILED.value,
                    success=False,
                    completed=False,
                    failed=True,
                    output=None,
                    error=str(error),
                    failure_reason=str(error),
                    error_code=(
                        ExecutionErrorCode.EXECUTION_PLAN_OUTPUT_RESOLUTION_FAILED.value
                    ),
                )
            except ExecutionPlanOutputError as error:
                self._trace_plan_output_resolution_failed(
                    trace,
                    result,
                    stats,
                    error_code=error.error_code,
                    duration_ms=_elapsed_ms(started),
                )
                return replace(
                    result,
                    plan_status=PlanExecutionStatus.FAILED.value,
                    success=False,
                    completed=False,
                    failed=True,
                    output=None,
                    error=str(error),
                    failure_reason=str(error),
                    error_code=(
                        ExecutionErrorCode.EXECUTION_PLAN_OUTPUT_RESOLUTION_FAILED.value
                    ),
                )
            self._trace_plan_output_resolution_succeeded(
                trace,
                result,
                stats,
                duration_ms=_elapsed_ms(started),
            )
            return replace(result, output=output)

        return replace(result, output=self._fallback_plan_output(result))

    def _context_from_success_result(
        self,
        plan: ExecutionPlan,
        result: PlanExecutionResult,
    ) -> ExecutionContext:
        execution_id = result.trace.execution_id if result.trace is not None else None
        snapshot = result.metadata.get("execution_context_snapshot")
        if isinstance(snapshot, ExecutionContextSnapshot):
            return ExecutionContext.restore(snapshot)
        context = ExecutionContext(execution_id)
        step_by_id = {step.id: step for step in plan.ordered_steps}
        for step_result in result.step_results:
            if not step_result.success or step_result.status != StepExecutionStatus.COMPLETED.value:
                if step_result.status == StepExecutionStatus.SKIPPED.value:
                    context.mark_step_skipped(step_result.step_id)
                continue
            context.mark_step_started(
                step_result.step_id,
                int(step_result.metadata.get("attempt_number", 1)),
            )
            context.mark_step_succeeded(step_result.step_id, step_result.output)
            step = step_by_id.get(step_result.step_id)
            if step is None or step.output_binding is None:
                continue
            binding = step.output_binding
            context.set_variable(
                binding.variable_name,
                navigate_structured_path(
                    step_result.output,
                    binding.path,
                    owner_label=f"output binding for step '{step.id}'",
                ),
            )
        return context

    def _fallback_plan_output(
        self,
        result: PlanExecutionResult,
    ) -> object | None:
        for step_result in reversed(result.step_results):
            if step_result.success and step_result.status == StepExecutionStatus.COMPLETED.value:
                return deepcopy(step_result.output)
        return None

    def _trace_step_event(
        self,
        trace: ExecutionTrace,
        *,
        action: str,
        status: str,
        step: ExecutionStep,
        step_index: int,
        total_steps: int,
        duration_ms: int | None = None,
        details: dict[str, object] | None = None,
    ) -> None:
        event_details: dict[str, object] = {
            "step_id": step.id,
            "step_index": step_index,
            "total_steps": total_steps,
            "argument_count": len(step.arguments),
            "argument_keys": sorted(step.arguments.keys()),
            "has_arguments": bool(step.arguments),
        }
        if step.tool is not None:
            event_details["tool_name"] = step.tool
        if details:
            event_details.update(
                {
                    key: value
                    for key, value in details.items()
                    if value is not None
                }
            )
        trace.add_event(
            component="ExecutionPlanExecutor",
            action=action,
            status=status,
            duration_ms=duration_ms,
            details=event_details,
        )

    def _trace_subplan_event(
        self,
        trace: ExecutionTrace,
        action: str,
        status: str,
        *,
        parent_execution_id: str,
        parent_step_id: str,
        child_execution_id: str | None,
        depth: int,
        attempt_number: int,
        child_status: str | None,
        child_step_count: int,
        plan_reference: ExecutionPlanReference | None = None,
        resolved_plan_signature: str | None = None,
    ) -> None:
        details: dict[str, object] = {
            "parent_execution_id": parent_execution_id,
            "parent_step_id": parent_step_id,
            "depth": depth,
            "attempt_number": attempt_number,
            "child_step_count": child_step_count,
        }
        if child_execution_id is not None:
            details["child_execution_id"] = child_execution_id
        if child_status is not None:
            details["child_status"] = child_status
        if plan_reference is not None:
            details["plan_id"] = plan_reference.plan_id
            details["version"] = plan_reference.version
        if resolved_plan_signature is not None:
            details["resolved_plan_signature"] = resolved_plan_signature
        trace.add_event(
            component="SubplanExecutor",
            action=action,
            status=status,
            details=details,
        )

    def _trace_branch_event(
        self,
        trace: ExecutionTrace,
        action: str,
        status: str,
        *,
        execution_id: str,
        step_id: str,
        selected_branch: str | None,
        depth: int,
        attempt_number: int,
        child_execution_id: str | None,
        child_status: str | None,
        child_step_count: int | None,
        error_code: str | None = None,
    ) -> None:
        details: dict[str, object] = {
            "execution_id": execution_id,
            "step_id": step_id,
            "depth": depth,
            "attempt_number": attempt_number,
        }
        if selected_branch is not None:
            details["selected_branch"] = selected_branch
        if child_execution_id is not None:
            details["child_execution_id"] = child_execution_id
        if child_status is not None:
            details["child_status"] = child_status
        if child_step_count is not None:
            details["child_step_count"] = child_step_count
        if error_code is not None:
            details["error_code"] = error_code
        trace.add_event(
            component="ExecutionPlanExecutor",
            action=action,
            status=status,
            details=details,
        )

    def _trace_loop_event(
        self,
        trace: ExecutionTrace,
        action: str,
        status: str,
        *,
        execution_id: str,
        step_id: str,
        iteration_index: int | None,
        iterations_completed: int,
        max_iterations: int,
        termination_reason: str | None,
        child_execution_id: str | None,
        child_status: str | None,
        error_code: str | None,
    ) -> None:
        details: dict[str, object] = {
            "execution_id": execution_id,
            "step_id": step_id,
            "iterations_completed": iterations_completed,
            "max_iterations": max_iterations,
        }
        if iteration_index is not None:
            details["iteration_index"] = iteration_index
        if termination_reason is not None:
            details["termination_reason"] = termination_reason
        if child_execution_id is not None:
            details["child_execution_id"] = child_execution_id
        if child_status is not None:
            details["child_status"] = child_status
        if error_code is not None:
            details["error_code"] = error_code
        trace.add_event(
            component="ExecutionPlanExecutor",
            action=action,
            status=status,
            details=details,
        )

    def _trace_plan_reference_resolution_started(
        self,
        trace: ExecutionTrace,
        reference: ExecutionPlanReference,
        *,
        parent_execution_id: str,
        parent_step_id: str,
        depth: int,
        attempt_number: int,
    ) -> None:
        self._trace_plan_reference_resolution_event(
            trace,
            "execution_plan_reference_resolution_started",
            TraceEventStatus.STARTED.value,
            reference,
            parent_execution_id=parent_execution_id,
            parent_step_id=parent_step_id,
            depth=depth,
            attempt_number=attempt_number,
        )

    def _trace_plan_reference_resolution_succeeded(
        self,
        trace: ExecutionTrace,
        reference: ExecutionPlanReference,
        *,
        parent_execution_id: str,
        parent_step_id: str,
        depth: int,
        attempt_number: int,
        resolved_plan_signature: str,
    ) -> None:
        self._trace_plan_reference_resolution_event(
            trace,
            "execution_plan_reference_resolution_succeeded",
            TraceEventStatus.FINISHED.value,
            reference,
            parent_execution_id=parent_execution_id,
            parent_step_id=parent_step_id,
            depth=depth,
            attempt_number=attempt_number,
            resolved_plan_signature=resolved_plan_signature,
        )

    def _trace_plan_reference_resolution_failed(
        self,
        trace: ExecutionTrace,
        reference: ExecutionPlanReference,
        *,
        parent_execution_id: str,
        parent_step_id: str,
        depth: int,
        attempt_number: int,
        error_code: str,
    ) -> None:
        self._trace_plan_reference_resolution_event(
            trace,
            "execution_plan_reference_resolution_failed",
            TraceEventStatus.FAILED.value,
            reference,
            parent_execution_id=parent_execution_id,
            parent_step_id=parent_step_id,
            depth=depth,
            attempt_number=attempt_number,
            error_code=error_code,
        )

    def _trace_plan_reference_resolution_event(
        self,
        trace: ExecutionTrace,
        action: str,
        status: str,
        reference: ExecutionPlanReference,
        *,
        parent_execution_id: str,
        parent_step_id: str,
        depth: int,
        attempt_number: int,
        resolved_plan_signature: str | None = None,
        error_code: str | None = None,
    ) -> None:
        details: dict[str, object] = {
            "parent_execution_id": parent_execution_id,
            "parent_step_id": parent_step_id,
            "plan_id": reference.plan_id,
            "version": reference.version,
            "depth": depth,
            "attempt_number": attempt_number,
        }
        if resolved_plan_signature is not None:
            details["resolved_plan_signature"] = resolved_plan_signature
        if error_code is not None:
            details["error_code"] = error_code
        trace.add_event(
            component="ExecutionPlanExecutor",
            action=action,
            status=status,
            details=details,
        )

    def _trace_plan_output_resolution_started(
        self,
        trace: ExecutionTrace,
        result: PlanExecutionResult,
        stats: Any,
    ) -> None:
        trace.add_event(
            component="ExecutionPlanExecutor",
            action="execution_plan_output_resolution_started",
            status=TraceEventStatus.STARTED.value,
            details=self._plan_output_trace_details(result, stats),
        )

    def _trace_plan_output_resolution_succeeded(
        self,
        trace: ExecutionTrace,
        result: PlanExecutionResult,
        stats: Any,
        *,
        duration_ms: int,
    ) -> None:
        trace.add_event(
            component="ExecutionPlanExecutor",
            action="execution_plan_output_resolution_succeeded",
            status=TraceEventStatus.FINISHED.value,
            duration_ms=duration_ms,
            details=self._plan_output_trace_details(result, stats),
        )

    def _trace_plan_output_resolution_failed(
        self,
        trace: ExecutionTrace,
        result: PlanExecutionResult,
        stats: Any,
        *,
        error_code: str,
        duration_ms: int,
    ) -> None:
        details = self._plan_output_trace_details(result, stats)
        details["error_code"] = error_code
        trace.add_event(
            component="ExecutionPlanExecutor",
            action="execution_plan_output_resolution_failed",
            status=TraceEventStatus.FAILED.value,
            duration_ms=duration_ms,
            details=details,
        )

    def _plan_output_trace_details(
        self,
        result: PlanExecutionResult,
        stats: Any,
    ) -> dict[str, object]:
        execution_id = result.trace.execution_id if result.trace is not None else None
        details: dict[str, object] = {
            "output_kind": stats.output_kind,
            "node_count": stats.node_count,
            "reference_count": stats.reference_count,
            "step_reference_count": stats.step_reference_count,
            "variable_reference_count": stats.variable_reference_count,
        }
        if execution_id is not None:
            details["execution_id"] = execution_id
        return details

    def _trace_condition_started(
        self,
        trace: ExecutionTrace,
        context: ExecutionContext,
        step: ExecutionStep,
    ) -> None:
        assert step.condition is not None
        trace.add_event(
            component="ExecutionPlanExecutor",
            action="execution_condition_started",
            status=TraceEventStatus.STARTED.value,
            details={
                "execution_id": context.execution_id,
                "step_id": step.id,
                "operator": self._condition_operator_label(step),
                "references": self._condition_reference_labels(step),
            },
        )

    def _trace_composite_condition_started(
        self,
        trace: ExecutionTrace,
        context: ExecutionContext,
        step: ExecutionStep,
    ) -> None:
        if not isinstance(step.condition, (AllOfCondition, AnyOfCondition, NotCondition)):
            return
        stats = condition_tree_stats(step.condition)
        trace.add_event(
            component="ExecutionPlanExecutor",
            action="composite_condition_started",
            status=TraceEventStatus.STARTED.value,
            details={
                "execution_id": context.execution_id,
                "step_id": step.id,
                "condition_kind": condition_kind(step.condition),
                "max_depth": stats["max_depth"],
                "node_count": stats["node_count"],
            },
        )

    def _trace_condition_succeeded(
        self,
        trace: ExecutionTrace,
        context: ExecutionContext,
        step: ExecutionStep,
        matched: bool,
    ) -> None:
        assert step.condition is not None
        trace.add_event(
            component="ExecutionPlanExecutor",
            action="execution_condition_succeeded",
            status=TraceEventStatus.FINISHED.value,
            details={
                "execution_id": context.execution_id,
                "step_id": step.id,
                "operator": self._condition_operator_label(step),
                "matched": matched,
                "references": self._condition_reference_labels(step),
            },
        )

    def _trace_composite_condition_succeeded(
        self,
        trace: ExecutionTrace,
        context: ExecutionContext,
        step: ExecutionStep,
        result: ExecutionConditionResult,
    ) -> None:
        if not isinstance(step.condition, (AllOfCondition, AnyOfCondition, NotCondition)):
            return
        trace.add_event(
            component="ExecutionPlanExecutor",
            action="composite_condition_succeeded",
            status=TraceEventStatus.FINISHED.value,
            details={
                "execution_id": context.execution_id,
                "step_id": step.id,
                "condition_kind": result.condition_kind,
                "matched": result.matched,
                "evaluated_nodes": result.evaluated_nodes,
                "skipped_nodes_due_to_short_circuit": result.skipped_nodes_due_to_short_circuit,
            },
        )

    def _trace_condition_failed(
        self,
        trace: ExecutionTrace,
        context: ExecutionContext,
        step: ExecutionStep,
        error_type: str,
    ) -> None:
        assert step.condition is not None
        trace.add_event(
            component="ExecutionPlanExecutor",
            action="execution_condition_failed",
            status=TraceEventStatus.FAILED.value,
            details={
                "execution_id": context.execution_id,
                "step_id": step.id,
                "operator": self._condition_operator_label(step),
                "error_code": ExecutionErrorCode.EXECUTION_CONDITION_FAILED.value,
                "error_type": error_type,
                "references": self._condition_reference_labels(step),
            },
        )

    def _trace_composite_condition_failed(
        self,
        trace: ExecutionTrace,
        context: ExecutionContext,
        step: ExecutionStep,
        error_type: str,
    ) -> None:
        if not isinstance(step.condition, (AllOfCondition, AnyOfCondition, NotCondition)):
            return
        stats = condition_tree_stats(step.condition)
        trace.add_event(
            component="ExecutionPlanExecutor",
            action="composite_condition_failed",
            status=TraceEventStatus.FAILED.value,
            details={
                "execution_id": context.execution_id,
                "step_id": step.id,
                "condition_kind": condition_kind(step.condition),
                "max_depth": stats["max_depth"],
                "node_count": stats["node_count"],
                "error_code": ExecutionErrorCode.EXECUTION_CONDITION_FAILED.value,
                "error_type": error_type,
            },
        )

    def _trace_condition_short_circuited(
        self,
        trace: ExecutionTrace,
        context: ExecutionContext,
        step: ExecutionStep,
        result: ExecutionConditionResult,
    ) -> None:
        if result.skipped_nodes_due_to_short_circuit <= 0:
            return
        trace.add_event(
            component="ExecutionPlanExecutor",
            action="condition_short_circuited",
            status=TraceEventStatus.FINISHED.value,
            details={
                "execution_id": context.execution_id,
                "step_id": step.id,
                "condition_kind": result.condition_kind,
                "evaluated_nodes": result.evaluated_nodes,
                "skipped_nodes_due_to_short_circuit": result.skipped_nodes_due_to_short_circuit,
                "matched": result.matched,
            },
        )

    def _trace_step_skipped(
        self,
        trace: ExecutionTrace,
        context: ExecutionContext,
        step: ExecutionStep,
    ) -> None:
        trace.add_event(
            component="ExecutionPlanExecutor",
            action="execution_step_skipped",
            status=TraceEventStatus.FINISHED.value,
            details={
                "execution_id": context.execution_id,
                "step_id": step.id,
                "tool_name": step.tool,
            },
        )

    def _trace_topology_started(
        self,
        trace: ExecutionTrace,
        plan: ExecutionPlan,
    ) -> None:
        trace.add_event(
            component="ExecutionPlanTopologicalSorter",
            action="execution_topology_started",
            status=TraceEventStatus.STARTED.value,
            details={
                "step_count": len(plan.ordered_steps),
                "dependency_count": sum(
                    len(step.depends_on)
                    for step in plan.ordered_steps
                ),
            },
        )

    def _trace_topology_succeeded(
        self,
        trace: ExecutionTrace,
        topology: TopologicalExecutionOrder,
    ) -> None:
        trace.add_event(
            component="ExecutionPlanTopologicalSorter",
            action="execution_topology_succeeded",
            status=TraceEventStatus.FINISHED.value,
            details={
                "step_count": len(topology.ordered_step_ids),
                "dependency_count": topology.dependency_count,
                "root_count": len(topology.root_step_ids),
                "reordered": topology.reordered,
                "ordered_step_ids": list(topology.ordered_step_ids),
            },
        )

    def _trace_topology_failed(
        self,
        trace: ExecutionTrace,
        plan: ExecutionPlan,
        error: ExecutionPlanTopologyError,
    ) -> None:
        trace.add_event(
            component="ExecutionPlanTopologicalSorter",
            action="execution_topology_failed",
            status=TraceEventStatus.FAILED.value,
            details={
                "step_count": len(plan.ordered_steps),
                "dependency_count": sum(
                    len(step.depends_on)
                    for step in plan.ordered_steps
                ),
                "error_type": type(error).__name__,
                "error_code": ExecutionErrorCode.INVALID_PLAN.value,
                "cycle_node_ids": [
                    step.id
                    for step in plan.ordered_steps
                    if step.id in str(error)
                ],
            },
        )

    def _trace_plan_reordered(
        self,
        trace: ExecutionTrace,
        topology: TopologicalExecutionOrder,
    ) -> None:
        trace.add_event(
            component="ExecutionPlanTopologicalSorter",
            action="execution_plan_reordered",
            status=TraceEventStatus.FINISHED.value,
            details={
                "step_count": len(topology.ordered_step_ids),
                "dependency_count": topology.dependency_count,
                "reordered": True,
                "ordered_step_ids": list(topology.ordered_step_ids),
            },
        )

    def _trace_dependency_check_started(
        self,
        trace: ExecutionTrace,
        context: ExecutionContext,
        step: ExecutionStep,
    ) -> None:
        trace.add_event(
            component="ExecutionDependencyChecker",
            action="execution_dependency_check_started",
            status=TraceEventStatus.STARTED.value,
            details={
                "execution_id": context.execution_id,
                "step_id": step.id,
                "dependency_count": len(step.depends_on),
                "dependency_ids": list(step.depends_on),
            },
        )

    def _trace_dependency_check_succeeded(
        self,
        trace: ExecutionTrace,
        context: ExecutionContext,
        step: ExecutionStep,
        check: ExecutionDependencyCheckResult,
    ) -> None:
        trace.add_event(
            component="ExecutionDependencyChecker",
            action="execution_dependency_check_succeeded",
            status=TraceEventStatus.FINISHED.value,
            details={
                "execution_id": context.execution_id,
                "step_id": step.id,
                "dependency_count": check.checked_count,
                "dependency_ids": list(check.dependency_ids),
            },
        )

    def _trace_dependency_check_failed(
        self,
        trace: ExecutionTrace,
        context: ExecutionContext,
        step: ExecutionStep,
        check: ExecutionDependencyCheckResult,
    ) -> None:
        trace.add_event(
            component="ExecutionDependencyChecker",
            action="execution_dependency_check_failed",
            status=TraceEventStatus.FAILED.value,
            details={
                "execution_id": context.execution_id,
                "step_id": step.id,
                "dependency_count": check.checked_count,
                "dependency_ids": list(check.dependency_ids),
                "blocking_dependency_ids": list(check.blocking_dependency_ids),
                "blocking_states": dict(check.blocking_states),
                "error_code": check.error_code,
            },
        )

    def _trace_step_blocked(
        self,
        trace: ExecutionTrace,
        context: ExecutionContext,
        step: ExecutionStep,
        check: ExecutionDependencyCheckResult,
    ) -> None:
        trace.add_event(
            component="ExecutionPlanExecutor",
            action="execution_step_blocked",
            status=TraceEventStatus.FINISHED.value,
            details={
                "execution_id": context.execution_id,
                "step_id": step.id,
                "tool_name": step.tool,
                "dependency_count": check.checked_count,
                "dependency_ids": list(check.dependency_ids),
                "blocking_dependency_ids": list(check.blocking_dependency_ids),
                "blocking_states": dict(check.blocking_states),
                "error_code": check.error_code,
            },
        )

    def _condition_reference_labels(
        self,
        step: ExecutionStep,
    ) -> list[str]:
        if step.condition is None:
            return []

        references: list[str] = []

        def visit(value: object) -> None:
            type_name = type(value).__name__
            if type_name == "StepOutputReference":
                step_id = object.__getattribute__(value, "step_id")
                references.append(f"steps.{step_id}.output")
                return
            if type_name == "ExecutionVariableReference":
                name = object.__getattribute__(value, "name")
                references.append(f"variables.{name}")
                return
            if isinstance(value, Mapping):
                for item in value.values():
                    visit(item)
                return
            if isinstance(value, (list, tuple)):
                for item in value:
                    visit(item)

        for operand in iter_condition_operands(step.condition):
            visit(operand)
        return sorted(set(references))

    def _condition_operator_label(
        self,
        step: ExecutionStep,
    ) -> str:
        if step.condition is None:
            return "none"
        if hasattr(step.condition, "operator"):
            return object.__getattribute__(step.condition, "operator").value
        return condition_kind(step.condition)

    def _trace_execution_context_created(
        self,
        trace: ExecutionTrace,
        context: ExecutionContext,
    ) -> None:
        trace.add_event(
            component="ExecutionPlanExecutor",
            action="execution_context_created",
            status=TraceEventStatus.FINISHED.value,
            details={
                "execution_id": context.execution_id,
                "result_count": len(context.results_snapshot()),
            },
        )

    def _trace_execution_context_restored(
        self,
        trace: ExecutionTrace,
        context: ExecutionContext,
    ) -> None:
        trace.add_event(
            component="ExecutionPlanExecutor",
            action="execution_context_restored",
            status=TraceEventStatus.FINISHED.value,
            details={
                "execution_id": context.execution_id,
                "result_count": len(context.results_snapshot()),
                "completed_count": len(context.completed_step_ids),
            },
        )

    def _trace_context_variable_events(
        self,
        trace: ExecutionTrace,
        context: ExecutionContext,
    ) -> None:
        for event in context.variable_events_snapshot():
            action = event.get("action")
            if action not in {"execution_variable_set", "execution_variable_deleted"}:
                continue
            trace.add_event(
                component="ExecutionPlanExecutor",
                action=str(action),
                status=TraceEventStatus.FINISHED.value,
                details={
                    "execution_id": context.execution_id,
                    "variable_name": event.get("variable_name"),
                    "variable_count": event.get("variable_count"),
                },
            )

    def _trace_step_state_changed(
        self,
        trace: ExecutionTrace,
        context: ExecutionContext,
        step_id: str,
        previous: str,
        current: str,
        attempt: int | None,
    ) -> None:
        details: dict[str, object] = {
            "execution_id": context.execution_id,
            "step_id": step_id,
            "previous_state": previous,
            "new_state": current,
            "result_count": len(context.results_snapshot()),
        }
        if attempt is not None:
            details["attempt"] = attempt
        trace.add_event(
            component="ExecutionPlanExecutor",
            action="step_state_changed",
            status=TraceEventStatus.FINISHED.value,
            details=details,
        )

    def _trace_execution_context_snapshot_created(
        self,
        trace: ExecutionTrace,
        context: ExecutionContext,
    ) -> None:
        trace.add_event(
            component="ExecutionPlanExecutor",
            action="execution_context_snapshot_created",
            status=TraceEventStatus.FINISHED.value,
            details={
                "execution_id": context.execution_id,
                "result_count": len(context.results_snapshot()),
                "completed_count": len(context.completed_step_ids),
            },
        )

    def _trace_retry_started(
        self,
        trace: ExecutionTrace,
        step: ExecutionStep,
        policy: RetryPolicy,
    ) -> None:
        trace.add_event(
            component="RetryEngine",
            action="execution_retry_started",
            status=TraceEventStatus.STARTED.value,
            details={
                "step_id": step.id,
                "tool_name": step.tool,
                "max_attempts": policy.max_attempts,
                "strategy": policy.strategy.value,
            },
        )

    def _trace_retry_attempt(
        self,
        trace: ExecutionTrace,
        step: ExecutionStep,
        attempt_number: int,
        policy: RetryPolicy,
    ) -> None:
        trace.add_event(
            component="RetryEngine",
            action="execution_retry_attempt",
            status=TraceEventStatus.STARTED.value,
            details={
                "step_id": step.id,
                "tool_name": step.tool,
                "attempt_number": attempt_number,
                "max_attempts": policy.max_attempts,
            },
        )

    def _trace_retry_succeeded(
        self,
        trace: ExecutionTrace,
        step: ExecutionStep,
        attempt_number: int,
        policy: RetryPolicy,
    ) -> None:
        trace.add_event(
            component="RetryEngine",
            action="execution_retry_succeeded",
            status=TraceEventStatus.FINISHED.value,
            details={
                "step_id": step.id,
                "tool_name": step.tool,
                "attempt_number": attempt_number,
                "max_attempts": policy.max_attempts,
            },
        )

    def _trace_retry_failed(
        self,
        trace: ExecutionTrace,
        step: ExecutionStep,
        attempt_number: int,
        policy: RetryPolicy,
        reason: RetryReason,
    ) -> None:
        trace.add_event(
            component="RetryEngine",
            action="execution_retry_failed",
            status=TraceEventStatus.FAILED.value,
            details={
                "step_id": step.id,
                "tool_name": step.tool,
                "attempt_number": attempt_number,
                "max_attempts": policy.max_attempts,
                "retry_reason": reason.value,
            },
        )

    def _trace_retry_aborted(
        self,
        trace: ExecutionTrace,
        step: ExecutionStep,
        attempt_number: int,
        policy: RetryPolicy,
        reason: RetryReason,
    ) -> None:
        trace.add_event(
            component="RetryEngine",
            action="execution_retry_aborted",
            status=TraceEventStatus.FINISHED.value,
            details={
                "step_id": step.id,
                "tool_name": step.tool,
                "attempt_number": attempt_number,
                "max_attempts": policy.max_attempts,
                "retry_reason": reason.value,
            },
        )

    def _trace_schema_validation_started(
        self,
        trace: ExecutionTrace,
        step: ExecutionStep,
        arguments: dict[str, object],
    ) -> None:
        if step.tool is None or self._tool_registry.arguments_schema(step.tool) is None:
            return
        trace.add_event(
            component="ExecutionPlanExecutor",
            action="schema_validation_started",
            status=TraceEventStatus.STARTED.value,
            details=self._schema_validation_trace_details(step, arguments),
        )

    def _trace_schema_validation_succeeded(
        self,
        trace: ExecutionTrace,
        step: ExecutionStep,
        arguments: dict[str, object],
    ) -> None:
        if step.tool is None or self._tool_registry.arguments_schema(step.tool) is None:
            return
        trace.add_event(
            component="ExecutionPlanExecutor",
            action="schema_validation_succeeded",
            status=TraceEventStatus.FINISHED.value,
            details=self._schema_validation_trace_details(step, arguments),
        )

    def _trace_schema_validation_failed(
        self,
        trace: ExecutionTrace,
        step: ExecutionStep,
        error: ToolSchemaValidationException,
    ) -> None:
        if step.tool is None or self._tool_registry.arguments_schema(step.tool) is None:
            return
        invalid_parameters = sorted(
            {
                item.parameter_name
                for item in error.result.errors
                if item.parameter_name is not None
            }
        )
        details = {
            "step_id": step.id,
            "tool_name": step.tool,
            "error_count": len(error.result.errors),
            "invalid_parameters": invalid_parameters,
        }
        trace.add_event(
            component="ExecutionPlanExecutor",
            action="schema_validation_failed",
            status=TraceEventStatus.FAILED.value,
            details=details,
        )

    def _schema_validation_trace_details(
        self,
        step: ExecutionStep,
        arguments: dict[str, object],
    ) -> dict[str, object]:
        return {
            "step_id": step.id,
            "tool_name": step.tool,
            "argument_count": len(arguments),
            "argument_keys": sorted(arguments.keys()),
        }

    def _trace_parameter_resolution_started(
        self,
        trace: ExecutionTrace,
        step: ExecutionStep,
    ) -> None:
        if not self._has_resolvable_reference(step.arguments):
            return
        trace.add_event(
            component="ExecutionPlanExecutor",
            action="parameter_resolution_started",
            status=TraceEventStatus.STARTED.value,
            details=self._parameter_resolution_trace_details(step),
        )

    def _trace_parameter_resolution_succeeded(
        self,
        trace: ExecutionTrace,
        step: ExecutionStep,
        resolution: Any,
    ) -> None:
        if not self._has_resolvable_reference(step.arguments):
            return
        details = self._parameter_resolution_trace_details(step)
        details.update(
            {
                "referenced_step_ids": sorted(set(resolution.used_step_ids)),
                "reference_count": len(resolution.used_references),
            }
        )
        trace.add_event(
            component="ExecutionPlanExecutor",
            action="parameter_resolution_succeeded",
            status=TraceEventStatus.FINISHED.value,
            details=details,
        )

    def _trace_parameter_resolution_failed(
        self,
        trace: ExecutionTrace,
        step: ExecutionStep,
        resolution: Any,
    ) -> None:
        if not self._has_resolvable_reference(step.arguments):
            return
        details = self._parameter_resolution_trace_details(step)
        details.update(
            {
                "referenced_step_ids": sorted(set(resolution.used_step_ids)),
                "reference_count": len(resolution.used_references),
                "error_code": resolution.error_code,
                "unresolved_references": list(resolution.unresolved_references),
            }
        )
        trace.add_event(
            component="ExecutionPlanExecutor",
            action="parameter_resolution_failed",
            status=TraceEventStatus.FAILED.value,
            details=details,
        )

    def _trace_variable_resolution_started(
        self,
        trace: ExecutionTrace,
        step: ExecutionStep,
    ) -> None:
        variable_references = self._variable_reference_labels(step.arguments)
        if not variable_references:
            return
        trace.add_event(
            component="ExecutionPlanExecutor",
            action="variable_resolution_started",
            status=TraceEventStatus.STARTED.value,
            details={
                "step_id": step.id,
                "tool_name": step.tool,
                "variable_reference_count": len(variable_references),
                "variable_references": variable_references,
            },
        )

    def _trace_variable_resolution_succeeded(
        self,
        trace: ExecutionTrace,
        step: ExecutionStep,
        resolution: Any,
    ) -> None:
        variable_references = self._variable_reference_labels(step.arguments)
        if not variable_references:
            return
        trace.add_event(
            component="ExecutionPlanExecutor",
            action="variable_resolution_succeeded",
            status=TraceEventStatus.FINISHED.value,
            details={
                "step_id": step.id,
                "tool_name": step.tool,
                "variable_names": sorted(set(resolution.used_variable_names)),
                "variable_reference_count": len(variable_references),
            },
        )

    def _trace_variable_resolution_failed(
        self,
        trace: ExecutionTrace,
        step: ExecutionStep,
        resolution: Any,
    ) -> None:
        variable_references = self._variable_reference_labels(step.arguments)
        if not variable_references:
            return
        trace.add_event(
            component="ExecutionPlanExecutor",
            action="variable_resolution_failed",
            status=TraceEventStatus.FAILED.value,
            details={
                "step_id": step.id,
                "tool_name": step.tool,
                "variable_names": sorted(set(resolution.used_variable_names)),
                "variable_reference_count": len(variable_references),
                "error_code": resolution.error_code,
                "unresolved_references": list(resolution.unresolved_references),
            },
        )

    def _trace_output_binding_started(
        self,
        trace: ExecutionTrace,
        context: ExecutionContext,
        step: ExecutionStep,
    ) -> None:
        trace.add_event(
            component="ExecutionPlanExecutor",
            action="execution_variable_binding_started",
            status=TraceEventStatus.STARTED.value,
            details=self._output_binding_trace_details(context, step),
        )

    def _trace_output_binding_succeeded(
        self,
        trace: ExecutionTrace,
        context: ExecutionContext,
        step: ExecutionStep,
    ) -> None:
        trace.add_event(
            component="ExecutionPlanExecutor",
            action="execution_variable_binding_succeeded",
            status=TraceEventStatus.FINISHED.value,
            details=self._output_binding_trace_details(context, step),
        )

    def _trace_output_binding_failed(
        self,
        trace: ExecutionTrace,
        context: ExecutionContext,
        step: ExecutionStep,
        *,
        error_code: str,
    ) -> None:
        details = self._output_binding_trace_details(context, step)
        details["error_code"] = error_code
        trace.add_event(
            component="ExecutionPlanExecutor",
            action="execution_variable_binding_failed",
            status=TraceEventStatus.FAILED.value,
            details=details,
        )

    def _output_binding_trace_details(
        self,
        context: ExecutionContext,
        step: ExecutionStep,
    ) -> dict[str, object]:
        binding = step.output_binding
        assert binding is not None
        return {
            "execution_id": context.execution_id,
            "step_id": step.id,
            "variable_name": binding.variable_name,
            "path": list(binding.path),
            "overwrite": binding.overwrite,
            "variable_count": len(context.variables_snapshot()),
        }

    def _parameter_resolution_trace_details(
        self,
        step: ExecutionStep,
    ) -> dict[str, object]:
        references = self._reference_labels(step.arguments)
        return {
            "step_id": step.id,
            "tool_name": step.tool,
            "reference_count": len(references),
            "references": references,
        }

    def _has_resolvable_reference(
        self,
        value: object,
    ) -> bool:
        return bool(self._reference_labels(value))

    def _reference_labels(
        self,
        value: object,
    ) -> list[str]:
        from core.execution_variable_reference import ExecutionVariableReference
        from core.step_output_reference import StepOutputReference

        labels: list[str] = []

        def visit(item: object) -> None:
            if isinstance(item, ExecutionVariableReference):
                labels.append(self._variable_reference_label(item))
                return

            if isinstance(item, StepOutputReference):
                if item.path:
                    path = ".".join(str(part) for part in item.path)
                    labels.append(f"steps.{item.step_id}.output:{path}")
                else:
                    labels.append(f"steps.{item.step_id}.output")
                return

            if isinstance(item, Mapping):
                if tuple(item.keys()) == ("$ref",) and isinstance(item.get("$ref"), str):
                    labels.append(str(item["$ref"]))
                    return
                if "$template" in item:
                    labels.append("<template>")
                    return
                for nested in item.values():
                    visit(nested)
                return

            if isinstance(item, (list, tuple)):
                for nested in item:
                    visit(nested)

        visit(value)
        return sorted(labels)

    def _variable_reference_labels(
        self,
        value: object,
    ) -> list[str]:
        from core.execution_variable_reference import ExecutionVariableReference

        labels: list[str] = []

        def visit(item: object) -> None:
            if isinstance(item, ExecutionVariableReference):
                labels.append(self._variable_reference_label(item))
                return

            if isinstance(item, Mapping):
                for nested in item.values():
                    visit(nested)
                return

            if isinstance(item, (list, tuple)):
                for nested in item:
                    visit(nested)

        visit(value)
        return sorted(labels)

    def _variable_reference_label(
        self,
        reference: Any,
    ) -> str:
        if reference.path:
            path = ".".join(str(part) for part in reference.path)
            return f"variables.{reference.name}:{path}"
        return f"variables.{reference.name}"

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
        del delay_ms
        before = self._retry_control_status(control)
        if before is not None:
            return before

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


def _safe_topological_execution_steps(
    plan: ExecutionPlan,
) -> tuple[ExecutionStep, ...]:
    try:
        return ExecutionPlanTopologicalSorter().sort(plan).ordered_steps(plan)
    except Exception:
        return tuple(plan.ordered_steps)


def build_partial_execution_state(
    *,
    objective: str,
    plan: ExecutionPlan,
    validation_result: PlanValidationResult | None,
    execution: PlanExecutionResult,
) -> PartialExecutionState:
    """Build a safe partial execution state from the executor result."""
    execution_steps = _safe_topological_execution_steps(plan)
    step_by_id = {step.id: step for step in execution_steps}
    raw_result_by_id = {result.step_id: result for result in execution.step_results}
    ordered_step_ids = tuple(step.id for step in execution_steps)
    raw_failed_step_ids = tuple(execution.failed_steps)
    if (
        not raw_failed_step_ids
        and execution.plan_status == PlanExecutionStatus.FAILED.value
        and execution.current_step is not None
    ):
        raw_failed_step_ids = (execution.current_step,)
    completed = tuple(step_id for step_id in ordered_step_ids if step_id in execution.completed_steps)
    failed = tuple(step_id for step_id in ordered_step_ids if step_id in raw_failed_step_ids)
    blocked = tuple(step_id for step_id in ordered_step_ids if step_id in execution.blocked_steps)
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
        and step_id not in blocked
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
    for step in execution_steps:
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
        blocked_step_ids=blocked,
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


def _trace_status_for_result(
    result: PlanExecutionResult,
) -> str:
    if result.cancelled or result.plan_status == PlanExecutionStatus.CANCELLED.value:
        return TraceStatus.CANCELLED.value
    if result.success or result.plan_status == PlanExecutionStatus.COMPLETED.value:
        return TraceStatus.SUCCESS.value
    return TraceStatus.FAILED.value


def _step_arguments_dict(
    step: ExecutionStep,
) -> dict[str, object]:
    if isinstance(step.arguments, ExecutionArguments):
        return step.arguments.as_dict()
    return dict(step.arguments)


def _elapsed_ms(
    started: float,
) -> int:
    return max(0, int((time.perf_counter() - started) * 1000))


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
        "blocked_step_ids": list(state.blocked_step_ids),
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
        blocked_step_ids=_payload_optional_str_tuple(
            payload,
            "blocked_step_ids",
            default=(),
        ),
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

    plan_step_ids = tuple(
        step.id for step in _safe_topological_execution_steps(state.original_plan)
    )
    plan_step_set = set(plan_step_ids)
    buckets = (
        state.completed_step_ids,
        state.failed_step_ids,
        state.blocked_step_ids,
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
        if state.pending_step_ids or state.failed_step_ids:
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
        if (
            not state.failed_step_ids
            and state.metadata.get("error_code")
            != ExecutionErrorCode.EXECUTION_PLAN_OUTPUT_RESOLUTION_FAILED.value
        ):
            raise ValueError("Failed execution requires at least one failed step.")

    if state.overall_status == PlanExecutionStatus.BLOCKED.value:
        if not state.blocked_step_ids:
            raise ValueError("Blocked execution requires at least one blocked step.")

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
    if step_id in execution.blocked_steps:
        return PartialStepStatus.BLOCKED.value
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
        if execution.plan_status == PlanExecutionStatus.BLOCKED_CONFIRMATION.value:
            return PartialStepStatus.BLOCKED_CONFIRMATION.value
        return PartialStepStatus.BLOCKED.value
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
    return bool(scheduled or exhausted or reason == RetryReason.MAX_RETRIES_REACHED.value)


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
    safe["blocked_count"] = len(execution.blocked_steps)
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


def _payload_optional_str_tuple(
    payload: dict[str, object],
    key: str,
    *,
    default: tuple[str, ...],
) -> tuple[str, ...]:
    if key not in payload:
        return default
    return _payload_str_tuple(payload, key)


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
