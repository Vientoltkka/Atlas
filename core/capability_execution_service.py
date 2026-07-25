"""Explicit capability execution service for Atlas."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType

from core.capability_orchestrator import (
    CapabilityOrchestrationPolicy,
    CapabilityOrchestrationRequest,
    CapabilityOrchestrationResult,
    CapabilityOrchestrationStatus,
    CapabilityOrchestrator,
)
from core.capability_planner import CapabilityPlanningRequest
from core.capability_resolver import CapabilityType, WorkflowCapabilitySource
from core.execution_plan_executor import ExecutionControl
from core.execution_replanner import ReplanningPolicy
from core.goal_verifier import GoalVerificationResult
from core.execution_plan_registry import ExecutionPlanReference
from core.multi_capability_planner import (
    MultiCapabilityPlanner,
    MultiCapabilityPlanningRequest,
    MultiCapabilityPlanningStatus,
)


MAX_CAPABILITY_EXECUTION_ITEMS = 64
MAX_CAPABILITY_EXECUTION_METADATA_ITEMS = 32
MAX_CAPABILITY_EXECUTION_INPUT_ITEMS = 32
MAX_CAPABILITY_EXECUTION_INPUT_DEPTH = 4
MAX_CAPABILITY_EXECUTION_OUTPUT_DEPTH = 8
MAX_CAPABILITY_EXECUTION_OUTPUT_NODES = 128
SENSITIVE_KEY_PARTS = ("secret", "token", "password", "api_key", "apikey", "authorization")


class CapabilityExecutionError(RuntimeError):
    """Base error for capability execution service contract violations."""


class InvalidCapabilityExecutionRequestError(CapabilityExecutionError):
    """Raised when a capability execution request is malformed."""


class CapabilityExecutionStatus(str, Enum):
    """Stable public states for explicit capability execution."""

    COMPLETED = "completed"
    SERVICE_UNAVAILABLE = "service_unavailable"
    INVALID_REQUEST = "invalid_request"
    NO_CAPABILITY_CANDIDATES = "no_capability_candidates"
    CAPABILITY_AMBIGUOUS = "capability_ambiguous"
    NO_WORKFLOW_CANDIDATES = "no_workflow_candidates"
    WORKFLOW_BELOW_MINIMUM_SCORE = "workflow_below_minimum_score"
    WORKFLOW_AMBIGUOUS = "workflow_ambiguous"
    PLANNING_FAILED = "planning_failed"
    PLAN_VALIDATION_FAILED = "plan_validation_failed"
    EXECUTION_FAILED = "execution_failed"
    CANCELLED = "cancelled"
    INTERNAL_ERROR = "internal_error"


@dataclass(frozen=True, slots=True)
class CapabilityExecutionSelectedCapability:
    """Safe selected capability metadata."""

    capability_id: str
    capability_type: str
    title: str
    categories: tuple[str, ...]
    tags: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CapabilityExecutionRequest:
    """Structured request for executing a workflow-backed capability."""

    objective: str = "execute capability"
    capability_id: str | None = None
    capability_type: CapabilityType | str | None = CapabilityType.WORKFLOW
    categories: tuple[str, ...] = ()
    excluded_categories: tuple[str, ...] = ()
    required_tags: tuple[str, ...] = ()
    preferred_tags: tuple[str, ...] = ()
    required_inputs: tuple[str, ...] = ()
    required_outputs: tuple[str, ...] = ()
    preferred_workflow_reference: ExecutionPlanReference | None = None
    minimum_score: int = 0
    minimum_workflow_score: int = 0
    require_unique_top_score: bool = True
    enabled_only: bool = True
    confirmation_granted: bool = False
    control: ExecutionControl | None = None
    replanning_policy: ReplanningPolicy | None = None
    inputs: Mapping[str, object] = field(default_factory=dict)
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.objective, str) or not self.objective.strip():
            raise InvalidCapabilityExecutionRequestError("objective must be a non-empty string.")
        object.__setattr__(self, "objective", " ".join(self.objective.split()))
        capability_type = _validate_capability_type(self.capability_type)
        if capability_type is not CapabilityType.WORKFLOW:
            raise InvalidCapabilityExecutionRequestError("only workflow capabilities can be executed.")
        object.__setattr__(self, "capability_type", capability_type)
        if self.preferred_workflow_reference is not None and not isinstance(
            self.preferred_workflow_reference,
            ExecutionPlanReference,
        ):
            raise InvalidCapabilityExecutionRequestError(
                "preferred_workflow_reference must be ExecutionPlanReference or None."
            )
        if not isinstance(self.confirmation_granted, bool):
            raise InvalidCapabilityExecutionRequestError("confirmation_granted must be a bool.")
        if self.control is not None and not isinstance(self.control, ExecutionControl):
            raise InvalidCapabilityExecutionRequestError("control must be ExecutionControl or None.")
        if self.replanning_policy is not None and not isinstance(self.replanning_policy, ReplanningPolicy):
            raise InvalidCapabilityExecutionRequestError("replanning_policy must be ReplanningPolicy or None.")
        object.__setattr__(self, "inputs", MappingProxyType(_safe_inputs(self.inputs)))
        object.__setattr__(self, "metadata", MappingProxyType(_safe_metadata(self.metadata)))
        _build_planning_request(self)


@dataclass(frozen=True, slots=True)
class CapabilityExecutionResult:
    """Safe public result for explicit capability execution."""

    status: CapabilityExecutionStatus
    selected_capability: CapabilityExecutionSelectedCapability | None = None
    selected_workflow_reference: ExecutionPlanReference | None = None
    plan_signature: str | None = None
    execution_id: str | None = None
    execution_status: str | None = None
    goal_verification_result: GoalVerificationResult | None = None
    output: object | None = None
    error_code: str | None = None
    message: str | None = None
    replanning_attempted: bool = False
    replan_attempts: int = 0
    replanning_status: str | None = None
    replanning_reason: str | None = None
    original_plan_signature: str | None = None
    final_plan_signature: str | None = None
    orchestration_result: CapabilityOrchestrationResult | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", _validate_status(self.status))
        if self.selected_workflow_reference is not None and not isinstance(
            self.selected_workflow_reference,
            ExecutionPlanReference,
        ):
            raise InvalidCapabilityExecutionRequestError(
                "selected_workflow_reference must be ExecutionPlanReference or None."
            )
        object.__setattr__(self, "output", _safe_output(self.output))
        object.__setattr__(self, "message", _safe_message(self.message))

    @property
    def completed(self) -> bool:
        """Return whether the explicit capability execution completed."""

        return self.status is CapabilityExecutionStatus.COMPLETED


class CapabilityExecutionService:
    """Convert explicit capability requests into controlled orchestration calls."""

    _STATUS_MAP = {
        CapabilityOrchestrationStatus.COMPLETED: CapabilityExecutionStatus.COMPLETED,
        CapabilityOrchestrationStatus.INVALID_REQUEST: CapabilityExecutionStatus.INVALID_REQUEST,
        CapabilityOrchestrationStatus.NO_CAPABILITY_CANDIDATES: CapabilityExecutionStatus.NO_CAPABILITY_CANDIDATES,
        CapabilityOrchestrationStatus.CAPABILITY_AMBIGUOUS: CapabilityExecutionStatus.CAPABILITY_AMBIGUOUS,
        CapabilityOrchestrationStatus.NO_WORKFLOW_CANDIDATES: CapabilityExecutionStatus.NO_WORKFLOW_CANDIDATES,
        CapabilityOrchestrationStatus.WORKFLOW_BELOW_MINIMUM_SCORE: (
            CapabilityExecutionStatus.WORKFLOW_BELOW_MINIMUM_SCORE
        ),
        CapabilityOrchestrationStatus.WORKFLOW_AMBIGUOUS: CapabilityExecutionStatus.WORKFLOW_AMBIGUOUS,
        CapabilityOrchestrationStatus.PLANNING_FAILED: CapabilityExecutionStatus.PLANNING_FAILED,
        CapabilityOrchestrationStatus.PLAN_VALIDATION_FAILED: CapabilityExecutionStatus.PLAN_VALIDATION_FAILED,
        CapabilityOrchestrationStatus.EXECUTION_FAILED: CapabilityExecutionStatus.EXECUTION_FAILED,
        CapabilityOrchestrationStatus.CANCELLED: CapabilityExecutionStatus.CANCELLED,
    }

    def __init__(
        self,
        capability_orchestrator: CapabilityOrchestrator,
        *,
        multi_capability_planner: MultiCapabilityPlanner | None = None,
    ) -> None:
        if not isinstance(capability_orchestrator, CapabilityOrchestrator):
            raise CapabilityExecutionError("CapabilityExecutionService requires CapabilityOrchestrator.")
        if multi_capability_planner is not None and not isinstance(
            multi_capability_planner,
            MultiCapabilityPlanner,
        ):
            raise CapabilityExecutionError("multi_capability_planner must be MultiCapabilityPlanner or None.")
        self._capability_orchestrator = capability_orchestrator
        self._multi_capability_planner = multi_capability_planner

    def execute(self, request: CapabilityExecutionRequest) -> CapabilityExecutionResult:
        """Execute one explicit workflow-backed capability request."""

        if not isinstance(request, CapabilityExecutionRequest):
            return CapabilityExecutionResult(
                CapabilityExecutionStatus.INVALID_REQUEST,
                error_code="INVALID_REQUEST",
                message="request must be CapabilityExecutionRequest.",
            )

        try:
            multi_result = self._execute_multi_capability_request(request)
            if multi_result is not None:
                return multi_result
            orchestration_request = CapabilityOrchestrationRequest(
                planning_request=_build_planning_request(request),
                policy=CapabilityOrchestrationPolicy(
                    confirmation_granted=request.confirmation_granted,
                    control=request.control,
                    replanning_policy=request.replanning_policy,
                ),
                inputs=request.inputs,
                metadata=request.metadata,
            )
            orchestration_result = self._capability_orchestrator.orchestrate(orchestration_request)
        except (CapabilityExecutionError, ValueError, TypeError, RuntimeError):
            return CapabilityExecutionResult(
                CapabilityExecutionStatus.INTERNAL_ERROR,
                error_code="INTERNAL_ERROR",
                message="Capability execution failed before orchestration completed.",
            )

        return _result_from_orchestration(orchestration_result)

    def _execute_multi_capability_request(
        self,
        request: CapabilityExecutionRequest,
    ) -> CapabilityExecutionResult | None:
        if self._multi_capability_planner is None:
            return None
        if request.capability_id is not None or request.preferred_workflow_reference is not None:
            return None
        if not request.required_outputs or not request.inputs:
            return None

        planning = self._multi_capability_planner.plan(
            MultiCapabilityPlanningRequest(
                initial_inputs=tuple(request.inputs.keys()),
                required_outputs=request.required_outputs,
            )
        )
        if planning.status is not MultiCapabilityPlanningStatus.PLANNED or planning.plan is None:
            return None
        if len(planning.plan.ordered_steps) <= 1:
            return None

        orchestration_result = self._capability_orchestrator.orchestrate_plan(
            planning.plan,
            policy=CapabilityOrchestrationPolicy(
                confirmation_granted=request.confirmation_granted,
                control=request.control,
                replanning_policy=request.replanning_policy,
            ),
            inputs=request.inputs,
            metadata={
                "multi_capability": True,
                "selected_capability_count": len(planning.selected_capability_ids),
            },
        )
        return _result_from_orchestration(orchestration_result)


def unavailable_capability_execution_result() -> CapabilityExecutionResult:
    """Return a stable result when no capability execution service is configured."""

    return CapabilityExecutionResult(
        CapabilityExecutionStatus.SERVICE_UNAVAILABLE,
        error_code="CAPABILITY_EXECUTION_SERVICE_UNAVAILABLE",
        message="Capability execution service is not configured.",
    )


def _build_planning_request(request: CapabilityExecutionRequest) -> CapabilityPlanningRequest:
    categories = tuple(dict.fromkeys(("workflow",) + tuple(request.categories)))
    return CapabilityPlanningRequest(
        objective=request.objective,
        capability_id=request.capability_id,
        required_categories=categories,
        excluded_categories=request.excluded_categories,
        required_tags=request.required_tags,
        preferred_tags=request.preferred_tags,
        required_inputs=request.required_inputs,
        required_outputs=request.required_outputs,
        preferred_workflow_reference=request.preferred_workflow_reference,
        minimum_capability_score=request.minimum_score,
        minimum_workflow_score=request.minimum_workflow_score,
        require_unique_workflow=request.require_unique_top_score,
        enabled_only=request.enabled_only,
        metadata=request.metadata,
    )


def _result_from_orchestration(
    orchestration_result: CapabilityOrchestrationResult,
) -> CapabilityExecutionResult:
    decision = orchestration_result.planning_decision
    execution = orchestration_result.execution_result
    selected_workflow_reference = None
    selected_capability = None
    if decision is not None:
        selected_workflow_reference = decision.selected_workflow_reference
        if decision.selected_capability is not None:
            capability = decision.selected_capability
            selected_capability = CapabilityExecutionSelectedCapability(
                capability_id=capability.capability_id,
                capability_type=capability.capability_type.value,
                title=capability.title,
                categories=capability.categories,
                tags=capability.tags,
            )
            if selected_workflow_reference is None and isinstance(
                capability.source_reference,
                WorkflowCapabilitySource,
            ):
                selected_workflow_reference = capability.source_reference.reference

    execution_id = None
    if execution is not None and execution.trace is not None:
        execution_id = execution.trace.execution_id

    status = CapabilityExecutionService._STATUS_MAP.get(
        orchestration_result.status,
        CapabilityExecutionStatus.INTERNAL_ERROR,
    )
    return CapabilityExecutionResult(
        status=status,
        selected_capability=selected_capability,
        selected_workflow_reference=selected_workflow_reference,
        plan_signature=(
            orchestration_result.validation_result.plan_signature
            if orchestration_result.validation_result is not None
            else None
        ),
        execution_id=execution_id,
        execution_status=execution.status if execution is not None else None,
        goal_verification_result=orchestration_result.goal_verification_result,
        output=execution.output if execution is not None else None,
        error_code=orchestration_result.error_code,
        message=_message_for(status, orchestration_result),
        replanning_attempted=orchestration_result.replanning_attempted,
        replan_attempts=orchestration_result.replan_attempts,
        replanning_status=orchestration_result.replanning_status,
        replanning_reason=orchestration_result.replanning_reason,
        original_plan_signature=orchestration_result.original_plan_signature,
        final_plan_signature=orchestration_result.final_plan_signature,
        orchestration_result=orchestration_result,
    )


def _message_for(
    status: CapabilityExecutionStatus,
    orchestration_result: CapabilityOrchestrationResult,
) -> str:
    if status is CapabilityExecutionStatus.COMPLETED:
        return "Capability execution completed."
    if status is CapabilityExecutionStatus.CANCELLED:
        return "Capability execution cancelled."
    if status in {
        CapabilityExecutionStatus.NO_CAPABILITY_CANDIDATES,
        CapabilityExecutionStatus.CAPABILITY_AMBIGUOUS,
        CapabilityExecutionStatus.NO_WORKFLOW_CANDIDATES,
        CapabilityExecutionStatus.WORKFLOW_BELOW_MINIMUM_SCORE,
        CapabilityExecutionStatus.WORKFLOW_AMBIGUOUS,
    }:
        return f"Capability execution stopped with status: {status.value}."
    if orchestration_result.status is CapabilityOrchestrationStatus.PLAN_VALIDATION_FAILED:
        return "Selected capability plan did not pass validation."
    if orchestration_result.status is CapabilityOrchestrationStatus.EXECUTION_FAILED:
        return "Capability execution failed."
    if orchestration_result.status is CapabilityOrchestrationStatus.PLANNING_FAILED:
        return "Capability planning failed."
    return "Capability execution did not complete."


def _validate_capability_type(value: CapabilityType | str | None) -> CapabilityType:
    if value is None:
        return CapabilityType.WORKFLOW
    if isinstance(value, CapabilityType):
        return value
    if isinstance(value, str):
        try:
            return CapabilityType(value.strip().lower())
        except ValueError as error:
            raise InvalidCapabilityExecutionRequestError("invalid capability_type.") from error
    raise InvalidCapabilityExecutionRequestError("capability_type must be CapabilityType, str, or None.")


def _validate_status(status: CapabilityExecutionStatus | str) -> CapabilityExecutionStatus:
    if isinstance(status, CapabilityExecutionStatus):
        return status
    if isinstance(status, str):
        try:
            return CapabilityExecutionStatus(status)
        except ValueError as error:
            raise InvalidCapabilityExecutionRequestError("invalid capability execution status.") from error
    raise InvalidCapabilityExecutionRequestError("status must be CapabilityExecutionStatus.")


def _safe_metadata(metadata: Mapping[str, object]) -> dict[str, object]:
    if not isinstance(metadata, Mapping):
        raise InvalidCapabilityExecutionRequestError("metadata must be a mapping.")
    if len(metadata) > MAX_CAPABILITY_EXECUTION_METADATA_ITEMS:
        raise InvalidCapabilityExecutionRequestError("metadata has too many items.")
    safe: dict[str, object] = {}
    for key, value in metadata.items():
        if not isinstance(key, str) or not key.strip():
            raise InvalidCapabilityExecutionRequestError("metadata keys must be non-empty strings.")
        if _is_sensitive_key(key):
            raise InvalidCapabilityExecutionRequestError("metadata cannot contain sensitive keys.")
        if not _is_safe_primitive(value):
            raise InvalidCapabilityExecutionRequestError("metadata values must be primitive safe values.")
        safe[key] = value
    return safe


def _safe_inputs(inputs: Mapping[str, object]) -> dict[str, object]:
    if not isinstance(inputs, Mapping):
        raise InvalidCapabilityExecutionRequestError("inputs must be a mapping.")
    if len(inputs) > MAX_CAPABILITY_EXECUTION_INPUT_ITEMS:
        raise InvalidCapabilityExecutionRequestError("inputs has too many items.")
    return _copy_safe_input_value(inputs, field_name="inputs", depth=0, counter={"nodes": 0})


def _copy_safe_input_value(
    value: object,
    *,
    field_name: str,
    depth: int,
    counter: dict[str, int],
) -> object:
    if depth > MAX_CAPABILITY_EXECUTION_INPUT_DEPTH:
        raise InvalidCapabilityExecutionRequestError(f"{field_name} is too deep.")
    counter["nodes"] += 1
    if counter["nodes"] > MAX_CAPABILITY_EXECUTION_ITEMS:
        raise InvalidCapabilityExecutionRequestError(f"{field_name} has too many nodes.")
    if _is_safe_primitive(value):
        if isinstance(value, float) and not math.isfinite(value):
            raise InvalidCapabilityExecutionRequestError(f"{field_name} floats must be finite.")
        return value
    if isinstance(value, Mapping):
        if len(value) > MAX_CAPABILITY_EXECUTION_INPUT_ITEMS:
            raise InvalidCapabilityExecutionRequestError(f"{field_name} has too many items.")
        copied: dict[str, object] = {}
        for raw_key, raw_value in value.items():
            if not isinstance(raw_key, str) or not raw_key.strip():
                raise InvalidCapabilityExecutionRequestError(f"{field_name} keys must be non-empty strings.")
            key = raw_key.strip()
            if _is_sensitive_key(key):
                raise InvalidCapabilityExecutionRequestError(f"{field_name} cannot contain sensitive keys.")
            copied[key] = _copy_safe_input_value(
                raw_value,
                field_name=field_name,
                depth=depth + 1,
                counter=counter,
            )
        return copied
    if isinstance(value, (list, tuple)):
        if len(value) > MAX_CAPABILITY_EXECUTION_INPUT_ITEMS:
            raise InvalidCapabilityExecutionRequestError(f"{field_name} has too many items.")
        return tuple(
            _copy_safe_input_value(item, field_name=field_name, depth=depth + 1, counter=counter)
            for item in value
        )
    raise InvalidCapabilityExecutionRequestError(f"{field_name} contains unsupported value.")


def _safe_output(value: object) -> object:
    return _copy_safe_output(value, path="$", depth=0, counter={"nodes": 0})


def _copy_safe_output(value: object, *, path: str, depth: int, counter: dict[str, int]) -> object:
    if depth > MAX_CAPABILITY_EXECUTION_OUTPUT_DEPTH:
        return None
    counter["nodes"] += 1
    if counter["nodes"] > MAX_CAPABILITY_EXECUTION_OUTPUT_NODES:
        return None
    if _is_safe_primitive(value):
        return value
    if isinstance(value, Mapping):
        safe: dict[str, object] = {}
        for raw_key, raw_value in value.items():
            key = str(raw_key)
            if _is_sensitive_key(key):
                safe[key] = "[redacted]"
                continue
            safe[key] = _copy_safe_output(raw_value, path=f"{path}.{key}", depth=depth + 1, counter=counter)
        return safe
    if isinstance(value, (tuple, list)):
        return tuple(
            _copy_safe_output(item, path=f"{path}[]", depth=depth + 1, counter=counter)
            for item in value
        )
    return None


def _safe_message(message: str | None) -> str | None:
    if message is None:
        return None
    text = " ".join(str(message).split())
    for key in SENSITIVE_KEY_PARTS:
        text = text.replace(key, "[redacted]")
    return text[:300]


def _is_safe_primitive(value: object) -> bool:
    return value is None or isinstance(value, (str, int, float, bool))


def _is_sensitive_key(key: str) -> bool:
    normalized = key.replace("-", "_").lower()
    return any(part in normalized for part in SENSITIVE_KEY_PARTS)
