"""Deterministic, bounded correction contracts for unmet execution objectives."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from hashlib import sha256
import json
from types import MappingProxyType
from typing import Mapping
from uuid import uuid4

from core.acceptance_criteria import AcceptanceCriterionKind
from core.execution_plan_validator import plan_signature
from core.goal_verifier import (
    CriterionEvaluationStatus,
    GoalVerificationResult,
    GoalVerificationStatus,
)
from core.planner import ExecutionPlan


MAX_CORRECTION_STEPS = 3
MAX_CORRECTION_RESOURCES = 1
MAX_CORRECTION_CONFIRMATIONS = 1


class CorrectionClassification(str, Enum):
    """Closed classification produced before any corrective action."""

    CORRECTABLE = "CORRECTABLE"
    NOT_CORRECTABLE = "NOT_CORRECTABLE"
    USER_INPUT_REQUIRED = "USER_INPUT_REQUIRED"
    UNSAFE_TO_CORRECT = "UNSAFE_TO_CORRECT"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    LIMIT_REACHED = "LIMIT_REACHED"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class CorrectionType(str, Enum):
    """Small allow-list of corrective actions."""

    REPEAT_VERIFICATION_STEP = "REPEAT_VERIFICATION_STEP"
    REWRITE_RESOURCE = "REWRITE_RESOURCE"
    REGENERATE_DERIVED_OUTPUT = "REGENERATE_DERIVED_OUTPUT"
    RESTORE_EXPECTED_VALUE = "RESTORE_EXPECTED_VALUE"
    COMPLETE_MISSING_STEP = "COMPLETE_MISSING_STEP"
    REQUEST_USER_ACTION = "REQUEST_USER_ACTION"
    NO_SAFE_CORRECTION = "NO_SAFE_CORRECTION"


class CorrectionLifecycleStatus(str, Enum):
    """Persistable lifecycle state for one correction request."""

    NOT_STARTED = "NOT_STARTED"
    REJECTED = "REJECTED"
    PENDING_CONFIRMATION = "PENDING_CONFIRMATION"
    AUTHORIZED = "AUTHORIZED"
    DISPATCHED = "DISPATCHED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    VERIFIED_AFTER_CORRECTION = "VERIFIED_AFTER_CORRECTION"
    LIMIT_REACHED = "LIMIT_REACHED"


@dataclass(frozen=True, slots=True)
class ObjectiveCorrectionPolicy:
    """Conservative limits; recursive correction is always disabled."""

    enabled: bool = True
    max_cycles: int = 1
    max_steps: int = MAX_CORRECTION_STEPS
    max_resources: int = MAX_CORRECTION_RESOURCES
    max_new_confirmations: int = MAX_CORRECTION_CONFIRMATIONS
    max_final_verifications: int = 1

    def __post_init__(self) -> None:
        if type(self.enabled) is not bool:
            raise TypeError("enabled must be a bool.")
        for name, upper in (
            ("max_cycles", 1),
            ("max_steps", MAX_CORRECTION_STEPS),
            ("max_resources", MAX_CORRECTION_RESOURCES),
            ("max_new_confirmations", MAX_CORRECTION_CONFIRMATIONS),
            ("max_final_verifications", 1),
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an int.")
            if value < 0 or value > upper:
                raise ValueError(f"{name} must be between 0 and {upper}.")


@dataclass(frozen=True, slots=True)
class CorrectionCriterionEvidence:
    """Sanitized evidence for one failed or partially satisfied criterion."""

    criterion_id: str
    kind: str
    required: bool
    status: str
    source: str
    expected_value: str | None
    observed_value: str | None
    evidence: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in ("criterion_id", "kind", "status", "source"):
            object.__setattr__(self, name, _required_text(getattr(self, name)))
        if type(self.required) is not bool:
            raise TypeError("required must be a bool.")
        for name in ("expected_value", "observed_value"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _safe_text(value))
        object.__setattr__(
            self,
            "evidence",
            tuple(dict.fromkeys(_safe_text(item) for item in self.evidence))[:16],
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "criterion_id": self.criterion_id,
            "kind": self.kind,
            "required": self.required,
            "status": self.status,
            "source": self.source,
            "expected_value": self.expected_value,
            "observed_value": self.observed_value,
            "evidence": list(self.evidence),
        }


@dataclass(frozen=True, slots=True)
class ObjectiveCorrectionRequest:
    """Serializable request linking an objective failure to a minimal repair."""

    correction_request_id: str
    plan_id: str
    session_id: str
    original_objective: str
    execution_status: str
    verification_status: str
    failed_criteria: tuple[CorrectionCriterionEvidence, ...]
    partially_satisfied_criteria: tuple[CorrectionCriterionEvidence, ...] = ()
    related_step_ids: tuple[str, ...] = ()
    tools_used: tuple[str, ...] = ()
    produced_resources: tuple[str, ...] = ()
    completed_step_ids: tuple[str, ...] = ()
    pending_step_ids: tuple[str, ...] = ()
    previous_attempts: int = 0
    remaining_cycles: int = 1
    required_confirmations: tuple[str, ...] = ()
    reason: str = ""

    def __post_init__(self) -> None:
        for name in (
            "correction_request_id",
            "plan_id",
            "session_id",
            "original_objective",
            "execution_status",
            "verification_status",
            "reason",
        ):
            object.__setattr__(self, name, _required_text(getattr(self, name)))
        for name in (
            "related_step_ids",
            "tools_used",
            "produced_resources",
            "completed_step_ids",
            "pending_step_ids",
            "required_confirmations",
        ):
            object.__setattr__(
                self,
                name,
                tuple(dict.fromkeys(_required_text(item) for item in getattr(self, name))),
            )
        if (
            isinstance(self.previous_attempts, bool)
            or not isinstance(self.previous_attempts, int)
            or self.previous_attempts < 0
        ):
            raise ValueError("previous_attempts must be a non-negative int.")
        if (
            isinstance(self.remaining_cycles, bool)
            or not isinstance(self.remaining_cycles, int)
            or self.remaining_cycles < 0
        ):
            raise ValueError("remaining_cycles must be a non-negative int.")

    def to_dict(self) -> dict[str, object]:
        return {
            "correction_request_id": self.correction_request_id,
            "plan_id": self.plan_id,
            "session_id": self.session_id,
            "original_objective": self.original_objective,
            "execution_status": self.execution_status,
            "verification_status": self.verification_status,
            "failed_criteria": [item.to_dict() for item in self.failed_criteria],
            "partially_satisfied_criteria": [
                item.to_dict() for item in self.partially_satisfied_criteria
            ],
            "evidence": list(
                dict.fromkeys(
                    evidence
                    for criterion in self.failed_criteria
                    for evidence in criterion.evidence
                )
            ),
            "related_step_ids": list(self.related_step_ids),
            "tools_used": list(self.tools_used),
            "produced_resources": list(self.produced_resources),
            "completed_step_ids": list(self.completed_step_ids),
            "pending_step_ids": list(self.pending_step_ids),
            "previous_attempts": self.previous_attempts,
            "remaining_cycles": self.remaining_cycles,
            "required_confirmations": list(self.required_confirmations),
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class ObjectiveCorrectionDecision:
    """Classification plus optional deterministic fragment."""

    classification: CorrectionClassification
    correction_type: CorrectionType
    request: ObjectiveCorrectionRequest
    fragment: ExecutionPlan | None = None
    affected_criterion_ids: tuple[str, ...] = ()
    expected_context: Mapping[str, object] = field(default_factory=dict)
    fragment_signature: str | None = None
    reason: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.classification, CorrectionClassification):
            object.__setattr__(
                self, "classification", CorrectionClassification(self.classification)
            )
        if not isinstance(self.correction_type, CorrectionType):
            object.__setattr__(self, "correction_type", CorrectionType(self.correction_type))
        object.__setattr__(
            self,
            "affected_criterion_ids",
            tuple(dict.fromkeys(_required_text(item) for item in self.affected_criterion_ids)),
        )
        object.__setattr__(
            self,
            "expected_context",
            MappingProxyType(dict(self.expected_context)),
        )
        if self.fragment is not None:
            signature = plan_signature(self.fragment)
            if self.fragment_signature is not None and self.fragment_signature != signature:
                raise ValueError("fragment_signature does not match fragment.")
            object.__setattr__(self, "fragment_signature", signature)
        elif self.classification is CorrectionClassification.CORRECTABLE:
            raise ValueError("CORRECTABLE decision requires a fragment.")
        object.__setattr__(self, "reason", _required_text(self.reason))

    def to_dict(self) -> dict[str, object]:
        return {
            "classification": self.classification.value,
            "correction_type": self.correction_type.value,
            "request": self.request.to_dict(),
            "affected_criterion_ids": list(self.affected_criterion_ids),
            "fragment_signature": self.fragment_signature,
            "reason": self.reason,
        }


def build_correction_request(
    plan: ExecutionPlan,
    execution_result: object,
    verification: GoalVerificationResult,
    *,
    session_id: str,
    previous_attempts: int = 0,
    policy: ObjectiveCorrectionPolicy | None = None,
) -> ObjectiveCorrectionRequest:
    """Build a sanitized request without inferring a remedy."""

    active_policy = policy or ObjectiveCorrectionPolicy()
    failed = tuple(
        _criterion_evidence(item)
        for item in verification.criteria
        if item.status is CriterionEvaluationStatus.FAILED
    )
    partial = tuple(
        _criterion_evidence(item)
        for item in verification.criteria
        if item.status is CriterionEvaluationStatus.INCONCLUSIVE
    )
    step_results = tuple(getattr(execution_result, "step_results", ()) or ())
    failed_ids = {item.criterion_id for item in failed + partial}
    related_steps = tuple(
        dict.fromkeys(
            criterion.source_step_id
            for criterion in plan.acceptance_criteria
            if criterion.criterion_id in failed_ids
            and criterion.source_step_id is not None
        )
    )
    resources = tuple(
        dict.fromkeys(
            criterion.resource_path
            for criterion in plan.acceptance_criteria
            if criterion.criterion_id in failed_ids
            and criterion.resource_path is not None
        )
    )
    tools = tuple(
        dict.fromkeys(
            item.tool_name
            for item in step_results
            if isinstance(getattr(item, "tool_name", None), str)
        )
    )
    return ObjectiveCorrectionRequest(
        correction_request_id=f"correction.{uuid4().hex}",
        plan_id=verification.plan_id or plan_signature(plan),
        session_id=session_id,
        original_objective=plan.goal,
        execution_status=verification.execution_status or "unknown",
        verification_status=verification.verification_status.value,
        failed_criteria=failed,
        partially_satisfied_criteria=partial,
        related_step_ids=related_steps,
        tools_used=tools,
        produced_resources=resources,
        completed_step_ids=tuple(
            getattr(execution_result, "completed_steps", ()) or ()
        ),
        pending_step_ids=tuple(getattr(execution_result, "pending_steps", ()) or ()),
        previous_attempts=previous_attempts,
        remaining_cycles=max(0, active_policy.max_cycles - previous_attempts),
        required_confirmations=("resource_write",) if resources else (),
        reason=verification.message or "Objective verification requires review.",
    )


def classify_correction(
    plan: ExecutionPlan,
    execution_result: object,
    verification: GoalVerificationResult,
    *,
    session_id: str,
    previous_attempts: int = 0,
    policy: ObjectiveCorrectionPolicy | None = None,
) -> tuple[CorrectionClassification, CorrectionType, ObjectiveCorrectionRequest, str]:
    """Classify only remedies that are safe and demonstrable."""

    active_policy = policy or ObjectiveCorrectionPolicy()
    request = build_correction_request(
        plan,
        execution_result,
        verification,
        session_id=session_id,
        previous_attempts=previous_attempts,
        policy=active_policy,
    )
    status = verification.verification_status
    if not active_policy.enabled or status in {
        GoalVerificationStatus.VERIFIED,
        GoalVerificationStatus.NOT_APPLICABLE,
    }:
        return (
            CorrectionClassification.NOT_APPLICABLE,
            CorrectionType.NO_SAFE_CORRECTION,
            request,
            "The verification result does not require correction.",
        )
    if previous_attempts >= active_policy.max_cycles:
        return (
            CorrectionClassification.LIMIT_REACHED,
            CorrectionType.NO_SAFE_CORRECTION,
            request,
            "The corrective cycle limit was reached.",
        )
    if status is GoalVerificationStatus.USER_ACTION_REQUIRED:
        return (
            CorrectionClassification.USER_INPUT_REQUIRED,
            CorrectionType.REQUEST_USER_ACTION,
            request,
            "User action is required before any correction.",
        )
    if status is GoalVerificationStatus.INCONCLUSIVE:
        return (
            CorrectionClassification.INSUFFICIENT_EVIDENCE,
            CorrectionType.NO_SAFE_CORRECTION,
            request,
            "Inconclusive verification cannot start speculative correction.",
        )
    failed_required = tuple(item for item in request.failed_criteria if item.required)
    if any(
        item.required
        and item.kind == AcceptanceCriterionKind.RESOURCE_CONTENT_EQUALS.value
        for item in request.partially_satisfied_criteria
    ):
        return (
            CorrectionClassification.INSUFFICIENT_EVIDENCE,
            CorrectionType.NO_SAFE_CORRECTION,
            request,
            "The failed resource criterion has no demonstrated expected value.",
        )
    if not failed_required:
        return (
            CorrectionClassification.NOT_CORRECTABLE,
            CorrectionType.NO_SAFE_CORRECTION,
            request,
            "No failed required criterion needs correction.",
        )
    content_failures = tuple(
        item
        for item in failed_required
        if item.kind == AcceptanceCriterionKind.RESOURCE_CONTENT_EQUALS.value
    )
    if len(content_failures) != 1:
        return (
            CorrectionClassification.NOT_CORRECTABLE,
            CorrectionType.NO_SAFE_CORRECTION,
            request,
            "No single deterministic resource-content correction is available.",
        )
    criterion = next(
        item
        for item in plan.acceptance_criteria
        if item.criterion_id == content_failures[0].criterion_id
    )
    if criterion.comparison_step_id is None and criterion.expected_value is None:
        return (
            CorrectionClassification.INSUFFICIENT_EVIDENCE,
            CorrectionType.NO_SAFE_CORRECTION,
            request,
            "The failed criterion has no demonstrated expected value.",
        )
    if criterion.resource_path is None or len(request.produced_resources) != 1:
        return (
            CorrectionClassification.UNSAFE_TO_CORRECT,
            CorrectionType.NO_SAFE_CORRECTION,
            request,
            "The correction is not limited to one declared resource.",
        )
    return (
        CorrectionClassification.CORRECTABLE,
        CorrectionType.REWRITE_RESOURCE,
        request,
        "One declared resource can be rewritten from demonstrated execution evidence.",
    )


def correction_fragment_fingerprint(
    request_id: str,
    fragment: ExecutionPlan,
) -> str:
    """Return a stable repeat-detection fingerprint."""

    payload = json.dumps(
        {
            "fragment_signature": plan_signature(fragment),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return sha256(payload.encode("utf-8")).hexdigest()


def merge_corrective_execution(
    original_plan: ExecutionPlan,
    original_execution: object,
    corrective_execution: object,
    decision: ObjectiveCorrectionDecision,
) -> object:
    """Overlay only corrected write/read evidence for final objective verification."""

    if decision.fragment is None or not decision.request.produced_resources:
        raise ValueError("A corrective fragment with one resource is required.")
    resource_path = decision.request.produced_resources[0]
    original_write = next(
        (
            step
            for step in original_plan.ordered_steps
            if step.tool == "write_file" and step.arguments.get("path") == resource_path
        ),
        None,
    )
    original_read = next(
        (
            step
            for step in reversed(original_plan.ordered_steps)
            if step.tool == "read_file" and step.arguments.get("path") == resource_path
        ),
        None,
    )
    corrective_results = tuple(
        getattr(corrective_execution, "step_results", ()) or ()
    )
    corrective_write = next(
        (item for item in corrective_results if getattr(item, "tool_name", None) == "write_file"),
        None,
    )
    corrective_read = next(
        (
            item
            for item in reversed(corrective_results)
            if getattr(item, "tool_name", None) == "read_file"
        ),
        None,
    )
    if original_write is None or original_read is None:
        raise ValueError("Original write/read steps for the corrected resource are required.")
    if corrective_read is None:
        raise ValueError("Corrective read result is required.")

    overlays = {
        original_read.id: replace(corrective_read, step_id=original_read.id),
    }
    if corrective_write is not None:
        overlays[original_write.id] = replace(
            corrective_write,
            step_id=original_write.id,
        )
    merged_results = []
    seen: set[str] = set()
    for item in tuple(getattr(original_execution, "step_results", ()) or ()):
        step_id = getattr(item, "step_id", "")
        merged_results.append(overlays.get(step_id, item))
        seen.add(step_id)
    for step_id, item in overlays.items():
        if step_id not in seen:
            merged_results.append(item)
    metadata = {
        **dict(getattr(original_execution, "metadata", {}) or {}),
        "correction_request_id": decision.request.correction_request_id,
        "correction_fragment_signature": decision.fragment_signature,
        "corrective_cycle": decision.request.previous_attempts + 1,
    }
    return replace(
        original_execution,
        step_results=merged_results,
        completed_steps=list(
            dict.fromkeys(
                (*tuple(getattr(original_execution, "completed_steps", ()) or ()),
                 original_write.id, original_read.id)
            )
        ),
        goal_verification_result=None,
        metadata=metadata,
    )


def _criterion_evidence(item: object) -> CorrectionCriterionEvidence:
    return CorrectionCriterionEvidence(
        criterion_id=getattr(item, "criterion_id"),
        kind=getattr(item, "kind"),
        required=getattr(item, "required"),
        status=getattr(getattr(item, "status"), "value", getattr(item, "status")),
        source=getattr(item, "source"),
        expected_value=getattr(item, "expected_value"),
        observed_value=getattr(item, "observed_value"),
        evidence=tuple(getattr(item, "evidence", ()) or ()),
    )


def _required_text(value: object) -> str:
    text = _safe_text(value).strip()
    if not text:
        raise ValueError("required text cannot be empty.")
    return text


def _safe_text(value: object) -> str:
    text = str(value).replace("\x00", "")
    return text if len(text) <= 4096 else text[:4093] + "..."
