"""Pure deterministic selection over resolved workflow capabilities."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
import math
import re
from types import MappingProxyType, ModuleType

from core.capability_resolver import (
    CapabilityCandidate,
    CapabilityDefinition,
    CapabilityResolutionResult,
    CapabilityType,
    ToolCapabilitySource,
    WorkflowCapabilitySource,
    capability_resolution_request_signature,
)
from core.execution_plan_registry import ExecutionPlanReference


MAX_WORKFLOW_SELECTION_IDS = 64
MAX_WORKFLOW_SELECTION_TAGS = 64
MAX_WORKFLOW_SELECTION_CATEGORIES = 64
MAX_WORKFLOW_SELECTION_CANDIDATES = 256
MAX_WORKFLOW_SELECTION_STRING_LENGTH = 200
MAX_WORKFLOW_SELECTION_METADATA_ITEMS = 32
MAX_WORKFLOW_SELECTION_METADATA_DEPTH = 4

PREFERRED_REFERENCE_BONUS = 6
PREFERRED_LIBRARY_BONUS = 3
PREFERRED_CATEGORY_BONUS = 2
PREFERRED_TAG_BONUS = 1

_IDENTIFIER_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,199}$")


class WorkflowSelectorError(RuntimeError):
    """Base error for workflow selector contract violations."""


class InvalidWorkflowSelectionPolicyError(WorkflowSelectorError):
    """Raised when a selection policy is malformed."""


class InvalidWorkflowSelectionRequestError(WorkflowSelectorError):
    """Raised when a selection request is malformed."""


class ConflictingWorkflowSelectionCandidateError(WorkflowSelectorError):
    """Raised when duplicate workflow candidate identities disagree."""


class WorkflowSelectionStatus(str, Enum):
    """Terminal states for deterministic workflow selection."""

    SELECTED = "selected"
    NO_CANDIDATES = "no_candidates"
    BELOW_MINIMUM_SCORE = "below_minimum_score"
    AMBIGUOUS = "ambiguous"
    INVALID_INPUT = "invalid_input"


class WorkflowSelectionReasonCode(str, Enum):
    """Structured reason codes for selection decisions."""

    WORKFLOW_TYPE_ACCEPTED = "workflow_type_accepted"
    NON_WORKFLOW_REJECTED = "non_workflow_rejected"
    DISABLED_REJECTED = "disabled_rejected"
    LIBRARY_NOT_ALLOWED = "library_not_allowed"
    LIBRARY_EXCLUDED = "library_excluded"
    CATEGORY_NOT_ALLOWED = "category_not_allowed"
    CATEGORY_EXCLUDED = "category_excluded"
    REQUIRED_TAG_MISSING = "required_tag_missing"
    EXCLUDED_TAG_PRESENT = "excluded_tag_present"
    PREFERRED_REFERENCE_MATCH = "preferred_reference_match"
    PREFERRED_LIBRARY_MATCH = "preferred_library_match"
    PREFERRED_CATEGORY_MATCH = "preferred_category_match"
    PREFERRED_TAG_MATCH = "preferred_tag_match"
    BELOW_MINIMUM_SCORE = "below_minimum_score"
    TOP_SCORE_TIE = "top_score_tie"
    UNIQUE_TOP_SELECTED = "unique_top_selected"
    INPUT_INCOHERENT = "input_incoherent"
    DUPLICATE_IDENTICAL = "duplicate_identical"


@dataclass(frozen=True, slots=True)
class WorkflowSelectionPolicy:
    """Deterministic policy for selecting one resolved workflow."""

    minimum_score: int = 0
    require_unique_top_score: bool = True
    enabled_only: bool = True
    allowed_library_ids: tuple[str, ...] = ()
    excluded_library_ids: tuple[str, ...] = ()
    allowed_categories: tuple[str, ...] = ()
    excluded_categories: tuple[str, ...] = ()
    required_tags: tuple[str, ...] = ()
    excluded_tags: tuple[str, ...] = ()
    preferred_library_ids: tuple[str, ...] = ()
    preferred_categories: tuple[str, ...] = ()
    preferred_tags: tuple[str, ...] = ()
    maximum_candidates_considered: int = MAX_WORKFLOW_SELECTION_CANDIDATES

    def __post_init__(self) -> None:
        if isinstance(self.minimum_score, bool) or not isinstance(self.minimum_score, int):
            raise InvalidWorkflowSelectionPolicyError("minimum_score must be a non-negative int.")
        if self.minimum_score < 0:
            raise InvalidWorkflowSelectionPolicyError("minimum_score must be non-negative.")
        if not isinstance(self.require_unique_top_score, bool):
            raise InvalidWorkflowSelectionPolicyError("require_unique_top_score must be a bool.")
        if not isinstance(self.enabled_only, bool):
            raise InvalidWorkflowSelectionPolicyError("enabled_only must be a bool.")
        if isinstance(self.maximum_candidates_considered, bool) or not isinstance(
            self.maximum_candidates_considered,
            int,
        ):
            raise InvalidWorkflowSelectionPolicyError("maximum_candidates_considered must be an int.")
        if self.maximum_candidates_considered <= 0 or self.maximum_candidates_considered > MAX_WORKFLOW_SELECTION_CANDIDATES:
            raise InvalidWorkflowSelectionPolicyError("maximum_candidates_considered is outside the allowed range.")

        object.__setattr__(
            self,
            "allowed_library_ids",
            _validate_identifier_group(self.allowed_library_ids, "allowed_library_ids", MAX_WORKFLOW_SELECTION_IDS),
        )
        object.__setattr__(
            self,
            "excluded_library_ids",
            _validate_identifier_group(self.excluded_library_ids, "excluded_library_ids", MAX_WORKFLOW_SELECTION_IDS),
        )
        object.__setattr__(
            self,
            "allowed_categories",
            _validate_identifier_group(self.allowed_categories, "allowed_categories", MAX_WORKFLOW_SELECTION_CATEGORIES),
        )
        object.__setattr__(
            self,
            "excluded_categories",
            _validate_identifier_group(self.excluded_categories, "excluded_categories", MAX_WORKFLOW_SELECTION_CATEGORIES),
        )
        object.__setattr__(
            self,
            "required_tags",
            _validate_identifier_group(self.required_tags, "required_tags", MAX_WORKFLOW_SELECTION_TAGS),
        )
        object.__setattr__(
            self,
            "excluded_tags",
            _validate_identifier_group(self.excluded_tags, "excluded_tags", MAX_WORKFLOW_SELECTION_TAGS),
        )
        object.__setattr__(
            self,
            "preferred_library_ids",
            _validate_identifier_group(self.preferred_library_ids, "preferred_library_ids", MAX_WORKFLOW_SELECTION_IDS),
        )
        object.__setattr__(
            self,
            "preferred_categories",
            _validate_identifier_group(self.preferred_categories, "preferred_categories", MAX_WORKFLOW_SELECTION_CATEGORIES),
        )
        object.__setattr__(
            self,
            "preferred_tags",
            _validate_identifier_group(self.preferred_tags, "preferred_tags", MAX_WORKFLOW_SELECTION_TAGS),
        )
        _reject_overlap("library_ids", self.allowed_library_ids, self.excluded_library_ids)
        _reject_overlap("categories", self.allowed_categories, self.excluded_categories)
        _reject_overlap("tags", self.required_tags, self.excluded_tags)


@dataclass(frozen=True, slots=True)
class WorkflowSelectionRequest:
    """Input for selecting a workflow from a capability resolution result."""

    resolution_result: CapabilityResolutionResult
    policy: WorkflowSelectionPolicy = field(default_factory=WorkflowSelectionPolicy)
    preferred_reference: ExecutionPlanReference | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.resolution_result, CapabilityResolutionResult):
            raise InvalidWorkflowSelectionRequestError("resolution_result must be CapabilityResolutionResult.")
        if not isinstance(self.policy, WorkflowSelectionPolicy):
            raise InvalidWorkflowSelectionRequestError("policy must be WorkflowSelectionPolicy.")
        if self.preferred_reference is not None:
            if not isinstance(self.preferred_reference, ExecutionPlanReference):
                raise InvalidWorkflowSelectionRequestError("preferred_reference must be ExecutionPlanReference or None.")
            object.__setattr__(
                self,
                "preferred_reference",
                ExecutionPlanReference(self.preferred_reference.plan_id, self.preferred_reference.version),
            )
        object.__setattr__(self, "metadata", MappingProxyType(_safe_metadata(self.metadata)))


@dataclass(frozen=True, slots=True)
class WorkflowSelectionReason:
    """Structured explanation for selection scoring or status."""

    code: WorkflowSelectionReasonCode
    candidate_id: str | None = None
    value: str | None = None
    score_delta: int = 0

    def __post_init__(self) -> None:
        if isinstance(self.score_delta, bool) or not isinstance(self.score_delta, int):
            raise InvalidWorkflowSelectionRequestError("score_delta must be an int.")


@dataclass(frozen=True, slots=True)
class WorkflowSelectionRejection:
    """Structured rejection for one candidate."""

    candidate_id: str
    reason_code: WorkflowSelectionReasonCode
    message: str


@dataclass(frozen=True, slots=True)
class WorkflowScoredCandidate:
    """Candidate with selector-visible score details."""

    candidate: CapabilityCandidate
    base_score: int
    policy_bonus: int
    final_score: int
    reasons: tuple[WorkflowSelectionReason, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "reasons", tuple(self.reasons))
        if self.base_score + self.policy_bonus != self.final_score:
            raise InvalidWorkflowSelectionRequestError("final_score must equal base_score plus policy_bonus.")
        if sum(reason.score_delta for reason in self.reasons) != self.policy_bonus:
            raise InvalidWorkflowSelectionRequestError("policy_bonus must equal selector reason deltas.")


@dataclass(frozen=True, slots=True)
class WorkflowSelectionResult:
    """Immutable result for a deterministic workflow selection."""

    status: WorkflowSelectionStatus
    selected_candidate: WorkflowScoredCandidate | None
    considered_candidates: tuple[WorkflowScoredCandidate, ...]
    rejected_candidates: tuple[WorkflowSelectionRejection, ...]
    reasons: tuple[WorkflowSelectionReason, ...]
    ambiguous_candidates: tuple[WorkflowScoredCandidate, ...]
    base_score: int | None
    policy_bonus: int
    final_score: int | None
    total_input_candidates: int
    total_workflow_candidates: int
    total_considered: int
    total_rejected: int
    policy_signature: str
    request_signature: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", _validate_status(self.status))
        object.__setattr__(self, "considered_candidates", tuple(self.considered_candidates))
        object.__setattr__(self, "rejected_candidates", tuple(self.rejected_candidates))
        object.__setattr__(self, "reasons", tuple(self.reasons))
        object.__setattr__(self, "ambiguous_candidates", tuple(self.ambiguous_candidates))


class WorkflowSelector:
    """Select one workflow candidate without execution or resolver calls."""

    def select(self, request: WorkflowSelectionRequest) -> WorkflowSelectionResult:
        if not isinstance(request, WorkflowSelectionRequest):
            raise InvalidWorkflowSelectionRequestError("request must be WorkflowSelectionRequest.")

        policy_signature = workflow_selection_policy_signature(request.policy)
        request_signature = workflow_selection_request_signature(request)
        invalid_reasons = _input_incoherence_reasons(request.resolution_result)
        if invalid_reasons:
            return _result(
                status=WorkflowSelectionStatus.INVALID_INPUT,
                request=request,
                policy_signature=policy_signature,
                request_signature=request_signature,
                reasons=invalid_reasons,
            )

        rejected: list[WorkflowSelectionRejection] = []
        scored: list[WorkflowScoredCandidate] = []
        workflow_candidates = 0
        accepted_seen: dict[tuple[object, ...], WorkflowScoredCandidate] = {}

        for candidate in request.resolution_result.candidates:
            candidate_id = _safe_candidate_id(candidate)
            if candidate.capability.capability_type is not CapabilityType.WORKFLOW:
                rejected.append(
                    WorkflowSelectionRejection(
                        candidate_id,
                        WorkflowSelectionReasonCode.NON_WORKFLOW_REJECTED,
                        "Candidate is not a workflow capability.",
                    )
                )
                continue

            workflow_candidates += 1
            source = candidate.capability.source_reference
            if not isinstance(source, WorkflowCapabilitySource):
                return _result(
                    status=WorkflowSelectionStatus.INVALID_INPUT,
                    request=request,
                    policy_signature=policy_signature,
                    request_signature=request_signature,
                    reasons=(
                        WorkflowSelectionReason(
                            WorkflowSelectionReasonCode.INPUT_INCOHERENT,
                            candidate_id,
                            "workflow source",
                        ),
                    ),
                )

            rejection = _policy_rejection(candidate, request.policy)
            if rejection is not None:
                rejected.append(rejection)
                continue

            selected = _scored_candidate(candidate, request)
            identity = _workflow_identity(candidate.capability)
            previous = accepted_seen.get(identity)
            if previous is not None:
                if _candidate_payload(previous.candidate) != _candidate_payload(selected.candidate):
                    raise ConflictingWorkflowSelectionCandidateError(
                        "duplicate workflow candidate identity has conflicting definitions."
                    )
                rejected.append(
                    WorkflowSelectionRejection(
                        candidate_id,
                        WorkflowSelectionReasonCode.DUPLICATE_IDENTICAL,
                        "Duplicate workflow candidate identity was deduplicated.",
                    )
                )
                continue
            accepted_seen[identity] = selected
            scored.append(selected)

        if not scored:
            status = WorkflowSelectionStatus.NO_CANDIDATES
            reasons = (WorkflowSelectionReason(WorkflowSelectionReasonCode.INPUT_INCOHERENT, value="no candidates"),)
            if workflow_candidates and all(item.reason_code is WorkflowSelectionReasonCode.DISABLED_REJECTED for item in rejected):
                reasons = (WorkflowSelectionReason(WorkflowSelectionReasonCode.DISABLED_REJECTED),)
            return _result(
                status=status,
                request=request,
                policy_signature=policy_signature,
                request_signature=request_signature,
                rejected=tuple(rejected),
                total_workflow_candidates=workflow_candidates,
                reasons=reasons,
            )

        ordered = tuple(sorted(scored, key=_scored_sort_key))
        considered = ordered[: request.policy.maximum_candidates_considered]
        top = considered[0]
        if top.final_score < request.policy.minimum_score:
            return _result(
                status=WorkflowSelectionStatus.BELOW_MINIMUM_SCORE,
                request=request,
                policy_signature=policy_signature,
                request_signature=request_signature,
                considered=considered,
                rejected=tuple(rejected),
                total_workflow_candidates=workflow_candidates,
                reasons=(
                    WorkflowSelectionReason(
                        WorkflowSelectionReasonCode.BELOW_MINIMUM_SCORE,
                        _safe_candidate_id(top.candidate),
                        str(request.policy.minimum_score),
                    ),
                ),
            )

        top_candidates = tuple(item for item in considered if item.final_score == top.final_score)
        if request.policy.require_unique_top_score and len(top_candidates) > 1:
            return _result(
                status=WorkflowSelectionStatus.AMBIGUOUS,
                request=request,
                policy_signature=policy_signature,
                request_signature=request_signature,
                considered=considered,
                rejected=tuple(rejected),
                ambiguous=top_candidates,
                total_workflow_candidates=workflow_candidates,
                reasons=(
                    WorkflowSelectionReason(
                        WorkflowSelectionReasonCode.TOP_SCORE_TIE,
                        value=str(top.final_score),
                    ),
                ),
            )

        return _result(
            status=WorkflowSelectionStatus.SELECTED,
            request=request,
            policy_signature=policy_signature,
            request_signature=request_signature,
            selected=top,
            considered=considered,
            rejected=tuple(rejected),
            total_workflow_candidates=workflow_candidates,
            reasons=(
                WorkflowSelectionReason(
                    WorkflowSelectionReasonCode.UNIQUE_TOP_SELECTED,
                    _safe_candidate_id(top.candidate),
                    str(top.final_score),
                ),
            ),
        )


def workflow_selection_policy_signature(policy: WorkflowSelectionPolicy) -> str:
    """Return a deterministic signature for a selection policy."""
    if not isinstance(policy, WorkflowSelectionPolicy):
        raise InvalidWorkflowSelectionPolicyError("policy must be WorkflowSelectionPolicy.")
    payload = {
        "minimum_score": policy.minimum_score,
        "require_unique_top_score": policy.require_unique_top_score,
        "enabled_only": policy.enabled_only,
        "allowed_library_ids": sorted(policy.allowed_library_ids),
        "excluded_library_ids": sorted(policy.excluded_library_ids),
        "allowed_categories": sorted(policy.allowed_categories),
        "excluded_categories": sorted(policy.excluded_categories),
        "required_tags": sorted(policy.required_tags),
        "excluded_tags": sorted(policy.excluded_tags),
        "preferred_library_ids": sorted(policy.preferred_library_ids),
        "preferred_categories": sorted(policy.preferred_categories),
        "preferred_tags": sorted(policy.preferred_tags),
        "maximum_candidates_considered": policy.maximum_candidates_considered,
    }
    return _signature(payload)


def workflow_selection_request_signature(request: WorkflowSelectionRequest) -> str:
    """Return a deterministic signature for a workflow selection request."""
    if not isinstance(request, WorkflowSelectionRequest):
        raise InvalidWorkflowSelectionRequestError("request must be WorkflowSelectionRequest.")
    preferred = None
    if request.preferred_reference is not None:
        preferred = {
            "plan_id": request.preferred_reference.plan_id,
            "version": request.preferred_reference.version,
        }
    payload = {
        "resolution_request_signature": capability_resolution_request_signature(request.resolution_result.request),
        "candidate_identities": sorted(
            _candidate_signature_payload(candidate)
            for candidate in request.resolution_result.candidates
        ),
        "policy_signature": workflow_selection_policy_signature(request.policy),
        "preferred_reference": preferred,
        "metadata": _jsonable_mapping(request.metadata),
    }
    return _signature(payload)


def _policy_rejection(
    candidate: CapabilityCandidate,
    policy: WorkflowSelectionPolicy,
) -> WorkflowSelectionRejection | None:
    capability = candidate.capability
    source = capability.source_reference
    if not isinstance(source, WorkflowCapabilitySource):
        return WorkflowSelectionRejection(
            _safe_candidate_id(candidate),
            WorkflowSelectionReasonCode.INPUT_INCOHERENT,
            "Workflow candidate has invalid source reference.",
        )
    library_id = source.library.library_id
    if policy.enabled_only and not capability.enabled:
        return WorkflowSelectionRejection(
            _safe_candidate_id(candidate),
            WorkflowSelectionReasonCode.DISABLED_REJECTED,
            "Workflow capability is disabled.",
        )
    if policy.allowed_library_ids and library_id not in policy.allowed_library_ids:
        return WorkflowSelectionRejection(
            _safe_candidate_id(candidate),
            WorkflowSelectionReasonCode.LIBRARY_NOT_ALLOWED,
            "Workflow library is not allowed by policy.",
        )
    if library_id in policy.excluded_library_ids:
        return WorkflowSelectionRejection(
            _safe_candidate_id(candidate),
            WorkflowSelectionReasonCode.LIBRARY_EXCLUDED,
            "Workflow library is excluded by policy.",
        )
    if policy.allowed_categories and not any(category in policy.allowed_categories for category in capability.categories):
        return WorkflowSelectionRejection(
            _safe_candidate_id(candidate),
            WorkflowSelectionReasonCode.CATEGORY_NOT_ALLOWED,
            "Workflow categories are not allowed by policy.",
        )
    if any(category in policy.excluded_categories for category in capability.categories):
        return WorkflowSelectionRejection(
            _safe_candidate_id(candidate),
            WorkflowSelectionReasonCode.CATEGORY_EXCLUDED,
            "Workflow category is excluded by policy.",
        )
    if any(tag not in capability.tags for tag in policy.required_tags):
        return WorkflowSelectionRejection(
            _safe_candidate_id(candidate),
            WorkflowSelectionReasonCode.REQUIRED_TAG_MISSING,
            "Workflow is missing a required tag.",
        )
    if any(tag in capability.tags for tag in policy.excluded_tags):
        return WorkflowSelectionRejection(
            _safe_candidate_id(candidate),
            WorkflowSelectionReasonCode.EXCLUDED_TAG_PRESENT,
            "Workflow has an excluded tag.",
        )
    return None


def _scored_candidate(
    candidate: CapabilityCandidate,
    request: WorkflowSelectionRequest,
) -> WorkflowScoredCandidate:
    reasons = [WorkflowSelectionReason(WorkflowSelectionReasonCode.WORKFLOW_TYPE_ACCEPTED, _safe_candidate_id(candidate))]
    source = candidate.capability.source_reference
    if not isinstance(source, WorkflowCapabilitySource):
        raise InvalidWorkflowSelectionRequestError("workflow candidate requires WorkflowCapabilitySource.")
    if request.preferred_reference is not None and source.reference == request.preferred_reference:
        reasons.append(
            WorkflowSelectionReason(
                WorkflowSelectionReasonCode.PREFERRED_REFERENCE_MATCH,
                _safe_candidate_id(candidate),
                source.reference.plan_id,
                PREFERRED_REFERENCE_BONUS,
            )
        )
    if source.library.library_id in request.policy.preferred_library_ids:
        reasons.append(
            WorkflowSelectionReason(
                WorkflowSelectionReasonCode.PREFERRED_LIBRARY_MATCH,
                _safe_candidate_id(candidate),
                source.library.library_id,
                PREFERRED_LIBRARY_BONUS,
            )
        )
    for category in request.policy.preferred_categories:
        if category in candidate.capability.categories:
            reasons.append(
                WorkflowSelectionReason(
                    WorkflowSelectionReasonCode.PREFERRED_CATEGORY_MATCH,
                    _safe_candidate_id(candidate),
                    category,
                    PREFERRED_CATEGORY_BONUS,
                )
            )
    for tag in request.policy.preferred_tags:
        if tag in candidate.capability.tags:
            reasons.append(
                WorkflowSelectionReason(
                    WorkflowSelectionReasonCode.PREFERRED_TAG_MATCH,
                    _safe_candidate_id(candidate),
                    tag,
                    PREFERRED_TAG_BONUS,
                )
            )
    policy_bonus = sum(reason.score_delta for reason in reasons)
    return WorkflowScoredCandidate(
        candidate=candidate,
        base_score=candidate.score,
        policy_bonus=policy_bonus,
        final_score=candidate.score + policy_bonus,
        reasons=tuple(reasons),
    )


def _input_incoherence_reasons(
    result: CapabilityResolutionResult,
) -> tuple[WorkflowSelectionReason, ...]:
    if not isinstance(result, CapabilityResolutionResult):
        return (WorkflowSelectionReason(WorkflowSelectionReasonCode.INPUT_INCOHERENT, value="result type"),)
    if result.matched_capabilities != len(result.candidates):
        return (WorkflowSelectionReason(WorkflowSelectionReasonCode.INPUT_INCOHERENT, value="matched count"),)
    if result.top_score is not None and not result.candidates:
        return (WorkflowSelectionReason(WorkflowSelectionReasonCode.INPUT_INCOHERENT, value="top score without candidates"),)
    if result.candidates:
        actual_top = max(candidate.score for candidate in result.candidates)
        if result.top_score != actual_top:
            return (WorkflowSelectionReason(WorkflowSelectionReasonCode.INPUT_INCOHERENT, value="top score mismatch"),)
        actual_ambiguous = sum(1 for candidate in result.candidates if candidate.score == actual_top) >= 2
        if result.ambiguous != actual_ambiguous:
            return (WorkflowSelectionReason(WorkflowSelectionReasonCode.INPUT_INCOHERENT, value="ambiguous mismatch"),)
    seen: dict[tuple[object, ...], CapabilityCandidate] = {}
    for candidate in result.candidates:
        score_error = _score_error(candidate.score)
        if score_error is not None:
            return (WorkflowSelectionReason(WorkflowSelectionReasonCode.INPUT_INCOHERENT, _safe_candidate_id(candidate), score_error),)
        if candidate.capability.capability_type is CapabilityType.WORKFLOW:
            try:
                identity = _workflow_identity(candidate.capability)
            except WorkflowSelectorError:
                return (
                    WorkflowSelectionReason(
                        WorkflowSelectionReasonCode.INPUT_INCOHERENT,
                        _safe_candidate_id(candidate),
                        "workflow identity",
                    ),
                )
            previous = seen.get(identity)
            if previous is not None and _candidate_payload(previous) != _candidate_payload(candidate):
                raise ConflictingWorkflowSelectionCandidateError(
                    "duplicate workflow candidate identity has conflicting definitions."
                )
            seen[identity] = candidate
    return ()


def _score_error(score: object) -> str | None:
    if isinstance(score, bool):
        return "bool score"
    if not isinstance(score, (int, float)):
        return "non numeric score"
    if isinstance(score, float) and not math.isfinite(score):
        return "non finite score"
    return None


def _result(
    *,
    status: WorkflowSelectionStatus,
    request: WorkflowSelectionRequest,
    policy_signature: str,
    request_signature: str,
    selected: WorkflowScoredCandidate | None = None,
    considered: tuple[WorkflowScoredCandidate, ...] = (),
    rejected: tuple[WorkflowSelectionRejection, ...] = (),
    ambiguous: tuple[WorkflowScoredCandidate, ...] = (),
    total_workflow_candidates: int | None = None,
    reasons: tuple[WorkflowSelectionReason, ...] = (),
) -> WorkflowSelectionResult:
    score_source = selected or (considered[0] if considered else None)
    all_reasons = tuple(reasons) + tuple(reason for item in considered for reason in item.reasons)
    return WorkflowSelectionResult(
        status=status,
        selected_candidate=selected,
        considered_candidates=considered,
        rejected_candidates=rejected,
        reasons=all_reasons,
        ambiguous_candidates=ambiguous,
        base_score=None if score_source is None else score_source.base_score,
        policy_bonus=0 if score_source is None else score_source.policy_bonus,
        final_score=None if score_source is None else score_source.final_score,
        total_input_candidates=len(request.resolution_result.candidates),
        total_workflow_candidates=(
            total_workflow_candidates
            if total_workflow_candidates is not None
            else sum(
                1
                for candidate in request.resolution_result.candidates
                if candidate.capability.capability_type is CapabilityType.WORKFLOW
            )
        ),
        total_considered=len(considered),
        total_rejected=len(rejected),
        policy_signature=policy_signature,
        request_signature=request_signature,
    )


def _workflow_identity(capability: CapabilityDefinition) -> tuple[object, ...]:
    source = capability.source_reference
    if not isinstance(source, WorkflowCapabilitySource):
        raise InvalidWorkflowSelectionRequestError("workflow capability requires WorkflowCapabilitySource.")
    return (
        CapabilityType.WORKFLOW.value,
        source.library.library_id,
        source.reference.plan_id,
        source.reference.version,
    )


def _safe_candidate_id(candidate: CapabilityCandidate) -> str:
    capability = candidate.capability
    source = capability.source_reference
    if isinstance(source, WorkflowCapabilitySource):
        version = source.reference.version or "unversioned"
        return f"{source.library.library_id}:{source.reference.plan_id}:{version}"
    if isinstance(source, ToolCapabilitySource):
        return source.tool_name
    return capability.capability_id


def _candidate_payload(candidate: CapabilityCandidate) -> tuple[object, ...]:
    capability = candidate.capability
    return (
        capability.capability_id,
        capability.capability_type.value,
        capability.title,
        capability.description,
        capability.categories,
        capability.tags,
        capability.input_names,
        capability.output_names,
        capability.enabled,
        candidate.score,
        _candidate_signature_payload(candidate),
    )


def _candidate_signature_payload(candidate: CapabilityCandidate) -> tuple[object, ...]:
    capability = candidate.capability
    source = capability.source_reference
    if isinstance(source, WorkflowCapabilitySource):
        identity = (
            "workflow",
            source.library.library_id,
            source.library.library_version,
            source.reference.plan_id,
            source.reference.version,
        )
    elif isinstance(source, ToolCapabilitySource):
        identity = ("tool", source.tool_name)
    else:
        identity = ("unknown", capability.capability_id)
    return (
        capability.capability_id,
        capability.capability_type.value,
        capability.categories,
        capability.tags,
        capability.enabled,
        candidate.score,
        identity,
    )


def _scored_sort_key(item: WorkflowScoredCandidate) -> tuple[int, int, int, int, int, int, str, str, str, str]:
    source = item.candidate.capability.source_reference
    if not isinstance(source, WorkflowCapabilitySource):
        raise InvalidWorkflowSelectionRequestError("workflow candidate requires WorkflowCapabilitySource.")
    reason_codes = {reason.code for reason in item.reasons}
    preferred_tag_count = sum(
        1
        for reason in item.reasons
        if reason.code is WorkflowSelectionReasonCode.PREFERRED_TAG_MATCH
    )
    return (
        -item.final_score,
        0 if item.candidate.capability.enabled else 1,
        0 if WorkflowSelectionReasonCode.PREFERRED_REFERENCE_MATCH in reason_codes else 1,
        0 if WorkflowSelectionReasonCode.PREFERRED_LIBRARY_MATCH in reason_codes else 1,
        0 if WorkflowSelectionReasonCode.PREFERRED_CATEGORY_MATCH in reason_codes else 1,
        -preferred_tag_count,
        source.library.library_id,
        source.reference.plan_id,
        source.reference.version or "",
        _safe_candidate_id(item.candidate),
    )


def _validate_identifier_group(values: Iterable[str], field: str, maximum: int) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Iterable):
        raise InvalidWorkflowSelectionPolicyError(f"{field} must be iterable.")
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        item = _validate_identifier(value, field)
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    if len(result) > maximum:
        raise InvalidWorkflowSelectionPolicyError(f"{field} exceeds the allowed count.")
    return tuple(result)


def _validate_identifier(value: str, field: str) -> str:
    if not isinstance(value, str):
        raise InvalidWorkflowSelectionPolicyError(f"{field} values must be strings.")
    normalized = value.strip().casefold()
    if not normalized:
        raise InvalidWorkflowSelectionPolicyError(f"{field} cannot contain empty values.")
    if len(normalized) > MAX_WORKFLOW_SELECTION_STRING_LENGTH:
        raise InvalidWorkflowSelectionPolicyError(f"{field} value exceeds the length limit.")
    if "/" in normalized or "\\" in normalized or ":" in normalized or ".." in normalized:
        raise InvalidWorkflowSelectionPolicyError(f"{field} cannot contain path-like values.")
    if any(ord(character) < 32 for character in normalized):
        raise InvalidWorkflowSelectionPolicyError(f"{field} cannot contain control characters.")
    if _IDENTIFIER_PATTERN.fullmatch(normalized) is None:
        raise InvalidWorkflowSelectionPolicyError(f"{field} contains unsupported characters.")
    return normalized


def _reject_overlap(field: str, first: tuple[str, ...], second: tuple[str, ...]) -> None:
    overlap = sorted(set(first).intersection(second))
    if overlap:
        raise InvalidWorkflowSelectionPolicyError(f"{field} has contradictory allowed/excluded values.")


def _safe_metadata(values: Mapping[str, object], *, depth: int = 0) -> dict[str, object]:
    if not isinstance(values, Mapping):
        raise InvalidWorkflowSelectionRequestError("metadata must be a mapping.")
    if depth > MAX_WORKFLOW_SELECTION_METADATA_DEPTH:
        raise InvalidWorkflowSelectionRequestError("metadata exceeds maximum depth.")
    if len(values) > MAX_WORKFLOW_SELECTION_METADATA_ITEMS:
        raise InvalidWorkflowSelectionRequestError("metadata exceeds maximum size.")
    result: dict[str, object] = {}
    for key, value in values.items():
        result[_validate_identifier(key, "metadata key")] = _safe_metadata_value(value, depth=depth + 1)
    return result


def _safe_metadata_value(value: object, *, depth: int) -> object:
    if value is None or type(value) in {str, int, bool}:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise InvalidWorkflowSelectionRequestError("metadata cannot contain non-finite floats.")
        return value
    if isinstance(value, tuple):
        return tuple(_safe_metadata_value(item, depth=depth + 1) for item in value)
    if isinstance(value, list):
        return tuple(_safe_metadata_value(item, depth=depth + 1) for item in value)
    if isinstance(value, Mapping):
        return MappingProxyType(_safe_metadata(value, depth=depth))
    if isinstance(value, (type, ModuleType)) or callable(value):
        raise InvalidWorkflowSelectionRequestError("metadata contains unsafe runtime objects.")
    raise InvalidWorkflowSelectionRequestError("metadata contains unsupported values.")


def _jsonable_mapping(values: Mapping[str, object]) -> tuple[tuple[str, object], ...]:
    return tuple(sorted((key, _jsonable_value(value)) for key, value in values.items()))


def _jsonable_value(value: object) -> object:
    if isinstance(value, Mapping):
        return _jsonable_mapping(value)
    if isinstance(value, tuple):
        return tuple(_jsonable_value(item) for item in value)
    return value


def _signature(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_status(status: WorkflowSelectionStatus | str) -> WorkflowSelectionStatus:
    if isinstance(status, WorkflowSelectionStatus):
        return status
    if isinstance(status, str):
        return WorkflowSelectionStatus(status)
    raise InvalidWorkflowSelectionRequestError("status must be WorkflowSelectionStatus.")
