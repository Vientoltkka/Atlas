"""Deterministic discovery for reusable Atlas workflows."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import re
from typing import Iterable

from core.execution_plan_library import ExecutionPlanLibrary, WorkflowDefinition
from core.execution_plan_registry import (
    ExecutionPlanReference,
    ExecutionPlanRegistryError,
    validate_plan_id,
)
from core.execution_plan_validator import plan_signature


MAX_DISCOVERY_CATEGORIES = 32
MAX_DISCOVERY_REQUIRED_TAGS = 32
MAX_DISCOVERY_PREFERRED_TAGS = 32
MAX_DISCOVERY_EXCLUDED_TAGS = 32
MAX_DISCOVERY_REFERENCES = 64
MAX_DISCOVERY_LIBRARY_IDS = 32
MAX_DISCOVERY_TITLE_TERMS = 16
MAX_DISCOVERY_TITLE_TERM_LENGTH = 80
MAX_DISCOVERY_LIMIT = 256

EXACT_REFERENCE_SCORE = 100
CATEGORY_MATCH_SCORE = 30
REQUIRED_TAG_MATCH_SCORE = 20
PREFERRED_TAG_MATCH_SCORE = 10
TITLE_TERM_MATCH_SCORE = 5
ENABLED_BONUS_SCORE = 1

_CATEGORY_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,127}$")
_TAG_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,63}$")


class WorkflowDiscoveryError(RuntimeError):
    """Base error for deterministic workflow discovery."""

    def __init__(
        self,
        message: str,
        *,
        field: str | None = None,
        value: object | None = None,
        reference: ExecutionPlanReference | None = None,
        library_id: str | None = None,
        code: str,
        reason: str | None = None,
    ) -> None:
        super().__init__(message)
        self.field = field
        self.value = value
        self.reference = reference
        self.library_id = library_id
        self.code = code
        self.reason = reason or message


class InvalidWorkflowDiscoveryRequestError(WorkflowDiscoveryError):
    """Raised when a discovery request is malformed."""


class InvalidWorkflowDiscoveryCategoryError(InvalidWorkflowDiscoveryRequestError):
    """Raised when a requested category is invalid."""


class InvalidWorkflowDiscoveryTagError(InvalidWorkflowDiscoveryRequestError):
    """Raised when a requested tag is invalid."""


class InvalidWorkflowDiscoveryReferenceError(InvalidWorkflowDiscoveryRequestError):
    """Raised when a requested reference is invalid."""


class InvalidWorkflowDiscoveryLibraryIdError(InvalidWorkflowDiscoveryRequestError):
    """Raised when a requested library id is invalid."""


class InvalidWorkflowDiscoveryTitleTermError(InvalidWorkflowDiscoveryRequestError):
    """Raised when a requested title term is invalid."""


class WorkflowDiscoveryLimitError(InvalidWorkflowDiscoveryRequestError):
    """Raised when request limits are invalid or exceeded."""


class DuplicateWorkflowCandidateError(WorkflowDiscoveryError):
    """Raised when duplicate workflow candidates cannot be deduplicated safely."""


class ConflictingWorkflowCandidateError(DuplicateWorkflowCandidateError):
    """Raised when duplicate references point to different plan signatures."""


class WorkflowMatchReasonCode(str, Enum):
    EXACT_REFERENCE = "exact_reference"
    CATEGORY_MATCH = "category_match"
    REQUIRED_TAG_MATCH = "required_tag_match"
    PREFERRED_TAG_MATCH = "preferred_tag_match"
    TITLE_TERM_MATCH = "title_term_match"
    ENABLED_BONUS = "enabled_bonus"


class WorkflowDiscoveryRejectionCode(str, Enum):
    LIBRARY_FILTERED = "library_filtered"
    REFERENCE_FILTERED = "reference_filtered"
    CATEGORY_MISMATCH = "category_mismatch"
    MISSING_REQUIRED_TAG = "missing_required_tag"
    EXCLUDED_TAG = "excluded_tag"
    DISABLED = "disabled"
    BELOW_MINIMUM_SCORE = "below_minimum_score"
    DUPLICATE_REFERENCE = "duplicate_reference"


@dataclass(frozen=True, slots=True)
class WorkflowMatchReason:
    """One structured explanation for candidate score."""

    code: WorkflowMatchReasonCode
    value: str | None
    score: int


@dataclass(frozen=True, slots=True)
class WorkflowLibraryReference:
    """Source library identity for a discovered workflow."""

    library_id: str
    library_version: str | None


@dataclass(frozen=True, slots=True)
class WorkflowDiscoveryRequest:
    """Explicit deterministic workflow discovery criteria."""

    categories: tuple[str, ...] = ()
    required_tags: tuple[str, ...] = ()
    preferred_tags: tuple[str, ...] = ()
    excluded_tags: tuple[str, ...] = ()
    references: tuple[ExecutionPlanReference, ...] = ()
    library_ids: tuple[str, ...] = ()
    title_terms: tuple[str, ...] = ()
    enabled_only: bool = True
    minimum_score: int = 0
    limit: int | None = None
    include_rejections: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "categories",
            _validate_identifier_group(
                self.categories,
                field="categories",
                maximum=MAX_DISCOVERY_CATEGORIES,
                pattern=_CATEGORY_PATTERN,
                error_type=InvalidWorkflowDiscoveryCategoryError,
                code="INVALID_WORKFLOW_DISCOVERY_CATEGORY",
            ),
        )
        required = _validate_identifier_group(
            self.required_tags,
            field="required_tags",
            maximum=MAX_DISCOVERY_REQUIRED_TAGS,
            pattern=_TAG_PATTERN,
            error_type=InvalidWorkflowDiscoveryTagError,
            code="INVALID_WORKFLOW_DISCOVERY_TAG",
        )
        preferred = _validate_identifier_group(
            self.preferred_tags,
            field="preferred_tags",
            maximum=MAX_DISCOVERY_PREFERRED_TAGS,
            pattern=_TAG_PATTERN,
            error_type=InvalidWorkflowDiscoveryTagError,
            code="INVALID_WORKFLOW_DISCOVERY_TAG",
        )
        excluded = _validate_identifier_group(
            self.excluded_tags,
            field="excluded_tags",
            maximum=MAX_DISCOVERY_EXCLUDED_TAGS,
            pattern=_TAG_PATTERN,
            error_type=InvalidWorkflowDiscoveryTagError,
            code="INVALID_WORKFLOW_DISCOVERY_TAG",
        )
        _reject_cross_group_tag_duplicates(required, preferred, excluded)
        object.__setattr__(self, "required_tags", required)
        object.__setattr__(self, "preferred_tags", preferred)
        object.__setattr__(self, "excluded_tags", excluded)
        object.__setattr__(self, "references", _validate_references(self.references))
        object.__setattr__(self, "library_ids", _validate_library_ids(self.library_ids))
        object.__setattr__(self, "title_terms", _validate_title_terms(self.title_terms))
        if not isinstance(self.enabled_only, bool):
            raise InvalidWorkflowDiscoveryRequestError(
                "enabled_only must be a bool.",
                field="enabled_only",
                value=self.enabled_only,
                code="INVALID_WORKFLOW_DISCOVERY_REQUEST",
            )
        if not isinstance(self.include_rejections, bool):
            raise InvalidWorkflowDiscoveryRequestError(
                "include_rejections must be a bool.",
                field="include_rejections",
                value=self.include_rejections,
                code="INVALID_WORKFLOW_DISCOVERY_REQUEST",
            )
        if isinstance(self.minimum_score, bool) or not isinstance(self.minimum_score, int):
            raise WorkflowDiscoveryLimitError(
                "minimum_score must be a non-negative integer.",
                field="minimum_score",
                value=self.minimum_score,
                code="INVALID_WORKFLOW_DISCOVERY_MINIMUM_SCORE",
            )
        if self.minimum_score < 0:
            raise WorkflowDiscoveryLimitError(
                "minimum_score cannot be negative.",
                field="minimum_score",
                value=self.minimum_score,
                code="INVALID_WORKFLOW_DISCOVERY_MINIMUM_SCORE",
            )
        if self.limit is not None:
            if isinstance(self.limit, bool) or not isinstance(self.limit, int):
                raise WorkflowDiscoveryLimitError(
                    "limit must be a positive integer or null.",
                    field="limit",
                    value=self.limit,
                    code="INVALID_WORKFLOW_DISCOVERY_LIMIT",
                )
            if self.limit <= 0 or self.limit > MAX_DISCOVERY_LIMIT:
                raise WorkflowDiscoveryLimitError(
                    "limit is outside the allowed range.",
                    field="limit",
                    value=self.limit,
                    code="INVALID_WORKFLOW_DISCOVERY_LIMIT",
                )


@dataclass(frozen=True, slots=True)
class WorkflowDiscoveryCandidate:
    """One scored reusable workflow candidate."""

    library_id: str
    library_version: str | None
    workflow: WorkflowDefinition
    score: int
    reasons: tuple[WorkflowMatchReason, ...]
    source_libraries: tuple[WorkflowLibraryReference, ...] = ()

    def __post_init__(self) -> None:
        reasons = tuple(self.reasons)
        sources = tuple(self.source_libraries) or (
            WorkflowLibraryReference(self.library_id, self.library_version),
        )
        object.__setattr__(self, "reasons", reasons)
        object.__setattr__(self, "source_libraries", sources)
        reason_score = sum(reason.score for reason in reasons)
        if reason_score != self.score:
            raise InvalidWorkflowDiscoveryRequestError(
                "Candidate score must equal the sum of match reasons.",
                field="score",
                value=self.score,
                reference=self.workflow.reference,
                library_id=self.library_id,
                code="INVALID_WORKFLOW_DISCOVERY_CANDIDATE",
            )


@dataclass(frozen=True, slots=True)
class WorkflowDiscoveryRejection:
    """Structured reason why a workflow did not become a candidate."""

    library_id: str
    library_version: str | None
    reference: ExecutionPlanReference
    reasons: tuple[WorkflowDiscoveryRejectionCode, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "reasons", tuple(self.reasons))


@dataclass(frozen=True, slots=True)
class WorkflowDiscoveryResult:
    """Immutable result for deterministic workflow discovery."""

    request: WorkflowDiscoveryRequest
    candidates: tuple[WorkflowDiscoveryCandidate, ...]
    rejections: tuple[WorkflowDiscoveryRejection, ...]
    scanned_libraries: int
    scanned_workflows: int
    matched_workflows: int
    rejected_workflows: int
    truncated: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "candidates", tuple(self.candidates))
        object.__setattr__(self, "rejections", tuple(self.rejections))

    @property
    def has_matches(self) -> bool:
        return bool(self.candidates)

    @property
    def best_score(self) -> int | None:
        return self.candidates[0].score if self.candidates else None

    @property
    def top_candidates(self) -> tuple[WorkflowDiscoveryCandidate, ...]:
        if not self.candidates:
            return ()
        best = self.candidates[0].score
        return tuple(candidate for candidate in self.candidates if candidate.score == best)

    @property
    def is_ambiguous(self) -> bool:
        top = self.top_candidates
        if len(top) < 2:
            return False
        return len({candidate.workflow.reference for candidate in top}) > 1


class WorkflowDiscoveryService:
    """Pure deterministic workflow discovery over execution-plan libraries."""

    def discover(
        self,
        request: WorkflowDiscoveryRequest,
        libraries: Iterable[ExecutionPlanLibrary],
    ) -> WorkflowDiscoveryResult:
        if not isinstance(request, WorkflowDiscoveryRequest):
            raise InvalidWorkflowDiscoveryRequestError(
                "request must be WorkflowDiscoveryRequest.",
                field="request",
                value=type(request).__name__,
                code="INVALID_WORKFLOW_DISCOVERY_REQUEST",
            )
        library_tuple = tuple(libraries)
        for library in library_tuple:
            if not isinstance(library, ExecutionPlanLibrary):
                raise InvalidWorkflowDiscoveryRequestError(
                    "libraries must contain ExecutionPlanLibrary instances.",
                    field="libraries",
                    value=type(library).__name__,
                    code="INVALID_WORKFLOW_DISCOVERY_LIBRARY",
                )

        scanned_libraries = 0
        scanned_workflows = 0
        rejected_workflows = 0
        rejections: list[WorkflowDiscoveryRejection] = []
        raw_candidates: list[WorkflowDiscoveryCandidate] = []

        for library in sorted(library_tuple, key=_library_sort_key):
            library_filtered = bool(request.library_ids) and library.library_id not in request.library_ids
            if library_filtered:
                if request.include_rejections:
                    for workflow in library.workflows():
                        rejections.append(
                            WorkflowDiscoveryRejection(
                                library_id=library.library_id,
                                library_version=library.version,
                                reference=workflow.reference,
                                reasons=(WorkflowDiscoveryRejectionCode.LIBRARY_FILTERED,),
                            )
                        )
                rejected_workflows += len(library.workflows())
                continue

            scanned_libraries += 1
            for workflow in library.workflows():
                scanned_workflows += 1
                rejection_codes = _rejection_codes(request, workflow)
                if rejection_codes:
                    rejected_workflows += 1
                    if request.include_rejections:
                        rejections.append(
                            WorkflowDiscoveryRejection(
                                library_id=library.library_id,
                                library_version=library.version,
                                reference=workflow.reference,
                                reasons=rejection_codes,
                            )
                        )
                    continue

                candidate = _candidate_for(request, library, workflow)
                if candidate.score < request.minimum_score:
                    rejected_workflows += 1
                    if request.include_rejections:
                        rejections.append(
                            WorkflowDiscoveryRejection(
                                library_id=library.library_id,
                                library_version=library.version,
                                reference=workflow.reference,
                                reasons=(WorkflowDiscoveryRejectionCode.BELOW_MINIMUM_SCORE,),
                            )
                        )
                    continue
                raw_candidates.append(candidate)

        candidates = _deduplicate_candidates(raw_candidates, include_rejections=request.include_rejections)
        sorted_candidates = tuple(sorted(candidates, key=_candidate_sort_key))
        truncated = request.limit is not None and len(sorted_candidates) > request.limit
        if request.limit is not None:
            sorted_candidates = sorted_candidates[: request.limit]

        return WorkflowDiscoveryResult(
            request=request,
            candidates=sorted_candidates,
            rejections=tuple(sorted(rejections, key=_rejection_sort_key)) if request.include_rejections else (),
            scanned_libraries=scanned_libraries,
            scanned_workflows=scanned_workflows,
            matched_workflows=len(candidates),
            rejected_workflows=rejected_workflows,
            truncated=truncated,
        )


def workflow_discovery_request_signature(request: WorkflowDiscoveryRequest) -> str:
    """Return a deterministic signature for discovery filters and preferences."""
    if not isinstance(request, WorkflowDiscoveryRequest):
        raise InvalidWorkflowDiscoveryRequestError(
            "request must be WorkflowDiscoveryRequest.",
            field="request",
            value=type(request).__name__,
            code="INVALID_WORKFLOW_DISCOVERY_REQUEST",
        )
    payload = {
        "categories": list(request.categories),
        "required_tags": list(request.required_tags),
        "preferred_tags": list(request.preferred_tags),
        "excluded_tags": list(request.excluded_tags),
        "references": [
            {"plan_id": reference.plan_id, "version": reference.version}
            for reference in request.references
        ],
        "library_ids": list(request.library_ids),
        "title_terms": list(request.title_terms),
        "enabled_only": request.enabled_only,
        "minimum_score": request.minimum_score,
        "limit": request.limit,
        "include_rejections": request.include_rejections,
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _candidate_for(
    request: WorkflowDiscoveryRequest,
    library: ExecutionPlanLibrary,
    workflow: WorkflowDefinition,
) -> WorkflowDiscoveryCandidate:
    reasons: list[WorkflowMatchReason] = []
    if workflow.reference in request.references:
        reasons.append(
            WorkflowMatchReason(
                WorkflowMatchReasonCode.EXACT_REFERENCE,
                workflow.reference.plan_id,
                EXACT_REFERENCE_SCORE,
            )
        )
    if workflow.category in request.categories:
        reasons.append(
            WorkflowMatchReason(
                WorkflowMatchReasonCode.CATEGORY_MATCH,
                workflow.category,
                CATEGORY_MATCH_SCORE,
            )
        )
    for tag in request.required_tags:
        if tag in workflow.tags:
            reasons.append(
                WorkflowMatchReason(
                    WorkflowMatchReasonCode.REQUIRED_TAG_MATCH,
                    tag,
                    REQUIRED_TAG_MATCH_SCORE,
                )
            )
    for tag in request.preferred_tags:
        if tag in workflow.tags:
            reasons.append(
                WorkflowMatchReason(
                    WorkflowMatchReasonCode.PREFERRED_TAG_MATCH,
                    tag,
                    PREFERRED_TAG_MATCH_SCORE,
                )
            )
    title = _normalize_title_text(workflow.title)
    for term in request.title_terms:
        if term in title:
            reasons.append(
                WorkflowMatchReason(
                    WorkflowMatchReasonCode.TITLE_TERM_MATCH,
                    term,
                    TITLE_TERM_MATCH_SCORE,
                )
            )
    if workflow.enabled:
        reasons.append(
            WorkflowMatchReason(
                WorkflowMatchReasonCode.ENABLED_BONUS,
                None,
                ENABLED_BONUS_SCORE,
            )
        )
    return WorkflowDiscoveryCandidate(
        library_id=library.library_id,
        library_version=library.version,
        workflow=workflow,
        score=sum(reason.score for reason in reasons),
        reasons=tuple(reasons),
        source_libraries=(WorkflowLibraryReference(library.library_id, library.version),),
    )


def _rejection_codes(
    request: WorkflowDiscoveryRequest,
    workflow: WorkflowDefinition,
) -> tuple[WorkflowDiscoveryRejectionCode, ...]:
    reasons: list[WorkflowDiscoveryRejectionCode] = []
    if request.references and workflow.reference not in request.references:
        reasons.append(WorkflowDiscoveryRejectionCode.REFERENCE_FILTERED)
    if request.categories and workflow.category not in request.categories:
        reasons.append(WorkflowDiscoveryRejectionCode.CATEGORY_MISMATCH)
    if any(tag not in workflow.tags for tag in request.required_tags):
        reasons.append(WorkflowDiscoveryRejectionCode.MISSING_REQUIRED_TAG)
    if any(tag in workflow.tags for tag in request.excluded_tags):
        reasons.append(WorkflowDiscoveryRejectionCode.EXCLUDED_TAG)
    if request.enabled_only and not workflow.enabled:
        reasons.append(WorkflowDiscoveryRejectionCode.DISABLED)
    return tuple(reasons)


def _deduplicate_candidates(
    candidates: list[WorkflowDiscoveryCandidate],
    *,
    include_rejections: bool,
) -> tuple[WorkflowDiscoveryCandidate, ...]:
    del include_rejections
    grouped: dict[ExecutionPlanReference, list[WorkflowDiscoveryCandidate]] = {}
    for candidate in candidates:
        grouped.setdefault(candidate.workflow.reference, []).append(candidate)

    deduplicated: list[WorkflowDiscoveryCandidate] = []
    for reference, group in grouped.items():
        signatures = {plan_signature(candidate.workflow.plan) for candidate in group}
        if len(signatures) > 1:
            raise ConflictingWorkflowCandidateError(
                "Duplicate workflow reference has conflicting plan signatures.",
                reference=reference,
                code="CONFLICTING_WORKFLOW_CANDIDATE",
            )
        winner = sorted(group, key=_candidate_representation_sort_key)[0]
        sources = tuple(
            sorted(
                {
                    source
                    for candidate in group
                    for source in candidate.source_libraries
                },
                key=lambda source: (source.library_id, _version_sort_value(source.library_version)),
            )
        )
        if len(group) > 1:
            winner = WorkflowDiscoveryCandidate(
                library_id=winner.library_id,
                library_version=winner.library_version,
                workflow=winner.workflow,
                score=winner.score,
                reasons=winner.reasons,
                source_libraries=sources,
            )
        deduplicated.append(winner)
    return tuple(deduplicated)


def _validate_identifier_group(
    values: Iterable[str],
    *,
    field: str,
    maximum: int,
    pattern: re.Pattern[str],
    error_type: type[WorkflowDiscoveryError],
    code: str,
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Iterable):
        raise error_type(
            f"{field} must be an iterable of strings.",
            field=field,
            value=type(values).__name__,
            code=code,
        )
    normalized: list[str] = []
    seen: set[str] = set()
    for value in values:
        item = _validate_identifier(value, field=field, pattern=pattern, error_type=error_type, code=code)
        if item in seen:
            raise error_type(
                f"{field} cannot contain duplicates.",
                field=field,
                value=item,
                code=code,
            )
        seen.add(item)
        normalized.append(item)
    if len(normalized) > maximum:
        raise WorkflowDiscoveryLimitError(
            f"{field} exceeds the allowed count.",
            field=field,
            value=len(normalized),
            code="WORKFLOW_DISCOVERY_LIMIT_EXCEEDED",
        )
    return tuple(normalized)


def _validate_identifier(
    value: str,
    *,
    field: str,
    pattern: re.Pattern[str],
    error_type: type[WorkflowDiscoveryError],
    code: str,
) -> str:
    if not isinstance(value, str):
        raise error_type(
            f"{field} values must be strings.",
            field=field,
            value=type(value).__name__,
            code=code,
        )
    normalized = value.strip().lower()
    if not normalized:
        raise error_type(
            f"{field} values cannot be empty.",
            field=field,
            value=value,
            code=code,
        )
    if "/" in normalized or "\\" in normalized or ":" in normalized or ".." in normalized:
        raise error_type(
            f"{field} values cannot be paths.",
            field=field,
            value=normalized,
            code=code,
        )
    if any(ord(character) < 32 for character in normalized):
        raise error_type(
            f"{field} values cannot contain control characters.",
            field=field,
            value=normalized,
            code=code,
        )
    if pattern.fullmatch(normalized) is None:
        raise error_type(
            f"{field} values have unsupported characters.",
            field=field,
            value=normalized,
            code=code,
        )
    return normalized


def _reject_cross_group_tag_duplicates(
    required: tuple[str, ...],
    preferred: tuple[str, ...],
    excluded: tuple[str, ...],
) -> None:
    groups = (
        ("required_tags", required),
        ("preferred_tags", preferred),
        ("excluded_tags", excluded),
    )
    seen: dict[str, str] = {}
    for field, tags in groups:
        for tag in tags:
            previous = seen.get(tag)
            if previous is not None:
                raise InvalidWorkflowDiscoveryTagError(
                    "Discovery tags cannot appear in multiple tag groups.",
                    field=field,
                    value=tag,
                    code="DUPLICATE_WORKFLOW_DISCOVERY_TAG",
                    reason=f"already present in {previous}",
                )
            seen[tag] = field


def _validate_references(
    references: Iterable[ExecutionPlanReference],
) -> tuple[ExecutionPlanReference, ...]:
    if isinstance(references, (str, bytes)) or not isinstance(references, Iterable):
        raise InvalidWorkflowDiscoveryReferenceError(
            "references must be an iterable of ExecutionPlanReference.",
            field="references",
            value=type(references).__name__,
            code="INVALID_WORKFLOW_DISCOVERY_REFERENCE",
        )
    result: list[ExecutionPlanReference] = []
    seen: set[ExecutionPlanReference] = set()
    for reference in references:
        if not isinstance(reference, ExecutionPlanReference):
            raise InvalidWorkflowDiscoveryReferenceError(
                "references must contain ExecutionPlanReference instances.",
                field="references",
                value=type(reference).__name__,
                code="INVALID_WORKFLOW_DISCOVERY_REFERENCE",
            )
        normalized = ExecutionPlanReference(reference.plan_id, reference.version)
        if normalized in seen:
            raise InvalidWorkflowDiscoveryReferenceError(
                "references cannot contain duplicates.",
                field="references",
                reference=normalized,
                code="DUPLICATE_WORKFLOW_DISCOVERY_REFERENCE",
            )
        seen.add(normalized)
        result.append(normalized)
    if len(result) > MAX_DISCOVERY_REFERENCES:
        raise WorkflowDiscoveryLimitError(
            "references exceed the allowed count.",
            field="references",
            value=len(result),
            code="WORKFLOW_DISCOVERY_LIMIT_EXCEEDED",
        )
    return tuple(result)


def _validate_library_ids(
    library_ids: Iterable[str],
) -> tuple[str, ...]:
    if isinstance(library_ids, (str, bytes)) or not isinstance(library_ids, Iterable):
        raise InvalidWorkflowDiscoveryLibraryIdError(
            "library_ids must be an iterable of strings.",
            field="library_ids",
            value=type(library_ids).__name__,
            code="INVALID_WORKFLOW_DISCOVERY_LIBRARY_ID",
        )
    result: list[str] = []
    seen: set[str] = set()
    for library_id in library_ids:
        try:
            normalized = validate_plan_id(library_id)
        except ExecutionPlanRegistryError as error:
            raise InvalidWorkflowDiscoveryLibraryIdError(
                "library_id is invalid.",
                field="library_ids",
                value=library_id if isinstance(library_id, str) else type(library_id).__name__,
                code="INVALID_WORKFLOW_DISCOVERY_LIBRARY_ID",
                reason=str(error),
            ) from error
        if normalized in seen:
            raise InvalidWorkflowDiscoveryLibraryIdError(
                "library_ids cannot contain duplicates.",
                field="library_ids",
                value=normalized,
                code="DUPLICATE_WORKFLOW_DISCOVERY_LIBRARY_ID",
            )
        seen.add(normalized)
        result.append(normalized)
    if len(result) > MAX_DISCOVERY_LIBRARY_IDS:
        raise WorkflowDiscoveryLimitError(
            "library_ids exceed the allowed count.",
            field="library_ids",
            value=len(result),
            code="WORKFLOW_DISCOVERY_LIMIT_EXCEEDED",
        )
    return tuple(result)


def _validate_title_terms(
    title_terms: Iterable[str],
) -> tuple[str, ...]:
    if isinstance(title_terms, (str, bytes)) or not isinstance(title_terms, Iterable):
        raise InvalidWorkflowDiscoveryTitleTermError(
            "title_terms must be an iterable of strings.",
            field="title_terms",
            value=type(title_terms).__name__,
            code="INVALID_WORKFLOW_DISCOVERY_TITLE_TERM",
        )
    result: list[str] = []
    seen: set[str] = set()
    for term in title_terms:
        if not isinstance(term, str):
            raise InvalidWorkflowDiscoveryTitleTermError(
                "title_terms values must be strings.",
                field="title_terms",
                value=type(term).__name__,
                code="INVALID_WORKFLOW_DISCOVERY_TITLE_TERM",
            )
        normalized = _normalize_title_text(term)
        if not normalized:
            raise InvalidWorkflowDiscoveryTitleTermError(
                "title_terms values cannot be empty.",
                field="title_terms",
                value=term,
                code="INVALID_WORKFLOW_DISCOVERY_TITLE_TERM",
            )
        if len(normalized) > MAX_DISCOVERY_TITLE_TERM_LENGTH:
            raise InvalidWorkflowDiscoveryTitleTermError(
                "title_terms value exceeds the length limit.",
                field="title_terms",
                value=normalized,
                code="INVALID_WORKFLOW_DISCOVERY_TITLE_TERM",
            )
        if any(ord(character) < 32 for character in normalized):
            raise InvalidWorkflowDiscoveryTitleTermError(
                "title_terms values cannot contain control characters.",
                field="title_terms",
                value=normalized,
                code="INVALID_WORKFLOW_DISCOVERY_TITLE_TERM",
            )
        if normalized in seen:
            raise InvalidWorkflowDiscoveryTitleTermError(
                "title_terms cannot contain duplicates.",
                field="title_terms",
                value=normalized,
                code="DUPLICATE_WORKFLOW_DISCOVERY_TITLE_TERM",
            )
        seen.add(normalized)
        result.append(normalized)
    if len(result) > MAX_DISCOVERY_TITLE_TERMS:
        raise WorkflowDiscoveryLimitError(
            "title_terms exceed the allowed count.",
            field="title_terms",
            value=len(result),
            code="WORKFLOW_DISCOVERY_LIMIT_EXCEEDED",
        )
    return tuple(result)


def _normalize_title_text(value: str) -> str:
    return " ".join(value.strip().casefold().split())


def _candidate_sort_key(
    candidate: WorkflowDiscoveryCandidate,
) -> tuple[int, int, str, str, tuple[int, str], str]:
    return (
        -candidate.score,
        0 if candidate.workflow.enabled else 1,
        candidate.workflow.category,
        candidate.workflow.reference.plan_id,
        _version_sort_value(candidate.workflow.reference.version),
        candidate.library_id,
    )


def _candidate_representation_sort_key(
    candidate: WorkflowDiscoveryCandidate,
) -> tuple[str, tuple[int, str], str, str, tuple[str, ...], int]:
    return (
        candidate.library_id,
        _version_sort_value(candidate.library_version),
        candidate.workflow.title,
        candidate.workflow.category,
        candidate.workflow.tags,
        0 if candidate.workflow.enabled else 1,
    )


def _library_sort_key(
    library: ExecutionPlanLibrary,
) -> tuple[str, tuple[int, str]]:
    return (library.library_id, _version_sort_value(library.version))


def _rejection_sort_key(
    rejection: WorkflowDiscoveryRejection,
) -> tuple[str, tuple[int, str], str, tuple[int, str], tuple[str, ...]]:
    return (
        rejection.library_id,
        _version_sort_value(rejection.library_version),
        rejection.reference.plan_id,
        _version_sort_value(rejection.reference.version),
        tuple(reason.value for reason in rejection.reasons),
    )


def _version_sort_value(version: str | None) -> tuple[int, str]:
    return (0, "") if version is None else (1, version)
