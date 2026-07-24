"""Capability-based planning without workflow execution."""

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
    CapabilityResolutionRequest,
    CapabilityResolutionResult,
    CapabilityResolver,
    CapabilityResolverError,
    CapabilityType,
    WorkflowCapabilitySource,
)
from core.execution_plan_library import ExecutionPlanLibrary, ExecutionPlanLibraryError, WorkflowDefinition
from core.execution_plan_registry import ExecutionPlanReference, ExecutionPlanRegistry, ExecutionPlanRegistryError
from core.execution_plan_validator import plan_signature
from core.planner import ExecutionPlan
from core.workflow_selector import (
    WorkflowSelectionPolicy,
    WorkflowSelectionRequest,
    WorkflowSelectionResult,
    WorkflowSelectionStatus,
    WorkflowSelector,
    WorkflowSelectorError,
)


MAX_CAPABILITY_PLANNING_ITEMS = 64
MAX_CAPABILITY_PLANNING_OBJECTIVE_LENGTH = 500
MAX_CAPABILITY_PLANNING_METADATA_ITEMS = 32
MAX_CAPABILITY_PLANNING_METADATA_DEPTH = 4
MAX_CAPABILITY_PLANNING_CANDIDATES = 256

_IDENTIFIER_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,199}$")


class CapabilityPlanningError(RuntimeError):
    """Base error for capability planning contract violations."""


class InvalidCapabilityPlanningRequestError(CapabilityPlanningError):
    """Raised when a capability planning request is malformed."""


class CapabilityResolutionFailedError(CapabilityPlanningError):
    """Raised when the capability resolver fails."""


class IncompatibleCapabilityError(CapabilityPlanningError):
    """Raised when a selected capability cannot represent a workflow."""


class WorkflowSelectionFailedError(CapabilityPlanningError):
    """Raised when workflow selection fails unexpectedly."""


class WorkflowNotResolvableError(CapabilityPlanningError):
    """Raised when a selected workflow cannot be resolved to a plan."""


class InconsistentWorkflowReferenceError(CapabilityPlanningError):
    """Raised when selected workflow identity and recovered plan source disagree."""


class InconsistentWorkflowPlanSignatureError(CapabilityPlanningError):
    """Raised when a selected workflow plan signature is inconsistent."""


class CapabilityPlanningStatus(str, Enum):
    """Stable capability-planning decision statuses."""

    SELECTED = "selected"
    NO_CAPABILITY_CANDIDATES = "no_capability_candidates"
    CAPABILITY_AMBIGUOUS = "capability_ambiguous"
    NO_WORKFLOW_CANDIDATES = "no_workflow_candidates"
    WORKFLOW_BELOW_MINIMUM_SCORE = "workflow_below_minimum_score"
    WORKFLOW_AMBIGUOUS = "workflow_ambiguous"
    INVALID_REQUEST = "invalid_request"
    RESOLUTION_FAILED = "resolution_failed"
    SELECTION_FAILED = "selection_failed"
    INCOMPATIBLE_CAPABILITY = "incompatible_capability"
    WORKFLOW_NOT_RESOLVABLE = "workflow_not_resolvable"
    INCONSISTENT_WORKFLOW_REFERENCE = "inconsistent_workflow_reference"
    INCONSISTENT_PLAN_SIGNATURE = "inconsistent_plan_signature"


@dataclass(frozen=True, slots=True)
class CapabilityPlanningPolicy:
    """Policy values that bridge capability resolution and workflow selection."""

    minimum_capability_score: int = 0
    minimum_workflow_score: int = 0
    require_unique_workflow: bool = True
    enabled_only: bool = True
    maximum_candidates: int = MAX_CAPABILITY_PLANNING_CANDIDATES

    def __post_init__(self) -> None:
        if isinstance(self.minimum_capability_score, bool) or not isinstance(self.minimum_capability_score, int):
            raise InvalidCapabilityPlanningRequestError("minimum_capability_score must be a non-negative int.")
        if isinstance(self.minimum_workflow_score, bool) or not isinstance(self.minimum_workflow_score, int):
            raise InvalidCapabilityPlanningRequestError("minimum_workflow_score must be a non-negative int.")
        if self.minimum_capability_score < 0 or self.minimum_workflow_score < 0:
            raise InvalidCapabilityPlanningRequestError("minimum scores must be non-negative.")
        if not isinstance(self.require_unique_workflow, bool):
            raise InvalidCapabilityPlanningRequestError("require_unique_workflow must be a bool.")
        if not isinstance(self.enabled_only, bool):
            raise InvalidCapabilityPlanningRequestError("enabled_only must be a bool.")
        if isinstance(self.maximum_candidates, bool) or not isinstance(self.maximum_candidates, int):
            raise InvalidCapabilityPlanningRequestError("maximum_candidates must be an int.")
        if self.maximum_candidates <= 0 or self.maximum_candidates > MAX_CAPABILITY_PLANNING_CANDIDATES:
            raise InvalidCapabilityPlanningRequestError("maximum_candidates is outside the allowed range.")


@dataclass(frozen=True, slots=True)
class CapabilityPlanningRequest:
    """Structured request for capability-based workflow planning."""

    objective: str
    capability_id: str | None = None
    required_categories: tuple[str, ...] = ()
    excluded_categories: tuple[str, ...] = ()
    required_tags: tuple[str, ...] = ()
    preferred_tags: tuple[str, ...] = ()
    required_inputs: tuple[str, ...] = ()
    required_outputs: tuple[str, ...] = ()
    preferred_workflow_reference: ExecutionPlanReference | None = None
    minimum_capability_score: int = 0
    minimum_workflow_score: int = 0
    require_unique_workflow: bool = True
    enabled_only: bool = True
    maximum_candidates: int = MAX_CAPABILITY_PLANNING_CANDIDATES
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "objective", _validate_objective(self.objective))
        if self.capability_id is not None:
            object.__setattr__(self, "capability_id", _validate_identifier(self.capability_id, "capability_id"))
        object.__setattr__(
            self,
            "required_categories",
            _validate_identifier_group(self.required_categories, "required_categories"),
        )
        object.__setattr__(
            self,
            "excluded_categories",
            _validate_identifier_group(self.excluded_categories, "excluded_categories"),
        )
        object.__setattr__(self, "required_tags", _validate_identifier_group(self.required_tags, "required_tags"))
        object.__setattr__(self, "preferred_tags", _validate_identifier_group(self.preferred_tags, "preferred_tags"))
        object.__setattr__(self, "required_inputs", _validate_identifier_group(self.required_inputs, "required_inputs"))
        object.__setattr__(
            self,
            "required_outputs",
            _validate_identifier_group(self.required_outputs, "required_outputs"),
        )
        _reject_overlap("categories", self.required_categories, self.excluded_categories)
        if self.preferred_workflow_reference is not None:
            if not isinstance(self.preferred_workflow_reference, ExecutionPlanReference):
                raise InvalidCapabilityPlanningRequestError(
                    "preferred_workflow_reference must be ExecutionPlanReference or None."
                )
            object.__setattr__(
                self,
                "preferred_workflow_reference",
                ExecutionPlanReference(
                    self.preferred_workflow_reference.plan_id,
                    self.preferred_workflow_reference.version,
                ),
            )
        CapabilityPlanningPolicy(
            minimum_capability_score=self.minimum_capability_score,
            minimum_workflow_score=self.minimum_workflow_score,
            require_unique_workflow=self.require_unique_workflow,
            enabled_only=self.enabled_only,
            maximum_candidates=self.maximum_candidates,
        )
        object.__setattr__(self, "metadata", MappingProxyType(_safe_metadata(self.metadata)))


@dataclass(frozen=True, slots=True)
class CapabilityPlanningDecision:
    """Traceable decision produced by capability-based planning."""

    status: CapabilityPlanningStatus
    request_signature: str
    capability_resolution_result: CapabilityResolutionResult | None
    workflow_selection_result: WorkflowSelectionResult | None
    selected_capability: CapabilityDefinition | None
    selected_workflow: WorkflowDefinition | None
    selected_workflow_reference: ExecutionPlanReference | None
    plan: ExecutionPlan | None
    reasons: tuple[str, ...] = ()
    library_id: str | None = None
    plan_id: str | None = None
    version: str | None = None
    plan_signature: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", _validate_status(self.status))
        object.__setattr__(self, "reasons", tuple(str(reason) for reason in self.reasons))
        if self.status is CapabilityPlanningStatus.SELECTED:
            if self.plan is None or self.selected_workflow is None or self.selected_workflow_reference is None:
                raise CapabilityPlanningError("selected decisions must include workflow, reference, and plan.")
        elif self.plan is not None:
            raise CapabilityPlanningError("plan can only exist when status is SELECTED.")


class CapabilityPlanner:
    """Bridge CapabilityResolver -> WorkflowSelector -> selected ExecutionPlan."""

    def __init__(
        self,
        resolver: CapabilityResolver,
        selector: WorkflowSelector,
        *,
        execution_plan_libraries: Iterable[ExecutionPlanLibrary] = (),
        execution_plan_registry: ExecutionPlanRegistry | None = None,
    ) -> None:
        if not isinstance(resolver, CapabilityResolver):
            raise CapabilityPlanningError("CapabilityPlanner requires CapabilityResolver.")
        if not isinstance(selector, WorkflowSelector):
            raise CapabilityPlanningError("CapabilityPlanner requires WorkflowSelector.")
        self._resolver = resolver
        self._selector = selector
        self._libraries = tuple(execution_plan_libraries)
        for library in self._libraries:
            if not isinstance(library, ExecutionPlanLibrary):
                raise CapabilityPlanningError("execution_plan_libraries must contain ExecutionPlanLibrary instances.")
        if execution_plan_registry is not None and not isinstance(execution_plan_registry, ExecutionPlanRegistry):
            raise CapabilityPlanningError("execution_plan_registry must be ExecutionPlanRegistry or None.")
        self._registry = execution_plan_registry

    def plan(self, request: CapabilityPlanningRequest) -> CapabilityPlanningDecision:
        if not isinstance(request, CapabilityPlanningRequest):
            return _decision(
                CapabilityPlanningStatus.INVALID_REQUEST,
                "invalid",
                reasons=("request must be CapabilityPlanningRequest",),
            )
        request_signature = capability_planning_request_signature(request)
        resolution_request = _resolution_request_for(request)
        try:
            resolution = self._resolver.resolve(resolution_request)
        except CapabilityResolverError as error:
            return _decision(
                CapabilityPlanningStatus.RESOLUTION_FAILED,
                request_signature,
                reasons=(type(error).__name__,),
            )

        if not resolution.candidates:
            return _decision(
                CapabilityPlanningStatus.NO_CAPABILITY_CANDIDATES,
                request_signature,
                capability_resolution_result=resolution,
                reasons=("resolver returned no candidates",),
            )

        workflow_candidates = tuple(
            candidate
            for candidate in resolution.candidates
            if candidate.capability.capability_type is CapabilityType.WORKFLOW
        )
        if not workflow_candidates:
            if resolution.ambiguous:
                return _decision(
                    CapabilityPlanningStatus.CAPABILITY_AMBIGUOUS,
                    request_signature,
                    capability_resolution_result=resolution,
                    reasons=("resolver returned ambiguous non-workflow capabilities",),
                )
            return _decision(
                CapabilityPlanningStatus.INCOMPATIBLE_CAPABILITY,
                request_signature,
                capability_resolution_result=resolution,
                selected_capability=resolution.candidates[0].capability,
                reasons=("top capability is not a workflow",),
            )

        selection_policy = _selection_policy_for(request, workflow_candidates)
        selection_request = WorkflowSelectionRequest(
            resolution,
            selection_policy,
            preferred_reference=request.preferred_workflow_reference,
            metadata={"request_signature": request_signature},
        )
        try:
            selection = self._selector.select(selection_request)
        except WorkflowSelectorError as error:
            return _decision(
                CapabilityPlanningStatus.SELECTION_FAILED,
                request_signature,
                capability_resolution_result=resolution,
                reasons=(type(error).__name__,),
            )

        if selection.status is not WorkflowSelectionStatus.SELECTED:
            return _decision(
                _planning_status_for_selection(selection.status),
                request_signature,
                capability_resolution_result=resolution,
                workflow_selection_result=selection,
                reasons=tuple(reason.code.value for reason in selection.reasons),
            )

        selected = selection.selected_candidate
        if selected is None:
            return _decision(
                CapabilityPlanningStatus.SELECTION_FAILED,
                request_signature,
                capability_resolution_result=resolution,
                workflow_selection_result=selection,
                reasons=("selector returned SELECTED without selected candidate",),
            )
        if selected.candidate.capability.capability_type is not CapabilityType.WORKFLOW:
            return _decision(
                CapabilityPlanningStatus.INCOMPATIBLE_CAPABILITY,
                request_signature,
                capability_resolution_result=resolution,
                workflow_selection_result=selection,
                selected_capability=selected.candidate.capability,
                reasons=("selected capability is not a workflow",),
            )

        try:
            workflow = self._workflow_for(selected.candidate.capability)
        except WorkflowNotResolvableError as error:
            return _decision(
                CapabilityPlanningStatus.WORKFLOW_NOT_RESOLVABLE,
                request_signature,
                capability_resolution_result=resolution,
                workflow_selection_result=selection,
                selected_capability=selected.candidate.capability,
                reasons=(str(error),),
            )
        except InconsistentWorkflowReferenceError as error:
            return _decision(
                CapabilityPlanningStatus.INCONSISTENT_WORKFLOW_REFERENCE,
                request_signature,
                capability_resolution_result=resolution,
                workflow_selection_result=selection,
                selected_capability=selected.candidate.capability,
                reasons=(str(error),),
            )

        source = selected.candidate.capability.source_reference
        if not isinstance(source, WorkflowCapabilitySource):
            raise IncompatibleCapabilityError("workflow capability must have WorkflowCapabilitySource.")
        signature = plan_signature(workflow.plan)
        if not signature:
            raise InconsistentWorkflowPlanSignatureError("selected workflow plan signature is empty.")
        return _decision(
            CapabilityPlanningStatus.SELECTED,
            request_signature,
            capability_resolution_result=resolution,
            workflow_selection_result=selection,
            selected_capability=selected.candidate.capability,
            selected_workflow=workflow,
            selected_workflow_reference=workflow.reference,
            plan=workflow.plan,
            reasons=("workflow selected",),
            library_id=source.library.library_id,
            plan_id=workflow.reference.plan_id,
            version=workflow.reference.version,
            plan_signature_value=signature,
        )

    def _workflow_for(self, capability: CapabilityDefinition) -> WorkflowDefinition:
        source = capability.source_reference
        if not isinstance(source, WorkflowCapabilitySource):
            raise InconsistentWorkflowReferenceError("selected capability source is not a workflow reference.")
        for library in self._libraries:
            if library.library_id != source.library.library_id:
                continue
            if library.version != source.library.library_version:
                continue
            try:
                workflow = library.get(source.reference)
            except ExecutionPlanLibraryError as error:
                raise WorkflowNotResolvableError("selected workflow reference is not in its source library.") from error
            _validate_workflow_identity(workflow, source)
            return workflow
        if self._registry is not None:
            try:
                plan = self._registry.resolve(source.reference)
            except ExecutionPlanRegistryError as error:
                raise WorkflowNotResolvableError("selected workflow reference is not in the registry.") from error
            return WorkflowDefinition(
                reference=source.reference,
                plan=plan,
                title=capability.title,
                description=capability.description,
                category=_workflow_category_from_capability(capability),
                tags=capability.tags,
                enabled=capability.enabled,
            )
        raise WorkflowNotResolvableError("no library or registry can resolve the selected workflow.")


def capability_planning_request_signature(request: CapabilityPlanningRequest) -> str:
    """Return a stable signature for a capability planning request."""
    if not isinstance(request, CapabilityPlanningRequest):
        raise InvalidCapabilityPlanningRequestError("request must be CapabilityPlanningRequest.")
    preferred = None
    if request.preferred_workflow_reference is not None:
        preferred = {
            "plan_id": request.preferred_workflow_reference.plan_id,
            "version": request.preferred_workflow_reference.version,
        }
    payload = {
        "objective": request.objective,
        "capability_id": request.capability_id,
        "required_categories": sorted(request.required_categories),
        "excluded_categories": sorted(request.excluded_categories),
        "required_tags": sorted(request.required_tags),
        "preferred_tags": sorted(request.preferred_tags),
        "required_inputs": sorted(request.required_inputs),
        "required_outputs": sorted(request.required_outputs),
        "preferred_workflow_reference": preferred,
        "minimum_capability_score": request.minimum_capability_score,
        "minimum_workflow_score": request.minimum_workflow_score,
        "require_unique_workflow": request.require_unique_workflow,
        "enabled_only": request.enabled_only,
        "maximum_candidates": request.maximum_candidates,
        "metadata": _jsonable_mapping(request.metadata),
    }
    encoded = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _resolution_request_for(request: CapabilityPlanningRequest) -> CapabilityResolutionRequest:
    return CapabilityResolutionRequest(
        required_capability_ids=() if request.capability_id is None else (request.capability_id,),
        required_categories=request.required_categories,
        excluded_categories=request.excluded_categories,
        required_tags=request.required_tags,
        preferred_tags=request.preferred_tags,
        required_inputs=request.required_inputs,
        desired_outputs=request.required_outputs,
        enabled_only=request.enabled_only,
        minimum_score=request.minimum_capability_score,
        limit=request.maximum_candidates,
        include_rejections=True,
    )


def _selection_policy_for(
    request: CapabilityPlanningRequest,
    workflow_candidates: tuple[CapabilityCandidate, ...],
) -> WorkflowSelectionPolicy:
    preferred_libraries: tuple[str, ...] = ()
    if request.preferred_workflow_reference is not None:
        preferred_libraries = tuple(
            dict.fromkeys(
                source.library.library_id
                for source in (
                    candidate.capability.source_reference
                    for candidate in workflow_candidates
                    if isinstance(candidate.capability.source_reference, WorkflowCapabilitySource)
                    and candidate.capability.source_reference.reference == request.preferred_workflow_reference
                )
            )
        )
    return WorkflowSelectionPolicy(
        minimum_score=request.minimum_workflow_score,
        require_unique_top_score=request.require_unique_workflow,
        enabled_only=request.enabled_only,
        allowed_categories=request.required_categories,
        excluded_categories=request.excluded_categories,
        required_tags=request.required_tags,
        preferred_tags=request.preferred_tags,
        preferred_library_ids=preferred_libraries,
        maximum_candidates_considered=request.maximum_candidates,
    )


def _planning_status_for_selection(status: WorkflowSelectionStatus) -> CapabilityPlanningStatus:
    if status is WorkflowSelectionStatus.NO_CANDIDATES:
        return CapabilityPlanningStatus.NO_WORKFLOW_CANDIDATES
    if status is WorkflowSelectionStatus.BELOW_MINIMUM_SCORE:
        return CapabilityPlanningStatus.WORKFLOW_BELOW_MINIMUM_SCORE
    if status is WorkflowSelectionStatus.AMBIGUOUS:
        return CapabilityPlanningStatus.WORKFLOW_AMBIGUOUS
    return CapabilityPlanningStatus.SELECTION_FAILED


def _validate_workflow_identity(
    workflow: WorkflowDefinition,
    source: WorkflowCapabilitySource,
) -> None:
    if workflow.reference != source.reference:
        raise InconsistentWorkflowReferenceError("resolved workflow reference does not match selected reference.")


def _workflow_category_from_capability(capability: CapabilityDefinition) -> str:
    for category in capability.categories:
        if category != "workflow":
            return category
    return "workflow"


def _decision(
    status: CapabilityPlanningStatus,
    request_signature: str,
    *,
    capability_resolution_result: CapabilityResolutionResult | None = None,
    workflow_selection_result: WorkflowSelectionResult | None = None,
    selected_capability: CapabilityDefinition | None = None,
    selected_workflow: WorkflowDefinition | None = None,
    selected_workflow_reference: ExecutionPlanReference | None = None,
    plan: ExecutionPlan | None = None,
    reasons: tuple[str, ...] = (),
    library_id: str | None = None,
    plan_id: str | None = None,
    version: str | None = None,
    plan_signature_value: str | None = None,
) -> CapabilityPlanningDecision:
    return CapabilityPlanningDecision(
        status=status,
        request_signature=request_signature,
        capability_resolution_result=capability_resolution_result,
        workflow_selection_result=workflow_selection_result,
        selected_capability=selected_capability,
        selected_workflow=selected_workflow,
        selected_workflow_reference=selected_workflow_reference,
        plan=plan,
        reasons=reasons,
        library_id=library_id,
        plan_id=plan_id,
        version=version,
        plan_signature=plan_signature_value,
    )


def _validate_objective(value: str) -> str:
    if not isinstance(value, str):
        raise InvalidCapabilityPlanningRequestError("objective must be a string.")
    normalized = " ".join(value.strip().split())
    if not normalized:
        raise InvalidCapabilityPlanningRequestError("objective cannot be empty.")
    if len(normalized) > MAX_CAPABILITY_PLANNING_OBJECTIVE_LENGTH:
        raise InvalidCapabilityPlanningRequestError("objective exceeds the length limit.")
    if any(ord(character) < 32 for character in normalized):
        raise InvalidCapabilityPlanningRequestError("objective cannot contain control characters.")
    return normalized


def _validate_identifier_group(values: Iterable[str], field_name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Iterable):
        raise InvalidCapabilityPlanningRequestError(f"{field_name} must be iterable.")
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = _validate_identifier(value, field_name)
        if normalized in seen:
            continue
        seen.add(normalized)
        result.append(normalized)
    if len(result) > MAX_CAPABILITY_PLANNING_ITEMS:
        raise InvalidCapabilityPlanningRequestError(f"{field_name} exceeds the allowed count.")
    return tuple(result)


def _validate_identifier(value: str, field_name: str) -> str:
    if not isinstance(value, str):
        raise InvalidCapabilityPlanningRequestError(f"{field_name} values must be strings.")
    normalized = value.strip().casefold()
    if not normalized:
        raise InvalidCapabilityPlanningRequestError(f"{field_name} cannot contain empty values.")
    if "/" in normalized or "\\" in normalized or ":" in normalized or ".." in normalized:
        raise InvalidCapabilityPlanningRequestError(f"{field_name} cannot contain path-like values.")
    if any(ord(character) < 32 for character in normalized):
        raise InvalidCapabilityPlanningRequestError(f"{field_name} cannot contain control characters.")
    if _IDENTIFIER_PATTERN.fullmatch(normalized) is None:
        raise InvalidCapabilityPlanningRequestError(f"{field_name} contains unsupported characters.")
    return normalized


def _reject_overlap(field_name: str, first: tuple[str, ...], second: tuple[str, ...]) -> None:
    if set(first).intersection(second):
        raise InvalidCapabilityPlanningRequestError(f"{field_name} contains contradictory required/excluded values.")


def _safe_metadata(values: Mapping[str, object], *, depth: int = 0) -> dict[str, object]:
    if not isinstance(values, Mapping):
        raise InvalidCapabilityPlanningRequestError("metadata must be a mapping.")
    if depth > MAX_CAPABILITY_PLANNING_METADATA_DEPTH:
        raise InvalidCapabilityPlanningRequestError("metadata exceeds maximum depth.")
    if len(values) > MAX_CAPABILITY_PLANNING_METADATA_ITEMS:
        raise InvalidCapabilityPlanningRequestError("metadata exceeds maximum size.")
    result: dict[str, object] = {}
    for key, value in values.items():
        result[_validate_identifier(key, "metadata key")] = _safe_metadata_value(value, depth=depth + 1)
    return result


def _safe_metadata_value(value: object, *, depth: int) -> object:
    if value is None or type(value) in {str, int, bool}:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise InvalidCapabilityPlanningRequestError("metadata cannot contain non-finite floats.")
        return value
    if isinstance(value, tuple):
        return tuple(_safe_metadata_value(item, depth=depth + 1) for item in value)
    if isinstance(value, list):
        return tuple(_safe_metadata_value(item, depth=depth + 1) for item in value)
    if isinstance(value, Mapping):
        return MappingProxyType(_safe_metadata(value, depth=depth))
    if isinstance(value, (type, ModuleType)) or callable(value):
        raise InvalidCapabilityPlanningRequestError("metadata contains unsafe runtime objects.")
    raise InvalidCapabilityPlanningRequestError("metadata contains unsupported values.")


def _jsonable_mapping(values: Mapping[str, object]) -> tuple[tuple[str, object], ...]:
    return tuple(sorted((key, _jsonable_value(value)) for key, value in values.items()))


def _jsonable_value(value: object) -> object:
    if isinstance(value, Mapping):
        return _jsonable_mapping(value)
    if isinstance(value, tuple):
        return tuple(_jsonable_value(item) for item in value)
    return value


def _validate_status(status: CapabilityPlanningStatus | str) -> CapabilityPlanningStatus:
    if isinstance(status, CapabilityPlanningStatus):
        return status
    if isinstance(status, str):
        return CapabilityPlanningStatus(status)
    raise CapabilityPlanningError("status must be CapabilityPlanningStatus.")
