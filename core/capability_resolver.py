"""Pure deterministic resolver for Atlas capability metadata."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
import math
import re
from types import MappingProxyType, ModuleType
from typing import Protocol

from core.execution_plan_library import ExecutionPlanLibrary, WorkflowDefinition
from core.execution_plan_registry import ExecutionPlanReference
from core.execution_variable_reference import ExecutionVariableReference
from core.workflow_discovery import WorkflowLibraryReference
from tools.registry import ToolRegistry


MAX_CAPABILITY_PROVIDERS = 16
MAX_SCANNED_CAPABILITIES = 1024
MAX_CAPABILITY_CATEGORIES = 32
MAX_CAPABILITY_TAGS = 64
MAX_CAPABILITY_INPUTS = 64
MAX_CAPABILITY_OUTPUTS = 64
MAX_CAPABILITY_TERMS = 16
MAX_CAPABILITY_RESULTS = 256
MAX_CAPABILITY_STRING_LENGTH = 200
MAX_CAPABILITY_DESCRIPTION_LENGTH = 2000
MAX_CAPABILITY_METADATA_ITEMS = 64
MAX_CAPABILITY_METADATA_DEPTH = 4

CAPABILITY_ID_MATCH_SCORE = 100
CAPABILITY_TYPE_MATCH_SCORE = 40
REQUIRED_CATEGORY_MATCH_SCORE = 30
REQUIRED_TAG_MATCH_SCORE = 20
REQUIRED_INPUT_MATCH_SCORE = 15
DESIRED_OUTPUT_MATCH_SCORE = 15
PREFERRED_TAG_MATCH_SCORE = 10
TITLE_TERM_MATCH_SCORE = 5
ENABLED_BONUS_SCORE = 1

_IDENTIFIER_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,199}$")


class CapabilityResolverError(RuntimeError):
    """Base error for capability resolver failures."""


class CapabilityValidationError(CapabilityResolverError):
    """Raised when capability metadata is unsafe or malformed."""


class CapabilityProviderError(CapabilityResolverError):
    """Raised when a provider cannot return valid capability metadata."""


class ConflictingCapabilityDefinitionError(CapabilityResolverError):
    """Raised when duplicate capability identities disagree."""


class InvalidCapabilityResolutionRequestError(CapabilityResolverError):
    """Raised when a capability resolution request is malformed."""


class CapabilityType(str, Enum):
    """Capability families supported by this phase."""

    TOOL = "tool"
    WORKFLOW = "workflow"


class CapabilityMatchReasonCode(str, Enum):
    """Stable score reason codes."""

    CAPABILITY_ID_MATCH = "capability_id_match"
    CAPABILITY_TYPE_MATCH = "capability_type_match"
    REQUIRED_CATEGORY_MATCH = "required_category_match"
    REQUIRED_TAG_MATCH = "required_tag_match"
    REQUIRED_INPUT_MATCH = "required_input_match"
    DESIRED_OUTPUT_MATCH = "desired_output_match"
    PREFERRED_TAG_MATCH = "preferred_tag_match"
    TITLE_TERM_MATCH = "title_term_match"
    ENABLED_BONUS = "enabled_bonus"


class CapabilityRejectionCode(str, Enum):
    """Stable rejection reason codes."""

    TYPE_MISMATCH = "type_mismatch"
    CAPABILITY_ID_MISMATCH = "capability_id_mismatch"
    CATEGORY_MISMATCH = "category_mismatch"
    EXCLUDED_CATEGORY = "excluded_category"
    MISSING_REQUIRED_TAG = "missing_required_tag"
    MISSING_REQUIRED_INPUT = "missing_required_input"
    DISABLED = "disabled"
    BELOW_MINIMUM_SCORE = "below_minimum_score"


@dataclass(frozen=True, slots=True)
class ToolCapabilitySource:
    """Structured source reference for a registered tool."""

    tool_name: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "tool_name", _validate_identifier(self.tool_name, "tool_name"))


@dataclass(frozen=True, slots=True)
class WorkflowCapabilitySource:
    """Structured source reference for a workflow in an execution-plan library."""

    library: WorkflowLibraryReference
    reference: ExecutionPlanReference

    def __post_init__(self) -> None:
        if not isinstance(self.library, WorkflowLibraryReference):
            raise CapabilityValidationError("workflow source library must be WorkflowLibraryReference.")
        if not isinstance(self.reference, ExecutionPlanReference):
            raise CapabilityValidationError("workflow source reference must be ExecutionPlanReference.")
        object.__setattr__(
            self,
            "library",
            WorkflowLibraryReference(
                _validate_identifier(self.library.library_id, "library_id"),
                self.library.library_version,
            ),
        )
        object.__setattr__(
            self,
            "reference",
            ExecutionPlanReference(self.reference.plan_id, self.reference.version),
        )


@dataclass(frozen=True, slots=True)
class CapabilityDefinition:
    """Immutable description of a real Atlas capability."""

    capability_id: str
    capability_type: CapabilityType
    title: str
    description: str
    categories: tuple[str, ...]
    tags: tuple[str, ...]
    input_names: tuple[str, ...]
    output_names: tuple[str, ...]
    enabled: bool
    source_reference: ToolCapabilitySource | WorkflowCapabilitySource
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "capability_id", _validate_identifier(self.capability_id, "capability_id"))
        object.__setattr__(self, "capability_type", _validate_capability_type(self.capability_type))
        object.__setattr__(self, "title", _validate_text(self.title, "title", MAX_CAPABILITY_STRING_LENGTH))
        object.__setattr__(
            self,
            "description",
            _validate_text(self.description, "description", MAX_CAPABILITY_DESCRIPTION_LENGTH),
        )
        object.__setattr__(
            self,
            "categories",
            _validate_identifier_group(self.categories, "categories", MAX_CAPABILITY_CATEGORIES),
        )
        object.__setattr__(
            self,
            "tags",
            _validate_identifier_group(self.tags, "tags", MAX_CAPABILITY_TAGS),
        )
        object.__setattr__(
            self,
            "input_names",
            _validate_identifier_group(self.input_names, "input_names", MAX_CAPABILITY_INPUTS),
        )
        object.__setattr__(
            self,
            "output_names",
            _validate_identifier_group(self.output_names, "output_names", MAX_CAPABILITY_OUTPUTS),
        )
        if not isinstance(self.enabled, bool):
            raise CapabilityValidationError("enabled must be a bool.")
        if self.capability_type is CapabilityType.TOOL and not isinstance(self.source_reference, ToolCapabilitySource):
            raise CapabilityValidationError("tool capabilities require ToolCapabilitySource.")
        if self.capability_type is CapabilityType.WORKFLOW and not isinstance(self.source_reference, WorkflowCapabilitySource):
            raise CapabilityValidationError("workflow capabilities require WorkflowCapabilitySource.")
        object.__setattr__(self, "metadata", MappingProxyType(_safe_metadata(self.metadata)))


@dataclass(frozen=True, slots=True)
class CapabilityResolutionRequest:
    """Explicit deterministic criteria for capability resolution."""

    capability_types: tuple[CapabilityType, ...] = ()
    required_capability_ids: tuple[str, ...] = ()
    required_categories: tuple[str, ...] = ()
    excluded_categories: tuple[str, ...] = ()
    required_tags: tuple[str, ...] = ()
    preferred_tags: tuple[str, ...] = ()
    required_inputs: tuple[str, ...] = ()
    desired_outputs: tuple[str, ...] = ()
    title_terms: tuple[str, ...] = ()
    enabled_only: bool = True
    minimum_score: int = 0
    limit: int | None = None
    include_rejections: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "capability_types", _validate_type_group(self.capability_types))
        object.__setattr__(
            self,
            "required_capability_ids",
            _validate_identifier_group(
                self.required_capability_ids,
                "required_capability_ids",
                MAX_CAPABILITY_RESULTS,
            ),
        )
        object.__setattr__(
            self,
            "required_categories",
            _validate_identifier_group(self.required_categories, "required_categories", MAX_CAPABILITY_CATEGORIES),
        )
        object.__setattr__(
            self,
            "excluded_categories",
            _validate_identifier_group(self.excluded_categories, "excluded_categories", MAX_CAPABILITY_CATEGORIES),
        )
        object.__setattr__(
            self,
            "required_tags",
            _validate_identifier_group(self.required_tags, "required_tags", MAX_CAPABILITY_TAGS),
        )
        object.__setattr__(
            self,
            "preferred_tags",
            _validate_identifier_group(self.preferred_tags, "preferred_tags", MAX_CAPABILITY_TAGS),
        )
        object.__setattr__(
            self,
            "required_inputs",
            _validate_identifier_group(self.required_inputs, "required_inputs", MAX_CAPABILITY_INPUTS),
        )
        object.__setattr__(
            self,
            "desired_outputs",
            _validate_identifier_group(self.desired_outputs, "desired_outputs", MAX_CAPABILITY_OUTPUTS),
        )
        object.__setattr__(self, "title_terms", _validate_title_terms(self.title_terms))
        if not isinstance(self.enabled_only, bool):
            raise InvalidCapabilityResolutionRequestError("enabled_only must be a bool.")
        if not isinstance(self.include_rejections, bool):
            raise InvalidCapabilityResolutionRequestError("include_rejections must be a bool.")
        if isinstance(self.minimum_score, bool) or not isinstance(self.minimum_score, int) or self.minimum_score < 0:
            raise InvalidCapabilityResolutionRequestError("minimum_score must be a non-negative int.")
        if self.limit is not None:
            if isinstance(self.limit, bool) or not isinstance(self.limit, int):
                raise InvalidCapabilityResolutionRequestError("limit must be a positive int or None.")
            if self.limit <= 0 or self.limit > MAX_CAPABILITY_RESULTS:
                raise InvalidCapabilityResolutionRequestError("limit is outside the allowed range.")


@dataclass(frozen=True, slots=True)
class CapabilityMatchReason:
    """One structured score contribution."""

    code: CapabilityMatchReasonCode
    value: str | None
    score: int

    def __post_init__(self) -> None:
        if isinstance(self.score, bool) or not isinstance(self.score, int):
            raise CapabilityValidationError("match reason score must be int.")


@dataclass(frozen=True, slots=True)
class CapabilityCandidate:
    """Scored capability candidate."""

    capability: CapabilityDefinition
    score: int
    reasons: tuple[CapabilityMatchReason, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "reasons", tuple(self.reasons))
        if sum(reason.score for reason in self.reasons) != self.score:
            raise CapabilityValidationError("candidate score must equal the sum of reasons.")


@dataclass(frozen=True, slots=True)
class CapabilityRejection:
    """Structured reason why a capability was filtered out."""

    capability: CapabilityDefinition
    reason_code: CapabilityRejectionCode
    message: str


@dataclass(frozen=True, slots=True)
class CapabilityResolutionResult:
    """Immutable result for a resolution attempt."""

    request: CapabilityResolutionRequest
    candidates: tuple[CapabilityCandidate, ...]
    rejected: tuple[CapabilityRejection, ...]
    scanned_capabilities: int
    matched_capabilities: int
    truncated: bool
    ambiguous: bool
    top_score: int | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "candidates", tuple(self.candidates))
        object.__setattr__(self, "rejected", tuple(self.rejected))


class CapabilityProvider(Protocol):
    """Read-only provider of real capability metadata."""

    def list_capabilities(self) -> tuple[CapabilityDefinition, ...]:
        """Return immutable capability definitions without side effects."""
        ...


class ToolCapabilityProvider:
    """Capability provider backed by the real ToolRegistry."""

    def __init__(self, registry: ToolRegistry) -> None:
        if not isinstance(registry, ToolRegistry):
            raise CapabilityProviderError("ToolCapabilityProvider requires ToolRegistry.")
        self._registry = registry

    def list_capabilities(self) -> tuple[CapabilityDefinition, ...]:
        capabilities: list[CapabilityDefinition] = []
        for descriptor in self._registry.descriptors():
            schema = descriptor.arguments_schema
            inputs = (
                tuple(parameter.name for parameter in schema.parameters)
                if schema is not None
                else tuple(descriptor.argument_names)
            )
            capabilities.append(
                CapabilityDefinition(
                    capability_id=f"tool.{descriptor.name}",
                    capability_type=CapabilityType.TOOL,
                    title=descriptor.name,
                    description=descriptor.description,
                    categories=_tool_categories(descriptor.name),
                    tags=_tool_tags(descriptor.name, descriptor.requires_confirmation),
                    input_names=inputs,
                    output_names=("result",) if descriptor.output_description else (),
                    enabled=True,
                    source_reference=ToolCapabilitySource(descriptor.name),
                    metadata={
                        "requires_confirmation": descriptor.requires_confirmation,
                        "dangerous": descriptor.dangerous,
                    },
                )
            )
        return tuple(capabilities)


class WorkflowCapabilityProvider:
    """Capability provider backed by one or more ExecutionPlanLibrary objects."""

    def __init__(self, libraries: Iterable[ExecutionPlanLibrary]) -> None:
        self._libraries = tuple(libraries)
        for library in self._libraries:
            if not isinstance(library, ExecutionPlanLibrary):
                raise CapabilityProviderError("WorkflowCapabilityProvider requires ExecutionPlanLibrary instances.")

    def list_capabilities(self) -> tuple[CapabilityDefinition, ...]:
        capabilities: list[CapabilityDefinition] = []
        for library in sorted(self._libraries, key=lambda item: (item.library_id, item.version or "")):
            workflows = library.workflows()
            if not isinstance(workflows, tuple):
                raise CapabilityProviderError("ExecutionPlanLibrary.workflows must return a tuple.")
            for workflow in workflows:
                if not isinstance(workflow, WorkflowDefinition):
                    raise CapabilityProviderError("ExecutionPlanLibrary returned an invalid workflow.")
                capabilities.append(
                    CapabilityDefinition(
                        capability_id=_workflow_capability_id(library.library_id, workflow.reference),
                        capability_type=CapabilityType.WORKFLOW,
                        title=workflow.title,
                        description=workflow.description,
                        categories=("workflow", workflow.category),
                        tags=workflow.tags,
                        input_names=_workflow_input_names(workflow),
                        output_names=_workflow_output_names(workflow),
                        enabled=workflow.enabled,
                        source_reference=WorkflowCapabilitySource(
                            WorkflowLibraryReference(library.library_id, library.version),
                            workflow.reference,
                        ),
                        metadata={"library_id": library.library_id, "library_version": library.version},
                    )
                )
        return tuple(capabilities)


def _workflow_input_names(workflow) -> tuple[str, ...]:
    names: list[str] = []
    for step in workflow.plan.ordered_steps:
        _collect_variable_reference_names(step.arguments.as_dict(), names)
    if workflow.plan.output is not None:
        _collect_variable_reference_names(workflow.plan.output.as_definition(), names)
    return tuple(dict.fromkeys(names))


def _collect_variable_reference_names(value: object, names: list[str]) -> None:
    if isinstance(value, ExecutionVariableReference):
        if value.name not in names:
            names.append(value.name)
        return
    if isinstance(value, Mapping):
        for item in value.values():
            _collect_variable_reference_names(item, names)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _collect_variable_reference_names(item, names)


def _workflow_output_names(workflow: WorkflowDefinition) -> tuple[str, ...]:
    if workflow.plan.output is None:
        return ()
    definition = workflow.plan.output.as_definition()
    if not isinstance(definition, Mapping):
        return ("result",)
    return tuple(
        key
        for key in definition
        if isinstance(key, str) and _IDENTIFIER_PATTERN.fullmatch(key) is not None
    )


class CapabilityResolver:
    """Resolve compatible capabilities from injected read-only providers."""

    def __init__(self, providers: Iterable[CapabilityProvider]) -> None:
        self._providers = tuple(providers)
        if len(self._providers) > MAX_CAPABILITY_PROVIDERS:
            raise CapabilityProviderError("too many capability providers.")

    def resolve(self, request: CapabilityResolutionRequest) -> CapabilityResolutionResult:
        if not isinstance(request, CapabilityResolutionRequest):
            raise InvalidCapabilityResolutionRequestError("request must be CapabilityResolutionRequest.")
        capabilities = self._load_capabilities()
        candidates: list[CapabilityCandidate] = []
        rejected: list[CapabilityRejection] = []
        for capability in capabilities:
            rejection = _first_rejection(request, capability)
            if rejection is not None:
                if request.include_rejections:
                    rejected.append(rejection)
                continue
            candidate = _candidate_for(request, capability)
            if candidate.score < request.minimum_score:
                if request.include_rejections:
                    rejected.append(
                        CapabilityRejection(
                            capability,
                            CapabilityRejectionCode.BELOW_MINIMUM_SCORE,
                            "Capability score is below the requested minimum.",
                        )
                    )
                continue
            candidates.append(candidate)

        candidates = list(_deduplicate_candidates(candidates))
        sorted_candidates = tuple(sorted(candidates, key=_candidate_sort_key))
        truncated = request.limit is not None and len(sorted_candidates) > request.limit
        if request.limit is not None:
            sorted_candidates = sorted_candidates[: request.limit]
        top_score = sorted_candidates[0].score if sorted_candidates else None
        ambiguous = top_score is not None and sum(1 for item in sorted_candidates if item.score == top_score) >= 2
        return CapabilityResolutionResult(
            request=request,
            candidates=sorted_candidates,
            rejected=tuple(sorted(rejected, key=_rejection_sort_key)) if request.include_rejections else (),
            scanned_capabilities=len(capabilities),
            matched_capabilities=len(candidates),
            truncated=truncated,
            ambiguous=ambiguous,
            top_score=top_score,
        )

    def _load_capabilities(self) -> tuple[CapabilityDefinition, ...]:
        capabilities: list[CapabilityDefinition] = []
        for provider in self._providers:
            try:
                provided = provider.list_capabilities()
            except CapabilityResolverError:
                raise
            except Exception as error:
                raise CapabilityProviderError("capability provider failed.") from error
            if not isinstance(provided, tuple):
                raise CapabilityProviderError("capability provider must return a tuple.")
            for capability in provided:
                if not isinstance(capability, CapabilityDefinition):
                    raise CapabilityProviderError("capability provider returned invalid data.")
                capabilities.append(capability)
                if len(capabilities) > MAX_SCANNED_CAPABILITIES:
                    raise CapabilityProviderError("scanned capabilities exceed the safety limit.")
        return tuple(capabilities)


def capability_resolution_request_signature(request: CapabilityResolutionRequest) -> str:
    """Return a deterministic signature for semantically equivalent requests."""
    if not isinstance(request, CapabilityResolutionRequest):
        raise InvalidCapabilityResolutionRequestError("request must be CapabilityResolutionRequest.")
    payload = {
        "capability_types": sorted(item.value for item in request.capability_types),
        "required_capability_ids": sorted(request.required_capability_ids),
        "required_categories": sorted(request.required_categories),
        "excluded_categories": sorted(request.excluded_categories),
        "required_tags": sorted(request.required_tags),
        "preferred_tags": sorted(request.preferred_tags),
        "required_inputs": sorted(request.required_inputs),
        "desired_outputs": sorted(request.desired_outputs),
        "title_terms": sorted(request.title_terms),
        "enabled_only": request.enabled_only,
        "minimum_score": request.minimum_score,
        "limit": request.limit,
        "include_rejections": request.include_rejections,
    }
    encoded = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _candidate_for(request: CapabilityResolutionRequest, capability: CapabilityDefinition) -> CapabilityCandidate:
    reasons: list[CapabilityMatchReason] = []
    if capability.capability_id in request.required_capability_ids:
        reasons.append(CapabilityMatchReason(CapabilityMatchReasonCode.CAPABILITY_ID_MATCH, capability.capability_id, CAPABILITY_ID_MATCH_SCORE))
    if capability.capability_type in request.capability_types:
        reasons.append(CapabilityMatchReason(CapabilityMatchReasonCode.CAPABILITY_TYPE_MATCH, capability.capability_type.value, CAPABILITY_TYPE_MATCH_SCORE))
    for category in request.required_categories:
        if category in capability.categories:
            reasons.append(CapabilityMatchReason(CapabilityMatchReasonCode.REQUIRED_CATEGORY_MATCH, category, REQUIRED_CATEGORY_MATCH_SCORE))
    for tag in request.required_tags:
        if tag in capability.tags:
            reasons.append(CapabilityMatchReason(CapabilityMatchReasonCode.REQUIRED_TAG_MATCH, tag, REQUIRED_TAG_MATCH_SCORE))
    for input_name in request.required_inputs:
        if input_name in capability.input_names:
            reasons.append(CapabilityMatchReason(CapabilityMatchReasonCode.REQUIRED_INPUT_MATCH, input_name, REQUIRED_INPUT_MATCH_SCORE))
    for output_name in request.desired_outputs:
        if output_name in capability.output_names:
            reasons.append(CapabilityMatchReason(CapabilityMatchReasonCode.DESIRED_OUTPUT_MATCH, output_name, DESIRED_OUTPUT_MATCH_SCORE))
    for tag in request.preferred_tags:
        if tag in capability.tags:
            reasons.append(CapabilityMatchReason(CapabilityMatchReasonCode.PREFERRED_TAG_MATCH, tag, PREFERRED_TAG_MATCH_SCORE))
    title = _normalize_text(capability.title)
    for term in request.title_terms:
        if term in title:
            reasons.append(CapabilityMatchReason(CapabilityMatchReasonCode.TITLE_TERM_MATCH, term, TITLE_TERM_MATCH_SCORE))
    if capability.enabled:
        reasons.append(CapabilityMatchReason(CapabilityMatchReasonCode.ENABLED_BONUS, None, ENABLED_BONUS_SCORE))
    return CapabilityCandidate(capability, sum(reason.score for reason in reasons), tuple(reasons))


def _first_rejection(
    request: CapabilityResolutionRequest,
    capability: CapabilityDefinition,
) -> CapabilityRejection | None:
    if request.capability_types and capability.capability_type not in request.capability_types:
        return CapabilityRejection(capability, CapabilityRejectionCode.TYPE_MISMATCH, "Capability type is not allowed.")
    if request.required_capability_ids and capability.capability_id not in request.required_capability_ids:
        return CapabilityRejection(capability, CapabilityRejectionCode.CAPABILITY_ID_MISMATCH, "Capability id is not allowed.")
    if request.required_categories and any(category not in capability.categories for category in request.required_categories):
        return CapabilityRejection(capability, CapabilityRejectionCode.CATEGORY_MISMATCH, "Capability is missing a required category.")
    if any(category in capability.categories for category in request.excluded_categories):
        return CapabilityRejection(capability, CapabilityRejectionCode.EXCLUDED_CATEGORY, "Capability has an excluded category.")
    if any(tag not in capability.tags for tag in request.required_tags):
        return CapabilityRejection(capability, CapabilityRejectionCode.MISSING_REQUIRED_TAG, "Capability is missing a required tag.")
    if any(input_name not in capability.input_names for input_name in request.required_inputs):
        return CapabilityRejection(capability, CapabilityRejectionCode.MISSING_REQUIRED_INPUT, "Capability is missing a required input.")
    if request.enabled_only and not capability.enabled:
        return CapabilityRejection(capability, CapabilityRejectionCode.DISABLED, "Capability is disabled.")
    return None


def _deduplicate_candidates(candidates: list[CapabilityCandidate]) -> tuple[CapabilityCandidate, ...]:
    grouped: dict[tuple[object, ...], CapabilityCandidate] = {}
    capability_ids: dict[str, CapabilityCandidate] = {}
    for candidate in candidates:
        previous_id = capability_ids.get(candidate.capability.capability_id)
        if previous_id is not None and _identity_key(previous_id.capability) != _identity_key(candidate.capability):
            raise ConflictingCapabilityDefinitionError("duplicate capability id has conflicting sources.")
        capability_ids[candidate.capability.capability_id] = candidate
        key = _identity_key(candidate.capability)
        previous = grouped.get(key)
        if previous is None:
            grouped[key] = candidate
            continue
        if _definition_payload(previous.capability) != _definition_payload(candidate.capability):
            raise ConflictingCapabilityDefinitionError("duplicate capability identity has conflicting definitions.")
    return tuple(grouped.values())


def _identity_key(capability: CapabilityDefinition) -> tuple[object, ...]:
    source = capability.source_reference
    if isinstance(source, ToolCapabilitySource):
        return (CapabilityType.TOOL.value, source.tool_name)
    if isinstance(source, WorkflowCapabilitySource):
        return (
            CapabilityType.WORKFLOW.value,
            source.library.library_id,
            source.reference.plan_id,
            source.reference.version,
        )
    raise CapabilityValidationError("unknown source reference.")


def _definition_payload(capability: CapabilityDefinition) -> tuple[object, ...]:
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
        _identity_key(capability),
        _jsonable_mapping(capability.metadata),
    )


def _candidate_sort_key(candidate: CapabilityCandidate) -> tuple[int, int, str, str, tuple[str, ...]]:
    capability = candidate.capability
    return (
        -candidate.score,
        0 if capability.enabled else 1,
        capability.capability_type.value,
        capability.capability_id,
        tuple(str(item) for item in _identity_key(capability)),
    )


def _rejection_sort_key(rejection: CapabilityRejection) -> tuple[str, str, str]:
    return (
        rejection.capability.capability_type.value,
        rejection.capability.capability_id,
        rejection.reason_code.value,
    )


def _validate_capability_type(value: CapabilityType | str) -> CapabilityType:
    if isinstance(value, CapabilityType):
        return value
    if isinstance(value, str):
        try:
            return CapabilityType(value.strip().casefold())
        except ValueError as error:
            raise CapabilityValidationError("unsupported capability type.") from error
    raise CapabilityValidationError("capability_type must be CapabilityType.")


def _validate_type_group(values: Iterable[CapabilityType | str]) -> tuple[CapabilityType, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Iterable):
        raise InvalidCapabilityResolutionRequestError("capability_types must be iterable.")
    result: list[CapabilityType] = []
    seen: set[CapabilityType] = set()
    for value in values:
        item = _validate_capability_type(value)
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return tuple(result)


def _validate_identifier_group(values: Iterable[str], field: str, maximum: int) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Iterable):
        raise InvalidCapabilityResolutionRequestError(f"{field} must be iterable.")
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        item = _validate_identifier(value, field)
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    if len(result) > maximum:
        raise InvalidCapabilityResolutionRequestError(f"{field} exceeds the allowed count.")
    return tuple(result)


def _validate_identifier(value: str, field: str) -> str:
    if not isinstance(value, str):
        raise CapabilityValidationError(f"{field} values must be strings.")
    normalized = value.strip().casefold()
    if not normalized:
        raise CapabilityValidationError(f"{field} cannot be empty.")
    if len(normalized) > MAX_CAPABILITY_STRING_LENGTH:
        raise CapabilityValidationError(f"{field} exceeds the length limit.")
    if "/" in normalized or "\\" in normalized or ":" in normalized or ".." in normalized:
        raise CapabilityValidationError(f"{field} cannot be a path-like value.")
    if any(ord(character) < 32 for character in normalized):
        raise CapabilityValidationError(f"{field} cannot contain control characters.")
    if _IDENTIFIER_PATTERN.fullmatch(normalized) is None:
        raise CapabilityValidationError(f"{field} has unsupported characters.")
    return normalized


def _validate_title_terms(values: Iterable[str]) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Iterable):
        raise InvalidCapabilityResolutionRequestError("title_terms must be iterable.")
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not isinstance(value, str):
            raise InvalidCapabilityResolutionRequestError("title_terms values must be strings.")
        item = _normalize_text(value)
        if not item:
            raise InvalidCapabilityResolutionRequestError("title_terms cannot contain empty values.")
        if len(item) > MAX_CAPABILITY_STRING_LENGTH:
            raise InvalidCapabilityResolutionRequestError("title_terms value exceeds the length limit.")
        if any(ord(character) < 32 for character in item):
            raise InvalidCapabilityResolutionRequestError("title_terms cannot contain control characters.")
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    if len(result) > MAX_CAPABILITY_TERMS:
        raise InvalidCapabilityResolutionRequestError("title_terms exceeds the allowed count.")
    return tuple(result)


def _validate_text(value: str, field: str, maximum: int) -> str:
    if not isinstance(value, str):
        raise CapabilityValidationError(f"{field} must be a string.")
    normalized = value.strip()
    if not normalized:
        raise CapabilityValidationError(f"{field} cannot be empty.")
    if len(normalized) > maximum:
        raise CapabilityValidationError(f"{field} exceeds the length limit.")
    if any(ord(character) < 32 and character not in "\r\n\t" for character in normalized):
        raise CapabilityValidationError(f"{field} cannot contain control characters.")
    return normalized


def _safe_metadata(values: Mapping[str, object], *, depth: int = 0) -> dict[str, object]:
    if not isinstance(values, Mapping):
        raise CapabilityValidationError("metadata must be a mapping.")
    if depth > MAX_CAPABILITY_METADATA_DEPTH:
        raise CapabilityValidationError("metadata exceeds maximum depth.")
    if len(values) > MAX_CAPABILITY_METADATA_ITEMS:
        raise CapabilityValidationError("metadata exceeds maximum size.")
    result: dict[str, object] = {}
    for key, value in values.items():
        result[_validate_identifier(key, "metadata key")] = _safe_metadata_value(value, depth=depth + 1)
    return result


def _safe_metadata_value(value: object, *, depth: int) -> object:
    if value is None or type(value) in {str, int, bool}:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise CapabilityValidationError("metadata cannot contain non-finite floats.")
        return value
    if isinstance(value, tuple):
        return tuple(_safe_metadata_value(item, depth=depth + 1) for item in value)
    if isinstance(value, list):
        return tuple(_safe_metadata_value(item, depth=depth + 1) for item in value)
    if isinstance(value, Mapping):
        return MappingProxyType(_safe_metadata(value, depth=depth))
    if isinstance(value, (type, ModuleType)) or callable(value):
        raise CapabilityValidationError("metadata contains unsafe runtime objects.")
    raise CapabilityValidationError("metadata contains unsupported values.")


def _jsonable_mapping(values: Mapping[str, object]) -> tuple[tuple[str, object], ...]:
    return tuple(sorted((key, _jsonable_value(value)) for key, value in values.items()))


def _jsonable_value(value: object) -> object:
    if isinstance(value, Mapping):
        return _jsonable_mapping(value)
    if isinstance(value, tuple):
        return tuple(_jsonable_value(item) for item in value)
    return value


def _normalize_text(value: str) -> str:
    return " ".join(value.strip().casefold().split())


def _tool_categories(tool_name: str) -> tuple[str, ...]:
    prefix = tool_name.split(".", 1)[0].split("_", 1)[0].casefold()
    return ("tool",) if prefix == "tool" else ("tool", _validate_identifier(prefix, "tool category"))


def _tool_tags(tool_name: str, requires_confirmation: bool) -> tuple[str, ...]:
    tags = [_validate_identifier(part, "tool tag") for part in re.split(r"[._]+", tool_name) if part]
    if requires_confirmation:
        tags.append("confirmation_required")
    return tuple(dict.fromkeys(tags))


def _workflow_capability_id(library_id: str, reference: ExecutionPlanReference) -> str:
    version = reference.version or "unversioned"
    return _validate_identifier(f"workflow.{library_id}.{reference.plan_id}.{version}", "capability_id")
