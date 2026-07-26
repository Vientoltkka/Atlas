"""Deterministic resolver for specialized Atlas agent definitions."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
from types import MappingProxyType
from typing import Any

from core.agent_registry import AgentDefinition, AgentRegistry, AgentType


MAX_AGENT_RESOLUTION_IDS = 32
MAX_AGENT_RESOLUTION_CANDIDATES = 64
MAX_AGENT_RESOLUTION_METADATA_ITEMS = 16
MAX_AGENT_RESOLUTION_METADATA_DEPTH = 3
MAX_AGENT_RESOLUTION_METADATA_NODES = 64
MAX_AGENT_RESOLUTION_SCORE = 10_000
_PERMISSION_IDS = frozenset(
    {
        "can_read_project",
        "can_write_files",
        "can_execute_tools",
        "can_modify_memory",
        "can_use_network",
        "requires_confirmation",
    }
)
_SENSITIVE_KEY_PARTS = ("secret", "token", "password", "api_key", "apikey", "authorization")


class AgentResolverError(RuntimeError):
    """Base error for deterministic agent resolution."""


class InvalidAgentResolutionRequestError(AgentResolverError):
    """Raised when an agent-resolution request is malformed."""


class AgentResolutionStatus(str, Enum):
    """Structured terminal statuses for agent resolution."""

    RESOLVED = "RESOLVED"
    NO_AGENTS = "NO_AGENTS"
    NO_MATCHING_AGENTS = "NO_MATCHING_AGENTS"
    BELOW_MINIMUM_SCORE = "BELOW_MINIMUM_SCORE"
    AMBIGUOUS = "AMBIGUOUS"
    INVALID_REQUEST = "INVALID_REQUEST"
    REGISTRY_UNAVAILABLE = "REGISTRY_UNAVAILABLE"
    INTERNAL_ERROR = "INTERNAL_ERROR"


class AgentResolutionReasonCode(str, Enum):
    """Structured reasons for scoring and selection."""

    PREFERRED_AGENT_ID_MATCH = "preferred_agent_id_match"
    REQUIRED_CAPABILITY_MATCH = "required_capability_match"
    PREFERRED_CAPABILITY_MATCH = "preferred_capability_match"
    REQUIRED_AGENT_TYPE_MATCH = "required_agent_type_match"
    PREFERRED_AGENT_TYPE_MATCH = "preferred_agent_type_match"
    REQUIRED_PERMISSION_MATCH = "required_permission_match"
    PREFERRED_PERMISSION_MATCH = "preferred_permission_match"
    ENABLED_AGENT = "enabled_agent"
    UNIQUE_TOP_SELECTED = "unique_top_selected"
    DETERMINISTIC_TIEBREAK_SELECTED = "deterministic_tiebreak_selected"
    TOP_SCORE_TIE = "top_score_tie"


class AgentResolutionRejectionCode(str, Enum):
    """Structured reasons for rejecting a registered agent."""

    DISABLED = "disabled"
    EXCLUDED_AGENT_ID = "excluded_agent_id"
    REQUIRED_CAPABILITY_MISSING = "required_capability_missing"
    REQUIRED_AGENT_TYPE_MISSING = "required_agent_type_missing"
    REQUIRED_PERMISSION_MISSING = "required_permission_missing"


@dataclass(frozen=True, slots=True)
class AgentResolutionRequest:
    """Structured deterministic criteria for selecting one agent."""

    required_capability_ids: tuple[str, ...] = ()
    preferred_capability_ids: tuple[str, ...] = ()
    required_agent_types: tuple[AgentType | str, ...] = ()
    preferred_agent_types: tuple[AgentType | str, ...] = ()
    required_permission_ids: tuple[str, ...] = ()
    preferred_permission_ids: tuple[str, ...] = ()
    required_agent_ids: tuple[str, ...] = ()
    excluded_agent_ids: tuple[str, ...] = ()
    preferred_agent_ids: tuple[str, ...] = ()
    enabled_only: bool = True
    minimum_score: int = 0
    require_unique_top_score: bool = True
    maximum_candidates_considered: int = MAX_AGENT_RESOLUTION_CANDIDATES
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "required_capability_ids",
            _identifier_tuple(self.required_capability_ids, "required_capability_ids"),
        )
        object.__setattr__(
            self,
            "preferred_capability_ids",
            _identifier_tuple(self.preferred_capability_ids, "preferred_capability_ids"),
        )
        object.__setattr__(
            self,
            "required_agent_types",
            _agent_type_tuple(self.required_agent_types, "required_agent_types"),
        )
        object.__setattr__(
            self,
            "preferred_agent_types",
            _agent_type_tuple(self.preferred_agent_types, "preferred_agent_types"),
        )
        object.__setattr__(
            self,
            "required_permission_ids",
            _permission_tuple(self.required_permission_ids, "required_permission_ids"),
        )
        object.__setattr__(
            self,
            "preferred_permission_ids",
            _permission_tuple(self.preferred_permission_ids, "preferred_permission_ids"),
        )
        object.__setattr__(
            self,
            "required_agent_ids",
            _identifier_tuple(self.required_agent_ids, "required_agent_ids"),
        )
        object.__setattr__(
            self,
            "excluded_agent_ids",
            _identifier_tuple(self.excluded_agent_ids, "excluded_agent_ids"),
        )
        object.__setattr__(
            self,
            "preferred_agent_ids",
            _identifier_tuple(self.preferred_agent_ids, "preferred_agent_ids"),
        )
        if not isinstance(self.enabled_only, bool):
            raise InvalidAgentResolutionRequestError("enabled_only must be a bool.")
        if not isinstance(self.require_unique_top_score, bool):
            raise InvalidAgentResolutionRequestError("require_unique_top_score must be a bool.")
        if isinstance(self.minimum_score, bool) or not isinstance(self.minimum_score, int):
            raise InvalidAgentResolutionRequestError("minimum_score must be an integer.")
        if self.minimum_score < 0 or self.minimum_score > MAX_AGENT_RESOLUTION_SCORE:
            raise InvalidAgentResolutionRequestError("minimum_score is outside the allowed range.")
        if isinstance(self.maximum_candidates_considered, bool) or not isinstance(
            self.maximum_candidates_considered,
            int,
        ):
            raise InvalidAgentResolutionRequestError("maximum_candidates_considered must be an integer.")
        if (
            self.maximum_candidates_considered <= 0
            or self.maximum_candidates_considered > MAX_AGENT_RESOLUTION_CANDIDATES
        ):
            raise InvalidAgentResolutionRequestError("maximum_candidates_considered is outside the allowed range.")
        object.__setattr__(self, "metadata", MappingProxyType(_safe_metadata(self.metadata)))


@dataclass(frozen=True, slots=True)
class AgentResolutionReason:
    """One structured score contribution or selection explanation."""

    code: AgentResolutionReasonCode
    value: str | None = None
    score: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "code", _reason_code(self.code))
        if self.value is not None:
            object.__setattr__(self, "value", _identifier(self.value, "value"))
        if isinstance(self.score, bool) or not isinstance(self.score, int):
            raise InvalidAgentResolutionRequestError("reason score must be an integer.")


@dataclass(frozen=True, slots=True)
class AgentResolutionCandidate:
    """Agent candidate with deterministic score and reasons."""

    agent: AgentDefinition
    score: int
    reasons: tuple[AgentResolutionReason, ...]
    satisfied_capability_count: int = 0
    preferred_agent_id_match: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.agent, AgentDefinition):
            raise InvalidAgentResolutionRequestError("candidate agent must be AgentDefinition.")
        object.__setattr__(self, "reasons", tuple(self.reasons))
        if not all(isinstance(reason, AgentResolutionReason) for reason in self.reasons):
            raise InvalidAgentResolutionRequestError("candidate reasons must be AgentResolutionReason values.")
        if sum(reason.score for reason in self.reasons) != self.score:
            raise InvalidAgentResolutionRequestError("candidate score must equal the sum of reasons.")
        if isinstance(self.satisfied_capability_count, bool) or not isinstance(self.satisfied_capability_count, int):
            raise InvalidAgentResolutionRequestError("satisfied_capability_count must be an integer.")
        if not isinstance(self.preferred_agent_id_match, bool):
            raise InvalidAgentResolutionRequestError("preferred_agent_id_match must be a bool.")


@dataclass(frozen=True, slots=True)
class AgentResolutionRejection:
    """Safe rejection for one registered agent."""

    agent_id: str
    reason_code: AgentResolutionRejectionCode
    message: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "agent_id", _identifier(self.agent_id, "agent_id"))
        object.__setattr__(self, "reason_code", _rejection_code(self.reason_code))
        object.__setattr__(self, "message", _safe_message(self.message))


@dataclass(frozen=True, slots=True)
class AgentResolutionResult:
    """Immutable result for deterministic agent resolution."""

    status: AgentResolutionStatus
    selected_agent: AgentDefinition | None
    candidates: tuple[AgentResolutionCandidate, ...]
    rejections: tuple[AgentResolutionRejection, ...]
    request_signature: str
    selected_agent_id: str | None = None
    scanned_agents: int = 0
    matched_agents: int = 0
    truncated: bool = False
    error_code: str | None = None
    error_message: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", _status(self.status))
        object.__setattr__(self, "candidates", tuple(self.candidates))
        object.__setattr__(self, "rejections", tuple(self.rejections))
        if self.selected_agent is not None and not isinstance(self.selected_agent, AgentDefinition):
            raise InvalidAgentResolutionRequestError("selected_agent must be AgentDefinition or None.")
        if self.selected_agent_id is not None:
            object.__setattr__(self, "selected_agent_id", _identifier(self.selected_agent_id, "selected_agent_id"))


class AgentResolver:
    """Resolve one registered specialized agent without executing anything."""

    def __init__(
        self,
        agent_registry: AgentRegistry,
    ) -> None:
        if not isinstance(agent_registry, AgentRegistry):
            raise AgentResolverError("AgentResolver requires AgentRegistry.")
        self._agent_registry = agent_registry

    def resolve(
        self,
        request: AgentResolutionRequest,
    ) -> AgentResolutionResult:
        """Return a deterministic resolution result from registered definitions."""

        if not isinstance(request, AgentResolutionRequest):
            return _result(
                AgentResolutionStatus.INVALID_REQUEST,
                "",
                error_code="INVALID_REQUEST",
                error_message="request must be AgentResolutionRequest.",
            )
        signature = agent_resolution_request_signature(request)
        try:
            agents = self._agent_registry.list_agents(enabled_only=False)
        except (TypeError, ValueError, RuntimeError):
            return _result(
                AgentResolutionStatus.REGISTRY_UNAVAILABLE,
                signature,
                error_code="REGISTRY_UNAVAILABLE",
                error_message="agent registry is unavailable.",
            )
        if not agents:
            return _result(AgentResolutionStatus.NO_AGENTS, signature, scanned_agents=0)

        rejections: list[AgentResolutionRejection] = []
        candidates: list[AgentResolutionCandidate] = []
        for agent in agents:
            rejection = _rejection(agent, request)
            if rejection is not None:
                rejections.append(rejection)
                continue
            candidates.append(_candidate(agent, request))

        if not candidates:
            return _result(
                AgentResolutionStatus.NO_MATCHING_AGENTS,
                signature,
                rejections=tuple(rejections),
                scanned_agents=len(agents),
            )

        ordered = tuple(sorted(candidates, key=lambda candidate: _candidate_sort_key(candidate, request)))
        considered = ordered[: request.maximum_candidates_considered]
        truncated = len(ordered) > len(considered)
        top = considered[0]
        if top.score < request.minimum_score:
            return _result(
                AgentResolutionStatus.BELOW_MINIMUM_SCORE,
                signature,
                candidates=considered,
                rejections=tuple(rejections),
                scanned_agents=len(agents),
                matched_agents=len(candidates),
                truncated=truncated,
                error_code="BELOW_MINIMUM_SCORE",
                error_message="top candidate score is below minimum_score.",
            )

        top_selection_key = _candidate_selection_key(top, request)
        ambiguous = tuple(
            candidate for candidate in considered if _candidate_selection_key(candidate, request) == top_selection_key
        )
        if request.require_unique_top_score and len(ambiguous) > 1:
            return _result(
                AgentResolutionStatus.AMBIGUOUS,
                signature,
                candidates=considered,
                rejections=tuple(rejections),
                scanned_agents=len(agents),
                matched_agents=len(candidates),
                truncated=truncated,
                error_code="AMBIGUOUS",
                error_message="multiple agents share the top score.",
            )

        return _result(
            AgentResolutionStatus.RESOLVED,
            signature,
            selected_agent=top.agent,
            candidates=considered,
            rejections=tuple(rejections),
            selected_agent_id=top.agent.agent_id,
            scanned_agents=len(agents),
            matched_agents=len(candidates),
            truncated=truncated,
        )


def agent_resolution_request_signature(
    request: AgentResolutionRequest,
) -> str:
    """Return a stable SHA-256 signature for a resolution request."""

    if not isinstance(request, AgentResolutionRequest):
        raise InvalidAgentResolutionRequestError("request must be AgentResolutionRequest.")
    payload = {
        "required_capability_ids": sorted(request.required_capability_ids),
        "preferred_capability_ids": sorted(request.preferred_capability_ids),
        "required_agent_types": sorted(item.value for item in request.required_agent_types),
        "preferred_agent_types": sorted(item.value for item in request.preferred_agent_types),
        "required_permission_ids": sorted(request.required_permission_ids),
        "preferred_permission_ids": sorted(request.preferred_permission_ids),
        "required_agent_ids": sorted(request.required_agent_ids),
        "excluded_agent_ids": sorted(request.excluded_agent_ids),
        "preferred_agent_ids": sorted(request.preferred_agent_ids),
        "enabled_only": request.enabled_only,
        "minimum_score": request.minimum_score,
        "require_unique_top_score": request.require_unique_top_score,
        "maximum_candidates_considered": request.maximum_candidates_considered,
        "metadata": _jsonable(request.metadata),
    }
    encoded = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _rejection(
    agent: AgentDefinition,
    request: AgentResolutionRequest,
) -> AgentResolutionRejection | None:
    if request.enabled_only and not agent.enabled:
        return AgentResolutionRejection(agent.agent_id, AgentResolutionRejectionCode.DISABLED, "agent is disabled")
    if request.required_agent_ids and agent.agent_id not in request.required_agent_ids:
        return AgentResolutionRejection(
            agent.agent_id,
            AgentResolutionRejectionCode.EXCLUDED_AGENT_ID,
            "agent id is not required",
        )
    if agent.agent_id in request.excluded_agent_ids:
        return AgentResolutionRejection(
            agent.agent_id,
            AgentResolutionRejectionCode.EXCLUDED_AGENT_ID,
            "agent id is excluded",
        )
    missing_capabilities = tuple(
        capability for capability in request.required_capability_ids if capability not in agent.capabilities.capabilities
    )
    if missing_capabilities:
        return AgentResolutionRejection(
            agent.agent_id,
            AgentResolutionRejectionCode.REQUIRED_CAPABILITY_MISSING,
            "required capability is missing",
        )
    if request.required_agent_types and agent.agent_type not in request.required_agent_types:
        return AgentResolutionRejection(
            agent.agent_id,
            AgentResolutionRejectionCode.REQUIRED_AGENT_TYPE_MISSING,
            "required agent type is missing",
        )
    missing_permissions = tuple(
        permission for permission in request.required_permission_ids if not _permission_enabled(agent, permission)
    )
    if missing_permissions:
        return AgentResolutionRejection(
            agent.agent_id,
            AgentResolutionRejectionCode.REQUIRED_PERMISSION_MISSING,
            "required permission is missing",
        )
    return None


def _candidate(
    agent: AgentDefinition,
    request: AgentResolutionRequest,
) -> AgentResolutionCandidate:
    reasons: list[AgentResolutionReason] = []
    if agent.agent_id in request.preferred_agent_ids:
        reasons.append(AgentResolutionReason(AgentResolutionReasonCode.PREFERRED_AGENT_ID_MATCH, agent.agent_id, 100))
    for capability in request.required_capability_ids:
        if capability in agent.capabilities.capabilities:
            reasons.append(AgentResolutionReason(AgentResolutionReasonCode.REQUIRED_CAPABILITY_MATCH, capability, 30))
    for capability in request.preferred_capability_ids:
        if capability in agent.capabilities.capabilities:
            reasons.append(AgentResolutionReason(AgentResolutionReasonCode.PREFERRED_CAPABILITY_MATCH, capability, 15))
    if request.required_agent_types and agent.agent_type in request.required_agent_types:
        reasons.append(AgentResolutionReason(AgentResolutionReasonCode.REQUIRED_AGENT_TYPE_MATCH, agent.agent_type.value, 20))
    if request.preferred_agent_types and agent.agent_type in request.preferred_agent_types:
        reasons.append(AgentResolutionReason(AgentResolutionReasonCode.PREFERRED_AGENT_TYPE_MATCH, agent.agent_type.value, 10))
    for permission in request.required_permission_ids:
        if _permission_enabled(agent, permission):
            reasons.append(AgentResolutionReason(AgentResolutionReasonCode.REQUIRED_PERMISSION_MATCH, permission, 10))
    for permission in request.preferred_permission_ids:
        if _permission_enabled(agent, permission):
            reasons.append(AgentResolutionReason(AgentResolutionReasonCode.PREFERRED_PERMISSION_MATCH, permission, 5))
    if agent.enabled:
        reasons.append(AgentResolutionReason(AgentResolutionReasonCode.ENABLED_AGENT, agent.agent_id, 1))
    required_and_preferred = set(request.required_capability_ids).union(request.preferred_capability_ids)
    satisfied_count = len(required_and_preferred.intersection(agent.capabilities.capabilities))
    score = sum(reason.score for reason in reasons)
    return AgentResolutionCandidate(
        agent=agent,
        score=score,
        reasons=tuple(reasons),
        satisfied_capability_count=satisfied_count,
        preferred_agent_id_match=agent.agent_id in request.preferred_agent_ids,
    )


def _candidate_sort_key(
    candidate: AgentResolutionCandidate,
    request: AgentResolutionRequest,
) -> tuple[object, ...]:
    preferred_index = (
        request.preferred_agent_ids.index(candidate.agent.agent_id)
        if candidate.agent.agent_id in request.preferred_agent_ids
        else MAX_AGENT_RESOLUTION_IDS + 1
    )
    preferred_type_index = _preferred_agent_type_index(candidate, request)
    return (
        -candidate.score,
        0 if candidate.agent.enabled else 1,
        0 if candidate.preferred_agent_id_match else 1,
        preferred_index,
        preferred_type_index,
        -candidate.satisfied_capability_count,
        candidate.agent.agent_type.value,
        candidate.agent.agent_id,
    )


def _candidate_selection_key(
    candidate: AgentResolutionCandidate,
    request: AgentResolutionRequest,
) -> tuple[object, ...]:
    preferred_index = (
        request.preferred_agent_ids.index(candidate.agent.agent_id)
        if candidate.agent.agent_id in request.preferred_agent_ids
        else MAX_AGENT_RESOLUTION_IDS + 1
    )
    return (
        candidate.score,
        candidate.agent.enabled,
        candidate.preferred_agent_id_match,
        preferred_index,
        _preferred_agent_type_index(candidate, request),
        candidate.satisfied_capability_count,
    )


def _preferred_agent_type_index(
    candidate: AgentResolutionCandidate,
    request: AgentResolutionRequest,
) -> int:
    if candidate.agent.agent_type in request.preferred_agent_types:
        return request.preferred_agent_types.index(candidate.agent.agent_type)
    return MAX_AGENT_RESOLUTION_IDS + 1


def _permission_enabled(
    agent: AgentDefinition,
    permission_id: str,
) -> bool:
    return bool(getattr(agent.permissions, permission_id))


def _result(
    status: AgentResolutionStatus,
    request_signature: str,
    *,
    selected_agent: AgentDefinition | None = None,
    candidates: tuple[AgentResolutionCandidate, ...] = (),
    rejections: tuple[AgentResolutionRejection, ...] = (),
    selected_agent_id: str | None = None,
    scanned_agents: int = 0,
    matched_agents: int = 0,
    truncated: bool = False,
    error_code: str | None = None,
    error_message: str | None = None,
) -> AgentResolutionResult:
    return AgentResolutionResult(
        status=status,
        selected_agent=selected_agent,
        candidates=candidates,
        rejections=rejections,
        request_signature=request_signature,
        selected_agent_id=selected_agent_id,
        scanned_agents=scanned_agents,
        matched_agents=matched_agents,
        truncated=truncated,
        error_code=error_code,
        error_message=_safe_message(error_message) if error_message is not None else None,
    )


def _identifier_tuple(
    values: Iterable[str],
    field_name: str,
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Iterable):
        raise InvalidAgentResolutionRequestError(f"{field_name} must be an iterable of strings.")
    normalized = tuple(dict.fromkeys(_identifier(value, field_name) for value in values))
    if len(normalized) > MAX_AGENT_RESOLUTION_IDS:
        raise InvalidAgentResolutionRequestError(f"{field_name} exceeds the item limit.")
    return normalized


def _permission_tuple(
    values: Iterable[str],
    field_name: str,
) -> tuple[str, ...]:
    normalized = _identifier_tuple(values, field_name)
    unknown = tuple(value for value in normalized if value not in _PERMISSION_IDS)
    if unknown:
        raise InvalidAgentResolutionRequestError(f"{field_name} contains an unknown permission id.")
    return normalized


def _agent_type_tuple(
    values: Iterable[AgentType | str],
    field_name: str,
) -> tuple[AgentType, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Iterable):
        raise InvalidAgentResolutionRequestError(f"{field_name} must be an iterable.")
    normalized: list[AgentType] = []
    for value in values:
        if isinstance(value, AgentType):
            agent_type = value
        elif isinstance(value, str):
            try:
                agent_type = AgentType(value.strip().lower())
            except ValueError as error:
                raise InvalidAgentResolutionRequestError(f"{field_name} contains an invalid AgentType.") from error
        else:
            raise InvalidAgentResolutionRequestError(f"{field_name} values must be AgentType or str.")
        if agent_type not in normalized:
            normalized.append(agent_type)
    if len(normalized) > MAX_AGENT_RESOLUTION_IDS:
        raise InvalidAgentResolutionRequestError(f"{field_name} exceeds the item limit.")
    return tuple(normalized)


def _identifier(
    value: str,
    field_name: str,
) -> str:
    if not isinstance(value, str):
        raise InvalidAgentResolutionRequestError(f"{field_name} must be a string.")
    if not value or value.strip() != value:
        raise InvalidAgentResolutionRequestError(f"{field_name} cannot be empty or padded.")
    if "/" in value or "\\" in value or ".." in value:
        raise InvalidAgentResolutionRequestError(f"{field_name} cannot be path-like.")
    if any(ord(character) < 32 for character in value):
        raise InvalidAgentResolutionRequestError(f"{field_name} cannot contain control characters.")
    if not all(character.isalnum() or character in "_.-" for character in value):
        raise InvalidAgentResolutionRequestError(f"{field_name} contains unsupported characters.")
    if len(value) > 128:
        raise InvalidAgentResolutionRequestError(f"{field_name} exceeds the length limit.")
    return value


def _safe_metadata(
    metadata: Mapping[str, object],
) -> dict[str, object]:
    if not isinstance(metadata, Mapping):
        raise InvalidAgentResolutionRequestError("metadata must be a mapping.")
    if len(metadata) > MAX_AGENT_RESOLUTION_METADATA_ITEMS:
        raise InvalidAgentResolutionRequestError("metadata has too many items.")
    return _safe_metadata_mapping(metadata, depth=0, counter={"nodes": 0})


def _safe_metadata_mapping(
    metadata: Mapping[str, object],
    *,
    depth: int,
    counter: dict[str, int],
) -> dict[str, object]:
    if depth > MAX_AGENT_RESOLUTION_METADATA_DEPTH:
        raise InvalidAgentResolutionRequestError("metadata is too deep.")
    safe: dict[str, object] = {}
    for key, value in metadata.items():
        if _is_sensitive_key(str(key)):
            raise InvalidAgentResolutionRequestError("metadata cannot contain sensitive keys.")
        safe[_identifier(str(key), "metadata key")] = _safe_metadata_value(value, depth=depth + 1, counter=counter)
    return safe


def _safe_metadata_value(
    value: object,
    *,
    depth: int,
    counter: dict[str, int],
) -> object:
    counter["nodes"] += 1
    if counter["nodes"] > MAX_AGENT_RESOLUTION_METADATA_NODES:
        raise InvalidAgentResolutionRequestError("metadata has too many nodes.")
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            raise InvalidAgentResolutionRequestError("metadata floats must be finite.")
        return value
    if isinstance(value, Mapping):
        return _safe_metadata_mapping(value, depth=depth, counter=counter)
    if isinstance(value, (tuple, list)):
        if len(value) > MAX_AGENT_RESOLUTION_METADATA_ITEMS:
            raise InvalidAgentResolutionRequestError("metadata sequence has too many items.")
        return tuple(_safe_metadata_value(item, depth=depth, counter=counter) for item in value)
    raise InvalidAgentResolutionRequestError("metadata contains unsupported values.")


def _jsonable(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(value[key]) for key in sorted(value)}
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    return value


def _safe_message(message: str) -> str:
    normalized = " ".join(str(message).split())
    for part in _SENSITIVE_KEY_PARTS:
        normalized = normalized.replace(part, "[redacted]")
    return normalized[:240]


def _is_sensitive_key(key: str) -> bool:
    normalized = key.replace("-", "_").lower()
    return any(part in normalized for part in _SENSITIVE_KEY_PARTS)


def _status(value: AgentResolutionStatus | str) -> AgentResolutionStatus:
    if isinstance(value, AgentResolutionStatus):
        return value
    if isinstance(value, str):
        return AgentResolutionStatus(value)
    raise InvalidAgentResolutionRequestError("status must be AgentResolutionStatus.")


def _reason_code(value: AgentResolutionReasonCode | str) -> AgentResolutionReasonCode:
    if isinstance(value, AgentResolutionReasonCode):
        return value
    if isinstance(value, str):
        return AgentResolutionReasonCode(value)
    raise InvalidAgentResolutionRequestError("reason code must be AgentResolutionReasonCode.")


def _rejection_code(value: AgentResolutionRejectionCode | str) -> AgentResolutionRejectionCode:
    if isinstance(value, AgentResolutionRejectionCode):
        return value
    if isinstance(value, str):
        return AgentResolutionRejectionCode(value)
    raise InvalidAgentResolutionRequestError("rejection code must be AgentResolutionRejectionCode.")
