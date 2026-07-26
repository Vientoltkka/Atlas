"""Deterministic resolver for reusable Atlas skills."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
from types import MappingProxyType

from core.agent_registry import AgentType
from core.skill_registry import SkillDefinition, SkillRegistry, SkillNotFoundError, validate_skill_id


class SkillResolutionStatus(str, Enum):
    """Terminal status for skill resolution."""

    RESOLVED = "RESOLVED"
    NO_MATCHING_SKILL = "NO_MATCHING_SKILL"
    AMBIGUOUS = "AMBIGUOUS"
    INVALID_REQUEST = "INVALID_REQUEST"


class SkillRejectionCode(str, Enum):
    """Reasons for skill rejection."""

    DISABLED = "DISABLED"
    EXCLUDED = "EXCLUDED"
    REQUIRED_SKILL_MISSING = "REQUIRED_SKILL_MISSING"
    CAPABILITY_MISSING = "CAPABILITY_MISSING"
    TAG_MISSING = "TAG_MISSING"
    AGENT_TYPE_INCOMPATIBLE = "AGENT_TYPE_INCOMPATIBLE"


@dataclass(frozen=True, slots=True)
class SkillResolutionRequest:
    """Declarative skill selection request."""

    required_skill_ids: tuple[str, ...] = ()
    required_capability_ids: tuple[str, ...] = ()
    preferred_capability_ids: tuple[str, ...] = ()
    required_tags: tuple[str, ...] = ()
    preferred_tags: tuple[str, ...] = ()
    required_agent_types: tuple[AgentType | str, ...] = ()
    preferred_agent_types: tuple[AgentType | str, ...] = ()
    excluded_skill_ids: tuple[str, ...] = ()
    enabled_only: bool = True
    require_unique_top_score: bool = True
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "required_skill_ids", _skill_ids(self.required_skill_ids))
        object.__setattr__(self, "excluded_skill_ids", tuple(sorted(_skill_ids(self.excluded_skill_ids))))
        for name in ("required_capability_ids", "preferred_capability_ids", "required_tags", "preferred_tags"):
            object.__setattr__(self, name, tuple(sorted(_ids(getattr(self, name)))))
        object.__setattr__(self, "required_agent_types", _agent_types(self.required_agent_types))
        object.__setattr__(self, "preferred_agent_types", _agent_types(self.preferred_agent_types))
        if not isinstance(self.enabled_only, bool) or not isinstance(self.require_unique_top_score, bool):
            raise ValueError("boolean flags are invalid.")
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@dataclass(frozen=True, slots=True)
class SkillResolutionCandidate:
    """Accepted skill candidate."""

    skill: SkillDefinition
    score: int
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class SkillResolutionRejection:
    """Rejected skill summary."""

    skill_id: str
    reason_code: SkillRejectionCode
    message: str


@dataclass(frozen=True, slots=True)
class SkillResolutionResult:
    """Structured skill resolution result."""

    status: SkillResolutionStatus
    selected_skill: SkillDefinition | None = None
    candidates: tuple[SkillResolutionCandidate, ...] = ()
    rejections: tuple[SkillResolutionRejection, ...] = ()
    request_signature: str = ""
    selected_skill_id: str | None = None
    error_code: str | None = None
    safe_message: str | None = None
    events: tuple[Mapping[str, object], ...] = ()
    metrics: Mapping[str, int] = field(default_factory=dict)


class SkillResolver:
    """Resolve one skill without executing it."""

    def __init__(self, registry: SkillRegistry) -> None:
        self._registry = registry

    def resolve(self, request: SkillResolutionRequest) -> SkillResolutionResult:
        if not isinstance(request, SkillResolutionRequest):
            return _result(SkillResolutionStatus.INVALID_REQUEST, "", error_code="INVALID_REQUEST")
        signature = skill_resolution_request_signature(request)
        rejections: list[SkillResolutionRejection] = []
        try:
            for skill_id in request.required_skill_ids:
                self._registry.get(skill_id)
        except SkillNotFoundError as error:
            return _result(
                SkillResolutionStatus.NO_MATCHING_SKILL,
                signature,
                rejections=(SkillResolutionRejection(str(error).split(":")[-1].strip(), SkillRejectionCode.REQUIRED_SKILL_MISSING, "required skill is not registered"),),
                error_code="REQUIRED_SKILL_MISSING",
            )
        pool = tuple(self._registry.get(skill_id) for skill_id in request.required_skill_ids) if request.required_skill_ids else self._registry.list_skills(enabled_only=False)
        candidates: list[SkillResolutionCandidate] = []
        for skill in pool:
            rejection = _rejection(skill, request)
            if rejection is not None:
                rejections.append(rejection)
                continue
            candidates.append(_candidate(skill, request))
        if not candidates:
            return _result(SkillResolutionStatus.NO_MATCHING_SKILL, signature, rejections=tuple(rejections), error_code="NO_MATCHING_SKILL")
        ordered = tuple(sorted(candidates, key=lambda candidate: (-candidate.score, candidate.skill.skill_id)))
        top = ordered[0]
        ambiguous = tuple(candidate for candidate in ordered if candidate.score == top.score)
        if request.require_unique_top_score and len(ambiguous) > 1:
            return _result(SkillResolutionStatus.AMBIGUOUS, signature, candidates=ordered, rejections=tuple(rejections), error_code="AMBIGUOUS")
        return _result(
            SkillResolutionStatus.RESOLVED,
            signature,
            selected_skill=top.skill,
            candidates=ordered,
            rejections=tuple(rejections),
            selected_skill_id=top.skill.skill_id,
        )


def skill_resolution_request_signature(request: SkillResolutionRequest) -> str:
    payload = {
        "required_skill_ids": request.required_skill_ids,
        "required_capability_ids": request.required_capability_ids,
        "preferred_capability_ids": request.preferred_capability_ids,
        "required_tags": request.required_tags,
        "preferred_tags": request.preferred_tags,
        "required_agent_types": tuple(agent_type.value for agent_type in request.required_agent_types),
        "preferred_agent_types": tuple(agent_type.value for agent_type in request.preferred_agent_types),
        "excluded_skill_ids": request.excluded_skill_ids,
        "enabled_only": request.enabled_only,
        "require_unique_top_score": request.require_unique_top_score,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _rejection(skill: SkillDefinition, request: SkillResolutionRequest) -> SkillResolutionRejection | None:
    if request.enabled_only and not skill.enabled:
        return SkillResolutionRejection(skill.skill_id, SkillRejectionCode.DISABLED, "skill is disabled")
    if skill.skill_id in request.excluded_skill_ids:
        return SkillResolutionRejection(skill.skill_id, SkillRejectionCode.EXCLUDED, "skill is excluded")
    if any(capability not in skill.required_capability_ids for capability in request.required_capability_ids):
        return SkillResolutionRejection(skill.skill_id, SkillRejectionCode.CAPABILITY_MISSING, "required capability is missing")
    if any(tag not in skill.tags for tag in request.required_tags):
        return SkillResolutionRejection(skill.skill_id, SkillRejectionCode.TAG_MISSING, "required tag is missing")
    if request.required_agent_types and not set(request.required_agent_types).intersection(skill.allowed_agent_types):
        return SkillResolutionRejection(skill.skill_id, SkillRejectionCode.AGENT_TYPE_INCOMPATIBLE, "agent type is incompatible")
    return None


def _candidate(skill: SkillDefinition, request: SkillResolutionRequest) -> SkillResolutionCandidate:
    score = 1
    reasons: list[str] = ["enabled_skill"] if skill.enabled else []
    if skill.skill_id in request.required_skill_ids:
        score += 100
        reasons.append("required_skill_id")
    for capability in request.required_capability_ids:
        score += 30
        reasons.append(f"required_capability:{capability}")
    for capability in request.preferred_capability_ids:
        if capability in skill.required_capability_ids:
            score += 15
            reasons.append(f"preferred_capability:{capability}")
    for tag in request.required_tags:
        score += 20
        reasons.append(f"required_tag:{tag}")
    for tag in request.preferred_tags:
        if tag in skill.tags:
            score += 10
            reasons.append(f"preferred_tag:{tag}")
    for agent_type in request.preferred_agent_types:
        if agent_type in skill.allowed_agent_types:
            score += 5
            reasons.append(f"preferred_agent_type:{agent_type.value}")
    return SkillResolutionCandidate(skill, score, tuple(reasons))


def _result(
    status: SkillResolutionStatus,
    signature: str,
    *,
    selected_skill: SkillDefinition | None = None,
    candidates: tuple[SkillResolutionCandidate, ...] = (),
    rejections: tuple[SkillResolutionRejection, ...] = (),
    selected_skill_id: str | None = None,
    error_code: str | None = None,
) -> SkillResolutionResult:
    return SkillResolutionResult(
        status,
        selected_skill,
        candidates,
        rejections,
        signature,
        selected_skill_id,
        error_code,
        events=(
            {"name": "skill_resolution_started", "status": "STARTED"},
            {
                "name": "skill_resolved" if status is SkillResolutionStatus.RESOLVED else "skill_resolution_failed",
                "status": status.value,
            },
        ),
        metrics={
            "skill_resolutions_requested": 1 if signature else 0,
            "skill_resolutions_succeeded": 1 if status is SkillResolutionStatus.RESOLVED else 0,
            "skill_resolutions_failed": 0 if status is SkillResolutionStatus.RESOLVED else 1,
        },
    )


def _skill_ids(values: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(validate_skill_id(value) for value in values))


def _ids(values: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(validate_skill_id(value) for value in values))


def _agent_types(values: tuple[AgentType | str, ...]) -> tuple[AgentType, ...]:
    normalized: list[AgentType] = []
    for value in values:
        agent_type = value if isinstance(value, AgentType) else AgentType(str(value).strip().lower())
        if agent_type not in normalized:
            normalized.append(agent_type)
    return tuple(normalized)
