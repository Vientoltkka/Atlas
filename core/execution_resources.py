"""Deterministic resource selection and execution-budget control."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
import math
from threading import Lock
from types import MappingProxyType
from typing import Any


class ResourceType(str, Enum):
    MODEL = "model"
    TOOL = "tool"
    AGENT = "agent"
    LOCAL_RUNTIME = "local_runtime"
    REMOTE_RUNTIME = "remote_runtime"


class ResourceHealthStatus(str, Enum):
    AVAILABLE = "available"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"
    UNKNOWN = "unknown"


class PrivacyLevel(str, Enum):
    PUBLIC = "public"
    INTERNAL = "internal"
    SENSITIVE = "sensitive"
    RESTRICTED = "restricted"


class OptimizationGoal(str, Enum):
    BALANCED = "balanced"
    MINIMIZE_COST = "minimize_cost"
    MINIMIZE_LATENCY = "minimize_latency"
    MAXIMIZE_QUALITY = "maximize_quality"
    MAXIMIZE_PRIVACY = "maximize_privacy"
    LOCAL_FIRST = "local_first"


class ResourceSelectionReason(str, Enum):
    POLICY_DISABLED = "policy_disabled"
    BEST_SCORE = "best_score"
    QUALITY_DEGRADED_FOR_BUDGET = "quality_degraded_for_budget"
    NO_COMPATIBLE_RESOURCE = "no_compatible_resource"
    BUDGET_EXCEEDED = "budget_exceeded"


class ResourceSelectionError(RuntimeError):
    def __init__(
        self,
        error_code: str,
        message: str,
        *,
        step_id: str | None = None,
        recoverable: bool = False,
        candidates_considered: tuple[str, ...] = (),
    ) -> None:
        super().__init__(message)
        self.error_code = error_code
        self.message = message
        self.step_id = step_id
        self.recoverable = recoverable
        self.candidates_considered = tuple(candidates_considered)


class NoCompatibleResourceError(ResourceSelectionError):
    pass


class ExecutionBudgetExceededError(ResourceSelectionError):
    pass


class ResourceUnavailableError(ResourceSelectionError):
    pass


class ResourceCapacityExceededError(ResourceSelectionError):
    pass


class InvalidResourcePolicyError(ValueError):
    pass


class BudgetReservationError(ResourceSelectionError):
    pass


@dataclass(frozen=True, slots=True)
class ExecutionResourceRequirements:
    required_capabilities: tuple[str, ...] = ()
    preferred_capabilities: tuple[str, ...] = ()
    minimum_quality: int = 0
    maximum_latency_seconds: float | None = None
    maximum_estimated_cost: float | None = None
    local_only: bool = False
    remote_allowed: bool = True
    privacy_level: PrivacyLevel = PrivacyLevel.PUBLIC
    requires_tool_execution: bool = False
    requires_vision: bool = False
    requires_audio: bool = False
    requires_code_execution: bool = False
    requires_long_context: bool = False
    minimum_context_window: int = 0
    preferred_model_ids: tuple[str, ...] = ()
    forbidden_model_ids: tuple[str, ...] = ()
    preferred_provider_ids: tuple[str, ...] = ()
    forbidden_provider_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in (
            "required_capabilities",
            "preferred_capabilities",
            "preferred_model_ids",
            "forbidden_model_ids",
            "preferred_provider_ids",
            "forbidden_provider_ids",
        ):
            object.__setattr__(self, name, _str_tuple(getattr(self, name), name))
        for name in (
            "local_only",
            "remote_allowed",
            "requires_tool_execution",
            "requires_vision",
            "requires_audio",
            "requires_code_execution",
            "requires_long_context",
        ):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be a bool.")
        if self.local_only and self.remote_allowed:
            raise ValueError("local_only and remote_allowed cannot both be true.")
        if self.minimum_quality < 0:
            raise ValueError("minimum_quality cannot be negative.")
        if self.minimum_context_window < 0:
            raise ValueError("minimum_context_window cannot be negative.")
        _non_negative_optional(self.maximum_latency_seconds, "maximum_latency_seconds")
        _non_negative_optional(self.maximum_estimated_cost, "maximum_estimated_cost")
        if set(self.preferred_model_ids) & set(self.forbidden_model_ids):
            raise ValueError("A model cannot be both preferred and forbidden.")
        if set(self.preferred_provider_ids) & set(self.forbidden_provider_ids):
            raise ValueError("A provider cannot be both preferred and forbidden.")
        if not isinstance(self.privacy_level, PrivacyLevel):
            object.__setattr__(self, "privacy_level", PrivacyLevel(self.privacy_level))


@dataclass(frozen=True, slots=True)
class ResourceCandidate:
    resource_id: str
    resource_type: ResourceType
    provider_id: str
    capabilities: tuple[str, ...] = ()
    quality_tier: int = 0
    estimated_cost: float | None = None
    estimated_latency: float | None = None
    context_window: int = 0
    local: bool = True
    available: bool = True
    health_status: ResourceHealthStatus = ResourceHealthStatus.AVAILABLE
    privacy_classification: PrivacyLevel = PrivacyLevel.PUBLIC
    concurrency_limit: int = 1
    metadata_version: int = 1

    def __post_init__(self) -> None:
        if not self.resource_id.strip() or not self.provider_id.strip():
            raise ValueError("resource_id and provider_id must be non-empty.")
        if not isinstance(self.resource_type, ResourceType):
            object.__setattr__(self, "resource_type", ResourceType(self.resource_type))
        if not isinstance(self.health_status, ResourceHealthStatus):
            object.__setattr__(
                self,
                "health_status",
                ResourceHealthStatus(self.health_status),
            )
        if not isinstance(self.privacy_classification, PrivacyLevel):
            object.__setattr__(
                self,
                "privacy_classification",
                PrivacyLevel(self.privacy_classification),
            )
        object.__setattr__(self, "capabilities", _str_tuple(self.capabilities, "capabilities"))
        if self.quality_tier < 0 or self.context_window < 0:
            raise ValueError("quality_tier and context_window cannot be negative.")
        if self.concurrency_limit < 1:
            raise ValueError("concurrency_limit must be greater than zero.")
        _non_negative_optional(self.estimated_cost, "estimated_cost")
        _non_negative_optional(self.estimated_latency, "estimated_latency")


@dataclass(frozen=True, slots=True)
class ExecutionResourceCatalog:
    candidates: tuple[ResourceCandidate, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "candidates", tuple(self.candidates))

    def list_candidates(self) -> tuple[ResourceCandidate, ...]:
        return tuple(
            sorted(
                self.candidates,
                key=lambda item: (item.provider_id, item.resource_id),
            )
        )


@dataclass(frozen=True, slots=True)
class ExecutionResourcePolicy:
    enabled: bool = False
    optimization_goal: OptimizationGoal = OptimizationGoal.BALANCED
    quality_weight: float = 1.0
    cost_weight: float = 1.0
    latency_weight: float = 1.0
    reliability_weight: float = 1.0
    privacy_weight: float = 1.0
    local_preference_weight: float = 1.0
    unknown_cost_penalty: float = 10.0
    degraded_resource_penalty: float = 5.0
    allow_quality_degradation: bool = False
    minimum_acceptable_quality: int = 0
    require_known_cost: bool = False
    require_available_resource: bool = True
    preserve_previous_behavior_on_disable: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.optimization_goal, OptimizationGoal):
            object.__setattr__(
                self,
                "optimization_goal",
                OptimizationGoal(self.optimization_goal),
            )
        for name in (
            "enabled",
            "allow_quality_degradation",
            "require_known_cost",
            "require_available_resource",
            "preserve_previous_behavior_on_disable",
        ):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be a bool.")
        for name in (
            "quality_weight",
            "cost_weight",
            "latency_weight",
            "reliability_weight",
            "privacy_weight",
            "local_preference_weight",
            "unknown_cost_penalty",
            "degraded_resource_penalty",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise InvalidResourcePolicyError(f"{name} must be a finite non-negative number.")
            if not math.isfinite(float(value)) or float(value) < 0:
                raise InvalidResourcePolicyError(f"{name} must be finite and non-negative.")
            object.__setattr__(self, name, float(value))
        if self.minimum_acceptable_quality < 0:
            raise InvalidResourcePolicyError("minimum_acceptable_quality cannot be negative.")


@dataclass(frozen=True, slots=True)
class ExecutionBudget:
    max_total_cost: float | None = None
    max_tokens: int | None = None
    max_duration_seconds: float | None = None
    max_remote_calls: int | None = None
    max_model_calls: int | None = None
    max_tool_calls: int | None = None
    max_replans: int | None = None
    reserved_cost: float = 0.0
    reserved_tokens: int = 0
    currency_code: str = "UNIT"
    hard_limit: bool = True

    def __post_init__(self) -> None:
        for name in ("max_total_cost", "max_duration_seconds"):
            _non_negative_optional(getattr(self, name), name)
        for name in (
            "max_tokens",
            "max_remote_calls",
            "max_model_calls",
            "max_tool_calls",
            "max_replans",
        ):
            _non_negative_int_optional(getattr(self, name), name)
        if self.reserved_cost < 0 or self.reserved_tokens < 0:
            raise ValueError("reserved budget cannot be negative.")
        if type(self.hard_limit) is not bool:
            raise TypeError("hard_limit must be a bool.")


@dataclass(frozen=True, slots=True)
class ExecutionBudgetUsage:
    estimated_cost: float = 0.0
    actual_cost: float | None = None
    estimated_tokens: int = 0
    actual_tokens: int | None = None
    elapsed_duration: float = 0.0
    remote_calls: int = 0
    model_calls: int = 0
    tool_calls: int = 0
    remaining_cost: float | None = None
    remaining_tokens: int | None = None
    exhausted_limits: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.estimated_cost < 0 or self.estimated_tokens < 0 or self.elapsed_duration < 0:
            raise ValueError("usage values cannot be negative.")
        _non_negative_optional(self.actual_cost, "actual_cost")
        _non_negative_int_optional(self.actual_tokens, "actual_tokens")
        object.__setattr__(self, "exhausted_limits", tuple(self.exhausted_limits))


@dataclass(frozen=True, slots=True)
class BudgetReservation:
    reservation_id: str
    step_id: str
    resource_id: str
    estimated_cost: float
    estimated_tokens: int = 0


@dataclass(frozen=True, slots=True)
class ResourceScore:
    resource_id: str
    capability_score: float
    quality_score: float
    cost_score: float
    latency_score: float
    reliability_score: float
    privacy_score: float
    locality_score: float
    availability_penalty: float
    budget_penalty: float
    final_score: float


@dataclass(frozen=True, slots=True)
class ResourceSelectionDecision:
    step_id: str
    selected_resource_id: str | None
    provider_id: str | None
    scores: tuple[ResourceScore, ...]
    rejected_candidate_ids: tuple[str, ...]
    reason: ResourceSelectionReason
    optimization_goal: OptimizationGoal
    degradation_applied: bool = False
    estimated_cost: float | None = None
    estimated_tokens: int | None = None
    budget_snapshot: ExecutionBudgetUsage | None = None
    tie_breaker_used: str | None = None


class ExecutionBudgetManager:
    def __init__(
        self,
        budget: ExecutionBudget | None = None,
        usage: ExecutionBudgetUsage | None = None,
    ) -> None:
        self._budget = budget or ExecutionBudget()
        self._usage = usage or self._usage_with_remaining(ExecutionBudgetUsage())
        self._reservations: dict[str, BudgetReservation] = {}
        self._lock = Lock()
        self._counter = 0

    def snapshot(self) -> ExecutionBudgetUsage:
        with self._lock:
            return self._usage

    def reserve(
        self,
        *,
        step_id: str,
        resource_id: str,
        estimated_cost: float | None,
        estimated_tokens: int | None = None,
    ) -> BudgetReservation:
        cost = float(estimated_cost or 0.0)
        tokens = int(estimated_tokens or 0)
        if cost < 0 or tokens < 0:
            raise BudgetReservationError("INVALID_BUDGET_RESERVATION", "Reservation cannot be negative.", step_id=step_id)
        with self._lock:
            if self._budget.max_total_cost is not None and (
                self._usage.estimated_cost + cost > self._budget.max_total_cost
            ):
                raise ExecutionBudgetExceededError(
                    "EXECUTION_BUDGET_EXCEEDED",
                    "Estimated cost exceeds execution budget.",
                    step_id=step_id,
                    recoverable=False,
                )
            if self._budget.max_tokens is not None and (
                self._usage.estimated_tokens + tokens > self._budget.max_tokens
            ):
                raise ExecutionBudgetExceededError(
                    "EXECUTION_TOKEN_BUDGET_EXCEEDED",
                    "Estimated tokens exceed execution budget.",
                    step_id=step_id,
                    recoverable=False,
                )
            self._counter += 1
            reservation = BudgetReservation(
                reservation_id=f"budget.reservation.{self._counter:06d}",
                step_id=step_id,
                resource_id=resource_id,
                estimated_cost=cost,
                estimated_tokens=tokens,
            )
            self._reservations[reservation.reservation_id] = reservation
            self._usage = self._usage_with_remaining(
                replace(
                    self._usage,
                    estimated_cost=self._usage.estimated_cost + cost,
                    estimated_tokens=self._usage.estimated_tokens + tokens,
                )
            )
            return reservation

    def confirm_consumption(
        self,
        reservation_id: str,
        *,
        actual_cost: float | None = None,
        actual_tokens: int | None = None,
    ) -> ExecutionBudgetUsage:
        with self._lock:
            reservation = self._reservations.pop(reservation_id, None)
            if reservation is None:
                raise BudgetReservationError("BUDGET_RESERVATION_NOT_FOUND", "Budget reservation was not found.")
            actual_cost_value = actual_cost
            actual_tokens_value = actual_tokens
            if actual_cost_value is not None:
                _non_negative_optional(actual_cost_value, "actual_cost")
            if actual_tokens_value is not None:
                _non_negative_int_optional(actual_tokens_value, "actual_tokens")
            estimated_cost = self._usage.estimated_cost
            estimated_tokens = self._usage.estimated_tokens
            if actual_cost_value is not None:
                estimated_cost = max(
                    0.0,
                    estimated_cost - reservation.estimated_cost + actual_cost_value,
                )
            if actual_tokens_value is not None:
                estimated_tokens = max(
                    0,
                    estimated_tokens - reservation.estimated_tokens + actual_tokens_value,
                )
            self._usage = self._usage_with_remaining(
                replace(
                    self._usage,
                    estimated_cost=estimated_cost,
                    estimated_tokens=estimated_tokens,
                    actual_cost=(
                        actual_cost_value
                        if self._usage.actual_cost is None
                        else self._usage.actual_cost + (actual_cost_value or 0.0)
                    )
                    if actual_cost_value is not None
                    else self._usage.actual_cost,
                    actual_tokens=(
                        actual_tokens_value
                        if self._usage.actual_tokens is None
                        else self._usage.actual_tokens + (actual_tokens_value or 0)
                    )
                    if actual_tokens_value is not None
                    else self._usage.actual_tokens,
                )
            )
            return self._usage

    def release(self, reservation_id: str) -> ExecutionBudgetUsage:
        with self._lock:
            reservation = self._reservations.pop(reservation_id, None)
            if reservation is None:
                return self._usage
            self._usage = self._usage_with_remaining(
                replace(
                    self._usage,
                    estimated_cost=max(0.0, self._usage.estimated_cost - reservation.estimated_cost),
                    estimated_tokens=max(0, self._usage.estimated_tokens - reservation.estimated_tokens),
                )
            )
            return self._usage

    def _usage_with_remaining(self, usage: ExecutionBudgetUsage) -> ExecutionBudgetUsage:
        remaining_cost = (
            None
            if self._budget.max_total_cost is None
            else max(0.0, self._budget.max_total_cost - usage.estimated_cost)
        )
        remaining_tokens = (
            None
            if self._budget.max_tokens is None
            else max(0, self._budget.max_tokens - usage.estimated_tokens)
        )
        exhausted = []
        if remaining_cost == 0.0:
            exhausted.append("max_total_cost")
        if remaining_tokens == 0:
            exhausted.append("max_tokens")
        return replace(
            usage,
            remaining_cost=remaining_cost,
            remaining_tokens=remaining_tokens,
            exhausted_limits=tuple(exhausted),
        )


class ExecutionResourceOptimizer:
    def __init__(self, policy: ExecutionResourcePolicy | None = None) -> None:
        self._policy = policy or ExecutionResourcePolicy()
        self._resource_counts: dict[str, int] = {}
        self._lock = Lock()

    @property
    def policy(self) -> ExecutionResourcePolicy:
        return self._policy

    def select(
        self,
        *,
        step_id: str,
        requirements: ExecutionResourceRequirements,
        catalog: ExecutionResourceCatalog,
        budget_usage: ExecutionBudgetUsage | None = None,
    ) -> ResourceSelectionDecision:
        candidates = catalog.list_candidates()
        if not self._policy.enabled:
            selected = next((item for item in candidates if item.available), None)
            return ResourceSelectionDecision(
                step_id=step_id,
                selected_resource_id=selected.resource_id if selected else None,
                provider_id=selected.provider_id if selected else None,
                scores=(),
                rejected_candidate_ids=(),
                reason=ResourceSelectionReason.POLICY_DISABLED,
                optimization_goal=self._policy.optimization_goal,
                estimated_cost=selected.estimated_cost if selected else None,
                budget_snapshot=budget_usage,
            )

        valid, rejected = self._filter_candidates(requirements, candidates, budget_usage)
        if not valid:
            raise NoCompatibleResourceError(
                "NO_COMPATIBLE_RESOURCE",
                "No compatible execution resource is available.",
                step_id=step_id,
                recoverable=True,
                candidates_considered=tuple(item.resource_id for item in candidates),
            )

        scores = tuple(self._score(item, requirements, budget_usage) for item in valid)
        score_by_id = {score.resource_id: score for score in scores}
        ordered = tuple(
            sorted(
                valid,
                key=lambda item: self._sort_key(item, score_by_id[item.resource_id]),
            )
        )
        selected = None
        with self._lock:
            for candidate in ordered:
                count = self._resource_counts.get(candidate.resource_id, 0)
                if count < candidate.concurrency_limit:
                    selected = candidate
                    self._resource_counts[candidate.resource_id] = count + 1
                    break
            if selected is None:
                raise ResourceCapacityExceededError(
                    "RESOURCE_CAPACITY_EXCEEDED",
                    "Resource concurrency capacity has been reached.",
                    step_id=step_id,
                    recoverable=True,
                    candidates_considered=tuple(item.resource_id for item in valid),
                )
        degradation = (
            self._policy.allow_quality_degradation
            and selected.quality_tier < max(item.quality_tier for item in candidates)
        )
        return ResourceSelectionDecision(
            step_id=step_id,
            selected_resource_id=selected.resource_id,
            provider_id=selected.provider_id,
            scores=tuple(score_by_id[item.resource_id] for item in ordered),
            rejected_candidate_ids=tuple(item.resource_id for item in rejected),
            reason=(
                ResourceSelectionReason.QUALITY_DEGRADED_FOR_BUDGET
                if degradation
                else ResourceSelectionReason.BEST_SCORE
            ),
            optimization_goal=self._policy.optimization_goal,
            degradation_applied=degradation,
            estimated_cost=selected.estimated_cost,
            budget_snapshot=budget_usage,
            tie_breaker_used="deterministic_resource_tie" if _has_tie(scores) else None,
        )

    def release(self, resource_id: str | None) -> None:
        if resource_id is None:
            return
        with self._lock:
            current = self._resource_counts.get(resource_id, 0)
            if current <= 1:
                self._resource_counts.pop(resource_id, None)
            else:
                self._resource_counts[resource_id] = current - 1

    def _filter_candidates(
        self,
        requirements: ExecutionResourceRequirements,
        candidates: tuple[ResourceCandidate, ...],
        budget_usage: ExecutionBudgetUsage | None,
    ) -> tuple[tuple[ResourceCandidate, ...], tuple[ResourceCandidate, ...]]:
        valid = []
        rejected = []
        for candidate in candidates:
            if not self._is_candidate_valid(requirements, candidate, budget_usage):
                rejected.append(candidate)
            else:
                valid.append(candidate)
        return tuple(valid), tuple(rejected)

    def _is_candidate_valid(
        self,
        requirements: ExecutionResourceRequirements,
        candidate: ResourceCandidate,
        budget_usage: ExecutionBudgetUsage | None,
    ) -> bool:
        if self._policy.require_available_resource and (
            not candidate.available
            or candidate.health_status is ResourceHealthStatus.UNAVAILABLE
        ):
            return False
        if candidate.health_status is ResourceHealthStatus.UNKNOWN and self._policy.require_available_resource:
            return False
        if self._policy.require_known_cost and candidate.estimated_cost is None:
            return False
        if not set(requirements.required_capabilities).issubset(candidate.capabilities):
            return False
        if requirements.requires_vision and "vision" not in candidate.capabilities:
            return False
        if requirements.requires_audio and "audio" not in candidate.capabilities:
            return False
        if requirements.requires_code_execution and "code_execution" not in candidate.capabilities:
            return False
        if requirements.requires_long_context and candidate.context_window <= 0:
            return False
        if candidate.quality_tier < requirements.minimum_quality:
            return False
        if candidate.context_window < requirements.minimum_context_window:
            return False
        if requirements.local_only and not candidate.local:
            return False
        if not requirements.remote_allowed and not candidate.local:
            return False
        if candidate.resource_id in requirements.forbidden_model_ids:
            return False
        if candidate.provider_id in requirements.forbidden_provider_ids:
            return False
        if not _privacy_allows(candidate.privacy_classification, requirements.privacy_level):
            return False
        if requirements.maximum_estimated_cost is not None and (
            candidate.estimated_cost is None
            or candidate.estimated_cost > requirements.maximum_estimated_cost
        ):
            return False
        if requirements.maximum_latency_seconds is not None and (
            candidate.estimated_latency is None
            or candidate.estimated_latency > requirements.maximum_latency_seconds
        ):
            return False
        if budget_usage is not None and budget_usage.remaining_cost is not None:
            if candidate.estimated_cost is None:
                return False
            if candidate.estimated_cost > budget_usage.remaining_cost:
                if not self._policy.allow_quality_degradation:
                    return False
                if candidate.quality_tier < self._policy.minimum_acceptable_quality:
                    return False
                return False
        return True

    def _score(
        self,
        candidate: ResourceCandidate,
        requirements: ExecutionResourceRequirements,
        budget_usage: ExecutionBudgetUsage | None,
    ) -> ResourceScore:
        preferred = len(set(requirements.preferred_capabilities).intersection(candidate.capabilities))
        capability_score = float(len(requirements.required_capabilities) + preferred)
        quality_score = float(candidate.quality_tier)
        cost_score = 0.0 if candidate.estimated_cost is None else 1.0 / (1.0 + candidate.estimated_cost)
        latency_score = 0.0 if candidate.estimated_latency is None else 1.0 / (1.0 + candidate.estimated_latency)
        reliability_score = 1.0 if candidate.health_status is ResourceHealthStatus.AVAILABLE else 0.5
        privacy_score = float(_privacy_rank(candidate.privacy_classification))
        locality_score = 1.0 if candidate.local else 0.0
        availability_penalty = self._policy.degraded_resource_penalty if candidate.health_status is ResourceHealthStatus.DEGRADED else 0.0
        if candidate.health_status is ResourceHealthStatus.UNKNOWN:
            availability_penalty += self._policy.degraded_resource_penalty
        budget_penalty = self._policy.unknown_cost_penalty if candidate.estimated_cost is None else 0.0
        if budget_usage is not None and budget_usage.remaining_cost is not None and candidate.estimated_cost is not None:
            if budget_usage.remaining_cost <= candidate.estimated_cost:
                budget_penalty += 1.0
        weights = self._goal_weights()
        final = (
            capability_score * 2.0
            + quality_score * weights["quality"]
            + cost_score * weights["cost"]
            + latency_score * weights["latency"]
            + reliability_score * self._policy.reliability_weight
            + privacy_score * weights["privacy"]
            + locality_score * weights["local"]
            - availability_penalty
            - budget_penalty
        )
        if candidate.resource_id in requirements.preferred_model_ids:
            final += 5.0
        if candidate.provider_id in requirements.preferred_provider_ids:
            final += 3.0
        return ResourceScore(
            resource_id=candidate.resource_id,
            capability_score=capability_score,
            quality_score=quality_score,
            cost_score=cost_score,
            latency_score=latency_score,
            reliability_score=reliability_score,
            privacy_score=privacy_score,
            locality_score=locality_score,
            availability_penalty=availability_penalty,
            budget_penalty=budget_penalty,
            final_score=final,
        )

    def _goal_weights(self) -> dict[str, float]:
        weights = {
            "quality": self._policy.quality_weight,
            "cost": self._policy.cost_weight,
            "latency": self._policy.latency_weight,
            "privacy": self._policy.privacy_weight,
            "local": self._policy.local_preference_weight,
        }
        if self._policy.optimization_goal is OptimizationGoal.MINIMIZE_COST:
            weights["cost"] *= 5.0
        elif self._policy.optimization_goal is OptimizationGoal.MINIMIZE_LATENCY:
            weights["latency"] *= 5.0
        elif self._policy.optimization_goal is OptimizationGoal.MAXIMIZE_QUALITY:
            weights["quality"] *= 5.0
        elif self._policy.optimization_goal is OptimizationGoal.MAXIMIZE_PRIVACY:
            weights["privacy"] *= 5.0
        elif self._policy.optimization_goal is OptimizationGoal.LOCAL_FIRST:
            weights["local"] *= 5.0
        return weights

    def _sort_key(
        self,
        candidate: ResourceCandidate,
        score: ResourceScore,
    ) -> tuple[float, float, int, float, float, float, int, str, str]:
        return (
            -score.final_score,
            -score.capability_score,
            -candidate.quality_tier,
            candidate.estimated_cost if candidate.estimated_cost is not None else float("inf"),
            candidate.estimated_latency if candidate.estimated_latency is not None else float("inf"),
            -score.reliability_score,
            0 if candidate.local else 1,
            candidate.provider_id,
            candidate.resource_id,
        )


def _has_tie(scores: tuple[ResourceScore, ...]) -> bool:
    values = [score.final_score for score in scores]
    return len(set(values)) != len(values)


def _privacy_rank(level: PrivacyLevel) -> int:
    return {
        PrivacyLevel.PUBLIC: 0,
        PrivacyLevel.INTERNAL: 1,
        PrivacyLevel.SENSITIVE: 2,
        PrivacyLevel.RESTRICTED: 3,
    }[level]


def _privacy_allows(candidate: PrivacyLevel, required: PrivacyLevel) -> bool:
    return _privacy_rank(candidate) >= _privacy_rank(required)


def _str_tuple(values: Any, name: str) -> tuple[str, ...]:
    if not isinstance(values, tuple):
        values = tuple(values)
    if not all(isinstance(item, str) and item.strip() for item in values):
        raise ValueError(f"{name} must contain non-empty strings.")
    return values


def _non_negative_optional(value: float | int | None, name: str) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite non-negative number or None.")
    if not math.isfinite(float(value)) or float(value) < 0:
        raise ValueError(f"{name} must be finite and non-negative.")


def _non_negative_int_optional(value: int | None, name: str) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer or None.")
