"""Typed deterministic acceptance criteria for execution plans."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
import math
from typing import Any, Mapping, Sequence


MAX_ACCEPTANCE_CRITERIA = 32
MAX_CRITERION_PATH_DEPTH = 16
MAX_EXPECTED_VALUE_DEPTH = 8
MAX_EXPECTED_VALUE_ITEMS = 256
MAX_EXPECTED_TEXT_LENGTH = 4096


class AcceptanceCriterionKind(str, Enum):
    """Closed criterion kinds supported by deterministic verification."""

    STEP_COMPLETED = "STEP_COMPLETED"
    OUTPUT_EXISTS = "OUTPUT_EXISTS"
    OUTPUT_EQUALS = "OUTPUT_EQUALS"
    OUTPUT_CONTAINS = "OUTPUT_CONTAINS"
    RESOURCE_EXISTS = "RESOURCE_EXISTS"
    RESOURCE_READABLE = "RESOURCE_READABLE"
    RESOURCE_CONTENT_EQUALS = "RESOURCE_CONTENT_EQUALS"
    EXPECTED_TOOL_USED = "EXPECTED_TOOL_USED"
    EXPECTED_STEP_COUNT = "EXPECTED_STEP_COUNT"
    NO_PENDING_CONFIRMATIONS = "NO_PENDING_CONFIRMATIONS"
    NO_CRITICAL_FAILURES = "NO_CRITICAL_FAILURES"
    USER_CONFIRMATION_REQUIRED = "USER_CONFIRMATION_REQUIRED"


@dataclass(frozen=True, slots=True)
class AcceptanceCriterion:
    """Declarative evidence requirement; it never executes behavior."""

    criterion_id: str
    kind: AcceptanceCriterionKind
    description: str
    required: bool = True
    source_step_id: str | None = None
    source_path: tuple[str | int, ...] = ()
    comparison_step_id: str | None = None
    comparison_path: tuple[str | int, ...] = ()
    expected_value: object | None = None
    resource_path: str | None = None
    tool_name: str | None = None
    expected_count: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "criterion_id",
            _required_text(self.criterion_id, "criterion_id"),
        )
        if not isinstance(self.kind, AcceptanceCriterionKind):
            object.__setattr__(self, "kind", AcceptanceCriterionKind(self.kind))
        object.__setattr__(
            self,
            "description",
            _required_text(self.description, "description"),
        )
        if type(self.required) is not bool:
            raise TypeError("required must be a bool.")
        object.__setattr__(
            self,
            "source_step_id",
            _optional_text(self.source_step_id, "source_step_id"),
        )
        object.__setattr__(
            self,
            "comparison_step_id",
            _optional_text(self.comparison_step_id, "comparison_step_id"),
        )
        object.__setattr__(
            self,
            "resource_path",
            _optional_text(self.resource_path, "resource_path"),
        )
        object.__setattr__(
            self,
            "tool_name",
            _optional_text(self.tool_name, "tool_name"),
        )
        object.__setattr__(
            self,
            "source_path",
            _path(self.source_path, "source_path"),
        )
        object.__setattr__(
            self,
            "comparison_path",
            _path(self.comparison_path, "comparison_path"),
        )
        if (
            self.expected_count is not None
            and (
                isinstance(self.expected_count, bool)
                or not isinstance(self.expected_count, int)
                or self.expected_count < 0
            )
        ):
            raise ValueError("expected_count must be a non-negative integer or None.")
        object.__setattr__(
            self,
            "expected_value",
            _safe_expected_value(self.expected_value),
        )
        _validate_kind_fields(self)


def normalize_acceptance_criteria(
    values: Sequence[AcceptanceCriterion] | None,
) -> tuple[AcceptanceCriterion, ...]:
    """Validate and deduplicate plan criteria at the model boundary."""
    if values is None:
        return ()
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError("acceptance_criteria must be a sequence.")
    if len(values) > MAX_ACCEPTANCE_CRITERIA:
        raise ValueError(
            f"acceptance_criteria cannot exceed {MAX_ACCEPTANCE_CRITERIA} items."
        )
    normalized: list[AcceptanceCriterion] = []
    identifiers: set[str] = set()
    for value in values:
        if not isinstance(value, AcceptanceCriterion):
            raise TypeError(
                "acceptance_criteria must contain AcceptanceCriterion values."
            )
        if value.criterion_id in identifiers:
            raise ValueError(
                f"Duplicate acceptance criterion id: {value.criterion_id}."
            )
        identifiers.add(value.criterion_id)
        normalized.append(value)
    return tuple(normalized)


def _validate_kind_fields(criterion: AcceptanceCriterion) -> None:
    if criterion.kind in {
        AcceptanceCriterionKind.STEP_COMPLETED,
        AcceptanceCriterionKind.OUTPUT_EXISTS,
        AcceptanceCriterionKind.OUTPUT_EQUALS,
        AcceptanceCriterionKind.OUTPUT_CONTAINS,
        AcceptanceCriterionKind.EXPECTED_TOOL_USED,
    } and criterion.source_step_id is None:
        raise ValueError(
            f"{criterion.kind.value} requires source_step_id."
        )
    if (
        criterion.kind is AcceptanceCriterionKind.OUTPUT_EQUALS
        and criterion.comparison_step_id is None
    ):
        raise ValueError("OUTPUT_EQUALS requires comparison_step_id.")
    if criterion.kind in {
        AcceptanceCriterionKind.RESOURCE_EXISTS,
        AcceptanceCriterionKind.RESOURCE_READABLE,
        AcceptanceCriterionKind.RESOURCE_CONTENT_EQUALS,
    } and criterion.resource_path is None:
        raise ValueError(
            f"{criterion.kind.value} requires resource_path."
        )
    if (
        criterion.kind is AcceptanceCriterionKind.RESOURCE_CONTENT_EQUALS
        and criterion.comparison_step_id is None
        and criterion.expected_value is None
    ):
        raise ValueError(
            "RESOURCE_CONTENT_EQUALS requires comparison_step_id or expected_value."
        )
    if (
        criterion.kind is AcceptanceCriterionKind.EXPECTED_TOOL_USED
        and criterion.tool_name is None
    ):
        raise ValueError("EXPECTED_TOOL_USED requires tool_name.")
    if (
        criterion.kind is AcceptanceCriterionKind.EXPECTED_STEP_COUNT
        and criterion.expected_count is None
    ):
        raise ValueError("EXPECTED_STEP_COUNT requires expected_count.")


def acceptance_criterion_to_dict(
    criterion: AcceptanceCriterion,
) -> dict[str, Any]:
    """Serialize one criterion declaration without evaluated evidence."""
    return {
        "criterion_id": criterion.criterion_id,
        "kind": criterion.kind.value,
        "description": criterion.description,
        "required": criterion.required,
        "source_step_id": criterion.source_step_id,
        "source_path": list(criterion.source_path),
        "comparison_step_id": criterion.comparison_step_id,
        "comparison_path": list(criterion.comparison_path),
        "expected_value": deepcopy(criterion.expected_value),
        "resource_path": criterion.resource_path,
        "tool_name": criterion.tool_name,
        "expected_count": criterion.expected_count,
    }


def acceptance_criterion_from_dict(
    payload: Mapping[str, Any],
) -> AcceptanceCriterion:
    """Load one persisted criterion declaration."""
    if not isinstance(payload, Mapping):
        raise ValueError("acceptance criterion must be an object.")
    return AcceptanceCriterion(
        criterion_id=_required_payload_text(payload, "criterion_id"),
        kind=AcceptanceCriterionKind(_required_payload_text(payload, "kind")),
        description=_required_payload_text(payload, "description"),
        required=_payload_bool(payload, "required", default=True),
        source_step_id=_payload_optional_text(payload, "source_step_id"),
        source_path=_payload_path(payload, "source_path"),
        comparison_step_id=_payload_optional_text(
            payload,
            "comparison_step_id",
        ),
        comparison_path=_payload_path(payload, "comparison_path"),
        expected_value=deepcopy(payload.get("expected_value")),
        resource_path=_payload_optional_text(payload, "resource_path"),
        tool_name=_payload_optional_text(payload, "tool_name"),
        expected_count=_payload_optional_int(payload, "expected_count"),
    )


def _required_text(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string.")
    return value.strip()


def _safe_expected_value(
    value: object,
    *,
    depth: int = 0,
    budget: list[int] | None = None,
) -> object:
    if depth > MAX_EXPECTED_VALUE_DEPTH:
        raise ValueError(
            f"expected_value cannot exceed depth {MAX_EXPECTED_VALUE_DEPTH}."
        )
    active_budget = [MAX_EXPECTED_VALUE_ITEMS] if budget is None else budget
    active_budget[0] -= 1
    if active_budget[0] < 0:
        raise ValueError(
            f"expected_value cannot exceed {MAX_EXPECTED_VALUE_ITEMS} items."
        )
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("expected_value floats must be finite.")
        return value
    if isinstance(value, str):
        if len(value) > MAX_EXPECTED_TEXT_LENGTH:
            raise ValueError(
                "expected_value strings exceed the safe length limit."
            )
        return value
    if isinstance(value, (list, tuple)):
        return [
            _safe_expected_value(
                item,
                depth=depth + 1,
                budget=active_budget,
            )
            for item in value
        ]
    if isinstance(value, Mapping):
        copied: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise ValueError(
                    "expected_value mappings require non-empty string keys."
                )
            copied[key] = _safe_expected_value(
                item,
                depth=depth + 1,
                budget=active_budget,
            )
        return copied
    raise TypeError("expected_value must be JSON-serializable structured data.")


def _optional_text(value: object, field_name: str) -> str | None:
    if value is None:
        return None
    return _required_text(value, field_name)


def _path(
    value: Sequence[str | int],
    field_name: str,
) -> tuple[str | int, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{field_name} must be a sequence.")
    if len(value) > MAX_CRITERION_PATH_DEPTH:
        raise ValueError(
            f"{field_name} cannot exceed {MAX_CRITERION_PATH_DEPTH} segments."
        )
    normalized: list[str | int] = []
    for segment in value:
        if isinstance(segment, bool) or not isinstance(segment, (str, int)):
            raise TypeError(f"{field_name} segments must be strings or integers.")
        if isinstance(segment, str) and not segment.strip():
            raise ValueError(f"{field_name} cannot contain empty segments.")
        if isinstance(segment, int) and segment < 0:
            raise ValueError(f"{field_name} integer segments cannot be negative.")
        normalized.append(segment.strip() if isinstance(segment, str) else segment)
    return tuple(normalized)


def _required_payload_text(
    payload: Mapping[str, Any],
    key: str,
) -> str:
    return _required_text(payload.get(key), key)


def _payload_optional_text(
    payload: Mapping[str, Any],
    key: str,
) -> str | None:
    return _optional_text(payload.get(key), key)


def _payload_bool(
    payload: Mapping[str, Any],
    key: str,
    *,
    default: bool,
) -> bool:
    value = payload.get(key, default)
    if type(value) is not bool:
        raise ValueError(f"{key} must be a bool.")
    return value


def _payload_optional_int(
    payload: Mapping[str, Any],
    key: str,
) -> int | None:
    value = payload.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{key} must be an integer or null.")
    return value


def _payload_path(
    payload: Mapping[str, Any],
    key: str,
) -> tuple[str | int, ...]:
    value = payload.get(key, ())
    if not isinstance(value, list):
        raise ValueError(f"{key} must be a list.")
    return _path(value, key)
