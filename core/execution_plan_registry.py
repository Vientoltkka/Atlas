"""In-memory registry for reusable Atlas execution plans."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from types import MappingProxyType
import re
from typing import TYPE_CHECKING, Iterator, Mapping

if TYPE_CHECKING:
    from core.planner import ExecutionPlan


_PLAN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_VERSION_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
_RESERVED_PLAN_IDS = frozenset({"__class__", "__dict__", "__mro__", "__subclasses__"})


class ExecutionPlanRegistryError(RuntimeError):
    """Base error for reusable execution-plan registry operations."""

    def __init__(
        self,
        message: str,
        *,
        plan_id: str | None = None,
        version: str | None = None,
        operation: str,
        code: str,
    ) -> None:
        super().__init__(message)
        self.plan_id = plan_id
        self.version = version
        self.operation = operation
        self.code = code


class InvalidExecutionPlanIdError(ExecutionPlanRegistryError):
    """Raised when a plan id is not a stable registry identifier."""


class InvalidExecutionPlanVersionError(ExecutionPlanRegistryError):
    """Raised when a version is not a stable registry identifier."""


class ExecutionPlanAlreadyRegisteredError(ExecutionPlanRegistryError):
    """Raised when registering an existing (plan_id, version) without replace."""


class ExecutionPlanNotFoundError(ExecutionPlanRegistryError):
    """Raised when a registry reference cannot be resolved."""


class ExecutionPlanRegistryUnavailableError(ExecutionPlanRegistryError):
    """Raised when execution requires a registry but none was injected."""


class InvalidExecutionPlanReferenceError(ExecutionPlanRegistryError):
    """Raised when a registry reference is malformed or ambiguous."""


class RecursiveRegisteredExecutionPlanError(ExecutionPlanRegistryError):
    """Raised when registered plan references recurse in the active branch."""


class RegisteredExecutionPlanSignatureMismatchError(ExecutionPlanRegistryError):
    """Raised when a resumed registered plan no longer matches its checkpoint."""


class RegisteredExecutionPlanValidationError(ExecutionPlanRegistryError):
    """Raised when a registered plan fails structural validation."""


@dataclass(frozen=True, slots=True)
class ExecutionPlanReference:
    """Stable logical reference to a registered ExecutionPlan."""

    plan_id: str
    version: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "plan_id", validate_plan_id(self.plan_id))
        object.__setattr__(self, "version", validate_plan_version(self.version))


@dataclass(frozen=True, slots=True)
class RegisteredExecutionPlan:
    """One immutable entry in the in-memory execution-plan registry."""

    reference: ExecutionPlanReference
    plan: "ExecutionPlan"
    description: str | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        from core.planner import ExecutionPlan

        if not isinstance(self.reference, ExecutionPlanReference):
            raise InvalidExecutionPlanReferenceError(
                "Registered plan reference must be ExecutionPlanReference.",
                plan_id=None,
                version=None,
                operation="register_entry",
                code="INVALID_EXECUTION_PLAN_REFERENCE",
            )
        if not isinstance(self.plan, ExecutionPlan):
            raise RegisteredExecutionPlanValidationError(
                "Registered plan must be an ExecutionPlan.",
                plan_id=self.reference.plan_id,
                version=self.reference.version,
                operation="register_entry",
                code="REGISTERED_EXECUTION_PLAN_VALIDATION_ERROR",
            )
        if self.description is not None and not isinstance(self.description, str):
            raise RegisteredExecutionPlanValidationError(
                "Registered plan description must be a string or null.",
                plan_id=self.reference.plan_id,
                version=self.reference.version,
                operation="register_entry",
                code="REGISTERED_EXECUTION_PLAN_VALIDATION_ERROR",
            )
        object.__setattr__(
            self,
            "metadata",
            MappingProxyType(_safe_metadata(self.metadata)),
        )


class ExecutionPlanRegistry:
    """Explicit local registry for reusable ExecutionPlan instances.

    The registry is in-memory and sequential. It does not perform file IO,
    dynamic loading, locking, or global singleton lookup. Replacing or removing
    entries during execution can affect future retries; an attempt that already
    resolved a reference keeps the resolved plan for that attempt.
    """

    def __init__(self) -> None:
        self._entries: OrderedDict[tuple[str, str | None], RegisteredExecutionPlan] = OrderedDict()

    def register(
        self,
        plan_id: str,
        plan: "ExecutionPlan",
        *,
        version: str | None = None,
        replace: bool = False,
    ) -> RegisteredExecutionPlan:
        return self.register_entry(
            RegisteredExecutionPlan(
                reference=ExecutionPlanReference(plan_id, version),
                plan=plan,
            ),
            replace=replace,
        )

    def register_entry(
        self,
        entry: RegisteredExecutionPlan,
        *,
        replace: bool = False,
    ) -> RegisteredExecutionPlan:
        if not isinstance(entry, RegisteredExecutionPlan):
            raise InvalidExecutionPlanReferenceError(
                "Registry entry must be RegisteredExecutionPlan.",
                plan_id=None,
                version=None,
                operation="register_entry",
                code="INVALID_EXECUTION_PLAN_REFERENCE",
            )
        key = _key(entry.reference)
        if key in self._entries and not replace:
            raise ExecutionPlanAlreadyRegisteredError(
                "Execution plan is already registered.",
                plan_id=entry.reference.plan_id,
                version=entry.reference.version,
                operation="register",
                code="EXECUTION_PLAN_ALREADY_REGISTERED",
            )
        self._entries[key] = entry
        return entry

    def resolve(
        self,
        reference: ExecutionPlanReference,
    ) -> "ExecutionPlan":
        if not isinstance(reference, ExecutionPlanReference):
            raise InvalidExecutionPlanReferenceError(
                "Execution plan reference must be ExecutionPlanReference.",
                plan_id=None,
                version=None,
                operation="resolve",
                code="INVALID_EXECUTION_PLAN_REFERENCE",
            )
        entry = self._entries.get(_key(reference))
        if entry is None:
            raise ExecutionPlanNotFoundError(
                "Execution plan reference was not found.",
                plan_id=reference.plan_id,
                version=reference.version,
                operation="resolve",
                code="EXECUTION_PLAN_NOT_FOUND",
            )
        return entry.plan

    def get(
        self,
        plan_id: str,
        *,
        version: str | None = None,
    ) -> "ExecutionPlan":
        return self.resolve(ExecutionPlanReference(plan_id, version))

    def contains(
        self,
        plan_id: str,
        *,
        version: str | None = None,
    ) -> bool:
        return _key(ExecutionPlanReference(plan_id, version)) in self._entries

    def unregister(
        self,
        plan_id: str,
        *,
        version: str | None = None,
    ) -> bool:
        key = _key(ExecutionPlanReference(plan_id, version))
        if key not in self._entries:
            return False
        del self._entries[key]
        return True

    def list_entries(self) -> tuple[RegisteredExecutionPlan, ...]:
        return tuple(self._entries.values())

    def clear(self) -> None:
        self._entries.clear()

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self) -> Iterator[RegisteredExecutionPlan]:
        return iter(self.list_entries())


def validate_plan_id(value: str) -> str:
    if not isinstance(value, str):
        raise InvalidExecutionPlanIdError(
            "Execution plan id must be a string.",
            plan_id=None,
            version=None,
            operation="validate_plan_id",
            code="INVALID_EXECUTION_PLAN_ID",
        )
    return _validate_identifier(
        value,
        pattern=_PLAN_ID_PATTERN,
        label="Execution plan id",
        error_type=InvalidExecutionPlanIdError,
        operation="validate_plan_id",
        code="INVALID_EXECUTION_PLAN_ID",
    )


def validate_plan_version(value: str | None) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise InvalidExecutionPlanVersionError(
            "Execution plan version must be a string or null.",
            plan_id=None,
            version=None,
            operation="validate_plan_version",
            code="INVALID_EXECUTION_PLAN_VERSION",
        )
    return _validate_identifier(
        value,
        pattern=_VERSION_PATTERN,
        label="Execution plan version",
        error_type=InvalidExecutionPlanVersionError,
        operation="validate_plan_version",
        code="INVALID_EXECUTION_PLAN_VERSION",
    )


def _validate_identifier(
    value: str,
    *,
    pattern: re.Pattern[str],
    label: str,
    error_type: type[ExecutionPlanRegistryError],
    operation: str,
    code: str,
) -> str:
    if value != value.strip():
        raise error_type(
            f"{label} cannot contain leading or trailing whitespace.",
            plan_id=value,
            version=None,
            operation=operation,
            code=code,
        )
    if not value:
        raise error_type(
            f"{label} cannot be empty.",
            plan_id=value,
            version=None,
            operation=operation,
            code=code,
        )
    if any(ord(character) < 32 for character in value):
        raise error_type(
            f"{label} cannot contain control characters.",
            plan_id=value,
            version=None,
            operation=operation,
            code=code,
        )
    if "/" in value or "\\" in value or ":" in value or ".." in value:
        raise error_type(
            f"{label} cannot be a path.",
            plan_id=value,
            version=None,
            operation=operation,
            code=code,
        )
    if value in _RESERVED_PLAN_IDS or value.startswith("__"):
        raise error_type(
            f"{label} is reserved.",
            plan_id=value,
            version=None,
            operation=operation,
            code=code,
        )
    if pattern.fullmatch(value) is None:
        raise error_type(
            f"{label} has unsupported characters.",
            plan_id=value,
            version=None,
            operation=operation,
            code=code,
        )
    return value


def _safe_metadata(metadata: Mapping[str, object]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in metadata.items():
        if not isinstance(key, str) or not key.strip():
            raise RegisteredExecutionPlanValidationError(
                "Registered plan metadata keys must be non-empty strings.",
                plan_id=None,
                version=None,
                operation="validate_metadata",
                code="REGISTERED_EXECUTION_PLAN_VALIDATION_ERROR",
            )
        result[key] = _safe_metadata_value(value)
    return result


def _safe_metadata_value(value: object) -> object:
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        if value == float("inf") or value == float("-inf") or value != value:
            raise RegisteredExecutionPlanValidationError(
                "Registered plan metadata cannot contain non-finite floats.",
                plan_id=None,
                version=None,
                operation="validate_metadata",
                code="REGISTERED_EXECUTION_PLAN_VALIDATION_ERROR",
            )
        return value
    if isinstance(value, tuple):
        return tuple(_safe_metadata_value(item) for item in value)
    if isinstance(value, list):
        return tuple(_safe_metadata_value(item) for item in value)
    if isinstance(value, Mapping):
        return MappingProxyType(_safe_metadata(value))
    raise RegisteredExecutionPlanValidationError(
        "Registered plan metadata contains an unsupported value.",
        plan_id=None,
        version=None,
        operation="validate_metadata",
        code="REGISTERED_EXECUTION_PLAN_VALIDATION_ERROR",
    )


def _key(reference: ExecutionPlanReference) -> tuple[str, str | None]:
    return (reference.plan_id, reference.version)
