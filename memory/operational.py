"""Immutable operational-memory models and conservative policy."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import re
from types import MappingProxyType
from typing import Any


_SECRET_TERMS = (
    "password",
    "contraseña",
    "contrasena",
    "token",
    "api_key",
    "apikey",
    "secret",
    "credential",
)
_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")


class MemoryCategory(str, Enum):
    USER_PREFERENCE = "user_preference"
    USER_FACT = "user_fact"
    PROJECT_FACT = "project_fact"
    DECISION = "decision"
    TASK_CONTEXT = "task_context"
    EXECUTION_RESULT = "execution_result"
    CONVERSATION_NOTE = "conversation_note"


class OperationalMemoryError(RuntimeError):
    code = "OPERATIONAL_MEMORY_ERROR"


class MemoryEntryNotFoundError(OperationalMemoryError):
    code = "MEMORY_ENTRY_NOT_FOUND"


class AmbiguousMemoryMatchError(OperationalMemoryError):
    code = "AMBIGUOUS_MEMORY_MATCH"


class InvalidMemoryEntryError(ValueError):
    code = "INVALID_MEMORY_ENTRY"


class SensitiveMemoryRejectedError(OperationalMemoryError):
    code = "SENSITIVE_MEMORY_REJECTED"


class MemoryUpdateError(OperationalMemoryError):
    code = "MEMORY_UPDATE_ERROR"


class MemoryForgetError(OperationalMemoryError):
    code = "MEMORY_FORGET_ERROR"


class MemoryContextLimitError(OperationalMemoryError):
    code = "MEMORY_CONTEXT_LIMIT"


@dataclass(frozen=True, slots=True)
class MemoryPolicy:
    automatic_write_enabled: bool = False
    allow_execution_result_storage: bool = False
    allow_decision_storage: bool = False
    allow_user_preferences: bool = True
    allow_sensitive_storage: bool = False
    deduplicate: bool = True
    max_entries_per_query: int = 20
    max_context_characters: int = 4_000
    max_context_entries: int = 8
    recent_entry_limit: int = 8

    def __post_init__(self) -> None:
        for name in (
            "automatic_write_enabled",
            "allow_execution_result_storage",
            "allow_decision_storage",
            "allow_user_preferences",
            "allow_sensitive_storage",
            "deduplicate",
        ):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be a bool.")
        for name in (
            "max_entries_per_query",
            "max_context_characters",
            "max_context_entries",
            "recent_entry_limit",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")


@dataclass(frozen=True, slots=True)
class MemoryEntry:
    memory_id: str
    content: str
    category: MemoryCategory
    created_at: datetime
    updated_at: datetime
    source_request_id: str | None = None
    user_id: str | None = None
    conversation_id: str | None = None
    importance: float = 0.5
    tags: tuple[str, ...] = ()
    active: bool = True
    sensitive: bool = False
    expires_at: datetime | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_id(self.memory_id, "memory_id")
        content = " ".join(self.content.split())
        if not content:
            raise InvalidMemoryEntryError("Memory content cannot be empty.")
        if contains_secret_material(content):
            raise InvalidMemoryEntryError("Memory content contains a credential marker.")
        category = (
            self.category
            if isinstance(self.category, MemoryCategory)
            else MemoryCategory(self.category)
        )
        _require_aware(self.created_at, "created_at")
        _require_aware(self.updated_at, "updated_at")
        if self.updated_at < self.created_at:
            raise InvalidMemoryEntryError("updated_at cannot precede created_at.")
        if self.expires_at is not None:
            _require_aware(self.expires_at, "expires_at")
        for name in ("source_request_id", "user_id", "conversation_id"):
            value = getattr(self, name)
            if value is not None:
                _validate_id(value, name)
        if not isinstance(self.importance, (int, float)) or isinstance(self.importance, bool):
            raise InvalidMemoryEntryError("importance must be a number.")
        if not 0.0 <= float(self.importance) <= 1.0:
            raise InvalidMemoryEntryError("importance must be between 0.0 and 1.0.")
        if type(self.active) is not bool or type(self.sensitive) is not bool:
            raise InvalidMemoryEntryError("active and sensitive must be bool values.")
        tags = tuple(
            sorted(
                {
                    normalize_tag(tag)
                    for tag in self.tags
                    if isinstance(tag, str) and normalize_tag(tag)
                }
            )
        )
        object.__setattr__(self, "content", content)
        object.__setattr__(self, "category", category)
        object.__setattr__(self, "importance", float(self.importance))
        object.__setattr__(self, "tags", tags)
        object.__setattr__(self, "metadata", freeze_safe_mapping(self.metadata))

    def is_expired(self, now: datetime) -> bool:
        _require_aware(now, "now")
        return self.expires_at is not None and self.expires_at <= now


@dataclass(frozen=True, slots=True)
class MemoryEvent:
    event_type: str
    request_id: str | None
    memory_id: str | None
    category: MemoryCategory | None
    operation: str | None
    selected_count: int
    content_length: int
    truncated: bool
    reason_code: str
    timestamp: datetime

    def __post_init__(self) -> None:
        _require_aware(self.timestamp, "timestamp")


def contains_secret_material(content: str) -> bool:
    normalized = content.casefold()
    return any(term in normalized for term in _SECRET_TERMS)


def normalize_content(content: str) -> str:
    return " ".join(content.casefold().split())


def normalize_tag(tag: str) -> str:
    return re.sub(r"[^a-z0-9_.-]+", "-", tag.strip().casefold()).strip("-")


def freeze_safe_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    frozen: dict[str, Any] = {}
    for key, item in sorted(value.items(), key=lambda pair: str(pair[0])):
        if not isinstance(key, str) or not key.strip():
            raise InvalidMemoryEntryError("Memory metadata keys must be non-empty strings.")
        if any(term in key.casefold() for term in _SECRET_TERMS):
            raise InvalidMemoryEntryError("Memory metadata contains a credential key.")
        frozen[key] = _freeze_safe_value(item)
    return MappingProxyType(frozen)


def _freeze_safe_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, datetime):
        _require_aware(value, "metadata datetime")
        return value.isoformat()
    if isinstance(value, Mapping):
        return freeze_safe_mapping(value)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_safe_value(item) for item in value)
    raise InvalidMemoryEntryError("Memory metadata must contain JSON-safe values.")


def _validate_id(value: str, name: str) -> None:
    if not isinstance(value, str) or not _ID_PATTERN.fullmatch(value):
        raise InvalidMemoryEntryError(f"{name} is invalid.")


def _require_aware(value: datetime, name: str) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise InvalidMemoryEntryError(f"{name} must be timezone-aware.")
