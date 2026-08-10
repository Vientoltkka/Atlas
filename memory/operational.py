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
_PREFERENCE_PART_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_RESTRICTED_PREFERENCE_PARTS = frozenset(
    {
        "anthropometric",
        "anthropometrics",
        "blood_pressure",
        "body_fat",
        "calorie",
        "calories",
        "glucose",
        "health",
        "height",
        "history",
        "log",
        "macro",
        "macros",
        "measurement",
        "measurements",
        "medical",
        "metric",
        "metrics",
        "nutrition_history",
        "record",
        "records",
        "training_history",
        "weight",
    }
)
_RESTRICTED_PREFERENCE_CONTENT = re.compile(
    r"\b(?:peso|weight|altura|height|grasa\s+corporal|body\s+fat|imc|bmi|"
    r"presi[o\u00f3]n\s+arterial|blood\s+pressure|glucosa|glucose|"
    r"colesterol|cholesterol|calor[i\u00ed]as?|calories|kcal|macros?)\b"
    r"[^.\n]{0,24}\d|\d[^.\n]{0,24}\b(?:kg|cm|peso|weight|altura|height|"
    r"grasa\s+corporal|body\s+fat|imc|bmi|presi[o\u00f3]n\s+arterial|"
    r"blood\s+pressure|glucosa|glucose|colesterol|cholesterol|"
    r"calor[i\u00ed]as?|calories|kcal|macros?)\b",
    re.IGNORECASE,
)


class MemoryCategory(str, Enum):
    USER_PREFERENCE = "user_preference"
    USER_PROFILE = "user_profile"
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
    domain: str | None = None
    key: str | None = None
    profile_id: str | None = None
    profile_name: str | None = None
    profile_role: str | None = None
    profile_is_primary: bool = False

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
        domain, key = normalize_preference_locator(self.domain, self.key)
        profile_id = normalize_profile_id(self.profile_id)
        profile_name = normalize_profile_text(
            self.profile_name,
            "profile_name",
            128,
        )
        profile_role = normalize_profile_text(
            self.profile_role,
            "profile_role",
            64,
        )
        if (domain is not None or key is not None) and (
            category is not MemoryCategory.USER_PREFERENCE
        ):
            raise InvalidMemoryEntryError(
                "domain and key are only valid for user preferences."
            )
        if domain is not None and self.user_id is None:
            raise InvalidMemoryEntryError(
                "Structured user preferences require a user_id."
            )
        if type(self.profile_is_primary) is not bool:
            raise InvalidMemoryEntryError("profile_is_primary must be a bool.")
        if category is MemoryCategory.USER_PROFILE:
            if profile_id is None or profile_name is None or self.user_id is None:
                raise InvalidMemoryEntryError(
                    "User profiles require profile_id, profile_name, and user_id."
                )
            if domain is not None or key is not None:
                raise InvalidMemoryEntryError(
                    "User profiles cannot define preference domain or key."
                )
        elif (
            profile_name is not None
            or profile_role is not None
            or self.profile_is_primary
        ):
            raise InvalidMemoryEntryError(
                "Profile attributes are only valid for user profiles."
            )
        if profile_id is not None and category not in {
            MemoryCategory.USER_PROFILE,
            MemoryCategory.USER_PREFERENCE,
        }:
            raise InvalidMemoryEntryError(
                "profile_id is only valid for profiles and user preferences."
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
        object.__setattr__(self, "domain", domain)
        object.__setattr__(self, "key", key)
        object.__setattr__(self, "profile_id", profile_id)
        object.__setattr__(self, "profile_name", profile_name)
        object.__setattr__(self, "profile_role", profile_role)
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


def contains_restricted_profile_data(
    content: str,
    *,
    domain: str | None = None,
    key: str | None = None,
) -> bool:
    """Reject profile metrics and histories that are outside this memory block."""
    locator_parts = {
        part
        for value in (domain, key)
        if value
        for part in re.split(r"[._-]+", value.casefold())
    }
    if locator_parts.intersection(_RESTRICTED_PREFERENCE_PARTS):
        return True
    return _RESTRICTED_PREFERENCE_CONTENT.search(content) is not None


def normalize_preference_locator(
    domain: str | None,
    key: str | None,
) -> tuple[str | None, str | None]:
    if domain is None and key is None:
        return None, None
    if not isinstance(domain, str) or not isinstance(key, str):
        raise InvalidMemoryEntryError(
            "Preference domain and key must be provided together."
        )
    normalized_domain = domain.strip().casefold()
    normalized_key = key.strip().casefold()
    if (
        _PREFERENCE_PART_PATTERN.fullmatch(normalized_domain) is None
        or _PREFERENCE_PART_PATTERN.fullmatch(normalized_key) is None
    ):
        raise InvalidMemoryEntryError("Preference domain or key is invalid.")
    return normalized_domain, normalized_key


def normalize_profile_id(profile_id: str | None) -> str | None:
    if profile_id is None:
        return None
    normalized = profile_id.strip().casefold() if isinstance(profile_id, str) else ""
    if not normalized or _ID_PATTERN.fullmatch(normalized) is None:
        raise InvalidMemoryEntryError("profile_id is invalid.")
    return normalized


def normalize_profile_text(
    value: str | None,
    name: str,
    max_length: int,
) -> str | None:
    if value is None:
        return None
    normalized = " ".join(value.split()) if isinstance(value, str) else ""
    if not normalized or len(normalized) > max_length:
        raise InvalidMemoryEntryError(f"{name} is invalid.")
    if contains_secret_material(normalized):
        raise InvalidMemoryEntryError(f"{name} contains a credential marker.")
    return normalized


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
