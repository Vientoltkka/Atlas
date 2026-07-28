"""Conversation memory."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from datetime import datetime, timezone
from threading import RLock
from typing import Any

from memory.operational import (
    InvalidMemoryEntryError,
    MemoryCategory,
    MemoryEntry,
    MemoryEntryNotFoundError,
    MemoryEvent,
    MemoryPolicy,
    SensitiveMemoryRejectedError,
    contains_secret_material,
    normalize_content,
)


class ConversationMemory:
    """Temporary conversation history plus explicit operational memory entries."""

    def __init__(
        self,
        *,
        policy: MemoryPolicy | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._messages: list[dict[str, str]] = []
        self._entries: dict[str, MemoryEntry] = {}
        self._events: list[MemoryEvent] = []
        self._policy = policy or MemoryPolicy()
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._counter = 0
        self._lock = RLock()

    @property
    def policy(self) -> MemoryPolicy:
        return self._policy

    @property
    def events(self) -> tuple[MemoryEvent, ...]:
        with self._lock:
            return tuple(self._events)

    def add_user(self, text: str) -> None:
        with self._lock:
            self._messages.append(
                {
                    "role": "user",
                    "content": text,
                }
            )

    def add_assistant(self, text: str) -> None:
        with self._lock:
            self._messages.append(
                {
                    "role": "assistant",
                    "content": text,
                }
            )

    def history(self) -> list[dict[str, str]]:
        with self._lock:
            return [dict(message) for message in self._messages]

    def store_entry(
        self,
        content: str,
        *,
        category: MemoryCategory = MemoryCategory.CONVERSATION_NOTE,
        source_request_id: str | None = None,
        user_id: str | None = None,
        conversation_id: str | None = None,
        importance: float = 0.5,
        tags: Sequence[str] = (),
        sensitive: bool = False,
        expires_at: datetime | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> MemoryEntry:
        """Store one explicit entry or return its deterministic duplicate."""
        now = self._clock()
        normalized_category = (
            category
            if isinstance(category, MemoryCategory)
            else MemoryCategory(category)
        )
        if (
            normalized_category is MemoryCategory.USER_PREFERENCE
            and not self._policy.allow_user_preferences
        ):
            raise InvalidMemoryEntryError("User preference storage is disabled.")
        if sensitive and not self._policy.allow_sensitive_storage:
            self._record(
                "sensitive_memory_rejected",
                request_id=source_request_id,
                memory_id=None,
                category=normalized_category,
                operation="store",
                content_length=len(content),
                reason_code="sensitive_storage_disabled",
            )
            raise SensitiveMemoryRejectedError("Sensitive memory storage is disabled.")
        if contains_secret_material(content):
            self._record(
                "sensitive_memory_rejected",
                request_id=source_request_id,
                memory_id=None,
                category=normalized_category,
                operation="store",
                content_length=len(content),
                reason_code="credential_marker",
            )
            raise SensitiveMemoryRejectedError("Credential-like memory content is rejected.")
        with self._lock:
            probe = MemoryEntry(
                memory_id="memory-probe",
                content=content,
                category=normalized_category,
                created_at=now,
                updated_at=now,
                source_request_id=source_request_id,
                user_id=user_id,
                conversation_id=conversation_id,
                importance=importance,
                tags=tuple(tags),
                active=True,
                sensitive=sensitive,
                expires_at=expires_at,
                metadata=metadata or {},
            )
            if self._policy.deduplicate:
                duplicate = self._find_duplicate(probe)
                if duplicate is not None:
                    self._record(
                        "memory_duplicate_detected",
                        request_id=source_request_id,
                        memory_id=duplicate.memory_id,
                        category=duplicate.category,
                        operation="store",
                        content_length=len(duplicate.content),
                        reason_code="normalized_exact_duplicate",
                    )
                    return duplicate
            self._counter += 1
            memory_id = f"memory-{self._counter:06d}"
            entry = replace(probe, memory_id=memory_id)
            self._entries[memory_id] = entry
            self._record(
                "memory_entry_stored",
                request_id=source_request_id,
                memory_id=memory_id,
                category=entry.category,
                operation="store",
                content_length=len(entry.content),
                reason_code="stored",
            )
            return entry

    def get_entry(self, memory_id: str) -> MemoryEntry:
        with self._lock:
            try:
                return self._entries[memory_id]
            except KeyError as error:
                raise MemoryEntryNotFoundError("Memory entry was not found.") from error

    def retrieve_entries(
        self,
        query: str = "",
        *,
        user_id: str | None = None,
        conversation_id: str | None = None,
        tags: Sequence[str] = (),
        categories: Sequence[MemoryCategory] = (),
        include_sensitive: bool = False,
        limit: int | None = None,
    ) -> tuple[MemoryEntry, ...]:
        """Retrieve active, non-expired entries with deterministic exact rules."""
        now = self._clock()
        normalized_query = normalize_content(query)
        query_tokens = frozenset(normalized_query.split())
        required_tags = frozenset(str(tag).strip().casefold() for tag in tags)
        category_filter = frozenset(
            item if isinstance(item, MemoryCategory) else MemoryCategory(item)
            for item in categories
        )
        with self._lock:
            candidates = []
            for entry in self._entries.values():
                if not entry.active or entry.is_expired(now):
                    continue
                if entry.sensitive and not include_sensitive:
                    continue
                if user_id is not None and entry.user_id not in {None, user_id}:
                    continue
                if conversation_id is not None and entry.conversation_id not in {
                    None,
                    conversation_id,
                }:
                    continue
                if required_tags and not required_tags.intersection(entry.tags):
                    continue
                if category_filter and entry.category not in category_filter:
                    continue
                entry_tokens = frozenset(normalize_content(entry.content).split())
                if query_tokens and not query_tokens.intersection(entry_tokens):
                    continue
                candidates.append(entry)
            ordered = tuple(
                sorted(
                    candidates,
                    key=lambda item: (
                        -item.importance,
                        -item.updated_at.timestamp(),
                        item.memory_id,
                    ),
                )
            )
            selected = ordered[: limit or self._policy.max_entries_per_query]
            self._record(
                "memory_entry_retrieved",
                request_id=None,
                memory_id=selected[0].memory_id if len(selected) == 1 else None,
                category=selected[0].category if len(selected) == 1 else None,
                operation="retrieve",
                selected_count=len(selected),
                content_length=sum(len(item.content) for item in selected),
                reason_code="deterministic_query",
            )
            return selected

    def list_entries(
        self,
        *,
        active_only: bool = True,
        include_expired: bool = False,
        include_sensitive: bool = False,
        user_id: str | None = None,
        conversation_id: str | None = None,
        tags: Sequence[str] = (),
        categories: Sequence[MemoryCategory] = (),
        limit: int | None = None,
    ) -> tuple[MemoryEntry, ...]:
        now = self._clock()
        required_tags = frozenset(str(tag).strip().casefold() for tag in tags)
        category_filter = frozenset(
            item if isinstance(item, MemoryCategory) else MemoryCategory(item)
            for item in categories
        )
        with self._lock:
            entries = tuple(
                entry
                for entry in self._entries.values()
                if (entry.active or not active_only)
                and (include_expired or not entry.is_expired(now))
                and (include_sensitive or not entry.sensitive)
                and (user_id is None or entry.user_id in {None, user_id})
                and (
                    conversation_id is None
                    or entry.conversation_id in {None, conversation_id}
                )
                and (
                    not required_tags
                    or bool(required_tags.intersection(entry.tags))
                )
                and (
                    not category_filter
                    or entry.category in category_filter
                )
            )
            return tuple(
                sorted(
                    entries,
                    key=lambda item: (-item.updated_at.timestamp(), item.memory_id),
                )
            )[: limit or self._policy.max_entries_per_query]

    def forget_entry(
        self,
        memory_id: str,
        *,
        source_request_id: str | None = None,
    ) -> MemoryEntry:
        """Soft-delete one exact entry."""
        with self._lock:
            current = self.get_entry(memory_id)
            if not current.active:
                return current
            updated = replace(current, active=False, updated_at=self._clock())
            self._entries[memory_id] = updated
            self._record(
                "memory_entry_forgotten",
                request_id=source_request_id,
                memory_id=memory_id,
                category=updated.category,
                operation="forget",
                content_length=len(updated.content),
                reason_code="soft_deleted",
            )
            return updated

    def update_entry(
        self,
        memory_id: str,
        *,
        content: str | None = None,
        category: MemoryCategory | None = None,
        importance: float | None = None,
        tags: Sequence[str] | None = None,
        sensitive: bool | None = None,
        expires_at: datetime | None = None,
        metadata: Mapping[str, Any] | None = None,
        source_request_id: str | None = None,
    ) -> MemoryEntry:
        """Update one exact entry while preserving its identity."""
        with self._lock:
            current = self.get_entry(memory_id)
            next_sensitive = current.sensitive if sensitive is None else sensitive
            next_content = current.content if content is None else content
            if next_sensitive and not self._policy.allow_sensitive_storage:
                raise SensitiveMemoryRejectedError("Sensitive memory storage is disabled.")
            if contains_secret_material(next_content):
                raise SensitiveMemoryRejectedError("Credential-like memory content is rejected.")
            updated = replace(
                current,
                content=next_content,
                category=current.category if category is None else category,
                updated_at=self._clock(),
                importance=current.importance if importance is None else importance,
                tags=current.tags if tags is None else tuple(tags),
                sensitive=next_sensitive,
                expires_at=current.expires_at if expires_at is None else expires_at,
                metadata=current.metadata if metadata is None else metadata,
            )
            duplicate = self._find_duplicate(updated, exclude_memory_id=memory_id)
            if self._policy.deduplicate and duplicate is not None:
                raise InvalidMemoryEntryError("Update would duplicate an existing memory entry.")
            self._entries[memory_id] = updated
            self._record(
                "memory_entry_updated",
                request_id=source_request_id,
                memory_id=memory_id,
                category=updated.category,
                operation="update",
                content_length=len(updated.content),
                reason_code="updated",
            )
            return updated

    def _find_duplicate(
        self,
        probe: MemoryEntry,
        *,
        exclude_memory_id: str | None = None,
    ) -> MemoryEntry | None:
        normalized = normalize_content(probe.content)
        for entry in sorted(self._entries.values(), key=lambda item: item.memory_id):
            if entry.memory_id == exclude_memory_id or not entry.active:
                continue
            if (
                normalize_content(entry.content) == normalized
                and entry.category is probe.category
                and entry.tags == probe.tags
                and entry.user_id == probe.user_id
                and entry.conversation_id == probe.conversation_id
            ):
                return entry
        return None

    def _record(
        self,
        event_type: str,
        *,
        request_id: str | None,
        memory_id: str | None,
        category: MemoryCategory | None,
        operation: str | None,
        selected_count: int = 0,
        content_length: int = 0,
        truncated: bool = False,
        reason_code: str,
    ) -> None:
        with self._lock:
            self._events.append(
                MemoryEvent(
                    event_type=event_type,
                    request_id=request_id,
                    memory_id=memory_id,
                    category=category,
                    operation=operation,
                    selected_count=selected_count,
                    content_length=content_length,
                    truncated=truncated,
                    reason_code=reason_code,
                    timestamp=self._clock(),
                )
            )
