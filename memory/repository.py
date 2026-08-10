"""Versioned local persistence for explicit personal preferences."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
import json
import os
from pathlib import Path
import re
from tempfile import NamedTemporaryFile
from typing import Any, Protocol

from memory.operational import MemoryCategory, MemoryEntry


SCHEMA_VERSION = 1
_MAX_FILE_BYTES = 1_000_000
_MEMORY_ID_PATTERN = re.compile(r"^memory-(\d{6})$")
_TOP_LEVEL_KEYS = frozenset({"schema_version", "last_memory_sequence", "entries"})
_ENTRY_KEYS = frozenset(
    {
        "memory_id",
        "content",
        "category",
        "created_at",
        "updated_at",
        "source_request_id",
        "user_id",
        "conversation_id",
        "importance",
        "tags",
        "active",
        "sensitive",
        "expires_at",
        "metadata",
    }
)


class MemoryRepositoryError(RuntimeError):
    """Base error raised by personal-memory persistence."""


class MemoryRepositoryWriteError(MemoryRepositoryError):
    """Raised when personal memory cannot be written atomically."""


class CorruptedMemoryRepositoryError(MemoryRepositoryError):
    """Raised when persisted personal memory is malformed or unauthorized."""


class UnsupportedMemoryRepositoryVersion(MemoryRepositoryError):
    """Raised when persisted personal memory uses an unknown schema version."""


class MemoryEntryRepository(Protocol):
    """Persistence contract for the existing operational MemoryEntry model."""

    def load(self) -> tuple[MemoryEntry, ...]:
        """Load persisted entries, or return an empty tuple when absent."""

    @property
    def last_memory_sequence(self) -> int:
        """Return the highest memory sequence ever persisted."""

    def save(
        self,
        entries: Sequence[MemoryEntry],
        *,
        last_memory_sequence: int,
    ) -> None:
        """Atomically replace the persisted set of entries."""


class FileMemoryEntryRepository:
    """Atomic, versioned JSON repository for explicit user preferences only."""

    def __init__(
        self,
        path: Path,
        *,
        max_file_bytes: int = _MAX_FILE_BYTES,
    ) -> None:
        self._path = path
        self._max_file_bytes = max_file_bytes
        self._last_memory_sequence = 0

    @property
    def path(self) -> Path:
        return self._path

    @property
    def last_memory_sequence(self) -> int:
        return self._last_memory_sequence

    def load(self) -> tuple[MemoryEntry, ...]:
        if not self._path.exists():
            return ()
        try:
            if self._path.stat().st_size > self._max_file_bytes:
                raise CorruptedMemoryRepositoryError(
                    "Personal memory file exceeds the maximum allowed size."
                )
            payload = json.loads(self._path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise CorruptedMemoryRepositoryError(
                "Personal memory file is not valid JSON."
            ) from error
        except OSError as error:
            raise MemoryRepositoryError("Could not load personal memory.") from error
        return self._payload_to_entries(payload)

    def save(
        self,
        entries: Sequence[MemoryEntry],
        *,
        last_memory_sequence: int,
    ) -> None:
        ordered = tuple(sorted(entries, key=lambda item: item.memory_id))
        self._validate_entries(ordered)
        self._validate_last_memory_sequence(ordered, last_memory_sequence)
        payload = {
            "schema_version": SCHEMA_VERSION,
            "last_memory_sequence": last_memory_sequence,
            "entries": [self._entry_to_payload(entry) for entry in ordered],
        }
        try:
            encoded = json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        except (TypeError, ValueError) as error:
            raise CorruptedMemoryRepositoryError(
                "Personal memory is not JSON serializable."
            ) from error
        if len(encoded.encode("utf-8")) > self._max_file_bytes:
            raise CorruptedMemoryRepositoryError(
                "Personal memory exceeds the maximum allowed size."
            )

        temp_path: Path | None = None
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with NamedTemporaryFile(
                "w",
                encoding="utf-8",
                newline="\n",
                dir=self._path.parent,
                prefix=f".{self._path.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temp_path = Path(handle.name)
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_path, self._path)
            self._last_memory_sequence = last_memory_sequence
        except OSError as error:
            if temp_path is not None:
                try:
                    temp_path.unlink(missing_ok=True)
                except OSError:
                    pass
            raise MemoryRepositoryWriteError(
                "Could not save personal memory atomically."
            ) from error

    def _payload_to_entries(self, payload: object) -> tuple[MemoryEntry, ...]:
        if not isinstance(payload, dict) or set(payload) != _TOP_LEVEL_KEYS:
            raise CorruptedMemoryRepositoryError(
                "Personal memory payload has an invalid structure."
            )
        version = payload.get("schema_version")
        if type(version) is not int or version != SCHEMA_VERSION:
            raise UnsupportedMemoryRepositoryVersion(
                f"Unsupported personal memory schema version: {version}."
            )
        raw_entries = payload.get("entries")
        if not isinstance(raw_entries, list):
            raise CorruptedMemoryRepositoryError(
                "Personal memory entries must be a list."
            )
        try:
            entries = tuple(self._entry_from_payload(item) for item in raw_entries)
        except (AttributeError, KeyError, OverflowError, TypeError, ValueError) as error:
            raise CorruptedMemoryRepositoryError(
                "Personal memory contains an invalid entry."
            ) from error
        self._validate_entries(entries)
        last_memory_sequence = payload.get("last_memory_sequence")
        self._validate_last_memory_sequence(entries, last_memory_sequence)
        self._last_memory_sequence = last_memory_sequence
        return tuple(sorted(entries, key=lambda item: item.memory_id))

    @staticmethod
    def _entry_to_payload(entry: MemoryEntry) -> dict[str, Any]:
        return {
            "memory_id": entry.memory_id,
            "content": entry.content,
            "category": entry.category.value,
            "created_at": entry.created_at.isoformat(),
            "updated_at": entry.updated_at.isoformat(),
            "source_request_id": entry.source_request_id,
            "user_id": entry.user_id,
            "conversation_id": entry.conversation_id,
            "importance": entry.importance,
            "tags": list(entry.tags),
            "active": entry.active,
            "sensitive": entry.sensitive,
            "expires_at": (
                None if entry.expires_at is None else entry.expires_at.isoformat()
            ),
            "metadata": _json_value(entry.metadata),
        }

    @staticmethod
    def _entry_from_payload(payload: object) -> MemoryEntry:
        if not isinstance(payload, dict) or set(payload) != _ENTRY_KEYS:
            raise CorruptedMemoryRepositoryError(
                "Personal memory entry has an invalid structure."
            )
        expires_at = payload["expires_at"]
        return MemoryEntry(
            memory_id=payload["memory_id"],
            content=payload["content"],
            category=MemoryCategory(payload["category"]),
            created_at=datetime.fromisoformat(payload["created_at"]),
            updated_at=datetime.fromisoformat(payload["updated_at"]),
            source_request_id=payload["source_request_id"],
            user_id=payload["user_id"],
            conversation_id=payload["conversation_id"],
            importance=payload["importance"],
            tags=tuple(payload["tags"]),
            active=payload["active"],
            sensitive=payload["sensitive"],
            expires_at=(
                None if expires_at is None else datetime.fromisoformat(expires_at)
            ),
            metadata=payload["metadata"],
        )

    @staticmethod
    def _validate_entries(entries: Sequence[MemoryEntry]) -> None:
        memory_ids: set[str] = set()
        for entry in entries:
            if not isinstance(entry, MemoryEntry):
                raise CorruptedMemoryRepositoryError(
                    "Personal memory contains an unsupported value."
                )
            if _MEMORY_ID_PATTERN.fullmatch(entry.memory_id) is None:
                raise CorruptedMemoryRepositoryError(
                    "Personal memory contains an invalid memory_id."
                )
            if entry.memory_id in memory_ids:
                raise CorruptedMemoryRepositoryError(
                    "Personal memory contains duplicate memory_id values."
                )
            if (
                entry.category is not MemoryCategory.USER_PREFERENCE
                or not entry.active
                or entry.sensitive
                or entry.source_request_id is None
            ):
                raise CorruptedMemoryRepositoryError(
                    "Personal memory contains a category that is not persistable."
                )
            memory_ids.add(entry.memory_id)

    @staticmethod
    def _validate_last_memory_sequence(
        entries: Sequence[MemoryEntry],
        last_memory_sequence: object,
    ) -> None:
        if type(last_memory_sequence) is not int or last_memory_sequence < 0:
            raise CorruptedMemoryRepositoryError(
                "Personal memory contains an invalid sequence counter."
            )
        highest_sequence = max(
            (
                int(entry.memory_id.removeprefix("memory-"))
                for entry in entries
            ),
            default=0,
        )
        if highest_sequence > last_memory_sequence:
            raise CorruptedMemoryRepositoryError(
                "Personal memory sequence counter precedes a stored memory_id."
            )


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    return value
