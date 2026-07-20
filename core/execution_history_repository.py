"""Explicit local persistence for execution history observability data."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
import os
from pathlib import Path
import tempfile

from core.execution_history import ExecutionHistory, ExecutionHistoryEntry
from core.execution_observability_deserializer import (
    ExecutionObservabilityDeserializer,
    ObservabilityDeserializationError,
)
from core.execution_observability_serializer import ExecutionObservabilitySerializer


EXECUTION_HISTORY_SCHEMA_VERSION = "1.0"
DEFAULT_EXECUTION_HISTORY_MAX_ENTRIES = 100


class ExecutionHistoryRepositoryError(Exception):
    """Base error for execution history repository failures."""


class ExecutionHistoryPersistenceError(ExecutionHistoryRepositoryError):
    """Raised when a history file cannot be written or cleared."""


class ExecutionHistoryLoadError(ExecutionHistoryRepositoryError):
    """Raised when a history file cannot be read or parsed."""


class ExecutionHistoryJsonError(ExecutionHistoryLoadError):
    """Raised when a history file does not contain valid JSON."""


class ExecutionHistorySchemaError(ExecutionHistoryLoadError):
    """Raised when the persisted history schema is unsupported or incomplete."""


class ExecutionHistoryValidationError(ExecutionHistoryLoadError):
    """Raised when persisted history data is internally inconsistent."""


class ExecutionHistoryPermissionError(ExecutionHistoryRepositoryError):
    """Raised when filesystem permissions block a repository operation."""


class ExecutionHistoryReadPermissionError(
    ExecutionHistoryLoadError,
    ExecutionHistoryPermissionError,
):
    """Raised when permissions block reading the history file."""


class ExecutionHistoryWritePermissionError(
    ExecutionHistoryPersistenceError,
    ExecutionHistoryPermissionError,
):
    """Raised when permissions block writing or clearing the history file."""


class ExecutionHistoryRepository(ABC):
    """Persistence boundary for execution history snapshots."""

    @abstractmethod
    def save(
        self,
        history: ExecutionHistory,
    ) -> None:
        """Persist the current history snapshot."""

    @abstractmethod
    def load(
        self,
        max_entries: int | None = None,
    ) -> ExecutionHistory:
        """Load a persisted history snapshot."""

    @abstractmethod
    def exists(self) -> bool:
        """Return whether the configured history file currently exists."""

    @abstractmethod
    def clear(self) -> None:
        """Remove the configured history file if present."""


class FileExecutionHistoryRepository(ExecutionHistoryRepository):
    """JSON file repository for explicit local execution history persistence.

    Concurrent writers are not coordinated in this phase. Atomic replacement
    prevents partially written final files, but the last successful replacement
    can prevail when several processes write the same path.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        indent: int | None = 2,
    ) -> None:
        self._path = Path(path)
        self._indent = indent
        self._serializer = ExecutionObservabilitySerializer()
        self._deserializer = ExecutionObservabilityDeserializer()

    @property
    def path(self) -> Path:
        """Return the normalized repository file path."""
        return self._path

    def exists(self) -> bool:
        """Return True only when the configured path exists as a regular file."""
        return self._path.is_file()

    def clear(self) -> None:
        """Delete the configured file without touching the parent directory."""
        try:
            self._path.unlink(missing_ok=True)
        except PermissionError as error:
            raise ExecutionHistoryWritePermissionError(
                f"Insufficient permissions to clear execution history file: {self._path}"
            ) from error
        except OSError as error:
            raise ExecutionHistoryPersistenceError(
                f"Could not clear execution history file: {self._path}"
            ) from error

    def save(
        self,
        history: ExecutionHistory,
    ) -> None:
        """Persist history using a same-directory temporary file and os.replace."""
        if self._path.exists() and self._path.is_dir():
            raise ExecutionHistoryPersistenceError(
                f"Execution history path is a directory: {self._path}"
            )
        payload = self._history_to_dict(history)
        try:
            serialized = self._serializer.to_json(payload, indent=self._indent)
        except TypeError as error:
            raise ExecutionHistoryPersistenceError(
                f"Could not serialize execution history for: {self._path}"
            ) from error

        temporary_path: Path | None = None
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                "w",
                encoding="utf-8",
                dir=self._path.parent,
                prefix=f".{self._path.name}.",
                suffix=".tmp",
                delete=False,
            ) as temporary_file:
                temporary_path = Path(temporary_file.name)
                temporary_file.write(serialized)
                temporary_file.flush()
                os.fsync(temporary_file.fileno())
            os.replace(temporary_path, self._path)
            temporary_path = None
        except PermissionError as error:
            raise ExecutionHistoryWritePermissionError(
                f"Insufficient permissions to write execution history file: {self._path}"
            ) from error
        except OSError as error:
            raise ExecutionHistoryPersistenceError(
                f"Could not write execution history file: {self._path}"
            ) from error
        finally:
            if temporary_path is not None:
                try:
                    temporary_path.unlink(missing_ok=True)
                except OSError:
                    pass

    def load(
        self,
        max_entries: int | None = None,
    ) -> ExecutionHistory:
        """Load the configured file, or return an empty history if it is absent."""
        requested_capacity = _validate_optional_capacity(max_entries)
        if not self._path.exists():
            return ExecutionHistory(
                max_entries=requested_capacity
                or DEFAULT_EXECUTION_HISTORY_MAX_ENTRIES
            )
        if self._path.is_dir():
            raise ExecutionHistoryLoadError(
                f"Execution history path is a directory: {self._path}"
            )

        try:
            content = self._path.read_text(encoding="utf-8")
        except PermissionError as error:
            raise ExecutionHistoryReadPermissionError(
                f"Insufficient permissions to read execution history file: {self._path}"
            ) from error
        except OSError as error:
            raise ExecutionHistoryLoadError(
                f"Could not read execution history file: {self._path}"
            ) from error
        if content == "":
            raise ExecutionHistoryJsonError(
                f"Execution history file is empty: {self._path}"
            )

        try:
            root = self._deserializer.parse_json(content)
        except ObservabilityDeserializationError as error:
            raise ExecutionHistoryJsonError(
                f"Invalid execution history JSON: {self._path}"
            ) from error

        payload = self._history_payload(root)
        stored_capacity = _required_positive_int(payload, "max_entries")
        capacity = requested_capacity or stored_capacity
        entries_payload = _required_list(payload, "entries")
        entries = tuple(
            self._entry_from_dict(_entry_payload(item, index), index)
            for index, item in enumerate(entries_payload)
        )

        history = ExecutionHistory(max_entries=capacity)
        for entry in entries:
            history.add_entry(entry)
        return history

    def load_entries(self) -> tuple[ExecutionHistoryEntry, ...]:
        """Load persisted entries with the stored history capacity policy applied."""
        return tuple(self.load())

    def save_entries(
        self,
        entries: tuple[ExecutionHistoryEntry, ...],
        *,
        max_entries: int = DEFAULT_EXECUTION_HISTORY_MAX_ENTRIES,
    ) -> None:
        """Persist explicit entries through a temporary in-memory history."""
        history = ExecutionHistory(max_entries=max_entries)
        for entry in entries:
            history.add_entry(entry)
        self.save(history)

    def _history_to_dict(
        self,
        history: ExecutionHistory,
    ) -> dict[str, object]:
        return {
            "schema_version": EXECUTION_HISTORY_SCHEMA_VERSION,
            "max_entries": history.max_entries,
            "entries": [
                self._entry_to_dict(entry)
                for entry in history
            ],
        }

    def _entry_to_dict(
        self,
        entry: ExecutionHistoryEntry,
    ) -> dict[str, object]:
        return {
            "execution_id": entry.execution_id,
            "status": entry.status,
            "started_at": entry.started_at.isoformat(),
            "finished_at": (
                entry.finished_at.isoformat()
                if entry.finished_at is not None
                else None
            ),
            "trace": self._serializer.trace_to_dict(entry.trace),
            "metrics": self._serializer.metrics_to_dict(entry.metrics),
        }

    def _history_payload(
        self,
        data: object,
    ) -> dict[str, object]:
        if not isinstance(data, dict):
            raise ExecutionHistorySchemaError(
                f"Execution history root must be an object: {self._path}"
            )
        version = _required_field(data, "schema_version")
        if not isinstance(version, str):
            raise ExecutionHistorySchemaError(
                f"Execution history schema_version must be a string: {self._path}"
            )
        if version != EXECUTION_HISTORY_SCHEMA_VERSION:
            raise ExecutionHistorySchemaError(
                f"Unsupported execution history schema_version: {version}."
            )
        return data

    def _entry_from_dict(
        self,
        payload: dict[str, object],
        index: int,
    ) -> ExecutionHistoryEntry:
        trace_payload = _required_dict(payload, "trace")
        metrics_payload = _required_dict(payload, "metrics")
        try:
            trace = self._deserializer.trace_from_dict(trace_payload)
            metrics = self._deserializer.metrics_from_dict(metrics_payload)
        except ObservabilityDeserializationError as error:
            raise ExecutionHistoryValidationError(
                f"Invalid execution history entry at index {index}: {self._path}"
            ) from error

        execution_id = _required_str(payload, "execution_id")
        status = _required_str(payload, "status")
        started_at = _required_timestamp(payload, "started_at")
        finished_at = _required_optional_timestamp(payload, "finished_at")

        _assert_equal(
            execution_id,
            trace.execution_id,
            "execution_id",
            index,
            self._path,
        )
        _assert_equal(status, trace.status, "status", index, self._path)
        _assert_equal(started_at, trace.started_at, "started_at", index, self._path)
        _assert_equal(
            finished_at,
            trace.finished_at,
            "finished_at",
            index,
            self._path,
        )
        _assert_equal(
            metrics.execution_id,
            trace.execution_id,
            "metrics.execution_id",
            index,
            self._path,
        )
        _assert_equal(
            metrics.execution_status,
            trace.status,
            "metrics.execution_status",
            index,
            self._path,
        )

        return ExecutionHistoryEntry(
            execution_id=execution_id,
            status=status,
            started_at=started_at,
            finished_at=finished_at,
            trace=trace,
            metrics=metrics,
            result=None,
        )


def _validate_optional_capacity(
    value: int | None,
) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("max_entries must be an integer.")
    if value <= 0:
        raise ValueError("max_entries must be greater than zero.")
    return value


def _required_field(
    payload: dict[str, object],
    field: str,
) -> object:
    if field not in payload:
        raise ExecutionHistorySchemaError(f"Missing required field: {field}.")
    return payload[field]


def _required_list(
    payload: dict[str, object],
    field: str,
) -> list[object]:
    value = _required_field(payload, field)
    if not isinstance(value, list):
        raise ExecutionHistorySchemaError(f"{field} must be a list.")
    return value


def _entry_payload(
    value: object,
    index: int,
) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ExecutionHistorySchemaError(f"entries[{index}] must be an object.")
    return value


def _required_dict(
    payload: dict[str, object],
    field: str,
) -> dict[str, object]:
    value = _required_field(payload, field)
    if not isinstance(value, dict):
        raise ExecutionHistorySchemaError(f"{field} must be an object.")
    return value


def _required_positive_int(
    payload: dict[str, object],
    field: str,
) -> int:
    value = _required_field(payload, field)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ExecutionHistorySchemaError(f"{field} must be an integer.")
    if value <= 0:
        raise ExecutionHistorySchemaError(f"{field} must be greater than zero.")
    return value


def _required_str(
    payload: dict[str, object],
    field: str,
) -> str:
    value = _required_field(payload, field)
    if not isinstance(value, str) or not value:
        raise ExecutionHistoryValidationError(f"{field} must be a non-empty string.")
    return value


def _required_timestamp(
    payload: dict[str, object],
    field: str,
) -> datetime:
    value = _required_str(payload, field)
    try:
        return datetime.fromisoformat(value)
    except ValueError as error:
        raise ExecutionHistoryValidationError(
            f"{field} must be a valid ISO 8601 timestamp."
        ) from error


def _required_optional_timestamp(
    payload: dict[str, object],
    field: str,
) -> datetime | None:
    value = _required_field(payload, field)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ExecutionHistoryValidationError(
            f"{field} must be a valid ISO 8601 timestamp or null."
        )
    try:
        return datetime.fromisoformat(value)
    except ValueError as error:
        raise ExecutionHistoryValidationError(
            f"{field} must be a valid ISO 8601 timestamp or null."
        ) from error


def _assert_equal(
    left: object,
    right: object,
    field: str,
    index: int,
    path: Path,
) -> None:
    if left != right:
        raise ExecutionHistoryValidationError(
            f"Inconsistent {field} in execution history entry {index}: {path}"
        )
