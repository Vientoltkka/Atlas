from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path

import pytest

from core.execution_history import ExecutionHistory, ExecutionHistoryEntry
from core.execution_history_query import (
    ExecutionHistoryQuery,
    ExecutionHistoryQueryService,
)
from core.execution_history_repository import (
    DEFAULT_EXECUTION_HISTORY_MAX_ENTRIES,
    EXECUTION_HISTORY_SCHEMA_VERSION,
    ExecutionHistoryJsonError,
    ExecutionHistoryLoadError,
    ExecutionHistoryPermissionError,
    ExecutionHistoryPersistenceError,
    ExecutionHistorySchemaError,
    ExecutionHistoryValidationError,
    FileExecutionHistoryRepository,
)
from core.execution_metrics import ExecutionMetrics, ExecutionMetricsCalculator
from core.execution_plan_executor import PlanExecutionResult
from core.execution_trace import ExecutionTrace, TraceEventStatus, TraceStatus


def _timestamp(seconds: int, *, offset: timezone = timezone.utc) -> datetime:
    return datetime(2026, 7, 20, 10, 0, seconds, tzinfo=offset)


def _result(
    execution_id: str,
    *,
    status: str = TraceStatus.SUCCESS.value,
    started_at: datetime | None = None,
    duration_ms: int = 100,
    unicode_text: str = "accion",
) -> PlanExecutionResult:
    started = started_at or _timestamp(0)
    trace = ExecutionTrace(execution_id=execution_id, started_at=started)
    trace.add_event(
        timestamp=started,
        component="ExecutionPlanExecutor",
        action="STEP_STARTED",
        status=TraceEventStatus.STARTED.value,
        details={"step_id": "step_1", "unicode": unicode_text},
    )
    terminal_action = (
        "STEP_FINISHED"
        if status == TraceStatus.SUCCESS.value
        else "STEP_FAILED"
    )
    terminal_status = (
        TraceEventStatus.FINISHED.value
        if status == TraceStatus.SUCCESS.value
        else TraceEventStatus.FAILED.value
    )
    trace.add_event(
        timestamp=started + timedelta(milliseconds=duration_ms),
        component="ExecutionPlanExecutor",
        action=terminal_action,
        status=terminal_status,
        duration_ms=duration_ms,
        details={"step_id": "step_1"},
    )
    trace.finish(status, finished_at=started + timedelta(milliseconds=duration_ms))
    return PlanExecutionResult(
        plan_status="completed" if status == TraceStatus.SUCCESS.value else "failed",
        success=status == TraceStatus.SUCCESS.value,
        trace=trace,
        metrics=ExecutionMetricsCalculator().calculate(trace),
    )


def _history(*results: PlanExecutionResult, max_entries: int = 100) -> ExecutionHistory:
    history = ExecutionHistory(max_entries=max_entries)
    for result in results:
        history.add(result)
    return history


def _saved_payload(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _entry_payload(path: Path, index: int = 0) -> dict[str, object]:
    payload = _saved_payload(path)
    entries = payload["entries"]
    assert isinstance(entries, list)
    entry = entries[index]
    assert isinstance(entry, dict)
    return entry


def _write_payload(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, allow_nan=False, sort_keys=True),
        encoding="utf-8",
    )


def test_repository_accepts_path_and_string_paths(tmp_path: Path) -> None:
    path = tmp_path / "history.json"

    path_repository = FileExecutionHistoryRepository(path)
    string_repository = FileExecutionHistoryRepository(str(path))

    assert path_repository.path == path
    assert string_repository.path == path


def test_exists_reports_regular_file_only_and_does_not_create_paths(tmp_path: Path) -> None:
    missing = tmp_path / "missing" / "history.json"
    repository = FileExecutionHistoryRepository(missing)

    assert repository.exists() is False
    assert missing.parent.exists() is False

    missing.parent.mkdir()
    missing.write_text("{}", encoding="utf-8")
    assert repository.exists() is True

    directory_repository = FileExecutionHistoryRepository(tmp_path)
    assert directory_repository.exists() is False


def test_save_and_load_empty_history_with_stored_capacity(tmp_path: Path) -> None:
    path = tmp_path / "history.json"
    repository = FileExecutionHistoryRepository(path)
    history = ExecutionHistory(max_entries=7)

    repository.save(history)
    loaded = repository.load()

    payload = _saved_payload(path)
    assert payload["schema_version"] == EXECUTION_HISTORY_SCHEMA_VERSION
    assert payload["max_entries"] == 7
    assert payload["entries"] == []
    assert loaded.max_entries == 7
    assert loaded.count() == 0


def test_missing_file_loads_empty_history_without_creating_file_or_parent(tmp_path: Path) -> None:
    path = tmp_path / "new" / "history.json"
    repository = FileExecutionHistoryRepository(path)

    loaded = repository.load()
    loaded_override = repository.load(max_entries=3)

    assert loaded.max_entries == DEFAULT_EXECUTION_HISTORY_MAX_ENTRIES
    assert loaded.count() == 0
    assert loaded_override.max_entries == 3
    assert loaded_override.count() == 0
    assert path.exists() is False
    assert path.parent.exists() is False


def test_save_creates_parent_directory_and_loads_one_entry(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "history.json"
    result = _result("exec-1")
    repository = FileExecutionHistoryRepository(path)

    repository.save(_history(result))
    loaded = repository.load()
    entry = loaded.latest()

    assert path.exists()
    assert entry is not None
    assert entry.execution_id == "exec-1"
    assert entry.status == TraceStatus.SUCCESS.value
    assert entry.started_at == result.trace.started_at  # type: ignore[union-attr]
    assert entry.finished_at == result.trace.finished_at  # type: ignore[union-attr]
    assert entry.trace.execution_id == result.trace.execution_id  # type: ignore[union-attr]
    assert entry.metrics == result.metrics
    assert entry.result is None


def test_load_preserves_multiple_entries_chronological_order_unicode_and_timezone(tmp_path: Path) -> None:
    offset = timezone(timedelta(hours=1))
    first = _result(
        "exec-1",
        started_at=_timestamp(0, offset=offset),
        unicode_text="accion tecnica",
    )
    second = _result(
        "exec-2",
        status=TraceStatus.FAILED.value,
        started_at=_timestamp(1, offset=offset),
        duration_ms=250,
        unicode_text="accion con tilde",
    )
    repository = FileExecutionHistoryRepository(tmp_path / "history.json")

    repository.save(_history(first, second, max_entries=5))
    loaded = repository.load()

    assert tuple(entry.execution_id for entry in loaded) == ("exec-1", "exec-2")
    assert loaded.latest() is not None
    assert loaded.latest().status == TraceStatus.FAILED.value  # type: ignore[union-attr]
    assert loaded.get("exec-1").started_at.tzinfo is not None  # type: ignore[union-attr]
    assert loaded.get("exec-1").trace.events[0].details["unicode"] == "accion tecnica"  # type: ignore[union-attr]
    assert "accion tecnica" in repository.path.read_text(encoding="utf-8")


def test_round_trip_replaces_existing_content_with_deterministic_json(tmp_path: Path) -> None:
    path = tmp_path / "history.json"
    repository = FileExecutionHistoryRepository(path)
    history = _history(_result("exec-1"), _result("exec-2"))

    repository.save(history)
    first = path.read_text(encoding="utf-8")
    repository.save(history)
    second = path.read_text(encoding="utf-8")

    assert first == second
    assert json.loads(second)["entries"][1]["execution_id"] == "exec-2"


def test_load_capacity_override_and_recent_entry_trimming(tmp_path: Path) -> None:
    repository = FileExecutionHistoryRepository(tmp_path / "history.json")
    repository.save(
        _history(
            _result("exec-1"),
            _result("exec-2"),
            _result("exec-3"),
            max_entries=10,
        )
    )

    loaded = repository.load(max_entries=2)

    assert loaded.max_entries == 2
    assert tuple(entry.execution_id for entry in loaded) == ("exec-2", "exec-3")


@pytest.mark.parametrize("value", [0, -1])
def test_load_rejects_invalid_capacity_override(tmp_path: Path, value: int) -> None:
    repository = FileExecutionHistoryRepository(tmp_path / "history.json")

    with pytest.raises(ValueError):
        repository.load(max_entries=value)


def test_load_applies_existing_duplicate_policy(tmp_path: Path) -> None:
    path = tmp_path / "history.json"
    repository = FileExecutionHistoryRepository(path)
    repository.save(_history(_result("exec-1", duration_ms=100)))
    payload = _saved_payload(path)
    replacement = _entry_payload(path).copy()
    replacement["metrics"] = dict(replacement["metrics"])
    replacement["metrics"]["total_duration_ms"] = 999
    payload["entries"].append(replacement)  # type: ignore[index, union-attr]
    _write_payload(path, payload)

    loaded = repository.load()

    assert loaded.count() == 1
    assert loaded.latest().metrics.total_duration_ms == 999  # type: ignore[union-attr]


def test_save_uses_temporary_file_and_keeps_final_file_intact_when_replace_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "history.json"
    repository = FileExecutionHistoryRepository(path)
    repository.save(_history(_result("exec-old")))
    original = path.read_text(encoding="utf-8")
    seen_temp_paths: list[Path] = []

    def fail_replace(source: str | os.PathLike[str], destination: str | os.PathLike[str]) -> None:
        seen_temp_paths.append(Path(source))
        raise PermissionError("denied")

    monkeypatch.setattr("core.execution_history_repository.os.replace", fail_replace)

    with pytest.raises(ExecutionHistoryPersistenceError) as error:
        repository.save(_history(_result("exec-new")))

    assert isinstance(error.value.__cause__, PermissionError)
    assert isinstance(error.value, ExecutionHistoryPermissionError)
    assert path.read_text(encoding="utf-8") == original
    assert seen_temp_paths
    assert all(not temporary_path.exists() for temporary_path in seen_temp_paths)


def test_clear_deletes_only_file_and_keeps_parent(tmp_path: Path) -> None:
    path = tmp_path / "history.json"
    sibling = tmp_path / "other.json"
    repository = FileExecutionHistoryRepository(path)
    repository.save(ExecutionHistory())
    sibling.write_text("keep", encoding="utf-8")

    repository.clear()
    repository.clear()

    assert path.exists() is False
    assert tmp_path.exists() is True
    assert sibling.read_text(encoding="utf-8") == "keep"


def test_directory_path_is_rejected_on_save_and_load(tmp_path: Path) -> None:
    repository = FileExecutionHistoryRepository(tmp_path)

    with pytest.raises(ExecutionHistoryPersistenceError):
        repository.save(ExecutionHistory())
    with pytest.raises(ExecutionHistoryLoadError):
        repository.load()


@pytest.mark.parametrize(
    ("payload", "expected_error"),
    [
        ("", ExecutionHistoryJsonError),
        ("{", ExecutionHistoryJsonError),
        ("[]", ExecutionHistorySchemaError),
        ({"entries": [], "max_entries": 100}, ExecutionHistorySchemaError),
        (
            {"schema_version": "2.0", "entries": [], "max_entries": 100},
            ExecutionHistorySchemaError,
        ),
        (
            {"schema_version": EXECUTION_HISTORY_SCHEMA_VERSION, "max_entries": 100},
            ExecutionHistorySchemaError,
        ),
        (
            {
                "schema_version": EXECUTION_HISTORY_SCHEMA_VERSION,
                "entries": {},
                "max_entries": 100,
            },
            ExecutionHistorySchemaError,
        ),
        (
            {
                "schema_version": EXECUTION_HISTORY_SCHEMA_VERSION,
                "entries": [1],
                "max_entries": 100,
            },
            ExecutionHistorySchemaError,
        ),
    ],
)
def test_load_rejects_corrupt_document_shapes(
    tmp_path: Path,
    payload: object,
    expected_error: type[Exception],
) -> None:
    path = tmp_path / "history.json"
    if isinstance(payload, str):
        path.write_text(payload, encoding="utf-8")
    else:
        _write_payload(path, payload)

    with pytest.raises(expected_error):
        FileExecutionHistoryRepository(path).load()


@pytest.mark.parametrize("field", ["trace", "metrics"])
def test_load_rejects_entry_missing_trace_or_metrics(tmp_path: Path, field: str) -> None:
    path = tmp_path / "history.json"
    repository = FileExecutionHistoryRepository(path)
    repository.save(_history(_result("exec-1")))
    payload = _saved_payload(path)
    del payload["entries"][0][field]  # type: ignore[index]
    _write_payload(path, payload)

    with pytest.raises(ExecutionHistorySchemaError):
        repository.load()


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("execution_id", "other"),
        ("status", TraceStatus.FAILED.value),
        ("started_at", "2026-07-20T10:00:09+00:00"),
        ("finished_at", "2026-07-20T10:00:09+00:00"),
    ],
)
def test_load_rejects_entry_inconsistency_with_trace(
    tmp_path: Path,
    field: str,
    replacement: object,
) -> None:
    path = tmp_path / "history.json"
    repository = FileExecutionHistoryRepository(path)
    repository.save(_history(_result("exec-1")))
    payload = _saved_payload(path)
    payload["entries"][0][field] = replacement  # type: ignore[index]
    _write_payload(path, payload)

    with pytest.raises(ExecutionHistoryValidationError):
        repository.load()


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("execution_id", "other"),
        ("execution_status", TraceStatus.FAILED.value),
    ],
)
def test_load_rejects_metrics_inconsistency_with_trace(
    tmp_path: Path,
    field: str,
    replacement: object,
) -> None:
    path = tmp_path / "history.json"
    repository = FileExecutionHistoryRepository(path)
    repository.save(_history(_result("exec-1")))
    payload = _saved_payload(path)
    payload["entries"][0]["metrics"][field] = replacement  # type: ignore[index]
    _write_payload(path, payload)

    with pytest.raises(ExecutionHistoryValidationError):
        repository.load()


def test_load_wraps_invalid_trace_and_metrics_with_exception_chaining(tmp_path: Path) -> None:
    path = tmp_path / "history.json"
    repository = FileExecutionHistoryRepository(path)
    repository.save(_history(_result("exec-1")))
    payload = _saved_payload(path)
    payload["entries"][0]["trace"]["schema_version"] = "bad"  # type: ignore[index]
    _write_payload(path, payload)

    with pytest.raises(ExecutionHistoryValidationError) as trace_error:
        repository.load()

    repository.save(_history(_result("exec-1")))
    payload = _saved_payload(path)
    payload["entries"][0]["metrics"]["success_rate"] = 2.0  # type: ignore[index]
    _write_payload(path, payload)

    with pytest.raises(ExecutionHistoryValidationError) as metrics_error:
        repository.load()

    assert trace_error.value.__cause__ is not None
    assert metrics_error.value.__cause__ is not None


def test_load_is_all_or_nothing_for_corrupt_entry(tmp_path: Path) -> None:
    path = tmp_path / "history.json"
    repository = FileExecutionHistoryRepository(path)
    repository.save(_history(_result("exec-1"), _result("exec-2")))
    payload = _saved_payload(path)
    payload["entries"][1]["execution_id"] = "bad"  # type: ignore[index]
    _write_payload(path, payload)

    with pytest.raises(ExecutionHistoryValidationError):
        repository.load()


def test_read_write_and_clear_system_errors_are_wrapped(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "history.json"
    path.write_text("{}", encoding="utf-8")
    repository = FileExecutionHistoryRepository(path)

    def fail_read_text(self: Path, *args: object, **kwargs: object) -> str:
        raise PermissionError("read denied")

    monkeypatch.setattr(Path, "read_text", fail_read_text)
    with pytest.raises(ExecutionHistoryLoadError) as read_error:
        repository.load()
    assert isinstance(read_error.value.__cause__, PermissionError)
    assert isinstance(read_error.value, ExecutionHistoryPermissionError)

    monkeypatch.undo()

    def fail_named_temporary_file(*args: object, **kwargs: object) -> object:
        raise PermissionError("write denied")

    monkeypatch.setattr(
        "core.execution_history_repository.tempfile.NamedTemporaryFile",
        fail_named_temporary_file,
    )
    with pytest.raises(ExecutionHistoryPersistenceError) as write_error:
        repository.save(ExecutionHistory())
    assert isinstance(write_error.value.__cause__, PermissionError)
    assert isinstance(write_error.value, ExecutionHistoryPermissionError)

    monkeypatch.undo()

    def fail_unlink(self: Path, *args: object, **kwargs: object) -> None:
        raise PermissionError("clear denied")

    monkeypatch.setattr(Path, "unlink", fail_unlink)
    with pytest.raises(ExecutionHistoryPersistenceError) as clear_error:
        repository.clear()
    assert isinstance(clear_error.value.__cause__, PermissionError)
    assert isinstance(clear_error.value, ExecutionHistoryPermissionError)


def test_save_does_not_modify_history_entries_trace_or_metrics(tmp_path: Path) -> None:
    result = _result("exec-1")
    history = _history(result)
    before_entries = tuple(history)
    before_events = tuple(result.trace.events)  # type: ignore[union-attr]
    before_metrics = result.metrics

    FileExecutionHistoryRepository(tmp_path / "history.json").save(history)

    assert tuple(history) == before_entries
    assert tuple(result.trace.events) == before_events  # type: ignore[union-attr]
    assert result.metrics is before_metrics


def test_load_returns_independent_objects_and_history_can_be_reused(tmp_path: Path) -> None:
    path = tmp_path / "history.json"
    repository = FileExecutionHistoryRepository(path)
    repository.save(_history(_result("exec-1")))

    first = repository.load()
    second = repository.load()
    first_entry = first.latest()
    second_entry = second.latest()
    assert first_entry is not None
    assert second_entry is not None
    assert first_entry is not second_entry
    assert first_entry.trace is not second_entry.trace
    assert first_entry.metrics is not second_entry.metrics

    first.add(_result("exec-2"))
    repository.save(first)
    loaded_again = repository.load()

    assert tuple(entry.execution_id for entry in loaded_again) == ("exec-1", "exec-2")


def test_loaded_history_is_queryable_and_accepts_new_results(tmp_path: Path) -> None:
    repository = FileExecutionHistoryRepository(tmp_path / "history.json")
    repository.save(_history(_result("exec-1"), _result("exec-2", status=TraceStatus.FAILED.value)))
    loaded = repository.load()

    query_result = ExecutionHistoryQueryService().query(
        loaded,
        ExecutionHistoryQuery(statuses=(TraceStatus.FAILED,)),
    )
    loaded.add(_result("exec-3"))

    assert tuple(entry.execution_id for entry in query_result.entries) == ("exec-2",)
    assert loaded.latest().execution_id == "exec-3"  # type: ignore[union-attr]


def test_save_rejects_non_finite_metric_values_with_chaining(tmp_path: Path) -> None:
    result = _result("exec-1")
    assert result.metrics is not None
    bad_metrics = replace(result.metrics, average_step_duration_ms=float("nan"))
    assert result.trace is not None
    entry = ExecutionHistoryEntry(
        execution_id=result.trace.execution_id,
        status=result.trace.status,
        started_at=result.trace.started_at,
        finished_at=result.trace.finished_at,
        trace=result.trace,
        metrics=bad_metrics,
        result=None,
    )
    history = ExecutionHistory()
    history.add_entry(entry)

    with pytest.raises(ExecutionHistoryPersistenceError) as error:
        FileExecutionHistoryRepository(tmp_path / "history.json").save(history)

    assert isinstance(error.value.__cause__, TypeError)


def test_repository_does_not_create_files_on_construction_and_uses_no_unsafe_loaders(tmp_path: Path) -> None:
    path = tmp_path / "history.json"
    FileExecutionHistoryRepository(path)

    source = Path("core/execution_history_repository.py").read_text(encoding="utf-8")

    assert path.exists() is False
    assert "pickle" not in source
    assert "eval(" not in source
    assert "exec(" not in source
    assert "ast.literal_eval" not in source
