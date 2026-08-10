from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import pytest

from bootstrap.bootstrap import Bootstrap, _personal_memory_path
from core.operational_route_executor import RouteExecutionStatus
from core.request_gateway import RequestSafetyContext
from memory.conversation import ConversationMemory
from memory.operational import MemoryCategory
from memory.repository import (
    CorruptedMemoryRepositoryError,
    FileMemoryEntryRepository,
    MemoryRepositoryWriteError,
    UnsupportedMemoryRepositoryVersion,
)
import memory.repository as memory_repository


NOW = datetime(2026, 8, 10, 12, 0, tzinfo=timezone.utc)


def _memory(path: Path, *, clock=lambda: NOW) -> ConversationMemory:
    return ConversationMemory(
        clock=clock,
        repository=FileMemoryEntryRepository(path),
    )


def _store_preference(
    memory: ConversationMemory,
    content: str,
    *,
    request_id: str,
    user_id: str = "user-a",
):
    return memory.store_entry(
        content,
        category=MemoryCategory.USER_PREFERENCE,
        source_request_id=request_id,
        user_id=user_id,
    )


def test_missing_file_is_empty_and_is_not_created(tmp_path: Path) -> None:
    path = tmp_path / "personal_memory.json"

    memory = _memory(path)

    assert memory.list_entries() == ()
    assert not path.exists()


def test_store_then_reconstruct_recovers_explicit_preference(tmp_path: Path) -> None:
    path = tmp_path / "personal_memory.json"
    first = _memory(path)
    stored = _store_preference(
        first,
        "prefiero respuestas breves",
        request_id="request-1",
    )

    restarted = _memory(path)
    recovered = restarted.retrieve_entries(
        "respuestas",
        user_id="user-a",
        categories=(MemoryCategory.USER_PREFERENCE,),
    )

    assert recovered == (stored,)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["last_memory_sequence"] == 1
    assert payload["entries"][0]["category"] == "user_preference"


def test_update_then_reconstruct_recovers_new_value(tmp_path: Path) -> None:
    path = tmp_path / "personal_memory.json"
    memory = _memory(path)
    stored = _store_preference(
        memory,
        "prefiero respuestas breves",
        request_id="request-1",
    )

    updated = memory.update_entry(
        stored.memory_id,
        content="prefiero respuestas detalladas",
        source_request_id="request-2",
    )
    restarted = _memory(path)

    assert restarted.get_entry(stored.memory_id) == updated
    assert "breves" not in path.read_text(encoding="utf-8")


def test_forget_physically_removes_content_and_restart_does_not_recover(
    tmp_path: Path,
) -> None:
    path = tmp_path / "personal_memory.json"
    memory = _memory(path)
    stored = _store_preference(
        memory,
        "prefiero tema oscuro",
        request_id="request-1",
    )

    forgotten = memory.forget_entry(
        stored.memory_id,
        source_request_id="request-2",
    )
    restarted = _memory(path)

    assert forgotten.active is False
    assert restarted.list_entries() == ()
    encoded = path.read_text(encoding="utf-8")
    assert "tema oscuro" not in encoded
    assert json.loads(encoded)["entries"] == []
    next_entry = _store_preference(
        restarted,
        "prefiero tema claro",
        request_id="request-3",
    )
    assert next_entry.memory_id == "memory-000002"


def test_restart_continues_global_id_sequence_without_collisions(
    tmp_path: Path,
) -> None:
    path = tmp_path / "personal_memory.json"
    first = _memory(path)
    one = _store_preference(first, "prefiero azul", request_id="request-1")
    two = _store_preference(first, "prefiero verde", request_id="request-2")

    restarted = _memory(path)
    three = _store_preference(restarted, "prefiero rojo", request_id="request-3")

    assert (one.memory_id, two.memory_id, three.memory_id) == (
        "memory-000001",
        "memory-000002",
        "memory-000003",
    )
    assert len({entry.memory_id for entry in restarted.list_entries()}) == 3


def test_persisted_preferences_remain_isolated_by_user_id(tmp_path: Path) -> None:
    path = tmp_path / "personal_memory.json"
    first = _memory(path)
    user_a = _store_preference(
        first,
        "prefiero respuestas breves",
        request_id="request-1",
        user_id="user-a",
    )
    user_b = _store_preference(
        first,
        "prefiero respuestas detalladas",
        request_id="request-2",
        user_id="user-b",
    )

    restarted = _memory(path)

    assert restarted.list_entries(user_id="user-a") == (user_a,)
    assert restarted.list_entries(user_id="user-b") == (user_b,)


def test_corrupt_file_fails_controlled_without_overwrite(tmp_path: Path) -> None:
    path = tmp_path / "personal_memory.json"
    original = "{not-json"
    path.write_text(original, encoding="utf-8")

    with pytest.raises(CorruptedMemoryRepositoryError):
        _memory(path)

    assert path.read_text(encoding="utf-8") == original


def test_invalid_utf8_file_fails_controlled_without_overwrite(
    tmp_path: Path,
) -> None:
    path = tmp_path / "personal_memory.json"
    original = b"\xff\xfe"
    path.write_bytes(original)

    with pytest.raises(CorruptedMemoryRepositoryError):
        _memory(path)

    assert path.read_bytes() == original


def test_incompatible_version_fails_controlled_without_overwrite(
    tmp_path: Path,
) -> None:
    path = tmp_path / "personal_memory.json"
    original = '{"entries":[],"last_memory_sequence":0,"schema_version":2}'
    path.write_text(original, encoding="utf-8")

    with pytest.raises(UnsupportedMemoryRepositoryVersion):
        _memory(path)

    assert path.read_text(encoding="utf-8") == original


def test_atomic_replace_failure_preserves_file_and_rolls_back_ram(
    monkeypatch,
    tmp_path: Path,
) -> None:
    path = tmp_path / "personal_memory.json"
    memory = _memory(path)
    stored = _store_preference(
        memory,
        "prefiero respuestas breves",
        request_id="request-1",
    )
    previous_file = path.read_bytes()

    def fail_replace(_source, _destination) -> None:
        raise OSError("simulated replace failure")

    monkeypatch.setattr(memory_repository.os, "replace", fail_replace)

    with pytest.raises(MemoryRepositoryWriteError):
        memory.update_entry(
            stored.memory_id,
            content="prefiero respuestas detalladas",
            source_request_id="request-2",
        )

    assert path.read_bytes() == previous_file
    assert memory.get_entry(stored.memory_id) == stored
    assert tuple(path.parent.glob(f".{path.name}.*.tmp")) == ()


def test_conversation_and_unauthorized_categories_are_not_persisted(
    tmp_path: Path,
) -> None:
    path = tmp_path / "personal_memory.json"
    memory = _memory(path)
    memory.add_user("conversacion privada")
    memory.add_assistant("respuesta transitoria")
    memory.store_entry(
        "nota conversacional",
        category=MemoryCategory.CONVERSATION_NOTE,
        source_request_id="request-1",
    )
    memory.store_entry(
        "dato de usuario",
        category=MemoryCategory.USER_FACT,
        source_request_id="request-2",
    )
    memory.store_entry(
        "preferencia sin orden explicita",
        category=MemoryCategory.USER_PREFERENCE,
    )

    restarted = _memory(path)
    encoded = path.read_text(encoding="utf-8")

    assert restarted.list_entries() == ()
    assert json.loads(encoded)["entries"] == []
    assert "conversacion privada" not in encoded
    assert "respuesta transitoria" not in encoded
    assert "nota conversacional" not in encoded
    assert "dato de usuario" not in encoded
    assert "preferencia sin orden explicita" not in encoded


def test_repository_rejects_unauthorized_persisted_category(tmp_path: Path) -> None:
    path = tmp_path / "personal_memory.json"
    memory = _memory(path)
    stored = _store_preference(
        memory,
        "prefiero respuestas breves",
        request_id="request-1",
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["entries"][0]["category"] = "conversation_note"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(CorruptedMemoryRepositoryError):
        _memory(path)

    assert stored.memory_id in path.read_text(encoding="utf-8")


def test_bootstrap_loads_persistent_memory_across_rebuilds(
    monkeypatch,
    tmp_path: Path,
) -> None:
    path = tmp_path / "personal_memory.json"
    monkeypatch.setenv("ATLAS_PERSONAL_MEMORY_PATH", str(path))
    monkeypatch.setenv(
        "ATLAS_EXECUTION_STATE_PATH",
        str(tmp_path / "execution_state.json"),
    )
    monkeypatch.setenv(
        "ATLAS_EXECUTION_HISTORY_PATH",
        str(tmp_path / "execution_sessions"),
    )

    first = Bootstrap.build()
    request = first._request_gateway.from_text(
        "recuerda que prefiero respuestas breves",
        safety_context=RequestSafetyContext(allow_side_effects=True),
    )
    stored = first.process_request(request)
    second = Bootstrap.build()
    recovered = second.process_prompt_result(
        "que recuerdas sobre respuestas"
    )

    assert stored.status is RouteExecutionStatus.COMPLETED
    assert recovered.status is RouteExecutionStatus.COMPLETED
    assert recovered.output["count"] == 1
    assert recovered.output["items"][0]["content"] == "prefiero respuestas breves"


def test_personal_memory_path_defaults_under_atlas_and_accepts_override(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv("ATLAS_PERSONAL_MEMORY_PATH", raising=False)
    assert _personal_memory_path().as_posix() == ".atlas/personal_memory.json"

    configured = tmp_path / "memory.json"
    monkeypatch.setenv("ATLAS_PERSONAL_MEMORY_PATH", str(configured))
    assert _personal_memory_path() == configured
