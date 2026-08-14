from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import pytest

from bootstrap.bootstrap import Bootstrap, _personal_memory_path
from core.operational_context import OperationalContextBuilder
from core.operational_request_router import RequestRoute, RouteDecision
from core.operational_route_executor import RouteExecutionStatus
from core.request_gateway import RequestSafetyContext
from core.request_gateway import RequestGateway
from memory.conversation import ConversationMemory
from memory.operational import (
    MemoryCategory,
    SensitiveMemoryRejectedError,
)
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
    domain: str | None = None,
    key: str | None = None,
    profile_id: str | None = None,
):
    return memory.store_entry(
        content,
        category=MemoryCategory.USER_PREFERENCE,
        source_request_id=request_id,
        user_id=user_id,
        domain=domain,
        key=key,
        profile_id=profile_id,
    )


def _store_profile(
    memory: ConversationMemory,
    profile_id: str,
    name: str,
    *,
    request_id: str,
    user_id: str = "user-a",
    role: str | None = None,
    primary: bool = False,
):
    return memory.store_entry(
        name,
        category=MemoryCategory.USER_PROFILE,
        source_request_id=request_id,
        user_id=user_id,
        profile_id=profile_id,
        profile_name=name,
        profile_role=role,
        profile_is_primary=primary,
    )



def _personal_request(content: str, request_id: str):
    return RequestGateway(
        clock=lambda: NOW,
        id_generator=lambda: request_id,
    ).from_text(content, request_id=request_id, user_id="user-a")


def _direct_response_decision(request_id: str) -> RouteDecision:
    return RouteDecision(
        request_id=request_id,
        route=RequestRoute.DIRECT_RESPONSE,
        confidence=1.0,
        reason="test direct response",
        matched_rules=("test.direct_response",),
        created_at=NOW,
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
    assert payload["schema_version"] == 3
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


def test_structured_preferences_are_selective_persistent_and_user_isolated(
    tmp_path: Path,
) -> None:
    path = tmp_path / "personal_memory.json"
    first = _memory(path)
    food_a = _store_preference(
        first,
        "prefiero comida italiana",
        request_id="request-1",
        user_id="user-a",
        domain="nutrition",
        key="food_preference",
    )
    training_schedule = _store_preference(
        first,
        "prefiero entrenar por la tarde",
        request_id="request-2",
        user_id="user-a",
        domain="training",
        key="schedule_preference",
    )
    travel_schedule = _store_preference(
        first,
        "prefiero viajar por la manana",
        request_id="request-3",
        user_id="user-a",
        domain="travel",
        key="schedule_preference",
    )
    food_b = _store_preference(
        first,
        "prefiero comida japonesa",
        request_id="request-4",
        user_id="user-b",
        domain="nutrition",
        key="food_preference",
    )

    assert first.retrieve_entries(
        user_id="user-a",
        domain="nutrition",
        key="food_preference",
    ) == (food_a,)
    assert first.retrieve_entries(
        user_id="user-b",
        domain="nutrition",
        key="food_preference",
    ) == (food_b,)
    assert first.retrieve_entries(
        user_id="user-a",
        domain="training",
        key="schedule_preference",
    ) == (training_schedule,)
    assert first.retrieve_entries(
        user_id="user-a",
        domain="travel",
        key="schedule_preference",
    ) == (travel_schedule,)

    updated_food_a = first.update_entry(
        food_a.memory_id,
        content="prefiero comida vegetariana",
        source_request_id="request-5",
    )
    first.forget_entry(
        travel_schedule.memory_id,
        source_request_id="request-6",
    )

    restarted = _memory(path)
    assert restarted.retrieve_entries(
        user_id="user-a",
        domain="nutrition",
        key="food_preference",
    ) == (updated_food_a,)
    assert restarted.retrieve_entries(
        user_id="user-b",
        domain="nutrition",
        key="food_preference",
    ) == (food_b,)
    assert restarted.retrieve_entries(
        user_id="user-a",
        domain="training",
        key="schedule_preference",
    ) == (training_schedule,)
    assert restarted.retrieve_entries(
        user_id="user-a",
        domain="travel",
        key="schedule_preference",
    ) == ()


def test_repository_loads_v1_json_without_structured_fields(tmp_path: Path) -> None:
    path = tmp_path / "personal_memory.json"
    first = _memory(path)
    stored = _store_preference(
        first,
        "prefiero respuestas breves",
        request_id="request-1",
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["schema_version"] = 1
    for entry in payload["entries"]:
        for key in (
            "domain",
            "key",
            "profile_id",
            "profile_name",
            "profile_role",
            "profile_is_primary",
        ):
            entry.pop(key)
    path.write_text(json.dumps(payload), encoding="utf-8")

    restarted = _memory(path)

    assert restarted.get_entry(stored.memory_id) == stored
    assert restarted.get_entry(stored.memory_id).domain is None
    assert restarted.get_entry(stored.memory_id).key is None




def test_repository_loads_v2_structured_preference_without_profile_id(
    tmp_path: Path,
) -> None:
    path = tmp_path / "personal_memory.json"
    first = _memory(path)
    stored = _store_preference(
        first,
        "prefiero entrenar por la tarde",
        request_id="request-1",
        domain="training",
        key="schedule_preference",
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["schema_version"] = 2
    for entry in payload["entries"]:
        for key in (
            "profile_id",
            "profile_name",
            "profile_role",
            "profile_is_primary",
        ):
            entry.pop(key)
    path.write_text(json.dumps(payload), encoding="utf-8")

    restarted = _memory(path)

    assert restarted.get_entry(stored.memory_id) == stored
    assert restarted.get_entry(stored.memory_id).profile_id is None


def test_profiles_are_isolated_selective_and_persistent(tmp_path: Path) -> None:
    path = tmp_path / "personal_memory.json"
    first = _memory(path)
    primary = _store_profile(
        first,
        "main",
        "Victor",
        request_id="request-1",
        primary=True,
    )
    client = _store_profile(
        first,
        "client-a",
        "Ana",
        request_id="request-2",
        role="client",
    )
    primary_preference = _store_preference(
        first,
        "prefiero entrenar por la manana",
        request_id="request-3",
        profile_id="main",
        domain="training",
        key="schedule_preference",
    )
    client_preference = _store_preference(
        first,
        "prefiero entrenar por la tarde",
        request_id="request-4",
        profile_id="client-a",
        domain="training",
        key="schedule_preference",
    )

    assert first.retrieve_entries(
        user_id="user-a",
        profile_id="main",
        domain="training",
        key="schedule_preference",
    ) == (primary_preference,)
    assert first.retrieve_entries(
        user_id="user-a",
        profile_id="client-a",
        domain="training",
        key="schedule_preference",
    ) == (client_preference,)

    updated_client = first.update_entry(
        client.memory_id,
        content="Ana Maria",
        source_request_id="request-5",
    )
    updated_client_preference = first.update_entry(
        client_preference.memory_id,
        content="prefiero entrenar al mediodia",
        source_request_id="request-6",
    )
    first.forget_entry(
        primary_preference.memory_id,
        source_request_id="request-7",
    )

    restarted = _memory(path)
    recovered_primary = restarted.retrieve_entries(
        user_id="user-a",
        profile_id="main",
        categories=(MemoryCategory.USER_PROFILE,),
    )
    recovered_client = restarted.retrieve_entries(
        user_id="user-a",
        profile_id="client-a",
        categories=(MemoryCategory.USER_PROFILE,),
    )
    recovered_primary_preference = restarted.retrieve_entries(
        user_id="user-a",
        profile_id="main",
        domain="training",
        key="schedule_preference",
    )
    recovered_client_preference = restarted.retrieve_entries(
        user_id="user-a",
        profile_id="client-a",
        domain="training",
        key="schedule_preference",
    )

    assert recovered_primary == (primary,)
    assert recovered_primary[0].profile_is_primary is True
    assert recovered_client == (updated_client,)
    assert recovered_client[0].profile_name == "Ana Maria"
    assert recovered_client[0].profile_role == "client"
    assert recovered_primary_preference == ()
    assert recovered_client_preference == (updated_client_preference,)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 3
    assert {
        (entry["profile_id"], entry["category"])
        for entry in payload["entries"]
    } == {
        ("main", "user_profile"),
        ("client-a", "user_profile"),
        ("client-a", "user_preference"),
    }
def test_structured_persistence_rejects_unauthorized_sensitive_and_metrics(
    tmp_path: Path,
) -> None:
    path = tmp_path / "personal_memory.json"
    memory = _memory(path)
    memory.store_entry(
        "prefiero comida italiana",
        category=MemoryCategory.USER_PREFERENCE,
        user_id="user-a",
        domain="nutrition",
        key="food_preference",
    )

    with pytest.raises(SensitiveMemoryRejectedError):
        memory.store_entry(
            "prefiero comida japonesa",
            category=MemoryCategory.USER_PREFERENCE,
            source_request_id="request-2",
            user_id="user-a",
            domain="nutrition",
            key="food_preference",
            sensitive=True,
        )
    with pytest.raises(SensitiveMemoryRejectedError):
        memory.store_entry(
            "objetivo diario 2200",
            category=MemoryCategory.USER_PREFERENCE,
            source_request_id="request-3",
            user_id="user-a",
            domain="nutrition",
            key="calorie_target",
        )
    with pytest.raises(SensitiveMemoryRejectedError):
        memory.store_entry(
            "Ana",
            category=MemoryCategory.USER_PROFILE,
            source_request_id="request-4",
            user_id="user-a",
            profile_id="client-a",
            profile_name="Ana",
            sensitive=True,
        )
    with pytest.raises(SensitiveMemoryRejectedError):
        memory.store_entry(
            "peso 80 kg",
            category=MemoryCategory.USER_PROFILE,
            source_request_id="request-5",
            user_id="user-a",
            profile_id="client-a",
            profile_name="peso 80 kg",
        )
    stored_food = memory.store_entry(
        "prefiero comida italiana",
        category=MemoryCategory.USER_PREFERENCE,
        source_request_id="request-6",
        user_id="user-a",
        profile_id="client-a",
        domain="nutrition",
        key="food_preference",
    )

    entries = json.loads(path.read_text(encoding="utf-8"))["entries"]
    assert len(entries) == 1
    assert entries[0]["memory_id"] == stored_food.memory_id
    assert entries[0]["profile_id"] == "client-a"


def test_personal_context_uses_domain_active_profile_user_and_five_entry_cap(
    tmp_path: Path,
) -> None:
    memory = _memory(tmp_path / "personal_memory.json")
    _store_profile(memory, "main", "Principal", request_id="profile-main", primary=True)
    _store_profile(memory, "client", "Cliente", request_id="profile-client")
    main_food = _store_preference(memory, "prefiero comida italiana", request_id="main-food", domain="nutrition", key="food_preference", profile_id="main")
    main_training = _store_preference(memory, "prefiero entrenar por la manana", request_id="main-training", domain="training", key="schedule_preference", profile_id="main")
    client_context = _store_preference(memory, "prefiero avisos del cliente", request_id="client-context", domain="client", key="notification_preference", profile_id="client")
    general_context = _store_preference(memory, "prefiero respuestas breves", request_id="general", domain="general", key="response_preference", profile_id="main")
    _store_preference(memory, "prefiero comida francesa", request_id="other-food", user_id="user-b", domain="nutrition", key="food_preference", profile_id="main")
    builder = OperationalContextBuilder(memory, clock=lambda: NOW)

    nutrition = builder.build(_personal_request("prepara un menu nutricional", "nutrition"), _direct_response_decision("nutrition"))
    training = builder.build(_personal_request("crea una rutina de entrenamiento", "training"), _direct_response_decision("training"))
    code = builder.build(_personal_request("revisa este codigo Python", "code"), _direct_response_decision("code"))
    client = builder.build(_personal_request("consulta para profile.client", "client"), _direct_response_decision("client"))
    general = builder.build(_personal_request("dame una recomendacion general", "general"), _direct_response_decision("general"))

    assert nutrition.relevant_memories == (main_food,)
    assert training.relevant_memories == (main_training,)
    assert code.relevant_memories == ()
    assert client.relevant_memories == (client_context,)
    assert general.relevant_memories == (general_context,)

    for index in range(6):
        _store_preference(memory, f"preferencia nutricional valida {index}", request_id=f"extra-{index}", domain="nutrition", key=f"extra_{index}_preference", profile_id="main")
    limited = builder.build(_personal_request("prepara una dieta", "limited"), _direct_response_decision("limited"))

    assert len(limited.relevant_memories) == 5
    assert limited.total_characters <= memory.policy.max_context_characters
    assert limited.truncated is True

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
    original = '{"entries":[],"last_memory_sequence":0,"schema_version":999}'
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
