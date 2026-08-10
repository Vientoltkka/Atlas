from __future__ import annotations

import json
from pathlib import Path

from bootstrap.bootstrap import Bootstrap
from core.operational_request_router import MemoryOperation, RequestRoute
from core.operational_route_executor import RouteExecutionStatus
from core.request_gateway import RequestSafetyContext


def _configure_runtime(monkeypatch, tmp_path: Path) -> Path:
    memory_path = tmp_path / "personal_memory.json"
    monkeypatch.setenv("ATLAS_PERSONAL_MEMORY_PATH", str(memory_path))
    monkeypatch.setenv(
        "ATLAS_EXECUTION_STATE_PATH",
        str(tmp_path / "execution_state.json"),
    )
    monkeypatch.setenv(
        "ATLAS_EXECUTION_HISTORY_PATH",
        str(tmp_path / "execution_sessions"),
    )
    return memory_path


def _execute_scoped(
    orchestrator,
    text: str,
    *,
    user_id: str,
    voice: bool = False,
    allow_side_effects: bool = False,
):
    adapter = (
        orchestrator._request_gateway.from_voice
        if voice
        else orchestrator._request_gateway.from_text
    )
    request = adapter(
        text,
        user_id=user_id,
        safety_context=RequestSafetyContext(
            allow_side_effects=allow_side_effects,
        ),
    )
    return orchestrator.process_request(request)


def test_natural_memory_lifecycle_survives_bootstrap_rebuilds_and_voice(
    monkeypatch,
    tmp_path: Path,
) -> None:
    memory_path = _configure_runtime(monkeypatch, tmp_path)

    first = Bootstrap.build()
    stored_visible = first.process_prompt(
        "recuerda que prefiero respuestas breves",
        confirm=lambda _prompt: "",
    )

    assert stored_visible == "Preferencia guardada: prefiero respuestas breves."
    payload = json.loads(memory_path.read_text(encoding="utf-8"))
    assert len(payload["entries"]) == 1
    assert payload["entries"][0]["category"] == "user_preference"

    second = Bootstrap.build()
    queried_visible = second.process_prompt(
        "qu\u00e9 recuerdas de mis preferencias",
        confirm=lambda _prompt: "",
    )
    incomplete_update_visible = second.process_prompt(
        "actualiza mi preferencia respuestas breves",
        confirm=lambda _prompt: "",
    )
    updated_visible = second.process_voice_prompt(
        "cambia mi preferencia respuestas breves por respuestas detalladas",
        confirm=lambda _prompt: "",
    )

    assert queried_visible == (
        "Se recuperaron estos recuerdos: prefiero respuestas breves."
    )
    assert incomplete_update_visible == "Cual es el nuevo contenido de la entrada?"
    assert updated_visible == (
        "Preferencia actualizada: prefiero respuestas detalladas."
    )
    assert "respuestas breves" not in memory_path.read_text(encoding="utf-8")

    third = Bootstrap.build()
    updated_query_visible = third.process_prompt(
        "qu\u00e9 recuerdas de mis preferencias",
        confirm=lambda _prompt: "",
    )
    forgotten_visible = third.process_prompt(
        "olvida respuestas detalladas",
        confirm=lambda _prompt: "",
    )

    assert updated_query_visible == (
        "Se recuperaron estos recuerdos: prefiero respuestas detalladas."
    )
    assert forgotten_visible == (
        "Preferencia olvidada: prefiero respuestas detalladas."
    )

    fourth = Bootstrap.build()
    empty_visible = fourth.process_voice_prompt(
        "qu\u00e9 recuerdas de mis preferencias",
        confirm=lambda _prompt: "",
    )

    assert empty_visible == "No se encontraron recuerdos coincidentes."
    final_payload = json.loads(memory_path.read_text(encoding="utf-8"))
    assert final_payload["entries"] == []
    assert "respuestas detalladas" not in memory_path.read_text(encoding="utf-8")


def test_natural_update_and_forget_are_isolated_by_user_id(
    monkeypatch,
    tmp_path: Path,
) -> None:
    _configure_runtime(monkeypatch, tmp_path)
    orchestrator = Bootstrap.build()

    stored_a = _execute_scoped(
        orchestrator,
        "recuerda que prefiero respuestas breves",
        user_id="user-a",
        allow_side_effects=True,
    )
    stored_b = _execute_scoped(
        orchestrator,
        "recuerda que prefiero respuestas breves",
        user_id="user-b",
        allow_side_effects=True,
    )
    updated_a = _execute_scoped(
        orchestrator,
        "actualiza mi preferencia respuestas breves a respuestas detalladas",
        user_id="user-a",
        voice=True,
        allow_side_effects=True,
    )
    queried_a = _execute_scoped(
        orchestrator,
        "qu\u00e9 recuerdas de mis preferencias",
        user_id="user-a",
    )
    queried_b = _execute_scoped(
        orchestrator,
        "qu\u00e9 recuerdas de mis preferencias",
        user_id="user-b",
        voice=True,
    )

    assert stored_a.status is RouteExecutionStatus.COMPLETED
    assert stored_b.status is RouteExecutionStatus.COMPLETED
    assert stored_a.output["memory_id"] != stored_b.output["memory_id"]
    assert updated_a.status is RouteExecutionStatus.COMPLETED
    assert updated_a.output["entry"]["content"] == "prefiero respuestas detalladas"
    assert tuple(item["content"] for item in queried_a.output["items"]) == (
        "prefiero respuestas detalladas",
    )
    assert tuple(item["content"] for item in queried_b.output["items"]) == (
        "prefiero respuestas breves",
    )

    forgotten_a = _execute_scoped(
        orchestrator,
        "olvida respuestas detalladas",
        user_id="user-a",
        allow_side_effects=True,
    )
    remaining_a = _execute_scoped(
        orchestrator,
        "qu\u00e9 recuerdas de mis preferencias",
        user_id="user-a",
    )
    remaining_b = _execute_scoped(
        orchestrator,
        "qu\u00e9 recuerdas de mis preferencias",
        user_id="user-b",
    )

    assert forgotten_a.status is RouteExecutionStatus.COMPLETED
    assert remaining_a.output["items"] == ()
    assert tuple(item["content"] for item in remaining_b.output["items"]) == (
        "prefiero respuestas breves",
    )


def test_only_explicit_non_sensitive_preference_intent_is_persisted(
    monkeypatch,
    tmp_path: Path,
) -> None:
    memory_path = _configure_runtime(monkeypatch, tmp_path)
    orchestrator = Bootstrap.build()

    implicit = orchestrator.classify_prompt("prefiero respuestas breves")
    sensitive_visible = orchestrator.process_prompt(
        "recuerda que mi password es ejemplo",
        confirm=lambda _prompt: "",
    )

    assert implicit.route is RequestRoute.DIRECT_RESPONSE
    assert implicit.memory_operation is None
    assert "rejected" in sensitive_visible.casefold()
    assert not memory_path.exists()


def test_natural_phrases_map_to_existing_memory_operations(
    monkeypatch,
    tmp_path: Path,
) -> None:
    _configure_runtime(monkeypatch, tmp_path)
    orchestrator = Bootstrap.build()

    cases = (
        ("recuerda que prefiero X", MemoryOperation.STORE),
        ("qu\u00e9 recuerdas de mis preferencias", MemoryOperation.RETRIEVE),
        (
            "cambia mi preferencia respuestas breves por respuestas detalladas",
            MemoryOperation.UPDATE,
        ),
        (
            "actualiza mi preferencia respuestas breves a respuestas detalladas",
            MemoryOperation.UPDATE,
        ),
        ("olvida respuestas detalladas", MemoryOperation.FORGET),
    )

    for text, operation in cases:
        decision = orchestrator.classify_prompt(text)
        assert decision.route is RequestRoute.MEMORY_QUERY
        assert decision.memory_operation is operation
