from __future__ import annotations

import json

from agents.nutrition_agent import NutritionAgent
from bootstrap.bootstrap import Bootstrap


NUTRITION_PROMPT = (
    "Calcula aproximadamente mis calorías y macronutrientes diarios para ganar "
    "masa muscular. Peso 74 kg, mido 1,80 m, entreno CrossFit 5 días por "
    "semana y quiero subir de peso minimizando la ganancia de grasa."
)


def _structured_response(text: str, *, follow_up: bool) -> str:
    return "```json\n" + json.dumps({"text": text, "requires_follow_up": follow_up}) + "\n```"


def _nutrition_orchestrator(monkeypatch, responses: list[str]):
    orchestrator = Bootstrap.build()
    nutrition = orchestrator._registry.get("nutrition")
    assert isinstance(nutrition, NutritionAgent)
    calls: list[list[dict[str, str]]] = []
    monkeypatch.setattr(nutrition._client, "check_model_health", lambda _model: None)

    def respond(*, model: str, messages: list[dict[str, str]]) -> str:
        del model
        calls.append(messages)
        return responses.pop(0)

    monkeypatch.setattr(nutrition._client, "ask", respond)
    return orchestrator, calls


def test_nutrition_followup_continues_with_prior_context_and_is_consumed(monkeypatch) -> None:
    orchestrator, calls = _nutrition_orchestrator(
        monkeypatch,
        [
            _structured_response("Objetivo y macros estimados.", follow_up=False),
        ],
    )

    assert orchestrator.process_prompt(NUTRITION_PROMPT, confirm=lambda _prompt: "") == (
        "Para calcularlo necesito tu edad y tu sexo."
    )
    assert orchestrator.pending_agent_followup is not None
    assert orchestrator.pending_agent_followup.agent_name == "nutrition"

    response = orchestrator.process_prompt("48 años, hombre.", confirm=lambda _prompt: "")
    assert response == "Objetivo y macros estimados."
    assert "requires_follow_up" not in response
    assert '{"text"' not in response
    assert "```json" not in response
    assert orchestrator.pending_agent_followup is None
    assert calls[0][-1] == {"role": "user", "content": "48 años, hombre."}
    assert {"role": "user", "content": NUTRITION_PROMPT} in calls[0]
    assert orchestrator.process_prompt("48 años, hombre.", confirm=lambda _prompt: "") == (
        "Atlas todavia no dispone de la capacidad necesaria para esa accion."
    )
    assert len(calls) == 1


def test_agent_followup_remains_active_when_the_agent_requests_another_turn(monkeypatch) -> None:
    orchestrator, _calls = _nutrition_orchestrator(
        monkeypatch,
        [
            _structured_response("Necesito otro dato.", follow_up=True),
            _structured_response("Cálculo completado.", follow_up=False),
        ],
    )

    orchestrator.process_prompt(NUTRITION_PROMPT, confirm=lambda _prompt: "")
    original_prompt = orchestrator.pending_agent_followup.original_prompt
    orchestrator.process_prompt("48 años, hombre.", confirm=lambda _prompt: "")
    assert orchestrator.pending_agent_followup is not None
    assert orchestrator.pending_agent_followup.original_prompt == original_prompt

    assert orchestrator.process_prompt("Sin alergias conocidas.", confirm=lambda _prompt: "") == (
        "Cálculo completado."
    )
    assert orchestrator.pending_agent_followup is None


def test_agent_followup_cancellation_clears_state(monkeypatch) -> None:
    orchestrator, calls = _nutrition_orchestrator(
        monkeypatch,
        [_structured_response("Necesito un dato.", follow_up=True)],
    )

    orchestrator.process_prompt(NUTRITION_PROMPT, confirm=lambda _prompt: "")
    assert orchestrator.process_prompt("cancelar", confirm=lambda _prompt: "") == (
        "Seguimiento del agente cancelado."
    )
    assert orchestrator.pending_agent_followup is None
    assert len(calls) == 0


def test_short_answer_without_followup_does_not_select_nutrition(monkeypatch) -> None:
    orchestrator = Bootstrap.build()
    nutrition = orchestrator._registry.get("nutrition")
    assert isinstance(nutrition, NutritionAgent)
    calls: list[list[dict[str, str]]] = []
    monkeypatch.setattr(nutrition._client, "check_model_health", lambda _model: None)

    def respond(*, model: str, messages: list[dict[str, str]]) -> str:
        del model
        calls.append(messages)
        return "Respuesta general."

    monkeypatch.setattr(nutrition._client, "ask", respond)

    assert orchestrator.process_prompt("48 años, hombre.", confirm=lambda _prompt: "") == (
        "Atlas todavia no dispone de la capacidad necesaria para esa accion."
    )
    assert calls == []
