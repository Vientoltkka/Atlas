"""Focal tests: the default text loop (python main.py) must route to
specialist agents instead of falling through to the execution conversation.

All builtin agents share one PromptClient instance, so the stub is keyed by
each agent's system prompt.
"""

from __future__ import annotations

import builtins
import json

from agents.nutrition_agent import NutritionAgent
from agents.training_agent import TrainingAgent
from bootstrap.bootstrap import Bootstrap


NUTRITION_PROMPT = (
    "Dame un plan de nutrición para una persona de 30 años que quiere ganar "
    "masa muscular"
)

EXPLICIT_NUTRITION_PROMPT = (
    "Usa el agente de nutrición y dame un plan de nutrición para ganar masa "
    "muscular."
)

TRAINING_PROMPT = (
    "Dame una sesión de entrenamiento de fuerza para el gimnasio de 60 minutos"
)

CHAT_PROMPT = "Que es Python?"


def _structured_response(text: str, *, follow_up: bool) -> str:
    return (
        "```json\n"
        + json.dumps({"text": text, "requires_follow_up": follow_up})
        + "\n```"
    )


def _stub_shared_prompt_client(monkeypatch, orchestrator) -> list[list[dict[str, str]]]:
    chat = orchestrator._registry.get("chat")
    nutrition = orchestrator._registry.get("nutrition")
    training = orchestrator._registry.get("training")
    assert nutrition._client is chat._client
    assert training._client is chat._client
    calls: list[list[dict[str, str]]] = []
    monkeypatch.setattr(chat._client, "check_model_health", lambda *_a, **_k: None)

    def respond(
        *,
        model: str,
        messages: list[dict[str, str]],
        provider_id: str | None = None,
    ) -> str:
        del model, provider_id
        calls.append(messages)
        system = messages[0]["content"] if messages else ""
        if "Nutrition Agent" in system:
            return _structured_response("Plan de nutricion listo.", follow_up=False)
        if "Training Agent" in system:
            return "Sesion de fuerza lista."
        return "Respuesta general."

    monkeypatch.setattr(chat._client, "ask", respond)
    return calls


def _system_agents_used(messages: list[list[dict[str, str]]]) -> list[str]:
    used = []
    for call in messages:
        system = call[0]["content"] if call else ""
        if "Nutrition Agent" in system:
            used.append("nutrition")
        elif "Training Agent" in system:
            used.append("training")
        else:
            used.append("chat")
    return used


def test_text_loop_routes_nutrition_prompt_to_nutrition_agent(monkeypatch, capsys) -> None:
    orchestrator = Bootstrap.build()
    assert isinstance(orchestrator._registry.get("nutrition"), NutritionAgent)
    calls = _stub_shared_prompt_client(monkeypatch, orchestrator)
    answers = iter([NUTRITION_PROMPT, "salir"])
    monkeypatch.setattr(builtins, "input", lambda *_a, **_k: next(answers))

    orchestrator.start()

    out = capsys.readouterr().out
    assert _system_agents_used(calls) == ["nutrition"]
    assert "Plan de nutricion listo." in out
    assert {"role": "user", "content": NUTRITION_PROMPT} in calls[0]
    assert (
        "Atlas todavia no dispone de la capacidad necesaria para esa accion."
        not in out
    )


def test_text_loop_routes_explicit_nutrition_request_to_nutrition_agent(
    monkeypatch,
    capsys,
) -> None:
    """Atlas supports explicit agent invocation: AgentOrchestrator domain
    markers ("nutric", "masa muscular") match the phrase "agente de nutrición",
    so the existing AgentOrchestrator selection resolves the explicit request;
    no new invocation feature is added."""

    orchestrator = Bootstrap.build()
    assert isinstance(orchestrator._registry.get("nutrition"), NutritionAgent)
    calls = _stub_shared_prompt_client(monkeypatch, orchestrator)
    answers = iter([EXPLICIT_NUTRITION_PROMPT, "salir"])
    monkeypatch.setattr(builtins, "input", lambda *_a, **_k: next(answers))

    orchestrator.start()

    out = capsys.readouterr().out
    assert _system_agents_used(calls) == ["nutrition"]
    assert "Plan de nutricion listo." in out


def test_text_loop_keeps_training_requests_on_training_agent(monkeypatch, capsys) -> None:
    orchestrator = Bootstrap.build()
    assert isinstance(orchestrator._registry.get("training"), TrainingAgent)
    calls = _stub_shared_prompt_client(monkeypatch, orchestrator)
    answers = iter([TRAINING_PROMPT, "salir"])
    monkeypatch.setattr(builtins, "input", lambda *_a, **_k: next(answers))

    orchestrator.start()

    out = capsys.readouterr().out
    assert _system_agents_used(calls) == ["training"]
    assert "Sesion de fuerza lista." in out


def test_text_loop_keeps_plain_chat_out_of_specialist_agents(monkeypatch, capsys) -> None:
    orchestrator = Bootstrap.build()
    calls = _stub_shared_prompt_client(monkeypatch, orchestrator)
    answers = iter([CHAT_PROMPT, "salir"])
    monkeypatch.setattr(builtins, "input", lambda *_a, **_k: next(answers))

    orchestrator.start()

    out = capsys.readouterr().out
    assert _system_agents_used(calls) == ["chat"]
    assert "Respuesta general." in out
