from __future__ import annotations

import pytest

from agents.nutrition_agent import NutritionAgent
from bootstrap.bootstrap import Bootstrap


class RecordingPromptClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, list[dict[str, str]]]] = []

    def ask(self, *, model: str, messages: list[dict[str, str]]) -> str:
        self.calls.append((model, messages))
        return "orientacion"


def test_nutrition_agent_preserves_v22_context_without_persisting_data() -> None:
    client = RecordingPromptClient()
    agent = NutritionAgent(client)  # type: ignore[arg-type]
    messages = [
        {"role": "system", "content": "Contexto operativo limitado:\n- user_id: user-a\n- profile_id: main\n- domain: nutrition\n- restricción declarada: vegetariana"},
        {"role": "user", "content": "Prepara mi timing para HYROX."},
    ]

    response = agent.run("local-model", messages)

    assert response == "orientacion"
    assert agent.name == "nutrition"
    assert client.calls == [("local-model", [{"role": "system", "content": agent.SYSTEM_PROMPT}, *messages])]


def test_nutrition_agent_covers_scope_safety_and_bootstrap_registration() -> None:
    agent = NutritionAgent(RecordingPromptClient())  # type: ignore[arg-type]
    prompt = agent.SYSTEM_PROMPT.casefold()

    for capability in ("grasa", "hipertrofia", "recompos", "crossfit", "hyrox", "halterofilia", "fuerza", "rendimiento", "recuper", "hidrat", "timing nutricional", "suplement", "restricciones alimentarias declaradas"):
        assert capability in prompt
    assert "no escribas recuerdos" in prompt
    assert "ni persistas datos" in prompt

    orchestrator = Bootstrap.build()

    assert isinstance(orchestrator._registry.get("nutrition"), NutritionAgent)


@pytest.mark.parametrize(
    "user_prompt",
    (
        "Prepara un plan para ganar masa muscular.",
        "Calcula mis macros para perder grasa: mujer, 30 anos, 65 kg, 165 cm y entreno 4 dias.",
        "Que como antes y despues de entrenar CrossFit?",
        "Que suplementacion basica tiene evidencia para fuerza?",
        "Calcula mis macros personalizados.",
    ),
)
def test_nutrition_agent_v1_requests_use_the_operational_contract(user_prompt: str) -> None:
    client = RecordingPromptClient()
    agent = NutritionAgent(client)  # type: ignore[arg-type]

    assert agent.run("simulated-model", [{"role": "user", "content": user_prompt}]) == "orientacion"
    assert client.calls == [
        (
            "simulated-model",
            [
                {"role": "system", "content": agent.SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )
    ]


def test_nutrition_agent_prompt_defines_v1_calculation_and_safety_contract() -> None:
    prompt = NutritionAgent(RecordingPromptClient()).SYSTEM_PROMPT.casefold()  # type: ignore[arg-type]

    for requirement in (
        "calor",
        "macros",
        "peso",
        "altura",
        "edad",
        "sexo",
        "distribuci",
        "pre/post entrenamiento",
        "hidrat",
        "suplement",
        "una sola aclaraci",
        "no asumas ni inventes",
        "no diagnostiques",
        "no prescribas medic",
    ):
        assert requirement in prompt


def test_conversation_executes_existing_nutrition_agent_with_mocked_model(monkeypatch) -> None:
    orchestrator = Bootstrap.build()
    nutrition = orchestrator._registry.get("nutrition")
    assert nutrition is not None
    calls: list[tuple[str, list[dict[str, str]]]] = []

    monkeypatch.setattr(nutrition._client, "check_model_health", lambda _model: None)

    def respond(*, model: str, messages: list[dict[str, str]]) -> str:
        calls.append((model, messages))
        return "Orientacion nutricional generada."

    monkeypatch.setattr(nutrition._client, "ask", respond)

    response = orchestrator.process_prompt(
        "Calcula mis macros: hombre, 35 anos, 80 kg, 180 cm y entreno 5 dias.",
        confirm=lambda _prompt: "",
    )

    assert response == "Orientacion nutricional generada."
    assert len(calls) == 1
    assert calls[0][1][-1] == {
        "role": "user",
        "content": "Calcula mis macros: hombre, 35 anos, 80 kg, 180 cm y entreno 5 dias.",
    }
