from __future__ import annotations

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

    for capability in ("pérdida de grasa", "hipertrofia", "recomposición corporal", "crossfit", "hyrox", "halterofilia", "fuerza", "rendimiento", "recuperación", "hidratación", "timing nutricional", "suplementación basada en evidencia", "restricciones alimentarias declaradas"):
        assert capability in prompt
    assert "no escribas recuerdos" in prompt
    assert "ni persistas\ndatos automáticamente" in prompt

    orchestrator = Bootstrap.build()

    assert isinstance(orchestrator._registry.get("nutrition"), NutritionAgent)