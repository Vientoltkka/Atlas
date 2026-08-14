from __future__ import annotations

from agents.training_agent import TrainingAgent


class RecordingPromptClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, list[dict[str, str]]]] = []

    def ask(self, *, model: str, messages: list[dict[str, str]]) -> str:
        self.calls.append((model, messages))
        return "plan"


def test_training_agent_preserves_v22_context_and_user_constraints() -> None:
    client = RecordingPromptClient()
    agent = TrainingAgent(client)  # type: ignore[arg-type]
    context = "Contexto operativo limitado:\n- prefiero sesiones de 45 minutos"
    messages = [
        {"role": "system", "content": context},
        {"role": "user", "content": "Quiero una sesión HYROX nivel principiante con mancuernas."},
    ]

    response = agent.run("local-model", messages)

    assert response == "plan"
    assert agent.name == "training"
    assert client.calls == [
        (
            "local-model",
            [
                {"role": "system", "content": agent.SYSTEM_PROMPT},
                *messages,
            ],
        )
    ]


def test_training_agent_prompt_covers_requested_domains_and_safety() -> None:
    agent = TrainingAgent(RecordingPromptClient())  # type: ignore[arg-type]

    prompt = agent.SYSTEM_PROMPT.casefold()

    for capability in ("crossfit", "hyrox", "hipertrofia", "gimnasia", "fuerza", "movilidad", "periodización"):
        assert capability in prompt
    assert "lesiones declaradas" in prompt
    assert "no afirmes resultados" in prompt
    assert "escribas recuerdos" in prompt