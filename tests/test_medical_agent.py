from __future__ import annotations

from agents.medical_agent import MedicalAgent
from bootstrap.bootstrap import Bootstrap


class RecordingPromptClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, list[dict[str, str]]]] = []

    def ask(self, *, model: str, messages: list[dict[str, str]]) -> str:
        self.calls.append((model, messages))
        return "guidance"


def test_medical_agent_preserves_v22_context_without_persisting_data() -> None:
    client = RecordingPromptClient()
    agent = MedicalAgent(client)  # type: ignore[arg-type]
    messages = [
        {"role": "system", "content": "Contexto operativo limitado:\n- user_id: user-a\n- profile_id: main\n- domain: medical"},
        {"role": "user", "content": "Tengo dolor persistente tras entrenar."},
    ]

    response = agent.run("local-model", messages)

    assert response == "guidance"
    assert agent.name == "medical"
    assert client.calls == [("local-model", [{"role": "system", "content": agent.SYSTEM_PROMPT}, *messages])]


def test_medical_agent_covers_evidence_safety_and_bootstrap_registration() -> None:
    agent = MedicalAgent(RecordingPromptClient())  # type: ignore[arg-type]
    prompt = agent.SYSTEM_PROMPT.casefold()

    for capability in ("medicina general", "medicina deportiva", "fisioterapia", "rehabilitacion", "farmacologia", "analiticas", "nutricion clinica", "prevencion", "factores de riesgo", "salud cardiovascular", "evidencia fuerte", "moderada", "limitada", "signos de alarma"):
        assert capability in prompt
    assert "no realices diagnosticos definitivos" in prompt
    assert "ni sustituyas la atencion medica" in prompt
    assert "derivacion urgente" in prompt
    assert "no escribas recuerdos" in prompt
    assert "ni persistas datos\nautomaticamente" in prompt

    orchestrator = Bootstrap.build()

    assert isinstance(orchestrator._registry.get("medical"), MedicalAgent)
    for name in ("training", "nutrition", "code", "coding"):
        assert orchestrator._registry.get(name) is not None