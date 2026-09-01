from __future__ import annotations

from bootstrap.bootstrap import Bootstrap
from agents.medical_agent import MedicalAgent


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


def test_medical_agent_defines_v1_guidance_and_safety_contract() -> None:
    agent = MedicalAgent(RecordingPromptClient())  # type: ignore[arg-type]
    prompt = agent.SYSTEM_PROMPT.casefold()

    for capability in (
        "orientacion general",
        "sintomas",
        "inflamacion",
        "senales a vigilar",
        "cuando consultar",
        "autocuidados basicos de bajo riesgo",
        "una sola aclaracion",
        "atencion urgente",
        "crossfit",
        "hyrox",
        "macros",
        "suplementacion",
    ):
        assert capability in prompt
    assert "sin certeza diagnostica" in prompt
    assert "no sustituyas una evaluacion medica" in prompt
    assert "no prescribas" in prompt
    assert "no inventes" in prompt
    assert "antecedentes" in prompt
    assert "no escribas" in prompt
    assert "recuerdos" in prompt


def test_medical_agent_bootstrap_registration() -> None:
    orchestrator = Bootstrap.build()

    assert isinstance(orchestrator._registry.get("medical"), MedicalAgent)
    for name in ("training", "nutrition", "code", "coding"):
        assert orchestrator._registry.get(name) is not None


def test_conversation_executes_medical_agent_with_mocked_model(monkeypatch) -> None:
    orchestrator = Bootstrap.build()
    medical = orchestrator._registry.get("medical")
    assert isinstance(medical, MedicalAgent)
    calls: list[tuple[str, list[dict[str, str]]]] = []

    monkeypatch.setattr(medical._client, "check_model_health", lambda _model: None)

    def respond(*, model: str, messages: list[dict[str, str]]) -> str:
        calls.append((model, messages))
        return "Orientacion medica prudente generada."

    monkeypatch.setattr(medical._client, "ask", respond)

    response = orchestrator.process_prompt(
        "Tengo fiebre y dolor de garganta desde ayer.",
        confirm=lambda _prompt: "",
    )

    assert response == "Orientacion medica prudente generada."
    assert len(calls) == 1
    assert calls[0][1][-1] == {
        "role": "user",
        "content": "Tengo fiebre y dolor de garganta desde ayer.",
    }
