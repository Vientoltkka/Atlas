from __future__ import annotations

from bootstrap.bootstrap import Bootstrap
from agents.finance_agent import FinanceAgent


class RecordingPromptClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, list[dict[str, str]]]] = []

    def ask(self, *, model: str, messages: list[dict[str, str]]) -> str:
        self.calls.append((model, messages))
        return "analysis"


def test_finance_agent_preserves_v22_context_without_persisting_data() -> None:
    client = RecordingPromptClient()
    agent = FinanceAgent(client)  # type: ignore[arg-type]
    messages = [
        {"role": "system", "content": "Contexto operativo limitado:\n- user_id: user-a\n- profile_id: main\n- domain: finance"},
        {"role": "user", "content": "Explica el riesgo de una cartera con ETFs."},
    ]

    response = agent.run("local-model", messages)

    assert response == "analysis"
    assert agent.name == "finance"
    assert client.calls == [("local-model", [{"role": "system", "content": agent.SYSTEM_PROMPT}, *messages])]


def test_finance_agent_defines_v1_personal_finance_and_safety_contract() -> None:
    agent = FinanceAgent(RecordingPromptClient())  # type: ignore[arg-type]
    prompt = " ".join(agent.SYSTEM_PROMPT.casefold().split())

    for capability in (
        "presupuestos mensuales",
        "control de gastos",
        "ahorro",
        "planificacion financiera personal",
        "fondo indexado",
        "compara opciones financieras",
        "interes o rentabilidad",
        "una sola aclaracion",
        "datos actuales de mercado",
    ):
        assert capability in prompt
    assert "no prometas ni garantices rentabilidad" in prompt
    assert "no inventes precios" in prompt
    assert "no ejecutes compras" in prompt
    assert "no des por hecho tolerancia al riesgo" in prompt
    assert "legalagent" in prompt
    assert "medicalagent" in prompt
    assert "nutrition" in prompt
    assert "coach" in prompt
    assert "no escribas recuerdos" in prompt
    assert "ni persistas datos automaticamente" in prompt


def test_finance_agent_bootstrap_registration() -> None:
    orchestrator = Bootstrap.build()

    assert isinstance(orchestrator._registry.get("finance"), FinanceAgent)
    for name in ("training", "nutrition", "code", "coding", "medical", "legal"):
        assert orchestrator._registry.get(name) is not None


def test_conversation_executes_finance_agent_with_mocked_model(monkeypatch) -> None:
    orchestrator = Bootstrap.build()
    finance = orchestrator._registry.get("finance")
    assert isinstance(finance, FinanceAgent)
    calls: list[tuple[str, list[dict[str, str]]]] = []

    monkeypatch.setattr(finance._client, "check_model_health", lambda _model: None)

    def respond(*, model: str, messages: list[dict[str, str]]) -> str:
        calls.append((model, messages))
        return "Orientacion financiera generada."

    monkeypatch.setattr(finance._client, "ask", respond)

    response = orchestrator.process_prompt(
        "Calcula el interes simple de 1000 euros al 5% anual durante 2 anos.",
        confirm=lambda _prompt: "",
    )

    assert response == "Orientacion financiera generada."
    assert len(calls) == 1
    assert calls[0][1][-1] == {
        "role": "user",
        "content": "Calcula el interes simple de 1000 euros al 5% anual durante 2 anos.",
    }
