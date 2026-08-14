from __future__ import annotations

from agents.finance_agent import FinanceAgent
from bootstrap.bootstrap import Bootstrap


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


def test_finance_agent_covers_scope_safety_and_bootstrap_registration() -> None:
    agent = FinanceAgent(RecordingPromptClient())  # type: ignore[arg-type]
    prompt = agent.SYSTEM_PROMPT.casefold().replace("\n", " ")

    for capability in ("acciones", "etfs", "indices", "bonos", "criptomonedas", "defi", "analisis fundamental", "analisis tecnico", "gestion del riesgo", "asignacion de activos", "diversificacion", "dca", "rebalanceo", "macroeconomia aplicada", "resultados empresariales", "escenarios probabilisticos", "conceptos financieros"):
        assert capability in prompt
    assert "no ofrezcas recomendaciones de inversion personalizadas" in prompt
    assert "ni garantices rendimientos" in prompt
    assert "no escribas recuerdos" in prompt
    assert "ni persistas datos automaticamente" in prompt

    orchestrator = Bootstrap.build()

    assert isinstance(orchestrator._registry.get("finance"), FinanceAgent)
    for name in ("training", "nutrition", "code", "coding", "medical", "legal"):
        assert orchestrator._registry.get(name) is not None