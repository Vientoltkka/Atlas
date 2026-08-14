from __future__ import annotations

from agents.code_agent import CodeAgent
from bootstrap.bootstrap import Bootstrap


class RecordingPromptClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, list[dict[str, str]]]] = []

    def ask(self, *, model: str, messages: list[dict[str, str]]) -> str:
        self.calls.append((model, messages))
        return "proposal"


def test_code_agent_preserves_v22_context_without_persisting_data() -> None:
    client = RecordingPromptClient()
    agent = CodeAgent(client)  # type: ignore[arg-type]
    messages = [
        {"role": "system", "content": "Contexto operativo limitado:\n- user_id: user-a\n- profile_id: main\n- domain: code"},
        {"role": "user", "content": "Crea una API FastAPI con PostgreSQL."},
    ]

    response = agent.run("local-model", messages)

    assert response == "proposal"
    assert agent.name == "code"
    assert client.calls == [("local-model", [{"role": "system", "content": agent.SYSTEM_PROMPT}, *messages])]


def test_code_agent_covers_scope_and_bootstrap_registration() -> None:
    agent = CodeAgent(RecordingPromptClient())  # type: ignore[arg-type]
    prompt = agent.SYSTEM_PROMPT.casefold()

    for capability in ("react", "next.js", "vite", "flutter", "react native", "fastapi", "flask", "express", "node", "supabase", "postgresql", "python", "bash", "powershell", "arquitectura de software", "debugging", "refactorizacion", "testing", "openai", "telegram", "whatsapp", "stripe", "github", "vercel", "proyectos completos"):
        assert capability in prompt
    assert "no escribas recuerdos" in prompt
    assert "ni persistas datos\nautomaticamente" in prompt

    orchestrator = Bootstrap.build()

    assert isinstance(orchestrator._registry.get("code"), CodeAgent)
    assert orchestrator._registry.get("coding") is not None
    assert orchestrator._registry.get("training") is not None
    assert orchestrator._registry.get("nutrition") is not None