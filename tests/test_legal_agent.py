from __future__ import annotations

from agents.legal_agent import LegalAgent
from bootstrap.bootstrap import Bootstrap


class RecordingPromptClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, list[dict[str, str]]]] = []

    def ask(self, *, model: str, messages: list[dict[str, str]]) -> str:
        self.calls.append((model, messages))
        return "analysis"


def test_legal_agent_preserves_v22_context_without_persisting_data() -> None:
    client = RecordingPromptClient()
    agent = LegalAgent(client)  # type: ignore[arg-type]
    messages = [
        {"role": "system", "content": "Contexto operativo limitado:\n- user_id: user-a\n- profile_id: main\n- domain: legal"},
        {"role": "user", "content": "Revisa este contrato en Espana."},
    ]

    response = agent.run("local-model", messages)

    assert response == "analysis"
    assert agent.name == "legal"
    assert client.calls == [("local-model", [{"role": "system", "content": agent.SYSTEM_PROMPT}, *messages])]


def test_legal_agent_covers_v1_scope_safety_and_bootstrap_registration() -> None:
    agent = LegalAgent(RecordingPromptClient())  # type: ignore[arg-type]
    prompt = " ".join(agent.SYSTEM_PROMPT.casefold().split())

    for capability in (
        "conceptos legales",
        "contratos y clausulas",
        "derechos y obligaciones generales",
        "relaciones laborales basicas",
        "consumo o reclamaciones basicas",
        "pais o jurisdiccion",
    ):
        assert capability in prompt
    assert "pide una sola aclaracion breve solo si falta un dato imprescindible" in prompt
    assert "no inventes ni cites leyes, articulos, sentencias, plazos o jurisdicciones" in prompt
    assert "no asegures resultados legales" in prompt
    assert "no ofrezcas representacion legal" in prompt
    assert "no sustituyas el asesoramiento" in prompt
    assert "no ejecutes tramites, reclamaciones, notificaciones ni otras acciones legales" in prompt
    assert "no escribas recuerdos" in prompt
    assert "ni persistas datos automaticamente" in prompt

    orchestrator = Bootstrap.build()

    assert isinstance(orchestrator._registry.get("legal"), LegalAgent)
    for name in ("training", "nutrition", "code", "coding", "medical"):
        assert orchestrator._registry.get(name) is not None
