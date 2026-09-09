from __future__ import annotations

import pytest

from bootstrap.bootstrap import Bootstrap
from agents.training_agent import TrainingAgent
from core.model_manager import ModelDescriptor, ModelManager
from models.prompt_client import InferenceBackendError


class RecordingPromptClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, list[dict[str, str]]]] = []

    def ask(self, *, model: str, messages: list[dict[str, str]]) -> str:
        self.calls.append((model, messages))
        return "plan"


class RecordingProviderPromptClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self.fail_models: set[str] = set()

    def ask(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        provider_id: str | None = None,
    ) -> str:
        self.calls.append(
            {"model": model, "provider_id": provider_id, "messages": messages}
        )
        if model in self.fail_models:
            raise InferenceBackendError(model, "simulated provider failure")
        return "plan"


class _NoModels:
    def list_models(self) -> list[str]:
        return []


def _gemini_model_manager() -> ModelManager:
    return ModelManager(
        client=_NoModels(),
        descriptors=(
            ModelDescriptor(
                logical_id="chat-gemini",
                provider_id="gemini",
                model_name="gemini-3.6-flash",
                capabilities=("general_chat", "chat", "fast_response"),
                local=False,
                priority=200,
                fallback_logical_ids=("chat-local",),
            ),
        ),
    )


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

def test_training_agent_prompt_defines_crossfit_session_quality_contract() -> None:
    agent = TrainingAgent(RecordingPromptClient())  # type: ignore[arg-type]

    prompt = agent.SYSTEM_PROMPT.casefold()

    for requirement in (
        "duraci",
        "warm-up",
        "exactamente la duraci",
        "series",
        "repeticiones",
        "%1rm",
        "wod principal",
        "time cap",
        "est",
        "escalado",
        "volumen excesivo",
        "crossfit habitual",
        "solo el contenido del entrenamiento",
        "markdown limpio",
    ):
        assert requirement in prompt


def _prompt_contract_text() -> str:
    raw = TrainingAgent(RecordingPromptClient()).SYSTEM_PROMPT  # type: ignore[arg-type]
    return " ".join(raw.casefold().split())


def test_training_agent_prompt_forbids_unconfirmed_material() -> None:
    prompt = _prompt_contract_text()

    assert "nunca afirmes que existe material" in prompt
    assert "si dispones" in prompt
    assert "material confirmado" in prompt
    # The old loophole that allowed assuming generic gym equipment is gone.
    assert "razonablemente asumible" not in prompt


def test_training_agent_prompt_preserves_explicit_restrictions() -> None:
    prompt = _prompt_contract_text()

    for requirement in (
        "respeta exactamente las cantidades",
        "restricciones explícitas",
        "no inventes restricciones",
        "no asignes más atletas simultáneos",
    ):
        assert requirement in prompt


def test_training_agent_prompt_defines_exact_duration_and_class_logistics_contract() -> None:
    prompt = _prompt_contract_text()

    for requirement in (
        "suma sea exactamente",
        "briefing",
        "transiciones",
        "clases grandes",
        "máquinas simultáneas",
        "cuellos de botella",
        "rotaciones",
    ):
        assert requirement in prompt


@pytest.mark.parametrize(
    "user_prompt",
    (
        "Hazme una sesión de CrossFit de 60 minutos.",
        "Programa un HYROX para 20 personas con 4 ergómetros.",
        "Quiero una sesión de fuerza de powerlifting.",
        "Prepara hipertrofia de tren superior para nivel intermedio.",
        "Quiero mejorar mi clean con una progresión simple.",
        "Hazme un entrenamiento personalizado.",
        "Me duele la rodilla desde ayer: dime qué lesión tengo y cómo tratarla.",
        "Programa mi back squat al 85% de mi 1RM.",
    ),
)
def test_training_agent_v1_requests_use_the_practical_coach_contract(user_prompt: str) -> None:
    client = RecordingPromptClient()
    agent = TrainingAgent(client)  # type: ignore[arg-type]

    assert agent.run("simulated-model", [{"role": "user", "content": user_prompt}]) == "plan"
    assert client.calls == [
        (
            "simulated-model",
            [
                {"role": "system", "content": agent.SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )
    ]


def test_training_agent_prompt_defines_v1_adaptation_and_safety_contract() -> None:
    prompt = TrainingAgent(RecordingPromptClient()).SYSTEM_PROMPT.casefold()  # type: ignore[arg-type]

    for requirement in (
        "powerlifting",
        "halterofilia",
        "levantamientos",
        "una sola aclaraci",
        "atletas",
        "material",
        "rir",
        "descanso",
        "time cap",
        "rotaci",
        "progresiones simples",
        "no inventes lesiones",
        "ni 1rm",
        "no diagnostiques",
        "no des diagn",
        "nutricional",
    ):
        assert requirement in prompt


def test_training_agent_crossfit_pdf_request_uses_reinforced_prompt_and_preserves_request() -> None:
    client = RecordingPromptClient()
    agent = TrainingAgent(client)  # type: ignore[arg-type]
    request = (
        "Créame un entrenamiento de CrossFit de 60 minutos para mañana, nivel "
        "avanzado, con fuerza y WOD y guárdalo en PDF."
    )

    response = agent.run("glm4:9b", [{"role": "user", "content": request}])

    assert response == "plan"
    assert agent.name == "training"
    assert client.calls == [
        (
            "glm4:9b",
            [
                {"role": "system", "content": agent.SYSTEM_PROMPT},
                {"role": "user", "content": request},
            ],
        )
    ]


def test_conversation_executes_existing_training_agent_with_mocked_model(
    monkeypatch,
) -> None:
    orchestrator = Bootstrap.build()
    training = orchestrator._registry.get("training")
    assert training is not None
    calls: list[tuple[str, list[dict[str, str]]]] = []

    monkeypatch.setattr(training._client, "check_model_health", lambda *_args, **_kwargs: None)

    def respond(*, model: str, messages: list[dict[str, str]]) -> str:
        calls.append((model, messages))
        return "Plan de entrenamiento generado."

    monkeypatch.setattr(training._client, "ask", respond)

    response = orchestrator.process_prompt(
        "hazme un entrenamiento de CrossFit de 60 minutos",
        confirm=lambda _prompt: "",
    )

    assert response == "Plan de entrenamiento generado."
    assert len(calls) == 1
    assert calls[0][1][-1] == {
        "role": "user",
        "content": "hazme un entrenamiento de CrossFit de 60 minutos",
    }


def test_training_agent_selects_configured_gemini_as_primary() -> None:
    client = RecordingProviderPromptClient()
    agent = TrainingAgent(client, _gemini_model_manager())

    response = agent.run(
        "glm4:9b",
        [{"role": "user", "content": "Sesión de CrossFit de 60 minutos."}],
    )

    assert response == "plan"
    assert len(client.calls) == 1
    assert client.calls[0]["model"] == "gemini-3.6-flash"
    assert client.calls[0]["provider_id"] == "gemini"


def test_training_agent_falls_back_to_local_qwen_when_gemini_fails() -> None:
    client = RecordingProviderPromptClient()
    client.fail_models = {"gemini-3.6-flash"}
    agent = TrainingAgent(client, _gemini_model_manager())

    response = agent.run(
        "glm4:9b",
        [{"role": "user", "content": "Sesión de CrossFit de 60 minutos."}],
    )

    assert response == "plan"
    assert [call["model"] for call in client.calls] == [
        "gemini-3.6-flash",
        "qwen3.6:latest",
    ]
    assert client.calls[1]["provider_id"] == "ollama"
    assert all(call["model"] != "glm4:9b" for call in client.calls)


def test_training_agent_never_uses_glm4_when_gemini_is_not_registered() -> None:
    client = RecordingProviderPromptClient()
    agent = TrainingAgent(client, ModelManager(client=_NoModels()))

    response = agent.run(
        "glm4:9b",
        [{"role": "user", "content": "Sesión de CrossFit de 60 minutos."}],
    )

    assert response == "plan"
    assert [call["model"] for call in client.calls] == ["qwen3.6:latest"]
    assert client.calls[0]["provider_id"] == "ollama"


def test_training_agent_without_model_manager_keeps_model_passthrough() -> None:
    client = RecordingPromptClient()
    agent = TrainingAgent(client)

    response = agent.run("some-other-model", [{"role": "user", "content": "WOD"}])

    assert response == "plan"
    assert client.calls == [
        (
            "some-other-model",
            [
                {"role": "system", "content": agent.SYSTEM_PROMPT},
                {"role": "user", "content": "WOD"},
            ],
        )
    ]
