from __future__ import annotations

import json

import pytest

from agents.base_agent import AgentResponse
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


def test_nutrition_agent_requests_all_required_data_for_an_empty_calculation() -> None:
    client = RecordingPromptClient()
    agent = NutritionAgent(client)  # type: ignore[arg-type]

    response = agent.run(
        "simulated-model",
        [{"role": "user", "content": "Calcula mis macros personalizados."}],
    )

    assert response == AgentResponse(
        text=(
            "Para calcularlo necesito tu objetivo, tu peso, tu altura, tu edad, "
            "tu sexo y tu actividad o carga de entrenamiento."
        ),
        requires_follow_up=True,
    )
    assert client.calls == []


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


def test_nutrition_agent_requests_missing_data_before_a_personalized_calculation() -> None:
    client = RecordingPromptClient()
    agent = NutritionAgent(client)  # type: ignore[arg-type]

    response = agent.run(
        "simulated-model",
        [
            {
                "role": "user",
                "content": (
                    "Calcula aproximadamente mis calorías y macronutrientes diarios "
                    "para ganar masa muscular. Peso 74 kg, mido 1,80 m, entreno "
                    "CrossFit 5 días por semana y quiero subir de peso minimizando "
                    "la ganancia de grasa."
                ),
            },
        ],
    )

    assert response == AgentResponse(
        text="Para calcularlo necesito tu edad y tu sexo.",
        requires_follow_up=True,
    )
    assert client.calls == []


def test_nutrition_agent_calculates_locally_when_all_data_is_available() -> None:
    agent = NutritionAgent(RecordingPromptClient())  # type: ignore[arg-type]

    response = agent.local_calculation_fallback(
        [
            {"role": "user", "content": "Calcula mis calorías y macros para ganar masa: peso 74 kg, mido 1,80 m y entreno CrossFit 5 días por semana."},
            {"role": "user", "content": "48 años, hombre."},
        ]
    )

    assert response == AgentResponse(
        text=(
            "### Estimación inicial para ganar masa muscular\n\n"
            "- **Calorías:** 3,050 kcal/día\n"
            "- **Proteínas:** 148 g/día\n"
            "- **Grasas:** 67 g/día\n"
            "- **Carbohidratos:** 464 g/día\n\n"
            "Es una estimación basada en Mifflin-St Jeor, actividad alta por "
            "CrossFit 5 días/semana y un superávit moderado. Mantén estas cifras "
            "2-3 semanas y ajusta 100-150 kcal según peso, rendimiento y perímetros."
        ),
        requires_follow_up=False,
    )


def test_nutrition_agent_unwraps_fenced_structured_response() -> None:
    client = RecordingPromptClient()
    agent = NutritionAgent(client)  # type: ignore[arg-type]

    def respond(*, model: str, messages: list[dict[str, str]]) -> str:
        return '```json\n{\n  "text": "Respuesta visible.",\n  "requires_follow_up": false\n}\n```'

    client.ask = respond  # type: ignore[method-assign]

    response = agent.run(
        "simulated-model",
        [{"role": "user", "content": "Ajusta lo que me queda por comer hoy."}],
    )

    assert response == AgentResponse(text="Respuesta visible.", requires_follow_up=False)


def test_conversation_executes_existing_nutrition_agent_with_mocked_model(monkeypatch) -> None:
    orchestrator = Bootstrap.build()
    nutrition = orchestrator._registry.get("nutrition")
    assert nutrition is not None
    calls: list[tuple[str, list[dict[str, str]]]] = []

    monkeypatch.setattr(nutrition._client, "check_model_health", lambda *_args, **_kwargs: None)

    def respond(
        *,
        model: str,
        messages: list[dict[str, str]],
        provider_id: str | None = None,
    ) -> str:
        calls.append((model, messages))
        return "Orientacion nutricional generada."

    monkeypatch.setattr(nutrition._client, "ask", respond)

    response = orchestrator.process_prompt(
        "Calcula mis macros para ganar masa: hombre, 35 anos, 80 kg, 180 cm y entreno 5 dias.",
        confirm=lambda _prompt: "",
    )

    assert response == "Orientacion nutricional generada."
    assert len(calls) == 1
    assert calls[0][1][-1] == {
        "role": "user",
        "content": "Calcula mis macros para ganar masa: hombre, 35 anos, 80 kg, 180 cm y entreno 5 dias.",
    }


def test_daily_coach_valid_plan_needs_only_one_inference() -> None:
    client = RecordingPromptClient()
    calls: list[list[dict[str, str]]] = []

    def respond(*, model: str, messages: list[dict[str, str]]) -> str:
        calls.append(messages)
        return json.dumps(
            {
                "text": "A las 16:00 toma arroz y huevos.",
                "requires_follow_up": False,
                "plan": [
                    {
                        "label": "merienda",
                        "time": "16:00",
                        "foods": ["arroz", "huevos"],
                    }
                ],
            }
        )

    client.ask = respond  # type: ignore[method-assign]
    agent = NutritionAgent(client)  # type: ignore[arg-type]

    response = agent.run(
        "simulated-model",
        [
            {
                "role": "user",
                "content": (
                    "[CONTEXTO OPERATIVO AUTORITATIVO DE NUTRICIÓN — SOLO LECTURA]\n"
                    "Hora local actual: 15:00\n"
                    "Inventario disponible:\n"
                    "- arroz: 1.85 kg\n"
                    "- huevos: 9 unit\n"
                    "Horario de trabajo:\n"
                    "- 07:00-15:00\n"
                    "[FIN DEL CONTEXTO OPERATIVO AUTORITATIVO]\n\n"
                    "[PETICIÓN ACTUAL DEL USUARIO]\n"
                    "Prepárame lo que debo comer hoy utilizando únicamente "
                    "los alimentos que tengo en casa."
                ),
            }
        ],
    )

    assert response == AgentResponse(
        text="A las 16:00 toma arroz y huevos.",
        requires_follow_up=False,
    )
    assert len(calls) == 1


def test_daily_coach_retries_once_when_plan_uses_past_time() -> None:
    client = RecordingPromptClient()
    calls: list[list[dict[str, str]]] = []

    responses = iter(
        (
            json.dumps(
                {
                    "text": "Desayuna a las 08:00.",
                    "requires_follow_up": False,
                    "plan": [
                        {
                            "label": "desayuno",
                            "time": "08:00",
                            "foods": ["arroz", "huevos"],
                        }
                    ],
                }
            ),
            json.dumps(
                {
                    "text": "A las 16:00 toma arroz y huevos.",
                    "requires_follow_up": False,
                    "plan": [
                        {
                            "label": "merienda",
                            "time": "16:00",
                            "foods": ["arroz", "huevos"],
                        }
                    ],
                }
            ),
        )
    )

    def respond(*, model: str, messages: list[dict[str, str]]) -> str:
        calls.append(messages)
        return next(responses)

    client.ask = respond  # type: ignore[method-assign]
    agent = NutritionAgent(client)  # type: ignore[arg-type]

    response = agent.run(
        "simulated-model",
        [
            {
                "role": "user",
                "content": (
                    "[CONTEXTO OPERATIVO AUTORITATIVO DE NUTRICIÓN — SOLO LECTURA]\n"
                    "Hora local actual: 15:00\n"
                    "Inventario disponible:\n"
                    "- arroz: 1.85 kg\n"
                    "- huevos: 9 unit\n"
                    "[FIN DEL CONTEXTO OPERATIVO AUTORITATIVO]\n\n"
                    "[PETICIÓN ACTUAL DEL USUARIO]\n"
                    "Prepárame lo que debo comer hoy utilizando únicamente "
                    "los alimentos que tengo en casa."
                ),
            }
        ],
    )

    assert response == AgentResponse(
        text="A las 16:00 toma arroz y huevos.",
        requires_follow_up=False,
    )
    assert len(calls) == 2
    assert "08:00" in calls[1][-1]["content"]
    assert "15:00" in calls[1][-1]["content"]


def test_daily_coach_retries_once_when_plan_invents_food() -> None:
    client = RecordingPromptClient()
    calls: list[list[dict[str, str]]] = []

    responses = iter(
        (
            json.dumps(
                {
                    "text": "A las 16:00 toma pollo con arroz.",
                    "requires_follow_up": False,
                    "plan": [
                        {
                            "label": "merienda",
                            "time": "16:00",
                            "foods": ["pollo", "arroz"],
                        }
                    ],
                }
            ),
            json.dumps(
                {
                    "text": "A las 16:00 toma arroz y huevos.",
                    "requires_follow_up": False,
                    "plan": [
                        {
                            "label": "merienda",
                            "time": "16:00",
                            "foods": ["arroz", "huevos"],
                        }
                    ],
                }
            ),
        )
    )

    def respond(*, model: str, messages: list[dict[str, str]]) -> str:
        calls.append(messages)
        return next(responses)

    client.ask = respond  # type: ignore[method-assign]
    agent = NutritionAgent(client)  # type: ignore[arg-type]

    response = agent.run(
        "simulated-model",
        [
            {
                "role": "user",
                "content": (
                    "[CONTEXTO OPERATIVO AUTORITATIVO DE NUTRICIÓN — SOLO LECTURA]\n"
                    "Hora local actual: 15:00\n"
                    "Inventario disponible:\n"
                    "- arroz: 1.85 kg\n"
                    "- huevos: 9 unit\n"
                    "[FIN DEL CONTEXTO OPERATIVO AUTORITATIVO]\n\n"
                    "[PETICIÓN ACTUAL DEL USUARIO]\n"
                    "Prepárame lo que debo comer hoy utilizando únicamente "
                    "los alimentos que tengo en casa."
                ),
            }
        ],
    )

    assert response == AgentResponse(
        text="A las 16:00 toma arroz y huevos.",
        requires_follow_up=False,
    )
    assert len(calls) == 2
    assert "pollo" in calls[1][-1]["content"].casefold()


def test_regular_nutrition_keeps_existing_two_key_contract() -> None:
    client = RecordingPromptClient()

    def respond(*, model: str, messages: list[dict[str, str]]) -> str:
        return json.dumps(
            {
                "text": "Orientación nutricional normal.",
                "requires_follow_up": False,
            }
        )

    client.ask = respond  # type: ignore[method-assign]
    agent = NutritionAgent(client)  # type: ignore[arg-type]

    response = agent.run(
        "simulated-model",
        [{"role": "user", "content": "Qué como antes de entrenar CrossFit?"}],
    )

    assert response == AgentResponse(
        text="Orientación nutricional normal.",
        requires_follow_up=False,
    )


def test_daily_coach_stops_after_second_invalid_plan() -> None:
    client = RecordingPromptClient()
    calls: list[list[dict[str, str]]] = []

    responses = iter(
        (
            json.dumps(
                {
                    "text": "A las 08:00 toma pollo.",
                    "requires_follow_up": False,
                    "plan": [
                        {
                            "label": "desayuno",
                            "time": "08:00",
                            "foods": ["pollo"],
                        }
                    ],
                }
            ),
            json.dumps(
                {
                    "text": "A las 09:00 toma pollo.",
                    "requires_follow_up": False,
                    "plan": [
                        {
                            "label": "desayuno",
                            "time": "09:00",
                            "foods": ["pollo"],
                        }
                    ],
                }
            ),
        )
    )

    def respond(*, model: str, messages: list[dict[str, str]]) -> str:
        calls.append(messages)
        return next(responses)

    client.ask = respond  # type: ignore[method-assign]
    agent = NutritionAgent(client)  # type: ignore[arg-type]

    response = agent.run(
        "simulated-model",
        [
            {
                "role": "user",
                "content": (
                    "[CONTEXTO OPERATIVO AUTORITATIVO DE NUTRICIÓN — SOLO LECTURA]\n"
                    "Hora local actual: 15:00\n"
                    "Inventario disponible:\n"
                    "- arroz: 1.85 kg\n"
                    "- huevos: 9 unit\n"
                    "[FIN DEL CONTEXTO OPERATIVO AUTORITATIVO]\n\n"
                    "[PETICIÓN ACTUAL DEL USUARIO]\n"
                    "Prepárame lo que debo comer hoy utilizando únicamente "
                    "los alimentos que tengo en casa."
                ),
            }
        ],
    )

    assert response == AgentResponse(
        text=(
            "No puedo generar ahora un plan diario que cumpla de forma fiable "
            "el horario y el inventario registrados. Reintenta la petición."
        ),
        requires_follow_up=False,
    )
    assert len(calls) == 2
