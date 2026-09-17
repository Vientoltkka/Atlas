"""Nutrition agent for evidence-informed sports nutrition guidance."""

from __future__ import annotations

import json
import re

from agents.base_agent import AgentResponse, BaseAgent, ask_prompt_client
from models.prompt_client import PromptClient


class NutritionAgent(BaseAgent):
    """Specialized nutrition coach that consumes bounded operational context."""

    SYSTEM_PROMPT = """
    Eres Atlas Nutrition Agent, especialista en nutrición deportiva basada en
    evidencia. Responde en español, Markdown limpio, de forma clara, práctica y
    prudente.

    Puedes orientar pérdida de grasa, ganancia muscular, hipertrofia, recomposición
    corporal, CrossFit, HYROX, halterofilia, fuerza, distribución de comidas,
    rendimiento, recuperación, hidratación, pre/post entrenamiento, timing nutricional
    y suplementación basada en evidencia, incluida la básica. Adapta la orientación al objetivo, carga de entrenamiento,
    preferencias, alimentos disponibles y restricciones alimentarias declaradas en
    el mensaje del usuario o en el contexto operativo limitado.

    Si el contexto operativo declara un inventario disponible y la petición pide
    planificar con lo disponible, en casa o con ese inventario, trátalo como una lista
    exhaustiva: usa únicamente esos alimentos. No inventes ni presupongas alimentos,
    suplementos o ingredientes adicionales. Si con ese inventario no se pueden cubrir
    los objetivos, dilo claramente y señala qué falta sin presentarlo como disponible.

    No presupongas si una cantidad de un alimento registrada por el usuario estaba
    cruda o cocida. Si esa diferencia cambia de forma material calorías o macros y el
    estado no está registrado, pide una aclaración breve cuando sea imprescindible o
    indica que el cálculo no puede ser preciso hasta conocerlo. Nunca elijas
    silenciosamente crudo o cocido.

    Pide una sola aclaración breve solo si falta un dato imprescindible para la
    petición concreta. Para calcular calorías o macros personalizados, solicita los
    datos necesarios que no se hayan aportado: objetivo, peso, altura, edad, sexo y
    actividad o carga de entrenamiento. No asumas ni inventes esos datos, calorías,
    macros, alergias, preferencias o alimentos disponibles. Si no se pide un cálculo
    personalizado o falta información no crítica, usa alternativas prácticas o
    porciones orientativas y declara cualquier supuesto útil.

    Cuando haya datos suficientes, muestra el objetivo, estimación de calorías,
    proteínas, carbohidratos y grasas, distribución diaria de comidas y ajustes
    simples según rendimiento, hambre, adherencia o evolución. Para planes de
    ganancia muscular o pérdida de grasa, prioriza hábitos sostenibles, proteína,
    energía suficiente y un ritmo gradual; no prometas resultados. Para pre/post
    entrenamiento, concreta objetivo, opciones de alimentos, cantidad o porción
    orientativa y momento. Para hidratación, ofrece pautas básicas y ajustables.

    En suplementación, limita la orientación a opciones básicas con evidencia,
    finalidad, uso general y precauciones; no prescribas medicación ni sustituyas
    atención profesional. Presenta recomendaciones ejecutables con listas o tablas
    breves y sustituciones acordes a preferencias y alimentos disponibles.

    No diagnostiques ni trates enfermedades, trastornos de la conducta alimentaria
    ni condiciones clínicas. Ante síntomas, dolor, enfermedad, embarazo, patología,
    medicación, restricciones complejas o señales de riesgo, no calcules ni
    prescribas: indica que requiere valoración por un profesional sanitario o
    dietista-nutricionista. Distingue recomendaciones de un registro real: no
    afirmes resultados, no escribas recuerdos ni persistas datos automáticamente.

    Devuelve exclusivamente un objeto JSON con las claves "text" y
    "requires_follow_up". "text" contiene la respuesta visible en español y
    "requires_follow_up" es true únicamente si necesitas que el usuario aporte
    otro dato para continuar; en caso contrario es false.
    """.strip()

    def __init__(self, prompt_client: PromptClient) -> None:
        self._client = prompt_client

    @property
    def name(self) -> str:
        return "nutrition"

    @property
    def description(self) -> str:
        return "Evidence-informed sports nutrition and dietary guidance."

    def run(
        self,
        model: str,
        messages: list[dict[str, str]],
        *,
        provider_id: str | None = None,
    ) -> str | AgentResponse:
        """Generate nutrition guidance without mutating memory or runtime state."""
        preflight_response = self.preflight(messages)
        if preflight_response is not None:
            return preflight_response

        operational = _daily_operational_constraints(messages)

        conversation = [{"role": "system", "content": self.SYSTEM_PROMPT}]
        conversation.extend(messages)

        if operational is not None:
            conversation.append(
                {
                    "role": "system",
                    "content": (
                        "Para esta petición Daily Coach añade una clave JSON adicional "
                        "\"plan\". Debe ser una lista de comidas. Cada comida debe tener "
                        "exactamente \"label\", \"time\" y \"foods\". \"time\" usa HH:MM "
                        "y \"foods\" contiene únicamente nombres del inventario disponible. "
                        "La clave \"text\" debe corresponder al mismo plan."
                    ),
                }
            )

        response = ask_prompt_client(
            self._client,
            model,
            conversation,
            provider_id,
        )

        if operational is None:
            return _parse_standard_response(response)

        parsed, errors = _parse_and_validate_daily_response(
            response,
            operational,
        )


        if not errors and parsed is not None:
            return parsed

        correction = (
            "Tu propuesta anterior incumple restricciones operativas autoritativas. "
            "Corrige únicamente estos errores:\n- "
            + "\n- ".join(errors)
            + "\nDebes conservar EXACTAMENTE este contrato JSON: "
            "{\"text\": string, \"requires_follow_up\": boolean, "
            "\"plan\": [{\"label\": string, \"time\": \"HH:MM\", "
            "\"foods\": [string]}]}. "
            "No cambies los nombres de las claves ni el tipo de sus valores. "
            "Todas las horas del plan deben ser iguales o posteriores a la hora "
            "local actual indicada en el contexto. "
            "Todos los nombres de foods deben coincidir con alimentos del "
            "inventario exhaustivo. "
            "Si por la hora actual ya no procede planificar ninguna comida hoy, "
            "devuelve \"plan\": [] y explícalo brevemente en text. "
            "Devuelve exclusivamente el objeto JSON, sin markdown."
        )

        retry_conversation = [
            *conversation,
            {"role": "assistant", "content": response},
            {"role": "user", "content": correction},
        ]

        retry = ask_prompt_client(
            self._client,
            model,
            retry_conversation,
            provider_id,
        )

        parsed, retry_errors = _parse_and_validate_daily_response(
            retry,
            operational,
        )


        if not retry_errors and parsed is not None:
            return parsed

        return AgentResponse(
            text=(
                "No puedo generar ahora un plan diario que cumpla de forma fiable "
                "el horario y el inventario registrados. Reintenta la petición."
            ),
            requires_follow_up=False,
        )
    def preflight(self, messages: list[dict[str, str]]) -> AgentResponse | None:
        """Return a local clarification before the model health check when needed."""
        missing_data = _missing_calculation_data(messages)
        if not missing_data:
            return None
        return AgentResponse(
            text=(
                "Para calcularlo necesito "
                f"{_format_missing_data(missing_data)}."
            ),
            requires_follow_up=True,
        )

    def local_calculation_fallback(
        self, messages: list[dict[str, str]]
    ) -> AgentResponse | None:
        """Return a bounded calorie/macronutrient estimate without a provider."""
        data = _calculation_data(messages)
        if data is None:
            return None
        weight, height_cm, age, sex, crossfit_days = data
        bmr = 10 * weight + 6.25 * height_cm - 5 * age + (5 if sex == "hombre" else -161)
        activity_factor = 1.725 if crossfit_days >= 5 else 1.55
        calories = round((bmr * activity_factor + 250) / 50) * 50
        protein = round(weight * 2)
        fat = round(weight * 0.9)
        carbohydrates = round((calories - protein * 4 - fat * 9) / 4)
        return AgentResponse(
            text=(
                "### Estimación inicial para ganar masa muscular\n\n"
                f"- **Calorías:** {calories:,} kcal/día\n"
                f"- **Proteínas:** {protein} g/día\n"
                f"- **Grasas:** {fat} g/día\n"
                f"- **Carbohidratos:** {carbohydrates} g/día\n\n"
                "Es una estimación basada en Mifflin-St Jeor, actividad alta por "
                f"CrossFit {crossfit_days} días/semana y un superávit moderado. "
                "Mantén estas cifras 2-3 semanas y ajusta 100-150 kcal según peso, "
                "rendimiento y perímetros."
            ),
            requires_follow_up=False,
        )


def _parse_standard_response(response: str) -> str | AgentResponse:
    """Parse the existing two-key Nutrition response contract."""
    try:
        payload = json.loads(_structured_response_content(response))
    except (TypeError, json.JSONDecodeError):
        return response

    if (
        not isinstance(payload, dict)
        or set(payload) != {"text", "requires_follow_up"}
        or not isinstance(payload["text"], str)
        or not isinstance(payload["requires_follow_up"], bool)
    ):
        return response

    return AgentResponse(
        text=payload["text"],
        requires_follow_up=payload["requires_follow_up"],
    )


def _daily_operational_constraints(
    messages: list[dict[str, str]],
) -> tuple[str, frozenset[str]] | None:
    """Extract current clock and exhaustive inventory from authoritative context."""
    content = "\n".join(
        message.get("content", "")
        for message in messages
    )

    marker = "[CONTEXTO OPERATIVO AUTORITATIVO DE NUTRICIÓN"
    if marker not in content:
        return None

    clock_match = re.search(
        r"Hora local actual:\s*(\d{2}:\d{2})",
        content,
        re.IGNORECASE,
    )
    if clock_match is None:
        return None

    inventory_match = re.search(
        r"Inventario disponible:\s*\n(.*?)(?=\nHorario de trabajo:|\Z)",
        content,
        re.DOTALL | re.IGNORECASE,
    )
    if inventory_match is None:
        return None

    foods: set[str] = set()

    for line in inventory_match.group(1).splitlines():
        match = re.match(r"\s*-\s*([^:]+):", line)
        if match:
            foods.add(match.group(1).strip().casefold())

    return clock_match.group(1), frozenset(foods)


def _parse_and_validate_daily_response(
    response: str,
    operational: tuple[str, frozenset[str]],
) -> tuple[AgentResponse | None, list[str]]:
    """Validate machine-readable Daily Coach facts before exposing model text."""
    try:
        payload = json.loads(_structured_response_content(response))
    except (TypeError, json.JSONDecodeError):
        return None, ["La respuesta no es JSON válido."]

    if not isinstance(payload, dict):
        return None, ["La respuesta JSON no es un objeto."]

    if set(payload) != {"text", "requires_follow_up", "plan"}:
        return None, [
            "El JSON debe contener exactamente text, requires_follow_up y plan."
        ]

    if (
        not isinstance(payload["text"], str)
        or not isinstance(payload["requires_follow_up"], bool)
        or not isinstance(payload["plan"], list)
    ):
        return None, ["Los tipos del contrato Daily Coach no son válidos."]

    current_clock, inventory = operational
    errors: list[str] = []

    for index, meal in enumerate(payload["plan"], start=1):
        if not isinstance(meal, dict) or set(meal) != {"label", "time", "foods"}:
            errors.append(
                f"La comida {index} debe contener exactamente label, time y foods."
            )
            continue

        label = meal["label"]
        meal_time = meal["time"]
        foods = meal["foods"]

        if (
            not isinstance(label, str)
            or not isinstance(meal_time, str)
            or re.fullmatch(r"\d{2}:\d{2}", meal_time) is None
            or not isinstance(foods, list)
            or not all(isinstance(food, str) for food in foods)
        ):
            errors.append(f"La comida {index} tiene datos inválidos.")
            continue

        if meal_time < current_clock:
            errors.append(
                f"La comida {index} usa {meal_time}, anterior a la hora actual "
                f"{current_clock}."
            )

        unknown = sorted(
            {
                food.strip().casefold()
                for food in foods
                if food.strip().casefold() not in inventory
            }
        )

        if unknown:
            errors.append(
                "La comida "
                f"{index} usa alimentos fuera del inventario: {', '.join(unknown)}."
            )

    if errors:
        return None, errors

    return (
        AgentResponse(
            text=payload["text"],
            requires_follow_up=payload["requires_follow_up"],
        ),
        [],
    )

def _structured_response_content(response: str) -> str:
    """Return the full JSON payload when the model encloses it in a JSON fence."""
    content = response.strip()
    if content.startswith("```json") and content.endswith("```"):
        return content[7:-3].strip()
    fenced_payloads = re.findall(r"```json\s*(\{.*?\})\s*```", content, re.DOTALL)
    if fenced_payloads:
        return fenced_payloads[-1]
    return content


def _calculation_data(
    messages: list[dict[str, str]],
) -> tuple[float, float, int, str, int] | None:
    """Extract only the values needed for a completed basic calculation."""
    user_content = "\n".join(
        message["content"] for message in messages if message.get("role") == "user"
    ).casefold()
    if not re.search(r"\bcalcul\w*\b", user_content) or not re.search(
        r"\b(calor[ií]as?|macros?|macronutrientes?)\b", user_content
    ):
        return None
    if _missing_calculation_data(messages):
        return None
    weight_match = re.search(r"\b(\d{2,3}(?:[.,]\d+)?)\s*(?:kg|kilos?)\b", user_content)
    height_match = re.search(r"\b(\d[.,]\d{1,2})\s*m\b|\b(\d{3})\s*(?:cm|cent[ií]metros?)\b", user_content)
    age_match = re.search(r"\b(\d{1,3})\s*a[nñ]os?\b", user_content)
    crossfit_match = re.search(r"crossfit\s*(\d+)\s*d[ií]as?", user_content)
    if not weight_match or not height_match or not age_match:
        return None
    weight = float(weight_match.group(1).replace(",", "."))
    height_cm = float((height_match.group(1) or height_match.group(2)).replace(",", "."))
    if height_cm < 10:
        height_cm *= 100
    sex = "hombre" if re.search(r"\b(hombre|masculino|var[oó]n)\b", user_content) else "mujer"
    return weight, height_cm, int(age_match.group(1)), sex, int(crossfit_match.group(1)) if crossfit_match else 4


def _missing_calculation_data(messages: list[dict[str, str]]) -> tuple[str, ...]:
    """Return required personal data absent from a requested calorie/macronutrient calculation."""
    user_content = "\n".join(
        message["content"]
        for message in messages
        if message.get("role") == "user"
    ).casefold()
    if not re.search(r"\bcalcul\w*\b", user_content) or not re.search(
        r"\b(calor[ií]as?|macros?|macronutrientes?)\b", user_content
    ):
        return ()

    missing = []
    if not re.search(r"\b(ganar|perder|mantener|subir|bajar|masa|hipertrof|recompos)\w*\b", user_content):
        missing.append("tu objetivo")
    if not re.search(r"\b\d{2,3}(?:[.,]\d+)?\s*(?:kg|kilos?)\b", user_content):
        missing.append("tu peso")
    if not re.search(
        r"\b(?:\d[.,]\d{1,2}\s*m|\d{3}\s*(?:cm|cent[ií]metros?))\b",
        user_content,
    ):
        missing.append("tu altura")
    if not re.search(r"\b\d{1,3}\s*a[nñ]os?\b", user_content):
        missing.append("tu edad")
    if not re.search(
        r"\b(hombre|mujer|masculino|femenino|var[oó]n|hembra)\b",
        user_content,
    ):
        missing.append("tu sexo")
    if not re.search(
        r"\b(entren\w*|crossfit|hyrox|actividad|sedentari\w*|ejercicio)\b",
        user_content,
    ):
        missing.append("tu actividad o carga de entrenamiento")
    return tuple(missing)


def _format_missing_data(missing_data: tuple[str, ...]) -> str:
    if len(missing_data) == 1:
        return missing_data[0]
    if len(missing_data) == 2:
        return f"{missing_data[0]} y {missing_data[1]}"
    return f"{', '.join(missing_data[:-1])} y {missing_data[-1]}"
