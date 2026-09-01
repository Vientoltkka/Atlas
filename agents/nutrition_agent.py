"""Nutrition agent for evidence-informed sports nutrition guidance."""

from __future__ import annotations

import json
import re

from agents.base_agent import AgentResponse, BaseAgent
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

    def run(self, model: str, messages: list[dict[str, str]]) -> str | AgentResponse:
        """Generate nutrition guidance without mutating memory or runtime state."""
        preflight_response = self.preflight(messages)
        if preflight_response is not None:
            return preflight_response
        conversation = [{"role": "system", "content": self.SYSTEM_PROMPT}]
        conversation.extend(messages)
        response = self._client.ask(model=model, messages=conversation)
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
