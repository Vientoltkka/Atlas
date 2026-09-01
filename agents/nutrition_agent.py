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
        missing_data = _missing_calculation_data(messages)
        if missing_data:
            return AgentResponse(
                text=(
                    "Para calcularlo necesito "
                    f"{_format_missing_data(missing_data)}."
                ),
                requires_follow_up=True,
            )
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


def _structured_response_content(response: str) -> str:
    """Return the full JSON payload when the model encloses it in a JSON fence."""
    content = response.strip()
    if content.startswith("```json") and content.endswith("```"):
        return content[7:-3].strip()
    return content


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
