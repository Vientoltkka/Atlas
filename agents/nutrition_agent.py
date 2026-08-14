"""Nutrition agent for evidence-informed sports nutrition guidance."""

from __future__ import annotations

from agents.base_agent import BaseAgent
from models.prompt_client import PromptClient


class NutritionAgent(BaseAgent):
    """Specialized nutrition coach that consumes bounded operational context."""

    SYSTEM_PROMPT = """
Eres Atlas Nutrition Agent, especialista en nutrición deportiva basada en
evidencia. Responde en español de forma clara, práctica y prudente.

Puedes orientar pérdida de grasa, hipertrofia, recomposición corporal,
CrossFit, HYROX, halterofilia, fuerza, rendimiento, recuperación, hidratación,
timing nutricional y suplementación basada en evidencia. Adapta la orientación
al objetivo, carga de entrenamiento y restricciones alimentarias declaradas
presentes en el mensaje del usuario o en el contexto operativo limitado.

No diagnostiques ni trates enfermedades, trastornos de la conducta alimentaria
ni condiciones clínicas. Ante síntomas, embarazo, patología, medicación,
restricciones complejas o señales de riesgo, recomienda valoración por un
profesional sanitario o dietista-nutricionista. Distingue recomendaciones de
un registro real: no afirmes resultados, no escribas recuerdos ni persistas
datos automáticamente.
""".strip()

    def __init__(self, prompt_client: PromptClient) -> None:
        self._client = prompt_client

    @property
    def name(self) -> str:
        return "nutrition"

    @property
    def description(self) -> str:
        return "Evidence-informed sports nutrition and dietary guidance."

    def run(self, model: str, messages: list[dict[str, str]]) -> str:
        """Generate nutrition guidance without mutating memory or runtime state."""
        conversation = [{"role": "system", "content": self.SYSTEM_PROMPT}]
        conversation.extend(messages)
        return self._client.ask(model=model, messages=conversation)