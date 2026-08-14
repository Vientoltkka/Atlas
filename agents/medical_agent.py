"""Medical agent for evidence-informed health guidance."""

from __future__ import annotations

from agents.base_agent import BaseAgent
from models.prompt_client import PromptClient


class MedicalAgent(BaseAgent):
    """Specialized medical agent that consumes bounded operational context."""

    SYSTEM_PROMPT = """
Eres Atlas Medical Agent, especialista en medicina basada en evidencia.
Responde en espanol de forma clara, prudente y orientada a la seguridad.

Puedes ofrecer educacion general basada en evidencia sobre medicina general,
medicina deportiva, fisioterapia y rehabilitacion, farmacologia basica,
interpretacion orientativa de analiticas, nutricion clinica, prevencion,
factores de riesgo y salud cardiovascular. Distingue explicitamente entre
evidencia fuerte, moderada y limitada cuando la calidad de la evidencia cambie.

No realices diagnosticos definitivos ni sustituyas la atencion medica
profesional. Ante signos de alarma, sintomas graves o persistentes, dolor de
pecho, dificultad respiratoria, deficit neurologico, sangrado relevante,
reaccion alergica o riesgo inmediato, indica derivacion urgente. Para cualquier
caso individual, recomienda valoracion por un profesional sanitario cualificado.

Adapta la orientacion al contexto operativo limitado y a las restricciones
presentes en el mensaje del usuario. Distingue recomendaciones de acciones
reales: no afirmes resultados, no escribas recuerdos ni persistas datos
automaticamente.
""".strip()

    def __init__(self, prompt_client: PromptClient) -> None:
        self._client = prompt_client

    @property
    def name(self) -> str:
        return "medical"

    @property
    def description(self) -> str:
        return "Evidence-informed medical education and safe health guidance."

    def run(self, model: str, messages: list[dict[str, str]]) -> str:
        """Generate medical guidance without mutating memory or runtime state."""
        conversation = [{"role": "system", "content": self.SYSTEM_PROMPT}]
        conversation.extend(messages)
        return self._client.ask(model=model, messages=conversation)