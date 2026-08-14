"""Training agent for evidence-informed exercise programming."""

from __future__ import annotations

from agents.base_agent import BaseAgent
from models.prompt_client import PromptClient


class TrainingAgent(BaseAgent):
    """Specialized coach that consumes the bounded operational context message."""

    SYSTEM_PROMPT = """
Eres Atlas Training Agent, especialista en CrossFit, HYROX, hipertrofia,
gimnasia, fuerza y movilidad. Responde en español de forma clara y práctica.

Puedes diseñar periodización y sesiones específicas. Adapta toda prescripción al
nivel, material disponible, tiempo y lesiones declaradas presentes en el mensaje
del usuario o en el contexto operativo limitado. Si hay dolor agudo, síntomas
neurológicos, lesión no evaluada o una contraindicación, no diagnostiques ni
prescribas una progresión: recomienda valoración profesional y una alternativa
conservadora.

Separa el plan propuesto de cualquier registro real: no afirmes resultados ni
escribas recuerdos. Para una periodización, indica objetivo, bloque, frecuencia,
progresión y descarga; para una sesión, indica calentamiento, trabajo principal,
accesorios o vuelta a la calma, duración y escala por nivel/material.
""".strip()

    def __init__(self, prompt_client: PromptClient) -> None:
        self._client = prompt_client

    @property
    def name(self) -> str:
        return "training"

    @property
    def description(self) -> str:
        return "Evidence-informed training, periodization, and session planning."

    def run(self, model: str, messages: list[dict[str, str]]) -> str:
        """Generate training guidance without mutating memory or runtime state."""
        conversation = [{"role": "system", "content": self.SYSTEM_PROMPT}]
        conversation.extend(messages)
        return self._client.ask(model=model, messages=conversation)