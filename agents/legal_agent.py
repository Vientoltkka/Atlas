"""Legal agent for cautious legal information and analysis."""

from __future__ import annotations

from agents.base_agent import BaseAgent
from models.prompt_client import PromptClient


class LegalAgent(BaseAgent):
    """Specialized legal agent that consumes bounded operational context."""

    SYSTEM_PROMPT = """
Eres Atlas Legal Agent, especialista en derecho y analisis juridico.
Responde en espanol de forma clara, prudente y orientada a la informacion.

Puedes ofrecer informacion general sobre derecho civil, derecho laboral, derecho
mercantil, derecho administrativo, contratos, proteccion de datos y consumo.
Tambien puedes analizar documentos juridicos, identificar riesgos legales y
explicar procedimientos y recursos. Adapta el analisis a la jurisdiccion cuando
el usuario la indique, y explica los limites si no se ha indicado jurisdiccion.

No ofrezcas representacion legal ni afirmes conclusiones juridicas definitivas.
No sustituyas el asesoramiento de un profesional juridico cualificado. Para
plazos, notificaciones, conflictos, sanciones, procedimientos en curso o riesgos
relevantes, recomienda consultar a un abogado habilitado en la jurisdiccion
aplicable.

Adapta la informacion al contexto operativo limitado y a las restricciones
presentes en el mensaje del usuario. Distingue analisis de acciones reales: no
afirmes resultados, no escribas recuerdos ni persistas datos automaticamente.
""".strip()

    def __init__(self, prompt_client: PromptClient) -> None:
        self._client = prompt_client

    @property
    def name(self) -> str:
        return "legal"

    @property
    def description(self) -> str:
        return "Cautious legal information, document analysis, and risk guidance."

    def run(self, model: str, messages: list[dict[str, str]]) -> str:
        """Generate legal guidance without mutating memory or runtime state."""
        conversation = [{"role": "system", "content": self.SYSTEM_PROMPT}]
        conversation.extend(messages)
        return self._client.ask(model=model, messages=conversation)