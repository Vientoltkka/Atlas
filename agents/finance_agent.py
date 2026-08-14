"""Finance agent for educational market and cryptocurrency analysis."""

from __future__ import annotations

from agents.base_agent import BaseAgent
from models.prompt_client import PromptClient


class FinanceAgent(BaseAgent):
    """Specialized finance agent that consumes bounded operational context."""

    SYSTEM_PROMPT = """
Eres Atlas Finance Agent, especialista en mercados financieros y criptomonedas.
Responde en espanol de forma clara, prudente y educativa.

Puedes explicar acciones, ETFs, indices y bonos; criptomonedas y DeFi; analisis
fundamental y analisis tecnico; gestion del riesgo; asignacion de activos y
diversificacion; DCA y rebalanceo; macroeconomia aplicada; interpretacion de
resultados empresariales; escenarios probabilisticos y conceptos financieros.

No ofrezcas recomendaciones de inversion personalizadas ni garantices
rendimientos. Explica incertidumbres, riesgos, costes, liquidez y volatilidad.
Toda informacion es educativa y no sustituye asesoramiento financiero profesional.

Adapta la informacion al contexto operativo limitado y a las restricciones
presentes en el mensaje del usuario. Distingue analisis de acciones reales: no
afirmes resultados, no escribas recuerdos ni persistas datos automaticamente.
""".strip()

    def __init__(self, prompt_client: PromptClient) -> None:
        self._client = prompt_client

    @property
    def name(self) -> str:
        return "finance"

    @property
    def description(self) -> str:
        return "Educational financial markets and cryptocurrency analysis."

    def run(self, model: str, messages: list[dict[str, str]]) -> str:
        """Generate finance guidance without mutating memory or runtime state."""
        conversation = [{"role": "system", "content": self.SYSTEM_PROMPT}]
        conversation.extend(messages)
        return self._client.ask(model=model, messages=conversation)