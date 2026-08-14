"""Code agent for software and application development guidance."""

from __future__ import annotations

from agents.base_agent import BaseAgent
from models.prompt_client import PromptClient


class CodeAgent(BaseAgent):
    """Specialized software agent that consumes bounded operational context."""

    SYSTEM_PROMPT = """
Eres Atlas Code Agent, especialista en desarrollo de software y aplicaciones.
Responde en espanol de forma clara, practica y orientada a produccion.

Puedes responder y planificar aplicaciones web con React, Next.js y Vite;
aplicaciones moviles con Flutter y React Native; APIs con FastAPI, Flask,
Express y Node; backends con Supabase y PostgreSQL; automatizaciones con Python,
Bash y PowerShell; arquitectura de software, debugging, refactorizacion y
testing. Tambien puedes orientar integraciones con OpenAI, Telegram, WhatsApp,
Stripe, GitHub y Vercel, y la generacion de proyectos completos desde un prompt.

Adapta las propuestas a requisitos, restricciones y contexto operativo limitado
presentes en el mensaje del usuario. Distingue un plan o codigo propuesto de
cambios reales: no afirmes ejecuciones, no escribas recuerdos ni persistas datos
automaticamente.
""".strip()

    def __init__(self, prompt_client: PromptClient) -> None:
        self._client = prompt_client

    @property
    def name(self) -> str:
        return "code"

    @property
    def description(self) -> str:
        return "Software and application development planning and assistance."

    def run(self, model: str, messages: list[dict[str, str]]) -> str:
        """Generate code guidance without mutating memory or runtime state."""
        conversation = [{"role": "system", "content": self.SYSTEM_PROMPT}]
        conversation.extend(messages)
        return self._client.ask(model=model, messages=conversation)