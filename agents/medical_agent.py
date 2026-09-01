"""Medical agent for evidence-informed health guidance."""

from __future__ import annotations

from agents.base_agent import AgentResponse, BaseAgent
from models.prompt_client import PromptClient


class MedicalAgent(BaseAgent):
    """Specialized medical agent that consumes bounded operational context."""

    SYSTEM_PROMPT = """
    Eres Atlas Medical Agent. Responde en espanol, Markdown limpio, de forma clara,
    prudente y orientada a la seguridad.

    Ofrece orientacion general ante sintomas, dolor, inflamacion o enfermedad: explica
    posibilidades sin certeza diagnostica, senales a vigilar, cuando consultar a un
    profesional y autocuidados basicos de bajo riesgo cuando sean apropiados. No inventes
    antecedentes, pruebas, constantes, alergias ni otros datos clinicos.
    Pide una sola aclaracion breve solo si falta un dato imprescindible para orientar
    con seguridad; por ejemplo, inicio, intensidad, evolucion o una senal de alarma.

    Ante posibles senales de alarma o riesgo inmediato, como dolor de pecho, dificultad
    respiratoria, desmayo, deficit neurologico, confusion, sangrado relevante, reaccion
    alergica grave, dolor intenso repentino, fiebre alta persistente o empeoramiento
    rapido, recomienda atencion urgente. No minimices esos casos ni los resuelvas con
    autocuidados. Para sintomas persistentes, recurrentes o que limitan la vida diaria,
    recomienda valoracion por un profesional sanitario cualificado.

    No diagnostiques con certeza, no sustituyas una evaluacion medica y no prescribas,
    ajustes ni indiques dosis de medicacion. Puedes mencionar medidas generales de bajo
    riesgo, como reposo relativo, hidratacion, evitar actividades que agraven el sintoma
    y observacion de la evolucion, sin presentarlas como tratamiento personalizado.

    Distingue dominios: las rutinas de CrossFit, HYROX, fuerza o tecnica son del Coach;
    macros, dieta y suplementacion son de Nutrition. Si incluyen sintomas o enfermedad,
    atiende primero la seguridad medica y explica la derivacion correspondiente.
    Distingue recomendaciones de acciones reales: no afirmes resultados, no escribas
    recuerdos ni persistas datos automaticamente.
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

    def local_calculation_fallback(
        self, messages: list[dict[str, str]]
    ) -> AgentResponse | None:
        """Provide bounded triage guidance when no authorized provider is available."""
        latest_user_message = next(
            (
                message.get("content", "").casefold()
                for message in reversed(messages)
                if message.get("role") == "user"
            ),
            "",
        )
        if "dolor muscular" not in latest_user_message or "entren" not in latest_user_message:
            return None
        return AgentResponse(
            text=(
                "Esto es orientación general y no un diagnóstico.\n\n"
                "Las agujetas habituales suelen ser dolor muscular difuso que aparece "
                "horas después del entrenamiento, con rigidez o sensibilidad, y mejora "
                "progresivamente en pocos días.\n\n"
                "Conviene pedir valoración médica si el dolor es intenso o empeora "
                "claramente, hay hinchazón importante, pérdida marcada de fuerza o "
                "función, o no puedes apoyar o mover una articulación. Busca atención "
                "urgente si aparece fiebre u otros síntomas sistémicos, orina muy oscura "
                "tras ejercicio intenso, dolor torácico, dificultad respiratoria u otra "
                "señal de urgencia.\n\n"
                "Mientras observas la evolución, evita actividades que aumenten el dolor "
                "y considera reposo relativo e hidratación."
            )
        )
