"""Legal agent for cautious legal information and analysis."""

from __future__ import annotations

from agents.base_agent import AgentResponse, BaseAgent
from models.prompt_client import PromptClient


class LegalAgent(BaseAgent):
    """Specialized legal agent that consumes bounded operational context."""

    SYSTEM_PROMPT = """
    Eres Atlas Legal Agent. Responde en espanol, Markdown limpio, de forma clara,
    prudente y orientada a informacion general.

    Explica conceptos legales, contratos y clausulas en lenguaje claro, derechos y
    obligaciones generales, relaciones laborales basicas y consumo o reclamaciones
    basicas. Puedes senalar cuestiones que conviene revisar, usando solo los hechos
    aportados. Pide una sola aclaracion breve solo si falta un dato imprescindible;
    si la respuesta depende de la legislacion aplicable y no consta la jurisdiccion,
    pide el pais o jurisdiccion antes de dar una respuesta especifica.

    No inventes ni cites leyes, articulos, sentencias, plazos o jurisdicciones que no
    hayas recibido. No asegures resultados legales. No ofrezcas representacion legal.
    No sustituyas el asesoramiento de un profesional cualificado. Cuando haya plazos,
    notificaciones, conflicto, sancion, procedimiento en curso, riesgo relevante o
    consecuencias importantes, recomienda asesoramiento profesional en la jurisdiccion
    aplicable.

    No ejecutes tramites, reclamaciones, notificaciones ni otras acciones legales;
    limita la respuesta a explicacion y orientacion general. Distingue dominios:
    presupuestos e inversiones son de Finance; sintomas o salud son de Medical;
    dieta, macros y suplementacion son de Nutrition; y CrossFit, HYROX o fuerza son
    del Coach. No escribas recuerdos ni persistas datos automaticamente.
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

    def local_calculation_fallback(
        self, messages: list[dict[str, str]]
    ) -> AgentResponse | None:
        """Provide bounded contract-clause guidance when no provider is available."""
        prompt = next(
            (
                message.get("content", "").casefold()
                for message in reversed(messages)
                if message.get("role") == "user"
            ),
            "",
        )
        if "contrat" not in prompt or ("clausula" not in prompt and "cláusula" not in prompt):
            return None
        return AgentResponse(
            text=(
                "Esta es orientación jurídica general, no un dictamen jurídico. Sin el "
                "texto exacto ni la legislación aplicable no se puede determinar si una "
                "cláusula es abusiva o ilegal.\n\n"
                "Para revisarla, conviene comprobar:\n"
                "- El texto literal de la cláusula y su contexto dentro del contrato.\n"
                "- El país o jurisdicción aplicable.\n"
                "- El tipo de contrato y las partes que lo han firmado.\n"
                "- Las obligaciones que impone a cada parte.\n"
                "- Las penalizaciones, gastos o costes asociados.\n"
                "- La duración, renovación y causas de resolución.\n"
                "- Posibles renuncias a derechos.\n"
                "- Si contradice normas imperativas aplicables.\n"
                "- Si está redactada de forma transparente, clara y comprensible.\n\n"
                "Revisa el texto exacto junto con la normativa de la jurisdicción "
                "aplicable. Si afecta a una obligación relevante, un conflicto, un plazo "
                "o una penalización, consulta a un profesional jurídico cualificado."
            )
        )
