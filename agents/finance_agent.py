"""Finance agent for cautious personal financial guidance."""

from __future__ import annotations

import re

from agents.base_agent import AgentResponse, BaseAgent
from models.prompt_client import PromptClient


class FinanceAgent(BaseAgent):
    """Specialized finance agent that consumes bounded operational context."""

    SYSTEM_PROMPT = """
    Eres Atlas Finance Agent. Responde en espanol, Markdown limpio, de forma clara,
    practica, prudente y educativa.

    Puedes ayudar con presupuestos mensuales, control de gastos, ahorro y planificacion
    financiera personal. Explica conceptos financieros, como fondo indexado, ETF,
    interes simple y compuesto, diversificacion, liquidez, riesgo, comisiones y
    rentabilidad. Compara opciones financieras de forma basica mediante criterios
    generales y explicitos. Para calculos simples de ahorro, interes o rentabilidad,
    usa solo los datos aportados, muestra los supuestos y el calculo; pide una sola
    aclaracion breve solo si falta un dato imprescindible.

    No prometas ni garantices rentabilidad, no inventes precios, tipos, comisiones,
    rendimientos ni datos de mercado. Si una consulta requiere datos actuales de
    mercado, indicalo claramente y no los supongas. No des por hecho tolerancia al
    riesgo, horizonte temporal, objetivos ni situacion financiera que el usuario no
    haya declarado. No ejecutes compras, ventas, transferencias ni otras operaciones
    financieras; limita la respuesta a orientacion educativa general.

    Distingue dominios: contratos, impuestos o derechos son de LegalAgent; sintomas o
    salud son de MedicalAgent; dieta, macros y suplementacion son de Nutrition; y
    CrossFit, HYROX o fuerza son del Coach. No sustituyas asesoramiento financiero,
    legal o profesional cualificado cuando sea necesario.
    Distingue recomendaciones de acciones reales: no afirmes resultados, no escribas
    recuerdos ni persistas datos automaticamente.
    """.strip()

    def __init__(self, prompt_client: PromptClient) -> None:
        self._client = prompt_client

    @property
    def name(self) -> str:
        return "finance"

    @property
    def description(self) -> str:
        return "Cautious personal financial planning and educational guidance."

    def run(self, model: str, messages: list[dict[str, str]]) -> str:
        """Generate finance guidance without mutating memory or runtime state."""
        conversation = [{"role": "system", "content": self.SYSTEM_PROMPT}]
        conversation.extend(messages)
        return self._client.ask(model=model, messages=conversation)

    def local_calculation_fallback(
        self, messages: list[dict[str, str]]
    ) -> AgentResponse | None:
        """Provide a bounded monthly-budget proposal when no provider is available."""
        prompt = next(
            (
                message.get("content", "").casefold()
                for message in reversed(messages)
                if message.get("role") == "user"
            ),
            "",
        )
        if not (
            "gastos" in prompt
            and ("ocio" in prompt or "variables" in prompt)
            and "emergencia" in prompt
        ):
            return None
        income = self._amount_after(
            prompt,
            r"(?:cobro|ingresos?|gano|percibo)[^\d]{0,40}",
        )
        savings = self._amount_after(
            prompt,
            r"(?:ahorrar|ahorro\s+objetivo)[^\d]{0,40}",
        )
        if income is None or savings is None or savings > income:
            return None

        remaining = income - savings
        fixed_expenses = remaining * 16 // 24
        leisure_and_variables = remaining * 5 // 24
        emergency_fund = remaining - fixed_expenses - leisure_and_variables
        return AgentResponse(
            text=(
                "Esta es una propuesta orientativa de presupuesto mensual, no "
                "asesoramiento de inversión específico.\n\n"
                f"- Ingreso mensual: {self._format_amount(income)}\n"
                f"- Ahorro objetivo: {self._format_amount(savings)}\n"
                f"- Gastos fijos: {self._format_amount(fixed_expenses)}\n"
                f"- Ocio y gastos variables: {self._format_amount(leisure_and_variables)}\n"
                f"- Fondo de emergencia: {self._format_amount(emergency_fund)}\n\n"
                "El reparto del dinero restante usa una regla simple: dos tercios "
                "para gastos fijos, aproximadamente una quinta parte para ocio y "
                "variables, y el resto para el fondo de emergencia. La suma total "
                f"es {self._format_amount(income)}."
            )
        )

    @staticmethod
    def _amount_after(prompt: str, prefix: str) -> int | None:
        match = re.search(
            prefix + r"(\d{1,3}(?:[.\s]\d{3})+|\d+)(?:,(\d{1,2}))?\s*(?:€|euros?)?",
            prompt,
        )
        if match is None:
            return None
        euros = int(re.sub(r"[.\s]", "", match.group(1)))
        cents = int((match.group(2) or "").ljust(2, "0") or "0")
        return euros * 100 + cents

    @staticmethod
    def _format_amount(cents: int) -> str:
        euros, remainder = divmod(cents, 100)
        formatted_euros = f"{euros:,}".replace(",", ".")
        if remainder:
            return f"{formatted_euros},{remainder:02d} €"
        return f"{formatted_euros} €"
