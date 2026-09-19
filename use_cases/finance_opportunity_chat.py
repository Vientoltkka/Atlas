"""Deterministic conversational interface for Finance Opportunity V1."""

from __future__ import annotations

import unicodedata

from finance.opportunity_service import FinanceOpportunityService


def _fold(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text or "")
    return "".join(
        char for char in normalized
        if not unicodedata.combining(char)
    ).lower().strip()


def handles_finance_opportunity_prompt(prompt: str) -> bool:
    folded = _fold(prompt)

    has_opportunity = any(
        token in folded
        for token in (
            "oportunidad",
            "oportunidades",
            "candidato",
            "candidatos",
        )
    )
    has_scope = any(
        token in folded
        for token in (
            "watchlist",
            "lista de seguimiento",
            "tactica",
            "tacticas",
            "tactico",
            "tacticos",
        )
    )

    return has_opportunity and has_scope


class FinanceOpportunityChat:
    def __init__(self, service: FinanceOpportunityService) -> None:
        self._service = service

    def handles(self, prompt: str) -> bool:
        return handles_finance_opportunity_prompt(prompt)

    def handle(self, prompt: str) -> str:
        if not self.handles(prompt):
            raise ValueError("unsupported finance opportunity prompt")

        result = self._service.scan_tactical_watchlist()

        lines = ["[PAPER][OPPORTUNITY] Escaneo táctico de watchlist."]

        if result.checked_symbols:
            lines.append(
                "Revisados: " + ", ".join(result.checked_symbols) + "."
            )
        else:
            lines.append(
                "No hay activos TACTICAL verificados para analizar."
            )

        if result.opportunities:
            lines.append("")
            lines.append("Candidatos detectados:")

            for item in result.opportunities:
                lines.append(f"- {item.symbol}: CANDIDATE")
                for reason in item.reasons:
                    lines.append(f"  Motivo: {reason}")
                for evidence in item.evidence:
                    lines.append(f"  {evidence}")
        else:
            lines.append("")
            lines.append(
                "No se detectaron candidatos con la regla cuantitativa actual."
            )

        if result.errors:
            lines.append("")
            lines.append("Incidencias:")
            for error in result.errors:
                lines.append(f"- {error}")

        lines.append("")
        lines.append(
            "Lectura PAPER y solo informativa: no se ha creado ni ejecutado ninguna orden."
        )

        return "\n".join(lines)
