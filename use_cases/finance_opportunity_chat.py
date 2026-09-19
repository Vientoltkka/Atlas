"""Deterministic conversational interface for Finance Opportunity V1."""

from __future__ import annotations

import unicodedata

from finance.opportunity_service import FinanceOpportunityService
from finance.strategy_evaluation import StrategyEvaluationStore


def _fold(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text or "")
    return "".join(
        char for char in normalized
        if not unicodedata.combining(char)
    ).lower().strip()


def handles_strategy_performance_prompt(prompt: str) -> bool:
    folded = _fold(prompt)
    return (
        "rendimiento" in folded
        and any(token in folded for token in ("estrategia", "senales"))
    )


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
        self._evaluation_store = StrategyEvaluationStore()
        self._last_candidates: tuple[str, ...] = ()

    def _render_strategy_performance(self) -> str:
        summary = self._evaluation_store.summary()

        total = summary["total_signals"]
        evaluated = summary["evaluated_20"]
        opened = summary["open"]
        win_rate = summary["win_rate_20"]
        average = summary["average_return_20"]

        lines = [
            "[PAPER][STRATEGY] Rendimiento de estrategia t?ctica.",
            f"Se?ales registradas: {total}.",
            f"Evaluadas a 20 sesiones: {evaluated}.",
            f"Abiertas: {opened}.",
        ]

        if evaluated == 0:
            lines.append(
                "Todav?a no existen se?ales con 20 sesiones posteriores; "
                "no hay evidencia suficiente para calcular win rate "
                "ni retorno medio."
            )
        else:
            lines.append(
                f"Win rate a 20 sesiones: {(win_rate * 100):.2f}%."
            )
            lines.append(
                f"Retorno medio a 20 sesiones: {(average * 100):.2f}%."
            )

        lines.append(
            "M?tricas descriptivas PAPER; no implican rentabilidad futura "
            "ni ejecutan ?rdenes."
        )

        return "\n".join(lines)

    @property
    def last_candidates(self) -> tuple[str, ...]:
        """Candidates from the latest explicit tactical scan."""
        return self._last_candidates

    def handles(self, prompt: str) -> bool:
        return handles_finance_opportunity_prompt(prompt)

    def handle(self, prompt: str) -> str:
        if handles_strategy_performance_prompt(prompt):
            return self._render_strategy_performance()
        if not self.handles(prompt):
            raise ValueError("unsupported finance opportunity prompt")

        result = self._service.scan_tactical_watchlist()
        self._last_candidates = tuple(
            item.symbol for item in result.opportunities
        )

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
            "Datos diarios del proveedor; no son cotizaciones en tiempo real."
        )
        lines.append(
            "Lectura PAPER y solo informativa: no se ha creado ni ejecutado ninguna orden."
        )

        if self._last_candidates:
            lines.append(
                "Para preparar una simulaci?n, importa primero el cierre diario "
                "del candidato elegido a PAPER; despu?s solicita la orden PAPER "
                "con una cantidad expl?cita. Cada paso mantiene su confirmaci?n independiente."
            )

        return "\n".join(lines)
