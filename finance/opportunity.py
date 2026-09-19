"""Deterministic tactical opportunity detection for Atlas Finance.

This layer interprets Atlas' existing QuantSnapshot. It does not create
orders, execute trades, mutate portfolios, call an LLM, or invent data.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from finance.quant.metrics import QuantSnapshot, TREND_UP


class OpportunitySignal(Enum):
    CANDIDATE = "CANDIDATE"
    NO_ACTION = "NO_ACTION"


@dataclass(frozen=True)
class FinanceOpportunity:
    symbol: str
    signal: OpportunitySignal
    reasons: tuple[str, ...]
    evidence: tuple[str, ...]


class TacticalOpportunityEngine:
    """Interpret Atlas' canonical quantitative trend classification."""

    def evaluate(
        self,
        symbol: str,
        snapshot: QuantSnapshot,
    ) -> FinanceOpportunity:
        normalized = (symbol or "").strip().upper()
        if not normalized:
            raise ValueError("symbol is required")

        evidence = (
            f"market_day={snapshot.last_day.isoformat()}",
            f"last_close={snapshot.last_close}",
            f"trend={snapshot.trend}",
            f"sma_20={snapshot.sma_20}",
            f"sma_50={snapshot.sma_50}",
            f"change_5={snapshot.change_5}",
            f"change_20={snapshot.change_20}",
            f"volatility_20_annualized={snapshot.volatility_20_annualized}",
            f"max_drawdown_60={snapshot.max_drawdown_60}",
        )

        if snapshot.trend == TREND_UP:
            return FinanceOpportunity(
                symbol=normalized,
                signal=OpportunitySignal.CANDIDATE,
                reasons=(
                    "Atlas quantitative trend is ALZA",
                    "canonical trend rule confirms tactical momentum",
                ),
                evidence=evidence,
            )

        return FinanceOpportunity(
            symbol=normalized,
            signal=OpportunitySignal.NO_ACTION,
            reasons=(
                f"Atlas quantitative trend is {snapshot.trend}",
            ),
            evidence=evidence,
        )
