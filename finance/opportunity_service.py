"""Read-only tactical opportunity scan over the Atlas watchlist."""

from __future__ import annotations

from dataclasses import dataclass

from finance.opportunity import (
    FinanceOpportunity,
    OpportunitySignal,
    TacticalOpportunityEngine,
)
from finance.paper.policy import PaperMode
from finance.quant.metrics import build_quant_snapshot
from tools.alpha_vantage import AlphaVantageClient, AlphaVantageError


MAX_OPPORTUNITY_SYMBOLS_PER_RUN = 5


@dataclass(frozen=True)
class OpportunityScanResult:
    opportunities: tuple[FinanceOpportunity, ...]
    checked_symbols: tuple[str, ...]
    errors: tuple[str, ...]


class FinanceOpportunityService:
    """Scan TACTICAL watchlist entries without changing financial state."""

    def __init__(
        self,
        watchlist_store,
        market_client: AlphaVantageClient,
        *,
        engine: TacticalOpportunityEngine | None = None,
    ) -> None:
        self._store = watchlist_store
        self._market_client = market_client
        self._engine = engine or TacticalOpportunityEngine()

    def scan_tactical_watchlist(self) -> OpportunityScanResult:
        entries = self._store.entries()

        tactical = [
            entry
            for entry in entries
            if str(entry.get("mode", "")).upper() == PaperMode.TACTICAL.value
            and not entry.get("needs_review")
        ][:MAX_OPPORTUNITY_SYMBOLS_PER_RUN]

        opportunities: list[FinanceOpportunity] = []
        checked: list[str] = []
        errors: list[str] = []

        for entry in tactical:
            symbol = entry["provider_symbol"]
            checked.append(symbol)

            try:
                series = self._market_client.daily_series(symbol)
                snapshot = build_quant_snapshot(series)
                opportunity = self._engine.evaluate(symbol, snapshot)
            except AlphaVantageError as error:
                errors.append(f"{symbol}: market data error {error.code}")
                continue
            except ValueError as error:
                errors.append(f"{symbol}: invalid market series: {error}")
                continue

            if opportunity.signal is OpportunitySignal.CANDIDATE:
                opportunities.append(opportunity)

        return OpportunityScanResult(
            opportunities=tuple(opportunities),
            checked_symbols=tuple(checked),
            errors=tuple(errors),
        )
