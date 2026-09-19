"""Finance Signal Discovery V1.

Usa un universo independiente y delega toda decisi?n CANDIDATE/NO_ACTION en
FinanceOpportunityService. No duplica reglas cuantitativas.
"""

from __future__ import annotations

from dataclasses import dataclass

from finance.discovery_universe import DiscoveryUniverseStore
from finance.opportunity import FinanceOpportunity
from finance.opportunity_service import (
    FinanceOpportunityService,
    OpportunityScanResult,
)


@dataclass(frozen=True, slots=True)
class SignalDiscoveryResult:
    opportunities: tuple[FinanceOpportunity, ...]
    checked_symbols: tuple[str, ...]
    errors: tuple[str, ...]
    universe_size: int


class FinanceSignalDiscovery:
    def __init__(
        self,
        universe: DiscoveryUniverseStore,
        opportunity_service: FinanceOpportunityService,
    ) -> None:
        self._universe = universe
        self._opportunity_service = opportunity_service

    @property
    def universe(self) -> DiscoveryUniverseStore:
        return self._universe

    def scan(self) -> SignalDiscoveryResult:
        symbols = self._universe.symbols()

        result: OpportunityScanResult = (
            self._opportunity_service.scan_symbols(symbols)
        )

        return SignalDiscoveryResult(
            opportunities=result.opportunities,
            checked_symbols=result.checked_symbols,
            errors=result.errors,
            universe_size=len(symbols),
        )
