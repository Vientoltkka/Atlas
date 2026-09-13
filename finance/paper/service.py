"""Fachada explicita de Atlas Finance V2.1 (motor paper).

Servicio unico, persistente y auditable para: consultar cartera, registrar
un MarketEvent declarado y ejecutar una orden paper confirmada. El
FinanceAgent sigue siendo solo-lectura: solo puede sugerir ordenes; la
ejecucion exige confirmacion explicita a traves de este servicio. Sin red,
sin broker y sin scheduler.
"""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path
from typing import Mapping

from finance.paper.engine import Decision, PaperEngine
from finance.paper.models import MarketEvent, PaperOrder, PaperPortfolio, utc_now
from finance.paper.store import PaperStore

DEFAULT_STARTING_CASH = Decimal("10000")


class PaperFinanceService:
    def __init__(
        self,
        store: PaperStore | None = None,
        *,
        starting_cash: Decimal = DEFAULT_STARTING_CASH,
        default_commission: Decimal = Decimal("0"),
        default_slippage_bps: Decimal = Decimal("0"),
    ) -> None:
        self._store = store or PaperStore()
        if self._store.exists():
            data = self._store.load()
            self._engine = PaperEngine.from_snapshot(data)
            return
        self._engine = PaperEngine(
            portfolio=PaperPortfolio(cash=starting_cash),
            default_commission=default_commission,
            default_slippage_bps=default_slippage_bps,
        )
        self._persist()

    @property
    def engine(self) -> PaperEngine:
        return self._engine

    def portfolio_summary(self) -> dict[str, object]:
        portfolio = self._engine.portfolio
        prices = self._engine.last_prices()
        positions = {
            symbol: {
                "qty": str(position.qty),
                "avg_price": str(position.avg_price),
                "last_price": str(prices.get(symbol, position.avg_price)),
            }
            for symbol, position in portfolio.positions.items()
        }
        ledger = self._engine.ledger
        updated_at = (
            str(ledger[-1]["timestamp"]) if ledger else utc_now().isoformat()
        )
        return {
            "cash": str(portfolio.cash.quantize(Decimal("0.01"))),
            "positions": positions,
            "nav": str(self._engine.nav()),
            "updated_at": updated_at,
        }

    def record_market_event(self, event: MarketEvent) -> bool:
        ingested = self._engine.process_event(event)
        if ingested:
            self._persist()
        return ingested

    def execute_confirmed_order(
        self, order: PaperOrder, event: MarketEvent | None = None
    ) -> Decision:
        decision = self._engine.execute_order(order, event)
        self._persist()
        return decision

    def pending_orders(self) -> tuple[PaperOrder, ...]:
        return self._engine.pending_orders()

    def ledger(self) -> tuple[dict[str, object], ...]:
        return self._engine.ledger

    def fills(self) -> tuple:
        return self._engine.fills

    def _persist(self) -> None:
        self._store.save(self._engine.snapshot())
