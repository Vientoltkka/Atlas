from datetime import date
from decimal import Decimal

from finance.opportunity import OpportunitySignal, TacticalOpportunityEngine
from finance.opportunity_service import FinanceOpportunityService
from finance.quant.metrics import (
    QuantSnapshot,
    TREND_DOWN,
    TREND_SIDEWAYS,
    TREND_UP,
)


def snapshot(trend: str) -> QuantSnapshot:
    return QuantSnapshot(
        symbol="TEST",
        total_bars=60,
        last_day=date(2026, 9, 18),
        last_close=Decimal("110"),
        change_5=Decimal("0.03"),
        change_20=Decimal("0.08"),
        sma_20=Decimal("105"),
        sma_50=Decimal("100"),
        volatility_20_annualized=Decimal("0.20"),
        max_drawdown_60=Decimal("-0.10"),
        close_vs_sma_20="POR ENCIMA",
        close_vs_sma_50="POR ENCIMA",
        trend=trend,
        unavailable=(),
    )


def test_real_atlas_alza_is_candidate():
    result = TacticalOpportunityEngine().evaluate("test", snapshot(TREND_UP))

    assert result.signal is OpportunitySignal.CANDIDATE
    assert result.symbol == "TEST"
    assert "trend=ALZA" in result.evidence


def test_non_alza_is_not_candidate():
    engine = TacticalOpportunityEngine()

    assert (
        engine.evaluate("TEST", snapshot(TREND_DOWN)).signal
        is OpportunitySignal.NO_ACTION
    )
    assert (
        engine.evaluate("TEST", snapshot(TREND_SIDEWAYS)).signal
        is OpportunitySignal.NO_ACTION
    )


class Store:
    def entries(self):
        return [
            {
                "provider_symbol": "AAA",
                "mode": "TACTICAL",
                "needs_review": False,
            },
            {
                "provider_symbol": "CORE1",
                "mode": "CORE",
                "needs_review": False,
            },
            {
                "provider_symbol": "LEGACY",
                "mode": "TACTICAL",
                "needs_review": True,
            },
            {
                "provider_symbol": "BBB",
                "mode": "TACTICAL",
                "needs_review": False,
            },
        ]


class Series:
    def __init__(self, symbol):
        self.symbol = symbol


class Market:
    def __init__(self):
        self.calls = []

    def daily_series(self, symbol):
        self.calls.append(symbol)
        return Series(symbol)


class Engine:
    def __init__(self):
        self.calls = []

    def evaluate(self, symbol, quant_snapshot):
        self.calls.append((symbol, quant_snapshot))
        from finance.opportunity import FinanceOpportunity

        signal = (
            OpportunitySignal.CANDIDATE
            if symbol == "AAA"
            else OpportunitySignal.NO_ACTION
        )
        return FinanceOpportunity(
            symbol=symbol,
            signal=signal,
            reasons=("test",),
            evidence=("test",),
        )


def test_service_scans_only_tactical_verified_entries(monkeypatch):
    market = Market()
    engine = Engine()

    monkeypatch.setattr(
        "finance.opportunity_service.build_quant_snapshot",
        lambda series: series,
    )

    result = FinanceOpportunityService(
        Store(),
        market,
        engine=engine,
    ).scan_tactical_watchlist()

    assert result.checked_symbols == ("AAA", "BBB")
    assert market.calls == ["AAA", "BBB"]
    assert [item.symbol for item in result.opportunities] == ["AAA"]
    assert result.errors == ()


def test_scan_does_not_mutate_watchlist_or_create_orders(monkeypatch):
    store = Store()
    original = store.entries()

    monkeypatch.setattr(
        "finance.opportunity_service.build_quant_snapshot",
        lambda series: series,
    )

    FinanceOpportunityService(
        store,
        Market(),
        engine=Engine(),
    ).scan_tactical_watchlist()

    assert store.entries() == original
