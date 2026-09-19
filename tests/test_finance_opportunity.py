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



class NoOpEvaluationStore:
    def evaluate_series(self, series):
        return ()

    def record_candidate(self, **kwargs):
        return None


def test_service_scans_only_tactical_verified_entries(monkeypatch):
    market = Market()
    engine = Engine()

    monkeypatch.setattr(
        "finance.opportunity_service.build_quant_snapshot",
        lambda series: snapshot(TREND_UP),
    )

    result = FinanceOpportunityService(
        Store(),
        market,
        engine=engine,
        evaluation_store=NoOpEvaluationStore(),
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
        lambda series: snapshot(TREND_UP),
    )

    FinanceOpportunityService(
        store,
        Market(),
        engine=Engine(),
        evaluation_store=NoOpEvaluationStore(),
    ).scan_tactical_watchlist()

    assert store.entries() == original


def test_candidate_is_recorded_for_strategy_evaluation(monkeypatch, tmp_path):
    from datetime import date
    from decimal import Decimal

    from finance.strategy_evaluation import StrategyEvaluationStore
    from tools.alpha_vantage import DailyBar, DailySeries

    class SingleTacticalStore:
        def entries(self):
            return (
                {
                    "symbol": "VUSA",
                    "provider_symbol": "VUSA.AMS",
                    "mode": "TACTICAL",
                    "needs_review": False,
                },
            )

    def real_series(symbol):
        bars = tuple(
            DailyBar(
                day=date(2026, 9, day),
                open=Decimal("125"),
                high=Decimal("127"),
                low=Decimal("124"),
                close=Decimal("125.9990"),
                volume=1000,
            )
            for day in range(1, 19)
        )
        return DailySeries(
            symbol=symbol,
            provider_last_refreshed="2026-09-18",
            provider_timezone="UTC",
            bars=bars,
        )

    class EvaluationMarket:
        def daily_series(self, symbol):
            return real_series(symbol)

    class CandidateEngine:
        def evaluate(self, symbol, quant_snapshot):
            from finance.opportunity import (
                FinanceOpportunity,
                OpportunitySignal,
            )
            return FinanceOpportunity(
                symbol=symbol,
                signal=OpportunitySignal.CANDIDATE,
                reasons=("trend=ALZA",),
                evidence=(),
            )

    quant = snapshot(TREND_UP)

    monkeypatch.setattr(
        "finance.opportunity_service.build_quant_snapshot",
        lambda series: quant,
    )

    evaluation_store = StrategyEvaluationStore(tmp_path)

    result = FinanceOpportunityService(
        SingleTacticalStore(),
        EvaluationMarket(),
        engine=CandidateEngine(),
        evaluation_store=evaluation_store,
    ).scan_tactical_watchlist()

    assert len(result.opportunities) == 1

    entries = evaluation_store.entries()
    assert len(entries) == 1
    assert entries[0].symbol == "VUSA.AMS"
    assert entries[0].signal_day == quant.last_day
    assert entries[0].signal_close == quant.last_close
    assert entries[0].rule == "trend=ALZA"


def test_repeated_scan_does_not_duplicate_strategy_signal(monkeypatch, tmp_path):
    from datetime import date
    from decimal import Decimal

    from finance.strategy_evaluation import StrategyEvaluationStore
    from tools.alpha_vantage import DailyBar, DailySeries

    class SingleTacticalStore:
        def entries(self):
            return (
                {
                    "symbol": "VUSA",
                    "provider_symbol": "VUSA.AMS",
                    "mode": "TACTICAL",
                    "needs_review": False,
                },
            )

    def real_series(symbol):
        bars = tuple(
            DailyBar(
                day=date(2026, 9, day),
                open=Decimal("125"),
                high=Decimal("127"),
                low=Decimal("124"),
                close=Decimal("125.9990"),
                volume=1000,
            )
            for day in range(1, 19)
        )
        return DailySeries(
            symbol=symbol,
            provider_last_refreshed="2026-09-18",
            provider_timezone="UTC",
            bars=bars,
        )

    class EvaluationMarket:
        def daily_series(self, symbol):
            return real_series(symbol)

    class CandidateEngine:
        def evaluate(self, symbol, quant_snapshot):
            from finance.opportunity import (
                FinanceOpportunity,
                OpportunitySignal,
            )
            return FinanceOpportunity(
                symbol=symbol,
                signal=OpportunitySignal.CANDIDATE,
                reasons=("trend=ALZA",),
                evidence=(),
            )

    quant = snapshot(TREND_UP)

    monkeypatch.setattr(
        "finance.opportunity_service.build_quant_snapshot",
        lambda series: quant,
    )

    evaluation_store = StrategyEvaluationStore(tmp_path)

    service = FinanceOpportunityService(
        SingleTacticalStore(),
        EvaluationMarket(),
        engine=CandidateEngine(),
        evaluation_store=evaluation_store,
    )

    service.scan_tactical_watchlist()
    service.scan_tactical_watchlist()

    entries = evaluation_store.entries()

    assert len(entries) == 1
    assert entries[0].symbol == "VUSA.AMS"
    assert entries[0].signal_day == quant.last_day
