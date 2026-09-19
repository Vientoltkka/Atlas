from datetime import date, timedelta
from decimal import Decimal

from finance.strategy_evaluation import StrategyEvaluationStore
from tools.alpha_vantage import DailyBar, DailySeries


def _series(symbol: str, start: date, closes: list[str]) -> DailySeries:
    bars = []
    day = start
    for close in closes:
        value = Decimal(close)
        bars.append(
            DailyBar(
                day=day,
                open=value,
                high=value,
                low=value,
                close=value,
                volume=1000,
            )
        )
        day += timedelta(days=1)

    return DailySeries(
        symbol=symbol,
        provider_last_refreshed=bars[-1].day.isoformat(),
        provider_timezone="UTC",
        bars=tuple(bars),
    )


def test_record_candidate_is_idempotent(tmp_path):
    store = StrategyEvaluationStore(tmp_path)

    first = store.record_candidate(
        symbol="vusa.ams",
        signal_day=date(2026, 9, 18),
        signal_close=Decimal("100"),
        rule="trend=ALZA",
    )
    second = store.record_candidate(
        symbol="VUSA.AMS",
        signal_day=date(2026, 9, 18),
        signal_close=Decimal("100"),
        rule="trend=ALZA",
    )

    assert first == second
    assert len(store.entries()) == 1
    assert store.entries()[0].status == "OPEN"


def test_evaluate_five_sessions(tmp_path):
    store = StrategyEvaluationStore(tmp_path)
    store.record_candidate(
        symbol="TEST",
        signal_day=date(2026, 1, 1),
        signal_close=Decimal("100"),
        rule="trend=ALZA",
    )

    series = _series(
        "TEST",
        date(2026, 1, 2),
        ["101", "102", "103", "104", "105"],
    )

    result = store.evaluate_series(series)[0]

    assert result.return_5 == Decimal("0.05")
    assert result.return_20 is None
    assert result.status == "OPEN"


def test_evaluate_twenty_sessions_and_summary(tmp_path):
    store = StrategyEvaluationStore(tmp_path)
    store.record_candidate(
        symbol="TEST",
        signal_day=date(2026, 1, 1),
        signal_close=Decimal("100"),
        rule="trend=ALZA",
    )

    closes = [str(100 + index) for index in range(1, 21)]
    result = store.evaluate_series(
        _series("TEST", date(2026, 1, 2), closes)
    )[0]

    assert result.return_5 == Decimal("0.05")
    assert result.return_20 == Decimal("0.20")
    assert result.max_favorable_20 == Decimal("0.20")
    assert result.max_adverse_20 == Decimal("0.01")
    assert result.status == "EVALUATED"

    summary = store.summary()
    assert summary["total_signals"] == 1
    assert summary["evaluated_20"] == 1
    assert summary["open"] == 0
    assert summary["win_rate_20"] == Decimal("1")
    assert summary["average_return_20"] == Decimal("0.20")


def test_evaluation_does_not_use_pre_signal_bars(tmp_path):
    store = StrategyEvaluationStore(tmp_path)
    store.record_candidate(
        symbol="TEST",
        signal_day=date(2026, 1, 10),
        signal_close=Decimal("100"),
        rule="trend=ALZA",
    )

    series = _series(
        "TEST",
        date(2026, 1, 1),
        [str(90 + index) for index in range(20)],
    )

    result = store.evaluate_series(series)[0]

    # Only bars strictly after Jan 10 count toward the horizon.
    assert result.return_5 == Decimal("0.04")
    assert result.return_20 is None


def test_store_survives_new_instance(tmp_path):
    first = StrategyEvaluationStore(tmp_path)
    first.record_candidate(
        symbol="VUSA.AMS",
        signal_day=date(2026, 9, 18),
        signal_close=Decimal("125.9990"),
        rule="trend=ALZA",
    )

    second = StrategyEvaluationStore(tmp_path)
    entries = second.entries()

    assert len(entries) == 1
    assert entries[0].symbol == "VUSA.AMS"
    assert entries[0].signal_day == date(2026, 9, 18)
    assert entries[0].signal_close == Decimal("125.9990")
    assert entries[0].rule == "trend=ALZA"
