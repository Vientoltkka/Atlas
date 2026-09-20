from datetime import datetime, timedelta, timezone
from decimal import Decimal
import json
import subprocess
import sys

from finance.intraday.evaluation import IndependentSignalEvent
from finance.intraday.models import IntradayObservation
from finance.intraday.outcome_analysis import (
    HORIZONS,
    MINIMUM_EXPLORATORY_EVENT_COUNT,
    aggregate_event_report,
    load_research_event_report,
    main,
    report_to_dict,
)
from finance.intraday.persistence import IntradayResearchLedger


START = datetime(2026, 9, 19, 12, 0, tzinfo=timezone.utc)


def observation(minute: int, price: str) -> IntradayObservation:
    value = Decimal(price)
    return IntradayObservation(
        symbol="BTC-USD",
        price=value,
        bid=value - Decimal("0.01"),
        ask=value + Decimal("0.01"),
        timestamp=START + timedelta(minutes=minute),
    )


def event(
    *,
    direction: str = "LONG",
    signal: str = "MOMENTUM",
    source: str = "alpha",
    cost: str = "0.001",
) -> IndependentSignalEvent:
    return IndependentSignalEvent(
        "BTC-USD",
        direction,
        signal,
        source,
        START,
        Decimal("100"),
        estimated_round_trip_cost=Decimal(cost),
    )


def test_report_aggregates_long_short_events_and_horizons() -> None:
    long_event = event()
    short_event = event(direction="SHORT")
    future = [
        observation(1, "101"),
        observation(5, "102"),
        observation(15, "99"),
        observation(30, "104"),
    ]
    long_event.calculate_outcomes(future)
    short_event.calculate_outcomes(future)

    report = aggregate_event_report([short_event, long_event])

    assert report.total_events == 2
    assert report.horizons[1].outcomes_available == 2
    assert report.horizons[1].mean_gross_return == Decimal("0")
    assert report.horizons[5].mean_gross_return == Decimal("0")
    assert report.horizons[15].mean_gross_return == Decimal("0")
    assert report.horizons[30].mean_gross_return == Decimal("0")
    assert report.horizons[15].net_win_rate == Decimal("0.5")


def test_net_mean_differs_from_gross_mean_when_cost_is_present() -> None:
    item = event(cost="0.003")
    item.calculate_outcomes([observation(1, "101")])

    horizon = aggregate_event_report([item]).horizons[1]

    assert horizon.mean_gross_return == Decimal("0.01")
    assert horizon.mean_net_return == Decimal("0.007")


def test_missing_outcomes_are_excluded_from_metrics() -> None:
    complete = event()
    complete.calculate_outcomes([observation(1, "101")])
    missing = event(signal="BREAKOUT")

    horizon = aggregate_event_report([complete, missing]).horizons[1]

    assert horizon.outcomes_available == 1
    assert horizon.outcomes_pending == 1
    assert horizon.mean_gross_return == Decimal("0.01")
    assert horizon.mean_net_return == Decimal("0.009")
    assert horizon.net_win_rate == Decimal("1")


def test_horizon_without_data_has_none_metrics() -> None:
    item = event()
    item.calculate_outcomes([observation(1, "101")])

    horizon = aggregate_event_report([item]).horizons[5]

    assert horizon.outcomes_available == 0
    assert horizon.outcomes_pending == 1
    assert horizon.mean_gross_return is None
    assert horizon.mean_net_return is None
    assert horizon.net_win_rate is None


def test_signal_source_groups_are_deterministically_sorted() -> None:
    events = [
        event(signal="Z", source="beta"),
        event(signal="A", source="zeta"),
        event(signal="A", source="alpha"),
        event(signal="A", source="alpha"),
    ]

    groups = aggregate_event_report(events).events_by_signal_source

    assert [(item.signal, item.source, item.event_count) for item in groups] == [
        ("A", "alpha", 2),
        ("A", "zeta", 1),
        ("Z", "beta", 1),
    ]


def test_report_module_has_no_execution_or_broker_dependency() -> None:
    import finance.intraday.outcome_analysis as outcome_analysis

    source = open(outcome_analysis.__file__, encoding="utf-8").read()
    assert "ExecutionIntent" not in source
    assert "BrokerAdapter" not in source
    assert "PaperFinanceService" not in source
    assert "finance.execution" not in source
    assert HORIZONS == (1, 5, 15, 30)


def test_ledger_reconstructs_event_updates_as_one_event(tmp_path) -> None:
    path = tmp_path / "events.jsonl"
    ledger = IntradayResearchLedger(path, strategy_version="v2")
    item = event()
    item.observation_count = 3
    item.last_observation_timestamp = START + timedelta(minutes=2)
    ledger.record_event(item)
    ledger.record_event_update(item)
    ledger.record_event_update(item)

    report = load_research_event_report(path, strategy_version="v2")

    assert report.total_events == 1
    assert report.candidate_observations == 0
    assert report.aggregated_by_cooldown == 2


def test_ledger_counts_candidates_and_filters_strategy_version(tmp_path) -> None:
    path = tmp_path / "events.jsonl"
    target = IntradayResearchLedger(path, strategy_version="v2")
    other = IntradayResearchLedger(path, strategy_version="other")
    for _ in range(16):
        target._append({"type": "SIGNAL", "strategy_version": "v2", "action": "CANDIDATE"})
    item = event()
    item.observation_count = 4
    target.record_event(item)
    other.record_event(event(signal="OTHER"))

    report = load_research_event_report(path, strategy_version="v2")

    assert report.total_events == 1
    assert report.candidate_observations == 16
    assert report.event_backed_candidate_observations == 4
    assert report.unlinked_candidate_observations == 12
    assert report.aggregated_by_cooldown == 3
    assert "Candidatos sin EVENT asociado no se usan para retornos independientes." in report.warnings


def test_exploratory_warning_uses_independent_event_count() -> None:
    one_event = aggregate_event_report([event()])
    assert "Muestra exploratoria; no permite inferir rentabilidad" in one_event.warnings

    enough_events = aggregate_event_report(
        [event(source=str(index)) for index in range(MINIMUM_EXPLORATORY_EVENT_COUNT)]
    )
    assert "Muestra exploratoria; no permite inferir rentabilidad" not in enough_events.warnings


def test_report_json_contains_candidate_linkage_fields() -> None:
    report = aggregate_event_report([event()])
    payload = report_to_dict(report)

    assert payload["event_backed_candidate_observations"] == 1
    assert payload["unlinked_candidate_observations"] == 0
    assert payload["warnings"] == ["Muestra exploratoria; no permite inferir rentabilidad"]


def test_ledger_report_has_resolved_pending_median_and_win_rate(tmp_path) -> None:
    path = tmp_path / "events.jsonl"
    ledger = IntradayResearchLedger(path, strategy_version="v2")
    first = event()
    second = event(signal="SECOND")
    first.calculate_outcomes([observation(1, "101"), observation(5, "102")])
    second.calculate_outcomes([observation(1, "99"), observation(5, "104")])
    ledger.record_event(first)
    ledger.record_event(second)

    horizon = load_research_event_report(path, strategy_version="v2").horizons[1]

    assert horizon.resolved_events == 2
    assert horizon.pending_events == 0
    assert horizon.mean_gross_return == Decimal("0")
    assert horizon.median_gross_return == Decimal("0")
    assert horizon.net_win_rate == Decimal("0.5")
    assert load_research_event_report(path, strategy_version="v2").horizons[15].pending_events == 2


def test_empty_ledger_is_safe_and_cli_json_does_not_write(tmp_path) -> None:
    path = tmp_path / "empty.jsonl"
    path.write_text("{\"type\":\"CONFIGURATION\",\"strategy_version\":\"v2\"}\n", encoding="utf-8")
    before = path.read_bytes()
    report = load_research_event_report(path, strategy_version="v2")
    assert report.total_events == 0
    assert "Muestra exploratoria; no permite inferir rentabilidad" in report.warnings

    result = subprocess.run(
        [sys.executable, "-m", "finance.intraday.outcome_analysis", "--ledger", str(path), "--strategy-version", "v2", "--json"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(result.stdout) == report_to_dict(report)
    assert path.read_bytes() == before


def test_main_without_arguments_uses_legacy_research_path(monkeypatch, capsys) -> None:
    import finance.intraday.outcome_analysis as outcome_analysis

    observations = [observation(0, "100")]
    loaded_paths = []
    analyzed = []

    def fake_load(path):
        loaded_paths.append(path)
        return observations

    monkeypatch.setattr(outcome_analysis, "load_observations", fake_load)
    monkeypatch.setattr(
        outcome_analysis,
        "analyze",
        lambda items, configuration: analyzed.append((items, configuration)),
    )

    assert main([]) is None

    assert [str(path) for path in loaded_paths] == [
        ".atlas\\finance_intraday\\live_research.jsonl"
    ]
    assert [
        (item.short_minutes, item.long_minutes)
        for _, item in analyzed
    ] == [(1, 3), (1, 5), (3, 5)]
    assert all(items is observations for items, _ in analyzed)
    assert "ATLAS INTRADAY OUTCOME ANALYZER V2" in capsys.readouterr().out
