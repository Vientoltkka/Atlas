import json
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from finance.intraday.policy_analysis import (
    PolicyThresholds,
    analyze_ledger,
    analyze_snapshots,
    format_results,
    load_snapshots,
)


START = datetime(2026, 9, 19, 12, 0, tzinfo=timezone.utc)
VERSION = "intraday-time-research-v2"


def observation(minute: int, price: str) -> dict:
    value = Decimal(price)
    return {
        "type": "OBSERVATION",
        "strategy_version": VERSION,
        "symbol": "BTC-USD",
        "price": str(value),
        "bid": str(value - Decimal("0.01")),
        "ask": str(value + Decimal("0.01")),
        "timestamp": (START + timedelta(minutes=minute)).isoformat(),
    }


def signal(
    minute: int,
    *,
    short: str,
    long: str,
    acceleration: str = "0.001",
    action: str = "NO_ACTION",
) -> dict:
    timestamp = (START + timedelta(minutes=minute)).isoformat()
    return {
        "type": "SIGNAL",
        "strategy_version": VERSION,
        "symbol": "BTC-USD",
        "timestamp": timestamp,
        "action": action,
        "direction": None,
        "reasons": ["TEST_POLICY"],
        "features": {
            "symbol": "BTC-USD",
            "timestamp": timestamp,
            "observations": 10,
            "last_price": "100",
            "return_short": short,
            "return_long": long,
            "acceleration": acceleration,
            "realized_volatility": "0",
            "relative_spread": "0",
        },
    }


def write_ledger(path, records) -> None:
    path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )


def result_for(results, short: str, long: str):
    return next(
        result
        for result in results
        if result.thresholds == PolicyThresholds(Decimal(short), Decimal(long))
    )


def test_analysis_does_not_modify_jsonl_and_is_deterministic(tmp_path):
    path = tmp_path / "research.jsonl"
    write_ledger(
        path,
        [
            observation(0, "100"),
            signal(0, short="0.001", long="0.002", action="CANDIDATE"),
            observation(1, "101"),
        ],
    )
    before = path.read_bytes()

    first, first_warnings = analyze_ledger(path, strategy_version=VERSION)
    second, second_warnings = analyze_ledger(path, strategy_version=VERSION)

    assert path.read_bytes() == before
    assert format_results(first, first_warnings) == format_results(second, second_warnings)


def test_strict_configuration_requires_all_conditions_with_and():
    snapshots = [
        signal(0, short="0.001", long="0.0019"),
        signal(1, short="0.0009", long="0.002"),
        signal(2, short="0.001", long="0.002", acceleration="0"),
        signal(3, short="0.001", long="0.002"),
    ]
    from finance.intraday.policy_analysis import _observation, _snapshot

    result = analyze_snapshots(
        [_observation(observation(0, "100"))],
        [_snapshot(item) for item in snapshots],
        PolicyThresholds(Decimal("0.001"), Decimal("0.002")),
        strategy_version=VERSION,
    )
    assert result.candidate_observations == 1


def test_less_strict_configuration_uses_historical_no_action_snapshot(tmp_path):
    path = tmp_path / "research.jsonl"
    write_ledger(
        path,
        [
            observation(0, "100"),
            signal(0, short="0.00050", long="0.00100", action="NO_ACTION"),
        ],
    )
    results, _ = analyze_ledger(path, strategy_version=VERSION)

    assert result_for(results, "0.00100", "0.00200").candidate_observations == 0
    assert result_for(results, "0.00025", "0.00050").candidate_observations == 1


def test_cooldown_groups_repeated_candidate_observations():
    from finance.intraday.policy_analysis import _observation, _snapshot

    observations = [_observation(observation(minute, "100")) for minute in (0, 1, 5)]
    snapshots = [
        _snapshot(signal(0, short="0.001", long="0.002", action="NO_ACTION")),
        _snapshot(signal(5, short="0.001", long="0.002", action="NO_ACTION")),
    ]
    result = analyze_snapshots(
        observations,
        snapshots,
        PolicyThresholds(Decimal("0.001"), Decimal("0.002")),
        strategy_version=VERSION,
    )

    assert result.candidate_observations == 2
    assert result.report.total_events == 1
    assert result.report.aggregated_by_cooldown == 1


def test_long_net_return_and_missing_future_are_correct():
    from finance.intraday.policy_analysis import _observation, _snapshot

    observations = [
        _observation(observation(0, "100")),
        _observation(observation(1, "101")),
        _observation(observation(5, "99")),
    ]
    result = analyze_snapshots(
        observations,
        [_snapshot(signal(0, short="0.001", long="0.002"))],
        PolicyThresholds(Decimal("0.001"), Decimal("0.002")),
        strategy_version=VERSION,
    )
    horizons = result.report.horizons

    assert horizons[1].mean_net_return == Decimal("0.0090")
    assert horizons[5].mean_net_return == Decimal("-0.0110")
    assert horizons[15].resolved_events == 0
    assert horizons[15].pending_events == 1
    assert horizons[15].mean_net_return is None


def test_policy_analysis_module_has_no_prohibited_dependencies():
    import finance.intraday.policy_analysis as module

    source = open(module.__file__, encoding="utf-8").read()
    assert "ExecutionIntent" not in source
    assert "BrokerAdapter" not in source
    assert "PaperFinanceService" not in source


def test_policy_analysis_rejects_intermediate_json_corruption(tmp_path):
    path = tmp_path / "corrupt.jsonl"
    path.write_text(
        json.dumps(observation(0, "100")) + "\n{broken\n" + json.dumps(signal(1, short="0.001", long="0.002")) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="invalid JSONL at line 2"):
        load_snapshots(path, strategy_version=VERSION)


def configuration(capture_id: str, **values) -> dict:
    return {
        "type": "CONFIGURATION",
        "strategy_version": VERSION,
        "capture_id": capture_id,
        "configuration": values,
    }


def test_analysis_displays_the_captured_active_v2_configuration(tmp_path):
    path = tmp_path / "research.jsonl"
    write_ledger(
        path,
        [
            configuration(
                "capture-a",
                minimum_short_return="0.123",
                minimum_long_return="0.456",
            ),
            {**observation(0, "100"), "capture_id": "capture-a"},
            {**signal(0, short="0.001", long="0.002"), "capture_id": "capture-a"},
        ],
    )

    results, warnings = analyze_ledger(
        path, strategy_version=VERSION, capture_id="capture-a"
    )
    text = format_results(results, warnings)

    assert results[0].captured_configuration == {
        "minimum_long_return": "0.456",
        "minimum_short_return": "0.123",
    }
    assert results[0].report.capture_id == "capture-a"
    assert "captured active V2 configuration" in text
    assert "configuración activa registrada de la captura" in text
    assert '"minimum_short_return":"0.123"' in text


def test_analysis_warns_when_capture_has_no_configuration(tmp_path):
    path = tmp_path / "research.jsonl"
    write_ledger(
        path,
        [
            {**observation(0, "100"), "capture_id": "capture-a"},
            {**signal(0, short="0.001", long="0.002"), "capture_id": "capture-a"},
        ],
    )

    results, warnings = analyze_ledger(
        path, strategy_version=VERSION, capture_id="capture-a"
    )
    text = format_results(results, warnings)

    assert any("Missing CONFIGURATION" in warning for warning in warnings)
    assert "no configuration values invented" in warnings[-1]
    assert "MISSING; no se inventan valores" in text


def test_analysis_rejects_incompatible_configurations_for_capture(tmp_path):
    path = tmp_path / "research.jsonl"
    write_ledger(
        path,
        [
            configuration("capture-a", threshold="0.001"),
            configuration("capture-a", threshold="0.002"),
        ],
    )

    with pytest.raises(ValueError, match="incompatible CONFIGURATION snapshots"):
        analyze_ledger(path, strategy_version=VERSION, capture_id="capture-a")


def test_sensitivity_matrix_remains_independent_of_captured_configuration(tmp_path):
    path = tmp_path / "research.jsonl"
    write_ledger(
        path,
        [
            configuration(
                "capture-a",
                minimum_short_return="999",
                minimum_long_return="999",
                cooldown_minutes=999,
                estimated_round_trip_cost="999",
            ),
            {**observation(0, "100"), "capture_id": "capture-a"},
            {
                **signal(0, short="0.00050", long="0.00100"),
                "capture_id": "capture-a",
            },
        ],
    )

    results, _ = analyze_ledger(
        path, strategy_version=VERSION, capture_id="capture-a"
    )

    assert results[0].captured_configuration["minimum_short_return"] == "999"
    assert result_for(results, "0.00025", "0.00050").candidate_observations == 1
    assert result_for(results, "0.00100", "0.00200").candidate_observations == 0
