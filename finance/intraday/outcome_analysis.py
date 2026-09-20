"""Outcome analysis for time-based intraday research.

Research only:
- no broker access
- no orders
- no execution
- no parameter optimization

Measures forward returns after historical momentum states.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import timedelta
from decimal import Decimal
from pathlib import Path
from statistics import mean, median
from typing import Iterable, Sequence

from finance.intraday.evaluation import IndependentSignalEvent
from finance.intraday.models import IntradayObservation
from finance.intraday.time_features import (
    TimeBasedIntradayFeatureEngine,
)
from finance.intraday.time_replay import (
    ReplayConfiguration,
    load_observations,
)


@dataclass(frozen=True)
class MomentumOutcome:
    timestamp: object
    return_short: Decimal
    return_long: Decimal
    acceleration: Decimal
    forward_1m: Decimal | None
    forward_5m: Decimal | None
    forward_15m: Decimal | None
    forward_30m: Decimal | None


HORIZONS = (1, 5, 15, 30)
MINIMUM_EXPLORATORY_EVENT_COUNT = 30
UNLINKED_CANDIDATE_WARNING = (
    "Candidatos sin EVENT asociado no se usan para retornos independientes."
)
EXPLORATORY_SAMPLE_WARNING = "Muestra exploratoria; no permite inferir rentabilidad"


@dataclass(frozen=True)
class EventSignalSourceGroup:
    """Deterministic event count for one signal/source pair."""

    signal: str
    source: str
    event_count: int


@dataclass(frozen=True)
class EventHorizonSummary:
    """Research-only aggregate for one forward horizon."""

    outcomes_available: int
    outcomes_pending: int
    mean_gross_return: Decimal | None
    median_gross_return: Decimal | None
    mean_net_return: Decimal | None
    median_net_return: Decimal | None
    net_win_rate: Decimal | None

    @property
    def resolved_events(self) -> int:
        return self.outcomes_available

    @property
    def pending_events(self) -> int:
        return self.outcomes_pending


@dataclass(frozen=True)
class ResearchEventReport:
    """Reproducible aggregate of already evaluated independent events."""

    total_events: int
    events_by_signal_source: tuple[EventSignalSourceGroup, ...]
    horizons: dict[int, EventHorizonSummary]
    events_by_direction: dict[str, int] | None = None
    events_by_cooldown: dict[int, int] | None = None
    candidate_observations: int = 0
    aggregated_by_cooldown: int = 0
    event_backed_candidate_observations: int = 0
    unlinked_candidate_observations: int = 0
    cooldown_minutes: tuple[int, ...] = ()
    estimated_round_trip_costs: tuple[Decimal, ...] = ()
    by_symbol: dict[str, "ResearchEventGroup"] | None = None
    by_direction: dict[str, "ResearchEventGroup"] | None = None
    by_signal: dict[str, "ResearchEventGroup"] | None = None
    by_source: dict[str, "ResearchEventGroup"] | None = None
    by_signal_source: dict[str, "ResearchEventGroup"] | None = None
    strategy_version: str | None = None
    warnings: tuple[str, ...] = ()

    @property
    def estimated_round_trip_cost(self) -> Decimal | None:
        return (
            self.estimated_round_trip_costs[0]
            if len(self.estimated_round_trip_costs) == 1 else None
        )


@dataclass(frozen=True)
class ResearchEventGroup:
    independent_events: int
    candidate_observations: int
    aggregated_by_cooldown: int
    horizons: dict[int, EventHorizonSummary]
    cooldown_minutes: tuple[int, ...]
    estimated_round_trip_costs: tuple[Decimal, ...]
    event_backed_candidate_observations: int = 0
    unlinked_candidate_observations: int = 0

    @property
    def estimated_round_trip_cost(self) -> Decimal | None:
        return (
            self.estimated_round_trip_costs[0]
            if len(self.estimated_round_trip_costs) == 1 else None
        )


def _mean_decimal(values: list[Decimal]) -> Decimal | None:
    if not values:
        return None
    return sum(values, Decimal("0")) / Decimal(len(values))


def _horizon_summary(events: Sequence[IndependentSignalEvent], horizon: int) -> EventHorizonSummary:
    outcomes = [
        getattr(event, f"outcome_{horizon}m")
        for event in events
    ]
    available = [
        outcome for outcome in outcomes
        if outcome is not None
        and outcome.gross_return is not None
        and outcome.net_return is not None
    ]
    gross_values = [outcome.gross_return for outcome in available]
    net_values = [outcome.net_return for outcome in available]
    return EventHorizonSummary(
        outcomes_available=len(available),
        outcomes_pending=len(events) - len(available),
        mean_gross_return=_mean_decimal(gross_values),
        median_gross_return=median(gross_values) if gross_values else None,
        mean_net_return=_mean_decimal(net_values),
        median_net_return=median(net_values) if net_values else None,
        net_win_rate=(
            Decimal(sum(value > 0 for value in net_values)) / Decimal(len(net_values))
            if net_values else None
        ),
    )


def _group(
    events: Sequence[IndependentSignalEvent],
    candidate_observations: int | None = None,
) -> ResearchEventGroup:
    counts: dict[int, int] = {}
    for event in events:
        counts[event.cooldown_minutes] = counts.get(event.cooldown_minutes, 0) + 1
    event_backed = sum(event.observation_count for event in events)
    candidates = event_backed if candidate_observations is None else candidate_observations
    return ResearchEventGroup(
        independent_events=len(events),
        candidate_observations=candidates,
        aggregated_by_cooldown=sum(max(event.observation_count - 1, 0) for event in events),
        horizons={horizon: _horizon_summary(events, horizon) for horizon in HORIZONS},
        cooldown_minutes=tuple(sorted(counts)),
        estimated_round_trip_costs=tuple(sorted({event.estimated_round_trip_cost for event in events})),
        event_backed_candidate_observations=event_backed,
        unlinked_candidate_observations=max(candidates - event_backed, 0),
    )


def _report_warnings(
    independent_events: int,
    unlinked_candidates: int,
) -> tuple[str, ...]:
    warnings = []
    if unlinked_candidates > 0:
        warnings.append(UNLINKED_CANDIDATE_WARNING)
    if independent_events < MINIMUM_EXPLORATORY_EVENT_COUNT:
        warnings.append(EXPLORATORY_SAMPLE_WARNING)
    return tuple(warnings)


def aggregate_event_report(
    events: Iterable[IndependentSignalEvent],
) -> ResearchEventReport:
    """Aggregate outcomes without evaluating or mutating the supplied events.

    ``estimated_round_trip_cost`` is already reflected in each event's
    ``net_return`` as a research-only hypothesis, not as a real commission.
    """

    event_list = tuple(events)
    grouped: dict[tuple[str, str], int] = {}
    for event in event_list:
        key = (event.signal, event.source)
        grouped[key] = grouped.get(key, 0) + 1

    groups = tuple(
        EventSignalSourceGroup(signal=signal, source=source, event_count=count)
        for (signal, source), count in sorted(grouped.items())
    )

    horizon_summaries = {
        horizon: _horizon_summary(event_list, horizon) for horizon in HORIZONS
    }
    group = _group(event_list)

    def grouped_by(key):
        values: dict[str, list[IndependentSignalEvent]] = {}
        for item in event_list:
            values.setdefault(key(item), []).append(item)
        return {name: _group(items) for name, items in sorted(values.items())}

    return ResearchEventReport(
        total_events=len(event_list),
        events_by_signal_source=groups,
        horizons=horizon_summaries,
        events_by_direction={
            direction: sum(1 for event in event_list if event.direction == direction)
            for direction in sorted({event.direction for event in event_list})
        },
        events_by_cooldown={
            cooldown: sum(
                1 for event in event_list
                if event.cooldown_minutes == cooldown
            )
            for cooldown in sorted({event.cooldown_minutes for event in event_list})
        },
        candidate_observations=sum(
            event.observation_count for event in event_list
        ),
        aggregated_by_cooldown=group.aggregated_by_cooldown,
        event_backed_candidate_observations=group.event_backed_candidate_observations,
        unlinked_candidate_observations=group.unlinked_candidate_observations,
        cooldown_minutes=group.cooldown_minutes,
        estimated_round_trip_costs=group.estimated_round_trip_costs,
        by_symbol=grouped_by(lambda item: item.symbol),
        by_direction=grouped_by(lambda item: item.direction),
        by_signal=grouped_by(lambda item: item.signal),
        by_source=grouped_by(lambda item: item.source),
        by_signal_source=grouped_by(lambda item: f"{item.signal}|{item.source}"),
        warnings=_report_warnings(
            len(event_list), group.unlinked_candidate_observations
        ),
    )


build_event_report = aggregate_event_report


def _decimal_value(value: Decimal | None) -> str | None:
    return None if value is None else str(value)


def _horizon_dict(summary: EventHorizonSummary) -> dict:
    return {
        "resolved_events": summary.resolved_events,
        "pending_events": summary.pending_events,
        "mean_gross_return": _decimal_value(summary.mean_gross_return),
        "median_gross_return": _decimal_value(summary.median_gross_return),
        "mean_net_return": _decimal_value(summary.mean_net_return),
        "median_net_return": _decimal_value(summary.median_net_return),
        "net_win_rate": _decimal_value(summary.net_win_rate),
    }


def _group_dict(group: ResearchEventGroup) -> dict:
    return {
        "independent_events": group.independent_events,
        "candidate_observations": group.candidate_observations,
        "aggregated_by_cooldown": group.aggregated_by_cooldown,
        "event_backed_candidate_observations": group.event_backed_candidate_observations,
        "unlinked_candidate_observations": group.unlinked_candidate_observations,
        "cooldown_minutes": list(group.cooldown_minutes),
        "estimated_round_trip_costs": [
            str(value) for value in group.estimated_round_trip_costs
        ],
        "estimated_round_trip_cost": _decimal_value(group.estimated_round_trip_cost),
        "horizons": {
            str(horizon): _horizon_dict(summary)
            for horizon, summary in sorted(group.horizons.items())
        },
    }


def report_to_dict(report: ResearchEventReport) -> dict:
    """Return a JSON-stable representation; returns remain decimal strings."""
    result = {
        "strategy_version": report.strategy_version,
        "independent_events": report.total_events,
        "candidate_observations": report.candidate_observations,
        "aggregated_by_cooldown": report.aggregated_by_cooldown,
        "event_backed_candidate_observations": report.event_backed_candidate_observations,
        "unlinked_candidate_observations": report.unlinked_candidate_observations,
        "cooldown_minutes": list(report.cooldown_minutes),
        "estimated_round_trip_costs": [
            str(value) for value in report.estimated_round_trip_costs
        ],
        "estimated_round_trip_cost": _decimal_value(report.estimated_round_trip_cost),
        "events_by_direction": dict(sorted((report.events_by_direction or {}).items())),
        "events_by_cooldown": {
            str(key): value
            for key, value in sorted((report.events_by_cooldown or {}).items())
        },
        "horizons": {
            str(horizon): _horizon_dict(summary)
            for horizon, summary in sorted(report.horizons.items())
        },
        "by_symbol": {
            key: _group_dict(value)
            for key, value in sorted((report.by_symbol or {}).items())
        },
        "by_direction": {
            key: _group_dict(value)
            for key, value in sorted((report.by_direction or {}).items())
        },
        "by_signal": {
            key: _group_dict(value)
            for key, value in sorted((report.by_signal or {}).items())
        },
        "by_source": {
            key: _group_dict(value)
            for key, value in sorted((report.by_source or {}).items())
        },
        "by_signal_source": {
            key: _group_dict(value)
            for key, value in sorted((report.by_signal_source or {}).items())
        },
        "warnings": list(report.warnings),
    }
    return result


def load_research_event_report(
    path: str | Path,
    *,
    strategy_version: str,
) -> ResearchEventReport:
    """Read and aggregate one strategy from an existing append-only ledger."""
    requested_version = strategy_version.strip()
    if not requested_version:
        raise ValueError("strategy_version is required")

    ledger_path = Path(path)
    if not ledger_path.exists():
        raise FileNotFoundError(f"ledger does not exist: {ledger_path}")

    from finance.intraday.persistence import IntradayResearchLedger

    snapshots: dict[str, object] = {}
    candidate_observations = 0
    try:
        with ledger_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"invalid JSONL at line {line_number}: {exc.msg}"
                    ) from exc
                if not isinstance(record, dict):
                    raise ValueError(f"invalid JSONL at line {line_number}: object required")
                if record.get("strategy_version") != requested_version:
                    continue
                if record.get("type") == "SIGNAL" and record.get("action") == "CANDIDATE":
                    candidate_observations += 1
                elif record.get("type") in {"EVENT", "EVENT_UPDATE"}:
                    try:
                        event = IntradayResearchLedger.event_from_record(record)
                    except (KeyError, TypeError, ValueError) as exc:
                        raise ValueError(
                            f"invalid event record at line {line_number}: {exc}"
                        ) from exc
                    snapshots[str(record.get("id", event.id))] = event
    except OSError as exc:
        raise ValueError(f"cannot read ledger: {ledger_path}: {exc}") from exc

    report = aggregate_event_report(snapshots.values())
    unlinked_candidates = max(
        candidate_observations - report.event_backed_candidate_observations,
        0,
    )
    warnings = list(_report_warnings(report.total_events, unlinked_candidates))
    if not report.total_events:
        warnings.insert(0, "Sin eventos EVENT para la strategy_version solicitada.")
    return ResearchEventReport(
        **{field: getattr(report, field) for field in report.__dataclass_fields__
           if field not in {
               "strategy_version", "warnings", "candidate_observations",
               "event_backed_candidate_observations", "unlinked_candidate_observations",
           }},
        candidate_observations=candidate_observations,
        event_backed_candidate_observations=report.event_backed_candidate_observations,
        unlinked_candidate_observations=unlinked_candidates,
        strategy_version=requested_version,
        warnings=tuple(warnings),
    )


def _text_percentage(value: Decimal | None) -> str:
    return "n/a" if value is None else f"{value * Decimal('100'):.4f}%"


def format_report_text(report: ResearchEventReport) -> str:
    lines = [
        "ATLAS INTRADAY RESEARCH EVENT REPORT V2",
        "RESEARCH ONLY - READ-ONLY; NO EXECUTION",
        f"strategy_version: {report.strategy_version}",
        f"independent_events: {report.total_events}",
        f"candidate_observations: {report.candidate_observations}",
        "event_backed_candidate_observations: "
        f"{report.event_backed_candidate_observations}",
        "unlinked_candidate_observations: "
        f"{report.unlinked_candidate_observations}",
        f"aggregated_by_cooldown: {report.aggregated_by_cooldown}",
        f"cooldown_minutes: {', '.join(map(str, report.cooldown_minutes)) or 'n/a'}",
        "estimated_round_trip_costs (decimal): "
        + (", ".join(map(str, report.estimated_round_trip_costs)) or "n/a"),
    ]
    for warning in report.warnings:
        lines.append(f"WARNING: {warning}")
    for horizon, summary in report.horizons.items():
        lines.append(
            f"+{horizon}m: resolved={summary.resolved_events} "
            f"pending={summary.pending_events} "
            f"gross_mean={_text_percentage(summary.mean_gross_return)} "
            f"gross_median={_text_percentage(summary.median_gross_return)} "
            f"net_mean={_text_percentage(summary.mean_net_return)} "
            f"net_median={_text_percentage(summary.median_net_return)} "
            f"net_win_rate={_text_percentage(summary.net_win_rate)}"
        )
    return "\n".join(lines)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Read-only V2 event outcome report")
    parser.add_argument("--ledger", required=True)
    parser.add_argument("--strategy-version", required=True)
    parser.add_argument("--json", action="store_true", dest="as_json")
    return parser


def _legacy_main() -> None:
    path = Path(
        ".atlas/finance_intraday/live_research.jsonl"
    )

    observations = load_observations(path)

    print("=== ATLAS INTRADAY OUTCOME ANALYZER V2 ===")
    print("RESEARCH ONLY - NO EXECUTION")
    print(f"Observations: {len(observations)}")

    configurations = (
        ReplayConfiguration(1, 3),
        ReplayConfiguration(1, 5),
        ReplayConfiguration(3, 5),
    )

    for configuration in configurations:
        analyze(
            observations,
            configuration,
        )


def main(argv: Sequence[str] | None = None) -> int | None:
    arguments = sys.argv[1:] if argv is None else list(argv)
    if not arguments:
        return _legacy_main()

    args = _parser().parse_args(arguments)
    try:
        report = load_research_event_report(
            args.ledger,
            strategy_version=args.strategy_version,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}")
        return 2

    if args.as_json:
        print(json.dumps(report_to_dict(report), sort_keys=True, separators=(",", ":")))
    else:
        print(format_report_text(report))
    return 0


def percentile(values, q):
    ordered = sorted(values)
    index = int((len(ordered) - 1) * q)
    return ordered[index]


def future_return(
    observations: list[IntradayObservation],
    *,
    index: int,
    minutes: int,
    maximum_lateness_seconds: int = 30,
) -> Decimal | None:
    current = observations[index]
    target = current.timestamp + timedelta(minutes=minutes)
    maximum = target + timedelta(seconds=maximum_lateness_seconds)

    for item in observations[index + 1:]:
        if item.symbol != current.symbol:
            continue

        if item.timestamp < target:
            continue

        if item.timestamp > maximum:
            return None

        return (item.price / current.price) - Decimal("1")

    return None


def build_outcomes(
    observations: list[IntradayObservation],
    configuration: ReplayConfiguration,
) -> list[MomentumOutcome]:
    engine = TimeBasedIntradayFeatureEngine(
        short_minutes=configuration.short_minutes,
        long_minutes=configuration.long_minutes,
    )

    outcomes = []

    for index in range(1, len(observations)):
        history = observations[: index + 1]

        try:
            features = engine.calculate(history)
        except ValueError:
            continue

        forward = {
            horizon: future_return(
                observations,
                index=index,
                minutes=horizon,
            )
            for horizon in HORIZONS
        }

        outcomes.append(
            MomentumOutcome(
                timestamp=features.timestamp,
                return_short=features.return_short,
                return_long=features.return_long,
                acceleration=features.acceleration,
                forward_1m=forward[1],
                forward_5m=forward[5],
                forward_15m=forward[15],
                forward_30m=forward[30],
            )
        )

    return outcomes


def report_forward(label, rows):
    print()
    print(label)
    print("-" * len(label))
    print(f"States: {len(rows)}")

    for horizon in HORIZONS:
        attribute = f"forward_{horizon}m"

        values = [
            getattr(row, attribute)
            for row in rows
            if getattr(row, attribute) is not None
        ]

        if not values:
            print(f"+{horizon:>2}m: no complete outcomes")
            continue

        floats = [float(value) for value in values]

        positive = sum(
            value > Decimal("0")
            for value in values
        )

        print(
            f"+{horizon:>2}m: "
            f"n={len(values):4d} "
            f"mean={mean(floats): .6%} "
            f"median={median(floats): .6%} "
            f"positive={100 * positive / len(values):6.2f}%"
        )


def analyze(observations, configuration):
    print()
    print("=" * 72)
    print(
        "WINDOW "
        f"{configuration.short_minutes}m/"
        f"{configuration.long_minutes}m"
    )
    print("=" * 72)

    rows = build_outcomes(
        observations,
        configuration,
    )

    print(f"Evaluable states: {len(rows)}")

    if not rows:
        return

    short_values = [row.return_short for row in rows]
    long_values = [row.return_long for row in rows]

    short_p90 = percentile(short_values, .90)
    short_p95 = percentile(short_values, .95)

    long_p90 = percentile(long_values, .90)
    long_p95 = percentile(long_values, .95)

    print(
        f"SHORT p90={float(short_p90):.6%} "
        f"p95={float(short_p95):.6%}"
    )

    print(
        f"LONG  p90={float(long_p90):.6%} "
        f"p95={float(long_p95):.6%}"
    )

    report_forward(
        "ALL EVALUABLE STATES",
        rows,
    )

    positive_momentum = [
        row
        for row in rows
        if (
            row.return_short > 0
            and row.return_long > 0
            and row.acceleration > 0
        )
    ]

    report_forward(
        "POSITIVE MOMENTUM",
        positive_momentum,
    )

    p90_momentum = [
        row
        for row in rows
        if (
            row.return_short >= short_p90
            and row.return_long >= long_p90
            and row.acceleration > 0
        )
    ]

    report_forward(
        "P90 SHORT + P90 LONG + POSITIVE ACCELERATION",
        p90_momentum,
    )

    p95_momentum = [
        row
        for row in rows
        if (
            row.return_short >= short_p95
            and row.return_long >= long_p95
            and row.acceleration > 0
        )
    ]

    report_forward(
        "P95 SHORT + P95 LONG + POSITIVE ACCELERATION",
        p95_momentum,
    )


if __name__ == "__main__":
    raise SystemExit(main())
