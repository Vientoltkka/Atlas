"""Read-only sensitivity analysis for persisted time-based V2 research.

This module evaluates historical feature snapshots only.  It never rebuilds or
updates the append-only ledger and does not participate in the active policy.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Sequence

from finance.intraday.evaluation import (
    EventDetector,
    IndependentSignalEvent,
)
from finance.intraday.models import IntradayFeatures, IntradayObservation
from finance.intraday.outcome_analysis import (
    HORIZONS,
    MINIMUM_EXPLORATORY_EVENT_COUNT,
    ResearchEventReport,
    aggregate_event_report,
    report_to_dict,
)
from finance.intraday.persistence import read_jsonl_records


SHORT_THRESHOLDS = (Decimal("0.00025"), Decimal("0.00050"), Decimal("0.00100"))
LONG_THRESHOLDS = (Decimal("0.00050"), Decimal("0.00100"), Decimal("0.00200"))
MINIMUM_ACCELERATION = Decimal("0")
MAXIMUM_VOLATILITY = Decimal("0.0100")
MAXIMUM_RELATIVE_SPREAD = Decimal("0.0030")
COOLDOWN_MINUTES = 15
RESEARCH_COST = Decimal("0.0010")


@dataclass(frozen=True)
class PolicyThresholds:
    minimum_short_return: Decimal
    minimum_long_return: Decimal

    @property
    def label(self) -> str:
        return (
            f"short={self.minimum_short_return} "
            f"long={self.minimum_long_return}"
        )


@dataclass(frozen=True)
class FeatureSnapshot:
    features: IntradayFeatures
    action: str


@dataclass(frozen=True)
class SensitivityResult:
    thresholds: PolicyThresholds
    candidate_observations: int
    report: ResearchEventReport
    captured_configuration: dict | None = None


def _decimal(value: object) -> Decimal:
    return Decimal(str(value))


def _observation(record: dict) -> IntradayObservation:
    return IntradayObservation(
        symbol=record["symbol"],
        price=_decimal(record["price"]),
        bid=_decimal(record["bid"]),
        ask=_decimal(record["ask"]),
        timestamp=datetime.fromisoformat(record["timestamp"]),
        volume=(
            None if record.get("volume") is None else _decimal(record["volume"])
        ),
    )


def _snapshot(record: dict) -> FeatureSnapshot:
    raw = record["features"]
    features = IntradayFeatures(
        symbol=raw["symbol"],
        timestamp=datetime.fromisoformat(raw["timestamp"]),
        observations=int(raw["observations"]),
        last_price=_decimal(raw["last_price"]),
        return_short=_decimal(raw["return_short"]),
        return_long=_decimal(raw["return_long"]),
        acceleration=_decimal(raw["acceleration"]),
        realized_volatility=_decimal(raw["realized_volatility"]),
        relative_spread=_decimal(raw["relative_spread"]),
    )
    return FeatureSnapshot(features=features, action=str(record.get("action", "")))


def load_snapshots(
    path: str | Path,
    *,
    strategy_version: str,
    capture_id: str | None = None,
) -> tuple[list[IntradayObservation], list[FeatureSnapshot], tuple[str, ...]]:
    """Read only observations and signal feature snapshots from one JSONL."""
    observations: list[IntradayObservation] = []
    snapshots: list[FeatureSnapshot] = []
    warnings: list[str] = []
    requested = strategy_version.strip()
    if not requested:
        raise ValueError("strategy_version is required")

    ledger_path = Path(path)
    if not ledger_path.exists():
        raise FileNotFoundError(f"ledger does not exist: {ledger_path}")

    records = read_jsonl_records(ledger_path)

    available_capture_ids = sorted({
        record["capture_id"] for record in records
        if record.get("strategy_version") == requested
        and isinstance(record.get("capture_id"), str)
        and record["capture_id"].strip()
    })
    if len(available_capture_ids) > 1 and capture_id is None:
        raise ValueError(
            "capture_id must be selected; available capture IDs: "
            + ", ".join(available_capture_ids)
        )
    if capture_id is None and len(available_capture_ids) == 1:
        capture_id = available_capture_ids[0]
    if capture_id is not None and capture_id not in available_capture_ids:
        raise ValueError(f"unknown capture_id: {capture_id}")
    for record in records:
        if record.get("strategy_version") != requested:
            continue
        if capture_id is not None:
            if record.get("capture_id") != capture_id:
                continue
        elif record.get("capture_id") is not None:
            continue
        try:
            if record.get("type") == "OBSERVATION":
                observations.append(_observation(record))
            elif record.get("type") == "SIGNAL":
                snapshots.append(_snapshot(record))
        except (KeyError, TypeError, ValueError) as exc:
            warnings.append(f"registro omitido: {exc}")

    observations.sort(key=lambda item: (item.timestamp, item.symbol))
    snapshots.sort(key=lambda item: (item.features.timestamp, item.features.symbol, item.action))
    if not observations:
        warnings.append("No hay OBSERVATION para la strategy_version solicitada.")
    if not snapshots:
        warnings.append("No hay SIGNAL con snapshots de features para la strategy_version solicitada.")
    return observations, snapshots, tuple(warnings)


def _load_captured_configuration(
    records: Sequence[dict],
    *,
    strategy_version: str,
    capture_id: str | None,
) -> dict | None:
    configurations = [
        record.get("configuration")
        for record in records
        if record.get("type") == "CONFIGURATION"
        and record.get("strategy_version") == strategy_version
        and record.get("capture_id") == capture_id
    ]
    if not configurations:
        return None
    if any(not isinstance(configuration, dict) for configuration in configurations):
        raise ValueError(
            "invalid CONFIGURATION snapshot for the selected strategy_version + capture_id"
        )
    encoded = {
        json.dumps(configuration, sort_keys=True, separators=(",", ":"))
        for configuration in configurations
    }
    if len(encoded) > 1:
        raise ValueError(
            "incompatible CONFIGURATION snapshots for the selected "
            "strategy_version + capture_id"
        )
    return deepcopy(configurations[0])


def _matches(features: IntradayFeatures, thresholds: PolicyThresholds) -> bool:
    # Keep the active V2 semantics: acceleration must be strictly positive
    # when its configured minimum is zero.
    return (
        features.return_short >= thresholds.minimum_short_return
        and features.return_long >= thresholds.minimum_long_return
        and features.acceleration > MINIMUM_ACCELERATION
        and features.realized_volatility <= MAXIMUM_VOLATILITY
        and features.relative_spread <= MAXIMUM_RELATIVE_SPREAD
    )


def analyze_snapshots(
    observations: list[IntradayObservation],
    snapshots: list[FeatureSnapshot],
    thresholds: PolicyThresholds,
    *,
    strategy_version: str,
    capture_id: str | None = None,
) -> SensitivityResult:
    """Evaluate one fixed threshold pair entirely in memory."""
    detector = EventDetector(
        cooldown_minutes=COOLDOWN_MINUTES,
        estimated_round_trip_cost=RESEARCH_COST,
    )
    events: dict[str, IndependentSignalEvent] = {}
    candidate_count = 0

    for snapshot in snapshots:
        features = snapshot.features
        if not _matches(features, thresholds):
            continue
        candidate_count += 1
        event = detector.observe(
            symbol=features.symbol,
            direction="LONG",
            signal="MOMENTUM",
            source=strategy_version,
            timestamp=features.timestamp,
            entry_price=features.last_price,
        )
        events.setdefault(event.id, event)

    for event in events.values():
        event.calculate_outcomes(observations)

    report = aggregate_event_report(events.values())
    return SensitivityResult(
        thresholds=thresholds,
        candidate_observations=candidate_count,
        report=ResearchEventReport(
            **{
                field: getattr(report, field)
                for field in report.__dataclass_fields__
                if field not in {"candidate_observations", "strategy_version", "capture_id"}
            },
            candidate_observations=candidate_count,
            strategy_version=strategy_version,
            capture_id=capture_id,
        ),
    )


def analyze_ledger(
    path: str | Path,
    *,
    strategy_version: str,
    capture_id: str | None = None,
) -> tuple[tuple[SensitivityResult, ...], tuple[str, ...]]:
    observations, snapshots, warnings = load_snapshots(
        path, strategy_version=strategy_version, capture_id=capture_id
    )
    records = read_jsonl_records(path)
    selected_capture_id = capture_id
    if selected_capture_id is None:
        available_capture_ids = sorted({
            record["capture_id"] for record in records
            if record.get("strategy_version") == strategy_version
            and isinstance(record.get("capture_id"), str)
            and record["capture_id"].strip()
        })
        if len(available_capture_ids) == 1:
            selected_capture_id = available_capture_ids[0]
    captured_configuration = _load_captured_configuration(
        records,
        strategy_version=strategy_version,
        capture_id=selected_capture_id,
    )
    if captured_configuration is None:
        warnings = warnings + (
            "Missing CONFIGURATION for the selected strategy_version + capture_id; "
            "no configuration values invented / falta CONFIGURATION y no se inventan valores.",
        )
    results = tuple(
        analyze_snapshots(
            observations,
            snapshots,
            PolicyThresholds(short, long),
            strategy_version=strategy_version,
            capture_id=selected_capture_id,
        )
        for short in SHORT_THRESHOLDS
        for long in LONG_THRESHOLDS
    )
    results = tuple(
        SensitivityResult(
            thresholds=result.thresholds,
            candidate_observations=result.candidate_observations,
            report=result.report,
            captured_configuration=deepcopy(captured_configuration),
        )
        for result in results
    )
    return results, warnings


def _text(value: Decimal | None) -> str:
    return "n/a" if value is None else str(value)


def format_results(
    results: Sequence[SensitivityResult],
    warnings: Sequence[str] = (),
) -> str:
    lines = [
        "V2 OFFLINE POLICY SENSITIVITY SCAN",
        "IN-SAMPLE EXPLORATORY — NOT A TRADING RECOMMENDATION",
        "READ-ONLY: SIGNAL/OBSERVATION snapshots only; active V2 policy unchanged",
        f"capture_id: {results[0].report.capture_id if results else 'legacy'}",
        "SENSITIVITY MATRIX: IN-SAMPLE EXPLORATORY ONLY; NOT THE ACTIVE POLICY",
        "matrix: short in {0.00025, 0.00050, 0.00100}; long in {0.00050, 0.00100, 0.00200}; "
        "acceleration > 0; volatility <= 0.0100; spread <= 0.0030",
        "cooldown_minutes: 15; research_cost: 0.0010",
    ]
    configuration = results[0].captured_configuration if results else None
    if configuration is None:
        lines.append(
            "captured active V2 configuration / configuración activa registrada de la captura: "
            "MISSING; no se inventan valores"
        )
    else:
        lines.append(
            "captured active V2 configuration / configuración activa registrada de la captura: "
            + json.dumps(configuration, sort_keys=True, separators=(",", ":"))
        )
    lines.extend(f"WARNING: {warning}" for warning in warnings)
    for result in results:
        report = result.report
        evidence = (
            "evidencia insuficiente"
            if report.total_events < MINIMUM_EXPLORATORY_EVENT_COUNT
            else "evidencia exploratoria"
        )
        lines.append(
            f"CONFIG {result.thresholds.label} | "
            f"candidate_observations={result.candidate_observations} | "
            f"independent_events={report.total_events} | "
            f"aggregated_by_cooldown={report.aggregated_by_cooldown} | {evidence}"
        )
        for horizon in HORIZONS:
            summary = report.horizons[horizon]
            lines.append(
                f"  +{horizon}m: resolved={summary.resolved_events} "
                f"pending={summary.pending_events} "
                f"mean_net_return={_text(summary.mean_net_return)}"
            )
    if results:
        lines.append("No se selecciona automáticamente ninguna configuración.")
    return "\n".join(lines)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Read-only V2 policy sensitivity scan")
    parser.add_argument("--ledger", required=True)
    parser.add_argument("--strategy-version", required=True)
    parser.add_argument("--capture-id")
    parser.add_argument("--json", action="store_true", dest="as_json")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(list(sys.argv[1:] if argv is None else argv))
    try:
        results, warnings = analyze_ledger(
            args.ledger,
            strategy_version=args.strategy_version,
            capture_id=args.capture_id,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}")
        return 2

    if args.as_json:
        payload = {
            "warning": "IN-SAMPLE EXPLORATORY — NOT A TRADING RECOMMENDATION",
            "sensitivity_scope": "in-sample exploratory; not the active policy",
            "captured_active_v2_configuration": (
                results[0].captured_configuration if results else None
            ),
            "results": [
                {
                    "minimum_short_return": str(result.thresholds.minimum_short_return),
                    "minimum_long_return": str(result.thresholds.minimum_long_return),
                    **report_to_dict(result.report),
                }
                for result in results
            ],
            "warnings": list(warnings),
        }
        print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
    else:
        print(format_results(results, warnings))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
