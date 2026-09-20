"""Append-only persistence for intraday research evidence."""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from datetime import datetime
from decimal import Decimal
from pathlib import Path

from finance.intraday.evaluation import (
    EventOutcome,
    IndependentSignalEvent,
    IntradaySignalOutcome,
)
from finance.intraday.models import (
    IntradayObservation,
    IntradaySignal,
)


def _json_value(value):
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if hasattr(value, "value"):
        return value.value
    raise TypeError(f"unsupported JSON value: {type(value)!r}")


class IntradayResearchLedger:
    DEFAULT_STRATEGY_VERSION = "intraday-momentum-v1"

    def __init__(
        self,
        path: str | Path,
        *,
        strategy_version: str = DEFAULT_STRATEGY_VERSION,
    ) -> None:
        version = strategy_version.strip()
        if not version:
            raise ValueError("strategy_version is required")

        self._path = Path(path)
        self._strategy_version = version

    @property
    def path(self) -> Path:
        return self._path

    @property
    def strategy_version(self) -> str:
        return self._strategy_version

    def _append(self, record: dict) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)

        payload = json.dumps(
            record,
            default=_json_value,
            sort_keys=True,
            separators=(",", ":"),
        )

        with self._path.open(
            "a",
            encoding="utf-8",
        ) as handle:
            handle.write(payload)
            handle.write(chr(10))
            handle.flush()
            os.fsync(handle.fileno())

    def record_observation(
        self,
        observation: IntradayObservation,
    ) -> None:
        self._append(
            {
                "type": "OBSERVATION",
                "strategy_version": self._strategy_version,
                **asdict(observation),
            }
        )

    def ensure_configuration(self, configuration: dict) -> None:
        """Persist the effective research configuration once per strategy."""
        if any(
            record.get("type") == "CONFIGURATION"
            and record.get("strategy_version") == self._strategy_version
            for record in self.records()
        ):
            return
        self._append(
            {
                "type": "CONFIGURATION",
                "strategy_version": self._strategy_version,
                "configuration": configuration,
            }
        )

    def record_signal(self, signal: IntradaySignal) -> None:
        self._append(
            {
                "type": "SIGNAL",
                "strategy_version": self._strategy_version,
                "symbol": signal.symbol,
                "timestamp": signal.timestamp,
                "action": signal.action,
                "reasons": list(signal.reasons),
                "direction": signal.direction,
                "features": asdict(signal.features),
            }
        )

    @staticmethod
    def _event_payload(event: IndependentSignalEvent) -> dict:
        outcomes = {}
        for horizon in (1, 5, 15, 30):
            outcome = getattr(event, f"outcome_{horizon}m")
            outcomes[str(horizon)] = None if outcome is None else asdict(outcome)
        return {
            "id": event.id,
            "symbol": event.symbol,
            "direction": event.direction,
            "signal": event.signal,
            "source": event.source,
            "timestamp": event.timestamp,
            "entry_price": event.entry_price,
            "observation_count": event.observation_count,
            "cooldown_minutes": event.cooldown_minutes,
            "estimated_round_trip_cost": event.estimated_round_trip_cost,
            "last_observation_timestamp": event.last_observation_timestamp,
            "outcomes": outcomes,
        }

    def record_event(self, event: IndependentSignalEvent) -> None:
        self._append({
            "type": "EVENT",
            "strategy_version": self._strategy_version,
            **self._event_payload(event),
        })

    def record_event_update(self, event: IndependentSignalEvent) -> None:
        self._append({
            "type": "EVENT_UPDATE",
            "strategy_version": self._strategy_version,
            **self._event_payload(event),
        })

    @staticmethod
    def event_from_record(record: dict) -> IndependentSignalEvent:
        outcomes = {}
        for horizon in (1, 5, 15, 30):
            raw = record.get("outcomes", {}).get(str(horizon))
            outcomes[f"outcome_{horizon}m"] = (
                None if raw is None else EventOutcome(
                    future_price=Decimal(raw["future_price"])
                    if raw["future_price"] is not None else None,
                    gross_return=Decimal(raw["gross_return"])
                    if raw["gross_return"] is not None else None,
                    net_return=Decimal(raw["net_return"])
                    if raw["net_return"] is not None else None,
                )
            )
        return IndependentSignalEvent(
            symbol=record["symbol"],
            direction=record["direction"],
            signal=record["signal"],
            source=record["source"],
            timestamp=datetime.fromisoformat(record["timestamp"]),
            entry_price=Decimal(record["entry_price"]),
            observation_count=int(record["observation_count"]),
            cooldown_minutes=int(record["cooldown_minutes"]),
            estimated_round_trip_cost=Decimal(record["estimated_round_trip_cost"]),
            last_observation_timestamp=datetime.fromisoformat(
                record.get("last_observation_timestamp", record["timestamp"])
            ),
            **outcomes,
        )

    def record_outcome(
        self,
        outcome: IntradaySignalOutcome,
    ) -> None:
        self._append(
            {
                "type": "OUTCOME",
                "strategy_version": self._strategy_version,
                **asdict(outcome),
            }
        )

    def records(self) -> tuple[dict, ...]:
        if not self._path.exists():
            return ()

        result: list[dict] = []

        with self._path.open(
            "r",
            encoding="utf-8",
        ) as handle:
            for line in handle:
                line = line.strip()
                if line:
                    result.append(json.loads(line))

        return tuple(result)
