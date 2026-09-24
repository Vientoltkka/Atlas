"""Append-only persistence for intraday research evidence."""

from __future__ import annotations

import json
import os
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from uuid import uuid4

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
        capture_id: str | None = None,
    ) -> None:
        version = strategy_version.strip()
        if not version:
            raise ValueError("strategy_version is required")

        self._path = Path(path)
        self._strategy_version = version
        existing_capture_ids = {
            record["capture_id"]
            for record in self.records()
            if record.get("strategy_version") == version
            and isinstance(record.get("capture_id"), str)
            and record["capture_id"].strip()
        }
        if capture_id is None:
            if len(existing_capture_ids) > 1:
                raise ValueError(
                    "capture_id is required for a ledger with multiple captures: "
                    + ", ".join(sorted(existing_capture_ids))
                )
            capture_id = next(iter(existing_capture_ids), None) or str(uuid4())
        capture_id = capture_id.strip()
        if not capture_id:
            raise ValueError("capture_id is required")
        self._capture_id = capture_id

    @property
    def path(self) -> Path:
        return self._path

    @property
    def strategy_version(self) -> str:
        return self._strategy_version

    @property
    def capture_id(self) -> str:
        return self._capture_id

    def _metadata(self) -> dict[str, str]:
        return {
            "strategy_version": self._strategy_version,
            "capture_id": self._capture_id,
        }

    def available_capture_ids(self) -> tuple[str, ...]:
        return tuple(sorted({
            record["capture_id"]
            for record in self.records()
            if record.get("strategy_version") == self._strategy_version
            and isinstance(record.get("capture_id"), str)
            and record["capture_id"].strip()
        }))

    def _append(self, record: dict) -> None:
        record.setdefault("strategy_version", self._strategy_version)
        record.setdefault("capture_id", self._capture_id)
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
                **self._metadata(),
                **asdict(observation),
            }
        )

    def ensure_configuration(self, configuration: dict) -> None:
        """Persist the effective research configuration once per strategy."""
        expected = json.dumps(
            configuration,
            default=_json_value,
            sort_keys=True,
            separators=(",", ":"),
        )
        for record in self.records():
            if (
                record.get("type") == "CONFIGURATION"
                and record.get("strategy_version") == self._strategy_version
                and record.get("capture_id") == self._capture_id
            ):
                actual = json.dumps(
                    record.get("configuration"),
                    default=_json_value,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                if actual != expected:
                    raise ValueError(
                        "configuration snapshot differs for capture_id: "
                        f"{self._capture_id}"
                    )
                return
        self._append(
            {
                "type": "CONFIGURATION",
                **self._metadata(),
                "configuration": deepcopy(configuration),
            }
        )

    def record_signal(
        self,
        signal: IntradaySignal,
        *,
        event_id: str | None = None,
    ) -> None:
        if event_id is not None:
            if not isinstance(event_id, str):
                raise ValueError("event_id must be a string")
            event_id = event_id.strip()
            if not event_id:
                raise ValueError("event_id cannot be empty")

        record = {
                "type": "SIGNAL",
                **self._metadata(),
                "symbol": signal.symbol,
                "timestamp": signal.timestamp,
                "action": signal.action,
                "reasons": list(signal.reasons),
                "direction": signal.direction,
                "features": asdict(signal.features),
        }
        if event_id is not None:
            record["event_id"] = event_id
        self._append(record)

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
            **self._metadata(),
            **self._event_payload(event),
        })

    def record_event_update(self, event: IndependentSignalEvent) -> None:
        self._append({
            "type": "EVENT_UPDATE",
            **self._metadata(),
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
                **self._metadata(),
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
