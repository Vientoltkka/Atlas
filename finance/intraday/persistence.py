"""Append-only persistence for intraday research evidence."""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from datetime import datetime
from decimal import Decimal
from pathlib import Path

from finance.intraday.evaluation import IntradaySignalOutcome
from finance.intraday.models import IntradaySignal


def _json_value(value):
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if hasattr(value, "value"):
        return value.value
    raise TypeError(f"unsupported JSON value: {type(value)!r}")


class IntradayResearchLedger:
    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)

    @property
    def path(self) -> Path:
        return self._path

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

    def record_signal(self, signal: IntradaySignal) -> None:
        self._append(
            {
                "type": "SIGNAL",
                "symbol": signal.symbol,
                "timestamp": signal.timestamp,
                "action": signal.action,
                "reasons": list(signal.reasons),
                "features": asdict(signal.features),
            }
        )

    def record_outcome(
        self,
        outcome: IntradaySignalOutcome,
    ) -> None:
        self._append(
            {
                "type": "OUTCOME",
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
