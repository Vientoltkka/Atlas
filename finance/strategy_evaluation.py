"""Persistent, deterministic evaluation of Atlas finance opportunity signals.

This module records CANDIDATE observations independently from PAPER trading.
It never creates orders, changes portfolios, or calls an LLM.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any, Sequence


SCHEMA_VERSION = 1
STATE_FILE = "signals.json"
EVALUATION_HORIZONS = (5, 20)


@dataclass(frozen=True)
class SignalEvaluation:
    symbol: str
    signal_day: date
    signal_close: Decimal
    rule: str
    return_5: Decimal | None = None
    return_20: Decimal | None = None
    max_favorable_20: Decimal | None = None
    max_adverse_20: Decimal | None = None

    @property
    def status(self) -> str:
        return "EVALUATED" if self.return_20 is not None else "OPEN"


class StrategyEvaluationStore:
    """Atomic JSON persistence, separate from PAPER portfolio state."""

    def __init__(
        self,
        directory: Path | str = Path(".atlas") / "finance_strategy_evaluation",
    ) -> None:
        self._directory = Path(directory)
        self._state_path = self._directory / STATE_FILE

    @property
    def state_path(self) -> Path:
        return self._state_path

    def entries(self) -> tuple[SignalEvaluation, ...]:
        if not self._state_path.exists():
            return ()

        try:
            data = json.loads(self._state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"strategy evaluation state corrupt: {exc}") from exc

        if not isinstance(data, dict) or data.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("unsupported strategy evaluation state")

        raw_entries = data.get("signals", [])
        if not isinstance(raw_entries, list):
            raise ValueError("strategy evaluation signals must be a list")

        return tuple(self._decode(item) for item in raw_entries)

    def record_candidate(
        self,
        *,
        symbol: str,
        signal_day: date,
        signal_close: Decimal,
        rule: str,
    ) -> SignalEvaluation:
        normalized = (symbol or "").strip().upper()
        if not normalized:
            raise ValueError("symbol is required")
        if signal_close <= 0:
            raise ValueError("signal_close must be positive")

        entries = list(self.entries())

        for entry in entries:
            if entry.symbol == normalized and entry.signal_day == signal_day:
                return entry

        signal = SignalEvaluation(
            symbol=normalized,
            signal_day=signal_day,
            signal_close=signal_close,
            rule=rule,
        )
        entries.append(signal)
        self._save(entries)
        return signal

    def evaluate_series(self, series) -> tuple[SignalEvaluation, ...]:
        """Update stored signals for one symbol from later daily closes."""
        symbol = (series.symbol or "").strip().upper()
        entries = list(self.entries())
        changed = False

        for index, entry in enumerate(entries):
            if entry.symbol != symbol:
                continue

            later = [bar for bar in series.bars if bar.day > entry.signal_day]
            if not later:
                continue

            return_5 = entry.return_5
            return_20 = entry.return_20
            favorable = entry.max_favorable_20
            adverse = entry.max_adverse_20

            if len(later) >= 5:
                return_5 = (later[4].close / entry.signal_close) - Decimal("1")

            if len(later) >= 20:
                window = later[:20]
                return_20 = (window[19].close / entry.signal_close) - Decimal("1")
                returns = [
                    (bar.close / entry.signal_close) - Decimal("1")
                    for bar in window
                ]
                favorable = max(returns)
                adverse = min(returns)

            updated = SignalEvaluation(
                symbol=entry.symbol,
                signal_day=entry.signal_day,
                signal_close=entry.signal_close,
                rule=entry.rule,
                return_5=return_5,
                return_20=return_20,
                max_favorable_20=favorable,
                max_adverse_20=adverse,
            )

            if updated != entry:
                entries[index] = updated
                changed = True

        if changed:
            self._save(entries)

        return tuple(entry for entry in entries if entry.symbol == symbol)

    def summary(self) -> dict[str, Any]:
        entries = self.entries()
        evaluated = [entry for entry in entries if entry.return_20 is not None]

        result: dict[str, Any] = {
            "total_signals": len(entries),
            "evaluated_20": len(evaluated),
            "open": len(entries) - len(evaluated),
            "win_rate_20": None,
            "average_return_20": None,
        }

        if evaluated:
            wins = sum(1 for entry in evaluated if entry.return_20 > 0)
            result["win_rate_20"] = Decimal(wins) / Decimal(len(evaluated))
            result["average_return_20"] = (
                sum((entry.return_20 for entry in evaluated), Decimal("0"))
                / Decimal(len(evaluated))
            )

        return result

    def _save(self, entries: Sequence[SignalEvaluation]) -> None:
        payload = {
            "schema_version": SCHEMA_VERSION,
            "signals": [self._encode(entry) for entry in entries],
        }

        self._directory.mkdir(parents=True, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(
            prefix=f".{STATE_FILE}.",
            suffix=".tmp",
            dir=str(self._directory),
        )

        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(
                    payload,
                    handle,
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
                handle.flush()
                os.fsync(handle.fileno())

            os.replace(tmp_path, self._state_path)
        except BaseException:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    @staticmethod
    def _encode(entry: SignalEvaluation) -> dict[str, Any]:
        return {
            "symbol": entry.symbol,
            "signal_day": entry.signal_day.isoformat(),
            "signal_close": str(entry.signal_close),
            "rule": entry.rule,
            "return_5": None if entry.return_5 is None else str(entry.return_5),
            "return_20": None if entry.return_20 is None else str(entry.return_20),
            "max_favorable_20": (
                None
                if entry.max_favorable_20 is None
                else str(entry.max_favorable_20)
            ),
            "max_adverse_20": (
                None
                if entry.max_adverse_20 is None
                else str(entry.max_adverse_20)
            ),
        }

    @staticmethod
    def _decode(item: dict[str, Any]) -> SignalEvaluation:
        def decimal_or_none(value):
            return None if value is None else Decimal(str(value))

        return SignalEvaluation(
            symbol=str(item["symbol"]).upper(),
            signal_day=date.fromisoformat(item["signal_day"]),
            signal_close=Decimal(str(item["signal_close"])),
            rule=str(item["rule"]),
            return_5=decimal_or_none(item.get("return_5")),
            return_20=decimal_or_none(item.get("return_20")),
            max_favorable_20=decimal_or_none(item.get("max_favorable_20")),
            max_adverse_20=decimal_or_none(item.get("max_adverse_20")),
        )
