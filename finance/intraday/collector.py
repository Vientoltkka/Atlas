"""Research-only collector connecting validated quotes to intraday evidence."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

from finance.intraday.bridge import observation_from_quote
from finance.intraday.evaluation import IntradaySignalEvaluator
from finance.intraday.models import (
    IntradayFeatures,
    IntradayObservation,
    IntradaySignal,
    IntradaySignalAction,
)
from finance.intraday.persistence import IntradayResearchLedger
from finance.intraday.service import IntradaySignalService
from finance.market_data.models import Quote


@dataclass(frozen=True)
class IntradayCollectorResult:
    accepted_observation: bool
    signal: IntradaySignal | None = None
    recorded_signal: bool = False
    recorded_outcomes: int = 0


class IntradayResearchCollector:
    """Accumulates research evidence without execution capability."""

    def __init__(
        self,
        *,
        signal_service: IntradaySignalService,
        ledger: IntradayResearchLedger,
        evaluator: IntradaySignalEvaluator | None = None,
        record_no_action: bool = False,
    ) -> None:
        self._signal_service = signal_service
        self._ledger = ledger
        self._evaluator = evaluator or IntradaySignalEvaluator()
        self._record_no_action = record_no_action

        self._observations: dict[str, list[IntradayObservation]] = {}
        self._candidates: dict[
            tuple[str, object],
            IntradaySignal,
        ] = {}
        self._completed: set[
            tuple[str, object, int]
        ] = set()

        self._restore_from_ledger()

    def _restore_from_ledger(self) -> None:
        for record in self._ledger.records():
            record_type = record.get("type")

            if record_type == "OBSERVATION":
                observation = self._observation_from_record(record)
                self._observations.setdefault(
                    observation.symbol,
                    [],
                ).append(observation)

                # Rebuild the signal service buffer without
                # writing new research evidence.
                self._signal_service.ingest(observation)

            elif (
                record_type == "SIGNAL"
                and record.get("action") == "CANDIDATE"
            ):
                signal = self._signal_from_record(record)
                self._candidates[
                    (signal.symbol, signal.timestamp)
                ] = signal

            elif record_type == "OUTCOME":
                self._completed.add(
                    (
                        str(record["symbol"]).strip().upper(),
                        datetime.fromisoformat(
                            record["signal_timestamp"]
                        ),
                        int(record["horizon_minutes"]),
                    )
                )

        for symbol in self._observations:
            self._observations[symbol].sort(
                key=lambda item: item.timestamp
            )

    @staticmethod
    def _observation_from_record(
        record: dict,
    ) -> IntradayObservation:
        volume = record.get("volume")

        return IntradayObservation(
            symbol=record["symbol"],
            price=Decimal(record["price"]),
            bid=Decimal(record["bid"]),
            ask=Decimal(record["ask"]),
            timestamp=datetime.fromisoformat(
                record["timestamp"]
            ),
            volume=(
                Decimal(volume)
                if volume is not None
                else None
            ),
        )

    @staticmethod
    def _signal_from_record(
        record: dict,
    ) -> IntradaySignal:
        raw = record["features"]

        features = IntradayFeatures(
            symbol=raw["symbol"],
            timestamp=datetime.fromisoformat(
                raw["timestamp"]
            ),
            observations=int(raw["observations"]),
            last_price=Decimal(raw["last_price"]),
            return_short=Decimal(raw["return_short"]),
            return_long=Decimal(raw["return_long"]),
            acceleration=Decimal(raw["acceleration"]),
            realized_volatility=Decimal(
                raw["realized_volatility"]
            ),
            relative_spread=Decimal(
                raw["relative_spread"]
            ),
        )

        return IntradaySignal(
            symbol=record["symbol"],
            timestamp=datetime.fromisoformat(
                record["timestamp"]
            ),
            action=IntradaySignalAction(
                record["action"]
            ),
            reasons=tuple(record["reasons"]),
            features=features,
        )

    def ingest_quote(
        self,
        quote: Quote,
    ) -> IntradayCollectorResult:
        observation = observation_from_quote(quote)

        observations = self._observations.setdefault(
            observation.symbol,
            [],
        )

        if (
            observations
            and observation.timestamp <= observations[-1].timestamp
        ):
            return IntradayCollectorResult(
                accepted_observation=False,
            )

        observations.append(observation)
        self._ledger.record_observation(observation)

        recorded_outcomes = self._evaluate_candidates(
            observation.symbol
        )

        signal = self._signal_service.ingest(observation)

        if signal is None:
            return IntradayCollectorResult(
                accepted_observation=True,
                recorded_outcomes=recorded_outcomes,
            )

        should_record = (
            signal.action is IntradaySignalAction.CANDIDATE
            or self._record_no_action
        )

        if should_record:
            self._ledger.record_signal(signal)

        if signal.action is IntradaySignalAction.CANDIDATE:
            key = (signal.symbol, signal.timestamp)
            self._candidates[key] = signal

        return IntradayCollectorResult(
            accepted_observation=True,
            signal=signal,
            recorded_signal=should_record,
            recorded_outcomes=recorded_outcomes,
        )

    def _evaluate_candidates(
        self,
        symbol: str,
    ) -> int:
        observations = self._observations.get(symbol, [])
        recorded = 0

        candidates = [
            signal
            for (candidate_symbol, _), signal
            in self._candidates.items()
            if candidate_symbol == symbol
        ]

        for signal in candidates:
            outcomes = self._evaluator.evaluate(
                signal,
                observations,
            )

            for outcome in outcomes:
                key = (
                    outcome.symbol,
                    outcome.signal_timestamp,
                    outcome.horizon_minutes,
                )

                if key in self._completed:
                    continue

                self._ledger.record_outcome(outcome)
                self._completed.add(key)
                recorded += 1

        return recorded
