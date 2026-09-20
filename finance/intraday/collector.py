"""Research-only collector connecting validated quotes to intraday evidence."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

from finance.intraday.bridge import observation_from_quote
from finance.intraday.evaluation import EventDetector, IntradaySignalEvaluator
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
        event_detector: EventDetector | None = None,
        event_source: str | None = None,
    ) -> None:
        self._signal_service = signal_service
        self._ledger = ledger
        self._evaluator = evaluator or IntradaySignalEvaluator()
        self._record_no_action = record_no_action
        self._event_detector = event_detector
        self._event_source = event_source
        self._events: dict[str, object] = {}

        self._observations: dict[str, list[IntradayObservation]] = {}
        self._candidates: dict[
            tuple[str, object],
            IntradaySignal,
        ] = {}
        self._completed: set[
            tuple[str, object, int]
        ] = set()

        self._restore_from_ledger()

    @classmethod
    def for_time_based_v2(
        cls,
        *,
        ledger: IntradayResearchLedger,
        signal_service=None,
        evaluator: IntradaySignalEvaluator | None = None,
    ) -> "IntradayResearchCollector":
        from finance.intraday.service import TimeBasedIntradaySignalService
        from finance.intraday.time_replay import TIME_BASED_STRATEGY_VERSION

        if ledger.strategy_version != TIME_BASED_STRATEGY_VERSION:
            raise ValueError(
                "time-based V2 requires the V2 strategy version"
            )

        service = signal_service or TimeBasedIntradaySignalService()
        ledger.ensure_configuration(service.configuration)
        return cls(
            signal_service=service,
            ledger=ledger,
            evaluator=evaluator,
            record_no_action=True,
            event_detector=EventDetector(),
            event_source=TIME_BASED_STRATEGY_VERSION,
        )

    def _restore_from_ledger(self) -> None:
        for record in self._ledger.records():
            if record.get("strategy_version") != self._ledger.strategy_version:
                continue
            record_type = record.get("type")

            if record_type == "OBSERVATION":
                observation = self._observation_from_record(record)
                existing = self._observations.setdefault(
                    observation.symbol,
                    [],
                )
                if any(
                    item.timestamp == observation.timestamp
                    for item in existing
                ):
                    continue
                existing.append(observation)

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

            elif record_type in {"EVENT", "EVENT_UPDATE"}:
                event = self._ledger.event_from_record(record)
                self._events[event.id] = event
                if self._event_detector is not None:
                    self._event_detector.restore(event)

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

        # Safely complete outcomes from observations already in this ledger.
        # The append-only history is preserved; only missing event snapshots
        # are added, so a restart cannot duplicate resolved updates.
        if self._event_detector is not None:
            for event in self._event_detector.update_pending(
                [item for items in self._observations.values() for item in items]
            ):
                self._ledger.record_event_update(event)

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
            direction=record.get("direction"),
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
            if self._event_detector is not None:
                for event in self._event_detector.update_pending(
                    observations,
                    symbol=observation.symbol,
                ):
                    self._ledger.record_event_update(event)
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
            if self._event_detector is not None:
                if signal.direction not in {"LONG", "SHORT"}:
                    raise ValueError("candidate signal requires a valid direction")
                event = self._event_detector.observe(
                    symbol=signal.symbol,
                    direction=signal.direction,
                    signal="MOMENTUM",
                    source=self._event_source or self._ledger.strategy_version,
                    timestamp=signal.timestamp,
                    entry_price=signal.features.last_price,
                )
                event.calculate_outcomes(observations)
                if event.id not in self._events:
                    self._ledger.record_event(event)
                    self._events[event.id] = event
                else:
                    self._ledger.record_event_update(event)

        if self._event_detector is not None:
            for event in self._event_detector.update_pending(
                observations,
                symbol=observation.symbol,
            ):
                self._ledger.record_event_update(event)

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
