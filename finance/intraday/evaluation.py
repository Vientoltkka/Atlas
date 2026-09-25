"""Forward evaluation of intraday candidate signals.

Evaluation is research-only and has no order-execution capability.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from hashlib import sha256

from finance.intraday.models import (
    IntradayObservation,
    IntradaySignal,
    IntradaySignalAction,
)

DEFAULT_MAXIMUM_LATENESS_SECONDS = 30


@dataclass(frozen=True)
class IntradaySignalOutcome:
    symbol: str
    signal_timestamp: datetime
    horizon_minutes: int
    entry_price: Decimal
    exit_price: Decimal
    forward_return: Decimal
    maximum_favorable_excursion: Decimal
    maximum_adverse_excursion: Decimal
    observations: int


class IntradaySignalEvaluator:
    def __init__(
        self,
        *,
        horizons_minutes: tuple[int, ...] = (1, 5, 15, 30),
        maximum_lateness_seconds: int = DEFAULT_MAXIMUM_LATENESS_SECONDS,
    ) -> None:
        if not horizons_minutes:
            raise ValueError("at least one horizon is required")

        if any(value <= 0 for value in horizons_minutes):
            raise ValueError("horizons must be positive")

        if len(set(horizons_minutes)) != len(horizons_minutes):
            raise ValueError("horizons must be unique")

        if maximum_lateness_seconds < 0:
            raise ValueError("maximum lateness cannot be negative")

        self._horizons = tuple(sorted(horizons_minutes))
        self._maximum_lateness = timedelta(
            seconds=maximum_lateness_seconds
        )

    def evaluate(
        self,
        signal: IntradaySignal,
        observations: list[IntradayObservation],
    ) -> tuple[IntradaySignalOutcome, ...]:
        if signal.action is not IntradaySignalAction.CANDIDATE:
            return ()

        future = [
            item
            for item in observations
            if item.symbol == signal.symbol
            and item.timestamp > signal.timestamp
        ]

        if not future:
            return ()

        future.sort(key=lambda item: item.timestamp)

        entry = signal.features.last_price
        outcomes: list[IntradaySignalOutcome] = []

        for horizon in self._horizons:
            target = signal.timestamp + timedelta(minutes=horizon)

            exit_observation = next(
                (
                    item
                    for item in future
                    if item.timestamp >= target
                ),
                None,
            )

            # Do not score an incomplete or excessively late horizon.
            if exit_observation is None:
                continue

            if (
                exit_observation.timestamp - target
                > self._maximum_lateness
            ):
                continue

            eligible = [
                item
                for item in future
                if item.timestamp <= exit_observation.timestamp
            ]

            prices = [item.price for item in eligible]
            exit_price = exit_observation.price

            forward_return = (exit_price / entry) - Decimal("1")
            mfe = (max(prices) / entry) - Decimal("1")
            mae = (min(prices) / entry) - Decimal("1")

            outcomes.append(
                IntradaySignalOutcome(
                    symbol=signal.symbol,
                    signal_timestamp=signal.timestamp,
                    horizon_minutes=horizon,
                    entry_price=entry,
                    exit_price=exit_price,
                    forward_return=forward_return,
                    maximum_favorable_excursion=mfe,
                    maximum_adverse_excursion=mae,
                    observations=len(eligible),
                )
            )

        return tuple(outcomes)


# Research-only default: an illustrative round-trip cost hypothesis, not a
# statement about any broker's actual commission or spread.
DEFAULT_ESTIMATED_ROUND_TRIP_COST = Decimal("0.0010")
INDEPENDENT_EVENT_HORIZONS = (1, 5, 15, 30)


@dataclass
class EventOutcome:
    future_price: Decimal | None
    gross_return: Decimal | None
    net_return: Decimal | None


@dataclass
class IndependentSignalEvent:
    """A deduplicated signal occurrence for research evaluation only."""

    symbol: str
    direction: str
    signal: str
    source: str
    timestamp: datetime
    entry_price: Decimal
    observation_count: int = 1
    cooldown_minutes: int = 15
    estimated_round_trip_cost: Decimal = DEFAULT_ESTIMATED_ROUND_TRIP_COST
    outcome_1m: EventOutcome | None = None
    outcome_5m: EventOutcome | None = None
    outcome_15m: EventOutcome | None = None
    outcome_30m: EventOutcome | None = None
    last_observation_timestamp: datetime | None = None
    _last_observation_timestamp: datetime = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.symbol = self.symbol.strip().upper()
        self.direction = self.direction.strip().upper()
        self.signal = self.signal.strip()
        self.source = self.source.strip()
        if not self.symbol or not self.direction or not self.signal or not self.source:
            raise ValueError("symbol, direction, signal, and source are required")
        if self.timestamp.tzinfo is None or self.timestamp.utcoffset() is None:
            raise ValueError("timestamp must be timezone-aware")
        if self.entry_price <= 0:
            raise ValueError("entry_price must be positive")
        if self.observation_count < 1:
            raise ValueError("observation_count must be positive")
        if self.cooldown_minutes < 0:
            raise ValueError("cooldown_minutes cannot be negative")
        if self.estimated_round_trip_cost < 0:
            raise ValueError("estimated_round_trip_cost cannot be negative")
        if self.last_observation_timestamp is None:
            self.last_observation_timestamp = self.timestamp
        if (
            self.last_observation_timestamp.tzinfo is None
            or self.last_observation_timestamp.utcoffset() is None
        ):
            raise ValueError("last_observation_timestamp must be timezone-aware")
        if self.last_observation_timestamp < self.timestamp:
            raise ValueError("last_observation_timestamp cannot precede timestamp")
        self._last_observation_timestamp = self.last_observation_timestamp

    @property
    def id(self) -> str:
        identity = "|".join(
            (self.symbol, self.direction, self.signal, self.source,
             self.timestamp.isoformat(), str(self.entry_price))
        )
        return sha256(identity.encode("utf-8")).hexdigest()

    @property
    def equivalence_key(self) -> tuple[str, str, str, str]:
        return (self.symbol, self.direction, self.signal, self.source)

    def calculate_outcomes(
        self,
        observations: list[IntradayObservation],
    ) -> bool:
        changed = False
        future = sorted(
            (item for item in observations
             if item.symbol == self.symbol and item.timestamp > self.timestamp),
            key=lambda item: item.timestamp,
        )
        for horizon in INDEPENDENT_EVENT_HORIZONS:
            current = getattr(self, f"outcome_{horizon}m")
            if current is not None and current.future_price is not None:
                continue
            target = self.timestamp + timedelta(minutes=horizon)
            item = next((item for item in future if item.timestamp >= target), None)
            valid_item = (
                item is not None
                and item.timestamp - target
                <= timedelta(seconds=DEFAULT_MAXIMUM_LATENESS_SECONDS)
            )
            if valid_item:
                gross = self._gross_return(item.price)
                outcome = EventOutcome(
                    future_price=item.price,
                    gross_return=gross,
                    net_return=gross - self.estimated_round_trip_cost,
                )
                changed = changed or current is None or current.future_price is None
                if current is None or current.future_price is None:
                    setattr(self, f"outcome_{horizon}m", outcome)
            elif current is None:
                # Preserve the existing public shape while the horizon is pending.
                setattr(self, f"outcome_{horizon}m", EventOutcome(None, None, None))
        return changed

    def _gross_return(self, future_price: Decimal) -> Decimal:
        if self.direction == "LONG":
            return (future_price - self.entry_price) / self.entry_price
        if self.direction == "SHORT":
            return (self.entry_price - future_price) / self.entry_price
        raise ValueError("direction must be LONG or SHORT")


class EventDetector:
    """Groups equivalent observations during a configurable cooldown."""

    def __init__(
        self,
        *,
        cooldown_minutes: int = 15,
        estimated_round_trip_cost: Decimal = DEFAULT_ESTIMATED_ROUND_TRIP_COST,
    ) -> None:
        if cooldown_minutes < 0:
            raise ValueError("cooldown_minutes cannot be negative")
        if estimated_round_trip_cost < 0:
            raise ValueError("estimated_round_trip_cost cannot be negative")
        self.cooldown_minutes = cooldown_minutes
        self.estimated_round_trip_cost = estimated_round_trip_cost
        self._active: dict[tuple[str, str, str, str], IndependentSignalEvent] = {}
        self._pending: dict[str, IndependentSignalEvent] = {}
        self._last_observation_timestamps: dict[
            tuple[str, str, str, str], datetime
        ] = {}

    def observe(
        self,
        *,
        symbol: str,
        direction: str,
        signal: str,
        source: str,
        timestamp: datetime,
        entry_price: Decimal,
    ) -> IndependentSignalEvent:
        normalized_direction = direction.strip().upper()
        if normalized_direction not in {"LONG", "SHORT"}:
            raise ValueError("direction must be LONG or SHORT")
        if timestamp.tzinfo is None or timestamp.utcoffset() is None:
            raise ValueError("timestamp must be timezone-aware")

        key = (symbol.strip().upper(), normalized_direction, signal.strip(), source.strip())
        last_timestamp = self._last_observation_timestamps.get(key)
        if last_timestamp is not None and timestamp < last_timestamp:
            raise ValueError("observation timestamp cannot move backwards")

        current = self._active.get(key)
        if current is not None:
            elapsed = timestamp - current._last_observation_timestamp
            if elapsed < timedelta(minutes=self.cooldown_minutes):
                current.observation_count += 1
                current._last_observation_timestamp = timestamp
                current.last_observation_timestamp = timestamp
                self._last_observation_timestamps[key] = timestamp
                return current

        event = IndependentSignalEvent(
            symbol=symbol,
            direction=normalized_direction,
            signal=signal,
            source=source,
            timestamp=timestamp,
            entry_price=entry_price,
            cooldown_minutes=self.cooldown_minutes,
            estimated_round_trip_cost=self.estimated_round_trip_cost,
        )
        self._active[key] = event
        self._pending[event.id] = event
        self._last_observation_timestamps[key] = timestamp
        return event

    def restore(self, event: IndependentSignalEvent) -> None:
        """Restore the latest append-only snapshot for an open event."""
        key = event.equivalence_key
        last_timestamp = event._last_observation_timestamp
        previous = self._last_observation_timestamps.get(key)
        if previous is not None and last_timestamp < previous:
            raise ValueError("event observation timestamp cannot move backwards")
        self._active[key] = event
        self._pending[event.id] = event
        self._last_observation_timestamps[key] = last_timestamp

    def update_pending(
        self,
        observations: list[IntradayObservation],
        *,
        symbol: str | None = None,
    ) -> tuple[IndependentSignalEvent, ...]:
        """Update open events from any accepted observation.

        Only a newly resolved horizon is reported as a material change. Empty
        pending outcome shells are intentionally not reported.
        """
        normalized_symbol = symbol.strip().upper() if symbol is not None else None
        changed: list[IndependentSignalEvent] = []
        # Cooldown selects the active event; every open event still needs its
        # own forward horizons evaluated after a newer event replaces it.
        for event_id, event in list(self._pending.items()):
            if normalized_symbol is not None and event.symbol != normalized_symbol:
                continue
            if event.calculate_outcomes(observations):
                changed.append(event)
            if all(
                getattr(event, f"outcome_{horizon}m").future_price is not None
                for horizon in INDEPENDENT_EVENT_HORIZONS
            ):
                self._pending.pop(event_id, None)
        return tuple(changed)
