"""Bounded per-symbol observation buffer."""

from __future__ import annotations

from collections import defaultdict, deque

from finance.intraday.models import IntradayObservation


class IntradayObservationBuffer:
    def __init__(self, *, maxlen: int = 120) -> None:
        if maxlen < 2:
            raise ValueError("maxlen must be >= 2")

        self._maxlen = maxlen
        self._items: dict[str, deque[IntradayObservation]] = defaultdict(
            lambda: deque(maxlen=self._maxlen)
        )

    def add(self, observation: IntradayObservation) -> bool:
        bucket = self._items[observation.symbol]

        if bucket and observation.timestamp <= bucket[-1].timestamp:
            return False

        bucket.append(observation)
        return True

    def observations(self, symbol: str) -> list[IntradayObservation]:
        return list(self._items.get(symbol.strip().upper(), ()))
