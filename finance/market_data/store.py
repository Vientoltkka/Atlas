"""Latest normalized market state."""

from __future__ import annotations

from finance.market_data.models import Bar, Quote, Trade


class MarketDataStore:
    def __init__(self) -> None:
        self._quotes: dict[str, Quote] = {}
        self._trades: dict[str, Trade] = {}
        self._bars: dict[tuple[str, str], Bar] = {}

    @staticmethod
    def _newer(
        current: Quote | Trade | None,
        incoming: Quote | Trade,
    ) -> bool:
        if current is None:
            return True

        if current.sequence is not None and incoming.sequence is not None:
            return incoming.sequence > current.sequence

        return incoming.timestamp > current.timestamp

    def put_quote(self, quote: Quote) -> bool:
        current = self._quotes.get(quote.symbol)
        if not self._newer(current, quote):
            return False
        self._quotes[quote.symbol] = quote
        return True

    def put_trade(self, trade: Trade) -> bool:
        current = self._trades.get(trade.symbol)
        if not self._newer(current, trade):
            return False
        self._trades[trade.symbol] = trade
        return True

    def put_bar(self, bar: Bar) -> bool:
        key = (bar.symbol, bar.interval)
        current = self._bars.get(key)

        if current is not None and bar.end <= current.end:
            return False

        self._bars[key] = bar
        return True

    def quote(self, symbol: str) -> Quote | None:
        return self._quotes.get(symbol.strip().upper())

    def trade(self, symbol: str) -> Trade | None:
        return self._trades.get(symbol.strip().upper())

    def bar(self, symbol: str, interval: str) -> Bar | None:
        return self._bars.get((symbol.strip().upper(), interval))
