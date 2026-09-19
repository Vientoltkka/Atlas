"""Bridge from normalized market data to intraday research."""

from __future__ import annotations

from decimal import Decimal

from finance.intraday.models import IntradayObservation
from finance.market_data.models import Quote


def observation_from_quote(quote: Quote) -> IntradayObservation:
    midpoint = (quote.bid + quote.ask) / Decimal("2")

    return IntradayObservation(
        symbol=quote.symbol,
        price=midpoint,
        bid=quote.bid,
        ask=quote.ask,
        timestamp=quote.timestamp,
    )
