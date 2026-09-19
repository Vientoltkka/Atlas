"""Public read-only Revolut X market-data adapter.

Uses only public market-data endpoints.
No credentials, account access or order execution.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from datetime import datetime, timezone
from decimal import Decimal
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from finance.market_data.models import Quote


class RevolutXPublicMarketDataProvider:
    BASE_URL = "https://revx.revolut.com/api"
    REGION = "EEA"

    def __init__(
        self,
        *,
        timeout_seconds: float = 10.0,
    ) -> None:
        self._timeout_seconds = timeout_seconds
        self._connected = False
        self._symbols: tuple[str, ...] = ()

    @property
    def name(self) -> str:
        return "REVOLUT_X_PUBLIC"

    def connect(self) -> None:
        # REST public provider: logical connection only.
        self._connected = True

    def disconnect(self) -> None:
        self._connected = False

    @staticmethod
    def _to_api_symbol(symbol: str) -> str:
        value = symbol.strip().upper().replace("/", "-")
        if not value:
            raise ValueError("symbol is required")
        return value

    @staticmethod
    def _to_atlas_symbol(symbol: str) -> str:
        return symbol.strip().upper().replace("/", "-")

    def subscribe(self, symbols: Iterable[str]) -> None:
        if not self._connected:
            raise RuntimeError("provider is not connected")

        normalized = tuple(
            dict.fromkeys(
                self._to_api_symbol(symbol)
                for symbol in symbols
                if symbol.strip()
            )
        )

        if not normalized:
            raise ValueError("at least one symbol is required")

        self._symbols = normalized

    def _request_tickers(self) -> dict:
        if not self._symbols:
            raise RuntimeError("no symbols subscribed")

        query = urlencode(
            {
                "symbols": ",".join(self._symbols),
                "region": self.REGION,
            }
        )

        url = f"{self.BASE_URL}/1.0/public/tickers?{query}"

        request = Request(
            url,
            headers={
                "Accept": "application/json",
                "User-Agent": "Atlas/1.0",
            },
            method="GET",
        )

        with urlopen(
            request,
            timeout=self._timeout_seconds,
        ) as response:
            if response.status != 200:
                raise ConnectionError(
                    f"Revolut X HTTP {response.status}"
                )

            return json.loads(
                response.read().decode("utf-8")
            )

    @staticmethod
    def _timestamp_from_metadata(payload: dict) -> datetime:
        metadata = payload.get("metadata")

        if not isinstance(metadata, dict):
            raise ValueError("missing metadata")

        raw_timestamp = metadata.get("timestamp")

        if raw_timestamp is None:
            raise ValueError("missing market-data timestamp")

        try:
            milliseconds = int(raw_timestamp)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "invalid market-data timestamp"
            ) from exc

        return datetime.fromtimestamp(
            milliseconds / 1000,
            tz=timezone.utc,
        )

    def events(self):
        if not self._connected:
            raise RuntimeError("provider is not connected")

        payload = self._request_tickers()
        timestamp = self._timestamp_from_metadata(payload)

        rows = payload.get("data")

        if not isinstance(rows, list):
            raise ValueError("invalid ticker data")

        wanted = set(self._symbols)

        for row in rows:
            if not isinstance(row, dict):
                continue

            raw_symbol = row.get("symbol")
            if not isinstance(raw_symbol, str):
                continue

            symbol = self._to_atlas_symbol(raw_symbol)

            if symbol not in wanted:
                continue

            try:
                bid = Decimal(str(row["bid"]))
                ask = Decimal(str(row["ask"]))
            except (KeyError, ValueError, TypeError) as exc:
                raise ValueError(
                    f"invalid ticker for {symbol}"
                ) from exc

            yield Quote(
                symbol=symbol,
                bid=bid,
                ask=ask,
                timestamp=timestamp,
                provider=self.name,
            )
