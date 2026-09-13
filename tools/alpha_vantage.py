"""Adaptador aislado de Alpha Vantage para Atlas Finance V2.4.

Vive en tools/ (junto a web_search) para mantener finance/ libre de red:
la pila paper solo importa stdlib y modulo finance, y este adaptador es
el unico punto de contacto con el proveedor de mercado.

Solo lectura y solo bajo peticion explicita del usuario: sin scheduler,
sin polling, sin alertas y sin llamadas automaticas. Devuelve modelos
tipados estrictos a partir de respuestas HTTP validadas; nunca expone,
registra ni incluye la API key en tests, y redacta secretos en los
errores. No toca MarketEvent, la cartera paper, las propuestas de orden
ni el broker.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import date
from decimal import Decimal, InvalidOperation
from typing import Any, Mapping

import httpx

ALPHAVANTAGE_ENV_VAR = "ALPHAVANTAGE_API_KEY"
DEFAULT_BASE_URL = "https://www.alphavantage.co/query"
DEFAULT_TIMEOUT_SECONDS = 10.0
DEFAULT_MAX_RESPONSE_BYTES = 2_000_000

_ERROR_MISSING_KEY = "MISSING_KEY"
_ERROR_RATE_LIMIT = "RATE_LIMIT"
_ERROR_INVALID_SYMBOL = "INVALID_SYMBOL"
_ERROR_INVALID_RESPONSE = "INVALID_RESPONSE"
_ERROR_NETWORK = "NETWORK"
_ERROR_HTTP_STATUS = "HTTP_STATUS"
_ERROR_RESPONSE_TOO_LARGE = "RESPONSE_TOO_LARGE"

_SYMBOL_PATTERN = re.compile(r"^[A-Za-z0-9.\-]{1,12}$")

_REDACTED = "***REDACTED***"


class AlphaVantageError(Exception):
    """Error controlado y seguro del adaptador Alpha Vantage."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True)
class SymbolMatch:
    """Resultado de busqueda de simbolo/activo en Alpha Vantage."""

    symbol: str
    name: str
    region: str
    currency: str
    asset_type: str


@dataclass(frozen=True)
class DailyBar:
    """Una sesion diaria OHLCV tal y como la devuelve el proveedor."""

    day: date
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: int


@dataclass(frozen=True)
class DailySeries:
    """Serie diaria OHLCV con los metadatos del proveedor."""

    symbol: str
    provider_last_refreshed: str
    provider_timezone: str
    bars: tuple[DailyBar, ...]

    @property
    def last_bar(self) -> DailyBar | None:
        return self.bars[-1] if self.bars else None


def normalize_symbol(symbol: str) -> str:
    """Normaliza y valida un simbolo de mercado sin inventar datos."""
    normalized = (symbol or "").strip().upper()
    if not normalized or _SYMBOL_PATTERN.match(normalized) is None:
        raise AlphaVantageError(
            _ERROR_INVALID_SYMBOL,
            f"Simbolo de mercado invalido: {symbol!r}.",
        )
    return normalized


class AlphaVantageClient:
    """Cliente HTTP de solo lectura para la API de Alpha Vantage.

    La API key se mantiene privada y nunca se incluye en mensajes de
    error, representaciones ni registros; las URLs con la clave no se
    registran en ningun sitio.
    """

    def __init__(
        self,
        api_key: str,
        *,
        transport: httpx.BaseTransport | None = None,
        base_url: str = DEFAULT_BASE_URL,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        max_response_bytes: int = DEFAULT_MAX_RESPONSE_BYTES,
    ) -> None:
        self._api_key = (api_key or "").strip()
        self._base_url = base_url
        self._max_response_bytes = max_response_bytes
        self._client = httpx.Client(
            transport=transport,
            timeout=httpx.Timeout(timeout_seconds),
            follow_redirects=True,
        )

    @classmethod
    def from_env(
        cls, environ: Mapping[str, str] | None = None, **kwargs: Any
    ) -> "AlphaVantageClient":
        env = os.environ if environ is None else environ
        return cls(env.get(ALPHAVANTAGE_ENV_VAR, ""), **kwargs)

    @property
    def configured(self) -> bool:
        """True solo si hay una API key no vacia; nunca revela su valor."""
        return bool(self._api_key)

    def search_symbol(self, query: str) -> tuple[SymbolMatch, ...]:
        """Busca simbolos/activos por texto (endpoint SYMBOL_SEARCH)."""
        keywords = (query or "").strip()
        if not keywords:
            raise AlphaVantageError(
                _ERROR_INVALID_SYMBOL, "La busqueda requiere un texto no vacio."
            )
        payload = self._get_json({"function": "SYMBOL_SEARCH", "keywords": keywords})
        matches = payload.get("bestMatches")
        if matches is None:
            return ()
        if not isinstance(matches, list):
            raise AlphaVantageError(
                _ERROR_INVALID_RESPONSE,
                "Respuesta de busqueda con formato inesperado en Alpha Vantage.",
            )
        parsed: list[SymbolMatch] = []
        for item in matches:
            if not isinstance(item, Mapping):
                raise AlphaVantageError(
                    _ERROR_INVALID_RESPONSE,
                    "Entrada de busqueda invalida en Alpha Vantage.",
                )
            symbol = item.get("1. symbol")
            if not isinstance(symbol, str) or not symbol.strip():
                raise AlphaVantageError(
                    _ERROR_INVALID_RESPONSE,
                    "Entrada de busqueda sin simbolo en Alpha Vantage.",
                )
            parsed.append(
                SymbolMatch(
                    symbol=symbol.strip().upper(),
                    name=_required_text(item, "2. name"),
                    region=_required_text(item, "4. region"),
                    currency=_required_text(item, "8. currency"),
                    asset_type=_required_text(item, "3. type"),
                )
            )
        return tuple(parsed)

    def daily_series(self, symbol: str) -> DailySeries:
        """Devuelve la serie diaria OHLCV (endpoint TIME_SERIES_DAILY)."""
        normalized = normalize_symbol(symbol)
        payload = self._get_json(
            {
                "function": "TIME_SERIES_DAILY",
                "symbol": normalized,
                "outputsize": "compact",
            }
        )
        meta = payload.get("Meta Data")
        series = payload.get("Time Series (Daily)")
        if not isinstance(meta, Mapping) or not isinstance(series, Mapping):
            raise AlphaVantageError(
                _ERROR_INVALID_RESPONSE,
                f"Respuesta de serie diaria invalida para {normalized}.",
            )
        bars = tuple(
            _parse_daily_bar(day, values) for day, values in sorted(series.items())
        )
        return DailySeries(
            symbol=str(meta.get("2. Symbol", normalized)).strip().upper(),
            provider_last_refreshed=_optional_text(meta, "3. Last Refreshed"),
            provider_timezone=_optional_text(meta, "4. Time Zone"),
            bars=bars,
        )

    def _get_json(self, params: Mapping[str, str]) -> Mapping[str, Any]:
        """Ejecuta una peticion GET validada sin registrar la URL con la clave."""
        if not self.configured:
            raise AlphaVantageError(
                _ERROR_MISSING_KEY,
                "Falta la variable ALPHAVANTAGE_API_KEY: no se consulta "
                "Alpha Vantage.",
            )
        query = dict(params)
        query["apikey"] = self._api_key
        request = self._client.build_request("GET", self._base_url, params=query)
        try:
            response = self._client.send(request, stream=True)
            try:
                body = self._read_bounded(response)
                status = response.status_code
            finally:
                response.close()
        except AlphaVantageError:
            raise
        except httpx.HTTPError:
            raise self._safe_error(
                _ERROR_NETWORK, "Error de red consultando Alpha Vantage."
            )
        if status != 200:
            raise self._safe_error(
                _ERROR_HTTP_STATUS, f"Alpha Vantage respondio HTTP {status}."
            )
        try:
            payload = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise self._safe_error(
                _ERROR_INVALID_RESPONSE,
                "Respuesta de Alpha Vantage no valida (JSON invalido).",
            )
        if not isinstance(payload, dict):
            raise self._safe_error(
                _ERROR_INVALID_RESPONSE,
                "Respuesta de Alpha Vantage no valida (raiz no es un objeto).",
            )
        note = payload.get("Note") or payload.get("Information")
        if isinstance(note, str) and note.strip():
            raise self._safe_error(
                _ERROR_RATE_LIMIT,
                "Limite de peticiones de Alpha Vantage alcanzado.",
            )
        error_message = payload.get("Error Message")
        if isinstance(error_message, str) and error_message.strip():
            raise self._safe_error(
                _ERROR_INVALID_SYMBOL,
                "Alpha Vantage rechazo la peticion (simbolo o parametros "
                "invalidos).",
            )
        return payload

    def _read_bounded(self, response: httpx.Response) -> bytes:
        """Lee la respuesta con limite estricto de tamaÃ±o."""
        chunks: list[bytes] = []
        total = 0
        try:
            for chunk in response.iter_bytes():
                total += len(chunk)
                if total > self._max_response_bytes:
                    raise self._safe_error(
                        _ERROR_RESPONSE_TOO_LARGE,
                        "La respuesta de Alpha Vantage excede el limite de "
                        "tamaÃ±o permitido.",
                    )
                chunks.append(chunk)
        except httpx.HTTPError:
            raise self._safe_error(
                _ERROR_NETWORK, "Error de red leyendo la respuesta de Alpha Vantage."
            )
        return b"".join(chunks)

    def _safe_error(self, code: str, message: str) -> AlphaVantageError:
        """Construye un error con cualquier secreto redactado."""
        if self._api_key and self._api_key in message:
            message = message.replace(self._api_key, _REDACTED)
        return AlphaVantageError(code, message)


def _required_text(item: Mapping[str, Any], key: str) -> str:
    value = item.get(key)
    if not isinstance(value, str):
        raise AlphaVantageError(
            _ERROR_INVALID_RESPONSE,
            f"Campo {key!r} ausente o invalido en Alpha Vantage.",
        )
    return value.strip()


def _optional_text(item: Mapping[str, Any], key: str) -> str:
    value = item.get(key)
    return value.strip() if isinstance(value, str) else ""


def _parse_daily_bar(day: str, values: Any) -> DailyBar:
    if not isinstance(day, str) or not isinstance(values, Mapping):
        raise AlphaVantageError(
            _ERROR_INVALID_RESPONSE,
            "Sesion diaria invalida en la respuesta de Alpha Vantage.",
        )
    parsed_day = _parse_day(day)
    return DailyBar(
        day=parsed_day,
        open=_parse_price(values, "1. open", day),
        high=_parse_price(values, "2. high", day),
        low=_parse_price(values, "3. low", day),
        close=_parse_price(values, "4. close", day),
        volume=_parse_volume(values, "5. volume", day),
    )


def _parse_day(raw: str) -> date:
    try:
        return date.fromisoformat(raw.strip())
    except ValueError:
        raise AlphaVantageError(
            _ERROR_INVALID_RESPONSE,
            "Fecha invalida en la serie diaria de Alpha Vantage.",
        )


def _parse_price(values: Mapping[str, Any], key: str, day: str) -> Decimal:
    value = values.get(key)
    if not isinstance(value, str):
        raise AlphaVantageError(
            _ERROR_INVALID_RESPONSE,
            f"Precio {key!r} invalido en la sesion {day} de Alpha Vantage.",
        )
    try:
        return Decimal(value.strip())
    except InvalidOperation:
        raise AlphaVantageError(
            _ERROR_INVALID_RESPONSE,
            f"Precio {key!r} no numerico en la sesion {day} de Alpha Vantage.",
        )


def _parse_volume(values: Mapping[str, Any], key: str, day: str) -> int:
    value = values.get(key)
    if not isinstance(value, str):
        raise AlphaVantageError(
            _ERROR_INVALID_RESPONSE,
            f"Volumen invalido en la sesion {day} de Alpha Vantage.",
        )
    try:
        return int(value.strip())
    except ValueError:
        raise AlphaVantageError(
            _ERROR_INVALID_RESPONSE,
            f"Volumen no numerico en la sesion {day} de Alpha Vantage.",
        )
