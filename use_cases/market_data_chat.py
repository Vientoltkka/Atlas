"""Chat de datos de mercado de Atlas Finance V2.4 (solo lectura).

Comandos conversacionales deterministas sobre el adaptador aislado de
Alpha Vantage: "datos mercado <simbolo>" (serie diaria OHLCV o ultimo
cierre disponible) y "buscar activo <texto>" (busqueda de simbolo).
Solo se consulta bajo peticion explicita del usuario; sin scheduler,
sin polling, sin alertas y sin llamadas automaticas.

Reglas: la respuesta lleva [MARKET DATA], la fuente Alpha Vantage y el
timestamp del proveedor si existe, con etiqueta clara DATOS DIARIOS o
RETRASADOS; nunca "tiempo real" salvo que el proveedor lo confirme
explicitamente. Los errores son breves y seguros, sin fallback al LLM
ni invencion de precios. Este bloque no registra MarketEvent, no
modifica la cartera paper, no crea propuestas ni ordenes y no conecta
ningun broker.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone

from tools.alpha_vantage import (
    AlphaVantageClient,
    AlphaVantageError,
    DailySeries,
)

MARKET_DATA_LABEL = "[MARKET DATA]"

_MAX_SEARCH_RESULTS = 5
_DAILY_BARS_SHOWN = 5

_DAILY_PATTERN = re.compile(
    r"^\s*datos\s+mercado\s+(?:de\s+|del\s+|para\s+|:)?\s*"
    r"(?P<symbol>[A-Za-z0-9.\-]{1,12})\s*[?.!]*\s*$",
    re.IGNORECASE,
)
_SEARCH_PATTERN = re.compile(
    r"^\s*buscar\s+activo\s+(?:con\s+nombre\s+|llamado\s+|de\s+|sobre\s+)?"
    r"(?P<query>.+?)\s*[?.!]*\s*$",
    re.IGNORECASE,
)

_ERROR_TEXTS = {
    "MISSING_KEY": (
        "Alpha Vantage no esta configurado: falta la variable "
        "ALPHAVANTAGE_API_KEY. No se consultan datos ni se inventan precios."
    ),
    "RATE_LIMIT": (
        "Limite de peticiones de Alpha Vantage alcanzado. No se consultan "
        "mas datos ahora mismo y no se inventan precios."
    ),
    "INVALID_SYMBOL": (
        "Alpha Vantage no reconoce el simbolo o los parametros solicitados. "
        "No se inventan precios."
    ),
    "INVALID_RESPONSE": (
        "La respuesta de Alpha Vantage no es valida. No se inventan precios."
    ),
    "NETWORK": (
        "Error de red consultando Alpha Vantage. No se inventan precios."
    ),
    "HTTP_STATUS": (
        "Alpha Vantage devolvio un error HTTP. No se inventan precios."
    ),
    "RESPONSE_TOO_LARGE": (
        "La respuesta de Alpha Vantage excede el limite de tamaño. No se "
        "inventan precios."
    ),
}

_UNKNOWN_COMMAND_TEXT = (
    f"{MARKET_DATA_LABEL} No he identificado el comando de datos de mercado. "
    f"Comandos disponibles (uno por mensaje):\n"
    f"- datos mercado SIMBOLO\n"
    f"- buscar activo TEXTO\n\n"
    f"Consulta de solo lectura; nada se ha modificado. {MARKET_DATA_LABEL}"
)


class MarketDataChat:
    """Maneja los turnos de datos de mercado sin LLM ni fallback inventado."""

    def __init__(self, client: AlphaVantageClient, now_provider=None) -> None:
        self._client = client
        self._now_provider = now_provider or (
            lambda: datetime.now(timezone.utc)
        )

    def handles(self, prompt: str) -> bool:
        """True solo ante un comando explicito de datos de mercado."""
        return _classify(prompt) is not None

    def handle(self, prompt: str) -> str:
        """Ejecuta un turno de datos de mercado y devuelve el texto visible."""
        classified = _classify(prompt)
        if classified is None:
            raise ValueError(f"turno de datos de mercado no reconocido: {prompt!r}")
        kind, argument = classified
        if kind == "daily":
            return self._daily_text(argument)
        return self._search_text(argument)

    def _daily_text(self, symbol: str) -> str:
        try:
            series = self._client.daily_series(symbol)
        except AlphaVantageError as error:
            return _error_text(error)
        return _compose_daily_report(series, now=self._now_provider())

    def _search_text(self, query: str) -> str:
        try:
            matches = self._client.search_symbol(query)
        except AlphaVantageError as error:
            return _error_text(error)
        if not matches:
            return _error_text(
                AlphaVantageError(
                    "INVALID_SYMBOL",
                    f"Sin resultados de busqueda para {query!r}.",
                )
            )
        lines = [
            f"{MARKET_DATA_LABEL} Busqueda de activos para \"{query}\" "
            "(fuente: Alpha Vantage):",
            "",
        ]
        for index, match in enumerate(matches[:_MAX_SEARCH_RESULTS], start=1):
            lines.append(
                f"{index}. {match.symbol} — {match.name} · "
                f"region: {match.region} · divisa: {match.currency} · "
                f"tipo: {match.asset_type}"
            )
        lines.append("")
        lines.append(
            "Solo metadatos de busqueda; sin cotizaciones. Consulta de solo "
            f"lectura; nada se ha modificado. {MARKET_DATA_LABEL}"
        )
        return "\n".join(lines)


def handles_market_data_prompt(prompt: str) -> bool:
    """Clasificador puro: True solo ante peticion explicita de datos."""
    return _classify(prompt) is not None


def _classify(prompt: str) -> "tuple[str, str] | None":
    text = prompt.strip()
    if "\n" in text:
        return None
    daily = _DAILY_PATTERN.match(text)
    if daily is not None:
        return "daily", daily.group("symbol")
    search = _SEARCH_PATTERN.match(text)
    if search is not None:
        query = search.group("query").strip()
        if query:
            return "search", query
    return None


def _compose_daily_report(series: DailySeries, *, now: datetime) -> str:
    if not series.bars:
        return _error_text(
            AlphaVantageError(
                "INVALID_RESPONSE",
                f"La serie diaria de {series.symbol} no contiene sesiones.",
            )
        )
    last = series.bars[-1]
    label = "DATOS DIARIOS" if last.day >= now.date() else "RETRASADOS"
    shown = series.bars[-_DAILY_BARS_SHOWN:]
    lines = [
        f"{MARKET_DATA_LABEL} Datos de mercado de {series.symbol} "
        "(fuente: Alpha Vantage):",
        "",
        f"- Ultimo cierre disponible: {_fmt_decimal(last.close)} "
        f"({last.day.isoformat()})",
        f"- Serie mostrada: ultimas {len(shown)} sesiones diarias",
    ]
    for bar in shown:
        lines.append(
            f"- {bar.day.isoformat()}: apertura {_fmt_decimal(bar.open)} · "
            f"maximo {_fmt_decimal(bar.high)} · minimo {_fmt_decimal(bar.low)} · "
            f"cierre {_fmt_decimal(bar.close)} · volumen {bar.volume}"
        )
    if series.provider_last_refreshed:
        refreshed = series.provider_last_refreshed
        if series.provider_timezone:
            refreshed = f"{refreshed} ({series.provider_timezone})"
        lines.append(f"- Timestamp del proveedor: {refreshed}")
    lines.append(f"- Etiqueta: {label}; no es tiempo real.")
    lines.append("")
    lines.append(
        "Consulta de solo lectura: no registra MarketEvent, no modifica la "
        f"cartera paper ni crea ordenes. {MARKET_DATA_LABEL}"
    )
    return "\n".join(lines)


def _error_text(error: AlphaVantageError) -> str:
    message = _ERROR_TEXTS.get(
        error.code,
        "Error consultando Alpha Vantage. No se inventan precios.",
    )
    return (
        f"{MARKET_DATA_LABEL} {message} Consulta de solo lectura; nada se "
        f"ha modificado. {MARKET_DATA_LABEL}"
    )


def _fmt_decimal(value) -> str:
    return f"{value.normalize():f}"
