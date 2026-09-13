"""Analisis de mercado de Atlas Finance V2.5 (solo lectura, sin LLM).

Comando conversacional determinista: "analiza mercado <simbolo>". Reutiliza
exclusivamente el adaptador Alpha Vantage de V2.4 (una unica consulta de
serie diaria bajo peticion explicita del usuario) y los calculos puros de
finance.quant.metrics: variaciones 5/20 sesiones, SMA 20/50, volatilidad
historica anualizada de 20 sesiones y drawdown maximo de las ultimas 60
sesiones.

Reglas: la respuesta lleva [MARKET ANALYSIS], la fuente Alpha Vantage, la
fecha del ultimo dato y etiqueta DATOS DIARIOS o RETRASADOS. Las
observaciones son descriptivas segun reglas explicitas, nunca
recomendaciones. Si faltan barras, el dato esta demasiado antiguo, hay
huecos invalidos o falla el proveedor, se explica que metrica no puede
calcularse; no se inventa nada y no hay fallback al LLM. Este bloque no
registra MarketEvent, no modifica la cartera paper, no crea propuestas ni
ordenes y no conecta ningun broker.
"""

from __future__ import annotations

import re
from datetime import datetime
from decimal import Decimal

from finance.quant.metrics import (
    TREND_DOWN,
    TREND_UNDETERMINED,
    TREND_UP,
    QuantSeriesError,
    QuantSnapshot,
    build_quant_snapshot,
)
from tools.alpha_vantage import AlphaVantageClient, AlphaVantageError, DailySeries

MARKET_ANALYSIS_LABEL = "[MARKET ANALYSIS]"

_ANALYSIS_PATTERN = re.compile(
    r"^\s*analiza\s+(?:el\s+)?mercado\s+(?:de\s+|del\s+|para\s+|:)?\s*"
    r"(?P<symbol>[A-Za-z0-9.\-]{1,12})\s*[?.!]*\s*$",
    re.IGNORECASE,
)

_ERROR_TEXTS = {
    "MISSING_KEY": (
        "Alpha Vantage no esta configurado: falta la variable "
        "ALPHAVANTAGE_API_KEY. No se consultan datos ni se calcula nada."
    ),
    "RATE_LIMIT": (
        "Limite de peticiones de Alpha Vantage alcanzado. No se consultan "
        "mas datos ahora mismo y no se calcula nada."
    ),
    "INVALID_SYMBOL": (
        "Alpha Vantage no reconoce el simbolo o los parametros solicitados. "
        "No se calcula nada."
    ),
    "INVALID_RESPONSE": (
        "La respuesta de Alpha Vantage no es valida. No se calcula nada."
    ),
    "NETWORK": (
        "Error de red consultando Alpha Vantage. No se calcula nada."
    ),
    "HTTP_STATUS": (
        "Alpha Vantage devolvio un error HTTP. No se calcula nada."
    ),
    "RESPONSE_TOO_LARGE": (
        "La respuesta de Alpha Vantage excede el limite de tamaño. No se "
        "calcula nada."
    ),
}

_UNKNOWN_COMMAND_TEXT = (
    f"{MARKET_ANALYSIS_LABEL} No he identificado el comando de analisis de "
    f"mercado. Comando disponible: analiza mercado SIMBOLO.\n\n"
    f"Consulta de solo lectura; nada se ha modificado. {MARKET_ANALYSIS_LABEL}"
)

_LIMITATIONS_TEXT = (
    "Limitaciones: datos diarios de cierre, no tiempo real; la volatilidad "
    "y las medias se calculan solo sobre el historial disponible; el "
    "comportamiento pasado no garantiza resultados futuros; esto no es una "
    "recomendacion de compra o venta."
)

_TREND_RULE_TEXTS = {
    TREND_UP: (
        "SMA20 > SMA50, variacion de 20 sesiones > 0 y ultimo cierre "
        "> SMA20."
    ),
    TREND_DOWN: (
        "SMA20 < SMA50, variacion de 20 sesiones < 0 y ultimo cierre "
        "< SMA20."
    ),
    "LATERAL": (
        "no se cumplen simultaneamente las condiciones de alza ni de baja."
    ),
    TREND_UNDETERMINED: (
        "faltan SMA20, SMA50 o la variacion de 20 sesiones para aplicar la "
        "regla."
    ),
}


class MarketAnalysisChat:
    """Maneja turnos de analisis de mercado sin LLM ni invencion de datos."""

    def __init__(self, client: AlphaVantageClient, now_provider=None) -> None:
        self._client = client
        self._now_provider = now_provider or (
            lambda: datetime.now().astimezone()
        )

    def handles(self, prompt: str) -> bool:
        """True solo ante el comando explicito "analiza mercado <simbolo>"."""
        return classify_market_analysis_prompt(prompt) is not None

    def handle(self, prompt: str) -> str:
        """Ejecuta un turno de analisis de mercado y devuelve el texto visible."""
        symbol = classify_market_analysis_prompt(prompt)
        if symbol is None:
            raise ValueError(
                f"turno de analisis de mercado no reconocido: {prompt!r}"
            )
        try:
            series = self._client.daily_series(symbol)
        except AlphaVantageError as error:
            return _error_text(error)
        try:
            snapshot = build_quant_snapshot(series)
        except QuantSeriesError as error:
            return _invalid_series_text(str(error))
        return _compose_analysis_report(
            series,
            snapshot,
            now=self._now_provider(),
        )


def handles_market_analysis_prompt(prompt: str) -> bool:
    """Clasificador puro: True solo ante "analiza mercado <simbolo>"."""
    return classify_market_analysis_prompt(prompt) is not None


def classify_market_analysis_prompt(prompt: str) -> "str | None":
    """Devuelve el simbolo si el prompt es el comando de analisis; si no, None."""
    text = (prompt or "").strip()
    if "\n" in text:
        return None
    match = _ANALYSIS_PATTERN.match(text)
    if match is None:
        return None
    symbol = match.group("symbol").strip().upper()
    return symbol or None


def _compose_analysis_report(
    series: DailySeries,
    snapshot: QuantSnapshot,
    *,
    now: datetime,
) -> str:
    label = "DATOS DIARIOS" if snapshot.last_day >= now.date() else "RETRASADOS"
    lines = [
        f"{MARKET_ANALYSIS_LABEL} Analisis cuantitativo de {snapshot.symbol} "
        "(fuente: Alpha Vantage, serie diaria):",
        "",
        f"- Simbolo: {snapshot.symbol}",
        f"- Fuente: Alpha Vantage (endpoint TIME_SERIES_DAILY)",
        f"- Fecha del ultimo dato: {snapshot.last_day.isoformat()}",
        f"- Etiqueta: {label}; no es tiempo real.",
        f"- Sesiones usadas: {snapshot.total_bars}",
        "",
    ]
    lines.extend(_metric_lines(snapshot))
    lines.append("")
    lines.append(
        f"Tendencia (regla explicita): {snapshot.trend} — "
        f"{_TREND_RULE_TEXTS.get(snapshot.trend, 'regla no definida.')}"
    )
    lines.append("")
    lines.append("Observaciones: descriptivas, no recomendaciones.")
    if snapshot.unavailable:
        lines.append(
            "Metricas no calculables por datos insuficientes: "
            + "; ".join(snapshot.unavailable)
            + "."
        )
    if label == "RETRASADOS":
        lines.append(
            "El ultimo dato esta retrasado respecto a hoy: las metricas "
            "reflejan el historial disponible y pueden estar desactualizadas."
        )
    lines.append(_LIMITATIONS_TEXT)
    lines.append("")
    lines.append(
        "Consulta de solo lectura: no registra MarketEvent, no modifica la "
        f"cartera paper ni crea ordenes. {MARKET_ANALYSIS_LABEL}"
    )
    return "\n".join(lines)


def _metric_lines(snapshot: QuantSnapshot) -> list[str]:
    lines = ["Metricas y periodo usados:"]
    if snapshot.change_5 is not None:
        lines.append(
            f"- Variacion 5 sesiones: {_fmt_pct(snapshot.change_5)}"
        )
    else:
        lines.append("- Variacion 5 sesiones: no calculable (faltan 6 sesiones).")
    if snapshot.change_20 is not None:
        lines.append(
            f"- Variacion 20 sesiones: {_fmt_pct(snapshot.change_20)}"
        )
    else:
        lines.append(
            "- Variacion 20 sesiones: no calculable (faltan 21 sesiones)."
        )
    if snapshot.sma_20 is not None:
        lines.append(
            f"- SMA20 (20 sesiones): {_fmt_decimal(snapshot.sma_20)}; "
            f"ultimo cierre {snapshot.close_vs_sma_20}."
        )
    else:
        lines.append("- SMA20: no calculable (faltan 20 sesiones).")
    if snapshot.sma_50 is not None:
        lines.append(
            f"- SMA50 (50 sesiones): {_fmt_decimal(snapshot.sma_50)}; "
            f"ultimo cierre {snapshot.close_vs_sma_50}."
        )
    else:
        lines.append("- SMA50: no calculable (faltan 50 sesiones).")
    if snapshot.volatility_20_annualized is not None:
        lines.append(
            "- Volatilidad historica anualizada (20 rentabilidades diarias, "
            "factor sqrt(252)): "
            f"{_fmt_pct_magnitude(snapshot.volatility_20_annualized)}"
        )
    else:
        lines.append(
            "- Volatilidad historica anualizada de 20 sesiones: no calculable "
            "(faltan 21 sesiones)."
        )
    if snapshot.max_drawdown_60 is not None:
        lines.append(
            "- Maximo drawdown pico-valle (ultimas 60 sesiones): "
            f"{_fmt_pct_magnitude(snapshot.max_drawdown_60)}"
        )
    else:
        lines.append(
            "- Maximo drawdown de las ultimas 60 sesiones: no calculable "
            "(faltan 60 sesiones)."
        )
    return lines


def _invalid_series_text(detail: str) -> str:
    return (
        f"{MARKET_ANALYSIS_LABEL} La serie diaria de Alpha Vantage contiene "
        f"huecos o valores invalidos: {detail} Ninguna metrica se calcula; "
        f"no se inventa nada. {MARKET_ANALYSIS_LABEL}"
    )


def _error_text(error: AlphaVantageError) -> str:
    message = _ERROR_TEXTS.get(
        error.code,
        "Error consultando Alpha Vantage. No se calcula nada.",
    )
    return (
        f"{MARKET_ANALYSIS_LABEL} {message} Consulta de solo lectura; nada "
        f"se ha modificado. {MARKET_ANALYSIS_LABEL}"
    )


def _fmt_decimal(value) -> str:
    return f"{value.normalize():f}"


def _fmt_pct(fraction) -> str:
    """Porcentaje con signo explicito (+/-) para variaciones."""
    percent = (fraction * 100).quantize(Decimal("0.01"))
    text = f"{percent:f}"
    if percent > 0:
        text = f"+{text}"
    return f"{text} %"


def _fmt_pct_magnitude(fraction) -> str:
    """Porcentaje sin signo para magnitudes (volatilidad, drawdown)."""
    percent = (fraction * 100).quantize(Decimal("0.01"))
    return f"{percent:f} %"

