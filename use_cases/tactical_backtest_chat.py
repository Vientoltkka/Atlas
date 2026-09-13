"""Backtest tactico de Atlas Finance V2.9 (solo lectura, sin LLM).

Comando conversacional determinista y explicito: "backtest tactical
<simbolo>". Reutiliza exclusivamente el adaptador Alpha Vantage de V2.4
(una unica consulta de serie diaria bajo peticion explicita del usuario)
y el motor puro de finance.tactical.backtest: estrategia unica y fija
(long-only, SMA20/SMA50, senal al cierre de t, ejecucion en la apertura
de t+1, coste de 10 bps por entrada y por salida, capital inicial
virtual fijo de 10000).

Reglas: la respuesta lleva [BACKTEST], la fuente Alpha Vantage, el numero
y rango de sesiones, la regla exacta y los costes. Nunca usa lenguaje de
recomendacion o promesa. Si hay menos de 252 sesiones muestra la
advertencia "muestra exploratoria, insuficiente para validar una
estrategia". Si faltan barras, hay fechas/valores invalidos o falla el
proveedor, la respuesta es segura y no inventa metricas; no hay fallback
al LLM. Los parametros no son variables desde el chat y no se optimiza
sobre el mismo historico. Este bloque no modifica watchlist, PAPER,
MarketEvent, ordenes ni politicas.
"""

from __future__ import annotations

import re
from datetime import datetime
from decimal import Decimal

from finance.tactical.backtest import (
    BacktestResult,
    TacticalBacktestError,
    run_tactical_backtest,
)
from tools.alpha_vantage import AlphaVantageClient, AlphaVantageError

BACKTEST_LABEL = "[BACKTEST]"

_BACKTEST_PATTERN = re.compile(
    r"^\s*backtest\s+tactical\s+(?:de\s+|del\s+|para\s+|:)?\s*"
    r"(?P<symbol>[A-Za-z0-9.\-]{1,12})\s*[?.!]*\s*$",
    re.IGNORECASE,
)

_ERROR_TEXTS = {
    "MISSING_KEY": (
        "Alpha Vantage no esta configurado: falta la variable "
        "ALPHAVANTAGE_API_KEY. No se consultan datos ni se simula nada."
    ),
    "RATE_LIMIT": (
        "Limite de peticiones de Alpha Vantage alcanzado. No se consultan "
        "mas datos ahora mismo y no se simula nada."
    ),
    "INVALID_SYMBOL": (
        "Alpha Vantage no reconoce el simbolo o los parametros solicitados. "
        "No se simula nada."
    ),
    "INVALID_RESPONSE": (
        "La respuesta de Alpha Vantage no es valida. No se simula nada."
    ),
    "NETWORK": (
        "Error de red consultando Alpha Vantage. No se simula nada."
    ),
    "HTTP_STATUS": (
        "Alpha Vantage devolvio un error HTTP. No se simula nada."
    ),
    "RESPONSE_TOO_LARGE": (
        "La respuesta de Alpha Vantage excede el limite de tamaño. No se "
        "simula nada."
    ),
}

_UNKNOWN_COMMAND_TEXT = (
    f"{BACKTEST_LABEL} No he identificado el comando de backtest tactico. "
    f"Comando disponible: backtest tactical SIMBOLO.\n\n"
    f"Consulta de solo lectura; nada se ha modificado. {BACKTEST_LABEL}"
)

_RULE_TEXT = (
    "long-only; senal al cierre de la sesion t (SMA20(t) > SMA50(t) estado "
    "largo deseado; SMA20(t) < SMA50(t) estado plano deseado; igualdad "
    "mantiene el estado); entrada y salida en la apertura de la sesion t+1; "
    "una posicion como maximo; sin apalancamiento, cortos ni derivados; "
    "estrategia unica y fija, sin parametros variables."
)

_COST_TEXT = (
    "10 bps por entrada y 10 bps por salida (0,10 % del capital ejecutado "
    "en cada lado)."
)

_LIMITATIONS_TEXT = (
    "Resultado historico descriptivo: el comportamiento pasado no garantiza "
    "resultados futuros y esto no es una recomendacion de compra o venta ni "
    "una promesa de rentabilidad."
)

_READ_ONLY_TEXT = (
    "Simulacion de solo lectura: no modifica watchlist, PAPER, MarketEvent, "
    "ordenes ni politicas."
)


class TacticalBacktestChat:
    """Maneja turnos de backtest tactico sin LLM ni invencion de datos."""

    def __init__(self, client: AlphaVantageClient, now_provider=None) -> None:
        self._client = client
        self._now_provider = now_provider or (
            lambda: datetime.now().astimezone()
        )

    def handles(self, prompt: str) -> bool:
        """True solo ante el comando explicito "backtest tactical <simbolo>"."""
        return classify_tactical_backtest_prompt(prompt) is not None

    def handle(self, prompt: str) -> str:
        """Ejecuta un turno de backtest tactico y devuelve el texto visible."""
        symbol = classify_tactical_backtest_prompt(prompt)
        if symbol is None:
            raise ValueError(
                f"turno de backtest tactico no reconocido: {prompt!r}"
            )
        try:
            series = self._client.daily_series(symbol)
        except AlphaVantageError as error:
            return _error_text(error)
        try:
            result = run_tactical_backtest(series)
        except TacticalBacktestError as error:
            return _invalid_series_text(str(error))
        return _compose_backtest_report(
            result,
            now=self._now_provider(),
        )


def handles_tactical_backtest_prompt(prompt: str) -> bool:
    """Clasificador puro: True solo ante "backtest tactical <simbolo>"."""
    return classify_tactical_backtest_prompt(prompt) is not None


def classify_tactical_backtest_prompt(prompt: str) -> "str | None":
    """Devuelve el simbolo si el prompt es el comando de backtest; si no, None."""
    text = (prompt or "").strip()
    if "\n" in text:
        return None
    match = _BACKTEST_PATTERN.match(text)
    if match is None:
        return None
    symbol = match.group("symbol").strip().upper()
    return symbol or None


def _compose_backtest_report(result: BacktestResult, *, now: datetime) -> str:
    label = (
        "DATOS DIARIOS"
        if result.last_day >= now.date()
        else "RETRASADOS"
    )
    lines = [
        f"{BACKTEST_LABEL} Backtest tactico paper de {result.symbol} "
        "(fuente: Alpha Vantage, serie diaria):",
        "",
        f"- Simbolo: {result.symbol}",
        "- Fuente: Alpha Vantage (endpoint TIME_SERIES_DAILY)",
        f"- Sesiones: {result.total_bars} "
        f"(rango {result.first_day.isoformat()} – {result.last_day.isoformat()})",
        f"- Etiqueta: {label}; no es tiempo real.",
        f"- Regla exacta: {_RULE_TEXT}",
        f"- Costes: {_COST_TEXT}",
        f"- Capital inicial virtual fijo: {_fmt_money(result.initial_capital)}",
        f"- Capital final virtual: {_fmt_money(result.final_capital)}",
        "- Rentabilidad neta de la estrategia: "
        f"{_fmt_pct(result.strategy_return)}",
        (
            "- Rentabilidad buy-and-hold (compra en la apertura de la "
            "primera sesion con 10 bps de coste y mantenimiento hasta el "
            f"ultimo cierre): {_fmt_pct(result.buy_and_hold_return)}"
        ),
        f"- Maximo drawdown de la estrategia: {_fmt_pct_magnitude(result.max_drawdown)}",
        f"- Operaciones completadas (ida y vuelta): {len(result.trades)}",
    ]
    if result.trades:
        lines.append(
            f"- Ratio de acierto: {_fmt_pct_magnitude(result.win_ratio)} "
            f"(de {len(result.trades)} operaciones completadas)"
        )
        lines.append(
            f"- Duracion media: {_fmt_sessions(result.average_duration)}"
        )
    else:
        lines.append(
            "- Ratio de acierto y duracion media: no calculables (sin "
            "operaciones completadas)."
        )
    if result.open_position:
        lines.append(
            "- Posicion abierta al final del historico: si; el capital "
            "final esta marcado al ultimo cierre y el coste de salida de "
            "10 bps no se ha pagado todavia."
        )
    else:
        lines.append("- Posicion abierta al final del historico: no.")
    if not result.signals_possible:
        lines.append(
            "- Sin senales posibles: faltan 50 sesiones para la primera "
            "SMA50 completa; la estrategia se mantiene plana."
        )
    if result.exploratory_sample:
        lines.append(
            "- ADVERTENCIA: muestra exploratoria, insuficiente para "
            "validar una estrategia."
        )
    lines.append("")
    lines.append(_LIMITATIONS_TEXT)
    lines.append(_READ_ONLY_TEXT)
    lines.append(BACKTEST_LABEL)
    return "\n".join(lines)


def _invalid_series_text(detail: str) -> str:
    return (
        f"{BACKTEST_LABEL} La serie diaria de Alpha Vantage contiene "
        f"fechas o valores invalidos: {detail} No se simula nada y no se "
        f"inventa ninguna metrica. {BACKTEST_LABEL}"
    )


def _error_text(error: AlphaVantageError) -> str:
    message = _ERROR_TEXTS.get(
        error.code,
        "Error consultando Alpha Vantage. No se simula nada.",
    )
    return (
        f"{BACKTEST_LABEL} {message} Consulta de solo lectura; nada se "
        f"ha modificado. {BACKTEST_LABEL}"
    )


def _fmt_money(value) -> str:
    return f"{value.quantize(Decimal('0.01')):f}"


def _fmt_pct(fraction) -> str:
    percent = (fraction * 100).quantize(Decimal("0.01"))
    text = f"{percent:f}"
    if percent > 0:
        text = f"+{text}"
    return f"{text} %"


def _fmt_pct_magnitude(fraction) -> str:
    if fraction is None:
        return "no calculable"
    percent = (fraction * 100).quantize(Decimal("0.01"))
    return f"{percent:f} %"


def _fmt_sessions(value) -> str:
    if value is None:
        return "no calculable"
    return f"{value.quantize(Decimal('0.01')):f} sesiones"
