"""Calculos cuantitativos deterministas de Atlas Finance V2.5.

Modulo puro (solo stdlib, sin red, sin LLM, sin persistencia) que consume
la serie diaria del adaptador Alpha Vantage V2.4 y produce un snapshot
cuantitativo reproducible: variaciones 5/20 sesiones, SMA 20/50,
volatilidad historica anualizada de 20 sesiones y drawdown maximo de las
ultimas 60 sesiones.

Reglas: si faltan barras, la metrica correspondiente no se calcula y se
declara como no disponible; nunca se inventa un valor. Una serie invalida
(fechas duplicadas o fuera de orden, cierres no positivos) aborta el
analisis completo con un error controlado. Las observaciones son
descriptivas, nunca recomendaciones.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from typing import Protocol, Sequence


class DailySeriesLike(Protocol):
    """Estructura minima de una serie diaria (compatible con DailySeries de tools.alpha_vantage).

    Se declara aqui para mantener finance/ libre de red: solo se necesita
    el simbolo y las sesiones con dia y cierre; nunca la libreria HTTP.
    """

    symbol: str
    bars: Sequence["DailyBarLike"]


class DailyBarLike(Protocol):
    """Estructura minima de una sesion diaria (compatible con DailyBar)."""

    day: date
    close: Decimal

ANNUALIZATION_TRADING_DAYS = Decimal("252")
CHANGE_SHORT_WINDOW = 5
CHANGE_LONG_WINDOW = 20
SMA_SHORT_WINDOW = 20
SMA_LONG_WINDOW = 50
VOLATILITY_WINDOW = 20
DRAWDOWN_WINDOW = 60

TREND_UP = "ALZA"
TREND_DOWN = "BAJA"
TREND_SIDEWAYS = "LATERAL"
TREND_UNDETERMINED = "INDETERMINADA"

POSITION_ABOVE = "POR ENCIMA"
POSITION_BELOW = "POR DEBAJO"
POSITION_EQUAL = "EN LINEA"


class QuantSeriesError(ValueError):
    """Serie diaria invalida: el analisis cuantitativo no puede ejecutarse."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


@dataclass(frozen=True)
class QuantSnapshot:
    """Resultado determinista del analisis cuantitativo de una serie diaria."""

    symbol: str
    total_bars: int
    last_day: date
    last_close: Decimal
    change_5: Decimal | None
    change_20: Decimal | None
    sma_20: Decimal | None
    sma_50: Decimal | None
    volatility_20_annualized: Decimal | None
    max_drawdown_60: Decimal | None
    close_vs_sma_20: str | None
    close_vs_sma_50: str | None
    trend: str
    unavailable: tuple[str, ...]

    @property
    def complete(self) -> bool:
        return not self.unavailable


def validate_daily_series(series: DailySeriesLike) -> None:
    """Valida la serie antes de calcular: fechas estrictas y cierres positivos.

    Los fines de semana y festivos son huecos naturales del mercado; solo
    se rechazan fechas duplicadas o fuera de orden y cierres no positivos.
    """
    if not series.bars:
        raise QuantSeriesError(
            f"La serie diaria de {series.symbol} no contiene sesiones."
        )
    previous_day: date | None = None
    for bar in series.bars:
        if bar.close <= 0:
            raise QuantSeriesError(
                "Cierre no positivo en la sesion "
                f"{bar.day.isoformat()} de {series.symbol}."
            )
        if previous_day is not None and bar.day <= previous_day:
            raise QuantSeriesError(
                "Sesion duplicada o fuera de orden en "
                f"{bar.day.isoformat()} de {series.symbol}."
            )
        previous_day = bar.day


def pct_change(closes: Sequence[Decimal], window: int) -> Decimal | None:
    """Variacion porcentual entre el ultimo cierre y el de hace `window` sesiones."""
    if len(closes) < window + 1:
        return None
    base = closes[-(window + 1)]
    last = closes[-1]
    return (last - base) / base


def simple_moving_average(
    closes: Sequence[Decimal], window: int
) -> Decimal | None:
    """Media movil simple de las ultimas `window` sesiones."""
    if len(closes) < window:
        return None
    tail = closes[-window:]
    total = sum(tail, Decimal(0))
    return total / Decimal(window)


def annualized_volatility(
    closes: Sequence[Decimal], window: int = VOLATILITY_WINDOW
) -> Decimal | None:
    """Volatilidad historica anualizada de las ultimas `window` rentabilidades.

    Usa la desviacion tipica muestral (denominador n-1) de las
    rentabilidades diarias simples y el factor de anualizacion sqrt(252).
    """
    if len(closes) < window + 1:
        return None
    tail = closes[-(window + 1):]
    returns = [
        (current / previous) - 1
        for previous, current in zip(tail, tail[1:])
    ]
    return _annualized_volatility_of_returns(tuple(returns))


def _annualized_volatility_of_returns(
    returns: tuple[Decimal, ...],
) -> Decimal:
    count = len(returns)
    mean = sum(returns, Decimal(0)) / Decimal(count)
    variance = (
        sum(((value - mean) ** 2 for value in returns), Decimal(0))
        / Decimal(count - 1)
    )
    return variance.sqrt() * ANNUALIZATION_TRADING_DAYS.sqrt()


def max_drawdown(
    closes: Sequence[Decimal], window: int = DRAWDOWN_WINDOW
) -> Decimal | None:
    """Maximo descenso pico-valle (fraccion no negativa) de las ultimas `window` sesiones."""
    if len(closes) < window:
        return None
    tail = closes[-window:]
    peak = tail[0]
    worst = Decimal(0)
    for close in tail:
        if close > peak:
            peak = close
        drawdown = (peak - close) / peak
        if drawdown > worst:
            worst = drawdown
    return worst


def position_of_close(
    last_close: Decimal, sma: Decimal | None
) -> str | None:
    """Posicion del ultimo cierre frente a una media movil."""
    if sma is None:
        return None
    if last_close > sma:
        return POSITION_ABOVE
    if last_close < sma:
        return POSITION_BELOW
    return POSITION_EQUAL


def classify_trend(
    last_close: Decimal,
    sma_20: Decimal | None,
    sma_50: Decimal | None,
    change_20: Decimal | None,
) -> str:
    """Tendencia descriptiva segun reglas explicitas (no recomendacion).

    - ALZA: SMA20 > SMA50, variacion 20 sesiones > 0 y ultimo cierre > SMA20.
    - BAJA: SMA20 < SMA50, variacion 20 sesiones < 0 y ultimo cierre < SMA20.
    - LATERAL: si no se cumplen las condiciones de alza ni de baja.
    - INDETERMINADA: si falta alguna entrada de la regla.
    """
    if sma_20 is None or sma_50 is None or change_20 is None:
        return TREND_UNDETERMINED
    up = sma_20 > sma_50 and change_20 > 0 and last_close > sma_20
    down = sma_20 < sma_50 and change_20 < 0 and last_close < sma_20
    if up:
        return TREND_UP
    if down:
        return TREND_DOWN
    return TREND_SIDEWAYS


def build_quant_snapshot(series: DailySeriesLike) -> QuantSnapshot:
    """Calcula el snapshot cuantitativo completo de una serie diaria validada."""
    validate_daily_series(series)
    closes = [bar.close for bar in series.bars]
    last_close = closes[-1]
    last_day = series.bars[-1].day
    sma_20 = simple_moving_average(closes, SMA_SHORT_WINDOW)
    sma_50 = simple_moving_average(closes, SMA_LONG_WINDOW)
    change_5 = pct_change(closes, CHANGE_SHORT_WINDOW)
    change_20 = pct_change(closes, CHANGE_LONG_WINDOW)
    volatility = annualized_volatility(closes, VOLATILITY_WINDOW)
    drawdown = max_drawdown(closes, DRAWDOWN_WINDOW)
    unavailable: list[str] = []
    if change_5 is None:
        unavailable.append("variacion 5 sesiones")
    if change_20 is None:
        unavailable.append("variacion 20 sesiones")
    if sma_20 is None:
        unavailable.append("SMA20")
    if sma_50 is None:
        unavailable.append("SMA50")
    if volatility is None:
        unavailable.append(
            "volatilidad historica anualizada de 20 sesiones"
        )
    if drawdown is None:
        unavailable.append("maximo drawdown de las ultimas 60 sesiones")
    return QuantSnapshot(
        symbol=series.symbol,
        total_bars=len(closes),
        last_day=last_day,
        last_close=last_close,
        change_5=change_5,
        change_20=change_20,
        sma_20=sma_20,
        sma_50=sma_50,
        volatility_20_annualized=volatility,
        max_drawdown_60=drawdown,
        close_vs_sma_20=position_of_close(last_close, sma_20),
        close_vs_sma_50=position_of_close(last_close, sma_50),
        trend=classify_trend(last_close, sma_20, sma_50, change_20),
        unavailable=tuple(unavailable),
    )
