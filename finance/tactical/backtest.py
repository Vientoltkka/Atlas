"""Backtest tactico determinista de Atlas Finance V2.9.

Motor puro (solo stdlib, sin red, sin LLM, sin persistencia) que consume la
serie diaria del adaptador Alpha Vantage V2.4 y simula una unica estrategia
fija e invariable:

- Long-only, una posicion como maximo; sin apalancamiento, cortos ni derivados.
- Senal generada al cierre de la sesion t: SMA20(t) > SMA50(t) indica estado
  largo deseado; SMA20(t) < SMA50(t) indica estado plano deseado; igualdad
  mantiene el estado anterior.
- La entrada y la salida se ejecutan en la apertura de la sesion t+1; la
  decision en t nunca usa datos de t ni de sesiones posteriores (sin
  look-ahead).
- Coste explicito de 10 bps (0,10 %) por entrada y 10 bps por salida,
  aplicado al capital ejecutado.
- Capital inicial virtual fijo e invariable: 10000. No admite parametros
  desde el chat ni optimizacion sobre el mismo historico.

Reglas de seguridad: si faltan barras o hay fechas/valores invalidos se
aborta con un error controlado y no se inventa ninguna metrica. El
resultado es historico y descriptivo; nunca es una recomendacion ni una
promesa de resultados. No modifica watchlist, PAPER, MarketEvent, ordenes
ni politicas.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from typing import Sequence

from finance.quant.metrics import QuantSeriesError, validate_daily_series

INITIAL_VIRTUAL_CAPITAL = Decimal("10000")
COST_RATE = Decimal("0.001")
COST_BPS = Decimal("10")
SMA_SHORT_WINDOW = 20
SMA_LONG_WINDOW = 50
EXPLORATORY_MIN_SESSIONS = 252


class TacticalBacktestError(ValueError):
    """Serie diaria invalida: el backtest no puede ejecutarse."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class DailySeriesLike:
    """Estructura minima esperada (compatible con DailySeries de tools.alpha_vantage)."""

    symbol: str
    bars: Sequence["DailyBarLike"]


class DailyBarLike:
    """Estructura minima esperada (compatible con DailyBar de tools.alpha_vantage)."""

    day: date
    open: Decimal
    close: Decimal


@dataclass(frozen=True)
class BacktestTrade:
    """Una operacion completa (entrada y salida) del backtest tactico."""

    entry_day: date
    entry_open: Decimal
    exit_day: date
    exit_open: Decimal
    shares: Decimal
    capital_before: Decimal
    capital_after: Decimal
    sessions_held: int
    win: bool


@dataclass(frozen=True)
class BacktestResult:
    """Resultado determinista del backtest tactico sobre una serie diaria."""

    symbol: str
    total_bars: int
    first_day: date
    last_day: date
    initial_capital: Decimal
    final_capital: Decimal
    strategy_return: Decimal
    buy_and_hold_return: Decimal
    max_drawdown: Decimal
    trades: tuple[BacktestTrade, ...]
    open_position: bool
    open_position_value: Decimal | None
    win_ratio: Decimal | None
    average_duration: Decimal | None
    exploratory_sample: bool
    signals_possible: bool


def validate_backtest_series(series: DailySeriesLike) -> None:
    """Valida la serie antes de simular: fechas estrictas y precios positivos.

    Reutiliza la validacion de cierres y fechas de finance.quant.metrics y
    anade la validacion de aperturas, que el backtest usa para ejecutar.
    """
    try:
        validate_daily_series(series)
    except QuantSeriesError as error:
        raise TacticalBacktestError(str(error)) from error
    for bar in series.bars:
        if bar.open <= 0:
            raise TacticalBacktestError(
                "Apertura no positiva en la sesion "
                f"{bar.day.isoformat()} de {series.symbol}."
            )


def _sma_at(closes: Sequence[Decimal], index: int, window: int) -> Decimal:
    tail = closes[index - window + 1 : index + 1]
    return sum(tail, Decimal(0)) / Decimal(window)


def _max_drawdown_of_curve(equity_curve: Sequence[Decimal]) -> Decimal:
    peak = equity_curve[0]
    worst = Decimal(0)
    for equity in equity_curve:
        if equity > peak:
            peak = equity
        drawdown = (peak - equity) / peak
        if drawdown > worst:
            worst = drawdown
    return worst


def run_tactical_backtest(series: DailySeriesLike) -> BacktestResult:
    """Ejecuta el backtest tactico completo sobre una serie diaria validada."""
    validate_backtest_series(series)
    closes = [bar.close for bar in series.bars]
    opens = [bar.open for bar in series.bars]
    days = [bar.day for bar in series.bars]
    total = len(closes)

    cash = INITIAL_VIRTUAL_CAPITAL
    shares = Decimal(0)
    entry_index: int | None = None
    capital_before_entry: Decimal | None = None
    trades: list[BacktestTrade] = []
    equity_curve: list[Decimal] = []

    for i in range(total):
        if i > 0:
            signal_index = i - 1
            desired_long: bool | None = None
            if signal_index >= SMA_LONG_WINDOW - 1:
                sma_short = _sma_at(closes, signal_index, SMA_SHORT_WINDOW)
                sma_long = _sma_at(closes, signal_index, SMA_LONG_WINDOW)
                if sma_short > sma_long:
                    desired_long = True
                elif sma_short < sma_long:
                    desired_long = False
            if desired_long is True and shares == 0:
                capital_before_entry = cash
                invested = cash * (Decimal(1) - COST_RATE)
                shares = invested / opens[i]
                cash = Decimal(0)
                entry_index = i
            elif desired_long is False and shares > 0:
                cash = shares * opens[i] * (Decimal(1) - COST_RATE)
                trades.append(
                    BacktestTrade(
                        entry_day=days[entry_index],
                        entry_open=opens[entry_index],
                        exit_day=days[i],
                        exit_open=opens[i],
                        shares=shares,
                        capital_before=capital_before_entry,
                        capital_after=cash,
                        sessions_held=i - entry_index,
                        win=cash > capital_before_entry,
                    )
                )
                shares = Decimal(0)
                entry_index = None
                capital_before_entry = None
        equity_curve.append(cash + shares * closes[i])

    final_capital = equity_curve[-1]
    open_position = shares > 0
    open_position_value = shares * closes[-1] if open_position else None
    strategy_return = final_capital / INITIAL_VIRTUAL_CAPITAL - Decimal(1)
    bh_shares = (INITIAL_VIRTUAL_CAPITAL * (Decimal(1) - COST_RATE)) / opens[0]
    buy_and_hold_return = (
        bh_shares * closes[-1] / INITIAL_VIRTUAL_CAPITAL - Decimal(1)
    )
    wins = sum(1 for trade in trades if trade.win)
    win_ratio = (
        Decimal(wins) / Decimal(len(trades)) if trades else None
    )
    average_duration = (
        sum(
            (Decimal(trade.sessions_held) for trade in trades),
            Decimal(0),
        )
        / Decimal(len(trades))
        if trades
        else None
    )
    return BacktestResult(
        symbol=series.symbol,
        total_bars=total,
        first_day=days[0],
        last_day=days[-1],
        initial_capital=INITIAL_VIRTUAL_CAPITAL,
        final_capital=final_capital,
        strategy_return=strategy_return,
        buy_and_hold_return=buy_and_hold_return,
        max_drawdown=_max_drawdown_of_curve(equity_curve),
        trades=tuple(trades),
        open_position=open_position,
        open_position_value=open_position_value,
        win_ratio=win_ratio,
        average_duration=average_duration,
        exploratory_sample=total < EXPLORATORY_MIN_SESSIONS,
        signals_possible=total >= SMA_LONG_WINDOW + 1,
    )
