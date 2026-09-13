"""Modulo tactico de Atlas Finance V2.9: backtest historico determinista.

Solo calculo puro sobre datos diarios; sin red, sin LLM y sin persistencia.
"""

from finance.tactical.backtest import (
    BacktestResult,
    BacktestTrade,
    TacticalBacktestError,
    run_tactical_backtest,
)

__all__ = [
    "BacktestResult",
    "BacktestTrade",
    "TacticalBacktestError",
    "run_tactical_backtest",
]
