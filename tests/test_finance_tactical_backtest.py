"""Tests de Atlas Finance V2.9: backtest tactico paper.

Cubren el calculo conocido con costes (Decimal exacto), la ausencia de
look-ahead (senal al cierre de t, ejecucion en la apertura de t+1, sin
uso de datos futuros), el cruce de entrada/salida, el caso sin
operaciones, drawdown y benchmark conocidos, datos insuficientes e
invalidos, errores del proveedor, el aislamiento total de PAPER y
watchlist, y la ruta real Atlas.process_prompt sin invocar el LLM.
Las respuestas HTTP son falsas (httpx.MockTransport); nunca hay red,
API key real, ordenes, MarketEvents ni fills.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path

import httpx
import pytest

from core.atlas import Atlas
from core.model_inference import ModelInferenceRunner
from finance.paper.service import PaperFinanceService
from finance.paper.store import PaperStore
from finance.tactical.backtest import (
    INITIAL_VIRTUAL_CAPITAL,
    BacktestResult,
    TacticalBacktestError,
    run_tactical_backtest,
)
from tools.alpha_vantage import AlphaVantageClient, DailyBar, DailySeries
from use_cases.tactical_backtest_chat import (
    BACKTEST_LABEL,
    TacticalBacktestChat,
    classify_tactical_backtest_prompt,
)

FAKE_KEY = "test-fake-key-not-real"
REVIEW_DAY = "2026-09-11"


def _bar(day: str, open_: str, close: str) -> DailyBar:
    open_value = Decimal(open_)
    close_value = Decimal(close)
    return DailyBar(
        day=date.fromisoformat(day),
        open=open_value,
        high=max(open_value, close_value),
        low=min(open_value, close_value),
        close=close_value,
        volume=1000,
    )


def _next_session(day: date) -> date:
    next_day = day
    while True:
        next_day = date.fromordinal(next_day.toordinal() + 1)
        if next_day.weekday() < 5:
            return next_day


def _series(
    opens: list[str],
    closes: list[str],
    *,
    symbol: str = "TEST",
) -> DailySeries:
    """Serie diaria sintetica de sesiones consecutivas desde 2026-06-01."""
    assert len(opens) == len(closes)
    bars = []
    day = date(2026, 6, 1)
    for open_, close in zip(opens, closes):
        bars.append(_bar(day.isoformat(), open_, close))
        day = _next_session(day)
    return DailySeries(
        symbol=symbol,
        provider_last_refreshed=bars[-1].day.isoformat() if bars else "",
        provider_timezone="US/Eastern",
        bars=tuple(bars),
    )


def _flat(value: str, count: int) -> list[str]:
    return [value] * count


def _series_from_closes(
    closes: list[str],
    *,
    symbol: str = "TEST",
) -> DailySeries:
    """Serie sin huecos: la apertura de cada sesion es el cierre anterior."""
    opens = [closes[0]] + closes[:-1]
    return _series(opens, closes, symbol=symbol)


def _gapless_roundtrip_200() -> DailySeries:
    """200 sesiones: 100 x50, 200 x50, 100 x100 (un cruce arriba y abajo)."""
    closes = (
        _flat("100", 50) + _flat("200", 50) + _flat("100", 100)
    )
    return _series_from_closes(closes)


def _series_winning_trade_220() -> DailySeries:
    """220 sesiones: entrada en 102 y salida en 180 (operacion ganadora)."""
    ramp = [str(100 + 2 * k) for k in range(50)]
    closes = (
        _flat("100", 50) + ramp + _flat("200", 100) + _flat("180", 20)
    )
    return _series_from_closes(closes)


def _payload_from_series(series: DailySeries) -> dict:
    time_series = {}
    for bar in series.bars:
        time_series[bar.day.isoformat()] = {
            "1. open": str(bar.open),
            "2. high": str(bar.high),
            "3. low": str(bar.low),
            "4. close": str(bar.close),
            "5. volume": str(bar.volume),
        }
    return {
        "Meta Data": {
            "1. Information": "Daily Prices",
            "2. Symbol": series.symbol,
            "3. Last Refreshed": series.provider_last_refreshed,
            "4. Time Zone": series.provider_timezone,
        },
        "Time Series (Daily)": time_series,
    }


def _now(day: str = REVIEW_DAY):
    fixed = datetime.fromisoformat(f"{day}T12:00:00+00:00")
    return lambda: fixed


def _client(handler) -> AlphaVantageClient:
    return AlphaVantageClient(
        FAKE_KEY,
        transport=httpx.MockTransport(handler),
    )


def _json_handler(payload):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=json.dumps(payload))

    return handler


def _client_for_series(series: DailySeries) -> AlphaVantageClient:
    return _client(_json_handler(_payload_from_series(series)))


def _trades_key(result: BacktestResult):
    return [
        (
            trade.entry_day,
            trade.entry_open,
            trade.exit_day,
            trade.exit_open,
            trade.shares,
        )
        for trade in result.trades
    ]


# --- Cálculo conocido con costes -----------------------------------------


def test_known_roundtrip_with_costs_exact_values() -> None:
    result = run_tactical_backtest(_gapless_roundtrip_200())
    assert result.total_bars == 200
    assert result.first_day == date(2026, 6, 1)
    assert len(result.trades) == 1
    trade = result.trades[0]
    # Entrada: senal en t=50 (SMA20=105 > SMA50=102), apertura t+1=200.
    assert trade.entry_open == Decimal("200")
    assert trade.capital_before == Decimal("10000")
    assert trade.shares == Decimal("49.95")
    # Salida: senal en t=100 (SMA20=195 < SMA50=198), apertura t+1=100.
    assert trade.exit_open == Decimal("100")
    assert trade.capital_after == Decimal("49.95") * Decimal("100") * Decimal(
        "0.999"
    )
    assert trade.capital_after == Decimal("4990.005")
    assert trade.win is False
    assert trade.sessions_held == 50
    assert result.final_capital == Decimal("4990.005")
    assert result.strategy_return == Decimal("4990.005") / Decimal(
        "10000"
    ) - Decimal(1)
    assert result.buy_and_hold_return == Decimal("-0.001")
    assert result.max_drawdown == Decimal("0.5009995")
    assert result.win_ratio == Decimal(0)
    assert result.average_duration == Decimal("50")
    assert result.initial_capital == INITIAL_VIRTUAL_CAPITAL
    assert result.exploratory_sample is True
    assert result.signals_possible is True


def test_known_winning_trade_exact_values() -> None:
    result = run_tactical_backtest(_series_winning_trade_220())
    assert len(result.trades) == 1
    trade = result.trades[0]
    assert trade.entry_open == Decimal("102")
    assert trade.exit_open == Decimal("180")
    expected_after = (
        Decimal("9990") / Decimal("102") * Decimal("180") * Decimal("0.999")
    )
    assert trade.capital_after == expected_after
    assert trade.win is True
    assert trade.sessions_held == 149
    assert result.win_ratio == Decimal(1)
    assert result.average_duration == Decimal("149")
    # Drawdown exacto: pico marcado a 200, valle final 180*0.999.
    assert result.max_drawdown.quantize(Decimal("0.000001")) == Decimal(
        "0.100900"
    )


# --- Ausencia de look-ahead ------------------------------------------------


def test_signal_on_last_bar_never_trades() -> None:
    """Senal en el ultimo cierre sin sesion t+1: no hay operacion."""
    closes = _flat("100", 50) + ["110"]
    result = run_tactical_backtest(_series_from_closes(closes))
    assert result.trades == ()
    assert result.open_position is False
    assert result.final_capital == Decimal("10000")


def test_entry_happens_on_session_after_signal_not_on_signal_day() -> None:
    result = run_tactical_backtest(_gapless_roundtrip_200())
    trade = result.trades[0]
    days = [bar.day for bar in _gapless_roundtrip_200().bars]
    assert trade.entry_day == days[51]
    assert trade.entry_day != days[50]


def test_future_data_does_not_change_past_trades() -> None:
    base = _gapless_roundtrip_200()
    closes = ["100"] * 50 + ["200"] * 50 + ["100"] * 100
    mutated_closes = list(closes)
    mutated_closes[150] = "999"
    mutated = _series_from_closes(mutated_closes)
    result_base = run_tactical_backtest(base)
    result_mutated = run_tactical_backtest(mutated)
    # Las operaciones previas al punto mutado no cambian; cualquier
    # operacion adicional ocurre estrictamente despues de la salida
    # original, con datos que solo existen a partir de entonces.
    past = len(result_base.trades)
    assert _trades_key(result_base) == _trades_key(result_mutated)[:past]
    for trade in result_mutated.trades[past:]:
        assert trade.entry_day > result_base.trades[-1].exit_day


def test_truncation_preserves_past_trades() -> None:
    closes = ["100"] * 50 + ["200"] * 50 + ["100"] * 100
    bars = _series_from_closes(closes).bars
    truncated = DailySeries(
        symbol="TEST",
        provider_last_refreshed="",
        provider_timezone="",
        bars=bars[:150],
    )
    result_full = run_tactical_backtest(_series_from_closes(closes))
    result_truncated = run_tactical_backtest(truncated)
    assert _trades_key(result_full) == _trades_key(result_truncated)


def test_execution_uses_next_open_never_signal_close() -> None:
    """Con hueco entre cierre y apertura, la ejecucion usa la apertura t+1."""
    closes = ["100"] * 50 + ["200"] * 50 + ["100"] * 100
    opens = ["105"] + [str(int(value) + 5) for value in closes[:-1]]
    result = run_tactical_backtest(_series(opens, closes))
    trade = result.trades[0]
    assert trade.entry_open == Decimal("205")
    assert trade.exit_open == Decimal("105")
    days = [bar.day for bar in _series(opens, closes).bars]
    assert trade.entry_day == days[51]
    assert trade.exit_day == days[101]


# --- Cruce de entrada/salida ------------------------------------------------


def test_entry_and_exit_occur_at_next_session_open() -> None:
    result = run_tactical_backtest(_gapless_roundtrip_200())
    trade = result.trades[0]
    assert trade.entry_open == Decimal("200")
    assert trade.exit_open == Decimal("100")
    days = [bar.day for bar in _gapless_roundtrip_200().bars]
    assert trade.entry_day == days[51]
    assert trade.exit_day == days[101]
    assert trade.exit_day > trade.entry_day


def test_equal_sma_never_triggers_and_flat_series_has_no_trades() -> None:
    closes = _flat("100", 60)
    result = run_tactical_backtest(_series_from_closes(closes))
    assert result.trades == ()
    assert result.final_capital == Decimal("10000")
    assert result.strategy_return == Decimal(0)
    assert result.max_drawdown == Decimal(0)
    assert result.open_position is False
    assert result.win_ratio is None
    assert result.average_duration is None
    assert result.buy_and_hold_return == Decimal("-0.001")


# --- Drawdown y benchmark ---------------------------------------------------


def test_max_drawdown_of_strategy_equity_curve() -> None:
    """El drawdown usa la curva de equity de la estrategia, no los cierres."""
    result = run_tactical_backtest(_gapless_roundtrip_200())
    assert result.max_drawdown == Decimal("0.5009995")


def test_buy_and_hold_benchmark_known_value() -> None:
    result = run_tactical_backtest(_gapless_roundtrip_200())
    assert result.buy_and_hold_return == Decimal("-0.001")


def test_buy_and_hold_rising_series_positive() -> None:
    closes = _flat("100", 50) + _flat("200", 150)
    result = run_tactical_backtest(_series_from_closes(closes))
    assert result.buy_and_hold_return == Decimal("0.998")


# --- Datos insuficientes e inválidos ----------------------------------------


def test_short_series_runs_without_signals_and_flags_exploratory() -> None:
    result = run_tactical_backtest(
        _series_from_closes(["100", "101", "102"])
    )
    assert result.total_bars == 3
    assert result.signals_possible is False
    assert result.trades == ()
    assert result.final_capital == Decimal("10000")
    assert result.strategy_return == Decimal(0)
    assert result.exploratory_sample is True


def test_series_below_252_but_above_50_is_exploratory() -> None:
    closes = _flat("100", 50) + _flat("200", 50)
    result = run_tactical_backtest(_series_from_closes(closes))
    assert result.exploratory_sample is True
    assert result.signals_possible is True


def test_non_positive_open_rejected() -> None:
    closes = _flat("100", 55)
    opens = ["100"] * 55
    opens[20] = "0"
    with pytest.raises(TacticalBacktestError):
        run_tactical_backtest(_series(opens, closes))


def test_non_positive_close_rejected() -> None:
    with pytest.raises(TacticalBacktestError):
        run_tactical_backtest(
            _series_from_closes(["100", "0", "101", "102", "103"])
        )


def test_duplicated_day_rejected() -> None:
    bar = _bar("2026-06-01", "100", "100")
    series = DailySeries(
        symbol="TEST",
        provider_last_refreshed="",
        provider_timezone="",
        bars=(bar, bar),
    )
    with pytest.raises(TacticalBacktestError):
        run_tactical_backtest(series)


def test_empty_series_rejected() -> None:
    series = DailySeries(
        symbol="TEST",
        provider_last_refreshed="",
        provider_timezone="",
        bars=(),
    )
    with pytest.raises(TacticalBacktestError):
        run_tactical_backtest(series)


# --- Chat [BACKTEST] ---------------------------------------------------------


def test_chat_report_contains_required_fields() -> None:
    series = _gapless_roundtrip_200()
    chat = TacticalBacktestChat(_client_for_series(series), now_provider=_now())
    assert chat.handles("backtest tactical TEST")
    text = chat.handle("backtest tactical TEST")
    assert text.startswith(BACKTEST_LABEL)
    assert "Alpha Vantage" in text
    assert "Sesiones: 200" in text
    assert "2026-06-01" in text
    assert "long-only" in text
    assert "SMA20(t) > SMA50(t)" in text
    assert "apertura de la sesion t+1" in text
    assert "10 bps por entrada y 10 bps por salida" in text
    assert "Capital inicial virtual fijo: 10000.00" in text
    assert "Capital final virtual: 4990.00" in text
    assert "-50.10 %" in text
    assert "-0.10 %" in text
    assert "50.10 %" in text
    assert "Operaciones completadas (ida y vuelta): 1" in text
    assert "0.00 %" in text
    assert "50.00 sesiones" in text
    assert "Posicion abierta al final del historico: no." in text
    assert (
        "muestra exploratoria, insuficiente para validar una estrategia"
        in text
    )
    assert "no es una recomendacion de compra o venta" in text
    assert "no modifica watchlist, PAPER, MarketEvent" in text


def test_chat_report_open_position_and_costs_note() -> None:
    closes = _flat("100", 50) + _flat("110", 20) + _flat("200", 30)
    series = _series_from_closes(closes)
    chat = TacticalBacktestChat(_client_for_series(series), now_provider=_now())
    text = chat.handle("backtest tactical TEST")
    assert "Posicion abierta al final del historico: si" in text
    assert "no se ha pagado todavia" in text
    assert "+81.64 %" in text
    assert "muestra exploratoria" in text


def test_chat_no_trades_report() -> None:
    series = _series_from_closes(_flat("100", 60))
    chat = TacticalBacktestChat(_client_for_series(series), now_provider=_now())
    text = chat.handle("backtest tactical TEST")
    assert "Operaciones completadas (ida y vuelta): 0" in text
    assert "no calculables (sin operaciones completadas)" in text


def test_chat_short_series_warns_and_reports_no_signals() -> None:
    series = _series_from_closes(["100", "101", "102"])
    chat = TacticalBacktestChat(_client_for_series(series), now_provider=_now())
    text = chat.handle("backtest tactical TEST")
    assert "muestra exploratoria, insuficiente para validar una estrategia" in text
    assert "Sin senales posibles" in text
    assert "Operaciones completadas (ida y vuelta): 0" in text


def test_chat_provider_error_is_safe() -> None:
    chat = TacticalBacktestChat(
        _client(_json_handler({"Note": "rate limit"})), now_provider=_now()
    )
    text = chat.handle("backtest tactical TEST")
    assert BACKTEST_LABEL in text
    assert "Limite de peticiones" in text
    assert "no se simula nada" in text.casefold()


def test_chat_missing_key_is_safe() -> None:
    client = AlphaVantageClient(
        "", transport=httpx.MockTransport(
            lambda request: httpx.Response(200, text="{}")
        )
    )
    chat = TacticalBacktestChat(client, now_provider=_now())
    text = chat.handle("backtest tactical TEST")
    assert "ALPHAVANTAGE_API_KEY" in text
    assert "No se consultan datos" in text


def test_chat_invalid_series_reports_without_metrics() -> None:
    series = _series_from_closes(["100", "0", "101"])
    chat = TacticalBacktestChat(_client_for_series(series), now_provider=_now())
    text = chat.handle("backtest tactical TEST")
    assert BACKTEST_LABEL in text
    assert "fechas o valores invalidos" in text
    assert "no se simula nada" in text.casefold()
    assert "50.10" not in text


def test_chat_never_exposes_api_key() -> None:
    series = _gapless_roundtrip_200()
    chat = TacticalBacktestChat(_client_for_series(series), now_provider=_now())
    assert FAKE_KEY not in chat.handle("backtest tactical TEST")


def test_chat_classifies_only_backtest_command() -> None:
    series = _gapless_roundtrip_200()
    chat = TacticalBacktestChat(_client_for_series(series), now_provider=_now())
    assert chat.handles("backtest tactical AAPL")
    assert chat.handles("backtest tactical de VUSA.AMS")
    assert chat.handles("Backtest tactical IBM?")
    assert not chat.handles("backtest tactical")
    assert not chat.handles("datos mercado AAPL")
    assert not chat.handles("analiza mercado TEST")
    assert not chat.handles("compra paper 10 de AAPL a mercado")
    assert not chat.handles("cartera paper")
    assert not chat.handles("revisar seguimiento")
    assert not chat.handles("")
    assert not chat.handles("backtest tactical TEST\ncompra paper 1 de TEST")
    with pytest.raises(ValueError):
        chat.handle("cartera paper")


def test_classifier_boundaries_do_not_capture_other_commands() -> None:
    assert classify_tactical_backtest_prompt("backtest tactical IBM") == "IBM"
    assert classify_tactical_backtest_prompt("cartera paper") is None
    assert classify_tactical_backtest_prompt("datos mercado IBM") is None
    assert classify_tactical_backtest_prompt("analiza mercado IBM") is None
    assert (
        classify_tactical_backtest_prompt("importa precio mercado IBM a paper")
        is None
    )
    assert classify_tactical_backtest_prompt("backtest optimizado IBM") is None


# --- Aislamiento de PAPER y watchlist ----------------------------------------


def test_backtest_never_modifies_paper_watchlist_or_orders() -> None:
    directory = Path(tempfile.mkdtemp(prefix="atlas_backtest_paper_"))
    try:
        paper_store = PaperStore(directory / "finance_paper")
        paper_service = PaperFinanceService(paper_store)
        snapshot_before = paper_service.engine.snapshot()
        paper_before = paper_store.state_path.read_text(encoding="utf-8")
        series = _gapless_roundtrip_200()
        chat = TacticalBacktestChat(
            _client_for_series(series), now_provider=_now()
        )
        chat.handle("backtest tactical TEST")
        chat.handle("backtest tactical NOEXISTE")
        assert paper_service.engine.snapshot() == snapshot_before
        assert paper_service.engine.fills == ()
        assert paper_service.engine.pending_orders() == ()
        assert paper_store.state_path.read_text(encoding="utf-8") == paper_before
        assert not (directory / "watchlist.json").exists()
    finally:
        shutil.rmtree(directory, ignore_errors=True)


# --- Ruta real Atlas.process_prompt sin LLM ----------------------------------


def _forbid_llm(monkeypatch) -> None:
    def forbidden(self, *args, **kwargs):
        raise AssertionError("El LLM no debe invocarse en el backtest tactico")

    monkeypatch.setattr(ModelInferenceRunner, "run", forbidden)


@pytest.fixture()
def atlas(monkeypatch):
    series = _gapless_roundtrip_200()
    instance = Atlas()
    instance._orchestrator._alpha_vantage_client_instance = (
        _client_for_series(series)
    )
    _forbid_llm(monkeypatch)
    return instance


def test_atlas_process_prompt_routes_backtest_without_llm(atlas) -> None:
    response = atlas.process_prompt("backtest tactical TEST")
    assert response.startswith(BACKTEST_LABEL)
    assert "Sesiones: 200" in response
    assert "10 bps" in response
    assert "muestra exploratoria" in response
