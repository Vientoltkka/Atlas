"""Tests de Atlas Finance V2.5: snapshot cuantitativo de mercado.

Los calculos se verifican contra valores conocidos con Decimal exacto;
las respuestas HTTP son falsas (httpx.MockTransport) y nunca se usa red
ni una API key real. Ningun test toca la cartera paper mas alla de
comprobar que el analisis no la modifica.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from datetime import date, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import httpx
import pytest

from finance.paper.service import PaperFinanceService
from finance.paper.store import PaperStore
from finance.quant.metrics import (
    TREND_DOWN,
    TREND_SIDEWAYS,
    TREND_UNDETERMINED,
    TREND_UP,
    QuantSeriesError,
    annualized_volatility,
    build_quant_snapshot,
    classify_trend,
    max_drawdown,
    pct_change,
    simple_moving_average,
)
from tools.alpha_vantage import AlphaVantageClient, DailyBar, DailySeries
from use_cases.market_analysis_chat import (
    MARKET_ANALYSIS_LABEL,
    MarketAnalysisChat,
)

FAKE_KEY = "test-fake-key-not-real"


def _bar(day: str, close: str, *, open_: str | None = None) -> DailyBar:
    close_value = Decimal(close)
    open_value = Decimal(open_) if open_ is not None else close_value
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


def _series(closes: list[str], *, symbol: str = "TEST") -> DailySeries:
    """Serie diaria sintetica de sesiones consecutivas desde 2026-06-01."""
    bars = []
    day = date(2026, 6, 1)
    for close in closes:
        bars.append(_bar(day.isoformat(), close))
        day = _next_session(day)
    return DailySeries(
        symbol=symbol,
        provider_last_refreshed=bars[-1].day.isoformat() if bars else "",
        provider_timezone="US/Eastern",
        bars=tuple(bars),
    )


def _now(day: str):
    fixed = datetime.fromisoformat(f"{day}T12:00:00+00:00")
    return lambda: fixed


def _client(handler, *, api_key: str = FAKE_KEY, **kwargs) -> AlphaVantageClient:
    return AlphaVantageClient(
        api_key,
        transport=httpx.MockTransport(handler),
        **kwargs,
    )


def _json_handler(payload):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=json.dumps(payload))

    return handler


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


FULL_CLOSES = (
    ["100"] * 30
    + [str(value) for value in range(101, 111)]
    + [str(value) for value in range(111, 131)]
)
FULL_SERIES = _series(FULL_CLOSES)
FULL_PAYLOAD = _payload_from_series(FULL_SERIES)
LAST_DAY = FULL_SERIES.bars[-1].day.isoformat()
STALE_DAY = (
    date.fromisoformat(LAST_DAY) + timedelta(days=30)
).isoformat()


# --- Calculos conocidos -------------------------------------------------


def test_pct_change_known_values() -> None:
    closes = [Decimal(v) for v in ("100", "100", "100", "100", "100", "110")]
    assert pct_change(closes, 5) == Decimal("0.1")
    assert pct_change(closes[:5], 5) is None


def test_sma_known_values() -> None:
    closes = [Decimal(v) for v in ("10", "20", "30")]
    assert simple_moving_average(closes, 3) == Decimal("20")
    assert simple_moving_average(closes, 4) is None


def test_volatility_zero_when_returns_are_constant() -> None:
    closes = []
    value = Decimal("100")
    for _ in range(21):
        closes.append(value)
        value = value * Decimal("1.1")
    result = annualized_volatility(closes, 20)
    assert result == Decimal(0)


def test_volatility_two_returns_closed_form() -> None:
    result = annualized_volatility(
        [Decimal("100"), Decimal("110"), Decimal("99")], 2
    )
    expected = (
        (
            Decimal("0.1") ** 2 + Decimal("-0.1") ** 2
        ).sqrt()
        * Decimal("252").sqrt()
    )
    assert result == expected


def test_max_drawdown_known_value() -> None:
    closes = [Decimal(v) for v in ("100", "110", "88", "95")]
    assert max_drawdown(closes, 4) == Decimal("0.2")
    assert max_drawdown(closes[:3], 4) is None


def test_snapshot_complete_series_metrics_match_known_values() -> None:
    snapshot = build_quant_snapshot(FULL_SERIES)
    assert snapshot.complete
    assert snapshot.total_bars == 60
    assert snapshot.last_close == Decimal("130")
    assert snapshot.change_5 == (Decimal("130") - Decimal("125")) / Decimal("125")
    assert snapshot.change_20 == (Decimal("130") - Decimal("110")) / Decimal("110")
    assert snapshot.sma_20 == Decimal("120.5")
    assert snapshot.sma_50 == Decimal("109.3")
    assert snapshot.max_drawdown_60 == Decimal(0)
    assert snapshot.close_vs_sma_20 == "POR ENCIMA"
    assert snapshot.close_vs_sma_50 == "POR ENCIMA"


def test_snapshot_declining_series_drawdown_known_value() -> None:
    declining = _series(
        ["110"] * 30
        + [str(value) for value in range(109, 99, -1)]
        + [str(value) for value in range(99, 79, -1)]
    )
    snapshot = build_quant_snapshot(declining)
    expected = (Decimal("110") - Decimal("80")) / Decimal("110")
    assert snapshot.max_drawdown_60 == expected


# --- Reglas de tendencia ------------------------------------------------


def test_trend_rules_are_explicit() -> None:
    assert classify_trend(
        Decimal("105"), Decimal("100"), Decimal("90"), Decimal("0.05")
    ) == TREND_UP
    assert classify_trend(
        Decimal("95"), Decimal("100"), Decimal("110"), Decimal("-0.05")
    ) == TREND_DOWN
    assert classify_trend(
        Decimal("100"), Decimal("100"), Decimal("100"), Decimal("0")
    ) == TREND_SIDEWAYS
    assert classify_trend(
        Decimal("100"), None, Decimal("100"), Decimal("0")
    ) == TREND_UNDETERMINED


def test_snapshot_trend_up_with_rising_series() -> None:
    snapshot = build_quant_snapshot(FULL_SERIES)
    assert snapshot.trend == TREND_UP


def test_snapshot_trend_down_with_falling_series() -> None:
    declining = _series(
        ["110"] * 30
        + [str(value) for value in range(109, 99, -1)]
        + [str(value) for value in range(99, 79, -1)]
    )
    snapshot = build_quant_snapshot(declining)
    assert snapshot.trend == TREND_DOWN
    assert snapshot.close_vs_sma_20 == "POR DEBAJO"


# --- Datos insuficientes ------------------------------------------------


def test_snapshot_with_few_bars_reports_unavailable_metrics() -> None:
    snapshot = build_quant_snapshot(_series(["100", "101", "102"]))
    assert snapshot.total_bars == 3
    assert not snapshot.complete
    assert snapshot.change_5 is None
    assert snapshot.change_20 is None
    assert snapshot.sma_20 is None
    assert snapshot.sma_50 is None
    assert snapshot.volatility_20_annualized is None
    assert snapshot.max_drawdown_60 is None
    assert snapshot.trend == TREND_UNDETERMINED
    assert "SMA50" in snapshot.unavailable
    assert "maximo drawdown de las ultimas 60 sesiones" in snapshot.unavailable


def test_chat_insufficient_data_explains_missing_metrics() -> None:
    payload = _payload_from_series(_series(["100", "101", "102"]))
    client = _client(_json_handler(payload))
    chat = MarketAnalysisChat(client, now_provider=_now(LAST_DAY))
    text = chat.handle("analiza mercado TEST")
    assert "[MARKET ANALYSIS]" in text
    assert "no calculable" in text
    assert "SMA50" in text
    assert "drawdown" in text
    assert "INDETERMINADA" in text


# --- Serie irregular ----------------------------------------------------


def test_snapshot_rejects_non_positive_close() -> None:
    with pytest.raises(QuantSeriesError):
        build_quant_snapshot(_series(["100", "0", "101"]))


def test_snapshot_rejects_duplicated_day() -> None:
    bar = _bar("2026-06-01", "100")
    series = DailySeries(
        symbol="TEST",
        provider_last_refreshed="",
        provider_timezone="",
        bars=(bar, bar),
    )
    with pytest.raises(QuantSeriesError):
        build_quant_snapshot(series)


def test_chat_invalid_series_reports_without_metrics() -> None:
    payload = _payload_from_series(_series(["100", "0", "101"]))
    client = _client(_json_handler(payload))
    chat = MarketAnalysisChat(client, now_provider=_now("2026-09-11"))
    text = chat.handle("analiza mercado TEST")
    assert "[MARKET ANALYSIS]" in text
    assert "invalidos" in text
    assert "Ninguna metrica se calcula" in text


# --- Reporte del chat ---------------------------------------------------


def test_chat_report_contains_metadata_and_metrics() -> None:
    client = _client(_json_handler(FULL_PAYLOAD))
    chat = MarketAnalysisChat(client, now_provider=_now(LAST_DAY))
    assert chat.handles("analiza mercado TEST")
    text = chat.handle("analiza mercado TEST")
    assert "[MARKET ANALYSIS]" in text
    assert "Alpha Vantage" in text
    assert LAST_DAY in text
    assert "DATOS DIARIOS" in text
    assert "no es tiempo real" in text
    assert "SMA20" in text
    assert "SMA50" in text
    assert "Volatilidad" in text
    assert "drawdown" in text.lower()
    assert "no garantiza resultados" in text
    assert "descriptivas, no recomendaciones" in text
    assert "no registra MarketEvent" in text


def test_chat_stale_data_is_labeled_retrasados() -> None:
    client = _client(_json_handler(FULL_PAYLOAD))
    chat = MarketAnalysisChat(client, now_provider=_now(STALE_DAY))
    text = chat.handle("analiza mercado TEST")
    assert "RETRASADOS" in text
    assert "DATOS DIARIOS" not in text
    assert "desactualizadas" in text


def test_chat_provider_error_is_safe_and_deterministic() -> None:
    client = _client(_json_handler({"Note": "rate limit reached."}))
    chat = MarketAnalysisChat(client, now_provider=_now(LAST_DAY))
    text = chat.handle("analiza mercado TEST")
    assert "[MARKET ANALYSIS]" in text
    assert "Limite de peticiones" in text
    assert "no se calcula nada" in text


def test_chat_classifies_only_analysis_command() -> None:
    client = _client(_json_handler(FULL_PAYLOAD))
    chat = MarketAnalysisChat(client, now_provider=_now(LAST_DAY))
    assert chat.handles("analiza mercado AAPL")
    assert chat.handles("Analiza el mercado BRK.B")
    assert not chat.handles("datos mercado AAPL")
    assert not chat.handles("compra paper 10 de AAPL a mercado")
    assert not chat.handles("cartera paper")
    assert not chat.handles("analiza mercado")
    assert not chat.handles("analiza mercado AAPL\ncompra paper 10 de AAPL")
    with pytest.raises(ValueError):
        chat.handle("cartera paper")


def test_chat_never_exposes_api_key() -> None:
    client = _client(_json_handler(FULL_PAYLOAD))
    chat = MarketAnalysisChat(client, now_provider=_now(LAST_DAY))
    assert FAKE_KEY not in chat.handle("analiza mercado TEST")


# --- Aislamiento de PAPER -----------------------------------------------


def test_paper_portfolio_is_never_modified_by_market_analysis() -> None:
    directory = Path(tempfile.mkdtemp(prefix="atlas_analysis_paper_"))
    try:
        store = PaperStore(directory / "paper")
        service = PaperFinanceService(store)
        snapshot_before = service.engine.snapshot()
        client = _client(_json_handler(FULL_PAYLOAD))
        chat = MarketAnalysisChat(client, now_provider=_now(LAST_DAY))
        chat.handle("analiza mercado TEST")
        chat.handle("analiza mercado NOEXISTE")
        snapshot_after = service.engine.snapshot()
        assert snapshot_before == snapshot_after
        assert service.engine.fills == ()
        assert service.engine.pending_orders() == ()
    finally:
        shutil.rmtree(directory, ignore_errors=True)
