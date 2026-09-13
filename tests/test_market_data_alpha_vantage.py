"""Tests de Atlas Finance V2.4: adaptador Alpha Vantage de solo lectura.

Todas las respuestas HTTP son falsas (httpx.MockTransport); nunca se usa
red ni una API key real. La clave ficticia de los tests nunca se envia a
ningun sitio y solo comprueba que no se filtra en las respuestas.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from decimal import Decimal

import httpx
import pytest

from tools.alpha_vantage import (
    ALPHAVANTAGE_ENV_VAR,
    AlphaVantageClient,
    AlphaVantageError,
)
from finance.paper.service import PaperFinanceService
from finance.paper.store import PaperStore
from use_cases.market_data_chat import MarketDataChat

FAKE_KEY = "test-fake-key-not-real"

DAILY_PAYLOAD = {
    "Meta Data": {
        "1. Information": "Daily Prices (open, high, low, close, volume)",
        "2. Symbol": "AAPL",
        "3. Last Refreshed": "2026-09-11",
        "4. Time Zone": "US/Eastern",
    },
    "Time Series (Daily)": {
        "2026-09-11": {
            "1. open": "227.10",
            "2. high": "229.30",
            "3. low": "226.40",
            "4. close": "228.55",
            "5. volume": "51234567",
        },
        "2026-09-10": {
            "1. open": "225.00",
            "2. high": "227.80",
            "3. low": "224.10",
            "4. close": "227.20",
            "5. volume": "49887766",
        },
    },
}

SEARCH_PAYLOAD = {
    "bestMatches": [
        {
            "1. symbol": "AAPL",
            "2. name": "APPLE INC",
            "3. type": "Equity",
            "4. region": "United States",
            "8. currency": "USD",
        }
    ]
}


def _now(day: str):
    fixed = datetime.fromisoformat(f"{day}T12:00:00+00:00")

    def provider():
        return fixed

    return provider


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


def test_daily_series_success_with_provider_timestamp() -> None:
    client = _client(_json_handler(DAILY_PAYLOAD))
    series = client.daily_series("aapl")
    assert series.symbol == "AAPL"
    assert series.provider_last_refreshed == "2026-09-11"
    assert series.provider_timezone == "US/Eastern"
    assert len(series.bars) == 2
    last = series.last_bar
    assert last is not None
    assert last.close == Decimal("228.55")
    assert last.volume == 51234567


def test_search_symbol_success() -> None:
    client = _client(_json_handler(SEARCH_PAYLOAD))
    matches = client.search_symbol("apple")
    assert len(matches) == 1
    assert matches[0].symbol == "AAPL"
    assert matches[0].name == "APPLE INC"
    assert matches[0].currency == "USD"


def test_chat_daily_command_success_label_and_source() -> None:
    client = _client(_json_handler(DAILY_PAYLOAD))
    chat = MarketDataChat(client, now_provider=_now("2026-09-11"))
    assert chat.handles("datos mercado AAPL")
    text = chat.handle("datos mercado AAPL")
    assert "[MARKET DATA]" in text
    assert "Alpha Vantage" in text
    assert "2026-09-11" in text
    assert "US/Eastern" in text
    assert "DATOS DIARIOS" in text
    assert "tiempo real" not in text.lower().replace("no es tiempo real", "")
    assert "228.55" in text


def test_chat_daily_command_stale_data_is_labeled_retrasados() -> None:
    client = _client(_json_handler(DAILY_PAYLOAD))
    chat = MarketDataChat(client, now_provider=_now("2026-09-13"))
    text = chat.handle("datos mercado AAPL")
    assert "RETRASADOS" in text
    assert "DATOS DIARIOS" not in text


def test_chat_search_command_success() -> None:
    client = _client(_json_handler(SEARCH_PAYLOAD))
    chat = MarketDataChat(client, now_provider=_now("2026-09-11"))
    assert chat.handles("buscar activo apple")
    text = chat.handle("buscar activo apple")
    assert "[MARKET DATA]" in text
    assert "AAPL" in text
    assert "APPLE INC" in text


def test_symbol_not_found_returns_safe_error() -> None:
    error_payload = {
        "Error Message": "Invalid API call. Please retry or visit documentation."
    }
    client = _client(_json_handler(error_payload))
    chat = MarketDataChat(client, now_provider=_now("2026-09-11"))
    text = chat.handle("datos mercado NOEXISTE")
    assert "[MARKET DATA]" in text
    assert "no reconoce el simbolo" in text
    assert "no se inventan precios" in text.lower()


def test_search_without_results_returns_safe_error() -> None:
    client = _client(_json_handler({"bestMatches": []}))
    chat = MarketDataChat(client, now_provider=_now("2026-09-11"))
    text = chat.handle("buscar activo zzzzz")
    assert "[MARKET DATA]" in text
    assert "no se inventan precios" in text.lower()


def test_missing_key_never_calls_http() -> None:
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200, text="{}")

    client = _client(handler, api_key="")
    assert client.configured is False
    chat = MarketDataChat(client, now_provider=_now("2026-09-11"))
    text = chat.handle("datos mercado AAPL")
    assert calls == []
    assert "ALPHAVANTAGE_API_KEY" in text
    assert "no se inventan precios" in text.lower().replace(
        "no se consultan datos ni se inventan precios",
        "no se inventan precios",
    )
    with pytest.raises(AlphaVantageError) as excinfo:
        client.daily_series("AAPL")
    assert excinfo.value.code == "MISSING_KEY"


def test_rate_limit_returns_safe_error() -> None:
    client = _client(
        _json_handler({"Note": "API call frequency limit reached."})
    )
    chat = MarketDataChat(client, now_provider=_now("2026-09-11"))
    text = chat.handle("datos mercado AAPL")
    assert "[MARKET_DATA]" not in text
    assert "[MARKET DATA]" in text
    assert "Limite de peticiones" in text
    assert "no se inventan precios" in text.lower()


def test_invalid_json_returns_safe_error() -> None:
    def bad_json_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="not-json")

    client = _client(bad_json_handler)
    chat = MarketDataChat(client, now_provider=_now("2026-09-11"))
    text = chat.handle("datos mercado AAPL")
    assert "[MARKET DATA]" in text
    assert "no es valida" in text
    assert "no se inventan precios" in text.lower()


def test_http_error_and_timeout_map_to_safe_errors() -> None:
    def http_error_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="oops")

    client = _client(http_error_handler)
    chat = MarketDataChat(client, now_provider=_now("2026-09-11"))
    text = chat.handle("datos mercado AAPL")
    assert "error HTTP" in text

    def timeout_handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("timed out")

    slow_client = _client(timeout_handler)
    chat_slow = MarketDataChat(slow_client, now_provider=_now("2026-09-11"))
    assert "Error de red" in chat_slow.handle("datos mercado AAPL")


def test_response_size_limit_is_enforced() -> None:
    big_payload = json.dumps({"Meta Data": {}, "pad": "x" * 3_000_000})

    def big_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=big_payload)

    client = _client(big_handler, max_response_bytes=1_000_000)
    with pytest.raises(AlphaVantageError) as excinfo:
        client.daily_series("AAPL")
    assert excinfo.value.code == "RESPONSE_TOO_LARGE"


def test_api_key_is_never_exposed_in_responses_or_errors() -> None:
    client = _client(_json_handler(DAILY_PAYLOAD))
    chat = MarketDataChat(client, now_provider=_now("2026-09-11"))
    for prompt in ("datos mercado AAPL", "buscar activo apple"):
        assert FAKE_KEY not in chat.handle(prompt)
    assert FAKE_KEY not in repr(client)
    def bad_body_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="bad")

    error_client = _client(bad_body_handler)
    with pytest.raises(AlphaVantageError) as excinfo:
        error_client.daily_series("AAPL")
    assert FAKE_KEY not in excinfo.value.message


def test_from_env_reads_documented_variable_only(monkeypatch) -> None:
    monkeypatch.setenv(ALPHAVANTAGE_ENV_VAR, FAKE_KEY)
    client = AlphaVantageClient.from_env(environ=dict(__import__("os").environ))
    assert client.configured is True
    monkeypatch.setenv(ALPHAVANTAGE_ENV_VAR, "")
    empty = AlphaVantageClient.from_env(environ={ALPHAVANTAGE_ENV_VAR: ""})
    assert empty.configured is False


def test_paper_portfolio_is_never_modified_by_market_data() -> None:
    import shutil
    import tempfile
    from pathlib import Path

    directory = Path(tempfile.mkdtemp(prefix="atlas_market_paper_"))
    try:
        store = PaperStore(directory / "paper")
        service = PaperFinanceService(store)
        snapshot_before = service.engine.snapshot()
        client = _client(_json_handler(DAILY_PAYLOAD))
        chat = MarketDataChat(client, now_provider=_now("2026-09-11"))
        chat.handle("datos mercado AAPL")
        chat.handle("buscar activo apple")
        chat.handle("datos mercado NOEXISTE")
        snapshot_after = service.engine.snapshot()
        assert snapshot_before == snapshot_after
        assert service.engine.fills == ()
        assert service.engine.pending_orders() == ()
    finally:
        shutil.rmtree(directory, ignore_errors=True)


def test_unknown_market_data_prompt_is_not_handled() -> None:
    client = _client(_json_handler(DAILY_PAYLOAD))
    chat = MarketDataChat(client, now_provider=_now("2026-09-11"))
    assert not chat.handles("cartera paper")
    assert not chat.handles("compra paper 10 de AAPL a mercado")
    assert not chat.handles("investiga apple")
    assert not chat.handles("datos mercado")
    assert not chat.handles("datos mercado AAPL\ncompra paper 10 de AAPL")
    with pytest.raises(ValueError):
        chat.handle("cartera paper")
