"""Tests de Atlas Finance V2.8: watchlist tactica paper.

Cubren alta/listado/borrado/duplicados/persistencia por provider_symbol
exacto (VUSA.AMS se consulta como VUSA.AMS; IBM como IBM), migracion del
esquema V2.8 original (simbolo+bolsa) sin perder registros, revision con
mezcla CORE/TACTICAL, limite de 5 activos por ejecucion, fallo y
rate-limit parciales, aislamiento total de PAPER y reglas de sintaxis.
Las respuestas HTTP son falsas (httpx.MockTransport); nunca hay red,
API key real, LLM, ordenes, MarketEvents ni fills. Incluye regresion de
aislamiento frente al estado paper V2.1-V2.7.
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

from finance.paper.policy import PaperMode
from finance.paper.service import PaperFinanceService
from finance.paper.store import PaperStore
from finance.watchlist.store import (
    SCHEMA_VERSION,
    WatchlistStore,
    WatchlistStoreError,
)
from tools.alpha_vantage import AlphaVantageClient, DailyBar, DailySeries
from use_cases.watchlist_chat import (
    MAX_REVIEW_SYMBOLS_PER_RUN,
    WATCHLIST_LABEL,
    WatchlistChat,
    classify_watchlist_prompt,
)

FAKE_KEY = "test-fake-key-not-real"

REVIEW_DAY = "2026-09-11"


def _bar(day: str, close: str) -> DailyBar:
    close_value = Decimal(close)
    return DailyBar(
        day=date.fromisoformat(day),
        open=close_value,
        high=close_value,
        low=close_value,
        close=close_value,
        volume=1000,
    )


def _next_session(day: date) -> date:
    next_day = day
    while True:
        next_day = date.fromordinal(next_day.toordinal() + 1)
        if next_day.weekday() < 5:
            return next_day


def _series(closes: list[str], *, symbol: str = "VUSA") -> DailySeries:
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


def _client_for_symbols(
    series_by_symbol: dict[str, DailySeries | Exception],
    *,
    calls: list[str] | None = None,
) -> AlphaVantageClient:
    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        symbol = None
        for part in url.split("&"):
            if part.startswith("symbol="):
                symbol = part.removeprefix("symbol=").upper()
        assert symbol is not None  # noqa: S101
        if calls is not None:
            calls.append(symbol)
        outcome = series_by_symbol.get(symbol)
        if isinstance(outcome, Exception):
            code = getattr(outcome, "code", None)
            if code == "RATE_LIMIT" or "rate" in str(outcome).lower():
                return httpx.Response(
                    200, text=json.dumps({"Note": "rate limit"})
                )
            return httpx.Response(
                200,
                text=json.dumps({"Error Message": "invalid"}),
            )
        return httpx.Response(200, text=json.dumps(_payload_from_series(outcome)))

    return _client(handler)


def _tmp_store() -> tuple[WatchlistStore, Path]:
    directory = Path(tempfile.mkdtemp(prefix="atlas_watchlist_"))
    return WatchlistStore(directory), directory


def _chat(
    store: WatchlistStore | None = None,
    client: AlphaVantageClient | None = None,
) -> tuple[WatchlistChat, WatchlistStore, Path]:
    if store is None:
        store, directory = _tmp_store()
    else:
        directory = store._directory
    return WatchlistChat(store, client, now_provider=_now()), store, directory


def _write_legacy_watchlist(directory: Path, entries: list[dict]) -> None:
    """Escribe un watchlist.json con el esquema V2.8 original (v1)."""
    payload = {"schema_version": 1, "entries": entries}
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "watchlist.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


# --- Store: provider_symbol exacto, alta/listado/borrado/duplicados --------


def test_store_add_list_remove_roundtrip() -> None:
    store, directory = _tmp_store()
    try:
        store.add_entry(
            provider_symbol="VUSA.AMS", mode=PaperMode.TACTICAL,
            note="etf mundo", added_on="2026-09-10",
        )
        store.add_entry(
            provider_symbol="IBM", mode=PaperMode.CORE,
            added_on="2026-09-11",
        )
        entries = store.entries()
        assert [entry["provider_symbol"] for entry in entries] == [
            "VUSA.AMS", "IBM"
        ]
        assert entries[0]["mode"] == "TACTICAL"
        assert entries[0]["note"] == "etf mundo"
        assert entries[1]["note"] is None
        removed = store.remove_entry("vusa.ams")
        assert removed["provider_symbol"] == "VUSA.AMS"
        assert [
            entry["provider_symbol"] for entry in store.entries()
        ] == ["IBM"]
    finally:
        shutil.rmtree(directory, ignore_errors=True)


def test_store_keeps_exact_provider_symbol_with_suffix() -> None:
    """VUSA.AMS se guarda y se devuelve exacto, sin alterar el sufijo."""
    store, directory = _tmp_store()
    try:
        store.add_entry(
            provider_symbol="VUSA.AMS", mode=PaperMode.TACTICAL,
            added_on="2026-09-10",
        )
        entry = store.entries()[0]
        assert entry["provider_symbol"] == "VUSA.AMS"
        raw = json.loads(
            store.state_path.read_text(encoding="utf-8")
        )
        assert raw["entries"][0]["provider_symbol"] == "VUSA.AMS"
        assert raw["schema_version"] == SCHEMA_VERSION
    finally:
        shutil.rmtree(directory, ignore_errors=True)


def test_store_allows_plain_symbol_without_exchange() -> None:
    """IBM sin bolsa es un identificador valido: no se exige .BOLSA."""
    store, directory = _tmp_store()
    try:
        store.add_entry(
            provider_symbol="IBM", mode=PaperMode.CORE,
            added_on="2026-09-10",
        )
        entries = store.entries()
        assert len(entries) == 1
        assert entries[0]["provider_symbol"] == "IBM"
        assert entries[0]["needs_review"] is False
    finally:
        shutil.rmtree(directory, ignore_errors=True)


def test_store_rejects_duplicates_same_provider_symbol() -> None:
    store, directory = _tmp_store()
    try:
        store.add_entry(
            provider_symbol="VUSA.AMS", mode=PaperMode.TACTICAL,
            added_on="2026-09-10",
        )
        with pytest.raises(WatchlistStoreError):
            store.add_entry(
                provider_symbol="VUSA.AMS", mode=PaperMode.CORE,
                added_on="2026-09-11",
            )
        assert len(store.entries()) == 1
    finally:
        shutil.rmtree(directory, ignore_errors=True)


def test_store_persists_atomically_and_reloads() -> None:
    store, directory = _tmp_store()
    try:
        store.add_entry(
            provider_symbol="IBM", mode=PaperMode.CORE,
            added_on="2026-09-10",
        )
        assert store.exists()
        reloaded = WatchlistStore(directory)
        entries = reloaded.entries()
        assert len(entries) == 1
        assert entries[0]["provider_symbol"] == "IBM"
        assert entries[0]["mode"] == "CORE"
        assert entries[0]["added_on"] == "2026-09-10"
        raw = json.loads(store.state_path.read_text(encoding="utf-8"))
        assert raw["schema_version"] == SCHEMA_VERSION
        tmp_files = [
            path for path in directory.iterdir() if path.suffix == ".tmp"
        ]
        assert tmp_files == []
    finally:
        shutil.rmtree(directory, ignore_errors=True)


def test_store_corrupt_file_raises_without_overwriting() -> None:
    store, directory = _tmp_store()
    try:
        store.state_path.write_text("{not-json", encoding="utf-8")
        with pytest.raises(WatchlistStoreError):
            store.entries()
        assert store.state_path.read_text(encoding="utf-8") == "{not-json"
    finally:
        shutil.rmtree(directory, ignore_errors=True)


def test_store_rejects_unsupported_version() -> None:
    store, directory = _tmp_store()
    try:
        store.state_path.write_text(
            json.dumps({"schema_version": 99, "entries": []}),
            encoding="utf-8",
        )
        with pytest.raises(WatchlistStoreError):
            store.entries()
    finally:
        shutil.rmtree(directory, ignore_errors=True)


def test_store_validates_note_and_inputs() -> None:
    store, directory = _tmp_store()
    try:
        with pytest.raises(WatchlistStoreError):
            store.add_entry(
                provider_symbol="VUSA.AMS", mode=PaperMode.TACTICAL,
                note="x" * 200, added_on="2026-09-10",
            )
        with pytest.raises(WatchlistStoreError):
            store.add_entry(
                provider_symbol="VUSA.AMS", mode=PaperMode.TACTICAL,
                added_on="10-09-2026",
            )
        with pytest.raises(WatchlistStoreError):
            store.add_entry(
                provider_symbol="", mode=PaperMode.TACTICAL,
                added_on="2026-09-10",
            )
        assert store.entries() == ()
    finally:
        shutil.rmtree(directory, ignore_errors=True)


# --- Migracion del esquema V2.8 original (simbolo + bolsa) -------------------


def test_migration_rebuilds_vusa_ams_provider_symbol() -> None:
    """VUSA + AMS (esquema v1) migra a provider_symbol VUSA.AMS sin perderlo."""
    store, directory = _tmp_store()
    try:
        _write_legacy_watchlist(
            directory,
            [
                {
                    "symbol": "VUSA",
                    "exchange": "AMS",
                    "mode": "TACTICAL",
                    "note": "etf sp500",
                    "added_on": "2026-09-13",
                    "last_reviewed": None,
                },
            ],
        )
        entries = store.entries()
        assert len(entries) == 1
        assert entries[0]["provider_symbol"] == "VUSA.AMS"
        # Toda entrada heredada queda pendiente de verificacion hasta
        # que una revision exitosa ante el proveedor la confirme.
        assert entries[0]["needs_review"] is True
        assert entries[0]["mode"] == "TACTICAL"
        assert entries[0]["note"] == "etf sp500"
        raw = json.loads(
            store.state_path.read_text(encoding="utf-8")
        )
        assert raw["schema_version"] == SCHEMA_VERSION
        assert raw["entries"][0]["provider_symbol"] == "VUSA.AMS"
    finally:
        shutil.rmtree(directory, ignore_errors=True)


def test_migration_flags_unverifiable_IBM_NYSE_for_review() -> None:
    """IBM + NYSE se conserva sin conversion inventada y queda con aviso."""
    store, directory = _tmp_store()
    try:
        _write_legacy_watchlist(
            directory,
            [
                {
                    "symbol": "IBM",
                    "exchange": "NYSE",
                    "mode": "CORE",
                    "note": None,
                    "added_on": "2026-09-13",
                    "last_reviewed": None,
                },
            ],
        )
        entries = store.entries()
        assert len(entries) == 1
        # No se elimina el sufijo (esa conversion universal inventaria
        # datos): el identificador heredado se conserva con aviso.
        assert entries[0]["provider_symbol"] == "IBM.NYSE"
        assert entries[0]["needs_review"] is True
        raw = json.loads(
            store.state_path.read_text(encoding="utf-8")
        )
        assert raw["schema_version"] == SCHEMA_VERSION
    finally:
        shutil.rmtree(directory, ignore_errors=True)


def test_migration_keeps_both_entries_and_persists() -> None:
    store, directory = _tmp_store()
    try:
        _write_legacy_watchlist(
            directory,
            [
                {
                    "symbol": "VUSA",
                    "exchange": "AMS",
                    "mode": "TACTICAL",
                    "note": "etf sp500",
                    "added_on": "2026-09-13",
                    "last_reviewed": None,
                },
                {
                    "symbol": "IBM",
                    "exchange": "NYSE",
                    "mode": "CORE",
                    "note": None,
                    "added_on": "2026-09-13",
                    "last_reviewed": None,
                },
            ],
        )
        entries = store.entries()
        assert [entry["provider_symbol"] for entry in entries] == [
            "VUSA.AMS", "IBM.NYSE"
        ]
        assert all(entry["needs_review"] for entry in entries)
        reloaded = WatchlistStore(directory)
        assert len(reloaded.entries()) == 2
    finally:
        shutil.rmtree(directory, ignore_errors=True)


def test_migration_persisted_needs_review_survives_reload() -> None:
    """La marca needs_review persistida se respeta entre recargas."""
    store, directory = _tmp_store()
    try:
        _write_legacy_watchlist(
            directory,
            [
                {
                    "symbol": "VUSA",
                    "exchange": "AMS",
                    "mode": "TACTICAL",
                    "note": None,
                    "added_on": "2026-09-13",
                    "last_reviewed": None,
                },
            ],
        )
        assert all(entry["needs_review"] for entry in store.entries())
        reloaded = WatchlistStore(directory)
        assert all(entry["needs_review"] for entry in reloaded.entries())
    finally:
        shutil.rmtree(directory, ignore_errors=True)


def test_review_clears_needs_review_after_successful_provider_check() -> None:
    """Una revision exitosa ante Alpha verifica el simbolo y quita el aviso."""
    store, directory = _tmp_store()
    try:
        _write_legacy_watchlist(
            store._directory,
            [
                {
                    "symbol": "VUSA",
                    "exchange": "AMS",
                    "mode": "TACTICAL",
                    "note": None,
                    "added_on": "2026-09-13",
                    "last_reviewed": None,
                },
            ],
        )
        chat = WatchlistChat(
            store,
            _client_for_symbols(
                {
                    "VUSA.AMS": _series(
                        [f"{100 + index}" for index in range(25)],
                        symbol="VUSA.AMS",
                    ),
                }
            ),
            now_provider=_now(),
        )
        text = chat.handle("revisar seguimiento")
        assert "ERROR" not in text
        entry = store.entries()[0]
        assert entry["needs_review"] is False
        chat_after = WatchlistChat(store, now_provider=_now())
        listing = chat_after.handle("lista seguimiento")
        assert "AVISO" not in listing
    finally:
        shutil.rmtree(directory, ignore_errors=True)


# --- Comandos de chat -------------------------------------------------------


def test_chat_add_command_with_tactical_and_note() -> None:
    chat, store, directory = _chat()
    try:
        text = chat.handle("añade a seguimiento VUSA.AMS tactical nota: etf mundo")
        assert WATCHLIST_LABEL in text
        assert "VUSA.AMS" in text
        assert "TACTICAL" in text
        assert "etf mundo" in text
        entries = store.entries()
        assert len(entries) == 1
        assert entries[0]["provider_symbol"] == "VUSA.AMS"
        assert entries[0]["mode"] == "TACTICAL"
    finally:
        shutil.rmtree(directory, ignore_errors=True)


def test_chat_add_plain_symbol_without_exchange() -> None:
    """La alta acepta el identificador exacto sin bolsa: IBM core."""
    chat, store, directory = _chat()
    try:
        text = chat.handle("anade a seguimiento IBM core")
        assert "CORE" in text
        assert "IBM" in text
        entries = store.entries()
        assert len(entries) == 1
        assert entries[0]["provider_symbol"] == "IBM"
        assert entries[0]["mode"] == "CORE"
        assert entries[0]["needs_review"] is False
    finally:
        shutil.rmtree(directory, ignore_errors=True)


def test_chat_add_with_suffix_still_works() -> None:
    chat, store, directory = _chat()
    try:
        text = chat.handle("añade a seguimiento VUSA.AMS tactical")
        assert "VUSA.AMS" in text
        entries = store.entries()
        assert entries[0]["provider_symbol"] == "VUSA.AMS"
    finally:
        shutil.rmtree(directory, ignore_errors=True)


def test_chat_add_duplicate_reports_and_does_not_modify() -> None:
    chat, store, directory = _chat()
    try:
        chat.handle("añade a seguimiento VUSA.AMS tactical")
        text = chat.handle("añade a seguimiento VUSA.AMS core")
        assert "ya esta en seguimiento" in text
        assert len(store.entries()) == 1
    finally:
        shutil.rmtree(directory, ignore_errors=True)


def test_chat_list_empty_has_clear_message() -> None:
    chat, store, directory = _chat()
    try:
        text = chat.handle("lista seguimiento")
        assert "vacia" in text
        assert "VUSA.AMS" in text
    finally:
        shutil.rmtree(directory, ignore_errors=True)


def test_chat_list_mixed_modes_and_notes() -> None:
    chat, store, directory = _chat()
    try:
        chat.handle("añade a seguimiento VUSA.AMS tactical nota: etf mundo")
        chat.handle("añade a seguimiento IBM core")
        text = chat.handle("lista seguimiento")
        assert "2 activos" in text
        assert "VUSA.AMS" in text
        assert "IBM" in text
        assert "TACTICAL" in text
        assert "CORE" in text
        assert "etf mundo" in text
    finally:
        shutil.rmtree(directory, ignore_errors=True)


def test_chat_list_shows_needs_review_warning_per_asset() -> None:
    """Un identificador heredado no verificable muestra aviso en la lista."""
    store, directory = _tmp_store()
    try:
        _write_legacy_watchlist(
            directory,
            [
                {
                    "symbol": "IBM",
                    "exchange": "NYSE",
                    "mode": "CORE",
                    "note": None,
                    "added_on": "2026-09-13",
                    "last_reviewed": None,
                },
            ],
        )
        chat = WatchlistChat(store, now_provider=_now())
        text = chat.handle("lista seguimiento")
        assert "AVISO" in text
        assert "IBM" in text
        assert "quita de seguimiento IBM" in text
    finally:
        shutil.rmtree(directory, ignore_errors=True)


def test_chat_remove_existing_and_missing() -> None:
    chat, store, directory = _chat()
    try:
        chat.handle("añade a seguimiento VUSA.AMS tactical")
        text = chat.handle("quita de seguimiento VUSA.AMS")
        assert "Baja de seguimiento registrada" in text
        assert store.entries() == ()
        text = chat.handle("quita de seguimiento VUSA.AMS")
        assert "no esta en seguimiento" in text
    finally:
        shutil.rmtree(directory, ignore_errors=True)


def test_chat_remove_plain_symbol() -> None:
    chat, store, directory = _chat()
    try:
        chat.handle("añade a seguimiento IBM core")
        text = chat.handle("quita de seguimiento IBM")
        assert "Baja de seguimiento registrada" in text
        assert store.entries() == ()
    finally:
        shutil.rmtree(directory, ignore_errors=True)


def test_chat_invalid_symbol_is_rejected_without_modification() -> None:
    chat, store, directory = _chat()
    try:
        text = chat.handle("añade a seguimiento .. core")
        assert "rechazada" in text
        assert store.entries() == ()
    finally:
        shutil.rmtree(directory, ignore_errors=True)


def test_classify_watchlist_prompt_boundaries() -> None:
    assert classify_watchlist_prompt("añade a seguimiento VUSA.AMS tactical") == "add"
    assert classify_watchlist_prompt("añade a seguimiento IBM core") == "add"
    assert classify_watchlist_prompt("lista seguimiento") == "list"
    assert classify_watchlist_prompt("seguimiento") == "list"
    assert classify_watchlist_prompt("quita de seguimiento VUSA.AMS") == "remove"
    assert classify_watchlist_prompt("revisar seguimiento") == "review"
    assert classify_watchlist_prompt("cartera paper") is None
    assert classify_watchlist_prompt("analiza mercado TEST") is None
    assert classify_watchlist_prompt("compra paper 10 de AAPL a mercado") is None
    assert classify_watchlist_prompt("") is None
    assert classify_watchlist_prompt("revisar seguimiento\nlista seguimiento") is None


# --- Revision -----------------------------------------------------------------


def _add_two(chat: WatchlistChat) -> None:
    chat.handle("añade a seguimiento VUSA.AMS tactical")
    chat.handle("añade a seguimiento IBM core")


def test_review_queries_alpha_with_exact_provider_symbols() -> None:
    """E2E: VUSA.AMS llega a Alpha como VUSA.AMS e IBM como IBM."""
    calls: list[str] = []
    chat, store, directory = _chat(
        client=_client_for_symbols(
            {
                "VUSA.AMS": _series(
                    [f"{100 + index}" for index in range(25)],
                    symbol="VUSA.AMS",
                ),
                "IBM": _series(
                    [f"{200 - index}" for index in range(25)], symbol="IBM"
                ),
            },
            calls=calls,
        )
    )
    try:
        _add_two(chat)
        text = chat.handle("revisar seguimiento")
        assert WATCHLIST_LABEL in text
        assert "VUSA.AMS" in text and "TACTICAL" in text
        assert "IBM" in text and "CORE" in text
        assert sorted(calls) == ["IBM", "VUSA.AMS"]
        assert "VUSA" not in [call for call in calls if call != "VUSA.AMS"]
        assert REVIEW_DAY in text
        assert "RETRASADOS" in text
        assert "+5 sesiones" in text or "5 sesiones" in text
        assert "tendencia" in text
        assert "Alpha Vantage" in text
        assert "no genera recomendaciones" in text
        assert "no produce fills" in text
    finally:
        shutil.rmtree(directory, ignore_errors=True)


def test_review_continues_after_rate_limit_on_one_asset() -> None:
    calls: list[str] = []
    chat, store, directory = _chat(
        client=_client_for_symbols(
            {
                "VUSA.AMS": RuntimeError("rate-limit-simulated"),
                "IBM": _series(
                    [f"{200 - index}" for index in range(25)], symbol="IBM"
                ),
            },
            calls=calls,
        )
    )
    try:
        _add_two(chat)
        text = chat.handle("revisar seguimiento")
        assert "ERROR" in text
        assert "Limite de peticiones" in text
        assert "IBM" in text
        assert "revision continua" in text
        assert sorted(calls) == ["IBM", "VUSA.AMS"]
    finally:
        shutil.rmtree(directory, ignore_errors=True)


def test_review_rate_limit_error_then_recovery_next_day() -> None:
    chat, store, directory = _chat(
        client=_client_for_symbols(
            {
                "VUSA.AMS": _series(
                    [f"{100 + index}" for index in range(25)],
                    symbol="VUSA.AMS",
                ),
            }
        )
    )
    try:
        chat.handle("añade a seguimiento VUSA.AMS tactical")
        first = chat.handle("revisar seguimiento")
        assert "VUSA.AMS" in first
        assert "ERROR" not in first
        second = chat.handle("revisar seguimiento")
        assert "una consulta diaria" in second
    finally:
        shutil.rmtree(directory, ignore_errors=True)


def test_review_stale_data_is_labeled_retrasados() -> None:
    chat, store, directory = _chat(
        client=_client_for_symbols(
            {
                "VUSA.AMS": _series(
                    [f"{100 + index}" for index in range(25)],
                    symbol="VUSA.AMS",
                ),
            }
        )
    )
    try:
        chat.handle("añade a seguimiento VUSA.AMS tactical")
        text = chat.handle("revisar seguimiento")
        assert "RETRASADOS" in text
    finally:
        shutil.rmtree(directory, ignore_errors=True)


def test_review_with_only_errors_reports_each_asset() -> None:
    chat, store, directory = _chat(
        client=_client_for_symbols(
            {
                "VUSA.AMS": RuntimeError("rate limit"),
                "IBM": RuntimeError("invalid symbol"),
            }
        )
    )
    try:
        _add_two(chat)
        text = chat.handle("revisar seguimiento")
        assert text.count("ERROR") == 2
        assert "no se han revisado" in text
        assert "VUSA.AMS" in text
        assert "IBM" in text
    finally:
        shutil.rmtree(directory, ignore_errors=True)


def test_review_invalid_symbol_reported_per_asset_not_fatal() -> None:
    """Un simbolo invalido se informa por activo y la revision continua."""
    calls: list[str] = []
    chat, store, directory = _chat(
        client=_client_for_symbols(
            {
                "BADSYM": RuntimeError("invalid symbol"),
                "IBM": _series(
                    [f"{200 - index}" for index in range(25)], symbol="IBM"
                ),
            },
            calls=calls,
        )
    )
    try:
        chat.handle("añade a seguimiento BADSYM core")
        chat.handle("añade a seguimiento IBM core")
        text = chat.handle("revisar seguimiento")
        assert "ERROR" in text
        assert "Alpha Vantage no reconoce este simbolo" in text
        assert "IBM" in text
        assert sorted(calls) == ["BADSYM", "IBM"]
    finally:
        shutil.rmtree(directory, ignore_errors=True)


def test_review_empty_watchlist_has_clear_message() -> None:
    chat, store, directory = _chat(client=_client(_json_handler({})))
    try:
        text = chat.handle("revisar seguimiento")
        assert "vacia" in text
        assert "No se consulta ningun proveedor" in text
    finally:
        shutil.rmtree(directory, ignore_errors=True)


def test_review_limit_five_assets_defers_the_rest() -> None:
    chat, store, directory = _chat(client=_client(_json_handler({})))
    try:
        for index in range(7):
            chat.handle(
                f"añade a seguimiento S{index}.AMS "
                f"{'core' if index % 2 == 0 else 'tactical'}"
            )
        text = chat.handle("revisar seguimiento")
        assert f"maximo {MAX_REVIEW_SYMBOLS_PER_RUN}" in text
        assert "Aplazados para la proxima revision: 2" in text
        entries = store.entries()
        assert len(entries) == 7
    finally:
        shutil.rmtree(directory, ignore_errors=True)


def test_review_missing_key_is_reported_per_asset() -> None:
    client = AlphaVantageClient("", transport=httpx.MockTransport(
        lambda request: httpx.Response(200, text="{}")
    ))
    chat, store, directory = _chat(client=client)
    try:
        chat.handle("añade a seguimiento VUSA.AMS tactical")
        text = chat.handle("revisar seguimiento")
        assert "ALPHAVANTAGE_API_KEY" in text
        assert "ERROR" in text
    finally:
        shutil.rmtree(directory, ignore_errors=True)


def test_review_insufficient_data_reports_unavailable_metrics() -> None:
    chat, store, directory = _chat(
        client=_client_for_symbols(
            {"VUSA.AMS": _series(["100", "101", "102"], symbol="VUSA.AMS")}
        )
    )
    try:
        chat.handle("añade a seguimiento VUSA.AMS tactical")
        text = chat.handle("revisar seguimiento")
        assert "no calculable" in text
        assert "metricas no calculables" in text
    finally:
        shutil.rmtree(directory, ignore_errors=True)


def test_review_migrated_vusa_ams_queries_full_symbol() -> None:
    """E2E migracion: VUSA+AMS heredado consulta Alpha como VUSA.AMS."""
    calls: list[str] = []
    store, directory = _tmp_store()
    try:
        _write_legacy_watchlist(
            store._directory,
            [
                {
                    "symbol": "VUSA",
                    "exchange": "AMS",
                    "mode": "TACTICAL",
                    "note": None,
                    "added_on": "2026-09-13",
                    "last_reviewed": None,
                },
            ],
        )
        chat = WatchlistChat(
            store,
            _client_for_symbols(
                {
                    "VUSA.AMS": _series(
                        [f"{100 + index}" for index in range(25)],
                        symbol="VUSA.AMS",
                    ),
                },
                calls=calls,
            ),
            now_provider=_now(),
        )
        text = chat.handle("revisar seguimiento")
        assert calls == ["VUSA.AMS"]
        assert "VUSA.AMS" in text
        assert "ERROR" not in text
    finally:
        shutil.rmtree(directory, ignore_errors=True)


# --- Aislamiento total de PAPER ------------------------------------------------


def test_review_never_modifies_paper_portfolio_or_creates_orders() -> None:
    chat, store, directory = _chat(
        client=_client_for_symbols(
            {
                "VUSA.AMS": _series(
                    [f"{100 + index}" for index in range(25)],
                    symbol="VUSA.AMS",
                ),
                "IBM": _series(
                    [f"{200 - index}" for index in range(25)], symbol="IBM"
                ),
            }
        )
    )
    paper_directory = directory / "finance_paper"
    try:
        paper_store = PaperStore(paper_directory)
        paper_service = PaperFinanceService(paper_store)
        snapshot_before = paper_service.engine.snapshot()
        _add_two(chat)
        chat.handle("revisar seguimiento")
        snapshot_after = paper_service.engine.snapshot()
        assert snapshot_before == snapshot_after
        assert paper_service.engine.fills == ()
        assert paper_service.engine.pending_orders() == ()
        watchlist_state = json.loads(
            (directory / "watchlist.json").read_text(encoding="utf-8")
        )
        assert watchlist_state["schema_version"] == SCHEMA_VERSION
        assert "positions" not in json.dumps(watchlist_state)
    finally:
        shutil.rmtree(directory, ignore_errors=True)


def test_paper_state_file_is_untouched_by_watchlist_operations() -> None:
    chat, store, directory = _chat()
    paper_directory = directory / "finance_paper"
    try:
        paper_store = PaperStore(paper_directory)
        PaperFinanceService(paper_store)
        paper_before = paper_store.state_path.read_text(encoding="utf-8")
        chat.handle("añade a seguimiento VUSA.AMS tactical")
        chat.handle("lista seguimiento")
        chat.handle("quita de seguimiento VUSA.AMS")
        paper_after = paper_store.state_path.read_text(encoding="utf-8")
        assert paper_before == paper_after
    finally:
        shutil.rmtree(directory, ignore_errors=True)


# --- Regresion V2.1-V2.7: los comandos paper siguen teniendo prioridad --------


def test_paper_commands_are_not_captured_by_watchlist() -> None:
    assert classify_watchlist_prompt("cartera paper") is None
    assert classify_watchlist_prompt("compra paper 10 de AAPL a mercado") is None
    assert classify_watchlist_prompt("registra precio paper AAPL 100") is None
    assert classify_watchlist_prompt("modo paper tactical") is None
    assert classify_watchlist_prompt("capital paper") is None
    assert classify_watchlist_prompt("politica paper") is None
    assert classify_watchlist_prompt("importa precio mercado IBM a paper") is None
    assert classify_watchlist_prompt("actualiza precio paper IBM desde mercado") is None
    assert classify_watchlist_prompt("maximo posiciones paper 5") is None


def test_watchlist_and_quant_modules_coexist() -> None:
    from finance.quant.metrics import build_quant_snapshot  # noqa: F401
    from use_cases.market_analysis_chat import classify_market_analysis_prompt

    assert classify_market_analysis_prompt("revisar seguimiento") is None
    assert classify_watchlist_prompt("analiza mercado TEST") is None
