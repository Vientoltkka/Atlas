"""Tests de Atlas Finance V2.6: importacion confirmada de mercado a paper.

Flujo: comando explicito -> una consulta Alpha Vantage -> propuesta [PAPER]
(dato retrasado, fuente alpha_vantage_daily, fecha del proveedor) ->
solo tras "si" se registra un MarketEvent paper. La importacion nunca crea
ni ejecuta ordenes; con LIMIT pendientes se rechaza. Todas las respuestas
HTTP son falsas (httpx.MockTransport): nunca se usa red ni una API key real.
"""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

import httpx
import pytest

from agents.finance_agent import FinanceAgent
from core.atlas import Atlas
from core.model_inference import ModelInferenceRunner
from finance.paper.models import OrderStatus, PaperOrder, OrderType, Side
from finance.paper.service import PaperFinanceService
from finance.paper.store import PaperStore
from tools.alpha_vantage import AlphaVantageClient
from use_cases.paper_finance_chat import IMPORT_SOURCE, PaperFinanceChat

FAKE_KEY = "test-fake-key-not-real"

IBM_DAILY_PAYLOAD = {
    "Meta Data": {
        "1. Information": "Daily Prices (open, high, low, close, volume)",
        "2. Symbol": "IBM",
        "3. Last Refreshed": "2026-09-11",
        "4. Time Zone": "US/Eastern",
    },
    "Time Series (Daily)": {
        "2026-09-11": {
            "1. open": "240.00",
            "2. high": "242.10",
            "3. low": "239.20",
            "4. close": "241.55",
            "5. volume": "3100000",
        },
        "2026-09-10": {
            "1. open": "238.00",
            "2. high": "241.00",
            "3. low": "237.50",
            "4. close": "239.80",
            "5. volume": "2950000",
        },
    },
}

EMPTY_SERIES_PAYLOAD = {
    "Meta Data": {
        "2. Symbol": "IBM",
        "3. Last Refreshed": "2026-09-11",
        "4. Time Zone": "US/Eastern",
    },
    "Time Series (Daily)": {},
}

PROVIDER_ERROR_PAYLOAD = {"Error Message": "Invalid API call."}


def _json_handler(payload):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=json.dumps(payload))

    return handler


def _network_error_handler():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    return handler


def _client(handler) -> AlphaVantageClient:
    return AlphaVantageClient(
        FAKE_KEY,
        transport=httpx.MockTransport(handler),
    )


def _chat(tmp_path: Path, client: AlphaVantageClient | None = None) -> PaperFinanceChat:
    return PaperFinanceChat(
        PaperFinanceService(PaperStore(tmp_path / "finance_paper")),
        market_client=client,
    )


def _last_event_entry(chat: PaperFinanceChat) -> dict:
    entries = list(chat.service.engine.ledger)
    return next(entry for entry in reversed(entries) if entry["kind"] == "EVENT")


def _summary(chat: PaperFinanceChat) -> dict:
    return chat.service.portfolio_summary()


# ---------------------------------------------------------------------------
# Propuesta y confirmacion
# ---------------------------------------------------------------------------


def test_import_proposal_shows_delayed_source_without_modifying(tmp_path) -> None:
    chat = _chat(tmp_path, _client(_json_handler(IBM_DAILY_PAYLOAD)))

    assert chat.handles("importa precio mercado IBM a paper")
    assert chat.handles("actualiza precio paper IBM desde mercado")

    proposal = chat.handle("importa precio mercado IBM a paper")

    assert proposal.startswith("[PAPER]")
    assert "IBM" in proposal
    assert "241,5500" in proposal
    assert "2026-09-11" in proposal
    assert "alpha_vantage_daily" in proposal
    assert "retrasado" in proposal.casefold()
    assert "no es tiempo real" in proposal.casefold()
    assert "no crea ni ejecuta ninguna orden" in proposal.casefold()
    assert "no se ha modificado" in proposal.casefold()
    pending = chat.pending_import
    assert pending is not None
    assert pending.symbol == "IBM"
    assert pending.price == Decimal("241.55")
    assert pending.provider_date.isoformat() == "2026-09-11"
    assert chat.service.engine.last_prices() == {}
    assert _summary(chat)["cash"] == "10000.00"
    assert _summary(chat)["positions"] == {}


def test_import_confirmation_records_alpha_vantage_tick(tmp_path) -> None:
    chat = _chat(tmp_path, _client(_json_handler(IBM_DAILY_PAYLOAD)))
    chat.handle("importa precio mercado IBM a paper")

    confirmation = chat.execute_pending()

    casefolded = confirmation.casefold()
    assert confirmation.startswith("[PAPER]")
    assert "tick paper importado" in casefolded
    assert "alpha_vantage_daily" in confirmation
    assert "2026-09-11" in confirmation
    assert "241,5500" in confirmation
    assert "retrasado" in casefolded
    assert "no es tiempo real" in casefolded
    assert "no crea ni ejecuta ordenes" in casefolded
    assert chat.service.engine.last_prices()["IBM"] == Decimal("241.55")
    entry = _last_event_entry(chat)
    assert entry["payload"]["source"] == IMPORT_SOURCE
    assert entry["payload"]["provider_date"] == "2026-09-11"
    assert chat.service.fills() == ()
    assert chat.service.pending_orders() == ()
    assert _summary(chat)["cash"] == "10000.00"
    assert _summary(chat)["positions"] == {}


def test_import_rejection_keeps_state_intact(tmp_path) -> None:
    chat = _chat(tmp_path, _client(_json_handler(IBM_DAILY_PAYLOAD)))
    chat.handle("actualiza precio paper IBM desde mercado")
    assert chat.pending_import is not None

    rejection = chat.cancel_pending()

    assert "descartada" in rejection.casefold()
    assert chat.pending_import is None
    assert chat.service.engine.last_prices() == {}
    assert _summary(chat)["cash"] == "10000.00"
    assert _summary(chat)["positions"] == {}

    late_confirmation = chat.execute_pending()
    assert "no se ha modificado" in late_confirmation.casefold()
    assert chat.service.engine.last_prices() == {}


def test_order_after_import_requires_second_confirmation(tmp_path) -> None:
    chat = _chat(tmp_path, _client(_json_handler(IBM_DAILY_PAYLOAD)))
    chat.handle("importa precio mercado IBM a paper")
    chat.execute_pending()
    assert chat.service.engine.last_prices()["IBM"] == Decimal("241.55")

    proposal = chat.handle("compra paper 1 de IBM a mercado")

    assert proposal.startswith("[PAPER]")
    assert "Propuesta de orden paper" in proposal
    assert chat.pending_proposal is not None
    assert chat.service.fills() == ()
    assert _summary(chat)["cash"] == "10000.00"
    assert _summary(chat)["positions"] == {}

    confirmation = chat.execute_pending()

    assert "FILLED" in confirmation
    assert _summary(chat)["positions"]["IBM"]["qty"] == "1"


# ---------------------------------------------------------------------------
# LIMIT pendientes bloquean la importacion
# ---------------------------------------------------------------------------


def test_import_rejected_while_limit_order_pending(tmp_path) -> None:
    chat = _chat(tmp_path, _client(_json_handler(IBM_DAILY_PAYLOAD)))
    chat.handle("registra precio paper IBM 100")
    chat.handle("compra paper 5 de IBM con limite 90")
    chat.execute_pending()
    pending_orders = chat.service.pending_orders()
    assert len(pending_orders) == 1

    response = chat.handle("importa precio mercado IBM a paper")

    casefolded = response.casefold()
    assert response.startswith("[PAPER]")
    assert "rechazada" in casefolded
    assert "limit" in casefolded
    assert "fills indirectos" in casefolded
    assert "cancelalas" in casefolded
    assert chat.pending_import is None
    assert chat.service.engine.last_prices()["IBM"] == Decimal("100")
    still_pending = chat.service.pending_orders()
    assert len(still_pending) == 1
    decision = chat.service.engine.decision_for(still_pending[0].order_id)
    assert decision is not None
    assert decision.status is OrderStatus.PENDING
    assert chat.service.fills() == ()


def test_import_blocked_again_at_confirmation_time(tmp_path) -> None:
    chat = _chat(tmp_path, _client(_json_handler(IBM_DAILY_PAYLOAD)))
    chat.handle("importa precio mercado IBM a paper")
    assert chat.pending_import is not None

    chat.service.execute_confirmed_order(
        PaperOrder(
            symbol="IBM",
            side=Side.BUY,
            qty=Decimal("5"),
            order_type=OrderType.LIMIT,
            limit_price=Decimal("90"),
        )
    )

    response = chat.execute_pending()

    casefolded = response.casefold()
    assert "no realizada" in casefolded
    assert "fills indirectos" in casefolded
    assert chat.service.engine.last_prices() == {}
    assert len(chat.service.pending_orders()) == 1
    assert chat.service.fills() == ()
    assert _summary(chat)["cash"] == "10000.00"


# ---------------------------------------------------------------------------
# Errores seguros sin modificar PAPER
# ---------------------------------------------------------------------------


def test_import_errors_are_safe_without_modification(tmp_path) -> None:
    missing_key = AlphaVantageClient("")
    network_failure = _client(_network_error_handler())
    empty_series = _client(_json_handler(EMPTY_SERIES_PAYLOAD))
    provider_error = _client(_json_handler(PROVIDER_ERROR_PAYLOAD))

    for client, expected_fragment in (
        (missing_key, "alphavantage_api_key"),
        (network_failure, "error de red"),
        (empty_series, "no contiene un cierre diario valido"),
        (provider_error, "no reconoce el simbolo"),
    ):
        chat = _chat(tmp_path / expected_fragment[:8], client)
        response = chat.handle("importa precio mercado IBM a paper")
        casefolded = response.casefold()
        assert response.startswith("[PAPER]")
        assert expected_fragment in casefolded
        assert "no se ha modificado" in casefolded
        assert chat.pending_import is None
        assert chat.service.engine.last_prices() == {}
        assert _summary(chat)["cash"] == "10000.00"
        assert _summary(chat)["positions"] == {}


def test_import_without_market_client_reports_unavailable(tmp_path) -> None:
    chat = _chat(tmp_path)

    response = chat.handle("importa precio mercado IBM a paper")

    casefolded = response.casefold()
    assert response.startswith("[PAPER]")
    assert "no disponible" in casefolded
    assert "no se ha modificado" in casefolded
    assert chat.pending_import is None
    assert chat.service.engine.last_prices() == {}


def test_import_commands_are_explicit_and_isolated(tmp_path) -> None:
    chat = _chat(tmp_path, _client(_json_handler(IBM_DAILY_PAYLOAD)))

    assert not chat.handles("datos mercado IBM")

    portfolio = chat.handle("cartera paper")
    assert portfolio.startswith("[PAPER]")
    assert chat.pending_import is None

    order = chat.handle("compra paper 1 de IBM a mercado")
    assert "falta un precio paper para ibm" in order.casefold()
    assert chat.pending_import is None

    multiline = chat.handle(
        "importa precio mercado IBM a paper\ncompra paper 1 de IBM",
    )
    assert multiline.startswith("[PAPER]")
    assert "varias lineas" in multiline.casefold()
    assert chat.pending_import is None


# ---------------------------------------------------------------------------
# Camino real de la UI grafica: Atlas.process_prompt -> orquestador.
# ---------------------------------------------------------------------------


def _forbid_llm(monkeypatch) -> None:
    def forbidden(self, *args, **kwargs):
        raise AssertionError("El LLM no debe invocarse en el camino paper")

    monkeypatch.setattr(ModelInferenceRunner, "run", forbidden)


@pytest.fixture()
def atlas(tmp_path: Path, monkeypatch):
    atlas = Atlas()
    finance = atlas._orchestrator._registry.get("finance")
    assert isinstance(finance, FinanceAgent)
    finance.paper_chat = PaperFinanceChat(
        PaperFinanceService(PaperStore(tmp_path / "finance_paper")),
        market_client=_client(_json_handler(IBM_DAILY_PAYLOAD)),
    )
    _forbid_llm(monkeypatch)
    return atlas


def test_atlas_ui_chat_import_flow_proposal_then_confirmation(atlas) -> None:
    finance = atlas._orchestrator._registry.get("finance")
    chat = finance.paper_chat
    assert chat is not None

    proposal = atlas.process_prompt("importa precio mercado IBM a paper")

    assert proposal.startswith("[PAPER]")
    assert "alpha_vantage_daily" in proposal
    assert "retrasado" in proposal.casefold()
    assert chat.pending_import is not None
    assert chat.service.engine.last_prices() == {}

    confirmation = atlas.process_prompt("sí")

    assert "tick paper importado" in confirmation.casefold()
    assert "alpha_vantage_daily" in confirmation
    assert chat.service.engine.last_prices()["IBM"] == Decimal("241.55")
    assert chat.pending_import is None

    order_proposal = atlas.process_prompt("compra paper 1 de IBM a mercado")
    assert "Propuesta de orden paper" in order_proposal
    assert chat.service.fills() == ()

    order_confirmation = atlas.process_prompt("sí")
    assert "FILLED" in order_confirmation
