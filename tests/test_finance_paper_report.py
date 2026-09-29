"""Pruebas focalizadas del informe paper determinista y de solo lectura."""

from __future__ import annotations

import json
from decimal import Decimal

import httpx

from agents.finance_agent import FinanceAgent
from core.atlas import Atlas
from finance.paper.service import PaperFinanceService
from finance.paper.store import PaperStore
from tools.alpha_vantage import AlphaVantageClient
from use_cases.paper_finance_chat import PaperFinanceChat


DAILY = {
    "Meta Data": {
        "2. Symbol": "IBM",
        "3. Last Refreshed": "2026-09-11",
        "4. Time Zone": "US/Eastern",
    },
    "Time Series (Daily)": {
        "2026-09-11": {
            "1. open": "240",
            "2. high": "242",
            "3. low": "239",
            "4. close": "241.55",
            "5. volume": "1000",
        }
    },
}


def _client() -> AlphaVantageClient:
    def handler(request: httpx.Request) -> httpx.Response:
        payload = (
            {
                "bestMatches": [
                    {
                        "1. symbol": "IBM",
                        "2. name": "International Business Machines",
                        "3. type": "Equity",
                        "4. region": "United States",
                        "8. currency": "USD",
                    }
                ]
            }
            if request.url.params["function"] == "SYMBOL_SEARCH"
            else DAILY
        )
        return httpx.Response(200, text=json.dumps(payload))

    return AlphaVantageClient("test-key", transport=httpx.MockTransport(handler))


def test_report_with_provider_contains_data_costs_risks_and_does_not_mutate(tmp_path):
    service = PaperFinanceService(PaperStore(tmp_path / "paper"))
    chat = PaperFinanceChat(service, market_client=_client())

    report = chat.handle("informe paper IBM")

    assert "Activo solicitado: IBM" in report
    assert "241,5500" in report
    assert "Moneda: USD" in report
    assert "alpha_vantage_daily" in report
    assert "Hora/fecha del proveedor: 2026-09-11" in report
    assert "COSTES Y SUPUESTOS" in report
    assert "RIESGOS Y LIMITES" in report
    assert "decision" in report.casefold()
    assert service.engine.last_prices() == {}
    assert service.fills() == ()
    assert service.pending_orders() == ()


def test_report_without_provider_is_explicit_about_missing_data(tmp_path):
    service = PaperFinanceService(
        PaperStore(tmp_path / "paper"),
        default_commission=Decimal("1.25"),
    )
    chat = PaperFinanceChat(service)

    report = chat.handle("informe paper BTC")

    assert "Precio: NO DISPONIBLE" in report
    assert "no hay proveedor" in report.casefold()
    assert "Moneda: NO VERIFICADA" in report
    assert "1,25 €" in report
    assert "COSTES Y SUPUESTOS" in report
    assert "RIESGOS Y LIMITES" in report
    assert "decision" in report.casefold()
    assert "no se ha modificado la cartera" in report.casefold()
    assert service.engine.last_prices() == {}


def test_report_from_atlas_entry_is_read_only_with_simulated_provider(tmp_path, monkeypatch):
    def forbidden_llm(*args, **kwargs):
        raise AssertionError("El informe paper no debe invocar el LLM")

    monkeypatch.setattr("core.model_inference.ModelInferenceRunner.run", forbidden_llm)
    atlas = Atlas()
    finance = atlas._orchestrator._registry.get("finance")
    assert isinstance(finance, FinanceAgent)
    service = PaperFinanceService(PaperStore(tmp_path / "paper"))
    chat = PaperFinanceChat(service, market_client=_client())
    finance.paper_chat = chat

    report = atlas.process_prompt("informe paper IBM")

    assert report.startswith("[PAPER]")
    assert "COSTES Y SUPUESTOS" in report
    assert "RIESGOS Y LIMITES" in report
    assert "decision" in report.casefold()
    assert service.engine.last_prices() == {}
    assert service.ledger() == ()
    assert chat.pending_proposal is None
    assert service.pending_orders() == ()
    assert service.fills() == ()
    summary = service.portfolio_summary()
    assert summary["cash"] == "10000.00"
    assert summary["positions"] == {}
