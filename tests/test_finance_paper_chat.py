"""Tests focalizados de Atlas Finance V2.2: chat paper workflow sobre V2.1.

Cubre el flujo conversacional completo: consulta de cartera paper,
registro de precio paper user_declared, propuesta de orden paper con
confirmacion/rechazo/expiracion y el rechazo de ordenes MARKET sin tick.
Reutiliza PaperFinanceService; no replica motor ni persistencia.
"""

from __future__ import annotations

from datetime import timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from agents.finance_agent import FinanceAgent
from bootstrap.bootstrap import Bootstrap
from core.atlas import Atlas
from core.model_inference import ModelInferenceRunner
from finance.paper.service import PaperFinanceService
from finance.paper.store import PaperStore
from use_cases.paper_finance_chat import PENDING_ORDER_TTL, PaperFinanceChat


def _noop_confirm(_prompt: str) -> str:
    return ""


@pytest.fixture()
def orchestrator(tmp_path: Path):
    orchestrator = Bootstrap.build()
    finance = orchestrator._registry.get("finance")
    assert isinstance(finance, FinanceAgent)
    store = PaperStore(tmp_path / "finance_paper")
    finance.paper_chat = PaperFinanceChat(PaperFinanceService(store))
    return orchestrator


def _finance(orchestrator) -> FinanceAgent:
    finance = orchestrator._registry.get("finance")
    assert isinstance(finance, FinanceAgent)
    return finance


def test_paper_portfolio_query_shows_paper_summary_without_llm(orchestrator, monkeypatch) -> None:
    finance = _finance(orchestrator)
    llm_calls: list[list[dict[str, str]]] = []

    def fail_ask(*, model: str, messages: list[dict[str, str]]) -> str:
        llm_calls.append(messages)
        return "analysis"

    monkeypatch.setattr(finance._client, "ask", fail_ask)

    response = orchestrator.process_prompt(
        "Muéstrame la cartera paper",
        confirm=_noop_confirm,
    )

    assert llm_calls == []
    assert "[PAPER]" in response
    assert "simulacion" in response.casefold()
    assert "Efectivo: 10.000,00 €" in response
    assert "Posiciones: ninguna" in response


def test_paper_price_declaration_is_recorded_with_user_declared_origin(orchestrator) -> None:
    finance = _finance(orchestrator)

    response = orchestrator.process_prompt(
        "Registra precio paper AAPL 185.50",
        confirm=_noop_confirm,
    )

    assert "user_declared" in response
    assert "185,5000" in response
    assert "[PAPER]" in response
    chat = finance.paper_chat
    assert chat is not None
    assert chat.service.engine.last_prices()["AAPL"] == Decimal("185.50")
    summary = chat.service.portfolio_summary()
    assert summary["nav"] == "10000.00"


def test_paper_order_proposal_then_confirmation_fills_and_modifies_portfolio(
    orchestrator,
) -> None:
    finance = _finance(orchestrator)
    chat = finance.paper_chat
    assert chat is not None

    orchestrator.process_prompt("Registra precio paper AAPL 100", confirm=_noop_confirm)

    proposal = orchestrator.process_prompt(
        "Compra paper 10 de AAPL a mercado",
        confirm=_noop_confirm,
    )

    assert "Propuesta de orden paper" in proposal
    assert "COMPRA" in proposal
    assert "AAPL" in proposal
    assert "no ejecutada todavia" in proposal.casefold()
    assert chat.pending_proposal is not None

    before = orchestrator.process_prompt("cartera paper", confirm=_noop_confirm)
    assert "Efectivo: 10.000,00 €" in before

    confirmation = orchestrator.process_prompt("sí", confirm=_noop_confirm)

    assert "FILLED" in confirmation
    assert chat.pending_proposal is None

    after = orchestrator.process_prompt("cartera paper", confirm=_noop_confirm)
    assert "Efectivo: 9.000,00 €" in after
    assert "AAPL: 10 uds" in after


def test_paper_order_rejected_on_no_leaves_portfolio_intact(orchestrator) -> None:
    finance = _finance(orchestrator)
    chat = finance.paper_chat
    assert chat is not None

    orchestrator.process_prompt("Registra precio paper AAPL 100", confirm=_noop_confirm)
    orchestrator.process_prompt("Compra paper 10 de AAPL a mercado", confirm=_noop_confirm)
    assert chat.pending_proposal is not None

    rejection = orchestrator.process_prompt("no", confirm=_noop_confirm)

    assert "descartada" in rejection.casefold()
    assert chat.pending_proposal is None

    late_confirmation = orchestrator.process_prompt("sí", confirm=_noop_confirm)
    assert "no se ha modificado" in late_confirmation.casefold()

    summary = chat.service.portfolio_summary()
    assert summary["cash"] == "10000.00"
    assert summary["positions"] == {}


def test_market_order_without_tick_is_rejected_without_modification(orchestrator) -> None:
    finance = _finance(orchestrator)
    chat = finance.paper_chat
    assert chat is not None

    response = orchestrator.process_prompt(
        "Compra paper 5 de MSFT a mercado",
        confirm=_noop_confirm,
    )

    casefolded = response.casefold()
    assert "[PAPER]" in response
    assert "rechazada" in casefolded
    assert "no invento precios" in casefolded
    assert "registra precio paper" in casefolded
    assert chat.pending_proposal is None
    summary = chat.service.portfolio_summary()
    assert summary["cash"] == "10000.00"
    assert summary["positions"] == {}


def test_pending_paper_order_expires_without_modification(orchestrator) -> None:
    finance = _finance(orchestrator)
    chat = finance.paper_chat
    assert chat is not None

    orchestrator.process_prompt("Registra precio paper AAPL 100", confirm=_noop_confirm)
    orchestrator.process_prompt("Compra paper 10 de AAPL a mercado", confirm=_noop_confirm)
    pending = chat.pending_proposal
    assert pending is not None

    expired = type(pending)(
        order=pending.order,
        text=pending.text,
        created_at=pending.created_at - PENDING_ORDER_TTL - timedelta(seconds=1),
    )
    chat._pending = expired

    response = orchestrator.process_prompt("sí", confirm=_noop_confirm)

    casefolded = response.casefold()
    assert "pendiente" in casefolded or "expir" in casefolded
    assert "no se ha modificado" in casefolded
    assert chat.pending_proposal is None
    summary = chat.service.portfolio_summary()
    assert summary["cash"] == "10000.00"
    assert summary["positions"] == {}


def test_limit_paper_order_proposal_mentions_pending_engine_state(orchestrator) -> None:
    finance = _finance(orchestrator)
    chat = finance.paper_chat
    assert chat is not None

    orchestrator.process_prompt("Registra precio paper AAPL 100", confirm=_noop_confirm)

    proposal = orchestrator.process_prompt(
        "Compra paper 10 de AAPL con limite 90",
        confirm=_noop_confirm,
    )

    assert "LIMIT" in proposal
    assert chat.pending_proposal is not None

    confirmation = orchestrator.process_prompt("sí", confirm=_noop_confirm)
    assert "PENDING" in confirmation
    assert chat.service.pending_orders() != ()


def test_finance_agent_exposes_optional_paper_chat_without_finance_paper_import() -> None:
    agent = FinanceAgent(None)  # type: ignore[arg-type]
    assert agent.paper_chat is None
    source = Path("agents/finance_agent.py").read_text(encoding="utf-8").casefold()
    assert "finance.paper" not in source
    assert "no ejecutes compras" in source


# ---------------------------------------------------------------------------
# Camino real de la UI grafica: TranscriptPanel.send_requested ->
# OrbeController.submit_text -> Atlas.process_prompt -> orquestador.
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
    store = PaperStore(tmp_path / "finance_paper")
    finance.paper_chat = PaperFinanceChat(PaperFinanceService(store))
    _forbid_llm(monkeypatch)
    return atlas


def _paper_chat(atlas) -> PaperFinanceChat:
    chat = atlas._orchestrator._registry.get("finance").paper_chat
    assert chat is not None
    return chat


def test_atlas_ui_chat_cartera_paper_returns_deterministic_paper(atlas) -> None:
    response = atlas.process_prompt("cartera paper")

    assert response.startswith("[PAPER]")
    assert "Efectivo: 10.000,00 €" in response
    assert "Posiciones: ninguna" in response


def test_atlas_ui_chat_order_symbol_first_without_tick_is_rejected(atlas) -> None:
    chat = _paper_chat(atlas)

    response = atlas.process_prompt("compra paper ABC 1 mercado")

    casefolded = response.casefold()
    assert response.startswith("[PAPER]")
    assert "falta un precio paper para abc" in casefolded
    assert "no invento precios" in casefolded
    assert chat.pending_proposal is None
    summary = chat.service.portfolio_summary()
    assert summary["cash"] == "10000.00"
    assert summary["positions"] == {}


def test_atlas_ui_chat_multiline_paper_commands_are_not_simulated(atlas) -> None:
    chat = _paper_chat(atlas)

    response = atlas.process_prompt(
        "compra paper 10 de AAPL a mercado\nvende paper 5 de AAPL",
    )

    assert response.startswith("[PAPER]")
    assert "un comando paper por mensaje" in response
    assert chat.pending_proposal is None
    summary = chat.service.portfolio_summary()
    assert summary["cash"] == "10000.00"
    assert summary["positions"] == {}


def test_atlas_ui_chat_order_proposal_and_confirmation_fill(atlas) -> None:
    chat = _paper_chat(atlas)

    atlas.process_prompt("registra precio paper AAPL 100")
    proposal = atlas.process_prompt("compra paper 10 de AAPL a mercado")

    assert proposal.startswith("[PAPER]")
    assert chat.pending_proposal is not None

    confirmation = atlas.process_prompt("sí")

    assert "FILLED" in confirmation
    assert chat.pending_proposal is None

    summary = atlas.process_prompt("cartera paper")
    assert "Efectivo: 9.000,00 €" in summary
    assert "AAPL: 10 uds" in summary


def test_atlas_ui_chat_short_price_tick_then_order_then_confirmation_fills(
    atlas,
) -> None:
    chat = _paper_chat(atlas)

    tick = atlas.process_prompt("precio paper TEST 100")

    assert tick.startswith("[PAPER]")
    assert "tick paper registrado" in tick.casefold()
    assert "user_declared" in tick

    proposal = atlas.process_prompt("compra paper TEST 1 mercado")

    assert proposal.startswith("[PAPER]")
    assert "Propuesta de orden paper" in proposal
    assert "no ejecutada todavia" in proposal.casefold()
    assert chat.pending_proposal is not None

    confirmation = atlas.process_prompt("sí")

    assert "FILLED" in confirmation
    assert chat.pending_proposal is None

    summary = chat.service.portfolio_summary()
    assert summary["positions"]["TEST"]["qty"] == "1"


def test_atlas_ui_chat_long_price_form_still_records_the_tick(atlas) -> None:
    chat = _paper_chat(atlas)

    tick = atlas.process_prompt("registra precio paper TEST 100")

    assert tick.startswith("[PAPER]")
    assert "user_declared" in tick
    assert chat.service.engine.last_prices()["TEST"] == Decimal("100")


def test_atlas_ui_chat_order_rejection_leaves_portfolio_intact(atlas) -> None:
    chat = _paper_chat(atlas)

    atlas.process_prompt("registra precio paper AAPL 100")
    atlas.process_prompt("compra paper 10 de AAPL a mercado")
    assert chat.pending_proposal is not None

    rejection = atlas.process_prompt("no")

    assert "descartada" in rejection.casefold()
    summary = chat.service.portfolio_summary()
    assert summary["cash"] == "10000.00"
    assert summary["positions"] == {}


def test_atlas_ui_chat_unmatched_paper_command_gets_deterministic_guidance(atlas) -> None:
    chat = _paper_chat(atlas)

    response = atlas.process_prompt("vende paper")

    assert response.startswith("[PAPER]")
    assert "comandos" in response.casefold()
    summary = chat.service.portfolio_summary()
    assert summary["cash"] == "10000.00"
    assert summary["positions"] == {}
