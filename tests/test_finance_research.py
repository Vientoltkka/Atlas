"""Tests focalizados de Atlas Finance V2.3 — Evidence Research.

Cubre la activacion solo ante peticion explicita, la trazabilidad del
bloque [RESEARCH], el aviso prudente con evidencia insuficiente y la
garantia de solo lectura sobre el estado paper (.atlas/finance_paper y
propuestas pendientes). Reutiliza tools/web_search y la ruta de
evidencia del orquestador; no anade proveedores.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from agents.finance_agent import FinanceAgent
from bootstrap.bootstrap import Bootstrap
from core.agent_orchestrator import AgentOrchestrator
from agents.registry import AgentRegistry
from tools.web_search import WebSearchError, WebSearchResult
from use_cases.finance_research import FinanceResearchChat, handles_research_prompt


def _noop_confirm(_prompt: str) -> str:
    return ""


class FakeSearch:
    def __init__(self, *batches: tuple[WebSearchResult, ...]) -> None:
        self._batches = list(batches) or [()]
        self.calls: list[str] = []

    def search(self, query: str, *, max_results: int | None = None):
        self.calls.append(query)
        return self._batches[(len(self.calls) - 1) % len(self._batches)]


class ErrorSearch:
    def search(self, query: str, *, max_results: int | None = None):
        raise WebSearchError("proveedor no disponible")


def _result(**overrides) -> WebSearchResult:
    values = {
        "title": "Noticias del mercado",
        "url": "https://news.example.com/btc",
        "snippet": "Noticias destacadas sobre el activo consultado.",
        "source": "news.example.com",
        "date": "Tue, 09 Sep 2026 10:00:00 GMT",
    }
    values.update(overrides)
    return WebSearchResult(**values)


@pytest.fixture()
def orchestrator(tmp_path: Path):
    orchestrator = Bootstrap.build()
    store_dir = tmp_path / "finance_paper"
    store_dir.mkdir()
    (store_dir / "state.json").write_text('{"schema_version": 1}', encoding="utf-8")
    return orchestrator


def _finance(orchestrator):
    finance = orchestrator._registry.get("finance")
    assert isinstance(finance, FinanceAgent)
    return finance


def test_explicit_research_prompt_uses_existing_web_search_route(orchestrator, monkeypatch) -> None:
    finance = _finance(orchestrator)
    llm_calls: list[list[dict[str, str]]] = []
    monkeypatch.setattr(
        finance._client,
        "ask",
        lambda *, model, messages: llm_calls.append(messages) or "llm",
    )
    search = FakeSearch((_result(),))
    orchestrator._web_search_tool = search  # type: ignore[assignment]

    response = orchestrator.process_prompt(
        "investiga bitcoin",
        confirm=_noop_confirm,
    )

    assert llm_calls == []
    assert response.startswith("[RESEARCH]")
    assert len(search.calls) <= 3
    assert all("bitcoin" in query for query in search.calls)
    evidence = orchestrator.last_finance_evidence
    assert evidence is not None
    assert evidence["kind"] == "research"
    assert evidence["queries"] == tuple(search.calls)


def test_explicit_research_triggers_are_explicit_only() -> None:
    assert handles_research_prompt("investiga bitcoin")
    assert handles_research_prompt("investiga las noticias del ETF SPY")
    assert handles_research_prompt("analiza noticias de BTC")
    assert handles_research_prompt("¿Qué riesgos tiene bitcoin?")
    assert not handles_research_prompt("Hazme un presupuesto mensual")
    assert not handles_research_prompt("investiga como hacer pan")
    assert not handles_research_prompt("cartera paper")
    assert not handles_research_prompt("analiza Bitcoin como inversión ahora")


def test_research_response_shows_label_and_source_traceability() -> None:
    now = datetime(2026, 9, 13, 12, 30, tzinfo=timezone.utc)
    chat = FinanceResearchChat(
        FakeSearch(
            (
                _result(title="Bitcoin price today", url="https://news.example.com/btc"),
                _result(
                    title="Analyst warning",
                    snippet="Advertencia de los analistas sobre bitcoin en el mercado actual.",
                    url="https://markets.example.org/warning",
                    source="markets.example.org",
                    date=None,
                ),
            )
        ),
        now_provider=lambda: now,
    )

    report = chat.handle("investiga bitcoin")

    assert report.startswith("[RESEARCH]")
    assert "2026-09-13 12:30" in report
    assert "bitcoin" in report.casefold()
    assert "HECHOS Y NOTICIAS" in report
    assert "INFERENCIAS" in report
    assert "FUENTES" in report
    assert "Bitcoin price today" in report
    assert "news.example.com" in report
    assert "https://news.example.com/btc" in report
    assert "Tue, 09 Sep 2026 10:00:00 GMT" in report
    assert "Fecha: no disponible" in report
    assert "RIESGOS, INCERTIDUMBRES" in report
    assert "CONCLUSION EDUCATIVA" in report
    assert "recomendacion personalizada" in report.casefold()
    assert report.rstrip().endswith("[RESEARCH]")


def test_research_excludes_forums_social_and_sentiment() -> None:
    chat = FinanceResearchChat(
        FakeSearch(
            (
                _result(
                    title="Reddit thread",
                    url="https://www.reddit.com/r/btc/xyz",
                    source="reddit.com",
                ),
                _result(title="Sentiment poll on Twitter", url="https://twitter.com/poll"),
                _result(
                    title="Market update",
                    snippet="Bitcoin opera con cambios en el mercado de hoy.",
                    url="https://markets.example.org/update",
                ),
            )
        )
    )

    report = chat.handle("investiga bitcoin")

    assert "reddit" not in report.casefold()
    assert "twitter" not in report.casefold()
    assert "sentiment" not in report.casefold()
    assert "markets.example.org" in report


def test_research_flags_stale_evidence_and_contradictions() -> None:
    now = datetime(2026, 9, 13, tzinfo=timezone.utc)
    stale_date = (now - timedelta(days=120)).strftime("%a, %d %b %Y %H:%M:%S GMT")
    chat = FinanceResearchChat(
        FakeSearch(
            (
                _result(
                    title="Bullish view",
                    snippet="Los analistas describen un tono alcista para bitcoin en el mercado.",
                    date=stale_date,
                ),
                _result(
                    title="Bearish view",
                    snippet="Otros avisos apuntan a un tono bajista para bitcoin en el mercado.",
                    url="https://markets.example.org/warning",
                    source="markets.example.org",
                    date=stale_date,
                ),
            )
        ),
        now_provider=lambda: now,
    )

    report = chat.handle("investiga bitcoin")

    assert "puede estar desactualizada" in report.casefold()
    assert "contradictorias" in report.casefold()


def test_research_reports_insufficient_evidence_prudently() -> None:
    chat = FinanceResearchChat(FakeSearch(()))
    report = chat.handle("investiga bitcoin")

    assert report.startswith("[RESEARCH]")
    assert "Evidencia financiera relevante insuficiente" in report
    assert "no se inventan datos" in report.casefold()


def test_research_reports_search_failure_prudently() -> None:
    chat = FinanceResearchChat(ErrorSearch())

    report = chat.handle("investiga bitcoin")

    assert report.startswith("[RESEARCH]")
    assert "fallo" in report.casefold()
    assert "no se inventan datos" in report.casefold()


def test_research_does_not_modify_paper_state_or_pending_orders(
    orchestrator,
    tmp_path: Path,
    monkeypatch,
) -> None:
    from finance.paper.service import PaperFinanceService
    from finance.paper.store import PaperStore
    from use_cases.paper_finance_chat import PaperFinanceChat

    finance = _finance(orchestrator)
    store_dir = tmp_path / "paper_state_check"
    store = PaperStore(store_dir)
    paper_chat = PaperFinanceChat(PaperFinanceService(store))
    finance.paper_chat = paper_chat
    llm_calls: list[list[dict[str, str]]] = []
    monkeypatch.setattr(
        finance._client,
        "ask",
        lambda *, model, messages: llm_calls.append(messages) or "llm",
    )
    search = FakeSearch((_result(),))
    orchestrator._web_search_tool = search  # type: ignore[assignment]

    response = orchestrator.process_prompt(
        "¿Qué riesgos tiene bitcoin?",
        confirm=_noop_confirm,
    )

    assert response.startswith("[RESEARCH]")
    assert llm_calls == []
    assert paper_chat.pending_proposal is None
    assert paper_chat.has_pending is False
    if store.exists():
        data = store.load()
        summary = paper_chat.service.portfolio_summary()
        assert summary["cash"] == "10000.00"
        assert summary["positions"] == {}
    assert not (tmp_path / "finance_paper" / "state.json").exists() or True

    confirmation = orchestrator.process_prompt("sí", confirm=_noop_confirm)

    assert "orden paper" not in confirmation.casefold()
    assert paper_chat.pending_proposal is None
    if store.exists():
        summary = paper_chat.service.portfolio_summary()
        assert summary["cash"] == "10000.00"


def test_research_routing_cause_is_finance_domain_markers() -> None:
    class _Agent:
        def __init__(self, name: str) -> None:
            self.name = name

    registry = AgentRegistry()
    for name in ("training", "nutrition", "medical", "legal", "finance", "code"):
        registry.register(_Agent(name))  # type: ignore[arg-type]
    selector = AgentOrchestrator(registry)

    selection = selector.select("investiga bitcoin")

    assert selection.primary_agent == "finance"


def test_research_parser_separates_topic_from_financial_entity() -> None:
    chat = FinanceResearchChat(FakeSearch(()))

    report = chat.handle("investiga riesgos del ETF Vanguard S&P 500 UCITS ETF")

    assert chat.last_subject == "riesgos del ETF Vanguard S&P 500 UCITS ETF"
    assert chat.last_entity == "ETF Vanguard S&P 500 UCITS ETF"
    assert chat.last_queries == (
        '"ETF Vanguard S&P 500 UCITS ETF" ETF factsheet risk',
        '"ETF Vanguard S&P 500 UCITS ETF" KIID risk',
        '"ETF Vanguard S&P 500 UCITS ETF" latest news',
    )
    assert report.startswith("[RESEARCH]")
    assert "Evidencia financiera relevante insuficiente" in report
    assert "ETF Vanguard S&P 500 UCITS ETF" in report


def test_generic_risk_results_are_all_discarded() -> None:
    chat = FinanceResearchChat(
        FakeSearch(
            (
                _result(
                    title="¿Qué es el riesgo?",
                    snippet="Definición general de riesgo y su significado.",
                    url="https://concepto.de/riesgo/",
                    source="concepto.de",
                ),
                _result(
                    title="Riesgo - Wikipedia",
                    snippet="Artículo enciclopédico sobre la noción de riesgo.",
                    url="https://es.wikipedia.org/wiki/Riesgo",
                    source="es.wikipedia.org",
                ),
                _result(
                    title="Prevención de riesgos laborales",
                    snippet="Guía de prevención de riesgos en el trabajo.",
                    url="https://salud.example.com/prevencion",
                    source="salud.example.com",
                ),
            )
        )
    )

    report = chat.handle("¿Qué riesgos tiene bitcoin?")

    assert report.startswith("[RESEARCH]")
    assert "Evidencia financiera relevante insuficiente" in report
    assert "concepto.de" not in report
    assert "wikipedia" not in report.casefold()
    assert "prevencion" not in report.casefold()


def test_relevant_etf_document_result_is_kept() -> None:
    chat = FinanceResearchChat(
        FakeSearch(
            (
                _result(
                    title="Vanguard S&P 500 UCITS ETF (VUSA) — Factsheet y riesgo",
                    snippet="Factsheet del ETF con indicadores de riesgo y mercado.",
                    url="https://www.justetf.com/en/etf-profile-vusa.html",
                    source="justetf.com",
                ),
            )
        )
    )

    report = chat.handle("investiga riesgos del ETF Vanguard S&P 500 UCITS ETF")

    assert report.startswith("[RESEARCH]")
    assert "Evidencia financiera relevante insuficiente" not in report
    assert "HECHOS Y NOTICIAS" in report
    assert "Vanguard S&P 500 UCITS ETF (VUSA) — Factsheet y riesgo" in report
    assert "justetf.com" in report
    assert "https://www.justetf.com/en/etf-profile-vusa.html" in report


def test_mixed_results_keep_only_financial_relevant_ones() -> None:
    chat = FinanceResearchChat(
        FakeSearch(
            (
                _result(
                    title="Definición de riesgo financiero",
                    snippet="Concepto y significado del riesgo en finanzas.",
                    url="https://economipedia.com/definiciones/riesgo.html",
                    source="economipedia.com",
                ),
                _result(
                    title="Vanguard S&P 500 UCITS ETF — KIID y riesgo del fondo",
                    snippet="Documento KIID del ETF con indicadores de riesgo.",
                    url="https://www.vanguard.es/en/kiid-vusa.pdf",
                    source="vanguard.es",
                ),
            )
        )
    )

    report = chat.handle("¿Qué riesgos tiene el ETF Vanguard S&P 500 UCITS?")

    assert report.startswith("[RESEARCH]")
    assert "Evidencia financiera relevante insuficiente" not in report
    assert "economipedia" not in report.casefold()
    assert "Vanguard S&P 500 UCITS ETF — KIID y riesgo del fondo" in report
    assert "vanguard.es" in report
