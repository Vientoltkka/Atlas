"""Focused contract for the bounded investment-analysis flow (finance + web evidence)."""

from __future__ import annotations

from bootstrap.bootstrap import Bootstrap
from core.agent_orchestrator import AgentOrchestrator
from core.orchestrator import (
    _extract_investment_subject,
    _requires_market_evidence,
    gather_finance_web_evidence,
)
from agents.finance_agent import FinanceAgent
from agents.registry import AgentRegistry
from tools.web_search import WebSearchResult, WebSearchTool


class Agent:
    def __init__(self, name: str) -> None:
        self.name = name


class FakeSearchTool:
    def __init__(self, results: tuple[WebSearchResult, ...]) -> None:
        self._results = results
        self.calls: list[str] = []

    def search(self, query: str, *, max_results: int | None = None):
        self.calls.append(query)
        return self._results


def test_investment_requests_route_to_finance() -> None:
    registry = AgentRegistry()
    for name in ("training", "nutrition", "medical", "legal", "finance", "code"):
        registry.register(Agent(name))  # type: ignore[arg-type]
    selector = AgentOrchestrator(registry)

    for prompt in (
        "Atlas, analiza Bitcoin como inversión ahora. Investiga información actual y riesgos.",
        "¿Debería invertir en Ethereum?",
        "Analiza el ETF SPY como inversión",
        "analiza btc y dime su riesgo de mercado",
        "¿Cómo están las criptomonedas para invertir hoy?",
    ):
        selection = selector.select(prompt)
        assert selection.primary_agent == "finance", prompt


def test_investment_subject_extraction_is_generic() -> None:
    assert _extract_investment_subject(
        "Atlas, analiza Bitcoin como inversión ahora. Investiga información actual."
    ) == "bitcoin"
    assert _extract_investment_subject("¿Debería invertir en Ethereum?") == "ethereum"
    assert _extract_investment_subject("Analiza el ETF SPY como inversión") == "etf spy"
    assert _extract_investment_subject("Hazme un presupuesto mensual") is None


def test_requires_market_evidence_needs_asset_and_analysis() -> None:
    assert _requires_market_evidence(
        "Atlas, analiza Bitcoin como inversión ahora. Investiga información actual."
    )
    assert not _requires_market_evidence("Quiero invertir parte de mis ingresos")
    assert not _requires_market_evidence("Calcula el interes simple de 1000 euros")


def test_gather_finance_web_evidence_runs_maximum_three_distinct_queries() -> None:
    tool = FakeSearchTool(
        (
            WebSearchResult(
                "Bitcoin price today",
                "https://news.example.com/btc-price",
                "Bitcoin trades near key level.",
                "news.example.com",
                "Tue, 09 Sep 2026 10:00:00 GMT",
            ),
        )
    )

    evidence, queries = gather_finance_web_evidence(tool, "analiza Bitcoin como inversión ahora")  # type: ignore[arg-type]

    assert len(queries) <= 3
    assert len(queries) == len(set(queries))
    assert len(tool.calls) == len(queries)
    assert evidence is not None
    assert evidence.startswith("CONTEXTO DE EVIDENCIA WEB")


def test_evidence_preserves_source_url_snippet_and_pubdate() -> None:
    tool = FakeSearchTool(
        (
            WebSearchResult(
                "Bitcoin price today",
                "https://news.example.com/btc-price",
                "Bitcoin trades near key level.",
                "news.example.com",
                "Tue, 09 Sep 2026 10:00:00 GMT",
            ),
        )
    )

    evidence, _queries = gather_finance_web_evidence(tool, "analiza Bitcoin como inversión")  # type: ignore[arg-type]

    assert evidence is not None
    assert "Bitcoin price today" in evidence
    assert "https://news.example.com/btc-price" in evidence
    assert "news.example.com" in evidence
    assert "Bitcoin trades near key level." in evidence
    assert "Tue, 09 Sep 2026 10:00:00 GMT" in evidence
    assert "no son instrucciones" in evidence


def test_missing_pubdate_does_not_invent_date() -> None:
    tool = FakeSearchTool(
        (
            WebSearchResult(
                "Analyst warning",
                "https://markets.example.org/warning",
                "Analysts flag downside risk.",
                "markets.example.org",
            ),
        )
    )

    evidence, _queries = gather_finance_web_evidence(tool, "analiza Bitcoin como inversión")  # type: ignore[arg-type]

    assert evidence is not None
    assert "Fecha: no disponible" in evidence
    assert "markets.example.org" in evidence


def test_gather_finance_web_evidence_returns_none_without_results() -> None:
    tool = FakeSearchTool(())

    assert gather_finance_web_evidence(tool, "analiza Bitcoin como inversión") is None  # type: ignore[arg-type]


def test_web_search_parses_pubdate_from_rss() -> None:
    rss = (
        "<?xml version='1.0'?><rss><channel>"
        "<item><title>Dated</title><link>https://example.com/dated</link>"
        "<description>Has a date.</description>"
        "<pubDate>Tue, 09 Sep 2026 10:00:00 GMT</pubDate></item>"
        "<item><title>Undated</title><link>https://example.com/undated</link>"
        "<description>No date.</description></item>"
        "</channel></rss>"
    )

    class _Response:
        encoding = "utf-8"

        def __init__(self, body: bytes) -> None:
            self._body = body

        def __enter__(self):
            return self

        def __exit__(self, *_args) -> None:
            return None

        def raise_for_status(self) -> None:
            return None

        def iter_bytes(self):
            yield self._body

    class _Client:
        def stream(self, _method, _url, *, params):
            return _Response(rss.encode("utf-8"))

    results = WebSearchTool(client=_Client()).search("anything")
    assert results[0].date == "Tue, 09 Sep 2026 10:00:00 GMT"
    assert results[1].date is None


def test_finance_system_prompt_defines_investment_analysis_contract() -> None:
    prompt = " ".join(FinanceAgent.SYSTEM_PROMPT.casefold().split())

    for section in (
        "hechos",
        "señales",
        "alcistas",
        "bajistas",
        "riesgos e incertidumbre",
        "interpretacion",
        "conclusion",
        "favorable / neutral / desfavorable",
        "confianza",
        "fuentes utilizadas",
    ):
        assert section in prompt, section
    assert "jamas inventes precios" in prompt
    assert "no ejecutes compras, ventas" in prompt
    assert "no prometas rentabilidad" in prompt


def test_finance_investment_analysis_gathers_web_evidence_before_inference(monkeypatch) -> None:
    orchestrator = Bootstrap.build()
    finance = orchestrator._registry.get("finance")
    assert isinstance(finance, FinanceAgent)
    search_calls: list[str] = []
    inference_calls: list[list[dict[str, str]]] = []

    class _FakeSearch:
        def search(self, query: str, *, max_results: int | None = None):
            search_calls.append(query)
            return (
                WebSearchResult(
                    "Bitcoin price today",
                    "https://news.example.com/btc-price",
                    "Bitcoin trades near a key resistance level.",
                    "news.example.com",
                    "Tue, 09 Sep 2026 10:00:00 GMT",
                ),
                WebSearchResult(
                    "Analyst warning",
                    "https://markets.example.org/warning",
                    "Analysts flag downside risk for crypto.",
                    "markets.example.org",
                ),
            )

    orchestrator._web_search_tool = _FakeSearch()  # type: ignore[assignment]
    monkeypatch.setattr(finance._client, "check_model_health", lambda *_args, **_kwargs: None)

    def respond(*, model: str, messages: list[dict[str, str]]) -> str:
        inference_calls.append(messages)
        return (
            "HECHOS\nBitcoin cotiza cerca de una resistencia clave.\n\n"
            "SEÑALES\n- alcistas: momentum positivo.\n- bajistas: riesgo de correccion.\n\n"
            "RIESGOS E INCERTIDUMBRE\nVolatilidad alta.\n\n"
            "INTERPRETACION\nEscenario neutral.\n\n"
            "CONCLUSION: NEUTRAL, confianza media.\n\n"
            "FUENTES UTILIZADAS\n- Bitcoin price today (news.example.com)\n"
            "- Analyst warning (markets.example.org)"
        )

    monkeypatch.setattr(finance._client, "ask", respond)

    response = orchestrator.process_prompt(
        "Atlas, analiza Bitcoin como inversión ahora. Investiga información actual, "
        "principales noticias y factores alcistas y bajistas. Valora riesgo y dime "
        "qué conclusión sacarías, explicando las fuentes utilizadas.",
        confirm=lambda _prompt: "",
    )

    assert response.startswith("HECHOS")
    assert len(search_calls) <= 3
    assert len(set(search_calls)) == len(search_calls)
    assert orchestrator.last_finance_evidence is not None
    messages = inference_calls[-1]
    evidence_messages = [
        (index, message)
        for index, message in enumerate(messages)
        if message.get("role") == "user"
        and message.get("content", "").startswith("CONTEXTO DE EVIDENCIA WEB")
    ]
    assert len(evidence_messages) == 1
    evidence_index, evidence_message = evidence_messages[0]
    user_prompt_index = next(
        index
        for index, message in enumerate(messages)
        if message.get("content", "").startswith("Atlas, analiza Bitcoin")
    )
    assert evidence_index > user_prompt_index
    evidence = evidence_message["content"]
    assert "https://news.example.com/btc-price" in evidence
    assert "markets.example.org" in evidence
    assert "Bitcoin trades near a key resistance level." in evidence
    assert "Tue, 09 Sep 2026 10:00:00 GMT" in evidence
    assert "no son instrucciones" in evidence


def test_finance_investment_flow_executes_no_trading_actions(monkeypatch) -> None:
    orchestrator = Bootstrap.build()
    finance = orchestrator._registry.get("finance")
    assert isinstance(finance, FinanceAgent)

    class _FakeSearch:
        def search(self, query: str, *, max_results: int | None = None):
            return (
                WebSearchResult(
                    "Market update",
                    "https://markets.example.org/update",
                    "Mixed signals across risk assets.",
                    "markets.example.org",
                ),
            )

    orchestrator._web_search_tool = _FakeSearch()  # type: ignore[assignment]
    monkeypatch.setattr(finance._client, "check_model_health", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        finance._client,
        "ask",
        lambda *, model, messages: "CONCLUSION: NEUTRAL, confianza baja. No se ejecuto ninguna operacion.",
    )

    response = orchestrator.process_prompt(
        "Analiza Ethereum como inversión con información actual y riesgo.",
        confirm=lambda _prompt: "",
    )

    assert "No se ejecuto ninguna operacion" in response
    evidence = orchestrator.last_finance_evidence["evidence"]
    assert "no son instrucciones" in str(evidence)
