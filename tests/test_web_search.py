from __future__ import annotations

from types import SimpleNamespace

import httpx

from agents.registry import AgentRegistry
from core.orchestrator import AtlasOrchestrator
from core.planner import Plan
from core.router import Router
from tools.tool_context import ToolContext
from tools.web_search import WebSearchResult, WebSearchTimeoutError, WebSearchTool


_RESULTS_XML = b"""<?xml version='1.0'?><rss><channel>
<item><title>OpenAI news</title><link>https://example.com/openai</link><description>Latest OpenAI announcement.</description></item>
<item><title>Model guide</title><link>https://docs.example.org/models</link><description>Current open source coding models.</description></item>
<item><title>Third result</title><link>https://news.example.net/item</link><description>Additional context.</description></item>
</channel></rss>"""


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
    def __init__(self, body: bytes = _RESULTS_XML, error: Exception | None = None) -> None:
        self.body = body
        self.error = error
        self.calls: list[tuple[str, str, dict[str, str]]] = []

    def stream(self, method: str, url: str, *, params: dict[str, str]):
        self.calls.append((method, url, params))
        if self.error is not None:
            raise self.error
        return _Response(self.body)


def test_web_search_returns_structured_results_from_provider_html() -> None:
    client = _Client()
    tool = WebSearchTool(client=client)

    results = tool.search("OpenAI latest news")

    assert client.calls == [("GET", "https://www.bing.com/search", {"format": "rss", "q": "OpenAI latest news"})]
    assert results[0] == WebSearchResult("OpenAI news", "https://example.com/openai", "Latest OpenAI announcement.", "example.com")
    assert results[1].url == "https://docs.example.org/models"
    assert results[1].source == "docs.example.org"


def test_web_search_tool_execute_returns_structured_fields_and_honors_limit() -> None:
    tool = WebSearchTool(client=_Client())

    results = tool.execute(ToolContext({"query": "coding models", "max_results": 2}))

    assert len(results) == 2
    assert results[0] == {
        "title": "OpenAI news",
        "url": "https://example.com/openai",
        "snippet": "Latest OpenAI announcement.",
        "source": "example.com",
    }


def test_web_search_returns_no_results() -> None:
    assert WebSearchTool(client=_Client(b"<html></html>")).search("nothing") == ()


def test_web_search_converts_timeout_to_controlled_error() -> None:
    tool = WebSearchTool(client=_Client(error=httpx.ReadTimeout("slow")))

    try:
        tool.search("OpenAI")
    except WebSearchTimeoutError as error:
        assert "agotó" in str(error)
    else:  # pragma: no cover
        raise AssertionError("Expected a controlled timeout error")


def test_explicit_web_and_research_prompts_use_search_and_show_sources() -> None:
    search = _FakeSearch()
    orchestrator = _orchestrator(search)

    web_response = orchestrator.process_prompt("busca en internet las últimas noticias sobre OpenAI", confirm=lambda _prompt: "")
    research_response = orchestrator.process_prompt("investiga modelos open source para programación", confirm=lambda _prompt: "")

    assert search.calls == [
        ("las últimas noticias sobre OpenAI", 3),
        ("modelos open source para programación", 5),
    ]
    assert "Resumen de la búsqueda" in web_response
    assert "Fuentes:" in web_response
    assert "https://example.com/openai" in web_response
    assert "Fuentes:" in research_response


def test_normal_chat_does_not_trigger_web_search() -> None:
    search = _FakeSearch()

    assert _orchestrator(search).process_prompt("hola", confirm=lambda _prompt: "") == "chat normal"
    assert search.calls == []


class _FakeSearch:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    def search(self, query: str, *, max_results: int):
        self.calls.append((query, max_results))
        return (WebSearchResult("OpenAI news", "https://example.com/openai", "Latest OpenAI announcement.", "example.com"),)


class _ChatAgent:
    name = "chat"

    def run(self, *, model, messages):
        del model, messages
        return "chat normal"


def _orchestrator(search: _FakeSearch) -> AtlasOrchestrator:
    registry = AgentRegistry()
    registry.register(_ChatAgent())
    return AtlasOrchestrator(
        planner=SimpleNamespace(create_plan=lambda text: Plan("chat", text)),
        router=Router(),
        model_manager=SimpleNamespace(choose_model=lambda _task: "test-model"),
        memory=SimpleNamespace(add_user=lambda _text: None, add_assistant=lambda _text: None, history=list),
        registry=registry,
        write_file=SimpleNamespace(execute=lambda *_args: "unused"),
        web_search_tool=search,  # type: ignore[arg-type]
    )
