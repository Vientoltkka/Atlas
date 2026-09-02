"""Bounded, read-only web search through Bing's public RSS endpoint."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse
from xml.etree import ElementTree

import httpx

from tools.base_tool import BaseTool
from tools.tool_context import ToolContext


_SEARCH_URL = "https://www.bing.com/search"


class WebSearchError(RuntimeError):
    """A controlled failure while retrieving search results."""


class WebSearchTimeoutError(WebSearchError):
    """The search provider did not respond within the configured timeout."""


@dataclass(frozen=True, slots=True)
class WebSearchResult:
    """One untrusted result returned by the web search provider."""

    title: str
    url: str
    snippet: str
    source: str

    def to_dict(self) -> dict[str, str]:
        return {"title": self.title, "url": self.url, "snippet": self.snippet, "source": self.source}


class WebSearchTool(BaseTool):
    """Search the public web without fetching or executing result content."""

    def __init__(self, *, client: Any | None = None, timeout_seconds: float = 8.0, max_results: int = 5, max_response_bytes: int = 500_000) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive.")
        if not 1 <= max_results <= 10:
            raise ValueError("max_results must be between 1 and 10.")
        if max_response_bytes < 1_024:
            raise ValueError("max_response_bytes must be at least 1024.")
        self._client = client
        self._timeout_seconds = timeout_seconds
        self._max_results = max_results
        self._max_response_bytes = max_response_bytes

    @property
    def name(self) -> str:
        return "web_search"

    @property
    def description(self) -> str:
        return "Busca resultados actuales en la web y devuelve título, URL, snippet y dominio."

    def semantic_metadata(self) -> dict[str, object]:
        return {
            "category": "web", "capabilities": ("web_search", "research"), "supported_intents": ("web.search",),
            "input_description": "Consulta de texto para buscar en la web.",
            "output_description": "Resultados estructurados con title, url, snippet y source.",
            "output_fields": ("title", "url", "snippet", "source"),
            "limitations": ("No abre ni descarga los resultados.", "El contenido de resultados se trata como datos no confiables.", "Máximo cinco resultados por consulta conversacional."),
            "tags": ("web", "search", "research", "internet"),
            "positive_examples": ("busca en internet noticias sobre OpenAI",),
        }

    def execute(self, context: ToolContext) -> tuple[dict[str, str], ...]:
        query = context.parameters.get("query")
        if not isinstance(query, str):
            raise ValueError("web_search requires a string query.")
        limit = context.parameters.get("max_results", self._max_results)
        if not isinstance(limit, int) or isinstance(limit, bool):
            raise ValueError("max_results must be an integer.")
        return tuple(result.to_dict() for result in self.search(query, max_results=limit))

    def search(self, query: str, *, max_results: int | None = None) -> tuple[WebSearchResult, ...]:
        cleaned_query = " ".join(query.split())
        if not cleaned_query:
            raise ValueError("web_search query cannot be empty.")
        limit = self._max_results if max_results is None else min(max_results, self._max_results)
        if not 1 <= limit <= self._max_results:
            raise ValueError(f"max_results must be between 1 and {self._max_results}.")
        try:
            html = self._fetch_html(cleaned_query)
        except httpx.TimeoutException as error:
            raise WebSearchTimeoutError("La búsqueda web agotó el tiempo de espera.") from error
        except httpx.HTTPError as error:
            raise WebSearchError("No se pudo consultar el proveedor de búsqueda.") from error
        try:
            results = _parse_rss_results(html)
        except ElementTree.ParseError as error:
            raise WebSearchError("El proveedor devolvió una respuesta no válida.") from error
        return tuple(results[:limit])

    def _fetch_html(self, query: str) -> str:
        own_client = self._client is None
        client = self._client or httpx.Client(timeout=self._timeout_seconds, follow_redirects=False, headers={"User-Agent": "Atlas/1.0 web-search"})
        try:
            with client.stream("GET", _SEARCH_URL, params={"format": "rss", "q": query}) as response:
                response.raise_for_status()
                chunks: list[bytes] = []
                size = 0
                for chunk in response.iter_bytes():
                    size += len(chunk)
                    if size > self._max_response_bytes:
                        raise WebSearchError("La respuesta del proveedor excedió el tamaño permitido.")
                    chunks.append(chunk)
                return b"".join(chunks).decode(response.encoding or "utf-8", errors="replace")
        finally:
            if own_client:
                client.close()


def _parse_rss_results(payload: str) -> list[WebSearchResult]:
    root = ElementTree.fromstring(payload)
    results: list[WebSearchResult] = []
    for item in root.findall("./channel/item"):
        title = _clean_text(item.findtext("title") or "")
        url = (item.findtext("link") or "").strip()
        snippet = _clean_text(item.findtext("description") or "")
        parsed = urlparse(url)
        if title and parsed.scheme in {"http", "https"} and parsed.netloc:
            results.append(WebSearchResult(title=title[:300], url=url, snippet=snippet[:800], source=parsed.netloc.removeprefix("www.")))
    return results


def _clean_text(value: str) -> str:
    return " ".join(value.split())
