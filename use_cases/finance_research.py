"""Atlas Finance V2.3 — Evidence Research (determinista, solo lectura).

Cuando el usuario pide explicitamente investigar un activo o tema
financiero ("investiga X", "analiza noticias de X", "que riesgos tiene
X"), este caso de uso recupera evidencia web con la herramienta
web_search ya existente (la misma instancia inyectada por el
orquestador; no duplica cliente ni llamadas web) y devuelve un informe
breve y trazable con etiqueta [RESEARCH].

Reglas: la evidencia no puede registrar MarketEvent, modificar la
cartera paper, crear propuestas de orden ni responder confirmaciones
paper; no se inventan cotizaciones, volumen, fundamentales ni noticias;
foros, redes sociales y "sentimiento" quedan fuera del bloque; si la
evidencia es insuficiente, antigua o contradictoria se dice
expresamente. No usa LLM y no toca los flujos V2.1/V2.2.
"""

from __future__ import annotations

import re
import unicodedata
from datetime import datetime, timedelta
from email.utils import parsedate_to_datetime

from tools.web_search import WebSearchError, WebSearchResult, WebSearchTool

RESEARCH_LABEL = "[RESEARCH]"

_MAX_RESEARCH_QUERIES = 3
_MAX_RESULTS_KEPT = 6
_SNIPPET_MAX_CHARS = 280
_STALE_EVIDENCE_AFTER = timedelta(days=14)

_ASSET_MARKERS = (
    "bitcoin",
    "btc",
    "ethereum",
    "eth",
    "cripto",
    "criptomoneda",
    "etf",
    "accion",
    "acciones",
    "bolsa",
    "ibex",
    "sp500",
    "nasdaq",
    "fondo indexado",
    "fondo",
    "oro",
    "plata",
    "petroleo",
    "divisa",
    "rentabilidad",
    "mercado",
    "inversion",
    "invertir",
    "invierte",
)

_FORUM_SOCIAL_MARKERS = (
    "reddit",
    "twitter",
    "x.com",
    "facebook",
    "tiktok",
    "instagram",
    "telegram",
    "discord",
    "quora",
    "foro",
    "forum",
    "sentimiento",
    "sentiment",
)

_QUERIES_BY_INTENT = {
    "investigate": (
        '"{entity}" ultimas noticias',
        '"{entity}" precio mercado actual',
        '"{entity}" riesgos inversion',
    ),
    "news": (
        '"{entity}" ultimas noticias',
        '"{entity}" precio mercado actual',
        '"{entity}" riesgos inversion',
    ),
    "risks": (
        '"{entity}" ETF factsheet risk',
        '"{entity}" KIID risk',
        '"{entity}" latest news',
    ),
}

_TOPIC_ENTITY_PREFIX = re.compile(
    r"^(?:los\s+|las\s+|el\s+|la\s+)?"
    r"(?:riesgos?|noticias|an[aá]lisis|evoluci[oó]n|precio|inversi[oó]n)\s+"
    r"(?:de\s+invertir\s+en\s+|del\s+|de\s+la\s+|de\s+los\s+|de\s+las\s+|"
    r"de\s+|al\s+|sobre\s+)?",
    re.IGNORECASE,
)

_GENERIC_RESULT_MARKERS = (
    "wikipedia",
    "wiktionary",
    "wikcionario",
    "diccionario",
    "dictionary",
    "definicion",
    "significado",
    "concepto",
    "enciclopedia",
    "prevencion",
    "psicosocial",
    "laboral",
    "laborales",
)

_FINANCIAL_CONTEXT_MARKERS = (
    "etf",
    "fondo",
    "fund",
    "ucits",
    "factsheet",
    "kiid",
    "prospecto",
    "prospectus",
    "mercado",
    "market",
    "bolsa",
    "stock",
    "share",
    "shares",
    "cotizacion",
    "price",
    "precio",
    "rentabilidad",
    "performance",
    "dividendo",
    "dividend",
    "isin",
    "ticker",
    "gestora",
    "riesgo",
    "risk",
    "volatilidad",
    "volatility",
    "inversion",
    "asset",
    "portfolio",
    "indice",
    "index",
    "noticias",
    "news",
)

_FINANCIAL_SOURCE_MARKERS = (
    "morningstar",
    "vanguard",
    "ishares",
    "blackrock",
    "justetf",
    "etfdb",
    "bloomberg",
    "reuters",
    "cnbc",
    "marketwatch",
    "investing.com",
    "finance.yahoo",
    "wsj",
    "ft.com",
    "expansion.com",
    "cincodias",
    "eleconomista",
    "msci",
    "spglobal",
    "banco de espana",
    "cnmv",
)

_ENTITY_GENERIC_TOKENS = frozenset(
    {
        "etf",
        "fondo",
        "fund",
        "funds",
        "de",
        "del",
        "la",
        "el",
        "los",
        "las",
        "un",
        "una",
        "y",
        "e",
        "o",
        "the",
        "of",
        "index",
        "inc",
        "corp",
        "plc",
        "group",
        "acciones",
        "accion",
    }
)

_ENTITY_TOKEN_SPLIT = re.compile(r"[^a-z0-9]+")

_INVESTIGATE_PATTERN = re.compile(
    r"^\s*investiga(?:r)?\b\s*(?:sobre\s+|las\s+noticias\s+(?:de|sobre)\s+|"
    r"los\s+riesgos\s+(?:de|de\s+invertir\s+en)\s+|el\s+|la\s+|los\s+|las\s+)?"
    r"(?P<subject>.+)$",
    re.IGNORECASE,
)
_NEWS_PATTERN = re.compile(
    r"^\s*(?:an[aá]liz[ae]|an[aá]lisis\s+de)\s+(?:las\s+)?noticias\s+"
    r"(?:de|sobre|del|de\s+la)\s+(?:el|la|los|las|un|una)?\s*"
    r"(?P<subject>.+)$",
    re.IGNORECASE,
)
_RISKS_PATTERN = re.compile(
    r"^\s*(?:qu[eé]|cu[aá]les\s+son\s+los)\s+riesgos\s+"
    r"(?:de|tiene|tiene\s+invertir\s+en|de\s+invertir\s+en)\s+"
    r"(?:el|la|los|las|un|una)?\s*(?P<subject>.+)$",
    re.IGNORECASE,
)

_SUBJECT_NOISE_PREFIXES = (
    "invertir en",
    "comprar",
    "vender",
    "el precio de",
    "la evolucion de",
    "los riesgos de",
    "el riesgo de",
)
_TICKER_PATTERN = re.compile(r"\b[A-Z]{2,5}\b")
_TRAILING_PUNCTUATION = " \t\r\n?¿!¡.;"

_POSITIVE_SIGNALS = ("sube", "subida", "alcista", "rally", "optimista", "repunte", "record")
_NEGATIVE_SIGNALS = ("baja", "bajada", "caida", "desplome", "bajista", "pesimista", "correccion")


class FinanceResearchChat:
    """Maneja los turnos de investigacion financiera sin LLM ni datos externos."""

    def __init__(self, search_tool: WebSearchTool, now_provider=None) -> None:
        self._tool = search_tool
        self._now_provider = now_provider or (lambda: datetime.now().astimezone())
        self.last_queries: tuple[str, ...] = ()
        self.last_subject: str | None = None
        self.last_entity: str | None = None

    def handles(self, prompt: str) -> bool:
        """True solo ante peticion de investigacion financiera explicita."""
        return _classify_research(prompt) is not None

    def handle(self, prompt: str) -> str:
        """Ejecuta un turno de investigacion y devuelve el informe [RESEARCH]."""
        classified = _classify_research(prompt)
        if classified is None:
            raise ValueError(f"turno de investigacion no reconocido: {prompt!r}")
        intent, subject, entity = classified
        self.last_subject = subject
        self.last_entity = entity
        topic = _topic_for(intent, subject)
        queries = tuple(
            template.format(entity=entity)
            for template in _QUERIES_BY_INTENT[topic]
        )[:_MAX_RESEARCH_QUERIES]

        facts: list[tuple[int, str, str, str, str, str | None]] = []
        executed: list[str] = []
        failed: list[str] = []
        seen_urls: set[str] = set()
        discarded = 0
        for query in queries[:_MAX_RESEARCH_QUERIES]:
            try:
                results = self._tool.search(query)
            except (WebSearchError, ValueError):
                failed.append(query)
                continue
            executed.append(query)
            for result in results:
                if _is_forum_or_social(result) or result.url in seen_urls:
                    continue
                if len(facts) >= _MAX_RESULTS_KEPT:
                    break
                if not _is_relevant(result, entity):
                    discarded += 1
                    continue
                seen_urls.add(result.url)
                facts.append(
                    (
                        len(facts) + 1,
                        result.title,
                        result.source,
                        result.url,
                        result.snippet or "Sin resumen disponible.",
                        result.date,
                    )
                )
            if len(facts) >= _MAX_RESULTS_KEPT:
                break
        self.last_queries = tuple(executed)
        return _compose_research_report(
            subject=subject,
            entity=entity,
            now=self._now_provider(),
            facts=facts,
            executed=executed,
            failed=failed,
            discarded=discarded,
        )


def handles_research_prompt(prompt: str) -> bool:
    """Clasificador puro: True solo ante peticion explicita de investigacion."""
    return _classify_research(prompt) is not None


def _classify_research(prompt: str) -> "tuple[str, str, str] | None":
    """Devuelve (intencion, tema solicitado, entidad financiera) o None."""
    text = prompt.strip().lstrip("¿¡").strip()
    if "\n" in text:
        text = " ".join(line.strip() for line in text.splitlines() if line.strip())
    for pattern, intent in (
        (_NEWS_PATTERN, "news"),
        (_RISKS_PATTERN, "risks"),
        (_INVESTIGATE_PATTERN, "investigate"),
    ):
        match = pattern.match(text)
        if match is None:
            continue
        subject = _clean_subject(match.group("subject"))
        if not subject:
            return None
        entity = _extract_entity(subject)
        if not entity or not _is_finance_subject(entity):
            return None
        return intent, subject, entity
    return None


def _topic_for(intent: str, subject: str) -> str:
    """Resuelve el tema solicitado, incluso bajo la intencion generica."""
    if intent != "investigate":
        return intent
    folded = _fold(subject)
    if folded.startswith("riesgo"):
        return "risks"
    if folded.startswith(("noticias", "analisis")):
        return "news"
    return "investigate"


def _extract_entity(subject: str) -> str:
    """Separa la entidad financiera del tema solicitado en el asunto."""
    match = _TOPIC_ENTITY_PREFIX.match(subject)
    if match is None or match.end() >= len(subject):
        return subject
    entity = subject[match.end():].strip()
    entity = re.sub(r"^(?:el|la|los|las|un|una)\s+", "", entity, flags=re.IGNORECASE)
    return entity.strip().rstrip(_TRAILING_PUNCTUATION).strip()


def _clean_subject(raw: str) -> str:
    subject = raw.strip().rstrip(_TRAILING_PUNCTUATION).strip()
    folded = _fold(subject)
    for prefix in _SUBJECT_NOISE_PREFIXES:
        if folded.startswith(prefix):
            subject = subject[len(prefix):].strip()
            folded = _fold(subject)
    subject = re.sub(r"^(?:el|la|los|las|un|una)\s+", "", subject, flags=re.IGNORECASE)
    return subject.strip().rstrip(_TRAILING_PUNCTUATION).strip()


def _is_finance_subject(subject: str) -> bool:
    folded = _fold(subject)
    if any(marker in folded for marker in _ASSET_MARKERS):
        return True
    return _TICKER_PATTERN.search(subject) is not None


def _is_forum_or_social(result: WebSearchResult) -> bool:
    haystack = _fold(f"{result.title} {result.snippet} {result.url} {result.source}")
    return any(marker in haystack for marker in _FORUM_SOCIAL_MARKERS)


def _is_relevant(result: WebSearchResult, entity: str) -> bool:
    """Exige que el resultado mencione la entidad y tenga contexto financiero."""
    haystack = _fold(
        f"{result.title} {result.snippet} {result.url} {result.source}"
    )
    if any(marker in haystack for marker in _GENERIC_RESULT_MARKERS):
        return False
    title_snippet = _fold(f"{result.title} {result.snippet}")
    identity_hits = [
        token
        for token in _significant_entity_tokens(entity)
        if token in title_snippet
    ]
    if not identity_hits:
        return False
    if all(not any(char.isalpha() for char in token) for token in identity_hits):
        return False
    if any(marker in haystack for marker in _FINANCIAL_CONTEXT_MARKERS):
        return True
    return any(marker in haystack for marker in _FINANCIAL_SOURCE_MARKERS)


def _significant_entity_tokens(entity: str) -> tuple[str, ...]:
    tokens = _ENTITY_TOKEN_SPLIT.split(_fold(entity))
    return tuple(
        token
        for token in tokens
        if token and len(token) >= 2 and token not in _ENTITY_GENERIC_TOKENS
    )


def _compose_research_report(
    *,
    subject: str,
    entity: str,
    now: datetime,
    facts: "list[tuple[int, str, str, str, str, str | None]]",
    executed: list[str],
    failed: list[str],
    discarded: int,
) -> str:
    if not facts:
        return _insufficient_report(entity, now, executed, failed, discarded)
    lines: list[str] = []
    lines.append(f"{RESEARCH_LABEL} Informe de investigacion web")
    lines.append("")
    lines.append(f"- Tema solicitado: {subject}")
    if entity != subject:
        lines.append(f"- Entidad financiera: {entity}")
    lines.append(f"- Fecha/hora de consulta: {now.strftime('%Y-%m-%d %H:%M %Z')}")
    if executed:
        lines.append(f"- Consultas ejecutadas ({len(executed)}): " + " | ".join(executed))
    lines.append("")
    lines.append("HECHOS Y NOTICIAS (textual de los fragmentos recuperados)")
    for index, _title, source, _url, snippet, _date in facts:
        lines.append(f"- [{index}] ({source}): «{snippet[:_SNIPPET_MAX_CHARS]}»")
    lines.append("")
    lines.append("INFERENCIAS")
    lines.append(
        "- No se han generado inferencias automaticas. Este informe solo cita lo "
        "textual de los fragmentos; cualquier interpretacion adicional exige "
        "revisar las fuentes citadas."
    )
    lines.append("")
    lines.append("FUENTES")
    for index, title, source, url, _snippet, date in facts:
        lines.append(
            f"- [{index}] {title} — {source} · {url} · "
            f"Fecha: {date if date else 'no disponible'}"
        )
    lines.append("")
    lines.append("RIESGOS, INCERTIDUMBRES, CONTRADICCIONES Y DATOS NO VERIFICADOS")
    lines.extend(_uncertainty_lines(facts, failed, now))
    lines.append("")
    lines.append("CONCLUSION EDUCATIVA")
    lines.append(
        "- Resumen educativo basado unicamente en la evidencia citada; no es una "
        "recomendacion personalizada de comprar o vender y no promete "
        f"rentabilidad. {RESEARCH_LABEL}"
    )
    return "\n".join(lines)


def _insufficient_report(
    entity: str,
    now: datetime,
    executed: list[str],
    failed: list[str],
    discarded: int,
) -> str:
    lines = [
        f"{RESEARCH_LABEL} Evidencia financiera relevante insuficiente para "
        f"{entity}.",
        "",
        f"- Fecha/hora de consulta: {now.strftime('%Y-%m-%d %H:%M %Z')}",
    ]
    if failed:
        lines.append(
            "- La busqueda web fallo o no devolvio resultados utilizables en: "
            + " | ".join(failed)
        )
    elif executed:
        lines.append(
            "- La busqueda web no devolvio resultados verificables (foros, redes "
            "sociales y sentimiento quedan excluidos del bloque)."
        )
    if discarded:
        lines.append(
            f"- Se descartaron {discarded} resultados por falta de relevancia "
            "financiera: definiciones generales, paginas educativas no "
            "financieras o resultados que no mencionan el activo."
        )
    lines.extend(
        [
            "- No se registran cotizaciones, volumen, fundamentales ni noticias: "
            "sin evidencia verificable no se inventan datos.",
            "",
            f"Consulta solo lectura; no se ha modificado nada. {RESEARCH_LABEL}",
        ]
    )
    return "\n".join(lines)


def _uncertainty_lines(
    facts: "list[tuple[int, str, str, str, str, str | None]]",
    failed: list[str],
    now: datetime,
) -> list[str]:
    lines = [
        "- La evidencia son fragmentos de resultados de busqueda; no se han "
        "verificado las fuentes originales ni los datos numericos que citan "
        "(precios, volumen, fundamentales).",
        "- Foros, redes sociales y analisis de sentimiento estan excluidos de "
        "este bloque.",
    ]
    dates: list[datetime] = []
    for item in facts:
        raw_date = item[5]
        if not raw_date:
            continue
        try:
            parsed = parsedate_to_datetime(raw_date)
        except (TypeError, ValueError):
            parsed = None
        if parsed is not None:
            dates.append(parsed)
    if not dates:
        lines.append(
            "- Ninguna fuente incluye fecha verificable; la antiguedad de la "
            "informacion no puede confirmarse."
        )
    elif all(now - date > _STALE_EVIDENCE_AFTER for date in dates):
        oldest = min(dates)
        lines.append(
            "- Toda la evidencia fechada es anterior a "
            f"{oldest:%Y-%m-%d}: puede estar desactualizada."
        )
    if failed:
        lines.append("- Consultas sin resultados utilizables: " + " | ".join(failed))
    contradiction = _contradiction_note(facts)
    if contradiction is not None:
        lines.append(contradiction)
    return lines


def _contradiction_note(
    facts: "list[tuple[int, str, str, str, str, str | None]]",
) -> "str | None":
    snippets = [item[4].casefold() for item in facts]
    has_positive = any(
        signal in snippet for snippet in snippets for signal in _POSITIVE_SIGNALS
    )
    has_negative = any(
        signal in snippet for snippet in snippets for signal in _NEGATIVE_SIGNALS
    )
    if has_positive and has_negative:
        return (
            "- Posibles senales contradictorias entre las fuentes (tono alcista y "
            "bajista en la misma evidencia): contrastalas antes de extraer "
            "conclusiones."
        )
    return None


def _fold(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text.casefold())
    return "".join(
        character
        for character in normalized
        if unicodedata.category(character) != "Mn"
    )
