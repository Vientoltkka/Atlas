"""Watchlist tactica de Atlas Finance V2.8 (solo lectura en revision).

Comandos conversacionales deterministas, sin LLM y sin scheduler:
- "añade a seguimiento IDENTIFICADOR core|tactical [nota: ...]"
- "lista seguimiento"
- "quita de seguimiento IDENTIFICADOR"
- "revisar seguimiento"

El IDENTIFICADOR es el `provider_symbol` EXACTO que devuelve "buscar
activo" del proveedor (Alpha Vantage): tanto si incluye bolsa
(VUSA.AMS) como si no (IBM). No se exige un sufijo ".BOLSA"
artificial y nunca se inventa uno: el identificador se consulta ante
el proveedor tal cual.

La watchlist vive en finance/watchlist/store.py, en un fichero JSON
atomico y versionado separado de la cartera PAPER. La revision reutiliza
el analisis cuantitativo existente (V2.5): una consulta de serie diaria
de Alpha Vantage (adaptador V2.4) y los calculos puros de
finance.quant.metrics por activo, con maximo 5 activos por ejecucion y
una consulta diaria por activo, usando exclusivamente el provider_symbol
guardado. Si Alpha Vantage limita o falla un activo, se informa ese
activo y la revision continua con los demas. La revision es de solo
lectura: no genera recomendaciones, no crea ordenes, no registra
MarketEvent, no produce fills ni modifica la cartera paper.
"""

from __future__ import annotations

import re
import unicodedata
from datetime import date, datetime
from decimal import Decimal

from finance.paper.policy import PaperMode
from finance.quant.metrics import build_quant_snapshot
from finance.watchlist.store import (
    WatchlistStore,
    WatchlistStoreError,
    normalize_note,
    normalize_provider_symbol,
)
from tools.alpha_vantage import AlphaVantageClient, AlphaVantageError

WATCHLIST_LABEL = "[WATCHLIST]"

REVIEW_SOURCE = "alpha_vantage_daily"

MAX_REVIEW_SYMBOLS_PER_RUN = 5

_NOTE_PREFIX_PATTERN = re.compile(r"(?:nota|notas)\s*:\s*(?P<note>.+)$")

_ADD_PATTERN = re.compile(
    r"^\s*(?:añade|anade|agrega|add)\s+(?:a\s+)?seguimiento\s+"
    r"(?P<target>[A-Za-z0-9.\-]{1,25})\s+"
    r"(?P<mode>core|tactical)\b"
    r"(?P<rest>.*)$",
    re.IGNORECASE,
)

_REMOVE_PATTERN = re.compile(
    r"^\s*(?:quita|elimina|borra|remove)\s+(?:del?\s+)?seguimiento\s+"
    r"(?P<target>[A-Za-z0-9.\-]{1,25})\s*[?.!]*\s*$",
    re.IGNORECASE,
)

_LIST_PATTERN = re.compile(
    r"^\s*(?:lista|listar|muestra|ver)\s+(?:la\s+)?seguimiento\s*[?.!]*\s*$"
    r"|^\s*seguimiento\s*[?.!]*\s*$",
    re.IGNORECASE,
)

_REVIEW_PATTERN = re.compile(
    r"^\s*(?:revisa|revisar|revisa\s+la\s+watchlist|revisar\s+la\s+watchlist|"
    r"revision\s+de\s+(?:la\s+)?seguimiento|revisar\s+seguimiento)\s*[?.!]*\s*$",
    re.IGNORECASE,
)

_HELP_TEXT = (
    "Comandos de seguimiento (uno por mensaje):\n"
    "- añade a seguimiento IDENTIFICADOR core|tactical\n"
    "- añade a seguimiento IDENTIFICADOR core|tactical nota: texto corto\n"
    "- lista seguimiento\n"
    "- quita de seguimiento IDENTIFICADOR\n"
    "- revisar seguimiento\n\n"
    "El IDENTIFICADOR es el simbolo exacto del proveedor tal y como lo "
    "devuelve \"buscar activo\": con bolsa si el proveedor la incluye "
    "(por ejemplo VUSA.AMS) o sin ella si no (por ejemplo IBM). No se "
    "exige un sufijo .BOLSA artificial. La revision consulta Alpha "
    "Vantage bajo peticion explicita: maximo 5 activos por ejecucion y "
    "una consulta diaria por activo."
)

_ERROR_TEXTS = {
    "MISSING_KEY": (
        "Alpha Vantage no esta configurado: falta la variable "
        "ALPHAVANTAGE_API_KEY. No se consulta ningun dato ni se calcula "
        "nada para este activo."
    ),
    "RATE_LIMIT": (
        "Limite de peticiones de Alpha Vantage alcanzado para este activo."
    ),
    "INVALID_SYMBOL": (
        "Alpha Vantage no reconoce este simbolo en el proveedor."
    ),
    "INVALID_RESPONSE": (
        "La respuesta de Alpha Vantage no es valida para este activo."
    ),
    "NETWORK": "Error de red consultando Alpha Vantage para este activo.",
    "HTTP_STATUS": "Alpha Vantage devolvio un error HTTP para este activo.",
    "RESPONSE_TOO_LARGE": (
        "La respuesta de Alpha Vantage excede el limite de tamaño para "
        "este activo."
    ),
}

_NEEDS_REVIEW_TEXT = (
    "AVISO: identificador heredado no verificable frente al proveedor "
    "(simbolo+bolsa del esquema anterior). Eliminalo con \"quita de "
    "seguimiento {symbol}\" y vuelve a darlo de alta con el identificador "
    "exacto de \"buscar activo\"."
)


class WatchlistChat:
    """Maneja los turnos de watchlist sin LLM, sin scheduler y sin paper."""

    def __init__(
        self,
        store: WatchlistStore,
        market_client: AlphaVantageClient | None = None,
        *,
        now_provider=None,
    ) -> None:
        self._store = store
        self._market_client = market_client
        self._now_provider = now_provider or (
            lambda: datetime.now().astimezone()
        )

    @property
    def store(self) -> WatchlistStore:
        return self._store

    def handles(self, prompt: str) -> bool:
        """True solo ante un comando explicito de seguimiento."""
        return classify_watchlist_prompt(prompt) is not None

    def handle(self, prompt: str) -> str:
        """Ejecuta un turno de watchlist y devuelve el texto visible."""
        intent = classify_watchlist_prompt(prompt)
        if intent is None:
            raise ValueError(f"turno de seguimiento no reconocido: {prompt!r}")
        if intent == "add":
            return self._handle_add(prompt)
        if intent == "remove":
            return self._handle_remove(prompt)
        if intent == "list":
            return self.list_text()
        if intent == "review":
            return self._handle_review()
        return self._help_text()

    # --- alta ---------------------------------------------------------------

    def _handle_add(self, prompt: str) -> str:
        match = _ADD_PATTERN.match(prompt)
        if match is None:
            return self._help_text()
        target = match.group("target").upper()
        try:
            provider_symbol = normalize_provider_symbol(target)
        except WatchlistStoreError as error:
            return (
                f"{WATCHLIST_LABEL} Alta en seguimiento rechazada: "
                f"{error} {_HELP_TEXT} Nada se ha modificado. "
                f"{WATCHLIST_LABEL}"
            )
        mode = PaperMode("CORE" if match.group("mode").lower() == "core" else "TACTICAL")
        note: str | None = None
        rest = (match.group("rest") or "").strip()
        if rest:
            note_match = _NOTE_PREFIX_PATTERN.search(rest)
            if note_match is None:
                return self._help_text()
            try:
                note = normalize_note(note_match.group("note"))
            except WatchlistStoreError as error:
                return (
                    f"{WATCHLIST_LABEL} Alta en seguimiento rechazada: "
                    f"{error} Nada se ha modificado. {WATCHLIST_LABEL}"
                )
        try:
            entry = self._store.add_entry(
                provider_symbol=provider_symbol,
                mode=mode,
                note=note,
                added_on=self._now_provider().date().isoformat(),
            )
        except WatchlistStoreError as error:
            return (
                f"{WATCHLIST_LABEL} Alta en seguimiento no realizada: "
                f"{error} Nada se ha modificado. {WATCHLIST_LABEL}"
            )
        lines = [
            f"{WATCHLIST_LABEL} Alta en seguimiento registrada:",
            "",
            f"- Simbolo: {entry['provider_symbol']}",
            f"- Modo: {entry['mode']}",
        ]
        if entry["note"]:
            lines.append(f"- Nota: {entry['note']}")
        lines.append(f"- Fecha de alta: {entry['added_on']}")
        lines.append("")
        lines.append(
            "Usa \"lista seguimiento\" para ver la lista y \"revisar "
            f"seguimiento\" para revisarla bajo peticion explicita. "
            f"{WATCHLIST_LABEL}"
        )
        return "\n".join(lines)

    # --- baja ---------------------------------------------------------------

    def _handle_remove(self, prompt: str) -> str:
        match = _REMOVE_PATTERN.match(prompt)
        if match is None:
            return self._help_text()
        target = match.group("target").upper()
        try:
            provider_symbol = normalize_provider_symbol(target)
            removed = self._store.remove_entry(provider_symbol)
        except WatchlistStoreError as error:
            return (
                f"{WATCHLIST_LABEL} Baja de seguimiento no realizada: "
                f"{error} Nada se ha modificado. {WATCHLIST_LABEL}"
            )
        return (
            f"{WATCHLIST_LABEL} Baja de seguimiento registrada:\n"
            f"\n"
            f"- Simbolo: {removed['provider_symbol']}\n"
            f"- Modo: {removed['mode']}\n"
            f"- Dado de alta el: {removed['added_on']}\n"
            f"\n"
            f"La cartera paper y el resto de la lista no se han modificado. "
            f"{WATCHLIST_LABEL}"
        )

    # --- listado ------------------------------------------------------------

    def list_text(self) -> str:
        entries = self._store.entries()
        if not entries:
            return (
                f"{WATCHLIST_LABEL} La watchlist esta vacia: no hay ningun "
                f"activo en seguimiento. Añade uno con, por ejemplo, "
                f"\"añade a seguimiento VUSA.AMS tactical\" o \"añade a "
                f"seguimiento IBM core\". Nada se ha modificado. "
                f"{WATCHLIST_LABEL}"
            )
        lines = [
            f"{WATCHLIST_LABEL} Watchlist de seguimiento "
            f"({len(entries)} activos):",
            "",
        ]
        for index, entry in enumerate(entries, start=1):
            line = (
                f"{index}. {entry['provider_symbol']} · "
                f"modo {entry['mode']} · alta {entry['added_on']}"
            )
            last_reviewed = entry.get("last_reviewed")
            if last_reviewed:
                line += f" · ultima revision {last_reviewed}"
            if entry["note"]:
                line += f" · nota: {entry['note']}"
            lines.append(line)
            if entry.get("needs_review"):
                lines.append(
                    f"   AVISO: {entry['provider_symbol']} proviene del "
                    "esquema anterior (simbolo+bolsa) y aun no se ha "
                    "verificado frente al proveedor; puede no ser un "
                    "identificador valido para Alpha Vantage. "
                    "\"revisar seguimiento\" lo verificara: si Alpha "
                    "Vantage lo reconoce, el aviso desaparecera. Si no, "
                    f"eliminalo con \"quita de seguimiento "
                    f"{entry['provider_symbol']}\" y vuelvelo a dar de "
                    "alta con el identificador exacto de \"buscar "
                    "activo\"."
                )
        lines.append("")
        lines.append(
            "Consulta de solo lectura; la revision cuantitativa solo "
            "ocurre con \"revisar seguimiento\". " f"{WATCHLIST_LABEL}"
        )
        return "\n".join(lines)

    # --- revision -------------------------------------------------------------

    def _handle_review(self) -> str:
        entries = self._store.entries()
        if not entries:
            return (
                f"{WATCHLIST_LABEL} No hay revision: la watchlist esta "
                f"vacia. Añade activos primero con, por ejemplo, \"añade a "
                f"seguimiento VUSA.AMS tactical\" o \"añade a seguimiento "
                f"IBM core\". No se consulta ningun proveedor y no se "
                f"calcula nada. {WATCHLIST_LABEL}"
            )
        client = self._market_client
        if client is None:
            return (
                f"{WATCHLIST_LABEL} Revision no disponible: no hay "
                f"adaptador de Alpha Vantage configurado en esta "
                f"instalacion. No se consulta ningun proveedor y no se "
                f"calcula nada. {WATCHLIST_LABEL}"
            )
        today = self._now_provider().date()
        today_key = today.isoformat()
        pending = [
            entry
            for entry in entries
            if entry.get("last_reviewed") != today_key
        ]
        already_reviewed = [
            entry
            for entry in entries
            if entry.get("last_reviewed") == today_key
        ]
        selected = pending[:MAX_REVIEW_SYMBOLS_PER_RUN]
        deferred_count = len(pending) - len(selected)
        results: list[_ReviewResult] = []
        for entry in selected:
            results.append(self._review_entry(entry, client, today))
        return _compose_review_report(
            results,
            already_reviewed=already_reviewed,
            deferred_count=deferred_count,
            today=today,
            total=len(entries),
        )

    def _review_entry(
        self,
        entry: dict,
        client: AlphaVantageClient,
        today: date,
    ) -> "_ReviewResult":
        # La revision usa EXCLUSIVAMENTE el provider_symbol guardado:
        # nunca se elimina ni se altera su sufijo.
        provider_symbol = entry["provider_symbol"]
        try:
            series = client.daily_series(provider_symbol)
        except AlphaVantageError as error:
            return _ReviewResult(entry=entry, error=_error_text(error.code))
        try:
            snapshot = build_quant_snapshot(series)
        except ValueError as error:
            return _ReviewResult(entry=entry, error=f"serie invalida: {error}")
        try:
            self._store.mark_reviewed(provider_symbol, today.isoformat())
        except WatchlistStoreError:
            return _ReviewResult(
                entry=entry,
                error="la watchlist no pudo registrar la revision de este activo",
            )
        label = "DATOS DIARIOS" if snapshot.last_day >= today else "RETRASADOS"
        return _ReviewResult(entry=entry, snapshot=snapshot, label=label)

    def _help_text(self) -> str:
        return (
            f"{WATCHLIST_LABEL} Sintaxis de seguimiento no reconocida. "
            f"{_HELP_TEXT} Nada se ha modificado. {WATCHLIST_LABEL}"
        )


class _ReviewResult:
    """Resultado de la revision de un activo (exito o error controlado)."""

    def __init__(
        self,
        *,
        entry: dict,
        snapshot=None,
        label: str | None = None,
        error: str | None = None,
    ) -> None:
        self.entry = entry
        self.snapshot = snapshot
        self.label = label
        self.error = error

    @property
    def ok(self) -> bool:
        return self.error is None


def _compose_review_report(
    results: list[_ReviewResult],
    *,
    already_reviewed: list[dict],
    deferred_count: int,
    today: date,
    total: int,
) -> str:
    lines = [
        f"{WATCHLIST_LABEL} Revision de seguimiento ({today.isoformat()}):",
        "",
    ]
    if results:
        lines.append(
            f"- Activos consultados en esta ejecucion: {len(results)} "
            f"de {total} en seguimiento."
        )
    elif already_reviewed:
        lines.append(
            "Ningun activo pendiente: todos los activos en seguimiento ya "
            "se han revisado hoy (una consulta diaria por activo)."
        )
    if deferred_count > 0:
        lines.append(
            f"- Aplazados para la proxima revision: {deferred_count} "
            f"(maximo {MAX_REVIEW_SYMBOLS_PER_RUN} activos por ejecucion)."
        )
    if already_reviewed:
        already = ", ".join(
            entry["provider_symbol"] for entry in already_reviewed
        )
        lines.append(
            f"- Ya revisados hoy (una consulta diaria por activo): {already}."
        )
    lines.append("")
    for result in results:
        entry = result.entry
        header = (
            f"- {entry['provider_symbol']} · modo {entry['mode']}"
        )
        if not result.ok:
            lines.append(f"{header} · ERROR: {result.error}")
            continue
        snapshot = result.snapshot
        assert snapshot is not None  # noqa: S101 - garantizado por result.ok
        lines.append(
            f"{header} · dato {snapshot.last_day.isoformat()} "
            f"({result.label}) · 5 sesiones "
            f"{_fmt_pct_or_na(snapshot.change_5)} · 20 sesiones "
            f"{_fmt_pct_or_na(snapshot.change_20)} · tendencia {snapshot.trend}"
        )
        if snapshot.unavailable:
            lines.append(
                f"  · metricas no calculables: "
                f"{'; '.join(snapshot.unavailable)}"
            )
    errors = [result for result in results if not result.ok]
    lines.append("")
    if errors:
        lines.append(
            "Los activos con error no se han revisado; el resto si. La "
            "revision continua a pesar de fallos parciales."
        )
    lines.append(
        f"Fuente: {REVIEW_SOURCE} (Alpha Vantage, serie diaria); dato "
        "diario/retrasado, no tiempo real."
    )
    lines.append(
        "Revision de solo lectura: no genera recomendaciones de compra o "
        "venta, no crea ordenes, no registra MarketEvent, no produce fills "
        "ni modifica la cartera paper. " f"{WATCHLIST_LABEL}"
    )
    return "\n".join(lines)


def _fmt_pct_or_na(fraction: Decimal | None) -> str:
    if fraction is None:
        return "no calculable"
    percent = (fraction * 100).quantize(Decimal("0.01"))
    text = f"{percent:f}"
    if percent > 0:
        text = f"+{text}"
    return f"{text} %"


def _error_text(code: str) -> str:
    return _ERROR_TEXTS.get(
        code, "Error consultando Alpha Vantage para este activo."
    )


def classify_watchlist_prompt(prompt: str) -> "str | None":
    """Clasificador puro de comandos de seguimiento; None si no lo es."""
    text = (prompt or "").strip()
    if not text or "\n" in text:
        return None
    normalized = _normalize(text)
    if _REVIEW_PATTERN.match(text) or _REVIEW_PATTERN.match(normalized):
        return "review"
    if _LIST_PATTERN.match(text) or _LIST_PATTERN.match(normalized):
        return "list"
    if _REMOVE_PATTERN.match(text) or _REMOVE_PATTERN.match(normalized):
        return "remove"
    if _ADD_PATTERN.match(text) or _ADD_PATTERN.match(normalized):
        return "add"
    if (
        "seguimiento" in normalized
        or "watchlist" in normalized
    ):
        return "help"
    return None


def handles_watchlist_prompt(prompt: str) -> bool:
    """Clasificador puro: True solo ante un comando explicito de seguimiento."""
    return classify_watchlist_prompt(prompt) is not None


def _normalize(text: str) -> str:
    """Minúsculas sin acentos para tolerar 'añade'/'anade' y tildes."""
    decomposed = unicodedata.normalize("NFKD", text.casefold())
    return "".join(
        character
        for character in decomposed
        if unicodedata.category(character) != "Mn"
    )
