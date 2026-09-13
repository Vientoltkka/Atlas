"""Chat paper workflow de Atlas Finance V2.2/V2.6.

Glue conversacional determinista sobre el PaperFinanceService existente
(V2.1): consulta de cartera paper (solo lectura), registro de un precio
paper declarado por el usuario (origen user_declared), creacion de
propuestas de orden paper legibles y, desde V2.6, importacion explicita
del ultimo cierre diario de Alpha Vantage a paper tras confirmacion
(fuente alpha_vantage_daily; la importacion nunca crea ordenes). La
ejecucion exige confirmacion explicita del usuario en el turno siguiente;
sin web en tiempo real, sin broker y sin dinero real. No replique el
motor ni la persistencia.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, replace
from datetime import date, datetime, timedelta
from decimal import Decimal, InvalidOperation

from finance.paper.models import (
    EventType,
    MarketEvent,
    OrderType,
    PaperOrder,
    Side,
    utc_now,
)
from finance.paper.service import PaperFinanceService
from tools.alpha_vantage import AlphaVantageClient, AlphaVantageError

PAPER_LABEL = "[PAPER]"

PENDING_ORDER_TTL = timedelta(minutes=15)

IMPORT_SOURCE = "alpha_vantage_daily"

_SIDE_WORDS = {
    "compra": Side.BUY,
    "comprar": Side.BUY,
    "vende": Side.SELL,
    "vender": Side.SELL,
}

_SYMBOL_STOPWORDS = {
    "paper",
    "mercado",
    "limite",
    "accion",
    "acciones",
    "unidades",
    "unidad",
    "uds",
    "de",
    "del",
    "la",
    "el",
    "a",
    "en",
    "con",
    "por",
    "tus",
    "mis",
}

_PRICE_INTENT_WORDS = ("registra", "registrar", "anota", "anotar", "declara", "declarar")
_PAPER_MARKERS = ("paper", "simulad")
_COMMAND_HINT_WORDS = (
    "compra",
    "comprar",
    "vende",
    "vender",
    "venta",
    "precio",
    "tick",
    "cartera",
    "portfolio",
    "orden",
    "ordenes",
)


@dataclass(frozen=True)
class PendingPaperProposal:
    """Una orden paper propuesta y pendiente de confirmacion del usuario."""

    order: PaperOrder
    text: str
    created_at: datetime
    status: str = "PENDING"

    @property
    def active(self) -> bool:
        """True solo mientras la propuesta sigue esperando confirmacion."""
        return self.status == "PENDING"


@dataclass(frozen=True)
class PendingPaperImport:
    """Un cierre diario de mercado propuesto para importar a paper.

    La importacion solo registra un MarketEvent tras confirmacion
    explicita; nunca crea ni ejecuta ordenes.
    """

    symbol: str
    price: Decimal
    provider_date: date
    provider_refreshed: str
    provider_timezone: str
    text: str
    created_at: datetime
    status: str = "PENDING"

    @property
    def active(self) -> bool:
        return self.status == "PENDING"


class PaperFinanceChat:
    """Maneja los turnos paper del FinanceAgent sin LLM ni datos externos."""

    def __init__(
        self,
        service: PaperFinanceService,
        *,
        market_client: AlphaVantageClient | None = None,
        market_client_factory=None,
    ) -> None:
        self._service = service
        self._market_client = market_client
        self._market_client_factory = market_client_factory
        self._pending: "PendingPaperProposal | PendingPaperImport | None" = None

    @property
    def service(self) -> PaperFinanceService:
        return self._service

    @property
    def has_market_client(self) -> bool:
        """True si hay cliente de mercado resuelto o factory pendiente."""
        return self._market_client is not None or self._market_client_factory is not None

    def set_market_client(self, client: AlphaVantageClient) -> None:
        """Inyecta el cliente de mercado de solo lectura (V2.4)."""
        self._market_client = client

    def set_market_client_factory(self, factory) -> None:
        """Inyecta una factory perezosa del cliente de mercado (V2.4)."""
        if self._market_client is None:
            self._market_client_factory = factory

    def _resolve_market_client(self) -> AlphaVantageClient | None:
        if self._market_client is None and self._market_client_factory is not None:
            self._market_client = self._market_client_factory()
        return self._market_client

    @property
    def has_pending(self) -> bool:
        """True while any paper proposal state exists (active or discarded)."""
        return self._pending is not None

    @property
    def pending_state(self) -> "PendingPaperProposal | PendingPaperImport | None":
        """Return the active pending state (order or import), or None."""
        if self._pending is None:
            return None
        if self._pending_expired() and self._pending.active:
            self._pending = replace(self._pending, status="EXPIRED")
        if not self._pending.active:
            return None
        return self._pending

    @property
    def pending_proposal(self) -> PendingPaperProposal | None:
        """Return the active paper order proposal, or None once discarded/expired."""
        pending = self.pending_state
        if isinstance(pending, PendingPaperProposal):
            return pending
        return None

    @property
    def pending_import(self) -> PendingPaperImport | None:
        """Return the active paper import proposal, or None once discarded/expired."""
        pending = self.pending_state
        if isinstance(pending, PendingPaperImport):
            return pending
        return None

    def handles(self, prompt: str) -> bool:
        """Return True when the prompt is an explicit paper chat command."""
        return _classify_paper_intent(prompt) is not None

    def handle(self, prompt: str) -> str:
        """Execute one paper chat turn and return the visible text."""
        intent = _classify_paper_intent(prompt)
        if intent == "multiline":
            return (
                f"{PAPER_LABEL} Mensaje con varias lineas: envia un comando "
                f"paper por mensaje para procesarlo de forma determinista "
                f"(por ejemplo: \"cartera paper\", \"registra precio paper "
                f"AAPL 185.50\", \"compra paper 10 de AAPL a mercado\"). No se "
                f"ha simulado nada con el modelo y la cartera paper no se ha "
                f"modificado. {PAPER_LABEL}"
            )
        if intent == "unknown":
            return (
                f"{PAPER_LABEL} No he identificado el comando paper. Comandos "
                f"disponibles (uno por mensaje):\n"
                f"- cartera paper\n"
                f"- registra precio paper SIMBOLO PRECIO\n"
                f"- precio paper SIMBOLO PRECIO (forma corta)\n"
                f"- importa precio mercado SIMBOLO a paper\n"
                f"- actualiza precio paper SIMBOLO desde mercado\n"
                f"- compra paper CANTIDAD de SIMBOLO a mercado\n"
                f"- compra paper CANTIDAD de SIMBOLO con limite PRECIO\n"
                f"- vende paper CANTIDAD de SIMBOLO a mercado\n\n"
                f"La cartera paper no se ha modificado. {PAPER_LABEL}"
            )
        if intent == "portfolio":
            return self.portfolio_text()
        if intent == "price":
            return self._handle_declared_price(prompt)
        if intent == "import":
            return self._handle_market_import(prompt)
        if intent == "order":
            return self._handle_order_request(prompt)
        raise ValueError(f"turno paper no reconocido: {prompt!r}")

    def describe_pending(self) -> str:
        pending = self.pending_proposal
        if pending is None:
            return self._discarded_message()
        return (
            f"{pending.text}\n\nLa propuesta sigue pendiente: responde \"si\" "
            f"para ejecutarla o \"no\" para cancelarla. {PAPER_LABEL}"
        )

    def execute_pending(self) -> str:
        """Execute the confirmed pending state (order or import), or expire it."""
        pending = self.pending_state
        if pending is None:
            return self._discarded_message()
        if isinstance(pending, PendingPaperImport):
            blocked = self._import_blocked_by_pending_orders()
            if blocked is not None:
                self._pending = replace(pending, status="CANCELLED")
                return self._import_blocked_execution_text()
            self._pending = replace(pending, status="CONSUMED")
            return self._execute_import(pending)
        self._pending = replace(pending, status="CONSUMED")
        order = pending.order
        try:
            decision = self._service.execute_confirmed_order(order)
        except ValueError as error:
            return (
                f"{PAPER_LABEL} Orden paper no ejecutada: {error} "
                f"La cartera no se ha modificado. {PAPER_LABEL}"
            )
        header = f"{PAPER_LABEL} Orden paper ejecutada con tu confirmacion (simulacion):"
        if decision.filled and decision.fill is not None:
            fill = decision.fill
            side = "COMPRA" if fill.side is Side.BUY else "VENTA"
            return (
                f"{header}\n"
                f"- Orden: {decision.order_id} · Estado: FILLED\n"
                f"- {side} {fill.qty} {fill.symbol} a {_fmt_price(fill.price)}\n"
                f"- Comision: {_fmt_money(fill.commission)} · "
                f"Slippage: {fill.slippage_bps} bps\n\n"
                f"Cartera paper actualizada. Etiqueta: PAPER, sin dinero real. {PAPER_LABEL}"
            )
        if decision.pending:
            return (
                f"{header}\n"
                f"- Orden: {decision.order_id} · Estado: PENDING\n"
                f"- Motivo: {decision.reason}\n\n"
                f"La orden paper queda registrada en el motor y se evaluara con "
                f"los ticks paper que declares. Etiqueta: PAPER. {PAPER_LABEL}"
            )
        return (
            f"{PAPER_LABEL} Orden paper no ejecutada (simulacion):\n"
            f"- Orden: {decision.order_id} · Estado: {decision.status.value}\n"
            f"- Motivo: {decision.reason}\n\n"
            f"La cartera paper no se ha modificado. Etiqueta: PAPER. {PAPER_LABEL}"
        )

    def cancel_pending(self) -> str:
        pending = self.pending_state
        if pending is not None:
            self._pending = replace(pending, status="CANCELLED")
        return self._discarded_message()

    def _discarded_message(self) -> str:
        noun = "de orden paper"
        if isinstance(self._pending, PendingPaperImport):
            noun = "de importacion de precio de mercado"
        if self._pending is not None and self._pending.status == "EXPIRED":
            return (
                f"{PAPER_LABEL} La propuesta {noun} ha expirado. "
                f"La cartera paper no se ha modificado. {PAPER_LABEL}"
            )
        return (
            f"{PAPER_LABEL} La propuesta {noun} ya fue descartada. "
            f"No hay ninguna orden pendiente y la cartera paper no se ha "
            f"modificado. {PAPER_LABEL}"
        )

    def portfolio_text(self) -> str:
        summary = self._service.portfolio_summary()
        lines = [
            f"{PAPER_LABEL} Cartera paper (simulacion, sin dinero real):",
            "",
            f"- Efectivo: {_fmt_money_str(str(summary['cash']))}",
            f"- NAV: {_fmt_money_str(str(summary['nav']))}",
        ]
        positions = summary["positions"]
        if positions:
            lines.append("- Posiciones:")
            for symbol, position in positions.items():
                lines.append(
                    f"  - {symbol}: {position['qty']} uds · "
                    f"precio medio {_fmt_price_str(str(position['avg_price']))} · "
                    f"ultimo precio {_fmt_price_str(str(position['last_price']))}"
                )
        else:
            lines.append("- Posiciones: ninguna")
        lines.append(f"- Actualizado: {summary['updated_at']}")
        lines.append("")
        lines.append(
            f"Consulta de solo lectura. Etiqueta: PAPER, sin dinero real. {PAPER_LABEL}"
        )
        return "\n".join(lines)

    def _handle_declared_price(self, prompt: str) -> str:
        match = _match_price(prompt)
        if match is None:
            return (
                f"{PAPER_LABEL} No he entendido el precio paper a registrar. "
                f"Usa por ejemplo: \"precio paper AAPL 185.50\" o \"registra "
                f"precio paper AAPL 185.50\". "
                f"La cartera no se ha modificado. {PAPER_LABEL}"
            )
        symbol = match.group("symbol").upper()
        price = _parse_number(match.group("price"))
        if price is None or price <= 0:
            return (
                f"{PAPER_LABEL} El precio paper declarado no es valido "
                f"({match.group('price')}). Debe ser un numero positivo. "
                f"La cartera no se ha modificado. {PAPER_LABEL}"
            )
        event = MarketEvent(
            symbol=symbol,
            price=price,
            timestamp=utc_now(),
            event_type=EventType.DECLARED,
        )
        try:
            ingested = self._service.record_market_event(event)
        except ValueError as error:
            return (
                f"{PAPER_LABEL} No he podido registrar el tick paper: {error} "
                f"La cartera no se ha modificado. {PAPER_LABEL}"
            )
        registered_at = utc_now().isoformat()
        if not ingested:
            return (
                f"{PAPER_LABEL} Ese tick paper ya estaba registrado "
                f"(idempotencia por evento). Simbolo {event.symbol} · precio "
                f"{_fmt_price(price)} · origen: user_declared. {PAPER_LABEL}"
            )
        return (
            f"{PAPER_LABEL} Tick paper registrado:\n"
            f"- Simbolo: {event.symbol}\n"
            f"- Precio: {_fmt_price(price)}\n"
            f"- Origen: user_declared\n"
            f"- Registrado: {registered_at}\n\n"
            f"Es un dato declarado por ti para la simulacion paper; no procede "
            f"de ningun mercado real. {PAPER_LABEL}"
        )

    def _handle_market_import(self, prompt: str) -> str:
        match = _match_import(prompt)
        if match is None:
            return (
                f"{PAPER_LABEL} No he entendido la importacion de precio de "
                f"mercado. Usa por ejemplo: \"importa precio mercado IBM a "
                f"paper\" o \"actualiza precio paper IBM desde mercado\". "
                f"La cartera paper no se ha modificado. {PAPER_LABEL}"
            )
        order_pending = self.pending_state
        if isinstance(order_pending, PendingPaperProposal):
            return (
                f"{PAPER_LABEL} Ya hay una propuesta de orden paper pendiente: "
                f"responde \"si\" para ejecutarla o \"no\" para descartarla "
                f"antes de importar precios de mercado. La cartera paper no "
                f"se ha modificado. {PAPER_LABEL}"
            )
        blocked = self._import_blocked_by_pending_orders()
        if blocked is not None:
            return blocked
        symbol = match.group("symbol").upper()
        client = self._resolve_market_client()
        if client is None:
            return (
                f"{PAPER_LABEL} Importacion de precio de mercado no "
                f"disponible: no hay adaptador de Alpha Vantage configurado "
                f"en esta instalacion. No se consulta ningun proveedor, no "
                f"se registra ningun tick y la cartera paper no se ha "
                f"modificado. {PAPER_LABEL}"
            )
        try:
            series = client.daily_series(symbol)
        except AlphaVantageError as error:
            return _import_error_text(error)
        last = series.last_bar
        if last is None or last.close <= 0:
            return (
                f"{PAPER_LABEL} La serie diaria de {series.symbol} en Alpha "
                f"Vantage no contiene un cierre diario valido. No se registra "
                f"ningun tick y la cartera paper no se ha modificado. "
                f"{PAPER_LABEL}"
            )
        pending = PendingPaperImport(
            symbol=series.symbol,
            price=last.close,
            provider_date=last.day,
            provider_refreshed=series.provider_last_refreshed,
            provider_timezone=series.provider_timezone,
            created_at=utc_now(),
            text=(
                f"{PAPER_LABEL} Propuesta de importacion de precio de mercado "
                f"a paper, no ejecutada todavia:\n"
                f"\n"
                f"- Simbolo: {series.symbol}\n"
                f"- Cierre diario disponible: {_fmt_price(last.close)}\n"
                f"- Fecha del proveedor: {last.day.isoformat()}\n"
                f"- Fuente: {IMPORT_SOURCE}\n"
                f"- Aviso: dato diario retrasado, no es tiempo real.\n"
                f"- La importacion solo registra un tick paper; no crea ni "
                f"ejecuta ninguna orden.\n"
                f"\n"
                f"Responde \"si\" para registrar el tick paper o \"no\" para "
                f"cancelarla. La cartera paper no se ha modificado. "
                f"{PAPER_LABEL}"
            ),
        )
        self._pending = pending
        return pending.text

    def _import_blocked_by_pending_orders(self) -> str | None:
        limits = [
            order
            for order in self._service.pending_orders()
            if order.order_type is OrderType.LIMIT
        ]
        if not limits:
            return None
        detail = ", ".join(f"{order.order_id} ({order.symbol})" for order in limits)
        return (
            f"{PAPER_LABEL} Importacion de precio de mercado rechazada: hay "
            f"ordenes LIMIT paper pendientes ({detail}). Cancelalas o "
            f"resuelvelas explicitamente antes de importar precios de "
            f"mercado: una importacion no puede causar fills indirectos. "
            f"No se consulta el proveedor ni se registra ningun tick y la "
            f"cartera paper no se ha modificado. {PAPER_LABEL}"
        )

    def _import_blocked_execution_text(self) -> str:
        return (
            f"{PAPER_LABEL} Importacion de precio de mercado no realizada: "
            f"hay ordenes LIMIT paper pendientes. Cancelalas o resuelvelas "
            f"explicitamente antes de importar precios de mercado: una "
            f"importacion no puede causar fills indirectos. No se ha "
            f"registrado ningun tick y la cartera paper no se ha modificado. "
            f"{PAPER_LABEL}"
        )

    def _execute_import(self, pending: PendingPaperImport) -> str:
        event = MarketEvent(
            symbol=pending.symbol,
            price=pending.price,
            timestamp=utc_now(),
            event_type=EventType.TICK,
            source=IMPORT_SOURCE,
            provider_date=pending.provider_date,
        )
        try:
            ingested = self._service.record_market_event(event)
        except ValueError as error:
            return (
                f"{PAPER_LABEL} No he podido registrar el tick paper "
                f"importado: {error} La cartera paper no se ha modificado. "
                f"{PAPER_LABEL}"
            )
        if not ingested:
            return (
                f"{PAPER_LABEL} Ese tick paper ya estaba registrado "
                f"(idempotencia por evento). Simbolo {event.symbol} · fuente "
                f"{IMPORT_SOURCE}. {PAPER_LABEL}"
            )
        refreshed = pending.provider_refreshed or pending.provider_date.isoformat()
        if pending.provider_timezone:
            refreshed = f"{refreshed} ({pending.provider_timezone})"
        return (
            f"{PAPER_LABEL} Tick paper importado con tu confirmacion "
            f"(simulacion):\n"
            f"\n"
            f"- Simbolo: {event.symbol}\n"
            f"- Precio: {_fmt_price(event.price)}\n"
            f"- Fecha del proveedor: {pending.provider_date.isoformat()}\n"
            f"- Fuente: {IMPORT_SOURCE} ({refreshed})\n"
            f"- Aviso: dato diario retrasado, no es tiempo real.\n"
            f"- Registrado: {utc_now().isoformat()}\n"
            f"\n"
            f"La importacion no crea ni ejecuta ordenes: cualquier compra o "
            f"venta paper seguira requiriendo su propia solicitud y una "
            f"segunda confirmacion independiente. Etiqueta: PAPER, sin "
            f"dinero real. {PAPER_LABEL}"
        )

    def _handle_order_request(self, prompt: str) -> str:
        match = _match_order(prompt)
        if match is None:
            return (
                f"{PAPER_LABEL} No he entendido la orden paper. Indica accion, "
                f"cantidad y simbolo, por ejemplo: \"compra paper 10 de AAPL a "
                f"mercado\" o \"compra paper 10 de AAPL con limite 180\". "
                f"La cartera no se ha modificado. {PAPER_LABEL}"
            )
        side = _SIDE_WORDS[match.group("side").lower()]
        qty = _parse_number(match.group("qty"))
        if qty is None or qty <= 0:
            return (
                f"{PAPER_LABEL} La cantidad de la orden paper no es valida "
                f"({match.group('qty')}). La cartera no se ha modificado. {PAPER_LABEL}"
            )
        symbol = match.group("symbol").upper()
        if symbol.lower() in _SYMBOL_STOPWORDS:
            return (
                f"{PAPER_LABEL} No he identificado el simbolo de la orden paper. "
                f"Indica un simbolo claro, por ejemplo: \"compra paper 10 de "
                f"AAPL a mercado\". La cartera no se ha modificado. {PAPER_LABEL}"
            )
        rest = match.group("rest") or ""
        limit_price = _extract_limit_price(rest)
        if limit_price is None and _mentions_limit(rest):
            return (
                f"{PAPER_LABEL} Para una orden LIMIT paper necesito el precio "
                f"limite, por ejemplo: \"compra paper 10 de {symbol} con limite "
                f"180\". La cartera no se ha modificado. {PAPER_LABEL}"
            )
        order_type = OrderType.LIMIT if limit_price is not None else OrderType.MARKET
        return self._propose_order(side, symbol, qty, order_type, limit_price)

    def _propose_order(
        self,
        side: Side,
        symbol: str,
        qty: Decimal,
        order_type: OrderType,
        limit_price: Decimal | None,
    ) -> str:
        try:
            order = PaperOrder(
                symbol=symbol,
                side=side,
                qty=qty,
                order_type=order_type,
                limit_price=limit_price,
            )
        except ValueError as error:
            return (
                f"{PAPER_LABEL} Orden paper invalida: {error} "
                f"La cartera no se ha modificado. {PAPER_LABEL}"
            )

        defaults = self._service.execution_defaults()
        reference_price = (
            limit_price
            if order_type is OrderType.LIMIT
            else self._service.engine.last_prices().get(symbol)
        )
        if reference_price is None:
            self._pending = None
            return (
                f"{PAPER_LABEL} Orden MARKET de {symbol} rechazada: falta un "
                f"precio paper para {symbol}. No hay ningun tick paper "
                f"registrado y una orden MARKET necesita un precio de "
                f"referencia. No invento precios. Registra primero un precio "
                f"paper, por ejemplo: \"registra precio paper {symbol} 100\". "
                f"La cartera paper no se ha modificado. {PAPER_LABEL}"
            )

        slippage_bps = defaults["slippage_bps"]
        commission = defaults["commission"]
        if side is Side.BUY:
            reference = reference_price * (Decimal("1") + slippage_bps / Decimal("10000"))
            estimated = reference * qty + commission
            estimate_label = "Coste estimado"
        else:
            reference = reference_price * (Decimal("1") - slippage_bps / Decimal("10000"))
            estimated = reference * qty - commission
            estimate_label = "Ingreso estimado"

        risk_lines = [
            "Simulacion paper sin dinero real; la ejecucion solo ocurre si "
            "confirmas.",
        ]
        if side is Side.SELL:
            position = self._service.engine.portfolio.positions.get(symbol)
            held = position.qty if position is not None else Decimal("0")
            if held < qty:
                risk_lines.append(
                    f"Posicion insuficiente: tienes {held} uds de {symbol}; el "
                    f"motor rechazaria la orden."
                )
        if order_type is OrderType.LIMIT:
            risk_lines.append(
                "La orden LIMIT paper queda pendiente en el motor hasta que un "
                "tick paper satisfaga el limite."
            )

        pending = PendingPaperProposal(
            order=order,
            text=(
                f"{PAPER_LABEL} Propuesta de orden paper, no ejecutada todavia:\n"
                f"\n"
                f"- Accion: {'COMPRA' if side is Side.BUY else 'VENTA'}\n"
                f"- Simbolo: {order.symbol}\n"
                f"- Cantidad: {order.qty}\n"
                f"- Tipo: {order_type.value}\n"
                f"- Precio de referencia (tick paper registrado): "
                f"{_fmt_price(reference_price)}\n"
                f"- {estimate_label}: {_fmt_money(estimated)}\n"
                f"- Comision: {_fmt_money(commission)} · "
                f"Slippage: {slippage_bps} bps\n"
                f"- Riesgo: {' '.join(risk_lines)}\n"
                f"\n"
                f"Responde \"si\" para ejecutar la orden paper o \"no\" para "
                f"descartarla. Nada se modifica hasta tu confirmacion. {PAPER_LABEL}"
            ),
            created_at=utc_now(),
        )
        self._pending = pending
        return pending.text

    def _pending_expired(self) -> bool:
        if self._pending is None:
            return False
        return utc_now() - self._pending.created_at > PENDING_ORDER_TTL


def _classify_paper_intent(prompt: str) -> str | None:
    """Classify one chat turn; multiline paper content is rejected, not simulated."""
    lines = [line for line in prompt.splitlines() if line.strip()]
    if len(lines) >= 2 and any(_classify_single(line) is not None for line in lines):
        return "multiline"
    return _classify_single(prompt)


def _classify_single(prompt: str) -> str | None:
    normalized = _normalize(prompt)
    if not any(marker in normalized for marker in _PAPER_MARKERS):
        return None
    if _PORTFOLIO_PATTERN.search(normalized) is not None:
        return "portfolio"
    if _match_import(prompt) is not None:
        return "import"
    if _match_order(prompt) is not None:
        return "order"
    if _match_price(prompt) is not None:
        return "price"
    if any(word in normalized for word in _PRICE_INTENT_WORDS) and (
        "precio" in normalized or "tick" in normalized
    ):
        return "price"
    if _looks_like_paper_command(normalized):
        return "unknown"
    return None


def _match_import(text: str):
    return _IMPORT_PATTERN.search(text) or _IMPORT_UPDATE_PATTERN.search(text)


def _looks_like_paper_command(normalized: str) -> bool:
    return any(word in normalized for word in _COMMAND_HINT_WORDS)


def _match_order(text: str):
    return _ORDER_PATTERN.search(text) or _ORDER_SYMBOL_FIRST_PATTERN.search(text)


def _match_price(text: str):
    return _PRICE_PATTERN.search(text) or _SHORT_PRICE_PATTERN.search(text)


def _normalize(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text.strip().lower())
    without_accents = "".join(
        character
        for character in normalized
        if unicodedata.category(character) != "Mn"
    )
    return without_accents


def _parse_number(raw: str) -> Decimal | None:
    try:
        return Decimal(raw.strip().replace(",", "."))
    except InvalidOperation:
        return None


def _mentions_limit(text: str) -> bool:
    return bool(re.search(r"\blimite\b", text))


def _extract_limit_price(text: str) -> Decimal | None:
    match = re.search(r"\blimite\b[^\d-]{0,12}(?P<price>\d+(?:[.,]\d+)?)", text)
    if match is None:
        return None
    return _parse_number(match.group("price"))


def _fmt_money(value: Decimal) -> str:
    return _fmt_money_str(f"{value.quantize(Decimal('0.01'))}")


def _fmt_money_str(raw: str) -> str:
    whole, _, fraction = raw.partition(".")
    grouped = f"{int(whole):,}".replace(",", ".")
    return f"{grouped},{fraction or '00'} €"


def _fmt_price(value: Decimal) -> str:
    return _fmt_price_str(f"{value.quantize(Decimal('0.0001'))}")


def _fmt_price_str(raw: str) -> str:
    return _fmt_money_str(raw).replace(" €", "")


_PORTFOLIO_PATTERN = re.compile(r"\b(?:cartera|portfolio|posiciones)\b")

_PRICE_PATTERN = re.compile(
    r"\b(?:registra|registrar|anota|anotar|declara|declarar)\b.{0,40}?"
    r"\b(?:precio|tick)\b(?:\s+paper)?\s*(?:de\s+|para\s+|del\s+|:)?\s*"
    r"(?P<symbol>[a-z][a-z0-9.\-]{0,9})\s*(?:a|en|de|=|:)?\s*"
    r"(?P<price>\d+(?:[.,]\d+)?)",
    re.IGNORECASE,
)

_SHORT_PRICE_PATTERN = re.compile(
    r"\b(?:precio|tick)\s+paper\b\s*(?:de\s+|para\s+|del\s+|:)?\s*"
    r"(?P<symbol>[a-z][a-z0-9.\-]{0,9})\s*(?:a|en|de|=|:)?\s*"
    r"(?P<price>\d+(?:[.,]\d+)?)",
    re.IGNORECASE,
)

_ORDER_PATTERN = re.compile(
    r"\b(?P<side>comprar|compra|vender|vende)\b\s*(?:me\s+)?(?:paper\s+)?"
    r"(?P<qty>\d+(?:[.,]\d+)?)\s*"
    r"(?:acciones?\s+|uds\.?\s+|unidades?\s+|u\s+)?"
    r"(?:de\s+|del\s+)?"
    r"(?P<symbol>[a-z][a-z0-9.\-]{0,9})\s*(?:paper)?(?P<rest>.*)$",
    re.IGNORECASE,
)

_ORDER_SYMBOL_FIRST_PATTERN = re.compile(
    r"\b(?P<side>comprar|compra|vender|vende)\b\s*(?:me\s+)?(?:paper\s+)?"
    r"(?P<symbol>[a-z][a-z0-9.\-]{0,9})\s*"
    r"(?P<qty>\d+(?:[.,]\d+)?)\s*"
    r"(?:acciones?\s+|uds\.?\s+|unidades?\s+)?"
    r"(?:de\s+|del\s+)?(?:paper)?(?P<rest>.*)$",
    re.IGNORECASE,
)

_IMPORT_PATTERN = re.compile(
    r"^\s*importar?\s+(?:el\s+)?precio\s+(?:de\s+)?mercado\s+"
    r"(?:de\s+|del\s+|para\s+|:)?\s*"
    r"(?P<symbol>[A-Za-z0-9.\-]{1,12})\s+(?:a|en|hacia)\s+(?:el\s+)?paper"
    r"\s*[?.!]*\s*$",
    re.IGNORECASE,
)

_IMPORT_UPDATE_PATTERN = re.compile(
    r"^\s*actualiza?\s+(?:el\s+)?precio\s+(?:de\s+)?paper\s+"
    r"(?P<symbol>[A-Za-z0-9.\-]{1,12})\s+desde\s+(?:el\s+)?mercado"
    r"\s*[?.!]*\s*$",
    re.IGNORECASE,
)

_IMPORT_ERROR_TEXTS = {
    "MISSING_KEY": (
        "Alpha Vantage no esta configurado: falta la variable "
        "ALPHAVANTAGE_API_KEY. No se consulta ningun proveedor ni se "
        "inventan precios."
    ),
    "RATE_LIMIT": (
        "Limite de peticiones de Alpha Vantage alcanzado. No se importa "
        "ningun precio ni se inventan precios."
    ),
    "INVALID_SYMBOL": (
        "Alpha Vantage no reconoce el simbolo solicitado. No se importa "
        "ningun precio ni se inventan precios."
    ),
    "INVALID_RESPONSE": (
        "La respuesta de Alpha Vantage no es valida. No se importa ningun "
        "precio ni se inventan precios."
    ),
    "NETWORK": (
        "Error de red consultando Alpha Vantage. No se importa ningun "
        "precio ni se inventan precios."
    ),
    "HTTP_STATUS": (
        "Alpha Vantage devolvio un error HTTP. No se importa ningun precio "
        "ni se inventan precios."
    ),
    "RESPONSE_TOO_LARGE": (
        "La respuesta de Alpha Vantage excede el limite permitido. No se "
        "importa ningun precio ni se inventan precios."
    ),
}


def _import_error_text(error: AlphaVantageError) -> str:
    message = _IMPORT_ERROR_TEXTS.get(
        error.code,
        "Error consultando Alpha Vantage. No se importan precios ni se "
        "inventan precios.",
    )
    return (
        f"{PAPER_LABEL} {message} No se registra ningun tick y la cartera "
        f"paper no se ha modificado. {PAPER_LABEL}"
    )
