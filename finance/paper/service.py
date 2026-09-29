"""Fachada explicita de Atlas Finance V2.1 (motor paper) + V2.7 (politica).

Servicio unico, persistente y auditable para: consultar cartera paper,
registrar un MarketEvent declarado y ejecutar una orden paper confirmada.
Desde V2.7 mantiene dos modos paper separados (CORE y TACTICAL), cada uno
con su propia cartera aislada y su InvestmentPolicy de riesgo versionada:
toda orden confirmada se valida contra la politica activa antes de
ejecutarse. El FinanceAgent sigue siendo solo-lectura: solo puede sugerir
ordenes; la ejecucion exige confirmacion explicita a traves de este
servicio. Sin red, sin broker y sin scheduler.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Mapping

from finance.paper.engine import Decision, PaperEngine
from finance.paper.models import (
    MarketEvent,
    PaperOrder,
    PaperPortfolio,
    Side,
    _decimal,
    utc_now,
)
from finance.paper.policy import InvestmentPolicy, PaperMode, PolicyViolation, validate_order
from finance.paper.store import PaperStore

DEFAULT_STARTING_CASH = Decimal("10000")

ENVELOPE_VERSION = 2


@dataclass
class _ModeState:
    """Cartera aislada + politica de riesgo de un modo paper."""

    engine: PaperEngine
    policy: InvestmentPolicy
    risk: "_RiskState"


@dataclass
class _RiskState:
    """Base y maximo NAV observados por una cartera paper aislada."""

    initial_nav: Decimal | None
    peak_nav: Decimal | None
    realized_pnl: Decimal = Decimal("0")

    def to_dict(self) -> dict[str, str | None]:
        return {
            "initial_nav": None if self.initial_nav is None else str(self.initial_nav),
            "peak_nav": None if self.peak_nav is None else str(self.peak_nav),
            "realized_pnl": str(self.realized_pnl),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "_RiskState":
        def decimal_or_none(value: object) -> Decimal | None:
            return None if value is None else Decimal(str(value))

        return cls(
            initial_nav=decimal_or_none(data.get("initial_nav")),
            peak_nav=decimal_or_none(data.get("peak_nav")),
            realized_pnl=Decimal(str(data.get("realized_pnl", "0"))),
        )


class PaperFinanceService:
    def __init__(
        self,
        store: PaperStore | None = None,
        *,
        starting_cash: Decimal = DEFAULT_STARTING_CASH,
        default_commission: Decimal = Decimal("0"),
        default_slippage_bps: Decimal = Decimal("0"),
        default_spread_bps: Decimal = Decimal("0"),
        entry_maker_fee_bps: Decimal = Decimal("0"),
        entry_taker_fee_bps: Decimal = Decimal("0"),
        exit_maker_fee_bps: Decimal = Decimal("0"),
        exit_taker_fee_bps: Decimal = Decimal("0"),
        default_is_maker: bool = False,
        mode: PaperMode = PaperMode.CORE,
    ) -> None:
        if not isinstance(mode, PaperMode):
            raise ValueError("modo paper invalido")
        self._store = store or PaperStore()
        self._starting_cash = starting_cash
        self._default_commission = default_commission
        self._default_slippage_bps = default_slippage_bps
        self._execution_costs = {
            "default_spread_bps": default_spread_bps,
            "entry_maker_fee_bps": entry_maker_fee_bps,
            "entry_taker_fee_bps": entry_taker_fee_bps,
            "exit_maker_fee_bps": exit_maker_fee_bps,
            "exit_taker_fee_bps": exit_taker_fee_bps,
            "default_is_maker": default_is_maker,
        }
        self._states: dict[PaperMode, _ModeState] = {}
        self._active_mode = mode
        if self._store.exists():
            data = self._store.load()
            if data.get("schema_version") == ENVELOPE_VERSION:
                self._load_envelope(data)
            else:
                # Estado V2.1-V2.6 (schema 1): cartera unica -> modo CORE.
                # El capital y las posiciones previas se conservan en CORE;
                # TACTICAL no recibe capital hasta una asignacion explicita
                # (nunca se duplica el capital paper inicial).
                legacy_engine = PaperEngine.from_snapshot(data)
                self._states[PaperMode.CORE] = _ModeState(
                    legacy_engine,
                    InvestmentPolicy(mode=PaperMode.CORE),
                    _RiskState(None, None),
                )
                self._bind_fill_validator(PaperMode.CORE, legacy_engine)
                self._active_mode = PaperMode.CORE
            return
        # Instalacion nueva: el capital paper inicial pertenece a CORE.
        # Cualquier otro modo empieza con efectivo 0 y sin posiciones.
        core_engine = PaperEngine(
            portfolio=PaperPortfolio(cash=self._starting_cash),
            default_commission=self._default_commission,
            default_slippage_bps=self._default_slippage_bps,
            **self._execution_costs,
        )
        self._states[PaperMode.CORE] = _ModeState(
            core_engine,
            InvestmentPolicy(mode=PaperMode.CORE),
            _RiskState(self._starting_cash, self._starting_cash),
        )
        self._bind_fill_validator(PaperMode.CORE, core_engine)
        if self._active_mode is not PaperMode.CORE:
            self._states[self._active_mode] = self._new_state(self._active_mode)
        self._persist()

    @property
    def engine(self) -> PaperEngine:
        """Motor paper del modo activo (compatibilidad V2.1-V2.6)."""
        return self._state_for(self._active_mode).engine

    def mode(self) -> PaperMode:
        return self._active_mode

    def policy(self) -> InvestmentPolicy:
        return self._state_for(self._active_mode).policy

    def set_mode(self, mode: PaperMode) -> PaperMode:
        """Cambia solo el contexto paper explicito; carteras aisladas."""
        if not isinstance(mode, PaperMode):
            raise ValueError("modo paper invalido")
        self._active_mode = mode
        self._state_for(mode)
        self._persist()
        return self._active_mode

    def update_policy(self, **changes: object) -> InvestmentPolicy:
        """Configura limites de la politica activa (usuario, no sistema)."""
        state = self._state_for(self._active_mode)
        new_policy = InvestmentPolicy(
            mode=state.policy.mode,
            max_open_positions=changes.get(
                "max_open_positions", state.policy.max_open_positions
            ),
            max_exposure_per_asset=changes.get(
                "max_exposure_per_asset", state.policy.max_exposure_per_asset
            ),
            max_total_exposure=changes.get(
                "max_total_exposure", state.policy.max_total_exposure
            ),
            max_loss_pct=changes.get("max_loss_pct", state.policy.max_loss_pct),
            max_drawdown_pct=changes.get(
                "max_drawdown_pct", state.policy.max_drawdown_pct
            ),
        )
        self._states[self._active_mode] = _ModeState(
            engine=state.engine, policy=new_policy, risk=state.risk
        )
        self._persist()
        return new_policy

    def allocate_capital(self, target: PaperMode, amount: Decimal) -> dict[str, object]:
        """Asigna capital paper a un modo moviendo efectivo entre modos.

        Determinista y exacto con Decimal: la cantidad debe ser positiva y
        la reasignacion nunca puede exceder el capital paper total
        disponible en el modo origen (no se inventa ni se duplica capital).
        Las posiciones no se tocan; la operacion queda en el ledger.
        """
        if not isinstance(target, PaperMode):
            raise ValueError("modo paper invalido")
        amount = _decimal(amount, "capital paper")
        if amount <= 0:
            raise ValueError(
                "la asignacion de capital paper debe ser un Decimal positivo"
            )
        source = (
            PaperMode.TACTICAL if target is PaperMode.CORE else PaperMode.CORE
        )
        source_state = self._state_for(source)
        target_state = self._state_for(target)
        available = source_state.engine.portfolio.cash
        if amount > available:
            raise ValueError(
                f"la reasignacion excede el capital paper total disponible: "
                f"{amount} > {available} en el modo {source.value}"
            )
        source_state.engine.transfer_cash(
            -amount, f"asignacion de capital paper al modo {target.value}"
        )
        target_state.engine.transfer_cash(
            amount, f"capital paper recibido del modo {source.value}"
        )
        for risk, delta in (
            (source_state.risk, -amount),
            (target_state.risk, amount),
        ):
            if risk.initial_nav is not None:
                risk.initial_nav += delta
            if risk.peak_nav is not None:
                risk.peak_nav += delta
        self._persist()
        return self.capital_summary()

    def capital_summary(self) -> dict[str, object]:
        """Capital paper por modo y total combinado (solo lectura)."""
        modes: dict[str, dict[str, object]] = {}
        total = Decimal("0")
        for mode in PaperMode:
            state = self._states.get(mode)
            if state is None:
                modes[mode.value] = {
                    "cash": "0",
                    "nav": "0",
                    "positions": {},
                    "created": False,
                }
                continue
            portfolio = state.engine.portfolio
            nav = state.engine.nav()
            total += nav
            modes[mode.value] = {
                "cash": str(portfolio.cash),
                "nav": str(nav),
                "positions": dict(portfolio.positions),
                "created": True,
            }
        return {
            "active_mode": self._active_mode.value,
            "modes": modes,
            "total": str(total.quantize(Decimal("0.01"))),
        }

    def check_policy(self, order: PaperOrder) -> None:
        """Valida la orden contra la politica activa (exacto, sin redondeos)."""
        self._validate(self._state_for(self._active_mode), order)

    def portfolio_summary(self) -> dict[str, object]:
        state = self._state_for(self._active_mode)
        portfolio = state.engine.portfolio
        prices = state.engine.last_prices()
        positions = {
            symbol: {
                "qty": str(position.qty),
                "avg_price": str(position.avg_price),
                "last_price": str(prices.get(symbol, position.avg_price)),
            }
            for symbol, position in portfolio.positions.items()
        }
        ledger = state.engine.ledger
        updated_at = (
            str(ledger[-1]["timestamp"]) if ledger else utc_now().isoformat()
        )
        return {
            "mode": self._active_mode.value,
            "cash": str(portfolio.cash.quantize(Decimal("0.01"))),
            "positions": positions,
            "nav": str(state.engine.nav()),
            "updated_at": updated_at,
        }

    def record_market_event(self, event: MarketEvent) -> bool:
        state = self._state_for(self._active_mode)
        ingested = state.engine.process_event(event)
        if ingested:
            self._refresh_risk(state)
            self._persist()
        return ingested

    def execute_confirmed_order(
        self, order: PaperOrder, event: MarketEvent | None = None
    ) -> Decision:
        state = self._state_for(self._active_mode)
        self._validate(state, order)
        decision = state.engine.execute_order(order, event)
        self._refresh_risk(state)
        self._persist()
        return decision

    def pending_orders(self) -> tuple[PaperOrder, ...]:
        return self._state_for(self._active_mode).engine.pending_orders()

    def execution_defaults(self) -> dict[str, Decimal | bool]:
        """Read-only synthetic costs for paper order estimates."""
        return self._state_for(self._active_mode).engine.defaults()

    def ledger(self) -> tuple[dict[str, object], ...]:
        return self._state_for(self._active_mode).engine.ledger

    def fills(self) -> tuple:
        return self._state_for(self._active_mode).engine.fills

    def execution_risk_state(self):
        """Build the common execution risk snapshot from paper state only."""
        from finance.execution.risk_gate import PortfolioRiskState

        state = self._state_for(self._active_mode)
        engine = state.engine
        today = utc_now().date()
        positions = {
            symbol: position.qty
            for symbol, position in engine.portfolio.positions.items()
        }
        fills = engine.fills
        daily_fills = [fill for fill in fills if fill.timestamp.date() == today]
        realized_loss = Decimal("0")
        average_costs: dict[str, Decimal] = {}
        for fill in fills:
            if fill.side.value == "BUY":
                previous_qty = sum(
                    item.qty
                    for item in fills
                    if item.symbol == fill.symbol
                    and item.timestamp <= fill.timestamp
                    and item is not fill
                    and item.side.value == "BUY"
                )
                prior_sells = sum(
                    item.qty
                    for item in fills
                    if item.symbol == fill.symbol
                    and item.timestamp <= fill.timestamp
                    and item is not fill
                    and item.side.value == "SELL"
                )
                held_before = previous_qty - prior_sells
                prior_cost = average_costs.get(fill.symbol, Decimal("0"))
                total_before = prior_cost * held_before
                average_costs[fill.symbol] = (
                    (total_before + fill.price * fill.qty + fill.commission)
                    / (held_before + fill.qty)
                )
            elif fill.timestamp.date() == today:
                cost = average_costs.get(fill.symbol, Decimal("0"))
                pnl = (fill.price - cost) * fill.qty - fill.commission
                if pnl < 0:
                    realized_loss -= pnl
            if fill.side.value == "SELL":
                average_costs[fill.symbol] = average_costs.get(
                    fill.symbol, Decimal("0")
                )
        return PortfolioRiskState(
            nav=engine.nav(),
            cash=engine.portfolio.cash,
            positions=positions,
            prices=engine.last_prices(),
            daily_trade_count=len(daily_fills),
            daily_realized_loss=realized_loss,
        )

    def risk_summary(self) -> dict[str, object]:
        """Metricas paper usadas por max_loss_pct y max_drawdown_pct."""
        state = self._state_for(self._active_mode)
        self._refresh_risk(state)
        metrics = self._risk_metrics(state)
        return {
            "initial_nav": metrics["initial_nav"],
            "peak_nav": metrics["peak_nav"],
            "nav": metrics["nav"],
            "realized_pnl": metrics["realized_pnl"],
            "loss_pct": metrics["loss_pct"],
            "drawdown_pct": metrics["drawdown_pct"],
            "data_complete": metrics["data_complete"],
            "missing": metrics["missing"],
        }

    def _validate(self, state: _ModeState, order: PaperOrder) -> None:
        defaults = state.engine.defaults()
        validate_order(
            state.policy,
            order,
            portfolio=state.engine.portfolio,
            last_prices=state.engine.last_prices(),
            commission=defaults["commission"],
            slippage_bps=defaults["slippage_bps"],
            spread_bps=defaults["spread_bps"],
            fee_bps=state.engine.fee_bps(order.side),
        )
        position = state.engine.portfolio.positions.get(order.symbol)
        reducing_sell = (
            order.side is Side.SELL
            and position is not None
            and order.qty <= position.qty
        )
        if not reducing_sell:
            self._validate_loss_and_drawdown(state)

    def _validate_loss_and_drawdown(self, state: _ModeState) -> None:
        policy = state.policy
        if policy.max_loss_pct is None and policy.max_drawdown_pct is None:
            return
        metrics = self._risk_metrics(state)
        if not metrics["data_complete"]:
            missing = ", ".join(metrics["missing"])
            raise PolicyViolation(
                "RISK_DATA_UNAVAILABLE",
                f"no se puede evaluar perdida/drawdown: falta {missing}; "
                f"bloqueo seguro",
            )
        if (
            policy.max_loss_pct is not None
            and metrics["loss_pct"] >= policy.max_loss_pct
        ):
            raise PolicyViolation(
                "MAX_LOSS_PCT",
                f"perdida realizada {metrics['loss_pct']} >= limite "
                f"{policy.max_loss_pct} (base NAV {metrics['initial_nav']})",
            )
        if (
            policy.max_drawdown_pct is not None
            and metrics["drawdown_pct"] >= policy.max_drawdown_pct
        ):
            raise PolicyViolation(
                "MAX_DRAWDOWN_PCT",
                f"drawdown {metrics['drawdown_pct']} >= limite "
                f"{policy.max_drawdown_pct} (maximo NAV {metrics['peak_nav']})",
            )

    def _risk_metrics(self, state: _ModeState) -> dict[str, object]:
        engine = state.engine
        missing: list[str] = []
        prices = engine.last_prices()
        for symbol in engine.portfolio.positions:
            price = prices.get(symbol)
            if (
                price is None
                or not isinstance(price, Decimal)
                or not price.is_finite()
                or price <= 0
            ):
                missing.append(f"precio paper valido de {symbol}")
        if state.risk.initial_nav is None or state.risk.initial_nav <= 0:
            missing.append("base NAV inicial persistida")
        if state.risk.peak_nav is None or state.risk.peak_nav <= 0:
            missing.append("maximo NAV persistido")
        nav = None if missing else engine.portfolio.nav(prices)
        peak = state.risk.peak_nav
        if peak is not None and nav is not None and nav > peak:
            peak = nav
        initial = state.risk.initial_nav
        loss = max(Decimal("0"), -state.risk.realized_pnl)
        return {
            "initial_nav": initial,
            "peak_nav": peak,
            "nav": nav,
            "realized_pnl": state.risk.realized_pnl,
            "loss_pct": None if initial is None or initial <= 0 else loss / initial,
            "drawdown_pct": (
                None
                if peak is None or peak <= 0 or nav is None
                else max(Decimal("0"), (peak - nav) / peak)
            ),
            "data_complete": not missing,
            "missing": missing,
        }

    def _refresh_risk(self, state: _ModeState) -> None:
        if state.risk.initial_nav is None:
            return
        metrics = self._risk_metrics(state)
        if not metrics["data_complete"]:
            return
        state.risk.realized_pnl = self._realized_pnl(state.engine)
        nav = state.engine.portfolio.nav(state.engine.last_prices())
        if state.risk.peak_nav is None or nav > state.risk.peak_nav:
            state.risk.peak_nav = nav

    @staticmethod
    def _realized_pnl(engine: PaperEngine) -> Decimal:
        quantities: dict[str, Decimal] = {}
        costs: dict[str, Decimal] = {}
        realized = Decimal("0")
        for fill in engine.fills:
            if fill.side.value == "BUY":
                quantities[fill.symbol] = quantities.get(fill.symbol, Decimal("0")) + fill.qty
                costs[fill.symbol] = costs.get(fill.symbol, Decimal("0")) + fill.price * fill.qty + fill.commission
                continue
            held = quantities.get(fill.symbol, Decimal("0"))
            if held < fill.qty or held <= 0:
                continue
            average_cost = costs[fill.symbol] / held
            realized += fill.price * fill.qty - fill.commission - average_cost * fill.qty
            quantities[fill.symbol] = held - fill.qty
            costs[fill.symbol] = costs[fill.symbol] - average_cost * fill.qty
        return realized

    def _state_for(self, mode: PaperMode) -> _ModeState:
        state = self._states.get(mode)
        if state is None:
            state = self._new_state(mode)
            self._states[mode] = state
            self._persist()
        return state

    def _new_state(self, mode: PaperMode) -> _ModeState:
        """Estado nuevo de un modo: sin capital ni posiciones.

        El capital paper solo existe en CORE al crear la instalacion o por
        asignacion explicita del usuario (allocate_capital): nunca se crea
        una segunda cartera con el capital inicial por defecto.
        """
        engine = PaperEngine(
            portfolio=PaperPortfolio(cash=Decimal("0")),
            default_commission=self._default_commission,
            default_slippage_bps=self._default_slippage_bps,
            **self._execution_costs,
        )
        self._bind_fill_validator(mode, engine)
        return _ModeState(
            engine,
            InvestmentPolicy(mode=mode),
            _RiskState(Decimal("0"), Decimal("0")),
        )

    def _bind_fill_validator(self, mode: PaperMode, engine: PaperEngine) -> None:
        engine.set_fill_validator(lambda order: self._validate(self._states[mode], order))

    def _load_envelope(self, data: Mapping[str, object]) -> None:
        modes_data = data.get("modes")
        assert isinstance(modes_data, Mapping)
        for mode_name, mode_data in modes_data.items():
            assert isinstance(mode_data, Mapping)
            mode = PaperMode(str(mode_name))
            policy_data = mode_data.get("policy")
            assert isinstance(policy_data, Mapping)
            engine_data = mode_data.get("engine")
            assert isinstance(engine_data, Mapping)
            engine = PaperEngine.from_snapshot(engine_data)
            self._states[mode] = _ModeState(
                engine=engine,
                policy=InvestmentPolicy.from_dict(policy_data),
                risk=(
                    _RiskState.from_dict(risk_data)
                    if isinstance(risk_data := mode_data.get("risk"), Mapping)
                    else _RiskState(None, None)
                ),
            )
            self._bind_fill_validator(mode, engine)
        stored_active = data.get("active_mode")
        if stored_active is not None:
            self._active_mode = PaperMode(str(stored_active))
        if self._active_mode not in self._states:
            self._states[self._active_mode] = self._new_state(self._active_mode)
            self._persist()

    def _persist(self) -> None:
        self._store.save(
            {
                "active_mode": self._active_mode.value,
                "modes": {
                    mode.value: {
                        "policy": state.policy.to_dict(),
                        "risk": state.risk.to_dict(),
                        "engine": state.engine.snapshot(),
                    }
                    for mode, state in self._states.items()
                },
            }
        )
