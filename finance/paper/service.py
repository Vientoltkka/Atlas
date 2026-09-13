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
    _decimal,
    utc_now,
)
from finance.paper.policy import InvestmentPolicy, PaperMode, validate_order
from finance.paper.store import PaperStore

DEFAULT_STARTING_CASH = Decimal("10000")

ENVELOPE_VERSION = 2


@dataclass
class _ModeState:
    """Cartera aislada + politica de riesgo de un modo paper."""

    engine: PaperEngine
    policy: InvestmentPolicy


class PaperFinanceService:
    def __init__(
        self,
        store: PaperStore | None = None,
        *,
        starting_cash: Decimal = DEFAULT_STARTING_CASH,
        default_commission: Decimal = Decimal("0"),
        default_slippage_bps: Decimal = Decimal("0"),
        mode: PaperMode = PaperMode.CORE,
    ) -> None:
        if not isinstance(mode, PaperMode):
            raise ValueError("modo paper invalido")
        self._store = store or PaperStore()
        self._starting_cash = starting_cash
        self._default_commission = default_commission
        self._default_slippage_bps = default_slippage_bps
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
                self._states[PaperMode.CORE] = _ModeState(
                    PaperEngine.from_snapshot(data),
                    InvestmentPolicy(mode=PaperMode.CORE),
                )
                self._active_mode = PaperMode.CORE
            return
        # Instalacion nueva: el capital paper inicial pertenece a CORE.
        # Cualquier otro modo empieza con efectivo 0 y sin posiciones.
        self._states[PaperMode.CORE] = _ModeState(
            PaperEngine(
                portfolio=PaperPortfolio(cash=self._starting_cash),
                default_commission=self._default_commission,
                default_slippage_bps=self._default_slippage_bps,
            ),
            InvestmentPolicy(mode=PaperMode.CORE),
        )
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
            engine=state.engine, policy=new_policy
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
            self._persist()
        return ingested

    def execute_confirmed_order(
        self, order: PaperOrder, event: MarketEvent | None = None
    ) -> Decision:
        state = self._state_for(self._active_mode)
        self._validate(state, order)
        decision = state.engine.execute_order(order, event)
        self._persist()
        return decision

    def pending_orders(self) -> tuple[PaperOrder, ...]:
        return self._state_for(self._active_mode).engine.pending_orders()

    def execution_defaults(self) -> dict[str, Decimal]:
        """Read-only commission and slippage defaults for order estimates."""
        return self._state_for(self._active_mode).engine.defaults()

    def ledger(self) -> tuple[dict[str, object], ...]:
        return self._state_for(self._active_mode).engine.ledger

    def fills(self) -> tuple:
        return self._state_for(self._active_mode).engine.fills

    def _validate(self, state: _ModeState, order: PaperOrder) -> None:
        defaults = state.engine.defaults()
        validate_order(
            state.policy,
            order,
            portfolio=state.engine.portfolio,
            last_prices=state.engine.last_prices(),
            commission=defaults["commission"],
            slippage_bps=defaults["slippage_bps"],
        )

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
        )
        return _ModeState(engine, InvestmentPolicy(mode=mode))

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
            self._states[mode] = _ModeState(
                engine=PaperEngine.from_snapshot(engine_data),
                policy=InvestmentPolicy.from_dict(policy_data),
            )
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
                        "engine": state.engine.snapshot(),
                    }
                    for mode, state in self._states.items()
                },
            }
        )
