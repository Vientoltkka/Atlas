"""Politica de riesgo paper de Atlas Finance V2.7 (CORE/TACTICAL).

InvestmentPolicy es la politica de riesgo versionada de cada modo PAPER:
- CORE: largo plazo.
- TACTICAL: corto plazo, con control mas estricto una vez configurado.

Los limites numericos no estan fijados por defecto: los configura el
usuario mas adelante (nunca se redondean ni se ignoran). Las prohibiciones
de derivados, apalancamiento y ventas en corto son explicitas y no se
pueden desactivar. Los limites de perdida y drawdown son configurables e
informativos: no hay ejecucion automatica. Todo el motor sigue siendo
PAPER, sin broker, sin scheduler y sin dinero real.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from typing import Mapping

from finance.paper.models import (
    OrderType,
    PaperOrder,
    PaperPortfolio,
    Side,
    _decimal,
)

RULE_NO_DERIVATIVES = "NO_DERIVATIVES"
RULE_NO_LEVERAGE = "NO_LEVERAGE"
RULE_NO_SHORTS = "NO_SHORTS"
RULE_MAX_OPEN_POSITIONS = "MAX_OPEN_POSITIONS"
RULE_MAX_EXPOSURE_PER_ASSET = "MAX_EXPOSURE_PER_ASSET"
RULE_MAX_TOTAL_EXPOSURE = "MAX_TOTAL_EXPOSURE"


class PaperMode(Enum):
    """Modo paper explicito del contexto de riesgo."""

    CORE = "CORE"
    TACTICAL = "TACTICAL"


MODE_DESCRIPTIONS: dict[PaperMode, str] = {
    PaperMode.CORE: "largo plazo",
    PaperMode.TACTICAL: "corto plazo, control mas estricto",
}


class PolicyViolation(ValueError):
    """Rechazo determinista de una orden paper por incumplir una regla."""

    def __init__(self, rule: str, reason: str) -> None:
        super().__init__(f"regla {rule}: {reason}")
        self.rule = rule
        self.reason = reason


@dataclass(frozen=True)
class InvestmentPolicy:
    """Politica de riesgo paper de un modo. Inmutable y versionada."""

    mode: PaperMode
    max_open_positions: int | None = None
    max_exposure_per_asset: Decimal | None = None
    max_total_exposure: Decimal | None = None
    max_loss_pct: Decimal | None = None
    max_drawdown_pct: Decimal | None = None
    prohibit_derivatives: bool = True
    prohibit_leverage: bool = True
    prohibit_shorts: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.mode, PaperMode):
            raise ValueError("modo paper invalido")
        limit = self.max_open_positions
        if limit is not None:
            if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
                raise ValueError("max_open_positions debe ser un entero >= 1 o None")
        for name in (
            "max_exposure_per_asset",
            "max_total_exposure",
            "max_loss_pct",
            "max_drawdown_pct",
        ):
            value = getattr(self, name)
            if value is None:
                continue
            decimal_value = _decimal(value, name)
            if decimal_value <= 0:
                raise ValueError(f"{name} debe ser positivo o None")
            if decimal_value > 1:
                raise ValueError(
                    f"{name} no puede exceder 1.0: mas del NAV seria apalancamiento"
                )
            object.__setattr__(self, name, decimal_value)
        prohibitions = {
            "prohibit_derivatives": ("derivados", self.prohibit_derivatives),
            "prohibit_leverage": ("apalancamiento", self.prohibit_leverage),
            "prohibit_shorts": ("ventas en corto", self.prohibit_shorts),
        }
        for name, (label, value) in prohibitions.items():
            if value is not True:
                raise ValueError(
                    f"la politica paper prohibe explicitamente {label}; "
                    f"{name} no se puede desactivar"
                )

    def to_dict(self) -> dict[str, object]:
        def raw(value: Decimal | None) -> str | None:
            return None if value is None else str(value)

        return {
            "mode": self.mode.value,
            "max_open_positions": self.max_open_positions,
            "max_exposure_per_asset": raw(self.max_exposure_per_asset),
            "max_total_exposure": raw(self.max_total_exposure),
            "max_loss_pct": raw(self.max_loss_pct),
            "max_drawdown_pct": raw(self.max_drawdown_pct),
            "prohibit_derivatives": True,
            "prohibit_leverage": True,
            "prohibit_shorts": True,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "InvestmentPolicy":
        def optional_decimal(name: str) -> Decimal | None:
            value = data.get(name)
            return None if value is None else Decimal(str(value))

        positions = data.get("max_open_positions")
        return cls(
            mode=PaperMode(str(data["mode"])),
            max_open_positions=None if positions is None else int(positions),
            max_exposure_per_asset=optional_decimal("max_exposure_per_asset"),
            max_total_exposure=optional_decimal("max_total_exposure"),
            max_loss_pct=optional_decimal("max_loss_pct"),
            max_drawdown_pct=optional_decimal("max_drawdown_pct"),
        )

    def describe_rules(self) -> dict[str, str]:
        """Reglas legibles con su estado: activa, no configurada o no
        aplicable todavia (los limites de perdida y drawdown no tienen
        calculos ni bloqueo reales y nunca protegen por si solos)."""

        def pct(value: Decimal) -> str:
            return f"{value * 100}%"

        return {
            "modo": f"{self.mode.value} ({MODE_DESCRIPTIONS[self.mode]})",
            "max_open_positions": (
                f"maximo {self.max_open_positions} posiciones abiertas"
                if self.max_open_positions is not None
                else "sin limite de posiciones abiertas"
            ),
            "max_exposure_per_asset": (
                f"{pct(self.max_exposure_per_asset)} del NAV por activo"
                if self.max_exposure_per_asset is not None
                else "sin limite por activo"
            ),
            "max_total_exposure": (
                f"{pct(self.max_total_exposure)} del NAV en total"
                if self.max_total_exposure is not None
                else "sin limite total"
            ),
            "max_loss_pct": (
                f"{pct(self.max_loss_pct)} del NAV (solo informativo)"
                if self.max_loss_pct is not None
                else "sin configurar (si se configura, solo informativo)"
            ),
            "max_drawdown_pct": (
                f"{pct(self.max_drawdown_pct)} del NAV (solo informativo)"
                if self.max_drawdown_pct is not None
                else "sin configurar (si se configura, solo informativo)"
            ),
            "derivados": "prohibidos",
            "apalancamiento": "prohibido",
            "cortos": "prohibidas",
        }

    def describe_rule_status(self) -> dict[str, str]:
        """Estado de cada regla: 'activa', 'no configurada' o 'no aplicable
        todavia'.

        - activa: la regla bloquea ordenes que la incumplen.
        - no configurada: todavia no limita ninguna orden.
        - no aplicable todavia: perdida y drawdown no tienen calculos ni
          bloqueo reales; aunque se configuren son solo informativos y no
          protegen por si solos.
        """
        return {
            "max_open_positions": (
                "activa" if self.max_open_positions is not None else "no configurada"
            ),
            "max_exposure_per_asset": (
                "activa" if self.max_exposure_per_asset is not None else "no configurada"
            ),
            "max_total_exposure": (
                "activa" if self.max_total_exposure is not None else "no configurada"
            ),
            "max_loss_pct": "no aplicable todavia",
            "max_drawdown_pct": "no aplicable todavia",
            "derivados": "activa",
            "apalancamiento": "activa",
            "cortos": "activa",
        }


def validate_order(
    policy: InvestmentPolicy,
    order: PaperOrder,
    *,
    portfolio: PaperPortfolio,
    last_prices: Mapping[str, Decimal],
    commission: Decimal = Decimal("0"),
    slippage_bps: Decimal = Decimal("0"),
) -> None:
    """Valida una orden paper contra la politica activa.

    Lanza PolicyViolation con la regla concreta si la orden incumple una
    regla. Todas las comparaciones son exactas con Decimal: nunca se
    redondea ni se ignora un limite. Para LIMIT se usa el precio limite
    como referencia conservadora; para MARKET el ultimo tick paper.
    """
    slippage = Decimal(slippage_bps)
    if order.side is Side.SELL:
        position = portfolio.positions.get(order.symbol)
        held = position.qty if position is not None else Decimal("0")
        if held < order.qty:
            raise PolicyViolation(
                RULE_NO_SHORTS,
                f"venta de {order.qty} > posicion en cartera {held}; "
                f"las ventas en corto estan prohibidas",
            )
        return

    if order.order_type is OrderType.LIMIT:
        reference = order.limit_price
    else:
        tick = last_prices.get(order.symbol)
        reference = (
            None
            if tick is None
            else tick * (Decimal("1") + slippage / Decimal("10000"))
        )
    if reference is None:
        # Sin precio de referencia la validacion numerica no aplica; el
        # motor rechazara la orden como "sin precio de referencia".
        return

    cost = reference * order.qty + Decimal(commission)
    if cost > portfolio.cash:
        raise PolicyViolation(
            RULE_NO_LEVERAGE,
            f"coste estimado {cost} > efectivo {portfolio.cash}; comprar sin "
            f"efectivo seria apalancarse",
        )

    position = portfolio.positions.get(order.symbol)
    existing_qty = position.qty if position is not None else Decimal("0")
    if position is None and policy.max_open_positions is not None:
        open_after = len(portfolio.positions) + 1
        if open_after > policy.max_open_positions:
            raise PolicyViolation(
                RULE_MAX_OPEN_POSITIONS,
                f"la orden abriria la posicion {open_after} y el maximo "
                f"configurado es {policy.max_open_positions}",
            )

    if policy.max_exposure_per_asset is None and policy.max_total_exposure is None:
        return

    nav = portfolio.nav(last_prices)
    if policy.max_exposure_per_asset is not None:
        asset_value = (existing_qty + order.qty) * reference
        asset_limit = policy.max_exposure_per_asset * nav
        if asset_value > asset_limit:
            raise PolicyViolation(
                RULE_MAX_EXPOSURE_PER_ASSET,
                f"exposicion de {order.symbol} tras la orden {asset_value} > "
                f"limite {asset_limit} ({policy.max_exposure_per_asset} del "
                f"NAV {nav})",
            )
    if policy.max_total_exposure is not None:
        positions_value = Decimal("0")
        for symbol, held_position in portfolio.positions.items():
            positions_value += held_position.qty * last_prices.get(
                symbol, held_position.avg_price
            )
        total = positions_value + order.qty * reference
        total_limit = policy.max_total_exposure * nav
        if total > total_limit:
            raise PolicyViolation(
                RULE_MAX_TOTAL_EXPOSURE,
                f"exposicion total tras la orden {total} > limite "
                f"{total_limit} ({policy.max_total_exposure} del NAV {nav})",
            )
