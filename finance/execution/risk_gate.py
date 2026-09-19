"""Fail-closed, LLM-free risk evaluation."""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from finance.execution.models import ExecutionIntent, ExecutionMode, RiskDecision
from finance.execution.policy import RiskPolicy
from finance.paper.models import Side


@dataclass(frozen=True)
class PortfolioRiskState:
    nav: Decimal | None
    cash: Decimal | None
    positions: dict[str, Decimal] | None
    prices: dict[str, Decimal] | None
    daily_trade_count: int | None
    daily_realized_loss: Decimal | None


class RiskGate:
    def evaluate(self, intent: ExecutionIntent, state: PortfolioRiskState | None, policy: RiskPolicy) -> RiskDecision:
        reasons: list[str] = []
        evidence: dict[str, str] = {}
        if intent.mode is ExecutionMode.REAL:
            reasons.append("REAL_EXECUTION_DISABLED")
        if policy.kill_switch:
            reasons.append("KILL_SWITCH_ACTIVE")
        if state is None or any(value is None for value in (state.nav, state.cash, state.positions, state.prices, state.daily_trade_count, state.daily_realized_loss)):
            reasons.append("MISSING_CRITICAL_RISK_DATA")
            return RiskDecision(intent.intent_id, False, tuple(reasons), evidence=evidence)
        assert state.nav is not None and state.cash is not None and state.positions is not None and state.prices is not None
        assert state.daily_trade_count is not None and state.daily_realized_loss is not None
        reference = intent.reference_price or state.prices.get(intent.symbol)
        if reference is None or reference <= 0:
            reasons.append("MISSING_REFERENCE_PRICE")
        elif state.nav <= 0:
            reasons.append("INVALID_NAV")
        else:
            value = intent.quantity * reference
            evidence.update(order_value=str(value), nav=str(state.nav))
            if value > policy.max_order_value: reasons.append("MAX_ORDER_VALUE")
            if state.daily_trade_count >= policy.max_daily_trades: reasons.append("MAX_DAILY_TRADES")
            if state.daily_realized_loss >= policy.max_daily_loss: reasons.append("MAX_DAILY_LOSS")
            if intent.asset_class not in policy.allowed_asset_classes: reasons.append("ASSET_CLASS_NOT_ALLOWED")
            if policy.prohibit_derivatives and intent.asset_class == "DERIVATIVE": reasons.append("NO_DERIVATIVES")
            if intent.side is Side.SELL and policy.prohibit_shorts and intent.quantity > state.positions.get(intent.symbol, Decimal("0")): reasons.append("NO_SHORTS")
            if intent.side is Side.BUY:
                if value > state.cash and policy.prohibit_leverage: reasons.append("NO_LEVERAGE")
                asset_after = state.positions.get(intent.symbol, Decimal("0")) * reference + value
                total_before = sum(qty * state.prices.get(symbol, Decimal("0")) for symbol, qty in state.positions.items())
                if asset_after > policy.max_exposure_per_asset * state.nav: reasons.append("MAX_EXPOSURE_PER_ASSET")
                if total_before + value > policy.max_total_exposure * state.nav: reasons.append("MAX_TOTAL_EXPOSURE")
                if intent.symbol not in state.positions and len(state.positions) >= policy.max_open_positions: reasons.append("MAX_OPEN_POSITIONS")
        return RiskDecision(intent.intent_id, not reasons, tuple(reasons or ("APPROVED",)), evidence=evidence)
