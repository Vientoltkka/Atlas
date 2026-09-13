"""Tests de Atlas Finance V2.7: politica de riesgo paper CORE/TACTICAL.

Cubre: cambio de modo explicito, reglas CORE/TACTICAL, ordenes bloqueadas
por regla concreta antes de crear propuesta, ordenes permitidas, aislamiento
de cartera entre modos, persistencia versionada (schema 2) y migracion
desde schema 1. Todo es PAPER: sin dinero real, sin broker, sin scheduler
y sin ejecucion automatica de limites de perdida/drawdown.
"""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

import pytest

from finance.paper.models import (
    EventType,
    MarketEvent,
    OrderType,
    PaperOrder,
    PaperPortfolio,
    Side,
)
from finance.paper.policy import (
    RULE_MAX_EXPOSURE_PER_ASSET,
    RULE_MAX_OPEN_POSITIONS,
    RULE_MAX_TOTAL_EXPOSURE,
    RULE_NO_LEVERAGE,
    RULE_NO_SHORTS,
    InvestmentPolicy,
    PaperMode,
    PolicyViolation,
    validate_order,
)
from finance.paper.service import PaperFinanceService
from finance.paper.store import PaperStore
from use_cases.paper_finance_chat import PaperFinanceChat


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _service(tmp_path: Path, **kwargs) -> PaperFinanceService:
    return PaperFinanceService(PaperStore(tmp_path / "finance_paper"), **kwargs)


def _chat(service: PaperFinanceService) -> PaperFinanceChat:
    return PaperFinanceChat(service)


def _register_tick(service: PaperFinanceService, symbol: str, price: str) -> None:
    service.record_market_event(
        MarketEvent(
            symbol=symbol,
            price=Decimal(price),
            event_type=EventType.DECLARED,
        )
    )


def _market_order(order_id: str, symbol: str, qty: str, side: Side = Side.BUY) -> PaperOrder:
    return PaperOrder(
        symbol=symbol,
        side=side,
        qty=Decimal(qty),
        order_type=OrderType.MARKET,
        order_id=order_id,
    )


# ---------------------------------------------------------------------------
# Politica por defecto: PAPER, sin porcentajes ni capital fijados
# ---------------------------------------------------------------------------


def test_default_policies_are_paper_with_unconfigured_limits() -> None:
    for mode in (PaperMode.CORE, PaperMode.TACTICAL):
        policy = InvestmentPolicy(mode=mode)
        rules = policy.describe_rules()
        status = policy.describe_rule_status()
        assert rules["modo"].startswith(mode.value)
        assert rules["max_open_positions"] == "sin limite de posiciones abiertas"
        assert rules["max_exposure_per_asset"] == "sin limite por activo"
        assert rules["max_total_exposure"] == "sin limite total"
        assert rules["max_loss_pct"] == "sin configurar (si se configura, solo informativo)"
        assert rules["max_drawdown_pct"] == "sin configurar (si se configura, solo informativo)"
        assert status["max_open_positions"] == "no configurada"
        assert status["max_exposure_per_asset"] == "no configurada"
        assert status["max_total_exposure"] == "no configurada"
        assert status["max_loss_pct"] == "no aplicable todavia"
        assert status["max_drawdown_pct"] == "no aplicable todavia"
        assert status["derivados"] == "activa"
        assert rules["derivados"] == "prohibidos"
        assert rules["apalancamiento"] == "prohibido"
        assert rules["cortos"] == "prohibidas"


def test_rule_status_distinguishes_active_unconfigured_and_not_yet_applicable() -> None:
    policy = InvestmentPolicy(
        mode=PaperMode.TACTICAL,
        max_open_positions=3,
        max_exposure_per_asset=Decimal("0.25"),
        max_total_exposure=Decimal("0.80"),
        max_loss_pct=Decimal("0.05"),
        max_drawdown_pct=Decimal("0.10"),
    )
    status = policy.describe_rule_status()
    assert status["max_open_positions"] == "activa"
    assert status["max_exposure_per_asset"] == "activa"
    assert status["max_total_exposure"] == "activa"
    # Aunque esten configurados, perdida y drawdown no protegen todavia:
    # no hay calculos ni bloqueo reales.
    assert status["max_loss_pct"] == "no aplicable todavia"
    assert status["max_drawdown_pct"] == "no aplicable todavia"
    rules = policy.describe_rules()
    assert "solo informativo" in rules["max_loss_pct"]
    assert "solo informativo" in rules["max_drawdown_pct"]


def test_policy_rejects_disabling_prohibitions_and_invalid_limits() -> None:
    with pytest.raises(ValueError):
        InvestmentPolicy(mode=PaperMode.CORE, prohibit_shorts=False)
    with pytest.raises(ValueError):
        InvestmentPolicy(mode=PaperMode.CORE, prohibit_leverage=False)
    with pytest.raises(ValueError):
        InvestmentPolicy(mode=PaperMode.CORE, prohibit_derivatives=False)
    with pytest.raises(ValueError):
        InvestmentPolicy(mode=PaperMode.CORE, max_total_exposure=Decimal("1.5"))
    with pytest.raises(ValueError):
        InvestmentPolicy(mode=PaperMode.CORE, max_exposure_per_asset=Decimal("0"))
    with pytest.raises(ValueError):
        InvestmentPolicy(mode=PaperMode.CORE, max_open_positions=0)
    with pytest.raises(ValueError):
        InvestmentPolicy(mode=PaperMode.CORE, max_open_positions=True)


def test_policy_dict_roundtrip_preserves_configured_limits() -> None:
    policy = InvestmentPolicy(
        mode=PaperMode.TACTICAL,
        max_open_positions=3,
        max_exposure_per_asset=Decimal("0.25"),
        max_total_exposure=Decimal("0.80"),
        max_loss_pct=Decimal("0.05"),
        max_drawdown_pct=Decimal("0.10"),
    )
    assert InvestmentPolicy.from_dict(policy.to_dict()) == policy


# ---------------------------------------------------------------------------
# Cambio de modo explicito
# ---------------------------------------------------------------------------


def test_mode_switch_changes_only_the_paper_context(tmp_path: Path) -> None:
    service = _service(tmp_path)
    assert service.mode() is PaperMode.CORE

    assert service.set_mode(PaperMode.TACTICAL) is PaperMode.TACTICAL
    assert service.mode() is PaperMode.TACTICAL
    assert service.policy().mode is PaperMode.TACTICAL

    assert service.set_mode(PaperMode.CORE) is PaperMode.CORE
    assert service.policy().mode is PaperMode.CORE


def test_mode_switch_is_persisted_across_service_instances(tmp_path: Path) -> None:
    store = PaperStore(tmp_path / "finance_paper")
    first = PaperFinanceService(store)
    first.set_mode(PaperMode.TACTICAL)

    second = PaperFinanceService(store)
    assert second.mode() is PaperMode.TACTICAL


def test_chat_mode_switch_command_and_guidance(tmp_path: Path) -> None:
    service = _service(tmp_path)
    chat = _chat(service)

    switched = chat.handle("modo paper tactical")
    assert service.mode() is PaperMode.TACTICAL
    assert "modo TACTICAL" in switched
    assert "politica paper" in switched.casefold()

    guidance = chat.handle("modo paper")
    assert "modo paper core" in guidance.casefold()
    assert "modo paper tactical" in guidance.casefold()
    assert "no se ha modificado" in guidance.casefold()


def test_chat_mode_switch_blocked_while_order_proposal_pending(tmp_path: Path) -> None:
    service = _service(tmp_path)
    chat = _chat(service)
    chat.handle("registra precio paper AAPL 100")
    chat.handle("compra paper 1 de AAPL a mercado")
    assert chat.pending_proposal is not None

    blocked = chat.handle("modo paper tactical")

    assert "no cambio el modo paper" in blocked.casefold()
    assert service.mode() is PaperMode.CORE
    assert chat.pending_proposal is not None


# ---------------------------------------------------------------------------
# Aislamiento de cartera entre modos
# ---------------------------------------------------------------------------


def test_portfolios_are_isolated_between_modes(tmp_path: Path) -> None:
    service = _service(tmp_path)
    _register_tick(service, "AAPL", "100")
    decision = service.execute_confirmed_order(_market_order("o1", "AAPL", "10"))
    assert decision.filled

    service.set_mode(PaperMode.TACTICAL)
    tactical_summary = service.portfolio_summary()
    assert tactical_summary["cash"] == "0.00"
    assert tactical_summary["positions"] == {}

    service.allocate_capital(PaperMode.TACTICAL, Decimal("4000"))
    tactical_summary = service.portfolio_summary()
    assert tactical_summary["cash"] == "4000.00"
    assert tactical_summary["positions"] == {}

    _register_tick(service, "TEST", "50")
    tactical_fill = service.execute_confirmed_order(_market_order("o2", "TEST", "2"))
    assert tactical_fill.filled

    service.set_mode(PaperMode.CORE)
    core_summary = service.portfolio_summary()
    assert core_summary["cash"] == "5000.00"
    assert set(core_summary["positions"]) == {"AAPL"}
    assert core_summary["positions"]["AAPL"]["qty"] == "10"


def test_isolated_portfolios_survive_reload(tmp_path: Path) -> None:
    store = PaperStore(tmp_path / "finance_paper")
    first = PaperFinanceService(store)
    _register_tick(first, "AAPL", "100")
    first.execute_confirmed_order(_market_order("o1", "AAPL", "10"))
    first.set_mode(PaperMode.TACTICAL)
    first.allocate_capital(PaperMode.TACTICAL, Decimal("5000"))
    _register_tick(first, "MSFT", "200")
    first.execute_confirmed_order(_market_order("o2", "MSFT", "1"))

    second = PaperFinanceService(store)
    second.set_mode(PaperMode.CORE)
    core_summary = second.portfolio_summary()
    assert set(core_summary["positions"]) == {"AAPL"}
    assert core_summary["cash"] == "4000.00"
    second.set_mode(PaperMode.TACTICAL)
    tactical_summary = second.portfolio_summary()
    assert set(tactical_summary["positions"]) == {"MSFT"}
    assert tactical_summary["cash"] == "4800.00"


# ---------------------------------------------------------------------------
# Ordenes bloqueadas por la politica (antes de crear propuesta)
# ---------------------------------------------------------------------------


def test_order_blocked_by_max_open_positions_rule(tmp_path: Path) -> None:
    service = _service(tmp_path)
    service.update_policy(max_open_positions=1)
    chat = _chat(service)
    chat.handle("registra precio paper AAPL 100")
    proposal = chat.handle("compra paper 1 de AAPL a mercado")
    assert "MAX_OPEN_POSITIONS" not in proposal

    confirmation = chat.execute_pending()
    assert "FILLED" in confirmation

    chat.handle("registra precio paper MSFT 200")
    blocked = chat.handle("compra paper 1 de MSFT a mercado")

    casefolded = blocked.casefold()
    assert f"regla {RULE_MAX_OPEN_POSITIONS}" in blocked
    assert "no se ha creado ninguna propuesta" in casefolded
    assert chat.pending_proposal is None
    summary = service.portfolio_summary()
    assert summary["cash"] == "9900.00"
    assert set(summary["positions"]) == {"AAPL"}


def test_order_blocked_by_max_exposure_per_asset_rule_without_rounding(tmp_path: Path) -> None:
    service = _service(tmp_path)
    service.update_policy(max_exposure_per_asset=Decimal("0.05"))  # 5% de NAV 10000 = 500
    chat = _chat(service)
    chat.handle("registra precio paper AAPL 100")

    blocked = chat.handle("compra paper 10 de AAPL a mercado")  # 1000 > 500
    assert f"regla {RULE_MAX_EXPOSURE_PER_ASSET}" in blocked
    assert chat.pending_proposal is None

    over_by_fraction = chat.handle("compra paper 5.000001 de AAPL a mercado")
    assert f"regla {RULE_MAX_EXPOSURE_PER_ASSET}" in over_by_fraction
    assert chat.pending_proposal is None

    exact = chat.handle("compra paper 5 de AAPL a mercado")  # 500 == limite
    assert "Propuesta de orden paper" in exact
    assert chat.pending_proposal is not None

    confirmation = chat.execute_pending()
    assert "FILLED" in confirmation
    assert service.portfolio_summary()["cash"] == "9500.00"


def test_order_blocked_by_max_total_exposure_rule(tmp_path: Path) -> None:
    service = _service(tmp_path)
    service.update_policy(max_total_exposure=Decimal("0.10"))  # 1000
    chat = _chat(service)
    chat.handle("registra precio paper AAPL 100")
    chat.handle("registra precio paper MSFT 50")

    within = chat.handle("compra paper 8 de AAPL a mercado")  # 800 <= 1000
    assert "Propuesta de orden paper" in within
    assert "FILLED" in chat.execute_pending()

    blocked = chat.handle("compra paper 5 de MSFT a mercado")  # 800 + 250 > 1000

    assert f"regla {RULE_MAX_TOTAL_EXPOSURE}" in blocked
    assert chat.pending_proposal is None
    assert service.portfolio_summary()["cash"] == "9200.00"


def test_sell_without_position_is_blocked_by_no_shorts_rule(tmp_path: Path) -> None:
    service = _service(tmp_path)
    chat = _chat(service)
    chat.handle("registra precio paper AAPL 100")

    blocked = chat.handle("vende paper 5 de AAPL a mercado")

    assert f"regla {RULE_NO_SHORTS}" in blocked
    assert "ventas en corto estan prohibidas" in blocked.casefold()
    assert chat.pending_proposal is None
    assert service.portfolio_summary()["cash"] == "10000.00"


def test_buy_without_cash_is_blocked_by_no_leverage_rule(tmp_path: Path) -> None:
    service = _service(tmp_path)
    chat = _chat(service)
    chat.handle("registra precio paper AAPL 100")

    blocked = chat.handle("compra paper 100.5 de AAPL a mercado")  # 10050 > 10000

    assert f"regla {RULE_NO_LEVERAGE}" in blocked
    assert "seria apalancarse" in blocked.casefold()
    assert chat.pending_proposal is None


# ---------------------------------------------------------------------------
# Orden permitida: propuesta con modo visible y confirmacion independiente
# ---------------------------------------------------------------------------


def test_allowed_order_proposal_shows_mode_and_fills_after_confirmation(tmp_path: Path) -> None:
    service = _service(tmp_path)
    chat = _chat(service)
    chat.handle("registra precio paper AAPL 100")

    proposal = chat.handle("compra paper 10 de AAPL a mercado")

    assert "Propuesta de orden paper" in proposal
    assert "- Modo: CORE (politica paper activa)" in proposal
    assert "prohibidos derivados, apalancamiento y ventas en corto" in proposal.casefold()
    assert chat.pending_proposal is not None
    assert service.portfolio_summary()["cash"] == "10000.00"

    confirmation = chat.execute_pending()

    assert "modo core" in confirmation.casefold()
    assert "FILLED" in confirmation
    summary = service.portfolio_summary()
    assert summary["cash"] == "9000.00"
    assert summary["positions"]["AAPL"]["qty"] == "10"


def test_politica_paper_command_shows_rule_states(tmp_path: Path) -> None:
    service = _service(tmp_path)
    chat = _chat(service)

    text = chat.handle("politica paper")

    casefolded = text.casefold()
    assert "politica de riesgo paper del modo core" in casefolded
    assert "no configurada" in casefolded
    assert "no aplicable todavia" in casefolded
    assert "no hay porcentajes ni capital fijados por defecto" in casefolded
    assert "derivados [activa]: prohibidos" in casefolded
    assert "ventas en corto [activa]: prohibidas" in casefolded
    assert "no protegen por si solos" in casefolded

    service.set_mode(PaperMode.TACTICAL)
    tactical_text = chat.handle("politica paper")
    assert "politica de riesgo paper del modo tactical" in tactical_text.casefold()


def test_politica_paper_shows_configured_limits(tmp_path: Path) -> None:
    service = _service(tmp_path)
    service.update_policy(
        max_open_positions=2,
        max_exposure_per_asset=Decimal("0.25"),
        max_total_exposure=Decimal("0.50"),
        max_loss_pct=Decimal("0.10"),
        max_drawdown_pct=Decimal("0.20"),
    )
    chat = _chat(service)

    text = chat.handle("politica paper")

    casefolded = text.casefold()
    assert "maximo de posiciones abiertas [activa]: maximo 2 posiciones" in casefolded
    assert "exposicion maxima por activo [activa]: 25.00% del nav" in casefolded
    assert "exposicion total maxima [activa]: 50.00% del nav" in casefolded
    assert "limite de perdida [no aplicable todavia]" in casefolded
    assert "limite de drawdown [no aplicable todavia]" in casefolded
    assert "10.00% del nav (solo informativo)" in casefolded
    assert "20.00% del nav (solo informativo)" in casefolded


# ---------------------------------------------------------------------------
# Persistencia versionada y migracion
# ---------------------------------------------------------------------------


def test_policy_limits_persist_across_service_instances(tmp_path: Path) -> None:
    store = PaperStore(tmp_path / "finance_paper")
    first = PaperFinanceService(store)
    first.update_policy(
        max_open_positions=4,
        max_exposure_per_asset=Decimal("0.30"),
        max_total_exposure=Decimal("0.60"),
        max_loss_pct=Decimal("0.08"),
        max_drawdown_pct=Decimal("0.15"),
    )

    second = PaperFinanceService(store)
    assert second.policy() == first.policy()
    assert second.policy().mode is PaperMode.CORE
    second.set_mode(PaperMode.TACTICAL)
    assert second.policy().max_open_positions is None


def test_tactical_and_core_policies_are_persisted_independently(tmp_path: Path) -> None:
    store = PaperStore(tmp_path / "finance_paper")
    first = PaperFinanceService(store)
    first.update_policy(max_open_positions=2)
    first.set_mode(PaperMode.TACTICAL)
    first.update_policy(max_open_positions=5, max_loss_pct=Decimal("0.03"))

    second = PaperFinanceService(store)
    second.set_mode(PaperMode.CORE)
    assert second.policy().max_open_positions == 2
    second.set_mode(PaperMode.TACTICAL)
    assert second.policy().max_open_positions == 5
    assert second.policy().max_loss_pct == Decimal("0.03")


def test_v1_state_migrates_to_core_mode_without_loss(tmp_path: Path) -> None:
    directory = tmp_path / "finance_paper"
    directory.mkdir(parents=True)
    (directory / "state.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "portfolio": {"cash": "5000", "positions": []},
            }
        ),
        encoding="utf-8",
    )
    service = PaperFinanceService(PaperStore(directory))

    assert service.mode() is PaperMode.CORE
    assert service.portfolio_summary()["cash"] == "5000.00"

    _register_tick(service, "AAPL", "100")
    reloaded = PaperFinanceService(PaperStore(directory))
    assert reloaded.mode() is PaperMode.CORE
    assert reloaded.portfolio_summary()["cash"] == "5000.00"
    assert reloaded.engine.last_prices()["AAPL"] == Decimal("100")
    reloaded.set_mode(PaperMode.TACTICAL)
    # La migracion conserva el capital previo en CORE y deja TACTICAL a 0:
    # nunca se crean dos carteras con el capital inicial duplicado.
    assert reloaded.portfolio_summary()["cash"] == "0.00"
    capital = reloaded.capital_summary()
    assert capital["modes"]["CORE"]["nav"] == "5000.00"
    assert capital["modes"]["TACTICAL"]["nav"] == "0.00"
    assert capital["total"] == "5000.00"


def test_fresh_install_never_duplicates_initial_capital_across_modes(
    tmp_path: Path,
) -> None:
    store = PaperStore(tmp_path / "finance_paper")
    service = PaperFinanceService(store)
    service.set_mode(PaperMode.TACTICAL)

    capital = service.capital_summary()
    assert capital["modes"]["CORE"]["nav"] == "10000.00"
    assert capital["modes"]["TACTICAL"]["nav"] == "0.00"
    assert capital["total"] == "10000.00"

    restored = PaperFinanceService(store)
    restored.set_mode(PaperMode.CORE)
    assert restored.portfolio_summary()["cash"] == "10000.00"
    restored.set_mode(PaperMode.TACTICAL)
    assert restored.portfolio_summary()["cash"] == "0.00"


def test_tactical_without_allocation_cannot_open_an_order(tmp_path: Path) -> None:
    service = _service(tmp_path)
    chat = _chat(service)

    assert "modo TACTICAL" in chat.handle("modo paper tactical")
    assert chat.handle("registra precio paper AAPL 100").startswith("[PAPER]")

    rejected = chat.handle("compra paper 1 de AAPL a mercado")

    casefolded = rejected.casefold()
    assert f"regla {RULE_NO_LEVERAGE}" in rejected
    assert "seria apalancarse" in casefolded
    assert "no se ha creado ninguna propuesta" in casefolded
    assert chat.pending_proposal is None
    assert service.portfolio_summary()["cash"] == "0.00"
    assert service.portfolio_summary()["positions"] == {}


# ---------------------------------------------------------------------------
# Revalidacion de la politica en el momento de la confirmacion
# ---------------------------------------------------------------------------


def test_pending_order_is_revalidated_against_policy_at_confirmation(tmp_path: Path) -> None:
    service = _service(tmp_path)
    chat = _chat(service)
    chat.handle("registra precio paper AAPL 100")
    chat.handle("compra paper 10 de AAPL a mercado")
    assert chat.pending_proposal is not None

    service.update_policy(max_exposure_per_asset=Decimal("0.05"))  # 500 < coste 1000
    confirmation = chat.execute_pending()

    casefolded = confirmation.casefold()
    assert "orden paper no ejecutada" in casefolded
    assert f"regla {RULE_MAX_EXPOSURE_PER_ASSET}" in confirmation
    summary = service.portfolio_summary()
    assert summary["cash"] == "10000.00"
    assert summary["positions"] == {}


def test_execute_confirmed_order_enforces_policy_directly(tmp_path: Path) -> None:
    service = _service(tmp_path)
    _register_tick(service, "AAPL", "100")
    filled = service.execute_confirmed_order(_market_order("o1", "AAPL", "10"))
    assert filled.filled

    service.update_policy(max_open_positions=1)
    _register_tick(service, "MSFT", "200")
    with pytest.raises(PolicyViolation) as excinfo:
        service.execute_confirmed_order(_market_order("o2", "MSFT", "1"))
    assert excinfo.value.rule == RULE_MAX_OPEN_POSITIONS

    service.update_policy(max_open_positions=None, max_exposure_per_asset=Decimal("0.10"))
    with pytest.raises(PolicyViolation) as excinfo:
        service.execute_confirmed_order(_market_order("o3", "AAPL", "5"))
    assert excinfo.value.rule == RULE_MAX_EXPOSURE_PER_ASSET


# ---------------------------------------------------------------------------
# Validador exacto (sin redondeos) a nivel de unidad
# ---------------------------------------------------------------------------


def test_validate_order_is_exact_about_limits() -> None:
    policy = InvestmentPolicy(
        mode=PaperMode.TACTICAL, max_exposure_per_asset=Decimal("0.5")
    )
    portfolio = PaperPortfolio(cash=Decimal("10000"))
    last_prices = {"AAPL": Decimal("100")}
    order = _market_order("o1", "AAPL", "50")

    validate_order(
        policy, order, portfolio=portfolio, last_prices=last_prices
    )  # 5000 == 5000: permitido, sin redondeo

    order_over = _market_order("o2", "AAPL", "50.000001")
    with pytest.raises(PolicyViolation) as excinfo:
        validate_order(
            policy, order_over, portfolio=portfolio, last_prices=last_prices
        )
    assert excinfo.value.rule == RULE_MAX_EXPOSURE_PER_ASSET

    sell_over = _market_order("o3", "AAPL", "1", side=Side.SELL)
    with pytest.raises(PolicyViolation) as excinfo:
        validate_order(policy, sell_over, portfolio=portfolio, last_prices=last_prices)
    assert excinfo.value.rule == RULE_NO_SHORTS


def test_validate_order_skips_numeric_checks_without_reference_price() -> None:
    policy = InvestmentPolicy(mode=PaperMode.CORE, max_total_exposure=Decimal("0.1"))
    portfolio = PaperPortfolio(cash=Decimal("10000"))
    order = _market_order("o1", "AAPL", "10")

    # Sin tick paper la orden MARKET no se puede estimar: el validador no
    # aplica limites numericos y el motor la rechazara como hoy.
    validate_order(policy, order, portfolio=portfolio, last_prices={})


# ---------------------------------------------------------------------------
# V2.7: comandos explicitos de capital y limites (sin LLM)
# ---------------------------------------------------------------------------


def test_capital_paper_command_shows_per_mode_and_combined_total(tmp_path: Path) -> None:
    service = _service(tmp_path)
    chat = _chat(service)

    text = chat.handle("capital paper")

    casefolded = text.casefold()
    assert "capital paper por modo" in casefolded
    assert "- core: efectivo 10.000,00 € · nav 10.000,00 € · posiciones 0" in casefolded
    assert "- tactical (sin asignar): efectivo 0,00 € · nav 0,00 € · posiciones 0" in casefolded
    assert "capital paper total combinado: 10.000,00 €" in casefolded
    assert "sin asignar" in casefolded


def test_capital_paper_assignment_moves_cash_without_duplicating(tmp_path: Path) -> None:
    service = _service(tmp_path)
    chat = _chat(service)

    assigned = chat.handle("capital paper tactical 3000")

    casefolded = assigned.casefold()
    assert "capital paper reasignado" in casefolded
    assert "- core: 7.000,00 € (nav)" in casefolded
    assert "- tactical: 3.000,00 € (nav)" in casefolded
    assert "capital paper total combinado: 10.000,00 €" in casefolded
    assert service.portfolio_summary()["cash"] == "7000.00"

    service.set_mode(PaperMode.TACTICAL)
    assert service.portfolio_summary()["cash"] == "3000.00"
    service.set_mode(PaperMode.CORE)
    assert service.portfolio_summary()["cash"] == "7000.00"


def test_capital_paper_reassignment_cannot_exceed_total_paper_capital(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path)
    chat = _chat(service)

    rejected = chat.handle("capital paper tactical 20000")

    casefolded = rejected.casefold()
    assert "asignacion de capital paper rechazada" in casefolded
    assert "excede el capital paper total" in casefolded
    assert "no se ha movido ningun euro paper" in casefolded
    summary = service.portfolio_summary()
    assert summary["cash"] == "10000.00"
    capital = service.capital_summary()
    assert capital["modes"]["TACTICAL"]["nav"] == "0.00"
    assert capital["total"] == "10000.00"


def test_capital_paper_invalid_amounts_and_currency_get_help(tmp_path: Path) -> None:
    service = _service(tmp_path)
    chat = _chat(service)

    zero = chat.handle("capital paper tactical 0")
    assert "importe de capital paper no es valido" in zero.casefold()

    currency = chat.handle("capital paper tactical 3000 usd")
    casefolded = currency.casefold()
    assert "moneda no coherente" in casefolded
    assert "euros" in casefolded

    syntax = chat.handle("capital paper tactical mucho")
    casefolded = syntax.casefold()
    assert "sintaxis de capital paper no reconocida" in casefolded
    assert "no se ha modificado" in casefolded

    empty = chat.handle("capital paper")
    assert "capital paper por modo" in empty.casefold()

    assert service.portfolio_summary()["cash"] == "10000.00"
    capital = service.capital_summary()
    assert capital["total"] == "10000.00"


def test_chat_config_commands_update_active_mode_policy(tmp_path: Path) -> None:
    service = _service(tmp_path)
    chat = _chat(service)

    positions = chat.handle("maximo posiciones paper 3")
    casefolded = positions.casefold()
    assert "politica paper del modo core configurada" in casefolded
    assert "maximo de posiciones abiertas [activa]: 3 posiciones" in casefolded
    assert service.policy().max_open_positions == 3
    assert service.policy().max_exposure_per_asset is None

    asset = chat.handle("maximo exposicion por activo paper 25%")
    assert "exposicion maxima por activo [activa]: 25% del nav" in asset.casefold()
    assert service.policy().max_exposure_per_asset == Decimal("0.25")

    total = chat.handle("maximo exposicion total paper 80")
    assert "exposicion total maxima [activa]: 80% del nav" in total.casefold()
    assert service.policy().max_total_exposure == Decimal("0.80")

    service.set_mode(PaperMode.TACTICAL)
    assert service.policy().max_open_positions is None
    assert service.policy().max_exposure_per_asset is None


def test_chat_config_invalid_values_are_rejected_with_help(tmp_path: Path) -> None:
    service = _service(tmp_path)
    chat = _chat(service)

    zero_positions = chat.handle("maximo posiciones paper 0")
    casefolded = zero_positions.casefold()
    assert "configuracion paper rechazada" in casefolded
    assert "comandos de configuracion paper" in casefolded
    assert service.policy().max_open_positions is None

    over = chat.handle("maximo exposicion total paper 150")
    assert "configuracion paper rechazada" in over.casefold()
    assert "no puede exceder 1.0" in over.casefold()
    assert service.policy().max_total_exposure is None

    garbage = chat.handle("maximo exposicion por activo paper mucho")
    casefolded = garbage.casefold()
    assert "sintaxis de configuracion paper no reconocida" in casefolded
    assert "no se ha modificado" in casefolded

    assert service.policy() == InvestmentPolicy(mode=PaperMode.CORE)


def test_configured_rule_blocks_order_before_proposal(tmp_path: Path) -> None:
    service = _service(tmp_path)
    chat = _chat(service)
    chat.handle("registra precio paper AAPL 100")

    configured = chat.handle("maximo exposicion total paper 10%")
    assert "exposicion total maxima [activa]: 10% del nav" in configured.casefold()

    # 10% de NAV 10000 = 1000; la orden de 15 uds a 100 cuesta 1500.
    blocked = chat.handle("compra paper 15 de AAPL a mercado")

    casefolded = blocked.casefold()
    assert f"regla {RULE_MAX_TOTAL_EXPOSURE}" in blocked
    assert "no se ha creado ninguna propuesta" in casefolded
    assert chat.pending_proposal is None
    assert service.portfolio_summary()["cash"] == "10000.00"
