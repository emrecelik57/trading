"""Generation des ordres : contraintes de tresorerie, quantites, garde-fous."""

from __future__ import annotations

import pandas as pd
import pytest

from quantfolio.config import PortfolioConfig
from quantfolio.orders import execute_plan, generate_orders
from quantfolio.state import PortfolioState


@pytest.fixture
def pcfg() -> PortfolioConfig:
    return PortfolioConfig(
        max_positions=5,
        max_weight=0.5,
        min_weight=0.01,
        no_trade_band=0.02,
        min_order_notional=100.0,
        cash_buffer=0.0,
        fractional_shares=False,
    )


@pytest.fixture
def prices() -> pd.Series:
    return pd.Series({"A": 100.0, "B": 50.0, "C": 25.0}, dtype="float64")


def test_buys_never_exceed_available_cash(pcfg, prices):
    state = PortfolioState(cash=1000.0)
    target = pd.Series({"A": 0.5, "B": 0.5, "C": 0.0})
    plan = generate_orders(target, prices, state, pcfg)

    spent = sum(order.notional for order in plan.buys)
    assert spent <= state.cash + 1e-9
    assert plan.cash_after >= -1e-9


def test_sells_are_capped_at_the_shares_held(pcfg, prices):
    state = PortfolioState(cash=0.0)
    state.set_position("A", shares=5, avg_price=100.0)
    target = pd.Series({"A": 0.0, "B": 0.0, "C": 0.0})
    plan = generate_orders(target, prices, state, pcfg)

    assert len(plan.sells) == 1
    assert plan.sells[0].shares == 5  # jamais de vente a decouvert


def test_zero_target_triggers_a_full_exit(pcfg, prices):
    state = PortfolioState(cash=100.0)
    state.set_position("A", shares=7, avg_price=90.0)
    target = pd.Series({"A": 0.0, "B": 0.0, "C": 0.0})
    plan = generate_orders(target, prices, state, pcfg)

    execute_plan(plan, state)
    assert "A" not in state.positions


def test_no_order_when_the_portfolio_already_matches(pcfg, prices):
    state = PortfolioState(cash=0.0)
    state.set_position("A", shares=10, avg_price=100.0)  # 1000, soit 100 %
    target = pd.Series({"A": 1.0, "B": 0.0, "C": 0.0})
    plan = generate_orders(target, prices, state, pcfg)
    assert plan.orders == []


def test_small_drift_is_ignored(pcfg, prices):
    """Un ecart sous la bande ne doit pas declencher d'ordre."""
    state = PortfolioState(cash=100.0)
    state.set_position("A", shares=9, avg_price=100.0)  # 900 sur 1000 = 90 %
    target = pd.Series({"A": 0.91, "B": 0.0, "C": 0.0})
    plan = generate_orders(target, prices, state, pcfg)
    assert plan.orders == []


def test_orders_below_the_minimum_notional_are_skipped(pcfg, prices):
    pcfg.min_order_notional = 10_000.0
    state = PortfolioState(cash=10_000.0)
    target = pd.Series({"A": 0.05, "B": 0.0, "C": 0.0})
    plan = generate_orders(target, prices, state, pcfg)
    assert plan.buys == []


def test_whole_shares_by_default(pcfg, prices):
    state = PortfolioState(cash=10_000.0)
    target = pd.Series({"A": 0.33, "B": 0.0, "C": 0.0})
    plan = generate_orders(target, prices, state, pcfg)
    for order in plan.orders:
        assert order.shares == int(order.shares)


def test_fractional_shares_when_enabled(pcfg, prices):
    pcfg.fractional_shares = True
    state = PortfolioState(cash=10_000.0)
    target = pd.Series({"A": 0.333, "B": 0.0, "C": 0.0})
    plan = generate_orders(target, prices, state, pcfg)
    assert plan.buys
    assert plan.buys[0].notional == pytest.approx(3330.0, rel=1e-6)


def test_sells_finance_buys_in_the_same_session(pcfg, prices):
    """Sans tresorerie, une rotation complete reste possible."""
    state = PortfolioState(cash=0.0)
    state.set_position("A", shares=10, avg_price=100.0)  # 1000, tout le capital
    target = pd.Series({"A": 0.0, "B": 1.0, "C": 0.0})
    plan = generate_orders(target, prices, state, pcfg)

    assert [o.ticker for o in plan.sells] == ["A"]
    assert [o.ticker for o in plan.buys] == ["B"]
    # Les ventes sont placees avant les achats dans le plan.
    assert plan.orders[0].side == "SELL"


def test_cash_buffer_is_preserved(pcfg, prices):
    pcfg.cash_buffer = 0.20
    state = PortfolioState(cash=10_000.0)
    target = pd.Series({"A": 0.8, "B": 0.0, "C": 0.0})
    plan = generate_orders(target, prices, state, pcfg)
    assert plan.cash_after >= 0.20 * plan.equity - prices["A"] - 1e-9


def test_unpriced_holdings_are_reported(pcfg, prices):
    state = PortfolioState(cash=1000.0)
    state.set_position("ZZZ", shares=10, avg_price=5.0)
    plan = generate_orders(pd.Series({"A": 1.0}), prices, state, pcfg)
    assert any("ZZZ" in warning for warning in plan.warnings)


def test_costs_are_charged_on_execution(pcfg, prices):
    state = PortfolioState(cash=10_000.0)
    target = pd.Series({"A": 0.5, "B": 0.0, "C": 0.0})
    plan = generate_orders(target, prices, state, pcfg, cost_bps=10.0)
    cash_before = state.cash
    execute_plan(plan, state, cost_bps=10.0)

    traded = sum(o.notional for o in plan.buys)
    assert state.cash == pytest.approx(cash_before - traded - traded * 10.0 / 10_000.0)
    assert state.total_costs == pytest.approx(traded * 10.0 / 10_000.0)


def test_plan_serialises_to_json_friendly_types(pcfg, prices):
    import json

    state = PortfolioState(cash=10_000.0)
    plan = generate_orders(pd.Series({"A": 0.5, "B": 0.5}), prices, state, pcfg)
    payload = json.loads(json.dumps(plan.as_dict()))
    assert payload["orders"]
    assert {"ticker", "side", "shares", "price"} <= set(payload["orders"][0])
