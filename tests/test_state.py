"""Comptabilite du portefeuille : PRU, plus-values, liquidites, persistance."""

from __future__ import annotations

import pandas as pd
import pytest

from quantfolio.state import PortfolioState


def test_buy_updates_cash_and_average_price():
    state = PortfolioState(cash=10_000.0)
    state.trade("A", 10, 100.0)
    assert state.cash == pytest.approx(9_000.0)
    assert state.positions["A"].shares == 10
    assert state.positions["A"].avg_price == pytest.approx(100.0)


def test_average_price_is_weighted_across_purchases():
    state = PortfolioState(cash=10_000.0)
    state.trade("A", 10, 100.0)
    state.trade("A", 10, 200.0)
    assert state.positions["A"].avg_price == pytest.approx(150.0)
    assert state.positions["A"].shares == 20


def test_selling_does_not_move_the_average_price():
    state = PortfolioState(cash=10_000.0)
    state.trade("A", 10, 100.0)
    state.trade("A", -5, 300.0)
    assert state.positions["A"].avg_price == pytest.approx(100.0)
    assert state.positions["A"].shares == 5


def test_realized_pnl_is_recorded_on_sale():
    state = PortfolioState(cash=10_000.0)
    state.trade("A", 10, 100.0)
    state.trade("A", -10, 150.0)
    assert state.realized_pnl == pytest.approx(500.0)
    assert "A" not in state.positions  # ligne soldee
    assert state.cash == pytest.approx(10_500.0)


def test_costs_reduce_cash_and_are_accumulated():
    state = PortfolioState(cash=10_000.0)
    state.trade("A", 10, 100.0, cost=5.0)
    assert state.cash == pytest.approx(8_995.0)
    assert state.total_costs == pytest.approx(5.0)


def test_equity_combines_cash_and_positions():
    state = PortfolioState(cash=1_000.0)
    state.trade("A", 5, 100.0)  # reste 500 en cash
    prices = pd.Series({"A": 120.0})
    assert state.market_value(prices) == pytest.approx(600.0)
    assert state.equity(prices) == pytest.approx(1_100.0)


def test_unpriced_positions_are_listed():
    state = PortfolioState(cash=0.0)
    state.set_position("A", 10, 100.0)
    state.set_position("B", 10, 50.0)
    assert state.unpriced(pd.Series({"A": 100.0})) == ["B"]


def test_history_records_every_trade():
    state = PortfolioState(cash=10_000.0)
    state.trade("A", 10, 100.0, date="2024-01-02")
    state.trade("A", -4, 110.0, date="2024-01-09")
    assert len(state.history) == 2
    assert state.history[0]["side"] == "BUY"
    assert state.history[1]["side"] == "SELL"
    assert state.history[1]["shares"] == 4


def test_save_and_load_roundtrip(tmp_path):
    state = PortfolioState(cash=5_000.0, currency="EUR")
    state.trade("A", 10, 100.0, cost=2.0)
    path = state.save(tmp_path / "portfolio.json")

    reloaded = PortfolioState.load(path)
    assert reloaded.cash == pytest.approx(state.cash)
    assert reloaded.currency == "EUR"
    assert reloaded.positions["A"].shares == 10
    assert reloaded.positions["A"].avg_price == pytest.approx(100.0)
    assert reloaded.total_costs == pytest.approx(2.0)
    assert len(reloaded.history) == 1


def test_load_missing_file_explains_what_to_do(tmp_path):
    with pytest.raises(FileNotFoundError, match="init-portfolio"):
        PortfolioState.load(tmp_path / "absent.json")


def test_load_or_create_falls_back_to_the_configured_capital(tmp_path):
    state = PortfolioState.load_or_create(tmp_path / "absent.json", cash=42_000.0)
    assert state.cash == pytest.approx(42_000.0)


def test_to_frame_reports_unrealised_pnl():
    state = PortfolioState(cash=0.0)
    state.set_position("A", 10, 100.0)
    frame = state.to_frame(pd.Series({"A": 130.0}))
    row = frame.iloc[0]
    assert row["pnl"] == pytest.approx(300.0)
    assert row["pnl_pct"] == pytest.approx(0.30)
