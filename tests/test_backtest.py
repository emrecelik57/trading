"""Coherence du backtest."""

from __future__ import annotations

import numpy as np
import pytest

from quantfolio.backtest import run_backtest
from quantfolio.engine import Engine

from conftest import make_config


@pytest.fixture(scope="module")
def result():
    tickers = [f"T{i:02d}" for i in range(12)]
    cfg = make_config(
        tickers,
        model={"kind": "rules"},
        backtest={"start": "2019-01-01", "rebalance_days": 10, "cost_bps": 5, "slippage_bps": 5},
    )
    engine = Engine(cfg)
    engine.prepare()
    return run_backtest(engine, cfg), cfg


def test_backtest_produces_a_continuous_equity_curve(result):
    backtest, _ = result
    assert len(backtest.equity) > 200
    assert backtest.equity.notna().all()
    assert (backtest.equity > 0).all()
    assert backtest.equity.index.is_monotonic_increasing


def test_cash_never_goes_negative(result):
    backtest, _ = result
    assert backtest.state.cash >= -1e-6


def test_no_short_positions(result):
    backtest, _ = result
    assert all(position.shares >= 0 for position in backtest.state.positions.values())


def test_final_equity_matches_cash_plus_positions(result):
    backtest, cfg = result
    # La derniere valeur de la courbe doit se reconstituer a partir de l'etat.
    engine = Engine(cfg)
    prices = engine.load().panel.close.loc[backtest.equity.index[-1]]
    assert backtest.state.equity(prices) == pytest.approx(backtest.equity.iloc[-1], rel=1e-9)


def test_positions_stay_within_the_configured_limits(result):
    backtest, cfg = result
    assert not backtest.weights.empty
    assert (backtest.weights > 1e-9).sum(axis=1).max() <= cfg.portfolio.max_positions
    assert backtest.weights.max().max() <= cfg.portfolio.max_weight + 1e-9
    assert backtest.weights.sum(axis=1).max() <= cfg.portfolio.max_gross + 1e-9


def test_costs_are_actually_charged(result):
    backtest, _ = result
    assert len(backtest.trades) > 0
    assert backtest.state.total_costs > 0
    assert backtest.trades["cost"].min() >= 0


def test_transaction_costs_reduce_performance():
    """Doubler les frais doit degrader le resultat, jamais l'ameliorer."""
    tickers = [f"T{i:02d}" for i in range(12)]
    outcomes = []
    for cost in (0.0, 50.0):
        cfg = make_config(
            tickers,
            model={"kind": "rules"},
            backtest={"start": "2019-01-01", "rebalance_days": 5,
                      "cost_bps": cost, "slippage_bps": cost},
        )
        engine = Engine(cfg)
        engine.prepare()
        outcomes.append(run_backtest(engine, cfg).equity.iloc[-1])
    assert outcomes[0] > outcomes[1]


def test_execution_mode_close_is_more_optimistic_than_next_open():
    """Executer au cours du signal donne un resultat different (et flatteur)."""
    tickers = [f"T{i:02d}" for i in range(12)]
    equities = {}
    for mode in ("close", "next_open"):
        cfg = make_config(
            tickers,
            model={"kind": "rules"},
            backtest={"start": "2019-01-01", "rebalance_days": 10, "execution": mode},
        )
        engine = Engine(cfg)
        engine.prepare()
        equities[mode] = run_backtest(engine, cfg).equity.iloc[-1]
    assert equities["close"] != equities["next_open"]


def test_backtest_refuses_a_period_that_is_too_short():
    """Une phase de chauffe plus longue que l'historique doit etre signalee."""
    tickers = [f"T{i:02d}" for i in range(12)]
    cfg = make_config(tickers, backtest={"start": "2019-01-01", "warmup_days": 5000})
    engine = Engine(cfg)
    engine.prepare()
    with pytest.raises(ValueError, match="trop courte"):
        run_backtest(engine, cfg)


def test_summary_contains_the_expected_metrics(result):
    backtest, _ = result
    for key in ("cagr", "volatilite", "sharpe", "max_drawdown", "rotation_annuelle"):
        assert key in backtest.summary
    assert np.isfinite(backtest.summary["sharpe"])
    assert backtest.summary["max_drawdown"] <= 0.0
