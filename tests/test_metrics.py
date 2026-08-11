"""Mesures de performance, verifiees sur des cas connus."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantfolio.metrics import (
    annual_volatility,
    beta_alpha,
    cagr,
    format_summary,
    max_drawdown,
    sharpe_ratio,
    summarize,
    to_returns,
)


def _equity(values: list[float], periods: int | None = None) -> pd.Series:
    index = pd.bdate_range("2020-01-01", periods=periods or len(values))
    return pd.Series(values, index=index)


def test_cagr_on_a_doubling_over_two_years():
    index = pd.DatetimeIndex(["2020-01-01", "2022-01-01"])
    equity = pd.Series([100.0, 200.0], index=index)
    # (2)^(1/2) - 1 = 41.4 %, a l'arrondi des annees bissextiles pres.
    assert cagr(equity) == pytest.approx(0.4142, abs=1e-3)


def test_max_drawdown_measures_the_worst_peak_to_trough():
    equity = _equity([100.0, 120.0, 60.0, 90.0])
    assert max_drawdown(equity) == pytest.approx(-0.5)


def test_max_drawdown_is_zero_when_never_losing():
    equity = _equity([100.0, 110.0, 120.0])
    assert max_drawdown(equity) == pytest.approx(0.0)


def test_annual_volatility_scales_by_sqrt_252():
    rng = np.random.default_rng(0)
    returns = pd.Series(rng.normal(0, 0.01, 2000))
    assert annual_volatility(returns) == pytest.approx(0.01 * np.sqrt(252), rel=0.05)


def test_sharpe_of_a_constant_return_series_is_infinite_free():
    returns = pd.Series([0.001] * 100)
    # Ecart-type nul : la mesure n'a pas de sens, on renvoie NaN plutot qu'inf.
    assert np.isnan(sharpe_ratio(returns))


def test_sharpe_is_positive_for_a_rising_curve():
    rng = np.random.default_rng(1)
    returns = pd.Series(rng.normal(0.0008, 0.01, 1000))
    assert sharpe_ratio(returns) > 0


def test_beta_of_a_series_against_itself_is_one():
    rng = np.random.default_rng(2)
    returns = pd.Series(rng.normal(0, 0.01, 500))
    beta, alpha = beta_alpha(returns, returns)
    assert beta == pytest.approx(1.0)
    assert alpha == pytest.approx(0.0, abs=1e-9)


def test_beta_doubles_with_leverage():
    rng = np.random.default_rng(3)
    market = pd.Series(rng.normal(0, 0.01, 500))
    beta, _ = beta_alpha(market * 2.0, market)
    assert beta == pytest.approx(2.0)


def test_to_returns_inverts_the_equity_curve():
    equity = _equity([100.0, 110.0, 99.0])
    returns = to_returns(equity)
    assert returns.iloc[0] == pytest.approx(0.10)
    assert returns.iloc[1] == pytest.approx(-0.10)


def test_summarize_reports_the_expected_keys():
    rng = np.random.default_rng(4)
    equity = _equity(list(100 * np.exp(np.cumsum(rng.normal(0.0005, 0.01, 500)))))
    summary = summarize(equity, benchmark=equity * 0.9)
    for key in ("cagr", "sharpe", "max_drawdown", "beta", "information_ratio"):
        assert key in summary
    assert summary["nb_seances"] == 500


def test_format_summary_handles_missing_values():
    rendered = format_summary({"sharpe": float("nan"), "cagr": 0.1234})
    assert "n/a" in rendered
    assert "12.34%" in rendered


def test_metrics_are_safe_on_a_degenerate_curve():
    equity = pd.Series([100.0], index=pd.DatetimeIndex(["2020-01-01"]))
    summary = summarize(equity)
    assert np.isnan(summary["cagr"])
    assert format_summary(summary)
