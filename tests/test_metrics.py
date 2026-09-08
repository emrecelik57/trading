"""Verification des mesures de risque sur des series aux proprietes connues."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from volatrade import metrics
from tests.conftest import make_frame


def test_volatilite_annualisee_sur_serie_calibree():
    # Rendements alternes +/-1 % : ecart-type quotidien connu.
    rng = np.random.default_rng(0)
    daily = pd.Series(0.01 * rng.standard_normal(2000))
    prices = 100 * np.exp(daily.cumsum())
    result = metrics.annualized_volatility(metrics.log_returns(prices))
    assert result == pytest.approx(0.01 * np.sqrt(252), rel=0.10)


def test_volatilite_nulle_sur_serie_constante():
    prices = pd.Series([10.0] * 300)
    assert metrics.annualized_volatility(metrics.log_returns(prices)) == pytest.approx(0.0)


def test_ewma_reagit_plus_vite_que_lecart_type():
    calme = np.full(200, 0.001)
    choc = np.full(30, 0.05) * np.tile([1, -1], 15)
    prices = pd.Series(100 * np.exp(np.cumsum(np.concatenate([calme, choc]))))
    returns = metrics.log_returns(prices)
    assert metrics.ewma_volatility(returns) > metrics.annualized_volatility(returns)


def test_regime_de_volatilite_detecte_lacceleration():
    rng = np.random.default_rng(4)
    calme = 0.005 * rng.standard_normal(200)
    agite = 0.04 * rng.standard_normal(30)
    prices = pd.Series(100 * np.exp(np.cumsum(np.concatenate([calme, agite]))))
    assert metrics.volatility_regime(metrics.log_returns(prices)) > 1.5


def test_max_drawdown_connu():
    prices = pd.Series([100, 120, 60, 80, 90])
    assert metrics.max_drawdown(prices) == pytest.approx(-0.5)
    assert metrics.current_drawdown(prices) == pytest.approx(-0.25)


def test_ulcer_index_positif_et_nul_si_monotone():
    assert metrics.ulcer_index(pd.Series([1.0, 2.0, 3.0, 4.0])) == pytest.approx(0.0)
    assert metrics.ulcer_index(pd.Series([100, 50, 80, 40])) > 0


def test_var_et_cvar_coherentes():
    rng = np.random.default_rng(1)
    returns = pd.Series(0.02 * rng.standard_normal(1000))
    var = metrics.historical_var(returns, 0.95)
    cvar = metrics.conditional_var(returns, 0.95)
    assert 0 < var < 0.10
    assert cvar > var, "la perte moyenne dans la queue depasse toujours la VaR"
    assert metrics.historical_var(returns, 0.95, horizon=10) == pytest.approx(var * np.sqrt(10))


def test_var_indisponible_si_historique_trop_court():
    assert np.isnan(metrics.historical_var(pd.Series([0.01, -0.02, 0.03])))


def test_beta_sur_titre_amplifiant_lindice():
    rng = np.random.default_rng(2)
    indice = pd.Series(0.01 * rng.standard_normal(500))
    titre = 2.0 * indice + 0.001 * rng.standard_normal(500)
    assert metrics.beta(titre, indice) == pytest.approx(2.0, rel=0.05)
    assert metrics.correlation(titre, indice) > 0.95


def test_sharpe_negatif_sur_serie_baissiere():
    prices = pd.Series(100 * np.exp(np.cumsum(np.full(300, -0.002))))
    assert metrics.sharpe_ratio(metrics.log_returns(prices)) < 0


def test_sortino_ignore_la_volatilite_haussiere():
    hausses = np.where(np.arange(300) % 2 == 0, 0.03, 0.001)
    prices = pd.Series(100 * np.exp(np.cumsum(hausses)))
    returns = metrics.log_returns(prices)
    assert metrics.sortino_ratio(returns) > metrics.sharpe_ratio(returns)


def test_atr_suit_lamplitude_reelle():
    frame = make_frame(np.full(100, 50.0), amplitude=0.03)
    # Amplitude haut-bas de ~6 % autour d'un cours stable.
    assert metrics.atr_percent(frame) == pytest.approx(0.06, rel=0.15)


def test_rsi_borne_et_extreme_sur_serie_monotone():
    hausse = pd.Series(np.linspace(10, 50, 200))
    baisse = pd.Series(np.linspace(50, 10, 200))
    assert metrics.rsi(hausse).iloc[-1] == pytest.approx(100.0)
    assert metrics.rsi(baisse).iloc[-1] == pytest.approx(0.0, abs=1e-6)
    melange = metrics.rsi(pd.Series(100 + np.sin(np.arange(300))))
    assert melange.dropna().between(0, 100).all()


def test_cagr_sur_doublement_annuel():
    prices = pd.Series(np.linspace(100, 200, 253))
    assert metrics.cagr(prices) == pytest.approx(1.0, rel=0.02)


def test_compute_metrics_renvoie_un_tableau_complet(uptrend_frame, benchmark_returns):
    result = metrics.compute_metrics("HAUT", uptrend_frame, benchmark_returns)
    assert result.ticker == "HAUT"
    assert result.observations == len(uptrend_frame)
    assert result.vol_ann_1y > 0
    assert result.max_drawdown <= 0
    assert set(result.as_dict()) >= {"vol_ewma", "var_95_1d", "sharpe", "beta"}
    assert result.expected_daily_move > 0
