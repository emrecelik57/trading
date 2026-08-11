"""Construction du portefeuille : selection, plafonds, hysterese, exposition."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantfolio.config import PortfolioConfig
from quantfolio.portfolio import (
    apply_no_trade_band,
    build_target,
    cap_weights,
    current_weights,
    select_and_weight,
)
from quantfolio.risk import portfolio_volatility, regime_is_on


@pytest.fixture
def pcfg() -> PortfolioConfig:
    return PortfolioConfig(
        max_positions=5, max_weight=0.30, min_weight=0.02, cash_buffer=0.0, vol_target=10.0
    )


def _scores(values: dict[str, float]) -> pd.Series:
    return pd.Series(values, dtype="float64")


def _flat_vols(index, value: float = 0.20) -> pd.Series:
    return pd.Series(value, index=index, dtype="float64")


# --------------------------------------------------------------------------
# Plafonnement
# --------------------------------------------------------------------------
def test_cap_weights_redistributes_the_excess():
    weights = pd.Series({"A": 0.6, "B": 0.2, "C": 0.2})
    capped = cap_weights(weights, cap=0.4)
    assert capped["A"] == pytest.approx(0.4)
    assert capped.sum() == pytest.approx(1.0)
    assert capped["B"] == pytest.approx(capped["C"])


def test_cap_weights_leaves_cash_when_everything_is_capped():
    weights = pd.Series({"A": 0.5, "B": 0.5})
    capped = cap_weights(weights, cap=0.3)
    # Impossible d'investir 100 % avec 2 lignes plafonnees a 30 % : le reste
    # doit rester en liquidites plutot que de violer la contrainte.
    assert capped.max() <= 0.3 + 1e-9
    assert capped.sum() == pytest.approx(0.6)


# --------------------------------------------------------------------------
# Selection
# --------------------------------------------------------------------------
def test_selection_respects_max_positions_and_threshold(pcfg):
    scores = _scores({f"T{i}": 0.9 - 0.1 * i for i in range(10)})
    weights, detail = select_and_weight(scores, _flat_vols(scores.index), pcfg)
    assert (weights > 0).sum() <= pcfg.max_positions
    # Aucun titre sous le seuil d'achat ne doit entrer.
    for ticker in detail.index:
        assert scores[ticker] > pcfg.score_threshold


def test_selection_prefers_the_best_scores(pcfg):
    # Plafond desserre : sinon les trois lignes saturent a max_weight et
    # l'ordre des convictions n'est plus observable.
    pcfg.max_weight = 1.0
    scores = _scores({"A": 0.9, "B": 0.5, "C": 0.1, "D": -0.5})
    weights, _ = select_and_weight(scores, _flat_vols(scores.index), pcfg)
    assert weights["A"] > weights["B"] > weights["C"]
    assert weights["D"] == 0.0


def test_lower_volatility_gets_a_bigger_weight(pcfg):
    pcfg.max_weight = 1.0
    scores = _scores({"A": 0.8, "B": 0.8})
    vols = pd.Series({"A": 0.10, "B": 0.40})
    weights, _ = select_and_weight(scores, vols, pcfg)
    # Conviction identique, risque quadruple : le poids doit etre bien plus petit.
    assert weights["A"] > weights["B"]
    assert weights["A"] / weights["B"] == pytest.approx(4.0, rel=0.05)


def test_weights_never_exceed_the_cap(pcfg):
    scores = _scores({"A": 1.0, "B": 0.05, "C": 0.04})
    weights, _ = select_and_weight(scores, _flat_vols(scores.index), pcfg)
    assert weights.max() <= pcfg.max_weight + 1e-9
    assert weights.sum() <= 1.0 + 1e-9


def test_small_positions_are_dropped(pcfg):
    pcfg.min_weight = 0.10
    scores = _scores({"A": 1.0, "B": 0.9, "C": 0.01})
    weights, _ = select_and_weight(scores, _flat_vols(scores.index), pcfg)
    assert weights[weights > 0].min() >= pcfg.min_weight - 1e-9


def test_empty_selection_when_nothing_beats_the_threshold(pcfg):
    scores = _scores({"A": -0.2, "B": -0.9})
    weights, detail = select_and_weight(scores, _flat_vols(scores.index), pcfg)
    assert weights.sum() == 0.0
    assert detail.empty


# --------------------------------------------------------------------------
# Hysterese
# --------------------------------------------------------------------------
def test_held_position_survives_a_small_score_drop(pcfg):
    """Un titre detenu qui passe legerement sous le seuil d'achat est conserve."""
    scores = _scores({"A": 0.9, "B": 0.8, "C": -0.10})
    vols = _flat_vols(scores.index)

    without = select_and_weight(scores, vols, pcfg)[0]
    with_holding = select_and_weight(scores, vols, pcfg, holdings={"C"})[0]

    assert without["C"] == 0.0
    assert with_holding["C"] > 0.0


def test_held_position_is_sold_below_the_exit_threshold(pcfg):
    scores = _scores({"A": 0.9, "B": 0.8, "C": -0.90})
    weights, _ = select_and_weight(scores, _flat_vols(scores.index), pcfg, holdings={"C"})
    assert weights["C"] == 0.0


def test_hold_bonus_avoids_swapping_for_a_marginal_candidate():
    cfg = PortfolioConfig(max_positions=1, max_weight=1.0, min_weight=0.0, hold_bonus=0.15)
    scores = _scores({"HELD": 0.50, "NEW": 0.55})
    weights, _ = select_and_weight(scores, _flat_vols(scores.index), cfg, holdings={"HELD"})
    assert weights["HELD"] > 0 and weights["NEW"] == 0.0

    # Un candidat nettement meilleur doit tout de meme prendre la place.
    better = _scores({"HELD": 0.50, "NEW": 0.95})
    weights, _ = select_and_weight(better, _flat_vols(better.index), cfg, holdings={"HELD"})
    assert weights["NEW"] > 0 and weights["HELD"] == 0.0


# --------------------------------------------------------------------------
# Exposition et risque
# --------------------------------------------------------------------------
def test_vol_targeting_reduces_exposure_when_risk_is_high():
    index = pd.bdate_range("2020-01-01", periods=120)
    rng = np.random.default_rng(0)
    calm = pd.DataFrame(rng.normal(0, 0.004, (120, 3)), index=index, columns=["A", "B", "C"])
    wild = pd.DataFrame(rng.normal(0, 0.030, (120, 3)), index=index, columns=["A", "B", "C"])
    scores = _scores({"A": 0.9, "B": 0.6, "C": 0.3})
    cfg = PortfolioConfig(max_positions=3, max_weight=0.5, cash_buffer=0.0, vol_target=0.15)

    calm_target = build_target(scores, _flat_vols(scores.index), calm, cfg)
    wild_target = build_target(scores, _flat_vols(scores.index), wild, cfg)

    assert calm_target.invested > wild_target.invested
    assert wild_target.exposure.estimated_vol > calm_target.exposure.estimated_vol


def test_exposure_never_exceeds_max_gross():
    index = pd.bdate_range("2020-01-01", periods=120)
    rng = np.random.default_rng(1)
    returns = pd.DataFrame(rng.normal(0, 0.001, (120, 3)), index=index, columns=["A", "B", "C"])
    scores = _scores({"A": 0.9, "B": 0.6, "C": 0.3})
    cfg = PortfolioConfig(max_positions=3, max_weight=0.5, cash_buffer=0.05, max_gross=1.0)

    target = build_target(scores, _flat_vols(scores.index), returns, cfg)
    assert target.invested <= cfg.max_gross * (1 - cfg.cash_buffer) + 1e-9
    assert target.cash_weight >= cfg.cash_buffer - 1e-9


def test_regime_filter_cuts_exposure_in_a_downtrend():
    index = pd.bdate_range("2019-01-01", periods=400)
    rising = pd.Series(np.linspace(100, 200, 400), index=index)
    falling = pd.Series(np.linspace(200, 100, 400), index=index)
    assert regime_is_on(rising, window=200)
    assert not regime_is_on(falling, window=200)

    rng = np.random.default_rng(2)
    returns = pd.DataFrame(rng.normal(0, 0.01, (120, 2)), index=index[:120], columns=["A", "B"])
    scores = _scores({"A": 0.9, "B": 0.6})
    cfg = PortfolioConfig(max_positions=2, max_weight=0.6, cash_buffer=0.0, regime_risk_off=0.4)

    on = build_target(scores, _flat_vols(scores.index), returns, cfg, market=rising, as_of=index[-1])
    off = build_target(scores, _flat_vols(scores.index), returns, cfg, market=falling, as_of=index[-1])
    assert off.invested == pytest.approx(on.invested * cfg.regime_risk_off, rel=1e-6)


def test_portfolio_volatility_accounts_for_diversification():
    index = pd.bdate_range("2020-01-01", periods=250)
    rng = np.random.default_rng(3)
    common = rng.normal(0, 0.01, 250)
    correlated = pd.DataFrame({"A": common, "B": common}, index=index)
    independent = pd.DataFrame(
        {"A": common, "B": rng.normal(0, 0.01, 250)}, index=index
    )
    weights = pd.Series({"A": 0.5, "B": 0.5})
    assert portfolio_volatility(weights, independent) < portfolio_volatility(weights, correlated)


# --------------------------------------------------------------------------
# Bande de non-negociation
# --------------------------------------------------------------------------
def test_no_trade_band_keeps_small_drifts():
    target = pd.Series({"A": 0.11, "B": 0.30})
    current = pd.Series({"A": 0.10, "B": 0.10})
    adjusted = apply_no_trade_band(target, current, band=0.02)
    assert adjusted["A"] == pytest.approx(0.10)  # ecart de 1 %, on ne touche pas
    assert adjusted["B"] == pytest.approx(0.30)  # ecart de 20 %, on ajuste


def test_no_trade_band_never_blocks_entries_or_exits():
    target = pd.Series({"NEW": 0.005, "OUT": 0.0})
    current = pd.Series({"NEW": 0.0, "OUT": 0.01})
    adjusted = apply_no_trade_band(target, current, band=0.05)
    assert adjusted["NEW"] == pytest.approx(0.005)
    assert adjusted["OUT"] == pytest.approx(0.0)


def test_current_weights_include_cash_in_the_total():
    prices = pd.Series({"A": 100.0, "B": 50.0})
    weights = current_weights({"A": 10.0}, prices, cash=1000.0)
    assert weights["A"] == pytest.approx(1000.0 / 2000.0)
    assert weights["B"] == 0.0
