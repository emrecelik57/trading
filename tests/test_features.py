"""Indicateurs techniques et normalisation en coupe transversale."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantfolio.features import (
    build_features,
    cross_sectional_rank,
    daily_returns,
    drawdown,
    momentum,
    momentum_12_1,
    realized_vol,
    rsi,
)


def _series(values: list[float]) -> pd.DataFrame:
    index = pd.bdate_range("2020-01-01", periods=len(values))
    return pd.DataFrame({"A": values}, index=index)


def test_momentum_is_a_simple_return():
    close = _series([100.0, 110.0, 121.0])
    assert momentum(close, 1)["A"].iloc[1] == pytest.approx(0.10)
    assert momentum(close, 2)["A"].iloc[2] == pytest.approx(0.21)
    assert np.isnan(momentum(close, 2)["A"].iloc[0])


def test_momentum_12_1_skips_the_last_month():
    close = _series(list(np.linspace(100, 200, 300)))
    value = momentum_12_1(close, long=252, skip=21)["A"].iloc[-1]
    expected = close["A"].iloc[-22] / close["A"].iloc[-253] - 1.0
    assert value == pytest.approx(expected)


def test_rsi_saturates_on_a_monotonic_series():
    close = _series([100.0 * 1.01**i for i in range(40)])
    assert rsi(close, 14)["A"].iloc[-1] == pytest.approx(100.0)

    falling = _series([100.0 * 0.99**i for i in range(40)])
    assert falling["A"].is_monotonic_decreasing
    assert rsi(falling, 14)["A"].iloc[-1] == pytest.approx(0.0, abs=1e-6)


def test_rsi_is_bounded():
    rng = np.random.default_rng(0)
    close = _series(list(100 * np.exp(np.cumsum(rng.normal(0, 0.02, 200)))))
    values = rsi(close, 14)["A"].dropna()
    assert len(values) > 100
    assert values.between(0.0, 100.0).all()


def test_realized_vol_matches_manual_computation():
    rng = np.random.default_rng(1)
    close = _series(list(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 100)))))
    returns = daily_returns(close)
    computed = realized_vol(returns, 21)["A"].iloc[-1]
    manual = returns["A"].iloc[-21:].std() * np.sqrt(252)
    assert computed == pytest.approx(manual)


def test_drawdown_is_zero_at_a_new_high():
    close = _series([100.0, 90.0, 80.0, 120.0])
    values = drawdown(close, window=3)["A"]
    # Fenetre incomplete tant qu'on n'a pas 3 observations.
    assert np.isnan(values.iloc[1])
    assert values.iloc[2] == pytest.approx(80.0 / 100.0 - 1.0)
    assert values.iloc[-1] == pytest.approx(0.0)


def test_cross_sectional_rank_is_centred_and_bounded():
    frame = pd.DataFrame(
        {"A": [1.0, 5.0], "B": [2.0, 4.0], "C": [3.0, 3.0], "D": [4.0, 2.0], "E": [5.0, 1.0]},
        index=pd.bdate_range("2020-01-01", periods=2),
    )
    ranked = cross_sectional_rank(frame, min_names=5)
    assert ranked.iloc[0].max() == pytest.approx(1.0)
    assert ranked.iloc[0].min() == pytest.approx(-0.6)
    assert ranked.loc[ranked.index[0], "E"] > ranked.loc[ranked.index[0], "A"]
    # L'ordre s'inverse le second jour.
    assert ranked.loc[ranked.index[1], "A"] > ranked.loc[ranked.index[1], "E"]


def test_cross_sectional_rank_needs_enough_names():
    frame = pd.DataFrame({"A": [1.0], "B": [2.0]}, index=pd.bdate_range("2020-01-01", periods=1))
    assert cross_sectional_rank(frame, min_names=5).isna().all().all()


def test_volatility_feature_is_inverted(panel, cfg):
    """Une volatilite elevee doit donner un indicateur bas (positif = favorable)."""
    featureset = build_features(panel, cfg.features, min_history=cfg.data.min_history)
    date = panel.dates[-1]
    raw = featureset.raw["vol_63"].loc[date].dropna()
    normalized = featureset.features["vol_63"].loc[date].dropna()
    common = raw.index.intersection(normalized.index)
    assert len(common) > 5
    assert raw[common].corr(normalized[common]) < -0.9


def test_features_only_use_past_data(panel, cfg):
    """Tronquer le futur ne change pas les indicateurs deja calcules."""
    cutoff = panel.dates[-60]
    full = build_features(panel, cfg.features, min_history=cfg.data.min_history)
    partial = build_features(panel.slice(end=cutoff), cfg.features, min_history=cfg.data.min_history)
    for name in full.names:
        left = full.features[name].loc[cutoff]
        right = partial.features[name].loc[cutoff]
        pd.testing.assert_series_equal(left, right, check_names=False, atol=1e-12)
