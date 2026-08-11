"""Le test le plus important : verifier qu'aucune donnee future ne fuit."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantfolio.engine import Engine
from quantfolio.labels import build_labels, forward_return, label_available_until


def test_forward_return_uses_only_future_prices():
    close = pd.DataFrame({"A": [10.0, 11.0, 12.0, 13.0]})
    forward = forward_return(close, horizon=2)
    # A t=0 : 12/10 - 1 = 0.2 ; les deux dernieres lignes ne sont pas connues.
    assert forward["A"].iloc[0] == pytest.approx(0.2)
    assert np.isnan(forward["A"].iloc[2])
    assert np.isnan(forward["A"].iloc[3])


def test_label_available_until_respects_horizon():
    dates = pd.bdate_range("2020-01-01", periods=20)
    # Au 20e jour, un label d'horizon 5 n'est connu que jusqu'au 15e.
    cutoff = label_available_until(dates, dates[19], horizon=5)
    assert cutoff == dates[14]


def test_training_sample_never_sees_the_future(cfg):
    """L'echantillon d'entrainement s'arrete avant horizon + embargo."""
    engine = Engine(cfg)
    engine.prepare()
    dates = engine.load().panel.dates
    as_of = dates[len(dates) // 2]

    X, y = engine.training_sample(as_of)
    assert len(X) > 0

    last_train_date = X.index.get_level_values("date").max()
    purge = cfg.model.horizon + cfg.model.embargo_days
    position_as_of = int(dates.searchsorted(as_of))
    position_last = int(dates.searchsorted(last_train_date))

    # Toute observation conservee a son label deja observable a `as_of`.
    assert position_as_of - position_last >= purge
    assert y.notna().all()


def test_scores_at_date_do_not_depend_on_later_data(cfg):
    """Tronquer l'historique apres la date de decision ne change pas le score.

    C'est la verification decisive : si un indicateur regardait vers l'avant,
    le score calcule sur l'historique complet differerait de celui calcule sur
    l'historique tronque a la date de decision.
    """
    engine_full = Engine(cfg)
    engine_full.prepare()
    dates = engine_full.load().panel.dates
    as_of = dates[-120]
    scores_full = engine_full.score_at(as_of, ranker=None)

    truncated = Engine(cfg)
    truncated.load()
    truncated.market_data.panel = truncated.market_data.panel.slice(end=as_of)
    truncated.prepare()
    scores_truncated = truncated.score_at(as_of, ranker=None)

    common = scores_full.dropna().index.intersection(scores_truncated.dropna().index)
    assert len(common) > 5
    pd.testing.assert_series_equal(
        scores_full.reindex(common),
        scores_truncated.reindex(common),
        atol=1e-10,
    )


def test_labels_match_realised_forward_returns(cfg):
    """Le label d'une date correspond bien au rendement futur, pas passe."""
    engine = Engine(cfg)
    engine.prepare()
    close = engine.load().panel.close
    horizon = cfg.model.horizon

    labels = build_labels(close, horizon=horizon, kind="excess")
    date = close.index[300]
    ticker = close.columns[0]
    realised = close[ticker].iloc[300 + horizon] / close[ticker].iloc[300] - 1.0
    universe_mean = (close.shift(-horizon) / close - 1.0).loc[date].mean()
    assert labels.loc[date, ticker] == pytest.approx(realised - universe_mean, abs=1e-9)
