"""Lissage du signal : effet attendu sur la stabilite et la rotation."""

from __future__ import annotations

import pandas as pd
import pytest

from quantfolio.backtest import run_backtest
from quantfolio.engine import Engine

from conftest import make_config


def _score_series(cfg, n_dates: int = 20) -> pd.DataFrame:
    engine = Engine(cfg)
    engine.prepare()
    dates = engine.load().panel.dates[-n_dates:]
    return pd.DataFrame({date: engine.score_at(date) for date in dates}).T


def test_smoothing_window_only_looks_backwards(cfg):
    cfg.model.score_smoothing = 10
    engine = Engine(cfg)
    engine.prepare()
    dates = engine.load().panel.dates
    date = dates[-30]

    window = engine._smoothing_window(date)
    assert len(window) == 10
    assert window.max() == date  # la seance du jour est incluse
    assert (window <= date).all()  # aucune seance future


def test_smoothing_window_is_truncated_at_the_start(cfg):
    cfg.model.score_smoothing = 50
    engine = Engine(cfg)
    engine.prepare()
    dates = engine.load().panel.dates
    window = engine._smoothing_window(dates[3])
    assert len(window) == 4  # on ne remonte pas avant le debut de l'historique


def test_smoothing_of_one_is_a_no_op(cfg):
    """score_smoothing = 1 doit redonner exactement le score du jour."""
    cfg.model.kind = "rules"
    cfg.model.score_smoothing = 1
    engine = Engine(cfg)
    dataset = engine.prepare()
    date = engine.last_date()

    from quantfolio.engine import rank_series

    scores = engine.score_at(date)
    raw = dataset.rules.xs(date, level="date").where(dataset.valid.xs(date, level="date"))
    expected = rank_series(rank_series(raw, cfg.features.min_cross_section), cfg.features.min_cross_section)
    pd.testing.assert_series_equal(scores.dropna(), expected.dropna(), check_names=False)


def test_smoothing_makes_scores_more_stable(cfg):
    """Le score lisse doit varier moins d'une seance a l'autre."""
    cfg.model.kind = "rules"

    cfg.model.score_smoothing = 1
    raw = _score_series(cfg)
    cfg.model.score_smoothing = 10
    smoothed = _score_series(cfg)

    raw_moves = raw.diff().abs().mean().mean()
    smoothed_moves = smoothed.diff().abs().mean().mean()
    assert smoothed_moves < raw_moves


@pytest.mark.slow
def test_smoothing_reduces_turnover():
    tickers = [f"T{i:02d}" for i in range(12)]
    turnovers = {}
    for span in (1, 10):
        cfg = make_config(
            tickers,
            model={"kind": "rules", "score_smoothing": span},
            backtest={"start": "2019-01-01", "rebalance_days": 5},
        )
        engine = Engine(cfg)
        engine.prepare()
        turnovers[span] = run_backtest(engine, cfg).summary["rotation_annuelle"]
    assert turnovers[10] < turnovers[1]
