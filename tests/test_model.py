"""Le modele apprend-il vraiment ?"""

from __future__ import annotations

import pandas as pd
import pytest

from quantfolio.config import Config
from quantfolio.engine import Engine
from quantfolio.model import Ranker, daily_information_coefficient


def _signal_config(tickers: list[str], dispersion: float, **model) -> Config:
    """Univers ou la tendance propre de chaque titre domine le bruit.

    Avec une forte dispersion des tendances, le momentum passe est reellement
    informatif sur le rendement futur relatif : un modele correctement cable
    doit le retrouver.
    """
    payload = {"kind": "gbm", "min_train_rows": 500, "rules_weight": 0.0}
    payload.update(model)
    return Config.from_dict(
        {
            "universe": {"tickers": tickers, "benchmark": None},
            "data": {
                "provider": "synthetic",
                "start": "2015-01-01",
                "end": "2022-12-31",
                "use_cache": False,
                "provider_args": {
                    "seed": 3,
                    "drift_dispersion": dispersion,
                    "vol_range": [0.008, 0.015],
                    "market_vol": 0.006,
                },
            },
            "model": payload,
        }
    )


def _out_of_sample_ic(cfg: Config, train_position: int = 700, step: int = 10) -> pd.Series:
    engine = Engine(cfg)
    engine.prepare()
    dates = engine.load().panel.dates
    ranker = engine.fit(as_of=dates[train_position])

    records = []
    for position in range(train_position, len(dates) - cfg.model.horizon, step):
        date = dates[position]
        scores = engine.score_at(date, ranker).dropna()
        if scores.empty:
            continue
        records.append(
            pd.Series(
                scores.to_numpy(),
                index=pd.MultiIndex.from_product(
                    [[date], scores.index], names=["date", "ticker"]
                ),
            )
        )
    assert records, "aucun score produit"
    return daily_information_coefficient(pd.concat(records), engine.dataset.y)


@pytest.mark.slow
def test_model_learns_when_a_signal_exists():
    """Sur un univers ou le momentum informe, l'IC hors echantillon est franchement positif."""
    tickers = [f"T{i:02d}" for i in range(15)]
    ic = _out_of_sample_ic(_signal_config(tickers, dispersion=0.60))
    assert ic.mean() > 0.15, f"IC trop faible : {ic.mean():.4f}"
    assert float((ic > 0).mean()) > 0.75


@pytest.mark.slow
def test_model_stays_neutral_without_signal():
    """Sur un univers sans structure, l'IC reste proche de zero.

    Un IC eleve ici signalerait une fuite de donnees plutot qu'un bon modele.
    """
    tickers = [f"T{i:02d}" for i in range(15)]
    ic = _out_of_sample_ic(_signal_config(tickers, dispersion=0.0))
    assert abs(ic.mean()) < 0.12, f"IC anormalement eleve sans signal : {ic.mean():.4f}"


def test_ranker_refuses_to_fit_without_enough_rows(cfg):
    engine = Engine(cfg)
    engine.prepare()
    cfg.model.min_train_rows = 10**9
    ranker = engine.fit(as_of=engine.last_date())
    assert not ranker.is_fitted
    assert "min_train_rows" in ranker.report.reason


def test_ranker_roundtrip(tmp_path, cfg):
    engine = Engine(cfg)
    engine.prepare()
    date = engine.last_date()
    ranker = engine.fit(as_of=date)
    assert ranker.is_fitted

    path = ranker.save(tmp_path / "model.joblib")
    reloaded = Ranker.load(path, cfg.model)
    assert reloaded.feature_names == ranker.feature_names

    rows = engine.dataset.X.xs(date, level="date")
    pd.testing.assert_series_equal(ranker.predict(rows), reloaded.predict(rows))


def test_predict_rejects_mismatched_features(cfg):
    engine = Engine(cfg)
    engine.prepare()
    ranker = engine.fit(as_of=engine.last_date())
    rows = engine.dataset.X.xs(engine.last_date(), level="date")
    with pytest.raises(ValueError, match="Indicateurs absents"):
        ranker.predict(rows.drop(columns=[ranker.feature_names[0]]))


def test_rules_only_model_produces_scores(cfg):
    cfg.model.kind = "rules"
    engine = Engine(cfg)
    engine.prepare()
    ranker = engine.fit(as_of=engine.last_date())
    assert not ranker.is_fitted

    scores = engine.score_at(engine.last_date(), ranker).dropna()
    assert len(scores) > 5
    assert scores.between(-1.0, 1.0).all()
