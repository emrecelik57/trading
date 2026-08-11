"""Modele de classement cross-sectionnel.

Le modele ne cherche pas a prevoir le niveau du marche : il apprend a ordonner
les titres de l'univers entre eux a un horizon de quelques jours. C'est un
probleme nettement plus facile et beaucoup plus stable qu'une prevision
directionnelle, et c'est exactement ce dont on a besoin pour decider quoi
acheter et quoi vendre.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import ModelConfig


@dataclass
class TrainReport:
    """Resume d'un entrainement."""

    fitted: bool
    n_rows: int = 0
    n_features: int = 0
    train_start: pd.Timestamp | None = None
    train_end: pd.Timestamp | None = None
    reason: str = ""
    metrics: dict[str, float] = field(default_factory=dict)

    def describe(self) -> str:
        if not self.fitted:
            return f"Modele non entraine ({self.reason})"
        span = ""
        if self.train_start is not None and self.train_end is not None:
            span = f" du {self.train_start:%Y-%m-%d} au {self.train_end:%Y-%m-%d}"
        extra = ""
        if self.metrics:
            extra = " | " + ", ".join(f"{k}={v:.4f}" for k, v in self.metrics.items())
        return f"{self.n_rows} observations x {self.n_features} indicateurs{span}{extra}"


def _build_estimator(cfg: ModelConfig):
    """Cree l'estimateur scikit-learn correspondant a la configuration."""
    if cfg.kind == "gbm":
        from sklearn.ensemble import HistGradientBoostingRegressor

        params: dict[str, Any] = {
            "loss": "squared_error",
            "learning_rate": 0.05,
            "max_iter": 300,
            "max_depth": 3,
            "min_samples_leaf": 50,
            "l2_regularization": 1.0,
            "max_features": 0.8,
            "early_stopping": False,
            "random_state": cfg.random_state,
        }
        params.update(cfg.params)
        return HistGradientBoostingRegressor(**params)

    if cfg.kind == "ridge":
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        params = {"alpha": 5.0}
        params.update(cfg.params)
        return Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("model", Ridge(**params)),
            ]
        )

    raise ValueError(f"model.kind non supporte pour l'apprentissage : {cfg.kind!r}")


def sample_weights(dates: pd.DatetimeIndex, half_life_years: float) -> np.ndarray | None:
    """Poids decroissants avec l'anciennete de l'observation."""
    if half_life_years is None or half_life_years <= 0:
        return None
    dates = pd.DatetimeIndex(dates)
    age_years = (dates.max() - dates).days / 365.25
    return np.power(0.5, age_years / half_life_years)


class Ranker:
    """Enveloppe autour de l'estimateur : entrainement, prevision, sauvegarde."""

    def __init__(self, cfg: ModelConfig):
        self.cfg = cfg
        self.estimator = None
        self.feature_names: list[str] = []
        self.report = TrainReport(fitted=False, reason="jamais entraine")

    # -- cycle de vie -------------------------------------------------------
    @property
    def is_fitted(self) -> bool:
        return self.estimator is not None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> TrainReport:
        """Entraine sur un echantillon deja purge des fuites temporelles."""
        if self.cfg.kind == "rules":
            self.report = TrainReport(fitted=False, reason="model.kind = rules")
            return self.report

        frame = X.copy()
        frame["__y__"] = y
        frame = frame[frame["__y__"].notna()]
        # Une ligne sans aucun indicateur n'apporte rien.
        frame = frame[frame[list(X.columns)].notna().any(axis=1)]

        if len(frame) < self.cfg.min_train_rows:
            self.report = TrainReport(
                fitted=False,
                n_rows=len(frame),
                reason=(
                    f"{len(frame)} observations < model.min_train_rows="
                    f"{self.cfg.min_train_rows}"
                ),
            )
            return self.report

        dates = frame.index.get_level_values("date")
        target = frame.pop("__y__")
        features = frame

        estimator = _build_estimator(self.cfg)
        weights = sample_weights(dates, self.cfg.sample_half_life_years)
        try:
            estimator.fit(features.to_numpy(dtype="float64"), target.to_numpy(dtype="float64"),
                          **({"sample_weight": weights} if weights is not None else {}))
        except TypeError:
            # Certains pipelines n'acceptent pas sample_weight directement.
            estimator.fit(features.to_numpy(dtype="float64"), target.to_numpy(dtype="float64"))

        self.estimator = estimator
        self.feature_names = list(features.columns)
        in_sample = pd.Series(
            estimator.predict(features.to_numpy(dtype="float64")), index=features.index
        )
        self.report = TrainReport(
            fitted=True,
            n_rows=len(features),
            n_features=features.shape[1],
            train_start=dates.min(),
            train_end=dates.max(),
            metrics={"ic_in_sample": information_coefficient(in_sample, target)},
        )
        return self.report

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Score brut par (date, ticker). Serie vide si le modele n'est pas pret."""
        if not self.is_fitted or len(X) == 0:
            return pd.Series(dtype="float64", index=X.index)
        missing = [c for c in self.feature_names if c not in X.columns]
        if missing:
            raise ValueError(
                f"Indicateurs absents lors de la prevision : {missing}. "
                "Le modele sauvegarde ne correspond pas a la configuration actuelle "
                "(relancez `quantfolio train`)."
            )
        values = X[self.feature_names].to_numpy(dtype="float64")
        return pd.Series(self.estimator.predict(values), index=X.index)

    # -- persistance --------------------------------------------------------
    def save(self, path: str | Path) -> Path:
        import joblib

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "estimator": self.estimator,
                "feature_names": self.feature_names,
                "cfg": self.cfg,
                "report": self.report,
                "saved_at": pd.Timestamp.now(),
            },
            path,
        )
        return path

    @classmethod
    def load(cls, path: str | Path, cfg: ModelConfig | None = None) -> "Ranker":
        import joblib

        payload = joblib.load(Path(path))
        ranker = cls(cfg or payload["cfg"])
        ranker.estimator = payload["estimator"]
        ranker.feature_names = payload["feature_names"]
        ranker.report = payload.get("report", TrainReport(fitted=True))
        return ranker


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------
def information_coefficient(predictions: pd.Series, target: pd.Series) -> float:
    """Correlation de Spearman globale entre prevision et realise."""
    joined = pd.DataFrame({"p": predictions, "y": target}).dropna()
    if len(joined) < 10:
        return float("nan")
    return float(joined["p"].corr(joined["y"], method="spearman"))


def daily_information_coefficient(predictions: pd.Series, target: pd.Series) -> pd.Series:
    """IC calcule date par date : la mesure de reference d'un modele de rang.

    Un IC moyen de 0.02 a 0.05 est deja exploitable sur un univers d'actions ;
    au-dela de 0.15 sur des donnees reelles, soupconnez une fuite de donnees.
    """
    joined = pd.DataFrame({"p": predictions, "y": target}).dropna()
    if joined.empty:
        return pd.Series(dtype="float64")

    def _ic(group: pd.DataFrame) -> float:
        if len(group) < 5:
            return float("nan")
        return group["p"].corr(group["y"], method="spearman")

    return joined.groupby(level="date").apply(_ic).dropna()


def feature_importance(ranker: Ranker, X: pd.DataFrame, y: pd.Series, n_repeats: int = 5,
                       max_rows: int = 20_000) -> pd.Series:
    """Importance par permutation (couteux : calcule sur un echantillon)."""
    from sklearn.inspection import permutation_importance

    if not ranker.is_fitted:
        return pd.Series(dtype="float64")
    frame = X[ranker.feature_names].copy()
    frame["__y__"] = y
    frame = frame.dropna(subset=["__y__"])
    if len(frame) > max_rows:
        frame = frame.sample(max_rows, random_state=ranker.cfg.random_state)
    target = frame.pop("__y__")
    result = permutation_importance(
        ranker.estimator,
        frame.to_numpy(dtype="float64"),
        target.to_numpy(dtype="float64"),
        n_repeats=n_repeats,
        random_state=ranker.cfg.random_state,
        scoring="r2",
    )
    return pd.Series(result.importances_mean, index=ranker.feature_names).sort_values(
        ascending=False
    )
