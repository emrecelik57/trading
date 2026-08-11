"""Moteur : assemble donnees, indicateurs, modele et portefeuille cible."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import Config
from .data.panel import MarketData, load_market_data
from .features import FeatureSet, build_features
from .labels import build_labels, label_available_until
from .model import Ranker, TrainReport
from .portfolio import PortfolioTarget, build_target
from .risk import market_index
from .rules import rules_score


def rank_series(values: pd.Series, min_names: int = 3) -> pd.Series:
    """Rang d'une coupe transversale, ramene dans [-1, 1]."""
    clean = values.dropna()
    if len(clean) < min_names:
        return pd.Series(np.nan, index=values.index, dtype="float64")
    ranked = (clean.rank(pct=True) - 0.5) * 2.0
    return ranked.reindex(values.index)


@dataclass
class Dataset:
    """Jeu de donnees complet, indexe par (date, ticker)."""

    X: pd.DataFrame
    y: pd.Series
    rules: pd.Series
    valid: pd.Series
    feature_names: list[str]

    def rows_until(self, cutoff: pd.Timestamp) -> pd.Index:
        dates = self.X.index.get_level_values("date")
        return self.X.index[dates <= pd.Timestamp(cutoff)]


class Engine:
    """Orchestre le pipeline complet, de la donnee brute a l'ordre a passer."""

    def __init__(self, cfg: Config, market_data: MarketData | None = None):
        self.cfg = cfg
        self.market_data: MarketData | None = market_data
        self.featureset: FeatureSet | None = None
        self.dataset: Dataset | None = None
        self._rules_wide: pd.DataFrame | None = None

    # -- preparation --------------------------------------------------------
    def load(self, refresh: bool = False, verbose: bool = False) -> MarketData:
        if self.market_data is None:
            self.market_data = load_market_data(self.cfg, refresh=refresh, verbose=verbose)
        return self.market_data

    def prepare(self, refresh: bool = False, verbose: bool = False) -> Dataset:
        """Calcule indicateurs, score a base de regles et labels."""
        market_data = self.load(refresh=refresh, verbose=verbose)
        panel = market_data.panel

        self.featureset = build_features(
            panel, self.cfg.features, min_history=self.cfg.data.min_history
        )
        self._rules_wide = rules_score(
            self.featureset, min_names=self.cfg.features.min_cross_section
        )
        labels = build_labels(
            panel.close,
            horizon=self.cfg.model.horizon,
            kind=self.cfg.model.label,
            mask=self.featureset.valid,
            min_names=self.cfg.features.min_cross_section,
        )

        X = self.featureset.to_long()
        index = X.index
        dates, tickers = panel.dates, panel.tickers
        self.dataset = Dataset(
            X=X,
            y=_to_long(labels, dates, tickers, index),
            rules=_to_long(self._rules_wide, dates, tickers, index),
            valid=_to_long(self.featureset.valid, dates, tickers, index).fillna(0.0).astype(bool),
            feature_names=self.featureset.names,
        )
        if verbose:
            usable = int(self.dataset.valid.sum())
            print(
                f"{len(panel)} seances, {len(panel.tickers)} titres, "
                f"{len(self.dataset.feature_names)} indicateurs, "
                f"{usable} observations exploitables."
            )
        return self.dataset

    def ensure_prepared(self) -> Dataset:
        if self.dataset is None:
            self.prepare()
        assert self.dataset is not None
        return self.dataset

    # -- apprentissage ------------------------------------------------------
    def training_sample(self, as_of: pd.Timestamp) -> tuple[pd.DataFrame, pd.Series]:
        """Echantillon utilisable a la date `as_of`, purge de toute fuite.

        Une observation datee de u utilise les prix jusqu'a u + horizon pour
        construire son label : elle n'existe donc pas encore a la date `as_of`
        si u + horizon > as_of. On ajoute une periode d'embargo pour couper le
        chevauchement residuel entre labels successifs.
        """
        dataset = self.ensure_prepared()
        dates = self.market_data.panel.dates  # type: ignore[union-attr]
        purge = self.cfg.model.horizon + self.cfg.model.embargo_days
        cutoff = label_available_until(dates, as_of, purge)

        row_dates = dataset.X.index.get_level_values("date")
        mask = (row_dates <= cutoff) & dataset.valid.to_numpy() & dataset.y.notna().to_numpy()
        if self.cfg.model.max_train_years > 0:
            floor = pd.Timestamp(cutoff) - pd.DateOffset(
                days=int(self.cfg.model.max_train_years * 365.25)
            )
            mask &= row_dates >= floor
        return dataset.X[mask], dataset.y[mask]

    def fit(self, as_of: pd.Timestamp | None = None, verbose: bool = False) -> Ranker:
        """Entraine le modele avec l'information disponible a `as_of`."""
        dataset = self.ensure_prepared()
        as_of = pd.Timestamp(as_of) if as_of is not None else dataset.X.index.get_level_values("date").max()
        ranker = Ranker(self.cfg.model)
        if self.cfg.model.kind == "rules":
            ranker.report = TrainReport(fitted=False, reason="model.kind = rules")
            return ranker
        X, y = self.training_sample(as_of)
        report = ranker.fit(X, y)
        if verbose:
            print(f"Entrainement au {pd.Timestamp(as_of):%Y-%m-%d} : {report.describe()}")
        return ranker

    # -- scoring ------------------------------------------------------------
    def _smoothing_window(self, date: pd.Timestamp) -> pd.DatetimeIndex:
        """Seances sur lesquelles moyenner le signal, `date` incluse.

        Uniquement des seances passees : le lissage ne doit pas faire entrer
        d'information posterieure a la decision.
        """
        span = max(1, int(self.cfg.model.score_smoothing))
        dates = self.load().panel.dates
        position = int(dates.searchsorted(pd.Timestamp(date), side="right"))
        return dates[max(0, position - span) : position]

    def score_at(self, date: pd.Timestamp, ranker: Ranker | None = None) -> pd.Series:
        """Score final par titre a une date : melange modele + regles.

        Le score est un rang cross-sectionnel dans [-1, 1] : +1 designe le
        titre le mieux classe du jour, -1 le moins bien classe. Il se lit comme
        une conviction relative, pas comme une prevision de rendement.
        """
        dataset = self.ensure_prepared()
        date = pd.Timestamp(date)
        try:
            rows = dataset.X.xs(date, level="date", drop_level=True)
        except KeyError:
            return pd.Series(dtype="float64")

        valid = dataset.valid.xs(date, level="date", drop_level=True)
        window = self._smoothing_window(date)
        # Le masque de validite est celui du jour de la decision : un titre
        # non negociable aujourd'hui ne doit pas etre score, meme s'il l'etait
        # les seances precedentes.
        rules = _mean_over_window(dataset.rules, window).reindex(valid.index).where(valid)
        rules_rank = rank_series(rules, self.cfg.features.min_cross_section)

        weight = float(self.cfg.model.rules_weight)
        tradable = valid[valid].index
        if ranker is None or not ranker.is_fitted or len(tradable) == 0:
            blended = rules_rank
        else:
            history = dataset.X.loc[
                dataset.X.index.get_level_values("date").isin(window)
                & dataset.X.index.get_level_values("ticker").isin(tradable)
            ]
            predictions = _mean_over_window(ranker.predict(history), window)
            predictions = predictions.reindex(rows.index)
            model_rank = rank_series(predictions, self.cfg.features.min_cross_section)
            if model_rank.isna().all():
                blended = rules_rank
            elif rules_rank.isna().all():
                blended = model_rank
            else:
                blended = (1.0 - weight) * model_rank.fillna(0.0) + weight * rules_rank.fillna(0.0)
                blended = blended.where(model_rank.notna() | rules_rank.notna())

        blended = blended.where(valid)
        return rank_series(blended, self.cfg.features.min_cross_section).rename("score")

    # -- portefeuille -------------------------------------------------------
    def market_series(self) -> pd.Series:
        """Serie de reference du filtre de regime."""
        market_data = self.load()
        if market_data.benchmark is not None:
            return market_data.benchmark
        return market_index(market_data.panel.close, market_data.panel.available)

    def volatilities_at(self, date: pd.Timestamp) -> pd.Series:
        """Volatilite annualisee par titre a une date (pour le dimensionnement)."""
        assert self.featureset is not None
        key = f"vol_{self.cfg.portfolio.vol_lookback}"
        if key not in self.featureset.raw:
            key = f"vol_{max(self.cfg.features.vol_windows)}"
        frame = self.featureset.raw[key]
        if pd.Timestamp(date) not in frame.index:
            return pd.Series(dtype="float64")
        return frame.loc[pd.Timestamp(date)]

    def returns_window(self, date: pd.Timestamp) -> pd.DataFrame:
        """Rendements recents utilises pour estimer la covariance."""
        market_data = self.load()
        returns = market_data.panel.returns()
        window = returns.loc[returns.index <= pd.Timestamp(date)]
        return window.tail(self.cfg.portfolio.vol_lookback)

    def target_at(
        self,
        date: pd.Timestamp,
        ranker: Ranker | None = None,
        scores: pd.Series | None = None,
        holdings: set[str] | None = None,
    ) -> PortfolioTarget:
        """Portefeuille cible a une date donnee.

        `holdings` active l'hysterese de selection : sans lui, la cible est
        calculee comme si l'on partait de zero, ce qui genere beaucoup plus
        d'allers-retours.
        """
        date = pd.Timestamp(date)
        scores = self.score_at(date, ranker) if scores is None else scores
        return build_target(
            scores=scores,
            vols=self.volatilities_at(date),
            returns_window=self.returns_window(date),
            cfg=self.cfg.portfolio,
            market=self.market_series(),
            as_of=date,
            holdings=holdings,
        )

    # -- diagnostic ---------------------------------------------------------
    def last_date(self) -> pd.Timestamp:
        return pd.Timestamp(self.load().panel.dates.max())


def _mean_over_window(values: pd.Series, window: pd.DatetimeIndex) -> pd.Series:
    """Moyenne par ticker d'une serie (date, ticker) sur les dates fournies."""
    if values.empty:
        return pd.Series(dtype="float64")
    selected = values[values.index.get_level_values("date").isin(window)]
    if selected.empty:
        return pd.Series(dtype="float64")
    return selected.groupby(level="ticker").mean()


def _to_long(
    wide: pd.DataFrame, dates: pd.DatetimeIndex, tickers: list[str], index: pd.MultiIndex
) -> pd.Series:
    """Aplati un tableau date x ticker sur un MultiIndex (date, ticker)."""
    aligned = wide.reindex(index=dates, columns=tickers)
    return pd.Series(aligned.to_numpy(dtype="float64").ravel(), index=index)
