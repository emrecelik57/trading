"""Estimation du risque et pilotage de l'exposition globale."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import PortfolioConfig

TRADING_DAYS = 252


@dataclass
class ExposureDecision:
    """Exposition brute cible et explication de sa valeur."""

    exposure: float
    estimated_vol: float
    regime_on: bool
    reason: str


def covariance(returns: pd.DataFrame, shrinkage: float = 0.3) -> pd.DataFrame:
    """Covariance annualisee, contractee vers sa diagonale.

    Le shrinkage evite les matrices instables quand le nombre d'observations
    est du meme ordre que le nombre de titres.
    """
    clean = returns.dropna(axis=1, how="all")
    if clean.shape[0] < 5 or clean.shape[1] == 0:
        return pd.DataFrame(index=returns.columns, columns=returns.columns, dtype="float64")
    cov = clean.cov(min_periods=5) * TRADING_DAYS
    diagonal = pd.DataFrame(
        np.diag(np.diag(cov.to_numpy())), index=cov.index, columns=cov.columns
    )
    shrunk = (1.0 - shrinkage) * cov + shrinkage * diagonal
    return shrunk.reindex(index=returns.columns, columns=returns.columns)


def portfolio_volatility(
    weights: pd.Series,
    returns: pd.DataFrame,
    shrinkage: float = 0.3,
    fallback_correlation: float = 0.35,
) -> float:
    """Volatilite annualisee du portefeuille pour un jeu de poids donne."""
    weights = weights[weights.abs() > 1e-12]
    if weights.empty:
        return 0.0

    tickers = [t for t in weights.index if t in returns.columns]
    if not tickers:
        return float("nan")
    window = returns[tickers]
    weight_vector = weights.reindex(tickers).to_numpy(dtype="float64")

    cov = covariance(window, shrinkage=shrinkage)
    if cov.isna().to_numpy().all():
        return float("nan")

    matrix = cov.to_numpy(dtype="float64")
    if np.isnan(matrix).any():
        # Repli : volatilites individuelles + correlation moyenne supposee.
        vols = window.std().to_numpy(dtype="float64") * np.sqrt(TRADING_DAYS)
        vols = np.nan_to_num(vols, nan=float(np.nanmean(vols)) if np.isfinite(np.nanmean(vols)) else 0.2)
        variance = float(
            np.sum((weight_vector * vols) ** 2)
            + fallback_correlation
            * (np.sum(weight_vector * vols) ** 2 - np.sum((weight_vector * vols) ** 2))
        )
        return float(np.sqrt(max(variance, 0.0)))

    variance = float(weight_vector @ matrix @ weight_vector)
    return float(np.sqrt(max(variance, 0.0)))


def market_index(close: pd.DataFrame, available: pd.DataFrame | None = None) -> pd.Series:
    """Indice equipondere de l'univers, utilise faute de benchmark explicite."""
    returns = close.div(close.shift(1)) - 1.0
    if available is not None:
        returns = returns.where(available)
    mean_returns = returns.mean(axis=1).fillna(0.0)
    return (1.0 + mean_returns).cumprod()


def regime_is_on(index: pd.Series, window: int, as_of: pd.Timestamp | None = None) -> bool:
    """Le marche est-il au-dessus de sa moyenne mobile longue ?"""
    series = index.dropna()
    if as_of is not None:
        series = series.loc[series.index <= pd.Timestamp(as_of)]
    if len(series) < max(20, window // 4):
        return True  # pas assez d'historique : on ne bride pas l'exposition
    moving_average = series.rolling(window, min_periods=max(20, window // 4)).mean()
    last_price = float(series.iloc[-1])
    last_ma = float(moving_average.iloc[-1])
    if not np.isfinite(last_ma):
        return True
    return last_price >= last_ma


def target_exposure(
    weights: pd.Series,
    returns_window: pd.DataFrame,
    cfg: PortfolioConfig,
    market: pd.Series | None = None,
    as_of: pd.Timestamp | None = None,
) -> ExposureDecision:
    """Determine l'exposition brute cible du portefeuille.

    Deux garde-fous se combinent :
      - ciblage de volatilite : on reduit la voilure quand le panier est agite ;
      - filtre de regime : on reduit encore quand le marche est sous sa moyenne
        mobile longue, la ou les strategies momentum souffrent le plus.
    """
    if weights.empty or weights.abs().sum() <= 0:
        return ExposureDecision(0.0, 0.0, True, "aucun titre selectionne")

    normalized = weights / weights.abs().sum()
    estimated_vol = portfolio_volatility(
        normalized, returns_window, shrinkage=cfg.covariance_shrinkage
    )

    if not np.isfinite(estimated_vol) or estimated_vol <= 1e-6:
        exposure = cfg.max_gross
        reason = "volatilite non estimable, exposition maximale"
    else:
        exposure = min(cfg.max_gross, cfg.vol_target / estimated_vol)
        reason = (
            f"vol estimee {estimated_vol:.1%} vs cible {cfg.vol_target:.1%}"
        )

    regime_on = True
    if cfg.regime_filter and market is not None:
        regime_on = regime_is_on(market, cfg.regime_ma, as_of=as_of)
        if not regime_on:
            exposure *= cfg.regime_risk_off
            reason += f" | marche sous sa MM{cfg.regime_ma}, exposition reduite"

    return ExposureDecision(
        exposure=float(max(0.0, min(exposure, cfg.max_gross))),
        estimated_vol=float(estimated_vol) if np.isfinite(estimated_vol) else float("nan"),
        regime_on=regime_on,
        reason=reason,
    )
