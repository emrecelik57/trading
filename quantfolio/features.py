"""Calcul des indicateurs et mise en forme du jeu de donnees d'apprentissage.

Deux idees structurent ce module :

1. Tous les indicateurs sont calcules de facon vectorisee sur des tableaux
   date x ticker, en n'utilisant que le passe (aucune fuite de donnees futures).
2. Chaque indicateur est ensuite normalise *en coupe transversale* : a une date
   donnee, on remplace la valeur brute par son rang parmi les titres cotes ce
   jour-la, ramene dans [-1, 1]. Le modele apprend donc a comparer les titres
   entre eux, ce qui reste stable quel que soit le regime de marche.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import FeaturesConfig
from .data.panel import Panel


@dataclass
class FeatureSet:
    """Indicateurs prets a l'emploi."""

    # Indicateurs normalises en coupe transversale, dans [-1, 1]
    features: dict[str, pd.DataFrame]
    # Valeurs brutes utiles ailleurs (volatilite pour le dimensionnement...)
    raw: dict[str, pd.DataFrame] = field(default_factory=dict)
    # Titre cotable et disposant d'assez d'historique
    valid: pd.DataFrame | None = None

    @property
    def names(self) -> list[str]:
        return sorted(self.features)

    def to_long(self) -> pd.DataFrame:
        """Empile les indicateurs en un DataFrame indexe par (date, ticker)."""
        names = self.names
        template = self.features[names[0]]
        index = pd.MultiIndex.from_product(
            [template.index, template.columns], names=["date", "ticker"]
        )
        data = {name: self.features[name].to_numpy(dtype="float64").ravel() for name in names}
        return pd.DataFrame(data, index=index)


# --------------------------------------------------------------------------
# Indicateurs elementaires
# --------------------------------------------------------------------------
def _min_periods(window: int, floor: int = 5) -> int:
    """Nombre minimum d'observations : la moitie de la fenetre, sans depasser
    la fenetre elle-meme (pandas refuse min_periods > window)."""
    return int(min(window, max(floor, window // 2)))


def daily_returns(close: pd.DataFrame) -> pd.DataFrame:
    return close.div(close.shift(1)) - 1.0


def momentum(close: pd.DataFrame, window: int) -> pd.DataFrame:
    return close.div(close.shift(window)) - 1.0


def momentum_12_1(close: pd.DataFrame, long: int = 252, skip: int = 21) -> pd.DataFrame:
    """Momentum 12 mois en sautant le dernier mois (effet de reversion court terme)."""
    return close.shift(skip).div(close.shift(long)) - 1.0


def realized_vol(returns: pd.DataFrame, window: int) -> pd.DataFrame:
    return returns.rolling(window, min_periods=_min_periods(window)).std() * np.sqrt(252)


def downside_vol(returns: pd.DataFrame, window: int) -> pd.DataFrame:
    downside = returns.clip(upper=0.0)
    return downside.rolling(window, min_periods=_min_periods(window)).std() * np.sqrt(252)


def rsi(close: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """RSI de Wilder, borne dans [0, 100]."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    alpha = 1.0 / period
    avg_gain = gain.ewm(alpha=alpha, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=alpha, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100.0 - 100.0 / (1.0 + rs)
    # Aucune perte sur la fenetre : RSI a 100 par convention.
    return out.where(avg_loss.ne(0.0) | avg_gain.isna(), 100.0)


def macd_histogram(close: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    ema_fast = close.ewm(span=fast, adjust=False, min_periods=fast).mean()
    ema_slow = close.ewm(span=slow, adjust=False, min_periods=slow).mean()
    macd = ema_fast - ema_slow
    hist = macd - macd.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return hist.div(close)  # normalise par le prix pour comparer les titres


def average_true_range(panel: Panel, period: int = 14) -> pd.DataFrame:
    prev_close = panel.close.shift(1)
    true_range = np.maximum(
        panel.high - panel.low,
        np.maximum((panel.high - prev_close).abs(), (panel.low - prev_close).abs()),
    )
    atr = true_range.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    return atr.div(panel.close)


def distance_to_ma(close: pd.DataFrame, window: int) -> pd.DataFrame:
    ma = close.rolling(window, min_periods=_min_periods(window)).mean()
    return close.div(ma) - 1.0


def bollinger_z(close: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    ma = close.rolling(window, min_periods=window).mean()
    sd = close.rolling(window, min_periods=window).std()
    return (close - ma).div(sd.replace(0.0, np.nan))


def drawdown(close: pd.DataFrame, window: int) -> pd.DataFrame:
    peak = close.rolling(window, min_periods=_min_periods(window)).max()
    return close.div(peak) - 1.0


def volume_ratio(volume: pd.DataFrame, short: int = 5, long: int = 63) -> pd.DataFrame:
    vol = volume.replace(0.0, np.nan)
    short_avg = vol.rolling(short, min_periods=short).mean()
    long_avg = vol.rolling(long, min_periods=_min_periods(long, floor=10)).mean()
    return short_avg.div(long_avg.replace(0.0, np.nan)) - 1.0


# --------------------------------------------------------------------------
# Normalisation en coupe transversale
# --------------------------------------------------------------------------
def cross_sectional_rank(
    df: pd.DataFrame, mask: pd.DataFrame | None = None, min_names: int = 5
) -> pd.DataFrame:
    """Rang par date, ramene dans [-1, 1]. NaN si trop peu de titres cotes."""
    values = df.where(mask) if mask is not None else df
    ranks = values.rank(axis=1, pct=True, na_option="keep")
    scaled = (ranks - 0.5) * 2.0
    counts = values.notna().sum(axis=1)
    scaled.loc[counts < min_names, :] = np.nan
    return scaled


# --------------------------------------------------------------------------
# Assemblage
# --------------------------------------------------------------------------
def compute_raw_indicators(panel: Panel, cfg: FeaturesConfig) -> dict[str, pd.DataFrame]:
    """Indicateurs bruts, avant normalisation."""
    close = panel.close
    returns = daily_returns(close)
    raw: dict[str, pd.DataFrame] = {}

    for window in cfg.momentum_windows:
        raw[f"mom_{window}"] = momentum(close, window)
    raw["mom_12_1"] = momentum_12_1(close)

    for window in cfg.vol_windows:
        raw[f"vol_{window}"] = realized_vol(returns, window)
    raw["downside_vol"] = downside_vol(returns, max(cfg.vol_windows))

    raw["rsi"] = rsi(close, cfg.rsi_period)
    raw["macd_hist"] = macd_histogram(close)
    raw["atr_pct"] = average_true_range(panel, cfg.atr_period)
    raw["dist_ma50"] = distance_to_ma(close, 50)
    raw["dist_ma200"] = distance_to_ma(close, 200)
    raw["ma_cross"] = (
        close.rolling(50, min_periods=25).mean().div(
            close.rolling(200, min_periods=100).mean()
        )
        - 1.0
    )
    raw["bollinger_z"] = bollinger_z(close, cfg.bollinger_window)
    raw["reversal"] = -momentum(close, cfg.reversal_window)
    raw["drawdown"] = drawdown(close, cfg.drawdown_window)
    raw["skew"] = returns.rolling(
        cfg.drawdown_window, min_periods=_min_periods(cfg.drawdown_window, floor=30)
    ).skew()
    raw["volume_ratio"] = volume_ratio(panel.volume)
    return raw


def valid_mask(panel: Panel, min_history: int) -> pd.DataFrame:
    """Titres cotables et disposant d'assez d'historique a chaque date."""
    history = panel.available.cumsum()
    return panel.available & (history >= min_history)


def build_features(panel: Panel, cfg: FeaturesConfig, min_history: int = 260) -> FeatureSet:
    """Construit le jeu d'indicateurs normalises pour tout le panel."""
    raw = compute_raw_indicators(panel, cfg)
    mask = valid_mask(panel, min_history)

    # Les indicateurs ou "plus grand = moins attractif" sont inverses pour que
    # le signe des features soit homogene (positif = favorable).
    inverted = {"vol", "downside_vol", "atr_pct"}
    features: dict[str, pd.DataFrame] = {}
    for name, frame in raw.items():
        base = name.split("_")[0]
        sign = -1.0 if (name in inverted or base in inverted) else 1.0
        features[name] = cross_sectional_rank(
            frame * sign, mask=mask, min_names=cfg.min_cross_section
        )

    return FeatureSet(features=features, raw=raw, valid=mask)
