"""Construction du portefeuille cible a partir des scores."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import PortfolioConfig
from .risk import ExposureDecision, target_exposure

EPSILON = 1e-9


@dataclass
class PortfolioTarget:
    """Portefeuille vise a une date donnee."""

    weights: pd.Series  # poids cibles, somme <= max_gross
    exposure: ExposureDecision
    detail: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def invested(self) -> float:
        return float(self.weights.sum())

    @property
    def cash_weight(self) -> float:
        return float(max(0.0, 1.0 - self.invested))

    @property
    def holdings(self) -> pd.Series:
        return self.weights[self.weights > EPSILON].sort_values(ascending=False)


def cap_weights(weights: pd.Series, cap: float, max_iter: int = 50) -> pd.Series:
    """Plafonne chaque poids a `cap` en redistribuant l'exces sur les autres.

    Si tous les titres sont plafonnes, la somme reste inferieure a 1 : le
    solde part en liquidites plutot que de forcer une concentration excessive.
    """
    weights = weights.astype("float64").copy()
    for _ in range(max_iter):
        over = weights > cap + EPSILON
        if not over.any():
            break
        excess = float((weights[over] - cap).sum())
        weights[over] = cap
        room = (weights < cap - EPSILON) & (weights > EPSILON)
        if not room.any() or excess <= EPSILON:
            break
        weights[room] += excess * weights[room] / float(weights[room].sum())
    return weights.clip(upper=cap)


def _apply_min_weight(weights: pd.Series, cfg: PortfolioConfig) -> pd.Series:
    """Supprime les lignes trop petites pour valoir la peine d'etre detenues."""
    for _ in range(10):
        total = float(weights.sum())
        if total <= EPSILON:
            return weights * 0.0
        too_small = (weights > EPSILON) & (weights < cfg.min_weight - EPSILON)
        if not too_small.any():
            break
        weights = weights[~too_small].reindex(weights.index).fillna(0.0)
        new_total = float(weights.sum())
        if new_total <= EPSILON:
            return weights * 0.0
        weights = weights / new_total * min(total, 1.0)
        weights = cap_weights(weights, cfg.max_weight)
    return weights


def select_and_weight(
    scores: pd.Series,
    vols: pd.Series,
    cfg: PortfolioConfig,
    holdings: set[str] | None = None,
) -> tuple[pd.Series, pd.DataFrame]:
    """Selectionne les titres et calcule leurs poids relatifs (somme <= 1).

    Deux ingredients : le score (conviction) et la volatilite (risque). Une
    conviction identique sur un titre deux fois plus volatil donne une position
    deux fois plus petite, pour que chaque ligne contribue au risque de facon
    comparable.

    La selection est hysteretique : un titre deja detenu reste eligible tant
    que son score depasse `exit_threshold`, et beneficie d'un bonus de rang.
    Sans cela le portefeuille se reconstruit entierement a chaque seance et les
    frais absorbent le signal.
    """
    universe = scores.index
    empty = pd.Series(0.0, index=universe, dtype="float64")
    holdings = holdings or set()
    columns = ["score", "vol", "raw_weight", "weight"]

    available = scores.dropna()
    if available.empty:
        return empty, pd.DataFrame(columns=columns)

    held = available.index.isin(holdings)
    eligible = np.where(
        held,
        available > cfg.exit_threshold + EPSILON,
        available > cfg.score_threshold + EPSILON,
    )
    candidates = available[eligible]
    if candidates.empty:
        return empty, pd.DataFrame(columns=columns)

    priority = candidates + pd.Series(
        np.where(candidates.index.isin(holdings), cfg.hold_bonus, 0.0),
        index=candidates.index,
    )
    keep = priority.sort_values(ascending=False).head(cfg.max_positions).index
    candidates = candidates.reindex(keep).sort_values(ascending=False)

    # Volatilite manquante : on prend la mediane de l'univers plutot que
    # d'exclure le titre (les jeunes cotations n'ont pas 3 mois d'historique).
    default_vol = float(vols.replace(0.0, np.nan).median())
    if not np.isfinite(default_vol) or default_vol <= 0:
        default_vol = 0.25
    selected_vols = vols.reindex(candidates.index).replace(0.0, np.nan).fillna(default_vol)
    selected_vols = selected_vols.clip(lower=0.05, upper=1.5)

    # La conviction se mesure a partir du seuil de sortie : tous les titres
    # retenus sont au-dessus, la force est donc strictement positive.
    strength = (candidates - cfg.exit_threshold).clip(lower=EPSILON)
    raw = (strength / selected_vols).clip(lower=0.0)
    if float(raw.sum()) <= EPSILON:
        return empty, pd.DataFrame(columns=columns)

    weights = raw / float(raw.sum())
    weights = cap_weights(weights, cfg.max_weight)
    weights = _apply_min_weight(weights, cfg)

    detail = pd.DataFrame(
        {
            "score": candidates,
            "vol": selected_vols,
            "raw_weight": raw / float(raw.sum()),
            "weight": weights.reindex(candidates.index).fillna(0.0),
        }
    )
    detail = detail[detail["weight"] > EPSILON].sort_values("weight", ascending=False)
    return weights.reindex(universe).fillna(0.0), detail


def build_target(
    scores: pd.Series,
    vols: pd.Series,
    returns_window: pd.DataFrame,
    cfg: PortfolioConfig,
    market: pd.Series | None = None,
    as_of: pd.Timestamp | None = None,
    holdings: set[str] | None = None,
) -> PortfolioTarget:
    """Portefeuille cible complet : selection, ponderation puis exposition."""
    relative, detail = select_and_weight(scores, vols, cfg, holdings=holdings)
    decision = target_exposure(relative, returns_window, cfg, market=market, as_of=as_of)

    investable = max(0.0, 1.0 - cfg.cash_buffer)
    weights = relative * decision.exposure * investable
    if not detail.empty:
        detail = detail.assign(weight=weights.reindex(detail.index).fillna(0.0))
        detail = detail[detail["weight"] > EPSILON]
    return PortfolioTarget(weights=weights, exposure=decision, detail=detail)


def current_weights(positions: dict[str, float], prices: pd.Series, cash: float) -> pd.Series:
    """Poids actuels du portefeuille, liquidites comprises dans le total."""
    tickers = list(prices.index)
    shares = pd.Series({t: float(positions.get(t, 0.0)) for t in tickers}, dtype="float64")
    values = shares * prices.reindex(tickers).astype("float64")
    equity = float(values.sum()) + float(cash)
    if equity <= 0:
        return pd.Series(0.0, index=tickers, dtype="float64")
    return values / equity


def apply_no_trade_band(
    target: pd.Series, current: pd.Series, band: float
) -> pd.Series:
    """Ignore les micro-ajustements de poids.

    Sans cette bande, le portefeuille genere des ordres tous les jours pour
    quelques dizaines d'euros : les frais mangent alors le signal. Les entrees
    et les sorties completes ne sont jamais filtrees.
    """
    if band <= 0:
        return target
    aligned_current = current.reindex(target.index).fillna(0.0)
    drift = (target - aligned_current).abs()
    entering = (aligned_current <= EPSILON) & (target > EPSILON)
    exiting = (aligned_current > EPSILON) & (target <= EPSILON)
    keep_current = (drift < band) & ~entering & ~exiting
    return target.where(~keep_current, aligned_current)
