"""Construction de la cible d'apprentissage (ce que le modele doit prevoir)."""

from __future__ import annotations

import pandas as pd

from .features import cross_sectional_rank


def forward_return(close: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Rendement des `horizon` prochaines seances, connu seulement a t+horizon."""
    return close.shift(-horizon).div(close) - 1.0


def excess_forward_return(
    close: pd.DataFrame, horizon: int, mask: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Rendement futur en exces de la moyenne de l'univers a la meme date.

    On retire la composante marche : le modele apprend le classement relatif,
    pas la direction generale des indices.
    """
    fwd = forward_return(close, horizon)
    if mask is not None:
        fwd = fwd.where(mask)
    return fwd.sub(fwd.mean(axis=1), axis=0)


def build_labels(
    close: pd.DataFrame,
    horizon: int,
    kind: str = "rank",
    mask: pd.DataFrame | None = None,
    min_names: int = 5,
) -> pd.DataFrame:
    """Cible d'apprentissage.

    - "rank"   : rang cross-sectionnel du rendement futur, dans [-1, 1].
                 Robuste aux valeurs extremes, recommande par defaut.
    - "excess" : rendement futur en exces de la moyenne de l'univers.
    """
    excess = excess_forward_return(close, horizon, mask=mask)
    if kind == "excess":
        return excess
    if kind == "rank":
        return cross_sectional_rank(excess, min_names=min_names)
    raise ValueError(f"Type de label inconnu : {kind!r} (rank | excess)")


def label_available_until(dates: pd.DatetimeIndex, as_of: pd.Timestamp, horizon: int) -> pd.Timestamp:
    """Derniere date dont le label est deja observable a la date `as_of`.

    Sert a purger l'echantillon d'apprentissage : une observation datee de u a
    besoin des prix jusqu'a u + horizon, donc elle n'est utilisable qu'a partir
    de cette date-la.
    """
    dates = pd.DatetimeIndex(dates)
    position = dates.searchsorted(pd.Timestamp(as_of), side="right") - 1
    cutoff = position - horizon
    if cutoff < 0:
        return pd.Timestamp.min
    return dates[cutoff]
