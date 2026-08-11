"""Score a base de regles : un multi-facteur classique, sans apprentissage.

Il sert a deux choses :
 - fournir un score exploitable des le premier jour, avant que le modele ait
   assez d'historique pour etre entraine ;
 - jouer le role d'a priori dans le melange final, ce qui stabilise nettement
   les recommandations quand le modele est incertain.

Ponderation : momentum long terme (le facteur le mieux documente), tendance,
faible volatilite, et un peu de reversion a court terme.
"""

from __future__ import annotations

import pandas as pd

from .features import FeatureSet, cross_sectional_rank

# Les poids portent sur des indicateurs deja normalises et orientes
# "positif = favorable" (la volatilite a donc deja ete inversee).
DEFAULT_WEIGHTS: dict[str, float] = {
    "mom_12_1": 0.30,
    "mom_63": 0.20,
    "dist_ma200": 0.15,
    "ma_cross": 0.10,
    "vol_63": 0.15,
    "reversal": 0.10,
}


def rules_score(
    featureset: FeatureSet,
    weights: dict[str, float] | None = None,
    min_names: int = 5,
) -> pd.DataFrame:
    """Score multi-facteur, renormalise en coupe transversale dans [-1, 1]."""
    weights = weights or DEFAULT_WEIGHTS
    available = {k: v for k, v in weights.items() if k in featureset.features}
    if not available:
        raise ValueError(
            "Aucun indicateur du score a base de regles n'est disponible. "
            f"Attendus : {sorted(weights)}"
        )

    total = sum(available.values())
    combined: pd.DataFrame | None = None
    for name, weight in available.items():
        # Un indicateur manquant pour un titre vaut 0 (neutre) plutot que NaN,
        # sinon un seul trou eliminerait le titre du classement.
        contribution = featureset.features[name].fillna(0.0) * (weight / total)
        combined = contribution if combined is None else combined + contribution

    assert combined is not None
    if featureset.valid is not None:
        combined = combined.where(featureset.valid)
    return cross_sectional_rank(combined, min_names=min_names)
