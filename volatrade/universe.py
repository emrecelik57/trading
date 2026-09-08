"""Selection de l'univers : les actions les plus volatiles d'un vivier liquide.

La liste `CANDIDATE_POOL` sert de vivier de depart (grandes capitalisations
americaines reputees nerveuses, valeurs de croissance, crypto-proxies,
nucleaire / spatial / quantique). Le classement final n'est PAS code en dur :
il est recalcule a chaque execution a partir de la volatilite realisee.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import Quote
from .metrics import (
    RiskMetrics,
    annualized_volatility,
    average_dollar_volume,
    compute_metrics,
    log_returns,
)

#: Vivier de candidats, par thematique. Modifiable via la configuration.
CANDIDATE_POOL: dict[str, list[str]] = {
    "tech_megacap_nerveuse": ["TSLA", "NVDA", "AMD", "MU", "AVGO", "ARM", "SMCI", "ON"],
    "logiciel_croissance": ["PLTR", "NET", "SNOW", "CRWD", "DDOG", "APP", "U", "RBLX", "SHOP"],
    "crypto_proxy": ["COIN", "MSTR", "MARA", "RIOT", "HOOD"],
    "fintech": ["SOFI", "AFRM", "UPST", "DKNG", "HIMS"],
    "vehicules_electriques": ["RIVN", "LCID", "NIO", "QS", "CHPT"],
    "energie_nouvelle": ["ENPH", "FSLR", "OKLO", "SMR", "VST"],
    "spatial_defense": ["RKLB", "ASTS", "LUNR", "JOBY", "ACHR"],
    "quantique_ia": ["IONQ", "RGTI", "SOUN", "BBAI", "TEM"],
    "consommation_volatile": ["CVNA", "GME", "SNAP", "PINS", "ROKU", "W"],
}

#: Indice de reference utilise pour le beta et la correlation.
DEFAULT_BENCHMARK = "SPY"


def default_pool() -> list[str]:
    """Vivier complet, dedoublonne, dans un ordre stable."""
    seen: dict[str, None] = {}
    for tickers in CANDIDATE_POOL.values():
        for ticker in tickers:
            seen.setdefault(ticker.upper(), None)
    return list(seen)


def theme_of(ticker: str) -> str:
    """Thematique d'un ticker du vivier (sert a limiter la concentration)."""
    ticker = ticker.upper()
    for theme, tickers in CANDIDATE_POOL.items():
        if ticker in {t.upper() for t in tickers}:
            return theme
    return "autre"


@dataclass(frozen=True)
class ScreenResult:
    """Resultat du filtrage d'un titre candidat."""

    ticker: str
    metrics: RiskMetrics
    theme: str
    volatility_score: float
    retained: bool
    reason: str


def _blended_volatility(returns: pd.Series) -> float:
    """Volatilite de selection : melange court terme reactif / moyen terme stable.

    On evite de ne regarder que 3 mois (un seul choc suffirait a classer un
    titre premier) et de ne regarder qu'un an (trop lent a detecter un reveil).
    """
    short = annualized_volatility(returns, window=63)
    medium = annualized_volatility(returns, window=126)
    parts = [value for value in (short, medium) if np.isfinite(value)]
    if not parts:
        return float("nan")
    if len(parts) == 1:
        return float(parts[0])
    return float(0.6 * short + 0.4 * medium)


def screen(
    quotes: dict[str, Quote],
    benchmark_returns: pd.Series | None = None,
    *,
    top_n: int = 10,
    min_dollar_volume: float = 30e6,
    min_price: float = 3.0,
    min_observations: int = 200,
    max_volatility: float = 2.5,
    risk_free: float = 0.0,
) -> list[ScreenResult]:
    """Classe les candidats par volatilite et retient les `top_n` eligibles.

    Les filtres de liquidite, de prix et d'historique ecartent les titres sur
    lesquels un plan de trade ne serait pas executable ; `max_volatility`
    ecarte les cas extremes (> 250 % annualise) ou le risque n'est plus
    pilotable par un stop.
    """
    results: list[ScreenResult] = []
    for ticker, quote in quotes.items():
        frame = quote.frame
        returns = log_returns(frame["close"])
        score = _blended_volatility(returns)
        metrics = compute_metrics(ticker, frame, benchmark_returns, risk_free)
        liquidity = average_dollar_volume(frame)

        reason = ""
        if len(frame) < min_observations:
            reason = f"historique trop court ({len(frame)} seances)"
        elif not np.isfinite(score):
            reason = "volatilite non calculable"
        elif float(frame['close'].iloc[-1]) < min_price:
            reason = f"cours trop bas ({frame['close'].iloc[-1]:.2f})"
        elif not np.isfinite(liquidity) or liquidity < min_dollar_volume:
            reason = f"liquidite insuffisante ({liquidity / 1e6:.1f} M/j)"
        elif score > max_volatility:
            reason = f"volatilite ingerable ({score:.0%})"

        results.append(
            ScreenResult(
                ticker=ticker.upper(),
                metrics=metrics,
                theme=theme_of(ticker),
                volatility_score=score,
                retained=False,
                reason=reason,
            )
        )

    eligible = [item for item in results if not item.reason]
    eligible.sort(key=lambda item: item.volatility_score, reverse=True)
    kept = {item.ticker for item in eligible[:top_n]}

    final = [
        ScreenResult(
            ticker=item.ticker,
            metrics=item.metrics,
            theme=item.theme,
            volatility_score=item.volatility_score,
            retained=item.ticker in kept,
            reason=item.reason or ("retenu" if item.ticker in kept else "hors du top"),
        )
        for item in results
    ]
    final.sort(
        key=lambda item: (
            not item.retained,
            -item.volatility_score if np.isfinite(item.volatility_score) else 0.0,
        )
    )
    return final


def retained(results: list[ScreenResult]) -> list[ScreenResult]:
    """Sous-ensemble retenu, du plus volatil au moins volatil."""
    return [item for item in results if item.retained]
