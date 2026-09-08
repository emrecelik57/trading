"""Dimensionnement des positions : combien acheter sans exploser le compte.

Trois contraintes s'appliquent a chaque ligne, la plus severe l'emporte :

1. risque par trade  : la perte si le stop est touche ne depasse pas X % du capital ;
2. cible de volatilite : le poids est inversement proportionnel a la volatilite du titre ;
3. poids maximum : aucune ligne ne depasse Y % du portefeuille.

Puis deux contraintes de portefeuille : la chaleur totale (somme des risques
ouverts) et l'exposition totale, avec une decote pour les titres redondants.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RiskSettings:
    """Parametres de gestion du risque."""

    capital: float = 10_000.0
    risk_per_trade: float = 0.01          # 1 % du capital perdu si le stop saute
    atr_stop_multiple: float = 2.5        # distance du stop en ATR(14)
    max_weight: float = 0.15              # poids max d'une ligne
    target_volatility: float = 0.45       # volatilite annuelle visee par ligne
    max_portfolio_heat: float = 0.06      # somme des risques ouverts (6 % du capital)
    max_exposure: float = 1.00            # part du capital investie au maximum
    max_positions_per_theme: int = 2
    correlation_haircut: float = 0.40     # decote max pour redondance


@dataclass(frozen=True)
class Sizing:
    """Taille de position calculee pour un titre."""

    ticker: str
    shares: int
    price: float
    stop_price: float
    stop_distance: float
    notional: float
    weight: float
    risk_amount: float
    risk_pct: float
    binding_constraint: str
    correlation_penalty: float


def stop_distance(atr_value: float, price: float, settings: RiskSettings) -> float:
    """Distance du stop en devise, plancher a 3 % du cours.

    Un plancher evite un stop absurdement serre quand l'ATR se contracte : sur
    un titre volatil, un stop a 1 % serait touche par le bruit intraday.
    """
    if not np.isfinite(atr_value) or atr_value <= 0:
        return float(0.10 * price)
    return float(max(settings.atr_stop_multiple * atr_value, 0.03 * price))


def _correlation_penalty(
    ticker: str, peers: list[str], correlations: pd.DataFrame | None, settings: RiskSettings
) -> float:
    """Coefficient <= 1 appliquant une decote aux titres trop correles au reste."""
    if correlations is None or ticker not in correlations.index:
        return 1.0
    others = [peer for peer in peers if peer != ticker and peer in correlations.columns]
    if not others:
        return 1.0
    average = float(correlations.loc[ticker, others].mean())
    if not np.isfinite(average) or average <= 0.5:
        return 1.0
    excess = min((average - 0.5) / 0.5, 1.0)
    return float(1.0 - settings.correlation_haircut * excess)


def size_position(
    ticker: str,
    price: float,
    atr_value: float,
    volatility: float,
    conviction: float,
    settings: RiskSettings,
    correlation_penalty: float = 1.0,
) -> Sizing:
    """Applique les trois contraintes et retient la plus contraignante."""
    distance = stop_distance(atr_value, price, settings)
    conviction = float(np.clip(conviction, 0.0, 1.0))
    if price <= 0 or conviction == 0.0:
        return Sizing(ticker, 0, price, price - distance, distance, 0.0, 0.0, 0.0, 0.0, "aucune", correlation_penalty)

    budget = settings.capital * settings.risk_per_trade * conviction * correlation_penalty
    shares_risk = budget / distance

    if np.isfinite(volatility) and volatility > 0:
        vol_weight = min(settings.target_volatility / volatility, settings.max_weight)
    else:
        vol_weight = settings.max_weight
    shares_vol = settings.capital * vol_weight * conviction * correlation_penalty / price
    shares_cap = settings.capital * settings.max_weight * conviction / price

    candidates = {
        "risque par trade": shares_risk,
        "cible de volatilite": shares_vol,
        "poids maximum": shares_cap,
    }
    binding = min(candidates, key=candidates.get)
    shares = int(math.floor(max(0.0, candidates[binding])))

    notional = shares * price
    risk_amount = shares * distance
    return Sizing(
        ticker=ticker.upper(),
        shares=shares,
        price=float(price),
        stop_price=float(price - distance),
        stop_distance=float(distance),
        notional=float(notional),
        weight=float(notional / settings.capital) if settings.capital else 0.0,
        risk_amount=float(risk_amount),
        risk_pct=float(risk_amount / settings.capital) if settings.capital else 0.0,
        binding_constraint=binding,
        correlation_penalty=float(correlation_penalty),
    )


def allocate(
    candidates: list[dict],
    settings: RiskSettings,
    correlations: pd.DataFrame | None = None,
) -> list[Sizing]:
    """Dimensionne un panier complet en respectant les limites de portefeuille.

    Chaque candidat est un dict : ticker, price, atr, volatility, conviction,
    theme, score. Les candidats sont traites par score decroissant ; les
    limites de theme puis les limites globales (chaleur, exposition) reduisent
    ou ecartent les lignes surnumeraires.
    """
    ordered = sorted(candidates, key=lambda item: item.get("score", 0.0), reverse=True)
    tickers = [item["ticker"] for item in ordered if item.get("conviction", 0.0) > 0]

    sizings: list[Sizing] = []
    theme_count: dict[str, int] = {}
    for item in ordered:
        conviction = float(item.get("conviction", 0.0))
        theme = item.get("theme", "autre")
        capped_by_theme = False
        if conviction > 0:
            used = theme_count.get(theme, 0)
            if used >= settings.max_positions_per_theme:
                conviction = 0.0
                capped_by_theme = True
            else:
                theme_count[theme] = used + 1

        penalty = _correlation_penalty(item["ticker"], tickers, correlations, settings)
        sizing = size_position(
            ticker=item["ticker"],
            price=float(item["price"]),
            atr_value=float(item.get("atr", float("nan"))),
            volatility=float(item.get("volatility", float("nan"))),
            conviction=conviction,
            settings=settings,
            correlation_penalty=penalty,
        )
        if capped_by_theme:
            sizing = replace(
                sizing,
                binding_constraint=(
                    f"limite de theme ({settings.max_positions_per_theme} lignes maximum "
                    f"sur {theme})"
                ),
            )
        sizings.append(sizing)

    return _apply_portfolio_limits(sizings, settings)


def _apply_portfolio_limits(sizings: list[Sizing], settings: RiskSettings) -> list[Sizing]:
    """Reduit proportionnellement les lignes si la chaleur ou l'exposition depasse."""
    total_risk = sum(item.risk_pct for item in sizings)
    total_weight = sum(item.weight for item in sizings)

    scale = 1.0
    if total_risk > settings.max_portfolio_heat > 0:
        scale = min(scale, settings.max_portfolio_heat / total_risk)
    if total_weight > settings.max_exposure > 0:
        scale = min(scale, settings.max_exposure / total_weight)
    if scale >= 1.0:
        return sizings

    scaled: list[Sizing] = []
    for item in sizings:
        shares = int(math.floor(item.shares * scale))
        notional = shares * item.price
        risk_amount = shares * item.stop_distance
        scaled.append(
            Sizing(
                ticker=item.ticker,
                shares=shares,
                price=item.price,
                stop_price=item.stop_price,
                stop_distance=item.stop_distance,
                notional=notional,
                weight=notional / settings.capital if settings.capital else 0.0,
                risk_amount=risk_amount,
                risk_pct=risk_amount / settings.capital if settings.capital else 0.0,
                binding_constraint=(
                    item.binding_constraint if item.shares == shares else "limite de portefeuille"
                ),
                correlation_penalty=item.correlation_penalty,
            )
        )
    return scaled


def portfolio_risk(
    sizings: list[Sizing], returns: pd.DataFrame, settings: RiskSettings
) -> dict[str, float]:
    """Risque agrege du panier : volatilite, VaR et perte si tous les stops sautent."""
    weights = {item.ticker: item.weight for item in sizings if item.shares > 0}
    summary = {
        "exposition": float(sum(weights.values())),
        "chaleur": float(sum(item.risk_pct for item in sizings)),
        "perte_si_tous_stops": float(sum(item.risk_amount for item in sizings)),
        "nombre_lignes": float(sum(1 for item in sizings if item.shares > 0)),
    }
    columns = [ticker for ticker in weights if ticker in returns.columns]
    if not columns:
        summary["volatilite_portefeuille"] = float("nan")
        summary["var_95_1j"] = float("nan")
        return summary

    aligned = returns[columns].dropna()
    if len(aligned) < 20:
        summary["volatilite_portefeuille"] = float("nan")
        summary["var_95_1j"] = float("nan")
        return summary

    vector = np.array([weights[ticker] for ticker in columns])
    covariance = aligned.cov().to_numpy() * 252
    variance = float(vector @ covariance @ vector)
    volatility = float(np.sqrt(max(variance, 0.0)))
    summary["volatilite_portefeuille"] = volatility
    # VaR parametrique a 95 % sur un jour (quantile normal 1.645).
    summary["var_95_1j"] = float(1.645 * volatility / np.sqrt(252) * settings.capital)
    return summary
