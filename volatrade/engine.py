"""Chaine complete : donnees -> selection -> scores -> tailles -> plans."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .config import AppConfig
from .data import Quote, close_panel, fetch_history, fetch_many
from .metrics import log_returns
from .plan import TradePlan, build_plan, is_actionable, summarize_basket
from .risk import allocate, portfolio_risk
from .signals import Signal, evaluate
from .universe import ScreenResult, retained, screen


@dataclass
class MarketView:
    """Resultat complet d'une analyse de marche."""

    config: AppConfig
    quotes: dict[str, Quote]
    screen: list[ScreenResult]
    signals: dict[str, Signal]
    plans: list[TradePlan]
    returns: pd.DataFrame
    correlations: pd.DataFrame
    basket: dict[str, float] = field(default_factory=dict)
    risk: dict[str, float] = field(default_factory=dict)
    errors: list[tuple[str, str]] = field(default_factory=list)

    @property
    def selection(self) -> list[ScreenResult]:
        return retained(self.screen)

    @property
    def actionable(self) -> list[TradePlan]:
        return [plan for plan in self.plans if is_actionable(plan)]

    def as_dict(self) -> dict:
        return {
            "configuration": self.config.as_dict(),
            "selection": [
                {
                    "ticker": item.ticker,
                    "theme": item.theme,
                    "volatilite_selection": item.volatility_score,
                    "metriques": item.metrics.as_dict(),
                }
                for item in self.selection
            ],
            "plans": [plan.as_dict() for plan in self.plans],
            "panier": self.basket,
            "risque_portefeuille": self.risk,
            "correlations": self.correlations.round(3).to_dict() if not self.correlations.empty else {},
            "erreurs": [{"ticker": ticker, "message": message} for ticker, message in self.errors],
        }


def _benchmark_returns(config: AppConfig, errors: list) -> pd.Series | None:
    try:
        quote = fetch_history(
            config.indice_reference,
            config.historique,
            cache_dir=config.dossier_cache,
            cache_ttl=config.cache_ttl_heures * 3600,
        )
    except Exception as exc:  # noqa: BLE001 - l'indice est optionnel
        errors.append((config.indice_reference, str(exc)))
        return None
    return log_returns(quote.close)


def load_market(config: AppConfig) -> tuple[dict[str, Quote], pd.Series | None, list]:
    """Telecharge l'univers et l'indice de reference."""
    errors: list[tuple[str, str]] = []
    benchmark = _benchmark_returns(config, errors)
    quotes = fetch_many(
        config.univers,
        config.historique,
        cache_dir=config.dossier_cache,
        cache_ttl=config.cache_ttl_heures * 3600,
        on_error=lambda ticker, exc: errors.append((ticker, str(exc))),
    )
    return quotes, benchmark, errors


def analyse(config: AppConfig, quotes: dict[str, Quote], benchmark: pd.Series | None,
            errors: list | None = None) -> MarketView:
    """Applique la chaine d'analyse a un jeu de cours deja telecharge."""
    errors = list(errors or [])
    results = screen(
        quotes,
        benchmark,
        top_n=config.nombre_titres,
        min_dollar_volume=config.volume_min,
        min_price=config.prix_min,
        max_volatility=config.volatilite_max,
        risk_free=config.taux_sans_risque,
    )
    selection = retained(results)

    signals: dict[str, Signal] = {}
    candidates: list[dict] = []
    for item in selection:
        quote = quotes[item.ticker]
        signal = evaluate(item.ticker, quote.frame, item.metrics)
        signals[item.ticker] = signal
        candidates.append(
            {
                "ticker": item.ticker,
                "price": item.metrics.last_price,
                "atr": signal.state.atr14,
                "volatility": item.metrics.vol_ewma,
                "conviction": signal.conviction,
                "score": signal.score,
                "theme": item.theme,
            }
        )

    panel = close_panel({item.ticker: quotes[item.ticker] for item in selection})
    returns = panel.pct_change().dropna(how="all") if not panel.empty else pd.DataFrame()
    correlations = returns.corr() if not returns.empty else pd.DataFrame()

    settings = config.risk_settings()
    sizings = allocate(candidates, settings, correlations if not correlations.empty else None)
    by_ticker = {sizing.ticker: sizing for sizing in sizings}

    plans = [
        build_plan(signals[item.ticker], item.metrics, by_ticker[item.ticker], item.theme, settings)
        for item in selection
        if item.ticker in by_ticker
    ]
    plans.sort(key=lambda plan: (not is_actionable(plan), -plan.score))

    return MarketView(
        config=config,
        quotes=quotes,
        screen=results,
        signals=signals,
        plans=plans,
        returns=returns,
        correlations=correlations,
        basket=summarize_basket(plans),
        risk=portfolio_risk(sizings, returns, settings),
        errors=errors,
    )


def run(config: AppConfig) -> MarketView:
    """Telecharge puis analyse : point d'entree unique de l'outil."""
    quotes, benchmark, errors = load_market(config)
    return analyse(config, quotes, benchmark, errors)
