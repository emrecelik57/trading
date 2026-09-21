"""Selection de l'univers : filtres de liquidite, de prix et de volatilite."""

from __future__ import annotations

import numpy as np
import pytest

from volatrade.universe import CANDIDATE_POOL, default_pool, retained, screen, theme_of
from tests.conftest import make_quote


def quote_volatile(ticker: str, vol: float, seed: int = 0, price: float = 50.0, volume: float = 5e6):
    rng = np.random.default_rng(seed)
    prices = price * np.exp(np.cumsum(vol / np.sqrt(252) * rng.standard_normal(300)))
    return make_quote(ticker, prices, volume=volume)


def test_vivier_par_defaut_sans_doublon():
    pool = default_pool()
    assert len(pool) == len(set(pool))
    assert all(ticker.isupper() for ticker in pool)
    assert len(pool) > 30


def test_theme_retrouve_pour_chaque_ticker_du_vivier():
    for theme, tickers in CANDIDATE_POOL.items():
        for ticker in tickers:
            assert theme_of(ticker) == theme
    assert theme_of("INCONNU") == "autre"


def test_les_plus_volatils_sont_retenus():
    quotes = {
        "CALME": quote_volatile("CALME", 0.15, seed=1),
        "MOYEN": quote_volatile("MOYEN", 0.50, seed=2),
        "FOU": quote_volatile("FOU", 1.20, seed=3),
    }
    results = screen(quotes, top_n=2, min_dollar_volume=0)
    noms = [item.ticker for item in retained(results)]
    assert noms == ["FOU", "MOYEN"]
    assert results[0].volatility_score > results[-1].volatility_score


def test_filtre_de_liquidite():
    quotes = {
        "LIQUIDE": quote_volatile("LIQUIDE", 0.80, seed=4, volume=5e6),
        "ILLIQUIDE": quote_volatile("ILLIQUIDE", 1.50, seed=5, volume=100),
    }
    results = screen(quotes, top_n=5, min_dollar_volume=1e6)
    noms = [item.ticker for item in retained(results)]
    assert noms == ["LIQUIDE"]
    ecarte = next(item for item in results if item.ticker == "ILLIQUIDE")
    assert "liquidite" in ecarte.reason


def test_filtre_de_prix_plancher():
    quotes = {"PENNY": quote_volatile("PENNY", 1.0, seed=6, price=0.5)}
    ecarte = screen(quotes, min_price=3.0, min_dollar_volume=0)[0]
    assert not ecarte.retained
    assert "cours trop bas" in ecarte.reason


def test_filtre_historique_trop_court():
    quotes = {"NEUF": make_quote("NEUF", np.linspace(10, 20, 50))}
    ecarte = screen(quotes, min_dollar_volume=0)[0]
    assert "historique trop court" in ecarte.reason


def test_volatilite_ingerable_ecartee():
    quotes = {"EXTREME": quote_volatile("EXTREME", 4.0, seed=7)}
    ecarte = screen(quotes, min_dollar_volume=0, min_price=0.0, max_volatility=2.5)[0]
    assert not ecarte.retained
    assert "ingerable" in ecarte.reason


def test_metriques_calculees_meme_pour_les_titres_ecartes():
    quotes = {"PENNY": quote_volatile("PENNY", 1.0, seed=8, price=0.5)}
    item = screen(quotes, min_price=3.0, min_dollar_volume=0)[0]
    assert item.metrics.observations == 300
    assert item.metrics.vol_ann_1y > 0


def test_top_n_respecte():
    quotes = {f"T{i}": quote_volatile(f"T{i}", 0.4 + i * 0.1, seed=i) for i in range(8)}
    assert len(retained(screen(quotes, top_n=3, min_dollar_volume=0))) == 3
