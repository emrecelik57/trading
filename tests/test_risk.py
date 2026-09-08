"""Dimensionnement des positions et limites de portefeuille."""

from __future__ import annotations

import pandas as pd
import pytest

from volatrade.risk import RiskSettings, allocate, portfolio_risk, size_position, stop_distance


SETTINGS = RiskSettings(capital=100_000, risk_per_trade=0.01, atr_stop_multiple=2.5,
                        max_weight=0.15, target_volatility=0.45, max_portfolio_heat=0.06)


def test_distance_de_stop_suit_latr():
    assert stop_distance(4.0, 100.0, SETTINGS) == pytest.approx(10.0)


def test_distance_de_stop_a_un_plancher():
    # ATR minuscule : le stop reste a 3 % du cours pour ne pas etre du bruit.
    assert stop_distance(0.1, 100.0, SETTINGS) == pytest.approx(3.0)


def test_distance_de_stop_sans_atr_disponible():
    assert stop_distance(float("nan"), 50.0, SETTINGS) == pytest.approx(5.0)


def test_taille_limitee_par_le_risque_par_trade():
    # Stop a 10 $ sous un cours de 100 $, 1 % de 100 000 $ = 1 000 $ de risque.
    sizing = size_position("A", 100.0, 4.0, 0.30, 1.0, SETTINGS)
    assert sizing.shares == 100
    assert sizing.risk_amount == pytest.approx(1000.0)
    assert sizing.risk_pct == pytest.approx(0.01)
    assert sizing.binding_constraint == "risque par trade"


def test_taille_limitee_par_la_cible_de_volatilite():
    # Volatilite de 150 % : la cible de 45 % impose un poids de 30 %, plafonne
    # a 15 % par max_weight ; le stop tres large ne mord pas.
    sizing = size_position("B", 100.0, 20.0, 1.50, 1.0, SETTINGS)
    assert sizing.weight <= SETTINGS.max_weight + 1e-9
    assert sizing.binding_constraint in ("cible de volatilite", "risque par trade")


def test_poids_maximum_respecte_meme_avec_un_stop_serre():
    sizing = size_position("C", 10.0, 0.05, 0.20, 1.0, SETTINGS)
    assert sizing.weight <= SETTINGS.max_weight + 1e-9


def test_conviction_partielle_divise_la_taille():
    pleine = size_position("D", 100.0, 4.0, 0.30, 1.0, SETTINGS)
    demie = size_position("D", 100.0, 4.0, 0.30, 0.5, SETTINGS)
    assert demie.shares == pytest.approx(pleine.shares / 2, abs=1)


def test_conviction_nulle_ne_genere_aucun_ordre():
    sizing = size_position("E", 100.0, 4.0, 0.30, 0.0, SETTINGS)
    assert sizing.shares == 0 and sizing.notional == 0


def candidats(n: int, theme: str = "t", price: float = 100.0) -> list[dict]:
    return [
        {
            "ticker": f"T{i}",
            "price": price,
            "atr": 4.0,
            "volatility": 0.60,
            "conviction": 1.0,
            "score": 90 - i,
            "theme": f"{theme}{i}",
        }
        for i in range(n)
    ]


def test_chaleur_totale_plafonnee():
    settings = RiskSettings(capital=100_000, risk_per_trade=0.01, max_portfolio_heat=0.03)
    sizings = allocate(candidats(10), settings)
    assert sum(item.risk_pct for item in sizings) <= 0.03 + 1e-6


def test_exposition_totale_plafonnee():
    settings = RiskSettings(capital=100_000, max_exposure=0.30, max_portfolio_heat=1.0,
                            max_weight=0.20, risk_per_trade=0.05)
    sizings = allocate(candidats(10), settings)
    assert sum(item.weight for item in sizings) <= 0.30 + 1e-6


def test_limite_de_lignes_par_theme():
    settings = RiskSettings(capital=100_000, max_positions_per_theme=2)
    candidates = candidats(5, theme="")  # tous sur le theme ""
    for item in candidates:
        item["theme"] = "meme_theme"
    sizings = allocate(candidates, settings)
    retenus = [item for item in sizings if item.shares > 0]
    assert len(retenus) == 2
    ecartes = [item for item in sizings if item.shares == 0]
    assert all("limite de theme" in item.binding_constraint for item in ecartes)


def test_decote_pour_titres_trop_correles():
    index = ["T0", "T1"]
    correles = pd.DataFrame([[1.0, 0.95], [0.95, 1.0]], index=index, columns=index)
    independants = pd.DataFrame([[1.0, 0.0], [0.0, 1.0]], index=index, columns=index)
    settings = RiskSettings(capital=100_000, max_positions_per_theme=5)
    avec = allocate(candidats(2), settings, correles)
    sans = allocate(candidats(2), settings, independants)
    assert avec[0].shares < sans[0].shares
    assert avec[0].correlation_penalty < 1.0


def test_risque_de_portefeuille_agrege():
    settings = RiskSettings(capital=100_000)
    sizings = allocate(candidats(3), settings)
    index = pd.bdate_range("2024-01-01", periods=300, tz="UTC")
    returns = pd.DataFrame(
        {item.ticker: [0.01, -0.01] * 150 for item in sizings}, index=index
    )
    summary = portfolio_risk(sizings, returns, settings)
    assert summary["nombre_lignes"] == 3
    assert summary["volatilite_portefeuille"] > 0
    assert summary["var_95_1j"] > 0
    assert summary["perte_si_tous_stops"] == pytest.approx(
        sum(item.risk_amount for item in sizings)
    )
