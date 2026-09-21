"""Mise en forme : les tableaux et fiches doivent rester lisibles et exacts."""

from __future__ import annotations

import numpy as np
import pytest

from volatrade import report
from volatrade.metrics import compute_metrics
from volatrade.plan import build_plan
from volatrade.risk import RiskSettings, size_position
from volatrade.signals import evaluate


def test_formatage_des_valeurs_manquantes():
    assert report.pct(float("nan")) == "n/d"
    assert report.num(None) == "n/d"
    assert report.money(float("nan")) == "n/d"
    assert report.compact(float("nan")) == "n/d"


def test_formatage_des_ratios_infinis():
    assert report.num(float("inf")) == "+inf"
    assert report.num(float("-inf")) == "-inf"


def test_formatage_lisible():
    assert report.pct(0.1234) == "12.3 %"
    assert report.money(1234.5, "$") == "1 234.50 $"
    assert report.compact(1_500_000) == "1.5 M"
    assert report.compact(2_400_000_000) == "2.4 Md"


def test_tableau_aligne_les_colonnes():
    rendu = report.table(["a", "bbbb"], [["1", "2"]])
    lignes = rendu.splitlines()
    assert len(lignes) == 3
    assert all(len(ligne) == len(lignes[0]) for ligne in lignes)


def test_tableau_vide():
    assert "aucune ligne" in report.table(["a"], [])


def test_fiche_de_plan_contient_les_regles_de_vente(uptrend_frame):
    settings = RiskSettings(capital=100_000)
    metrics = compute_metrics("TEST", uptrend_frame)
    signal = evaluate("TEST", uptrend_frame, metrics)
    sizing = size_position("TEST", metrics.last_price, signal.state.atr14,
                           metrics.vol_ewma, 1.0, settings)
    plan = build_plan(signal, metrics, sizing, "theme", settings)
    fiche = report.format_plan_detail(plan)
    assert "Quand vendre" in fiche
    assert "stop initial" in fiche
    assert "Objectif 1" in fiche
    assert f"{plan.shares}" in fiche


def test_explications_des_mesures_completes():
    texte = report.explain_metrics()
    for terme in ("VaR", "CVaR", "Sharpe", "Beta", "ATR"):
        assert terme in texte
