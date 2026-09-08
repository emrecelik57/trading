"""Score d'achat : tendance, momentum, qualite d'entree, vetos."""

from __future__ import annotations

import numpy as np
import pytest

from volatrade.metrics import compute_metrics
from volatrade.signals import (
    ACTION_AVOID,
    ACTION_BUY,
    ACTION_PARTIAL,
    ACTION_WATCH,
    WEIGHTS,
    evaluate,
    technical_state,
)
from tests.conftest import make_frame, trending_prices


def signal_for(frame, benchmark=None):
    metrics = compute_metrics("TEST", frame, benchmark)
    return evaluate("TEST", frame, metrics)


def test_ponderations_totalisent_un():
    assert sum(WEIGHTS.values()) == pytest.approx(1.0)


def test_etat_technique_calcule_les_moyennes_et_distances(uptrend_frame):
    state = technical_state(uptrend_frame)
    assert state.sma20 > 0 and state.sma50 > 0 and state.sma200 > 0
    assert state.high52w >= state.price >= state.low52w
    assert state.distance_high52w <= 0
    assert 0 <= state.rsi14 <= 100


def test_tendance_haussiere_donne_un_score_eleve(uptrend_frame, benchmark_returns):
    signal = signal_for(uptrend_frame, benchmark_returns)
    assert signal.components["tendance"] >= 60
    assert signal.action in (ACTION_BUY, ACTION_PARTIAL)
    assert signal.conviction > 0


def test_tendance_baissiere_est_ecartee(downtrend_frame, benchmark_returns):
    signal = signal_for(downtrend_frame, benchmark_returns)
    assert signal.action == ACTION_AVOID
    assert signal.conviction == 0.0
    assert any("MM200" in warning or "200 jours" in warning for warning in signal.warnings)


def test_veto_sous_la_mm200_avec_momentum_negatif():
    # Hausse puis effondrement : sous la MM200 et momentum 6 mois tres negatif.
    montee = np.linspace(20, 80, 260)
    chute = np.linspace(80, 30, 140)
    signal = signal_for(make_frame(np.concatenate([montee, chute])))
    assert signal.action == ACTION_AVOID


def test_titre_etire_est_penalise_sur_lentree():
    calme = trending_prices(n=380, drift=0.001, noise=0.01, seed=5)
    envolee = calme[-1] * np.exp(np.cumsum(np.full(20, 0.06)))
    signal = signal_for(make_frame(np.concatenate([calme, envolee])))
    assert signal.components["entree"] < 45
    assert any("surchauffe" in warning or "etire" in warning for warning in signal.warnings)


def test_repli_sain_est_recompense():
    montee = trending_prices(n=390, drift=0.003, noise=0.008, seed=9)
    repli = montee[-1] * np.exp(np.cumsum(np.full(6, -0.011)))
    signal = signal_for(make_frame(np.concatenate([montee, repli])))
    assert signal.components["entree"] > 60
    assert any("repli" in reason for reason in signal.reasons)


def test_serie_sans_momentum_nobtient_pas_une_position_pleine(flat_frame):
    # Serie quasi plate : meme au-dessus de ses moyennes mobiles, elle ne
    # merite qu'une demi-position faute de dynamique.
    signal = signal_for(flat_frame)
    assert signal.components["momentum"] < 45
    assert signal.action != ACTION_BUY
    assert 0 <= signal.score <= 100


def test_score_borne_et_composantes_completes(uptrend_frame):
    signal = signal_for(uptrend_frame)
    assert 0 <= signal.score <= 100
    assert set(signal.components) == set(WEIGHTS)
    assert all(0 <= value <= 100 for value in signal.components.values())


def test_conviction_reduite_pour_un_achat_partiel(uptrend_frame):
    signal = signal_for(uptrend_frame)
    convictions = {ACTION_BUY: 1.0, ACTION_PARTIAL: 0.5, ACTION_WATCH: 0.0, ACTION_AVOID: 0.0}
    assert signal.conviction == convictions[signal.action]
