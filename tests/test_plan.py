"""Construction du plan de trade : objectifs, stops, regles de sortie."""

from __future__ import annotations

import numpy as np
import pytest

from volatrade.metrics import compute_metrics
from volatrade.plan import (
    TARGET_LADDER,
    TIME_STOP_DAYS,
    build_plan,
    chandelier_stop,
    is_actionable,
    summarize_basket,
)
from volatrade.risk import RiskSettings, size_position
from volatrade.signals import evaluate
from tests.conftest import make_frame


SETTINGS = RiskSettings(capital=50_000, risk_per_trade=0.01)


def plan_for(frame, capital: float = 50_000):
    settings = RiskSettings(capital=capital, risk_per_trade=0.01)
    metrics = compute_metrics("TEST", frame)
    signal = evaluate("TEST", frame, metrics)
    sizing = size_position(
        "TEST", metrics.last_price, signal.state.atr14, metrics.vol_ewma,
        signal.conviction, settings,
    )
    return build_plan(signal, metrics, sizing, "theme", settings), signal, sizing


def test_objectifs_places_a_1_5_r_et_3_r(uptrend_frame):
    plan, _, sizing = plan_for(uptrend_frame)
    unit = sizing.stop_distance
    assert plan.unit_risk == pytest.approx(unit)
    assert plan.target1 == pytest.approx(plan.price + TARGET_LADDER[0][0] * unit)
    assert plan.target2 == pytest.approx(plan.price + TARGET_LADDER[1][0] * unit)
    assert plan.stop_price == pytest.approx(plan.price - unit)


def test_zone_dachat_encadre_le_cours(uptrend_frame):
    plan, _, _ = plan_for(uptrend_frame)
    assert plan.entry_low < plan.price < plan.entry_high
    assert plan.entry_high - plan.entry_low > 0


def test_toutes_les_regles_de_sortie_sont_presentes(uptrend_frame):
    plan, _, _ = plan_for(uptrend_frame)
    noms = {rule.nom for rule in plan.exits}
    assert noms == {
        "stop initial",
        "objectif 1 (+1,5 R)",
        "objectif 2 (+3 R)",
        "stop suiveur",
        "rupture de tendance",
        "stop temporel",
        "choc de volatilite",
    }
    assert plan.time_stop_days == TIME_STOP_DAYS
    assert all(rule.action for rule in plan.exits)


def test_risque_annonce_egale_quantite_fois_distance_de_stop(uptrend_frame):
    plan, _, _ = plan_for(uptrend_frame)
    assert plan.risk_amount == pytest.approx(plan.shares * plan.unit_risk)
    assert plan.risk_pct <= 0.01 + 1e-9


def test_plan_non_actionnable_sur_tendance_baissiere(downtrend_frame):
    plan, signal, _ = plan_for(downtrend_frame)
    assert signal.conviction == 0
    assert plan.shares == 0
    assert not is_actionable(plan)


def test_avertissement_si_le_capital_ne_permet_pas_une_ligne(uptrend_frame):
    plan, signal, _ = plan_for(uptrend_frame, capital=200.0)
    if signal.conviction > 0:
        assert plan.shares == 0
        assert any("capital" in warning for warning in plan.warnings)


def test_horizon_borne_et_coherent(uptrend_frame):
    plan, _, _ = plan_for(uptrend_frame)
    assert 5 <= plan.horizon_days <= 180


def test_stop_suiveur_sous_le_plus_haut_recent():
    frame = make_frame(np.linspace(10, 20, 100), amplitude=0.0)
    stop = chandelier_stop(frame, atr_value=1.0)
    assert stop == pytest.approx(20.0 - 3.0)
    assert np.isnan(chandelier_stop(frame, atr_value=float("nan")))


def test_bilan_du_panier_agrege_les_lignes_actionnables(uptrend_frame, downtrend_frame):
    haut, _, _ = plan_for(uptrend_frame)
    bas, _, _ = plan_for(downtrend_frame)
    summary = summarize_basket([haut, bas])
    assert summary["lignes"] == float(len([p for p in (haut, bas) if is_actionable(p)]))
    assert summary["capital_engage"] == pytest.approx(
        sum(p.notional for p in (haut, bas) if is_actionable(p))
    )


def test_plan_serialisable_en_json(uptrend_frame):
    import json

    plan, _, _ = plan_for(uptrend_frame)
    data = plan.as_dict()
    assert isinstance(data["exits"], list)
    json.dumps(data)
