"""Backtest : absence de biais de look-ahead et coherence des sorties."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from volatrade.backtest import WARMUP, run_backtest, summarize
from volatrade.risk import RiskSettings
from tests.conftest import make_quote, trending_prices


SETTINGS = RiskSettings(capital=100_000, risk_per_trade=0.01, atr_stop_multiple=2.5)


def test_rapport_vide_sans_trade():
    report = summarize([])
    assert report.n_trades == 0
    assert np.isnan(report.avg_r)


def test_entree_a_louverture_de_la_seance_suivante():
    quote = make_quote("HAUT", trending_prices(n=400, drift=0.004, noise=0.01, seed=2))
    _, per_ticker = run_backtest({"HAUT": quote}, None, SETTINGS, decision_every=5)
    trades = per_ticker["HAUT"].trades
    assert trades, "une serie franchement haussiere doit declencher des trades"
    frame = quote.frame
    for trade in trades:
        date = pd.Timestamp(trade.entry_date, tz="UTC")
        assert trade.entry_price == pytest.approx(float(frame.loc[date, "open"]))


def test_aucun_trade_avant_la_periode_de_chauffe():
    quote = make_quote("HAUT", trending_prices(n=400, drift=0.004, noise=0.01, seed=2))
    _, per_ticker = run_backtest({"HAUT": quote}, None, SETTINGS, decision_every=5)
    premiere_date_possible = quote.frame.index[WARMUP]
    for trade in per_ticker["HAUT"].trades:
        assert pd.Timestamp(trade.entry_date, tz="UTC") >= premiere_date_possible


def test_perte_bornee_a_une_unite_de_risque():
    # Hausse suivie d'un effondrement : les trades perdants sortent au stop,
    # donc au-dela de -1 R seulement en cas de gap a l'ouverture.
    prices = np.concatenate([trending_prices(n=300, drift=0.003, noise=0.01, seed=1),
                             np.linspace(1.0, 0.5, 120) * trending_prices(1, seed=1)[0]])
    quote = make_quote("CASSE", prices)
    report, _ = run_backtest({"CASSE": quote}, None, SETTINGS, decision_every=5)
    if report.n_trades:
        assert report.worst_r >= -1.6


def test_serie_baissiere_ne_declenche_aucun_achat(downtrend_frame):
    from volatrade.data import Quote

    quote = Quote("BAS", "USD", "TEST", downtrend_frame)
    report, _ = run_backtest({"BAS": quote}, None, SETTINGS, decision_every=5)
    assert report.n_trades == 0


def test_historique_trop_court_ignore():
    quote = make_quote("COURT", np.linspace(10, 20, 100))
    report, per_ticker = run_backtest({"COURT": quote}, None, SETTINGS)
    assert report.n_trades == 0
    assert per_ticker["COURT"].n_trades == 0


def test_statistiques_coherentes():
    quote = make_quote("HAUT", trending_prices(n=500, drift=0.003, noise=0.015, seed=8))
    report, _ = run_backtest({"HAUT": quote}, None, SETTINGS, decision_every=5)
    if report.n_trades:
        resultats = [trade.r_multiple for trade in report.trades]
        assert report.total_r == pytest.approx(sum(resultats))
        assert report.avg_r == pytest.approx(np.mean(resultats))
        assert report.best_r == pytest.approx(max(resultats))
        assert report.worst_r == pytest.approx(min(resultats))
        assert 0 <= report.win_rate <= 1
        assert report.max_drawdown_r <= 0
        assert report.capital_return == pytest.approx(report.total_r * SETTINGS.risk_per_trade)


def test_cadence_de_decision_reduit_le_nombre_de_trades():
    quote = make_quote("HAUT", trending_prices(n=600, drift=0.002, noise=0.02, seed=12))
    rapide, _ = run_backtest({"HAUT": quote}, None, SETTINGS, decision_every=1)
    lent, _ = run_backtest({"HAUT": quote}, None, SETTINGS, decision_every=20)
    assert rapide.n_trades >= lent.n_trades


def test_reference_acheter_conserver_calculee():
    quote = make_quote("HAUT", trending_prices(n=500, drift=0.002, noise=0.01, seed=3))
    report, _ = run_backtest({"HAUT": quote}, None, SETTINGS, decision_every=5)
    attendu = float(quote.frame["close"].iloc[-1] / quote.frame["close"].iloc[WARMUP] - 1)
    assert report.buy_hold == pytest.approx(attendu)
