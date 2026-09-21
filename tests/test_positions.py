"""Suivi des positions ouvertes : verdicts de vente, d'allegement, de conservation."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from volatrade.positions import (
    PortfolioError,
    Position,
    VERDICT_HOLD,
    VERDICT_SELL,
    VERDICT_TIGHTEN,
    VERDICT_TRIM,
    load_portfolio,
    parse_date,
    review_portfolio,
    review_position,
)
from volatrade.risk import RiskSettings
from tests.conftest import make_quote


def test_lecture_du_fichier_de_portefeuille(tmp_path):
    chemin = tmp_path / "pf.json"
    chemin.write_text(json.dumps([
        {"ticker": "abc", "shares": 10, "entry_price": 5.0, "entry_date": "2024-01-02"}
    ]))
    positions = load_portfolio(chemin)
    assert positions[0].ticker == "ABC"
    assert positions[0].shares == 10


def test_lecture_accepte_la_forme_objet(tmp_path):
    chemin = tmp_path / "pf.json"
    chemin.write_text(json.dumps({"positions": [{"ticker": "A", "shares": 1, "entry_price": 2.0}]}))
    assert len(load_portfolio(chemin)) == 1


def test_erreurs_de_fichier_explicites(tmp_path):
    with pytest.raises(PortfolioError, match="introuvable"):
        load_portfolio(tmp_path / "absent.json")
    mauvais = tmp_path / "mauvais.json"
    mauvais.write_text("{pas du json")
    with pytest.raises(PortfolioError, match="JSON invalide"):
        load_portfolio(mauvais)
    incomplet = tmp_path / "incomplet.json"
    incomplet.write_text(json.dumps([{"ticker": "A"}]))
    with pytest.raises(PortfolioError, match="shares"):
        load_portfolio(incomplet)


def test_stop_touche_declenche_une_vente():
    quote = make_quote("A", np.linspace(100, 60, 260))
    position = Position("A", 10, entry_price=100.0, stop_price=80.0)
    review = review_position(position, quote)
    assert review.verdict == VERDICT_SELL
    assert "stop touche" in review.actions[0]
    assert review.r_multiple < 0


def test_objectif_atteint_declenche_un_allegement():
    quote = make_quote("A", np.linspace(100, 140, 260))
    position = Position("A", 10, entry_price=100.0, stop_price=90.0, target1=115.0, target2=200.0)
    review = review_position(position, quote)
    assert review.verdict == VERDICT_TRIM
    assert "objectif 1" in review.actions[0]
    assert review.r_multiple == pytest.approx((quote.last_price - 100.0) / 10.0)


def test_objectif_deja_encaisse_nest_pas_repropose():
    quote = make_quote("A", np.linspace(100, 140, 260))
    position = Position("A", 10, entry_price=100.0, stop_price=90.0, target1=115.0,
                        target2=500.0, trimmed=["objectif1"])
    assert review_position(position, quote).verdict != VERDICT_TRIM


def test_rupture_de_tendance_declenche_une_vente():
    # Longue hausse puis cassure franche : le cours passe nettement sous la MM50.
    montee = np.linspace(50, 120, 240)
    chute = np.linspace(120, 70, 30)
    quote = make_quote("A", np.concatenate([montee, chute]), amplitude=0.005)
    position = Position("A", 10, entry_price=60.0, stop_price=40.0, target1=500.0, target2=600.0)
    review = review_position(position, quote)
    assert review.verdict == VERDICT_SELL
    assert "moyenne 50 jours" in review.actions[0]


def test_bruit_sous_la_mm50_ne_declenche_pas_de_vente():
    # Serie haussiere bruitee : des clotures passent sous la MM50 sans casser
    # la tendance ; la marge d'un ATR doit eviter la sortie.
    rng = np.random.default_rng(5)
    prices = 50 * np.exp(np.cumsum(0.002 + 0.015 * rng.standard_normal(300)))
    quote = make_quote("A", prices)
    position = Position("A", 10, entry_price=float(prices[-30]), stop_price=float(prices[-30]) * 0.5,
                        target1=1e9, target2=1e9)
    assert review_position(position, quote).verdict != VERDICT_SELL


def test_stop_temporel_solde_une_position_qui_stagne():
    quote = make_quote("A", np.full(260, 100.0), start="2024-01-01")
    date_entree = str(quote.frame.index[-40].date())
    position = Position("A", 10, entry_price=100.0, entry_date=date_entree,
                        stop_price=90.0, target1=200.0, target2=300.0)
    review = review_position(position, quote)
    assert review.verdict == VERDICT_SELL
    assert "stop temporel" in review.actions[0]
    assert review.days_held >= 25


def test_position_gagnante_fait_remonter_le_stop():
    quote = make_quote("A", np.linspace(100, 130, 260), amplitude=0.005)
    position = Position("A", 10, entry_price=100.0, stop_price=90.0,
                        target1=1e9, target2=1e9)
    review = review_position(position, quote)
    assert review.verdict == VERDICT_TIGHTEN
    assert review.suggested_stop > review.stop_price


def test_position_calme_est_conservee():
    quote = make_quote("A", np.linspace(100, 104, 260), amplitude=0.001)
    position = Position("A", 10, entry_price=100.0, stop_price=95.0,
                        target1=1e9, target2=1e9)
    review = review_position(position, quote)
    assert review.verdict in (VERDICT_HOLD, VERDICT_TIGHTEN)
    assert review.pnl == pytest.approx((review.price - 100.0) * 10)


def test_stop_reconstruit_si_absent():
    quote = make_quote("A", np.linspace(100, 110, 260))
    review = review_position(Position("A", 10, entry_price=100.0), quote)
    assert review.stop_price < 100.0
    assert review.target1 > 100.0


def test_revue_ignore_les_positions_sans_cours():
    quote = make_quote("A", np.linspace(100, 110, 260))
    positions = [Position("A", 1, 100.0), Position("INCONNU", 1, 10.0)]
    reviews = review_portfolio(positions, {"A": quote})
    assert [review.ticker for review in reviews] == ["A"]


# --------------------------------------------------------------------------
# Regle "resultats" : sortie totale avant une publication
# --------------------------------------------------------------------------
def quote_calme(n: int = 260, debut: str = "2026-01-01"):
    """Serie sans signal : aucune autre regle ne doit se declencher."""
    return make_quote("A", np.linspace(100, 104, n), start=debut, amplitude=0.001)


def position_calme(**extra) -> Position:
    base = dict(ticker="A", shares=10, entry_price=100.0, stop_price=95.0,
                target1=1e9, target2=1e9)
    base.update(extra)
    return Position(**base)


def test_lecture_des_dates_valides_et_invalides():
    assert parse_date("2026-11-02") == pd.Timestamp("2026-11-02")
    assert parse_date(pd.Timestamp("2026-11-02", tz="UTC")) == pd.Timestamp("2026-11-02")
    for mauvais in (None, "", "   ", "pas une date", 42.0, 2026, True):
        assert parse_date(mauvais) is None, mauvais


def test_date_resultats_illisible_rejetee_au_chargement(tmp_path):
    chemin = tmp_path / "pf.json"
    chemin.write_text(json.dumps([
        {"ticker": "A", "shares": 1, "entry_price": 10.0, "date_resultats": "32/13/2026"}
    ]))
    with pytest.raises(PortfolioError, match="date_resultats"):
        load_portfolio(chemin)


def test_date_resultats_absente_ne_change_rien():
    quote = quote_calme()
    review = review_position(position_calme(), quote)
    assert review.days_to_earnings is None
    assert review.verdict != VERDICT_SELL


def test_publication_imminente_declenche_une_vente_totale():
    quote = quote_calme()
    dans_deux_seances = str((quote.frame.index[-1] + pd.offsets.BDay(2)).date())
    review = review_position(position_calme(date_resultats=dans_deux_seances), quote)
    assert review.verdict == VERDICT_SELL
    assert review.days_to_earnings == 2
    assert "publication des resultats" in review.actions[0]
    assert "gap" in review.actions[0]


def test_publication_le_jour_meme_declenche_aussi():
    quote = quote_calme()
    aujourd_hui = str(quote.frame.index[-1].date())
    review = review_position(position_calme(date_resultats=aujourd_hui), quote)
    assert review.verdict == VERDICT_SELL
    assert "aujourd'hui" in review.actions[0]


def test_publication_lointaine_ne_declenche_pas_mais_previent():
    quote = quote_calme()
    dans_un_mois = str((quote.frame.index[-1] + pd.offsets.BDay(22)).date())
    review = review_position(position_calme(date_resultats=dans_un_mois), quote)
    assert review.verdict != VERDICT_SELL
    assert review.days_to_earnings == 22
    assert any("publication dans 22 seances" in note for note in review.notes)
    assert any("sortie prevue dans 19 seances" in note for note in review.notes)


def test_seuil_de_declenchement_configurable():
    quote = quote_calme()
    dans_cinq = str((quote.frame.index[-1] + pd.offsets.BDay(5)).date())
    position = position_calme(date_resultats=dans_cinq)
    assert review_position(position, quote).verdict != VERDICT_SELL
    large = RiskSettings(earnings_exit_days=7)
    assert review_position(position, quote, large).verdict == VERDICT_SELL


def test_politique_desactivable_ne_fait_qu_alerter():
    quote = quote_calme()
    demain = str((quote.frame.index[-1] + pd.offsets.BDay(1)).date())
    position = position_calme(date_resultats=demain)
    sans_sortie = RiskSettings(exit_before_earnings=False)
    review = review_position(position, quote, sans_sortie)
    assert review.verdict != VERDICT_SELL
    assert review.days_to_earnings == 1
    assert any("publication dans 1 seances" in note for note in review.notes)


def test_date_resultats_passee_signalee_sans_vendre():
    quote = quote_calme()
    passee = str((quote.frame.index[-1] - pd.offsets.BDay(10)).date())
    review = review_position(position_calme(date_resultats=passee), quote)
    assert review.verdict != VERDICT_SELL
    assert review.days_to_earnings == -10
    assert any("depassee" in note for note in review.notes)


def test_stop_touche_prime_sur_la_publication():
    # Le stop reste la regle la plus urgente : on ne conserve pas une ligne
    # stoppee sous pretexte qu'une publication approche.
    quote = make_quote("A", np.linspace(100, 60, 260), start="2026-01-01")
    demain = str((quote.frame.index[-1] + pd.offsets.BDay(1)).date())
    review = review_position(
        Position("A", 10, entry_price=100.0, stop_price=80.0, date_resultats=demain), quote
    )
    assert review.verdict == VERDICT_SELL
    assert "stop touche" in review.actions[0]


def test_publication_prime_sur_les_objectifs():
    # Tout ou rien : on solde avant la publication plutot que d'alleger.
    quote = make_quote("A", np.linspace(100, 140, 260), start="2026-01-01", amplitude=0.001)
    demain = str((quote.frame.index[-1] + pd.offsets.BDay(1)).date())
    review = review_position(
        Position("A", 10, entry_price=100.0, stop_price=90.0, target1=115.0,
                 target2=500.0, date_resultats=demain),
        quote,
    )
    assert review.verdict == VERDICT_SELL
    assert "publication" in review.actions[0]
