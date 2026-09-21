"""Calendrier de publication : lecture des reponses Nasdaq et projection."""

from __future__ import annotations

import pandas as pd
import pytest

from volatrade import data, earnings
from volatrade.data import DataError
from volatrade.earnings import (
    SOURCE_ANNOUNCE,
    SOURCE_PROJECTED,
    EarningsDate,
    complete_positions,
    fetch_earnings_date,
    parse_announcement,
    parse_reported_dates,
    project_next,
)
from volatrade.positions import Position

AUJOURD_HUI = pd.Timestamp("2026-09-21")


def annonce(texte: str) -> dict:
    return {"data": {"announcement": texte, "heading": "AMD Earnings Date"}}


def historique(dates: list[str]) -> dict:
    return {
        "data": {
            "symbol": "amd",
            "earningsSurpriseTable": {
                "rows": [{"dateReported": d, "eps": 1.0} for d in dates]
            },
        }
    }


# --------------------------------------------------------------------------
# Lecture des reponses
# --------------------------------------------------------------------------
def test_lecture_de_lannonce():
    payload = annonce("Earnings announcement* for AMD: Nov 3, 2026")
    assert parse_announcement(payload) == pd.Timestamp("2026-11-03")


def test_lecture_de_lannonce_formats_alternatifs():
    assert parse_announcement(annonce("x: November 3, 2026")) == pd.Timestamp("2026-11-03")
    assert parse_announcement(annonce("x: 2026-11-03")) == pd.Timestamp("2026-11-03")


@pytest.mark.parametrize("texte", [
    "Earnings announcement* for NVDA: ",      # champ vide, cas reel
    "Earnings announcement* for X: TBA",      # date non communiquee
    "Earnings announcement* for X: N/A",
    "pas de deux points",
    "",
])
def test_annonce_illisible_renvoie_none(texte):
    assert parse_announcement(annonce(texte)) is None


def test_annonce_absente_du_payload():
    assert parse_announcement({}) is None
    assert parse_announcement({"data": None}) is None


def test_lecture_des_dates_publiees_triees():
    payload = historique(["8/4/2026", "5/5/2026", "2/3/2026", "11/4/2025"])
    dates = parse_reported_dates(payload)
    assert dates == [
        pd.Timestamp("2025-11-04"), pd.Timestamp("2026-02-03"),
        pd.Timestamp("2026-05-05"), pd.Timestamp("2026-08-04"),
    ]


def test_dates_publiees_illisibles_ignorees():
    payload = historique(["8/4/2026", "pas une date", ""])
    assert parse_reported_dates(payload) == [pd.Timestamp("2026-08-04")]
    assert parse_reported_dates({}) == []


# --------------------------------------------------------------------------
# Projection trimestrielle
# --------------------------------------------------------------------------
def test_projection_suit_la_cadence_observee():
    # Quatre publications espacees d'environ 91 jours (cas reel AMD).
    reported = parse_reported_dates(historique(["11/4/2025", "2/3/2026", "5/5/2026", "8/4/2026"]))
    projete = project_next(reported, AUJOURD_HUI)
    assert projete is not None
    assert pd.Timestamp("2026-10-25") <= projete <= pd.Timestamp("2026-11-15")


def test_projection_saute_les_trimestres_deja_passes():
    # Historique ancien : la projection doit depasser la date du jour.
    reported = [pd.Timestamp("2025-01-15"), pd.Timestamp("2025-04-15")]
    projete = project_next(reported, AUJOURD_HUI)
    assert projete is not None and projete > AUJOURD_HUI


def test_projection_ignore_les_ecarts_aberrants():
    # Un ecart de 400 jours ne doit pas devenir la cadence de reference.
    reported = [pd.Timestamp("2024-01-10"), pd.Timestamp("2025-02-14"), pd.Timestamp("2025-05-15")]
    projete = project_next(reported, AUJOURD_HUI)
    assert projete is not None
    ecart = (projete - pd.Timestamp("2025-05-15")).days
    assert ecart % 91 < 8 or ecart < 8 * 95


def test_projection_sans_historique():
    assert project_next([], AUJOURD_HUI) is None


def test_projection_abandonne_si_historique_trop_vieux():
    # Huit trimestres au maximum : un historique de 2015 ne projette rien.
    assert project_next([pd.Timestamp("2015-01-10")], AUJOURD_HUI) is None


# --------------------------------------------------------------------------
# Recuperation complete
# --------------------------------------------------------------------------
@pytest.fixture
def faux_nasdaq(monkeypatch):
    """Remplace les appels HTTP ; `reponses` pilote ce que renvoie chaque endpoint."""
    reponses: dict[str, dict] = {}

    def faux_get(url, timeout, attempts=3):
        if "analyst" in url:
            if reponses.get("annonce_echoue"):
                raise DataError("indisponible")
            return reponses.get("annonce", annonce(""))
        if reponses.get("historique_echoue"):
            raise DataError("indisponible")
        return reponses.get("historique", historique([]))

    monkeypatch.setattr(data, "http_get_json", faux_get)
    return reponses


def test_annonce_prioritaire_sur_la_projection(faux_nasdaq, tmp_path):
    faux_nasdaq["annonce"] = annonce("Earnings announcement* for AMD: Nov 3, 2026")
    faux_nasdaq["historique"] = historique(["11/4/2025", "2/3/2026", "5/5/2026", "8/4/2026"])
    trouve = fetch_earnings_date("AMD", today=AUJOURD_HUI, cache_dir=tmp_path)
    assert trouve == EarningsDate("AMD", "2026-11-03", SOURCE_ANNOUNCE)
    assert not trouve.is_projected
    assert "estimation" in trouve.describe()


def test_repli_sur_la_projection_quand_lannonce_est_vide(faux_nasdaq, tmp_path):
    faux_nasdaq["historique"] = historique(["11/4/2025", "2/3/2026", "5/5/2026", "8/4/2026"])
    trouve = fetch_earnings_date("NVDA", today=AUJOURD_HUI, cache_dir=tmp_path)
    assert trouve is not None and trouve.source == SOURCE_PROJECTED


def test_annonce_deja_passee_bascule_sur_la_projection(faux_nasdaq, tmp_path):
    faux_nasdaq["annonce"] = annonce("Earnings announcement* for X: Aug 4, 2026")
    faux_nasdaq["historique"] = historique(["11/4/2025", "2/3/2026", "5/5/2026", "8/4/2026"])
    trouve = fetch_earnings_date("X", today=AUJOURD_HUI, cache_dir=tmp_path)
    assert trouve is not None and trouve.source == SOURCE_PROJECTED
    assert trouve.date > "2026-09-21"


def test_aucune_donnee_renvoie_none(faux_nasdaq, tmp_path):
    assert fetch_earnings_date("INCONNU", today=AUJOURD_HUI, cache_dir=tmp_path) is None


def test_endpoint_en_panne_nempeche_pas_lautre(faux_nasdaq, tmp_path):
    faux_nasdaq["annonce_echoue"] = True
    faux_nasdaq["historique"] = historique(["11/4/2025", "2/3/2026", "5/5/2026", "8/4/2026"])
    trouve = fetch_earnings_date("AMD", today=AUJOURD_HUI, cache_dir=tmp_path)
    assert trouve is not None and trouve.source == SOURCE_PROJECTED


def test_les_deux_endpoints_en_panne(faux_nasdaq, tmp_path):
    faux_nasdaq["annonce_echoue"] = True
    faux_nasdaq["historique_echoue"] = True
    assert fetch_earnings_date("AMD", today=AUJOURD_HUI, cache_dir=tmp_path) is None


def test_cache_evite_un_second_appel(faux_nasdaq, tmp_path, monkeypatch):
    faux_nasdaq["annonce"] = annonce("Earnings announcement* for AMD: Nov 3, 2026")
    fetch_earnings_date("AMD", today=AUJOURD_HUI, cache_dir=tmp_path)
    monkeypatch.setattr(data, "http_get_json", lambda *a, **k: pytest.fail("appel reseau inattendu"))
    trouve = fetch_earnings_date("AMD", today=AUJOURD_HUI, cache_dir=tmp_path)
    assert trouve.date == "2026-11-03"


# --------------------------------------------------------------------------
# Completion des positions
# --------------------------------------------------------------------------
def test_date_manquante_est_completee(faux_nasdaq, tmp_path):
    faux_nasdaq["annonce"] = annonce("Earnings announcement* for AMD: Nov 3, 2026")
    position = Position("AMD", 1, 500.0)
    messages = complete_positions([position], today=AUJOURD_HUI, cache_dir=tmp_path)
    assert position.date_resultats == "2026-11-03"
    assert "recuperee automatiquement" in messages[0]


def test_date_saisie_a_la_main_nest_jamais_ecrasee(faux_nasdaq, tmp_path):
    faux_nasdaq["annonce"] = annonce("Earnings announcement* for AMD: Nov 3, 2026")
    position = Position("AMD", 1, 500.0, date_resultats="2026-11-02")
    messages = complete_positions([position], today=AUJOURD_HUI, cache_dir=tmp_path)
    assert position.date_resultats == "2026-11-02"
    assert "conservee" in messages[0] and "2026-11-03" in messages[0]


def test_accord_avec_nasdaq_ne_produit_aucun_message(faux_nasdaq, tmp_path):
    faux_nasdaq["annonce"] = annonce("Earnings announcement* for AMD: Nov 3, 2026")
    position = Position("AMD", 1, 500.0, date_resultats="2026-11-03")
    assert complete_positions([position], today=AUJOURD_HUI, cache_dir=tmp_path) == []


def test_date_introuvable_est_signalee(faux_nasdaq, tmp_path):
    position = Position("INCONNU", 1, 10.0)
    messages = complete_positions([position], today=AUJOURD_HUI, cache_dir=tmp_path)
    assert position.date_resultats is None
    assert "aucune date" in messages[0]
