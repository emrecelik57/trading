"""Interface en ligne de commande et chaine complete, sans acces reseau."""

from __future__ import annotations

import json

import numpy as np
import pytest

from volatrade import data
from volatrade.cli import build_parser, main
from volatrade.config import load_config
from volatrade.engine import analyse
from tests.conftest import make_quote, trending_prices


TICKERS = ["HAUT", "BAS", "PLAT", "SPY"]


def _series(ticker: str) -> np.ndarray:
    rng = np.random.default_rng(abs(hash(ticker)) % 1000)
    if ticker == "HAUT":
        return trending_prices(n=420, drift=0.004, noise=0.025, seed=21)
    if ticker == "BAS":
        return 120 * np.exp(np.cumsum(-0.004 + 0.03 * rng.standard_normal(420)))
    if ticker == "SPY":
        return 400 * np.exp(np.cumsum(0.0005 + 0.008 * rng.standard_normal(420)))
    return 30 * np.exp(np.cumsum(0.001 + 0.02 * rng.standard_normal(420)))


@pytest.fixture
def faux_reseau(monkeypatch):
    """Remplace l'appel HTTP par des series synthetiques."""

    def faux_get(url, timeout, attempts=3):
        ticker = url.split("/chart/")[1].split("?")[0]
        closes = _series(ticker)
        return {
            "chart": {
                "result": [
                    {
                        "meta": {"currency": "USD", "fullExchangeName": "TEST"},
                        "timestamp": [1600000000 + 86400 * i for i in range(len(closes))],
                        "indicators": {
                            "quote": [
                                {
                                    "open": list(closes),
                                    "high": list(closes * 1.03),
                                    "low": list(closes * 0.97),
                                    "close": list(closes),
                                    "volume": [8_000_000] * len(closes),
                                }
                            ]
                        },
                    }
                ],
                "error": None,
            }
        }

    monkeypatch.setattr(data, "_http_get_json", faux_get)


@pytest.fixture
def config_path(tmp_path):
    chemin = tmp_path / "config.json"
    chemin.write_text(json.dumps({
        "capital": 100_000,
        "nombre_titres": 3,
        "univers": ["HAUT", "BAS", "PLAT"],
        "indice_reference": "SPY",
        "volume_min": 0,
        "dossier_cache": str(tmp_path / "cache"),
    }))
    return chemin


def lancer(args, config_path) -> int:
    return main(["--config", str(config_path)] + args)


def test_parser_accepte_les_options_avant_ou_apres_la_sous_commande():
    parser = build_parser()
    avant = parser.parse_args(["--capital", "500", "plan"])
    apres = parser.parse_args(["plan", "--capital", "500"])
    assert avant.capital == apres.capital == 500
    assert avant.commande == apres.commande == "plan"


def test_commande_par_defaut_est_le_plan(faux_reseau, config_path, capsys):
    assert main(["--config", str(config_path)]) == 0
    sortie = capsys.readouterr().out
    assert "Plans d'achat" in sortie
    assert "Bilan du panier" in sortie


def test_commande_selection(faux_reseau, config_path, capsys):
    assert lancer(["selection"], config_path) == 0
    sortie = capsys.readouterr().out
    assert "Selection" in sortie and "HAUT" in sortie


def test_commande_risque_affiche_les_mesures(faux_reseau, config_path, capsys):
    assert lancer(["risque"], config_path) == 0
    sortie = capsys.readouterr().out
    for colonne in ("VaR 95 %", "CVaR 95 %", "Sharpe", "Sortino", "Correlations"):
        assert colonne in sortie


def test_commande_plan_affiche_les_regles_de_vente(faux_reseau, config_path, capsys):
    assert lancer(["plan", "--tout"], config_path) == 0
    sortie = capsys.readouterr().out
    assert "Quand vendre" in sortie
    assert "stop initial" in sortie
    assert "stop temporel" in sortie


def test_export_json(faux_reseau, config_path, tmp_path, capsys):
    cible = tmp_path / "rapport.json"
    assert lancer(["plan", "--json", str(cible)], config_path) == 0
    capsys.readouterr()
    data_export = json.loads(cible.read_text())
    assert {"configuration", "selection", "plans", "panier"} <= set(data_export)
    assert data_export["selection"], "la selection exportee ne doit pas etre vide"


def test_commande_analyse(faux_reseau, config_path, capsys):
    assert lancer(["analyse", "HAUT"], config_path) == 0
    sortie = capsys.readouterr().out
    assert "Composantes du score" in sortie


def test_commande_backtest(faux_reseau, config_path, capsys):
    assert lancer(["backtest", "--cadence", "10"], config_path) == 0
    sortie = capsys.readouterr().out
    assert "Backtest" in sortie


def test_commande_suivi(faux_reseau, config_path, tmp_path, capsys):
    portefeuille = tmp_path / "pf.json"
    portefeuille.write_text(json.dumps([
        {"ticker": "HAUT", "shares": 10, "entry_price": 50.0, "entry_date": "2021-01-04"}
    ]))
    assert lancer(["suivi", "-p", str(portefeuille)], config_path) == 0
    sortie = capsys.readouterr().out
    assert "Suivi de 1 position" in sortie
    assert "HAUT" in sortie


def test_suivi_signale_un_fichier_manquant(faux_reseau, config_path, tmp_path, capsys):
    code = lancer(["suivi", "-p", str(tmp_path / "absent.json")], config_path)
    assert code == 2
    assert "introuvable" in capsys.readouterr().err


def test_configuration_invalide_retourne_un_code_derreur(capsys):
    assert main(["--capital", "-5", "plan"]) == 2
    assert "configuration" in capsys.readouterr().err


def test_chaine_complete_sans_reseau(faux_reseau, config_path):
    """L'analyse produit un plan coherent de bout en bout."""
    from volatrade.engine import load_market

    config = load_config(config_path)
    quotes, benchmark, errors = load_market(config)
    view = analyse(config, quotes, benchmark, errors)
    assert len(view.selection) == 3
    assert len(view.plans) == 3
    assert not view.correlations.empty
    for plan in view.plans:
        assert plan.stop_price < plan.price < plan.target1 < plan.target2
        assert plan.risk_pct <= config.risque_par_trade + 1e-9
    total_risque = sum(plan.risk_pct for plan in view.plans)
    assert total_risque <= config.chaleur_max + 1e-9
    json.dumps(view.as_dict(), default=str)
