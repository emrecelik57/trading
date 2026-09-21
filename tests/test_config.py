"""Chargement et validation de la configuration."""

from __future__ import annotations

import json

import pytest

from volatrade.config import AppConfig, ConfigError, load_config


def test_valeurs_par_defaut_coherentes():
    config = load_config()
    assert config.capital == 10_000
    assert config.nombre_titres == 10
    assert 0 < config.risque_par_trade < 0.1
    assert len(config.univers) > 30
    settings = config.risk_settings()
    assert settings.capital == config.capital
    assert settings.risk_per_trade == config.risque_par_trade


def test_fichier_json_ecrase_les_defauts(tmp_path):
    chemin = tmp_path / "config.json"
    chemin.write_text(json.dumps({"capital": 50_000, "nombre_titres": 5}))
    config = load_config(chemin)
    assert config.capital == 50_000
    assert config.nombre_titres == 5


def test_les_options_cli_priment_sur_le_fichier(tmp_path):
    chemin = tmp_path / "config.json"
    chemin.write_text(json.dumps({"capital": 50_000}))
    assert load_config(chemin, capital=1_000).capital == 1_000
    # Une option non fournie (None) ne doit pas ecraser le fichier.
    assert load_config(chemin, capital=None).capital == 50_000


def test_univers_accepte_une_chaine_separee_par_virgules():
    config = load_config(univers="tsla, nvda ,amd")
    assert config.univers == ["TSLA", "NVDA", "AMD"]


def test_cle_inconnue_rejetee(tmp_path):
    chemin = tmp_path / "config.json"
    chemin.write_text(json.dumps({"capitale": 1000}))
    with pytest.raises(ConfigError, match="cles inconnues"):
        load_config(chemin)


def test_fichier_absent_ou_invalide(tmp_path):
    with pytest.raises(ConfigError, match="introuvable"):
        load_config(tmp_path / "rien.json")
    mauvais = tmp_path / "mauvais.json"
    mauvais.write_text("[]")
    with pytest.raises(ConfigError, match="objet JSON"):
        load_config(mauvais)


@pytest.mark.parametrize(
    "champ,valeur",
    [
        ("capital", -1),
        ("risque_par_trade", 0.0),
        ("risque_par_trade", 0.5),
        ("poids_max", 0.0),
        ("nombre_titres", 0),
        ("multiple_atr_stop", -2),
    ],
)
def test_bornes_de_validation(champ, valeur):
    with pytest.raises(ConfigError):
        AppConfig(**{champ: valeur}).validate()


def test_univers_vide_rejete():
    with pytest.raises(ConfigError, match="univers"):
        AppConfig(univers=[]).validate()


def test_serialisation():
    data = load_config().as_dict()
    json.dumps(data)
    assert "capital" in data and "univers" in data
