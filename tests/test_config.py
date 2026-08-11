"""Validation de la configuration : les erreurs doivent etre explicites."""

from __future__ import annotations

import pytest
import yaml

from quantfolio.config import Config, ConfigError, load_config
from quantfolio.templates import DEFAULT_CONFIG_YAML


def _config(**overrides) -> Config:
    payload = {"universe": {"tickers": ["A", "B"]}}
    for section, values in overrides.items():
        payload.setdefault(section, {}).update(values)
    return Config.from_dict(payload)


def test_shipped_template_is_valid():
    """Le modele genere par `quantfolio init` doit se charger tel quel."""
    cfg = Config.from_dict(yaml.safe_load(DEFAULT_CONFIG_YAML))
    assert len(cfg.universe.tickers) >= 15
    assert cfg.model.kind in {"gbm", "ridge", "rules"}


def test_unknown_key_is_rejected():
    with pytest.raises(ConfigError, match="Cle\\(s\\) inconnue\\(s\\)"):
        _config(portfolio={"max_positon": 5})


def test_unknown_section_is_rejected():
    with pytest.raises(ConfigError, match="Section"):
        Config.from_dict({"universe": {"tickers": ["A"]}, "portfeuille": {}})


def test_empty_universe_is_rejected():
    with pytest.raises(ConfigError, match="universe.tickers est vide"):
        Config.from_dict({})


def test_duplicate_tickers_are_rejected():
    with pytest.raises(ConfigError, match="doublons"):
        Config.from_dict({"universe": {"tickers": ["A", "A"]}})


def test_unreachable_exposure_is_rejected():
    # 3 lignes plafonnees a 10 % ne peuvent pas faire 100 % d'exposition.
    with pytest.raises(ConfigError, match="max_weight x max_positions"):
        _config(portfolio={"max_positions": 3, "max_weight": 0.10})


def test_exit_threshold_must_be_below_the_buy_threshold():
    with pytest.raises(ConfigError, match="exit_threshold"):
        _config(portfolio={"score_threshold": 0.0, "exit_threshold": 0.5})


@pytest.mark.parametrize(
    "section,values,message",
    [
        ("model", {"kind": "reseau_de_neurones"}, "model.kind"),
        ("model", {"label": "autre"}, "model.label"),
        ("model", {"horizon": 0}, "horizon"),
        ("model", {"rules_weight": 1.5}, "rules_weight"),
        ("portfolio", {"capital": -1}, "capital"),
        ("portfolio", {"vol_target": 0}, "vol_target"),
        ("portfolio", {"cash_buffer": 1.0}, "cash_buffer"),
        ("backtest", {"execution": "demain"}, "execution"),
        ("backtest", {"rebalance_days": 0}, "rebalance_days"),
        ("data", {"provider": "bloomberg"}, "provider"),
    ],
)
def test_invalid_values_are_reported(section, values, message):
    with pytest.raises(ConfigError, match=message):
        _config(**{section: values})


def test_custom_provider_requires_a_specification():
    with pytest.raises(ConfigError, match="data.custom"):
        _config(data={"provider": "custom"})


def test_yaml_roundtrip(tmp_path):
    directory = tmp_path / "config"
    directory.mkdir()
    path = directory / "config.yaml"
    path.write_text(DEFAULT_CONFIG_YAML, encoding="utf-8")

    cfg = Config.from_yaml(path)
    # Les chemins relatifs se resolvent par rapport a la racine du projet.
    assert cfg.state_path == tmp_path / "state/portfolio.json"
    assert cfg.model_path == tmp_path / "models/model.joblib"


def test_missing_config_file_explains_the_fix(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ConfigError, match="Aucune configuration trouvee"):
        load_config(None)
