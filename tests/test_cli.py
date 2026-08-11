"""Parcours utilisateur complet, en ligne de commande."""

from __future__ import annotations

import json

import pytest
import yaml

from quantfolio.cli import main
from quantfolio.state import PortfolioState
from quantfolio.templates import DEFAULT_CONFIG_YAML


@pytest.fixture
def project(tmp_path):
    """Un projet jetable sur donnees synthetiques."""
    payload = yaml.safe_load(DEFAULT_CONFIG_YAML)
    payload["universe"] = {"tickers": [f"T{i:02d}" for i in range(12)], "benchmark": None}
    payload["data"].update(
        {"provider": "synthetic", "start": "2015-01-01", "end": "2022-12-31", "use_cache": False}
    )
    payload["model"].update({"kind": "rules"})
    payload["backtest"].update({"start": "2021-01-01", "rebalance_days": 10})
    payload["portfolio"].update({"capital": 50_000})

    directory = tmp_path / "config"
    directory.mkdir()
    path = directory / "config.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return tmp_path, str(path)


def test_init_writes_a_usable_config(tmp_path, capsys):
    target = tmp_path / "config" / "config.yaml"
    assert main(["init", "--output", str(target)]) == 0
    assert target.exists()

    # Sans --force, on ne doit pas ecraser un fichier existant.
    assert main(["init", "--output", str(target)]) == 1
    assert "existe deja" in capsys.readouterr().out


def test_fetch_reports_the_universe(project, capsys):
    _, config_path = project
    assert main(["--config", config_path, "fetch"]) == 0
    assert "12 titres" in capsys.readouterr().out


def test_full_daily_workflow(project, capsys):
    """init-portfolio -> daily -> apply -> portfolio."""
    root, config_path = project

    assert main(["--config", config_path, "init-portfolio", "--cash", "50000"]) == 0
    assert (root / "state" / "portfolio.json").exists()

    assert main(["--config", config_path, "daily"]) == 0
    output = capsys.readouterr().out
    assert "ORDRES A PASSER" in output
    assert "PORTEFEUILLE CIBLE" in output

    orders_files = sorted((root / "output").glob("orders_*.json"))
    assert len(orders_files) == 1
    plan = json.loads(orders_files[0].read_text(encoding="utf-8"))
    assert plan["orders"], "le premier jour doit proposer des achats"
    assert all(order["side"] == "BUY" for order in plan["orders"])

    # Enregistrement de l'execution.
    assert main(["--config", config_path, "apply", "--file", str(orders_files[0])]) == 0
    state = PortfolioState.load(root / "state" / "portfolio.json")
    assert state.positions
    assert state.cash < 50_000

    # Le portefeuille detenu s'affiche et se valorise.
    assert main(["--config", config_path, "portfolio"]) == 0
    assert "PORTEFEUILLE ACTUEL" in capsys.readouterr().out

    # Relancer daily sur le meme jour ne doit plus rien proposer :
    # le portefeuille est deja conforme a la cible.
    assert main(["--config", config_path, "daily"]) == 0
    assert "Aucun ordre" in capsys.readouterr().out


def test_apply_dry_run_changes_nothing(project):
    root, config_path = project
    main(["--config", config_path, "init-portfolio", "--cash", "50000"])
    main(["--config", config_path, "daily"])
    orders_file = sorted((root / "output").glob("orders_*.json"))[0]

    main(["--config", config_path, "apply", "--file", str(orders_file), "--dry-run"])
    state = PortfolioState.load(root / "state" / "portfolio.json")
    assert state.cash == 50_000
    assert not state.positions


def test_set_position_declares_an_existing_holding(project):
    root, config_path = project
    main(["--config", config_path, "init-portfolio", "--cash", "10000"])
    assert (
        main([
            "--config", config_path, "set-position", "T00",
            "--shares", "25", "--price", "42.5", "--cash", "7000",
        ])
        == 0
    )
    state = PortfolioState.load(root / "state" / "portfolio.json")
    assert state.positions["T00"].shares == 25
    assert state.cash == 7000


def test_backtest_writes_its_outputs(project, capsys):
    root, config_path = project
    assert main(["--config", config_path, "backtest"]) == 0
    output = capsys.readouterr().out
    assert "PERFORMANCE" in output
    assert (root / "output" / "backtest_equity.csv").exists()
    assert (root / "output" / "backtest_summary.json").exists()

    summary = json.loads((root / "output" / "backtest_summary.json").read_text())
    assert "sharpe" in summary


def test_daily_accepts_a_past_date(project, capsys):
    root, config_path = project
    main(["--config", config_path, "init-portfolio", "--cash", "50000"])
    assert main(["--config", config_path, "daily", "--date", "2022-06-15"]) == 0
    assert "SEANCE DU 2022-06-15" in capsys.readouterr().out
    assert (root / "output" / "orders_2022-06-15.json").exists()


def test_portfolio_without_state_explains_the_fix(project, capsys):
    _, config_path = project
    assert main(["--config", config_path, "portfolio"]) == 1
    assert "init-portfolio" in capsys.readouterr().out


def test_invalid_config_returns_an_error_code(tmp_path, capsys):
    directory = tmp_path / "config"
    directory.mkdir()
    path = directory / "config.yaml"
    path.write_text("universe:\n  tickers: []\n", encoding="utf-8")
    assert main(["--config", str(path), "fetch"]) == 2
    assert "Erreur de configuration" in capsys.readouterr().err


def test_train_saves_a_model(project, capsys):
    root, config_path = project
    payload = yaml.safe_load((root / "config" / "config.yaml").read_text())
    payload["model"].update({"kind": "gbm", "min_train_rows": 500})
    (root / "config" / "config.yaml").write_text(yaml.safe_dump(payload), encoding="utf-8")

    assert main(["--config", config_path, "train"]) == 0
    assert (root / "models" / "model.joblib").exists()
    assert "Modele sauvegarde" in capsys.readouterr().out
