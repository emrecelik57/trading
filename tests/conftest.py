"""Fixtures partagees."""

from __future__ import annotations

import pandas as pd
import pytest

from quantfolio.config import Config
from quantfolio.data.panel import build_panel
from quantfolio.data.providers import SyntheticProvider


def make_config(tickers: list[str], **overrides) -> Config:
    """Config minimale sur donnees synthetiques, sans reseau ni cache."""
    payload: dict = {
        "universe": {"tickers": tickers, "benchmark": None},
        "data": {
            "provider": "synthetic",
            "start": "2015-01-01",
            "end": "2022-12-31",
            "use_cache": False,
        },
        "model": {"kind": "gbm", "min_train_rows": 500},
        "backtest": {"start": "2019-01-01"},
    }
    for section, values in overrides.items():
        payload.setdefault(section, {}).update(values)
    return Config.from_dict(payload)


@pytest.fixture
def tickers() -> list[str]:
    return [f"T{i:02d}" for i in range(12)]


@pytest.fixture
def cfg(tickers) -> Config:
    return make_config(tickers)


@pytest.fixture
def panel(cfg, tickers):
    provider = SyntheticProvider(seed=1)
    histories = provider.get_many(
        tickers, pd.Timestamp(cfg.data.start), pd.Timestamp(cfg.data.end)
    )
    return build_panel(histories, cfg, tickers=tickers)
