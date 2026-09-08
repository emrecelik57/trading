"""Fixtures communes : series de cours synthetiques, sans acces reseau."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from volatrade.data import Quote


def make_frame(
    closes,
    *,
    start: str = "2023-01-02",
    amplitude: float = 0.02,
    volume: float = 5e6,
) -> pd.DataFrame:
    """Construit un tableau OHLCV coherent a partir d'une serie de clotures."""
    closes = np.asarray(closes, dtype=float)
    index = pd.bdate_range(start=start, periods=len(closes), tz="UTC")
    opens = np.concatenate([[closes[0]], closes[:-1]])
    highs = np.maximum(opens, closes) * (1 + amplitude)
    lows = np.minimum(opens, closes) * (1 - amplitude)
    return pd.DataFrame(
        {
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": np.full(len(closes), volume),
        },
        index=index,
    )


def make_quote(ticker: str, closes, **kwargs) -> Quote:
    """Enveloppe `make_frame` dans un `Quote`."""
    return Quote(ticker=ticker.upper(), currency="USD", exchange="TEST",
                 frame=make_frame(closes, **kwargs))


def trending_prices(n: int = 400, drift: float = 0.0025, noise: float = 0.02, seed: int = 7):
    """Serie haussiere bruitee."""
    rng = np.random.default_rng(seed)
    steps = drift + noise * rng.standard_normal(n)
    return 50.0 * np.exp(np.cumsum(steps))


def falling_prices(n: int = 400, drift: float = -0.004, noise: float = 0.03, seed: int = 11):
    """Serie baissiere bruitee (le cas 'couteau qui tombe')."""
    rng = np.random.default_rng(seed)
    steps = drift + noise * rng.standard_normal(n)
    return 120.0 * np.exp(np.cumsum(steps))


def flat_prices(n: int = 400, level: float = 25.0, noise: float = 0.015, seed: int = 3):
    """Serie sans tendance."""
    rng = np.random.default_rng(seed)
    return level * np.exp(np.cumsum(noise * rng.standard_normal(n)) * 0.2)


@pytest.fixture
def uptrend_frame() -> pd.DataFrame:
    return make_frame(trending_prices())


@pytest.fixture
def downtrend_frame() -> pd.DataFrame:
    return make_frame(falling_prices())


@pytest.fixture
def flat_frame() -> pd.DataFrame:
    return make_frame(flat_prices())


@pytest.fixture
def quotes(uptrend_frame, downtrend_frame, flat_frame) -> dict[str, Quote]:
    return {
        "HAUT": Quote("HAUT", "USD", "TEST", uptrend_frame),
        "BAS": Quote("BAS", "USD", "TEST", downtrend_frame),
        "PLAT": Quote("PLAT", "USD", "TEST", flat_frame),
    }


@pytest.fixture
def benchmark_returns(flat_frame) -> pd.Series:
    from volatrade.metrics import log_returns

    return log_returns(flat_frame["close"])
