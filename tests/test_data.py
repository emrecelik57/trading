"""Normalisation des sources de donnees et alignement des calendriers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantfolio.data.base import DataError, load_custom_provider, normalize_ohlcv
from quantfolio.data.panel import build_panel
from quantfolio.data.providers import CsvProvider, SyntheticProvider

from conftest import make_config


def test_normalize_accepts_common_header_variants():
    raw = pd.DataFrame(
        {
            "Date": ["2024-01-02", "2024-01-03"],
            "Open": [10.0, 11.0],
            "High": [11.0, 12.0],
            "Low": [9.0, 10.0],
            "Adj Close": [10.5, 11.5],
            "Volume": [1000, 2000],
        }
    )
    frame = normalize_ohlcv(raw)
    assert list(frame.columns) == ["open", "high", "low", "close", "volume"]
    assert frame.index[0] == pd.Timestamp("2024-01-02")
    assert frame["close"].iloc[1] == pytest.approx(11.5)


def test_normalize_fills_missing_ohlc_from_close():
    raw = pd.DataFrame({"date": ["2024-01-02"], "close": [42.0]})
    frame = normalize_ohlcv(raw)
    assert frame["open"].iloc[0] == pytest.approx(42.0)
    assert frame["high"].iloc[0] == pytest.approx(42.0)
    assert np.isnan(frame["volume"].iloc[0])


def test_normalize_requires_a_close_column():
    with pytest.raises(DataError, match="cloture"):
        normalize_ohlcv(pd.DataFrame({"date": ["2024-01-02"], "truc": [1.0]}))


def test_normalize_sorts_deduplicates_and_drops_invalid_rows():
    raw = pd.DataFrame(
        {
            "date": ["2024-01-03", "2024-01-02", "2024-01-03", "2024-01-04"],
            "close": [11.0, 10.0, 12.0, None],
        }
    )
    frame = normalize_ohlcv(raw)
    assert frame.index.is_monotonic_increasing
    assert len(frame) == 2
    assert frame["close"].loc[pd.Timestamp("2024-01-03")] == pytest.approx(12.0)


def test_normalize_repairs_inconsistent_bounds():
    raw = pd.DataFrame(
        {"date": ["2024-01-02"], "open": [10.0], "high": [9.0], "low": [11.0], "close": [12.0]}
    )
    frame = normalize_ohlcv(raw)
    assert frame["high"].iloc[0] >= frame["close"].iloc[0]
    assert frame["low"].iloc[0] <= frame["close"].iloc[0]


def test_normalize_handles_an_empty_frame():
    frame = normalize_ohlcv(pd.DataFrame())
    assert frame.empty
    assert list(frame.columns) == ["open", "high", "low", "close", "volume"]


def test_csv_provider_reads_a_directory(tmp_path):
    frame = pd.DataFrame(
        {"date": pd.bdate_range("2024-01-01", periods=10).strftime("%Y-%m-%d"),
         "close": np.linspace(10, 20, 10)}
    )
    frame.to_csv(tmp_path / "ABC.csv", index=False)
    provider = CsvProvider(tmp_path)
    history = provider.get_history("ABC", pd.Timestamp("2024-01-01"), pd.Timestamp("2024-12-31"))
    assert len(history) == 10

    with pytest.raises(DataError, match="Aucun fichier"):
        provider.get_history("ZZZ", pd.Timestamp("2024-01-01"), pd.Timestamp("2024-12-31"))


def test_synthetic_provider_is_reproducible():
    """Deux instances identiques doivent produire exactement la meme serie."""
    args = (pd.Timestamp("2020-01-01"), pd.Timestamp("2020-12-31"))
    first = SyntheticProvider(seed=7).get_history("AAPL", *args)
    second = SyntheticProvider(seed=7).get_history("AAPL", *args)
    pd.testing.assert_frame_equal(first, second)

    different = SyntheticProvider(seed=8).get_history("AAPL", *args)
    assert not first["close"].equals(different["close"])


def test_panel_aligns_different_calendars():
    """Un titre cotant moins souvent ne doit pas trouer le panel."""
    cfg = make_config(["A", "B"])
    full = pd.bdate_range("2024-01-01", periods=10)
    histories = {
        "A": normalize_ohlcv(pd.DataFrame({"date": full, "close": np.arange(10.0) + 100})),
        "B": normalize_ohlcv(pd.DataFrame({"date": full[::2], "close": np.arange(5.0) + 50})),
    }
    panel = build_panel(histories, cfg, tickers=["A", "B"])

    assert len(panel) == 10
    assert panel.close["B"].notna().all()  # prix propages
    # B cote un jour sur deux : il reste disponible grace a la tolerance.
    assert panel.available["B"].sum() == 10


def test_panel_marks_a_delisted_ticker_unavailable():
    cfg = make_config(["A", "B"], data={"max_staleness": 2})
    full = pd.bdate_range("2024-01-01", periods=20)
    histories = {
        "A": normalize_ohlcv(pd.DataFrame({"date": full, "close": np.arange(20.0) + 100})),
        # B s'arrete de coter apres 5 seances.
        "B": normalize_ohlcv(pd.DataFrame({"date": full[:5], "close": np.arange(5.0) + 50})),
    }
    panel = build_panel(histories, cfg, tickers=["A", "B"])
    assert panel.available["B"].iloc[:5].all()
    assert not panel.available["B"].iloc[-1]
    assert panel.available["A"].all()


def test_panel_marks_a_late_listing_unavailable_before_ipo():
    cfg = make_config(["A", "B"])
    full = pd.bdate_range("2024-01-01", periods=20)
    histories = {
        "A": normalize_ohlcv(pd.DataFrame({"date": full, "close": np.arange(20.0) + 100})),
        "B": normalize_ohlcv(pd.DataFrame({"date": full[10:], "close": np.arange(10.0) + 50})),
    }
    panel = build_panel(histories, cfg, tickers=["A", "B"])
    assert not panel.available["B"].iloc[:10].any()
    assert panel.available["B"].iloc[10:].all()


def test_custom_provider_specification_is_validated():
    with pytest.raises(DataError, match="Format attendu"):
        load_custom_provider("mon_module_sans_deux_points")
    with pytest.raises(DataError, match="Impossible d'importer"):
        load_custom_provider("module_qui_nexiste_pas:Classe")


def test_custom_provider_loads_a_user_class():
    provider = load_custom_provider(
        "quantfolio.data.providers:SyntheticProvider", {"seed": 11}
    )
    history = provider.get_history("X", pd.Timestamp("2024-01-01"), pd.Timestamp("2024-03-01"))
    assert len(history) > 30
