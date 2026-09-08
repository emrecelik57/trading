"""Lecture des reponses du fournisseur de cours et cache disque."""

from __future__ import annotations

import json

import pytest

from volatrade import data
from volatrade.data import DataError, close_panel, parse_chart_payload


def payload(closes, adjcloses=None, timestamps=None) -> dict:
    timestamps = timestamps or [1700000000 + 86400 * i for i in range(len(closes))]
    indicators = {
        "quote": [
            {
                "open": list(closes),
                "high": [value * 1.02 for value in closes],
                "low": [value * 0.98 for value in closes],
                "close": list(closes),
                "volume": [1_000_000] * len(closes),
            }
        ]
    }
    if adjcloses is not None:
        indicators["adjclose"] = [{"adjclose": list(adjcloses)}]
    return {
        "chart": {
            "result": [
                {
                    "meta": {"currency": "USD", "fullExchangeName": "TEST"},
                    "timestamp": timestamps,
                    "indicators": indicators,
                }
            ],
            "error": None,
        }
    }


def test_parse_lit_les_colonnes_et_les_metadonnees():
    quote = parse_chart_payload("test", payload([10.0, 11.0, 12.0]))
    assert quote.ticker == "TEST"
    assert quote.currency == "USD"
    assert len(quote) == 3
    assert quote.last_price == pytest.approx(12.0)
    assert list(quote.frame.columns) == ["open", "high", "low", "close", "volume"]


def test_parse_applique_le_facteur_dajustement_a_tout_lohlc():
    # Split 2:1 : les clotures ajustees valent la moitie des clotures brutes.
    quote = parse_chart_payload("test", payload([100.0, 100.0], [50.0, 50.0]))
    assert quote.frame["close"].iloc[0] == pytest.approx(50.0)
    assert quote.frame["high"].iloc[0] == pytest.approx(51.0)
    assert quote.frame["low"].iloc[0] == pytest.approx(49.0)


def test_parse_signale_une_erreur_du_fournisseur():
    with pytest.raises(DataError, match="inconnu"):
        parse_chart_payload("XXXX", {"chart": {"error": {"description": "ticker inconnu"}}})


def test_parse_refuse_une_reponse_vide():
    with pytest.raises(DataError):
        parse_chart_payload("XXXX", {"chart": {"result": []}})
    with pytest.raises(DataError, match="aucune barre"):
        parse_chart_payload("XXXX", payload([]))


def test_parse_supprime_les_dates_dupliquees():
    quote = parse_chart_payload("test", payload([10.0, 11.0], timestamps=[1700000000] * 2))
    assert len(quote) == 1
    assert quote.last_price == pytest.approx(11.0)


def test_cache_evite_un_second_appel_reseau(tmp_path, monkeypatch):
    appels = []

    def faux_get(url, timeout, attempts=3):
        appels.append(url)
        return payload([10.0, 11.0, 12.0])

    monkeypatch.setattr(data, "_http_get_json", faux_get)
    premier = data.fetch_history("TEST", cache_dir=tmp_path)
    second = data.fetch_history("TEST", cache_dir=tmp_path)
    assert len(appels) == 1
    assert premier.last_price == second.last_price


def test_cache_expire_selon_le_ttl(tmp_path, monkeypatch):
    appels = []
    monkeypatch.setattr(
        data, "_http_get_json", lambda url, timeout, attempts=3: (appels.append(url), payload([1.0, 2.0]))[1]
    )
    data.fetch_history("TEST", cache_dir=tmp_path, cache_ttl=0)
    data.fetch_history("TEST", cache_dir=tmp_path, cache_ttl=0)
    assert len(appels) == 2


def test_cache_corrompu_est_ignore(tmp_path, monkeypatch):
    chemin = data._cache_path("TEST", "2y", "1d", tmp_path)
    chemin.parent.mkdir(parents=True, exist_ok=True)
    chemin.write_text("{ceci n'est pas du json")
    monkeypatch.setattr(data, "_http_get_json", lambda url, timeout, attempts=3: payload([5.0, 6.0]))
    assert data.fetch_history("TEST", cache_dir=tmp_path).last_price == pytest.approx(6.0)


def test_fetch_many_ignore_les_tickers_en_echec(tmp_path, monkeypatch):
    def faux_get(url, timeout, attempts=3):
        if "BOOM" in url:
            raise DataError("indisponible")
        return payload([10.0, 11.0])

    monkeypatch.setattr(data, "_http_get_json", faux_get)
    erreurs = []
    quotes = data.fetch_many(
        ["OK", "BOOM"], cache_dir=tmp_path, on_error=lambda ticker, exc: erreurs.append(ticker)
    )
    assert set(quotes) == {"OK"}
    assert erreurs == ["BOOM"]


def test_close_panel_aligne_les_series():
    a = parse_chart_payload("A", payload([1.0, 2.0, 3.0]))
    b = parse_chart_payload("B", payload([10.0, 20.0, 30.0]))
    panel = close_panel({"A": a, "B": b})
    assert list(panel.columns) == ["A", "B"]
    assert len(panel) == 3
    assert close_panel({}).empty
