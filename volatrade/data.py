"""Recuperation des cours de bourse (API publique Yahoo Finance) avec cache disque.

Le module n'utilise que la bibliotheque standard pour le reseau : aucune cle
d'API n'est necessaire. Les reponses sont mises en cache sur disque afin que
les executions repetees (et les tests) ne saturent pas le fournisseur.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
USER_AGENT = "Mozilla/5.0 (compatible; volatrade/1.0)"
DEFAULT_CACHE_DIR = Path(os.environ.get("VOLATRADE_CACHE", ".volatrade_cache"))
DEFAULT_TTL_SECONDS = 6 * 3600


class DataError(RuntimeError):
    """Erreur de recuperation ou de qualite des donnees pour un ticker."""


@dataclass(frozen=True)
class Quote:
    """Historique de cours d'un titre.

    `frame` contient les colonnes open/high/low/close/volume ajustees des
    splits et dividendes, indexees par date (UTC, tri croissant).
    """

    ticker: str
    currency: str
    exchange: str
    frame: pd.DataFrame

    @property
    def close(self) -> pd.Series:
        return self.frame["close"]

    @property
    def last_price(self) -> float:
        return float(self.frame["close"].iloc[-1])

    def __len__(self) -> int:
        return len(self.frame)


def _cache_path(ticker: str, period: str, interval: str, cache_dir: Path) -> Path:
    safe = ticker.replace("/", "_").replace("^", "idx_")
    return cache_dir / f"{safe}__{period}__{interval}.json"


def _read_cache(path: Path, ttl: float) -> dict | None:
    if ttl <= 0 or not path.exists():
        return None
    if time.time() - path.stat().st_mtime > ttl:
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def _write_cache(path: Path, payload: dict) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload))
    except OSError:
        pass  # un cache indisponible ne doit jamais faire echouer une analyse


def _http_get_json(url: str, timeout: float, attempts: int = 3) -> dict:
    last_error: Exception | None = None
    for attempt in range(attempts):
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_error = exc
            if attempt < attempts - 1:
                time.sleep(2 ** attempt)
    raise DataError(f"echec de la requete {url} : {last_error}")


def parse_chart_payload(ticker: str, payload: dict) -> Quote:
    """Convertit une reponse brute de l'API chart en `Quote` ajuste."""
    chart = payload.get("chart") or {}
    if chart.get("error"):
        raise DataError(f"{ticker}: {chart['error'].get('description', chart['error'])}")
    results = chart.get("result") or []
    if not results:
        raise DataError(f"{ticker}: reponse vide du fournisseur")

    result = results[0]
    timestamps = result.get("timestamp") or []
    if not timestamps:
        raise DataError(f"{ticker}: aucune barre de cotation renvoyee")

    quote = (result.get("indicators", {}).get("quote") or [{}])[0]
    frame = pd.DataFrame(
        {
            "open": quote.get("open"),
            "high": quote.get("high"),
            "low": quote.get("low"),
            "close": quote.get("close"),
            "volume": quote.get("volume"),
        },
        index=pd.to_datetime(timestamps, unit="s", utc=True).normalize(),
    ).astype(float)

    adj_block = result.get("indicators", {}).get("adjclose") or []
    if adj_block and adj_block[0].get("adjclose"):
        adj_close = pd.Series(adj_block[0]["adjclose"], index=frame.index, dtype=float)
        # Le facteur d'ajustement s'applique aussi a open/high/low pour que
        # l'ATR et les plus-hauts restent coherents apres un split.
        factor = (adj_close / frame["close"]).replace([np.inf, -np.inf], np.nan).ffill().fillna(1.0)
        for column in ("open", "high", "low", "close"):
            frame[column] = frame[column] * factor

    frame = frame.dropna(subset=["close"])
    frame = frame[~frame.index.duplicated(keep="last")].sort_index()
    if frame.empty:
        raise DataError(f"{ticker}: aucune cloture exploitable")

    meta = result.get("meta", {})
    return Quote(
        ticker=ticker.upper(),
        currency=meta.get("currency", "USD"),
        exchange=meta.get("fullExchangeName", meta.get("exchangeName", "")),
        frame=frame,
    )


def fetch_history(
    ticker: str,
    period: str = "2y",
    interval: str = "1d",
    *,
    cache_dir: Path | str = DEFAULT_CACHE_DIR,
    cache_ttl: float = DEFAULT_TTL_SECONDS,
    timeout: float = 20.0,
) -> Quote:
    """Telecharge (ou relit depuis le cache) l'historique d'un titre."""
    cache_dir = Path(cache_dir)
    path = _cache_path(ticker, period, interval, cache_dir)
    payload = _read_cache(path, cache_ttl)
    if payload is None:
        url = f"{CHART_URL.format(ticker=urllib.parse.quote(ticker))}?range={period}&interval={interval}&events=div%2Csplit"
        payload = _http_get_json(url, timeout=timeout)
        _write_cache(path, payload)
    return parse_chart_payload(ticker, payload)


def fetch_many(
    tickers: list[str],
    period: str = "2y",
    interval: str = "1d",
    *,
    cache_dir: Path | str = DEFAULT_CACHE_DIR,
    cache_ttl: float = DEFAULT_TTL_SECONDS,
    timeout: float = 20.0,
    on_error=None,
) -> dict[str, Quote]:
    """Telecharge plusieurs titres ; un ticker en echec est ignore, pas fatal."""
    quotes: dict[str, Quote] = {}
    for ticker in tickers:
        try:
            quotes[ticker.upper()] = fetch_history(
                ticker,
                period,
                interval,
                cache_dir=cache_dir,
                cache_ttl=cache_ttl,
                timeout=timeout,
            )
        except DataError as exc:
            if on_error is not None:
                on_error(ticker, exc)
    return quotes


def close_panel(quotes: dict[str, Quote]) -> pd.DataFrame:
    """Assemble les clotures ajustees en un tableau colonnes = tickers."""
    if not quotes:
        return pd.DataFrame()
    panel = pd.DataFrame({ticker: quote.close for ticker, quote in quotes.items()})
    return panel.sort_index()
