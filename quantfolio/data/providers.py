"""Fournisseurs de donnees fournis en standard."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

from .base import DataError, DataProvider, normalize_ohlcv


class CsvProvider(DataProvider):
    """Lit des fichiers `<repertoire>/<TICKER>.csv`.

    Le CSV doit contenir une colonne de date et une colonne de cloture ajustee ;
    open/high/low/volume sont optionnels. Les en-tetes usuels sont reconnus
    (Date, Adj Close, Volume...).
    """

    name = "csv"

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def _file_for(self, ticker: str) -> Path | None:
        for candidate in (
            self.path / f"{ticker}.csv",
            self.path / f"{ticker.upper()}.csv",
            self.path / f"{ticker.lower()}.csv",
        ):
            if candidate.exists():
                return candidate
        return None

    def get_history(
        self, ticker: str, start: pd.Timestamp, end: pd.Timestamp
    ) -> pd.DataFrame:
        file = self._file_for(ticker)
        if file is None:
            raise DataError(
                f"Aucun fichier pour {ticker} dans {self.path} "
                f"(attendu : {self.path / (ticker + '.csv')})."
            )
        df = normalize_ohlcv(pd.read_csv(file))
        return df.loc[(df.index >= start) & (df.index <= end)]


class YahooProvider(DataProvider):
    """Implementation de reference sur l'API publique de Yahoo Finance.

    Sans dependance externe (urllib). Pratique pour demarrer, mais non
    contractuelle : pour un usage serieux, branchez votre propre fournisseur.
    """

    name = "yahoo"
    URL = "https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"

    def __init__(self, timeout: float = 20.0):
        self.timeout = timeout

    def get_history(
        self, ticker: str, start: pd.Timestamp, end: pd.Timestamp
    ) -> pd.DataFrame:
        params = (
            f"?period1={int(pd.Timestamp(start).timestamp())}"
            f"&period2={int(pd.Timestamp(end).timestamp()) + 86400}"
            "&interval=1d&events=div%2Csplit"
        )
        url = self.URL.format(ticker=ticker) + params
        request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, ValueError) as exc:
            raise DataError(f"Appel Yahoo echoue pour {ticker} : {exc}") from exc

        chart = (payload or {}).get("chart") or {}
        if chart.get("error"):
            raise DataError(f"Yahoo a renvoye une erreur pour {ticker} : {chart['error']}")
        results = chart.get("result") or []
        if not results:
            return normalize_ohlcv(pd.DataFrame())

        result = results[0]
        timestamps = result.get("timestamp") or []
        quote = (result.get("indicators", {}).get("quote") or [{}])[0]
        frame = pd.DataFrame(
            {
                "date": pd.to_datetime(timestamps, unit="s", utc=True),
                "open": quote.get("open"),
                "high": quote.get("high"),
                "low": quote.get("low"),
                "close": quote.get("close"),
                "volume": quote.get("volume"),
            }
        )
        # Prix ajustes : on rescale l'OHLC par le ratio adjclose / close.
        adj = (result.get("indicators", {}).get("adjclose") or [{}])[0].get("adjclose")
        if adj is not None and len(adj) == len(frame):
            adjclose = pd.Series(adj, dtype="float64")
            ratio = (adjclose / frame["close"]).replace([np.inf, -np.inf], np.nan)
            ratio = ratio.ffill().fillna(1.0)
            for col in ("open", "high", "low", "close"):
                frame[col] = frame[col] * ratio
        return normalize_ohlcv(frame)


class SyntheticProvider(DataProvider):
    """Genere des historiques simules, reproductibles, sans reseau.

    Utile pour tester l'outil, faire tourner la demo et les tests unitaires.
    Les series partagent un facteur de marche commun, ont des tendances et des
    volatilites differentes : de quoi verifier que le modele sait classer.
    """

    name = "synthetic"

    def __init__(
        self,
        seed: int = 42,
        annual_drift: float = 0.06,
        # Dispersion des tendances propres a chaque titre, en rythme annuel.
        # C'est elle qui rend (ou non) le classement previsible : plus elle est
        # forte devant la volatilite, plus le momentum porte de l'information.
        drift_dispersion: float = 0.10,
        vol_range: tuple[float, float] = (0.012, 0.028),
        market_vol: float = 0.009,
    ):
        self.seed = seed
        self.annual_drift = annual_drift
        self.drift_dispersion = drift_dispersion
        self.vol_range = vol_range
        self.market_vol = market_vol

    def get_history(
        self, ticker: str, start: pd.Timestamp, end: pd.Timestamp
    ) -> pd.DataFrame:
        dates = pd.bdate_range(start=start, end=end)
        if len(dates) == 0:
            return normalize_ohlcv(pd.DataFrame())

        # Un tirage propre a chaque titre, plus un facteur de marche commun.
        # zlib.crc32 plutot que hash() : ce dernier est randomise a chaque
        # processus, les series ne seraient pas reproductibles d'un run a l'autre.
        ticker_seed = zlib.crc32(f"{ticker}:{self.seed}".encode("utf-8"))
        rng = np.random.default_rng(ticker_seed)
        market_rng = np.random.default_rng(self.seed)
        n = len(dates)

        market = market_rng.normal(0.0003, self.market_vol, n)
        beta = rng.uniform(0.6, 1.4)
        vol = rng.uniform(*self.vol_range)
        drift = self.annual_drift / 252 + rng.normal(0, self.drift_dispersion / 252)

        # Une composante lente et persistante rend le momentum exploitable.
        slow = pd.Series(rng.normal(0, 1, n)).ewm(span=60).mean().to_numpy()
        slow = slow / (np.std(slow) + 1e-9) * 0.0012

        returns = drift + beta * market + slow + rng.normal(0, vol, n)
        close = 50.0 * np.exp(np.cumsum(returns))
        noise = np.abs(rng.normal(0, 0.004, n))
        frame = pd.DataFrame(
            {
                "date": dates,
                "open": close * (1 + rng.normal(0, 0.002, n)),
                "high": close * (1 + noise),
                "low": close * (1 - noise),
                "close": close,
                "volume": rng.lognormal(14, 0.4, n).round(),
            }
        )
        return normalize_ohlcv(frame)
