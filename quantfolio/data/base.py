"""Interface des sources de donnees.

C'est ici que vous branchez votre API. Un fournisseur doit uniquement savoir
renvoyer un historique OHLCV journalier pour un ticker donne ; tout le reste
(normalisation, cache, alignement des calendriers) est gere par l'outil.
"""

from __future__ import annotations

import abc
import importlib
from typing import Any

import pandas as pd

# Colonnes attendues en sortie d'un fournisseur, dans cet ordre.
OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]

# Alias tolerees pour les en-tetes de colonnes (CSV, API tierces...).
_COLUMN_ALIASES = {
    "date": "date",
    "datetime": "date",
    "timestamp": "date",
    "time": "date",
    "open": "open",
    "o": "open",
    "high": "high",
    "h": "high",
    "low": "low",
    "l": "low",
    "close": "close",
    "c": "close",
    "adj close": "close",
    "adj_close": "close",
    "adjclose": "close",
    "close_adj": "close",
    "price": "close",
    "volume": "volume",
    "v": "volume",
    "vol": "volume",
}


class DataError(RuntimeError):
    """Erreur de recuperation ou de format des donnees."""


class DataProvider(abc.ABC):
    """Source d'historiques de prix.

    Pour brancher votre propre API, sous-classez et implementez `get_history`,
    puis renseignez dans la config :

        data:
          provider: custom
          custom: "mon_module:MonProvider"
          custom_args: {api_key: "..."}
    """

    name = "base"

    @abc.abstractmethod
    def get_history(
        self, ticker: str, start: pd.Timestamp, end: pd.Timestamp
    ) -> pd.DataFrame:
        """Renvoie l'historique journalier de `ticker` entre `start` et `end`.

        Le DataFrame attendu est indexe par date (croissante) et contient les
        colonnes open, high, low, close, volume. Les prix doivent etre ajustes
        des splits et dividendes, sinon les rendements calcules seront faux.
        Renvoyer un DataFrame vide si le titre est inconnu.
        """

    def get_many(
        self, tickers: list[str], start: pd.Timestamp, end: pd.Timestamp
    ) -> dict[str, pd.DataFrame]:
        """Recupere plusieurs titres. Surchargez si votre API fait du batch."""
        out: dict[str, pd.DataFrame] = {}
        for ticker in tickers:
            try:
                out[ticker] = normalize_ohlcv(self.get_history(ticker, start, end))
            except Exception as exc:  # pragma: no cover - depend du provider
                raise DataError(
                    f"Echec de la recuperation de {ticker} via {self.name} : {exc}"
                ) from exc
        return out


def normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise un historique brut au format attendu par l'outil.

    Tolere les variantes d'en-tetes, une colonne date ou un index de dates,
    et complete open/high/low a partir de close si elles manquent.
    """
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=OHLCV_COLUMNS, index=pd.DatetimeIndex([], name="date"))

    df = df.copy()
    df.columns = [_COLUMN_ALIASES.get(str(c).strip().lower(), str(c).strip().lower()) for c in df.columns]

    if "date" in df.columns:
        index = pd.to_datetime(df["date"], errors="coerce", utc=True)
        df = df.drop(columns=[c for c in df.columns if c == "date"])
    else:
        index = pd.to_datetime(pd.Index(df.index), errors="coerce", utc=True)

    # On travaille en dates naives (une seance = un jour), sans fuseau.
    df.index = pd.DatetimeIndex(index).tz_localize(None).normalize()
    df.index.name = "date"

    if "close" not in df.columns:
        raise DataError(
            "Colonne de cloture absente. Colonnes recues : " f"{sorted(set(df.columns))}"
        )

    for col in ("open", "high", "low"):
        if col not in df.columns:
            df[col] = df["close"]
    if "volume" not in df.columns:
        df["volume"] = float("nan")

    df = df[OHLCV_COLUMNS]
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df[df.index.notna()]
    df = df[df["close"].notna() & (df["close"] > 0)]
    df = df[~df.index.duplicated(keep="last")].sort_index()

    # Coherence des bornes : certaines API renvoient un high < close.
    df["high"] = df[["high", "low", "open", "close"]].max(axis=1)
    df["low"] = df[["high", "low", "open", "close"]].min(axis=1)
    return df


def load_custom_provider(spec: str, kwargs: dict[str, Any] | None = None) -> DataProvider:
    """Instancie un fournisseur utilisateur depuis "module:Classe"."""
    if ":" not in spec:
        raise DataError(
            f"Specification de provider invalide : {spec!r}. "
            'Format attendu : "mon_module:MaClasse".'
        )
    module_name, attr = spec.split(":", 1)
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise DataError(
            f"Impossible d'importer le module {module_name!r} : {exc}. "
            "Verifiez qu'il est sur le PYTHONPATH."
        ) from exc
    try:
        obj = getattr(module, attr)
    except AttributeError as exc:
        raise DataError(f"{module_name!r} ne definit pas {attr!r}.") from exc

    provider = obj(**(kwargs or {})) if isinstance(obj, type) else obj
    if not hasattr(provider, "get_history"):
        raise DataError(
            f"{spec} n'expose pas de methode get_history(ticker, start, end)."
        )
    return provider
