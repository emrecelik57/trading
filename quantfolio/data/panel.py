"""Chargement et alignement des historiques en tableaux date x ticker."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ..config import Config
from .base import DataError, DataProvider, load_custom_provider, normalize_ohlcv
from .providers import CsvProvider, SyntheticProvider, YahooProvider


@dataclass
class Panel:
    """Historiques alignes sur un calendrier commun.

    Chaque attribut est un DataFrame indexe par date, avec un ticker par colonne.
    `available` indique, pour chaque case, si le titre cotait reellement ce
    jour-la : les prix sont propages (ffill) pour eviter les trous dans les
    indicateurs, mais on ne doit jamais traiter un titre non disponible.
    """

    open: pd.DataFrame
    high: pd.DataFrame
    low: pd.DataFrame
    close: pd.DataFrame
    volume: pd.DataFrame
    available: pd.DataFrame

    @property
    def tickers(self) -> list[str]:
        return list(self.close.columns)

    @property
    def dates(self) -> pd.DatetimeIndex:
        return pd.DatetimeIndex(self.close.index)

    def __len__(self) -> int:
        return len(self.close.index)

    def returns(self) -> pd.DataFrame:
        """Rendements journaliers simples."""
        return self.close.div(self.close.shift(1)) - 1.0

    def slice(self, start=None, end=None) -> "Panel":
        mask = pd.Series(True, index=self.close.index)
        if start is not None:
            mask &= self.close.index >= pd.Timestamp(start)
        if end is not None:
            mask &= self.close.index <= pd.Timestamp(end)
        return Panel(**{
            field: getattr(self, field).loc[mask.to_numpy()]
            for field in ("open", "high", "low", "close", "volume", "available")
        })

    def last_prices(self) -> pd.Series:
        """Derniere cloture connue par titre."""
        return self.close.ffill().iloc[-1]


def build_provider(cfg: Config) -> DataProvider:
    """Instancie le fournisseur de donnees decrit par la configuration."""
    kind = cfg.data.provider
    if kind == "csv":
        return CsvProvider(cfg.resolve(cfg.data.path))
    if kind == "yahoo":
        return YahooProvider()
    if kind == "synthetic":
        return SyntheticProvider(**cfg.data.provider_args)
    if kind == "custom":
        return load_custom_provider(cfg.data.custom or "", cfg.data.provider_args)
    raise DataError(f"Fournisseur inconnu : {kind!r}")


def _cache_file(cfg: Config, ticker: str) -> Path:
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in ticker)
    return cfg.resolve(cfg.data.cache_dir) / cfg.data.provider / f"{safe}.csv"


def _read_cache(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        return normalize_ohlcv(pd.read_csv(path))
    except Exception:
        return None


def _write_cache(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    out.insert(0, "date", out.index.strftime("%Y-%m-%d"))
    out.to_csv(path, index=False)


def fetch_histories(
    cfg: Config,
    provider: DataProvider | None = None,
    refresh: bool = False,
    verbose: bool = False,
) -> dict[str, pd.DataFrame]:
    """Recupere les historiques de l'univers (+ benchmark), avec cache disque."""
    provider = provider or build_provider(cfg)
    start = pd.Timestamp(cfg.data.start)
    end = pd.Timestamp(cfg.data.end) if cfg.data.end else pd.Timestamp.today().normalize()

    tickers = list(cfg.universe.tickers)
    if cfg.universe.benchmark and cfg.universe.benchmark not in tickers:
        tickers.append(cfg.universe.benchmark)

    histories: dict[str, pd.DataFrame] = {}
    to_fetch: list[str] = []
    for ticker in tickers:
        cached = None if refresh or not cfg.data.use_cache else _read_cache(_cache_file(cfg, ticker))
        # Le cache n'est utilise que s'il couvre la periode demandee.
        if cached is not None and len(cached) and cached.index.max() >= end - pd.Timedelta(days=5):
            histories[ticker] = cached.loc[(cached.index >= start) & (cached.index <= end)]
        else:
            to_fetch.append(ticker)

    if to_fetch:
        if verbose:
            print(f"Telechargement de {len(to_fetch)} titre(s) via {provider.name}...")
        fetched = provider.get_many(to_fetch, start, end)
        for ticker, df in fetched.items():
            if cfg.data.use_cache and len(df):
                _write_cache(_cache_file(cfg, ticker), df)
            histories[ticker] = df

    missing = [t for t, df in histories.items() if df is None or len(df) == 0]
    for ticker in missing:
        histories.pop(ticker, None)
    if missing and verbose:
        print(f"Avertissement : aucune donnee pour {', '.join(sorted(missing))}")
    if not histories:
        raise DataError(
            "Aucune donnee recuperee. Verifiez data.provider, data.path et les "
            "tickers de l'univers."
        )
    return histories


def build_panel(
    histories: dict[str, pd.DataFrame], cfg: Config, tickers: list[str] | None = None
) -> Panel:
    """Aligne des historiques heterogenes sur un calendrier commun."""
    tickers = [t for t in (tickers or sorted(histories)) if t in histories and len(histories[t])]
    if not tickers:
        raise DataError("Aucun titre exploitable pour construire le panel.")

    calendar = pd.DatetimeIndex(sorted(set().union(*[histories[t].index for t in tickers])))
    frames: dict[str, dict[str, pd.Series]] = {f: {} for f in ("open", "high", "low", "close", "volume")}
    available: dict[str, pd.Series] = {}

    for ticker in tickers:
        hist = histories[ticker].reindex(calendar)
        traded = hist["close"].notna()
        # Une ligne reste "disponible" quelques seances apres sa derniere
        # cotation (jour ferie local, donnee manquante), puis devient invalide.
        staleness = _bars_since_true(traded)
        available[ticker] = traded | (staleness <= cfg.data.max_staleness)
        # Avant la premiere cotation, rien n'est disponible.
        first = traded.idxmax() if traded.any() else None
        if first is not None:
            available[ticker] &= hist.index >= first
        else:
            available[ticker] = pd.Series(False, index=calendar)

        filled = hist.ffill()
        for field in ("open", "high", "low", "close"):
            frames[field][ticker] = filled[field]
        frames["volume"][ticker] = hist["volume"].fillna(0.0)

    panel = Panel(
        open=pd.DataFrame(frames["open"], index=calendar)[tickers],
        high=pd.DataFrame(frames["high"], index=calendar)[tickers],
        low=pd.DataFrame(frames["low"], index=calendar)[tickers],
        close=pd.DataFrame(frames["close"], index=calendar)[tickers],
        volume=pd.DataFrame(frames["volume"], index=calendar)[tickers],
        available=pd.DataFrame(available, index=calendar)[tickers].fillna(False).astype(bool),
    )
    for frame in (panel.open, panel.high, panel.low, panel.close, panel.volume, panel.available):
        frame.index.name = "date"
    return panel


def _bars_since_true(mask: pd.Series) -> pd.Series:
    """Nombre de seances depuis le dernier True (0 si True aujourd'hui)."""
    positions = pd.Series(range(len(mask)), index=mask.index)
    last_true = positions.where(mask.to_numpy()).ffill()
    return (positions - last_true).fillna(len(mask))


@dataclass
class MarketData:
    """Panel de l'univers investissable + serie du benchmark (hors univers)."""

    panel: Panel
    benchmark: pd.Series | None = None
    benchmark_name: str | None = None

    @property
    def tickers(self) -> list[str]:
        return self.panel.tickers

    @property
    def dates(self) -> pd.DatetimeIndex:
        return self.panel.dates


def load_market_data(cfg: Config, refresh: bool = False, verbose: bool = False) -> MarketData:
    """Raccourci : recuperation + alignement de l'univers et du benchmark.

    Le benchmark n'entre pas dans l'univers investissable : il ne sert qu'au
    filtre de regime et a la comparaison de performance.
    """
    histories = fetch_histories(cfg, refresh=refresh, verbose=verbose)
    universe = [t for t in cfg.universe.tickers if t in histories]
    if not universe:
        raise DataError(
            "Aucun titre de l'univers n'a de donnees exploitables. "
            f"Tickers demandes : {', '.join(cfg.universe.tickers)}"
        )
    panel = build_panel(histories, cfg, tickers=universe)

    benchmark = None
    bench_name = cfg.universe.benchmark
    if bench_name and bench_name in histories and len(histories[bench_name]):
        benchmark = histories[bench_name]["close"].reindex(panel.dates).ffill()
        benchmark.name = bench_name
    elif bench_name and verbose:
        print(
            f"Avertissement : pas de donnees pour le benchmark {bench_name}, "
            "un indice equipondere de l'univers sera utilise."
        )
    return MarketData(panel=panel, benchmark=benchmark, benchmark_name=bench_name)
