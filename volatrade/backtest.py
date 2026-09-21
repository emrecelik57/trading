"""Verification historique des regles d'achat et de vente.

Le backtest rejoue exactement la logique de production : le score est calcule
sur les seules donnees disponibles a la date de decision, l'entree se fait a
l'ouverture de la seance suivante, et les sorties suivent les regles du plan
(stop initial, objectifs partiels, stop suiveur, rupture de tendance, stop
temporel).

Il ne modelise ni les frais, ni le glissement, ni les gaps intraday au-dela de
l'ouverture : les resultats sont un ordre de grandeur destine a valider la
coherence des regles, pas une promesse de rendement.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from .metrics import atr, compute_metrics, sma
from .plan import (
    TARGET_LADDER,
    TIME_STOP_DAYS,
    TRAILING_ATR_MULTIPLE,
    TRAILING_LOOKBACK,
    TREND_BREAK_ATR_BUFFER,
)
from .risk import RiskSettings, stop_distance
from .signals import ACTION_BUY, ACTION_PARTIAL, evaluate

WARMUP = 220


@dataclass
class Trade:
    """Un aller-retour complet."""

    ticker: str
    entry_date: str
    entry_price: float
    exit_date: str = ""
    exit_price: float = 0.0
    r_multiple: float = 0.0
    days_held: int = 0
    reason: str = ""
    partials: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class BacktestReport:
    """Statistiques agregees d'une serie de trades."""

    trades: list[Trade]
    n_trades: int
    win_rate: float
    avg_r: float
    median_r: float
    expectancy: float
    profit_factor: float
    total_r: float
    max_drawdown_r: float
    avg_days: float
    best_r: float
    worst_r: float
    capital_return: float
    buy_hold: float

    def as_dict(self) -> dict:
        data = asdict(self)
        data["trades"] = [trade.as_dict() for trade in self.trades]
        return data


def _run_single(
    ticker: str,
    frame: pd.DataFrame,
    benchmark_returns: pd.Series | None,
    settings: RiskSettings,
    decision_every: int,
) -> tuple[list[Trade], float]:
    """Rejoue les regles sur un titre : (trades, performance acheter-conserver)."""
    frame = frame.dropna(subset=["open", "high", "low", "close"])
    if len(frame) < WARMUP + 30:
        return [], float("nan")
    buy_hold = float(frame["close"].iloc[-1] / frame["close"].iloc[WARMUP] - 1.0)

    close = frame["close"]
    sma50 = sma(close, 50)
    atr_series = atr(frame, 14)
    highs = frame["high"]

    trades: list[Trade] = []
    position: dict | None = None

    for index in range(WARMUP, len(frame) - 1):
        today = frame.index[index]
        next_open = float(frame["open"].iloc[index + 1])

        if position is not None:
            high = float(frame["high"].iloc[index])
            low = float(frame["low"].iloc[index])
            entry = position["entry"]
            unit = position["unit"]
            atr_now = float(atr_series.iloc[index]) if np.isfinite(atr_series.iloc[index]) else unit / 2.5

            # 1. Stop (prioritaire : on suppose toujours le pire ordre intraday).
            if low <= position["stop"]:
                fill = min(position["stop"], float(frame["open"].iloc[index]))
                position["realized_r"] += position["remaining"] * (fill - entry) / unit
                trades.append(
                    _close_trade(position, ticker, today, fill, "stop touche", index)
                )
                position = None
                continue

            # 2. Objectifs partiels.
            for level, (multiple, fraction) in enumerate(TARGET_LADDER, start=1):
                key = f"objectif{level}"
                target = entry + multiple * unit
                if key not in position["hit"] and high >= target:
                    fill = max(target, float(frame["open"].iloc[index]))
                    taken = min(fraction, position["remaining"])
                    position["realized_r"] += taken * (fill - entry) / unit
                    position["remaining"] -= taken
                    position["hit"].append(key)
                    position["partials"].append(f"{key} a {fill:.2f}")
                    if level == 1:
                        position["stop"] = max(position["stop"], entry)

            if position["remaining"] <= 1e-9:
                trades.append(_close_trade(position, ticker, today, float(close.iloc[index]), "objectifs atteints", index))
                position = None
                continue

            # 3. Stop suiveur une fois le premier objectif encaisse.
            if position["hit"]:
                window_high = float(highs.iloc[max(0, index - TRAILING_LOOKBACK + 1): index + 1].max())
                trailing = window_high - TRAILING_ATR_MULTIPLE * atr_now
                position["stop"] = max(position["stop"], trailing)

            # 4. Rupture de tendance : deux clotures sous la MM50.
            buffer_now = TREND_BREAK_ATR_BUFFER * atr_now
            broke = (
                np.isfinite(sma50.iloc[index])
                and np.isfinite(sma50.iloc[index - 1])
                and close.iloc[index] < sma50.iloc[index] - buffer_now
                and close.iloc[index - 1] < sma50.iloc[index - 1] - buffer_now
            )
            # 5. Stop temporel.
            stale = (
                index - position["entry_index"] >= TIME_STOP_DAYS
                and (float(close.iloc[index]) - entry) / unit < 0.5
            )
            if broke or stale:
                position["realized_r"] += position["remaining"] * (next_open - entry) / unit
                trades.append(
                    _close_trade(
                        position,
                        ticker,
                        frame.index[index + 1],
                        next_open,
                        "rupture de tendance" if broke else "stop temporel",
                        index + 1,
                    )
                )
                position = None
            continue

        # Pas de position : on evalue le signal a cadence reduite.
        if (index - WARMUP) % decision_every != 0:
            continue
        window = frame.iloc[: index + 1]
        bench = benchmark_returns.loc[: today] if benchmark_returns is not None else None
        metrics = compute_metrics(ticker, window, bench)
        signal = evaluate(ticker, window, metrics)
        if signal.action not in (ACTION_BUY, ACTION_PARTIAL):
            continue

        atr_value = float(atr_series.iloc[index])
        unit = stop_distance(atr_value, next_open, settings)
        position = {
            "entry": next_open,
            "unit": unit,
            "stop": next_open - unit,
            "remaining": 1.0,
            "realized_r": 0.0,
            "hit": [],
            "partials": [],
            "entry_index": index + 1,
            "entry_date": frame.index[index + 1],
        }

    if position is not None:
        last_close = float(close.iloc[-1])
        position["realized_r"] += position["remaining"] * (last_close - position["entry"]) / position["unit"]
        trades.append(
            _close_trade(position, ticker, frame.index[-1], last_close, "position encore ouverte", len(frame) - 1)
        )
    return trades, buy_hold


def _close_trade(position: dict, ticker: str, date, price: float, reason: str, index: int) -> Trade:
    return Trade(
        ticker=ticker,
        entry_date=str(pd.Timestamp(position["entry_date"]).date()),
        entry_price=float(position["entry"]),
        exit_date=str(pd.Timestamp(date).date()),
        exit_price=float(price),
        r_multiple=float(position["realized_r"]),
        days_held=int(index - position["entry_index"]),
        reason=reason,
        partials=list(position["partials"]),
    )


def summarize(
    trades: list[Trade], buy_hold: float = float("nan"), risk_per_trade: float = 0.01
) -> BacktestReport:
    """Calcule les statistiques d'une liste de trades.

    `capital_return` traduit le resultat en R en performance du capital : en
    risquant `risk_per_trade` a chaque trade, +18 R valent +18 % de capital
    (approximation : ni compose, ni ajuste des positions simultanees).
    """
    if not trades:
        return BacktestReport([], 0, *([float("nan")] * 10), float("nan"), buy_hold)

    results = np.array([trade.r_multiple for trade in trades], dtype=float)
    wins = results[results > 0]
    losses = results[results <= 0]
    equity = np.cumsum(results)
    peak = np.maximum.accumulate(equity)

    gross_win = float(wins.sum()) if wins.size else 0.0
    gross_loss = float(-losses.sum()) if losses.size else 0.0

    return BacktestReport(
        trades=trades,
        n_trades=len(trades),
        win_rate=float(wins.size / results.size),
        avg_r=float(results.mean()),
        median_r=float(np.median(results)),
        expectancy=float(results.mean()),
        profit_factor=float(gross_win / gross_loss) if gross_loss > 0 else float("inf"),
        total_r=float(results.sum()),
        max_drawdown_r=float((equity - peak).min()),
        avg_days=float(np.mean([trade.days_held for trade in trades])),
        best_r=float(results.max()),
        worst_r=float(results.min()),
        capital_return=float(results.sum() * risk_per_trade),
        buy_hold=float(buy_hold),
    )


def run_backtest(
    quotes: dict,
    benchmark_returns: pd.Series | None = None,
    settings: RiskSettings | None = None,
    decision_every: int = 5,
) -> tuple[BacktestReport, dict[str, BacktestReport]]:
    """Backteste tous les titres et renvoie (rapport global, rapports par titre)."""
    settings = settings or RiskSettings()
    all_trades: list[Trade] = []
    per_ticker: dict[str, BacktestReport] = {}
    holds: list[float] = []
    for ticker, quote in quotes.items():
        trades, buy_hold = _run_single(
            ticker, quote.frame, benchmark_returns, settings, decision_every
        )
        per_ticker[ticker] = summarize(trades, buy_hold, settings.risk_per_trade)
        all_trades.extend(trades)
        if np.isfinite(buy_hold):
            holds.append(buy_hold)
    all_trades.sort(key=lambda trade: trade.entry_date)
    average_hold = float(np.mean(holds)) if holds else float("nan")
    return summarize(all_trades, average_hold, settings.risk_per_trade), per_ticker
