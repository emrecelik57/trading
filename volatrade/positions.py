"""Suivi des positions ouvertes : conserver, alleger ou vendre aujourd'hui.

Le fichier de portefeuille est un JSON simple :

    [
      {"ticker": "TSLA", "shares": 12, "entry_price": 250.0,
       "entry_date": "2026-06-02", "stop_price": 221.5,
       "target1": 294.3, "target2": 382.9, "trimmed": []}
    ]

Seuls `ticker`, `shares` et `entry_price` sont obligatoires : le stop et les
objectifs manquants sont reconstruits a partir de l'ATR courant.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .data import Quote
from .metrics import atr, sma
from .plan import (
    TARGET_LADDER,
    TIME_STOP_DAYS,
    TREND_BREAK_ATR_BUFFER,
    chandelier_stop,
)
from .risk import RiskSettings, stop_distance

VERDICT_SELL = "VENDRE"
VERDICT_TRIM = "ALLEGER"
VERDICT_HOLD = "CONSERVER"
VERDICT_TIGHTEN = "REMONTER LE STOP"


class PortfolioError(ValueError):
    """Fichier de portefeuille invalide."""


@dataclass
class Position:
    """Une ligne detenue."""

    ticker: str
    shares: float
    entry_price: float
    entry_date: str | None = None
    stop_price: float | None = None
    target1: float | None = None
    target2: float | None = None
    trimmed: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, raw: dict) -> "Position":
        missing = [key for key in ("ticker", "shares", "entry_price") if key not in raw]
        if missing:
            raise PortfolioError(f"position incomplete, champs manquants : {', '.join(missing)}")
        return cls(
            ticker=str(raw["ticker"]).upper(),
            shares=float(raw["shares"]),
            entry_price=float(raw["entry_price"]),
            entry_date=raw.get("entry_date"),
            stop_price=float(raw["stop_price"]) if raw.get("stop_price") is not None else None,
            target1=float(raw["target1"]) if raw.get("target1") is not None else None,
            target2=float(raw["target2"]) if raw.get("target2") is not None else None,
            trimmed=list(raw.get("trimmed", [])),
        )


@dataclass(frozen=True)
class PositionReview:
    """Verdict du jour sur une position ouverte."""

    ticker: str
    verdict: str
    price: float
    entry_price: float
    shares: float
    pnl: float
    pnl_pct: float
    r_multiple: float
    stop_price: float
    suggested_stop: float
    target1: float
    target2: float
    days_held: int
    actions: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return asdict(self)


def load_portfolio(path: Path | str) -> list[Position]:
    """Lit le fichier de portefeuille."""
    path = Path(path)
    if not path.exists():
        raise PortfolioError(f"fichier de portefeuille introuvable : {path}")
    try:
        raw = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise PortfolioError(f"JSON invalide dans {path} : {exc}") from exc
    if isinstance(raw, dict):
        raw = raw.get("positions", [])
    if not isinstance(raw, list):
        raise PortfolioError("le fichier doit contenir une liste de positions")
    return [Position.from_dict(item) for item in raw]


def _days_held(entry_date: str | None, last_date) -> int:
    if not entry_date:
        return 0
    try:
        start = pd.Timestamp(entry_date, tz="UTC")
    except (ValueError, TypeError):
        return 0
    end = pd.Timestamp(last_date)
    if end.tzinfo is None:
        end = end.tz_localize("UTC")
    # ~21 seances de bourse par mois calendaire.
    return int(max(0, np.busday_count(start.date(), end.date())))


def review_position(
    position: Position, quote: Quote, settings: RiskSettings | None = None
) -> PositionReview:
    """Compare la position aux regles de sortie et rend un verdict."""
    settings = settings or RiskSettings()
    frame = quote.frame
    price = quote.last_price
    atr_series = atr(frame, 14).dropna()
    atr_value = float(atr_series.iloc[-1]) if len(atr_series) else float("nan")

    stop = position.stop_price
    if stop is None:
        stop = position.entry_price - stop_distance(atr_value, position.entry_price, settings)
    unit_risk = max(position.entry_price - stop, 1e-9)
    target1 = position.target1 or position.entry_price + TARGET_LADDER[0][0] * unit_risk
    target2 = position.target2 or position.entry_price + TARGET_LADDER[1][0] * unit_risk

    pnl = (price - position.entry_price) * position.shares
    pnl_pct = price / position.entry_price - 1.0 if position.entry_price else float("nan")
    r_multiple = (price - position.entry_price) / unit_risk
    days = _days_held(position.entry_date, frame.index[-1])

    sma50_series = sma(frame["close"], 50).dropna()
    sma50 = float(sma50_series.iloc[-1]) if len(sma50_series) else float("nan")
    closes = frame["close"].dropna()
    # Marge d'un ATR sous la MM50 : sans elle, le bruit quotidien d'un titre
    # volatil declenche une sortie tous les trois jours (cf. backtest).
    buffer_value = TREND_BREAK_ATR_BUFFER * atr_value if np.isfinite(atr_value) else 0.0
    broke_trend = (
        len(closes) >= 2
        and len(sma50_series) >= 2
        and bool(
            closes.iloc[-1] < sma50_series.iloc[-1] - buffer_value
            and closes.iloc[-2] < sma50_series.iloc[-2] - buffer_value
        )
    )

    trailing = chandelier_stop(frame, atr_value)
    suggested_stop = stop
    if "objectif1" in position.trimmed:
        # Apres la premiere prise de benefice le stop ne redescend jamais.
        candidates = [value for value in (stop, position.entry_price, trailing) if np.isfinite(value)]
        suggested_stop = max(candidates)
    elif np.isfinite(trailing):
        suggested_stop = max(stop, trailing) if r_multiple >= 1.0 else stop

    actions: list[str] = []
    notes: list[str] = []
    verdict = VERDICT_HOLD

    if price <= stop:
        verdict = VERDICT_SELL
        actions.append(f"stop touche ({price:.2f} <= {stop:.2f}) : solder la totalite")
    elif broke_trend:
        verdict = VERDICT_SELL
        actions.append(
            f"deux clotures a plus d'un ATR sous la moyenne 50 jours ({sma50:.2f}) : "
            "sortir, la tendance est cassee"
        )
    elif price >= target2 and "objectif2" not in position.trimmed:
        verdict = VERDICT_TRIM
        actions.append(
            f"objectif 2 atteint ({price:.2f} >= {target2:.2f}) : vendre "
            f"{TARGET_LADDER[1][1]:.0%} et suivre le solde au stop suiveur"
        )
    elif price >= target1 and "objectif1" not in position.trimmed:
        verdict = VERDICT_TRIM
        actions.append(
            f"objectif 1 atteint ({price:.2f} >= {target1:.2f}) : vendre "
            f"{TARGET_LADDER[0][1]:.0%} et remonter le stop au point mort "
            f"({position.entry_price:.2f})"
        )
    elif days >= TIME_STOP_DAYS and r_multiple < 0.5:
        verdict = VERDICT_SELL
        actions.append(
            f"stop temporel : {days} seances sans depasser +0,5 R ({r_multiple:+.2f} R), liberer le capital"
        )
    elif np.isfinite(suggested_stop) and suggested_stop > stop + 1e-9:
        verdict = VERDICT_TIGHTEN
        actions.append(f"remonter le stop de {stop:.2f} a {suggested_stop:.2f}")

    if np.isfinite(atr_value):
        notes.append(f"amplitude quotidienne typique : {atr_value / price:.1%} du cours")
    notes.append(f"gain/perte latent : {pnl:+,.0f} ({pnl_pct:+.1%}, {r_multiple:+.2f} R)")
    if verdict == VERDICT_HOLD:
        notes.append(
            f"prochain declencheur : stop {stop:.2f} en bas, objectif "
            f"{(target1 if price < target1 else target2):.2f} en haut"
        )

    return PositionReview(
        ticker=position.ticker,
        verdict=verdict,
        price=price,
        entry_price=position.entry_price,
        shares=position.shares,
        pnl=float(pnl),
        pnl_pct=float(pnl_pct),
        r_multiple=float(r_multiple),
        stop_price=float(stop),
        suggested_stop=float(suggested_stop) if np.isfinite(suggested_stop) else float(stop),
        target1=float(target1),
        target2=float(target2),
        days_held=days,
        actions=actions,
        notes=notes,
    )


def review_portfolio(
    positions: list[Position], quotes: dict[str, Quote], settings: RiskSettings | None = None
) -> list[PositionReview]:
    """Passe en revue toutes les positions dont les cours sont disponibles."""
    reviews: list[PositionReview] = []
    for position in positions:
        quote = quotes.get(position.ticker)
        if quote is None:
            continue
        reviews.append(review_position(position, quote, settings))
    return reviews
