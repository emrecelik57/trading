"""Score d'achat : tendance, momentum, qualite du point d'entree, risque.

Le score final va de 0 a 100 et se traduit en une action (`ACHAT`,
`ACHAT PARTIEL`, `SURVEILLER`, `EVITER`). La logique assumee est celle du
suivi de tendance sur repli : on achete un titre haussier qui respire, pas un
titre en chute libre ni un titre en surchauffe verticale.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .metrics import RiskMetrics, atr, rsi, sma, total_return

ACTION_BUY = "ACHAT"
ACTION_PARTIAL = "ACHAT PARTIEL"
ACTION_WATCH = "SURVEILLER"
ACTION_AVOID = "EVITER"

#: Ponderation des quatre blocs du score.
WEIGHTS = {"tendance": 0.35, "momentum": 0.25, "entree": 0.20, "risque": 0.20}


def _ramp(value: float, low: float, high: float) -> float:
    """Projette `value` sur 0..100 lineairement entre `low` et `high`."""
    if not np.isfinite(value):
        return 50.0
    if high == low:
        return 50.0
    return float(np.clip((value - low) / (high - low), 0.0, 1.0) * 100.0)


def _band(value: float, ideal_low: float, ideal_high: float, span: float) -> float:
    """100 dans la plage ideale, decroissant lineairement de part et d'autre."""
    if not np.isfinite(value):
        return 50.0
    if ideal_low <= value <= ideal_high:
        return 100.0
    distance = ideal_low - value if value < ideal_low else value - ideal_high
    return float(np.clip(1.0 - distance / span, 0.0, 1.0) * 100.0)


@dataclass(frozen=True)
class TechnicalState:
    """Photographie technique d'un titre a la derniere seance."""

    price: float
    sma20: float
    sma50: float
    sma200: float
    sma50_slope: float
    rsi14: float
    atr14: float
    high20: float
    high52w: float
    low52w: float
    distance_high20: float
    distance_high52w: float
    extension_atr: float
    momentum_6m: float
    momentum_3m: float
    momentum_12_1: float


def technical_state(frame: pd.DataFrame) -> TechnicalState:
    """Calcule les moyennes mobiles, le RSI et les distances aux extremes."""
    close = frame["close"].astype(float)
    price = float(close.iloc[-1])
    sma20 = sma(close, 20)
    sma50 = sma(close, 50)
    sma200 = sma(close, 200)
    atr14 = atr(frame, 14)

    def last(series: pd.Series) -> float:
        clean = series.dropna()
        return float(clean.iloc[-1]) if len(clean) else float("nan")

    sma50_last = last(sma50)
    sma50_prev = float(sma50.dropna().iloc[-21]) if len(sma50.dropna()) > 21 else float("nan")
    slope = (
        (sma50_last / sma50_prev - 1.0)
        if np.isfinite(sma50_last) and np.isfinite(sma50_prev) and sma50_prev
        else float("nan")
    )

    high20 = float(close.iloc[-20:].max())
    window52 = close.iloc[-252:]
    high52w = float(window52.max())
    low52w = float(window52.min())
    atr_last = last(atr14)
    sma20_last = last(sma20)

    return TechnicalState(
        price=price,
        sma20=sma20_last,
        sma50=sma50_last,
        sma200=last(sma200),
        sma50_slope=slope,
        rsi14=last(rsi(close, 14)),
        atr14=atr_last,
        high20=high20,
        high52w=high52w,
        low52w=low52w,
        distance_high20=price / high20 - 1.0 if high20 else float("nan"),
        distance_high52w=price / high52w - 1.0 if high52w else float("nan"),
        extension_atr=(
            (price - sma20_last) / atr_last
            if np.isfinite(sma20_last) and np.isfinite(atr_last) and atr_last
            else float("nan")
        ),
        momentum_6m=total_return(close, 126),
        momentum_3m=total_return(close, 63),
        momentum_12_1=(
            float(close.iloc[-21] / close.iloc[-252] - 1.0) if len(close) > 252 else float("nan")
        ),
    )


@dataclass(frozen=True)
class Signal:
    """Verdict d'achat pour un titre."""

    ticker: str
    score: float
    action: str
    components: dict[str, float]
    state: TechnicalState
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def conviction(self) -> float:
        """Fraction de la taille de position nominale a engager (0 a 1)."""
        if self.action == ACTION_BUY:
            return 1.0
        if self.action == ACTION_PARTIAL:
            return 0.5
        return 0.0


def _trend_score(state: TechnicalState, reasons: list[str], warnings: list[str]) -> float:
    points = 0.0
    if np.isfinite(state.sma200) and state.price > state.sma200:
        points += 40.0
        reasons.append("cours au-dessus de la moyenne 200 jours (tendance de fond haussiere)")
    else:
        warnings.append("cours sous la moyenne 200 jours : tendance de fond baissiere")
    if np.isfinite(state.sma50) and state.price > state.sma50:
        points += 25.0
    else:
        warnings.append("cours sous la moyenne 50 jours : dynamique de moyen terme cassee")
    if np.isfinite(state.sma50) and np.isfinite(state.sma200) and state.sma50 > state.sma200:
        points += 20.0
    if np.isfinite(state.sma50_slope) and state.sma50_slope > 0:
        points += 15.0
        reasons.append(f"moyenne 50 jours orientee a la hausse ({state.sma50_slope:+.1%} sur 1 mois)")
    return points


def _momentum_score(state: TechnicalState, reasons: list[str]) -> float:
    six = _ramp(state.momentum_6m, -0.30, 0.60)
    three = _ramp(state.momentum_3m, -0.20, 0.40)
    twelve = _ramp(state.momentum_12_1, -0.30, 0.80) if np.isfinite(state.momentum_12_1) else 50.0
    score = 0.45 * six + 0.30 * three + 0.25 * twelve
    if np.isfinite(state.momentum_6m) and state.momentum_6m > 0.25:
        reasons.append(f"momentum 6 mois solide ({state.momentum_6m:+.0%})")
    return float(score)


def _entry_score(state: TechnicalState, reasons: list[str], warnings: list[str]) -> float:
    # RSI : ni survendu (couteau qui tombe) ni surachete (risque de reflux).
    rsi_score = _band(state.rsi14, 45.0, 65.0, 25.0)
    # Repli ideal : 2 a 12 % sous le plus haut des 20 seances.
    pullback = -state.distance_high20 if np.isfinite(state.distance_high20) else float("nan")
    pullback_score = _band(pullback, 0.02, 0.12, 0.12)
    # Extension : acheter a plus de 3 ATR au-dessus de la MM20 se paie cher.
    extension_score = _band(state.extension_atr, -1.0, 2.0, 2.5)

    if np.isfinite(state.rsi14) and state.rsi14 > 75:
        warnings.append(f"RSI en surchauffe ({state.rsi14:.0f}) : attendre un repli")
    if np.isfinite(pullback) and 0.02 <= pullback <= 0.12:
        reasons.append(f"repli sain de {pullback:.1%} sous le plus haut des 20 seances")
    if np.isfinite(state.extension_atr) and state.extension_atr > 3:
        warnings.append(f"cours etire a {state.extension_atr:.1f} ATR au-dessus de la MM20")

    return float(0.35 * rsi_score + 0.40 * pullback_score + 0.25 * extension_score)


def _risk_score(metrics: RiskMetrics, reasons: list[str], warnings: list[str]) -> float:
    sortino = _ramp(metrics.sortino, -0.5, 2.5)
    calmar = _ramp(metrics.calmar, -0.5, 2.0)
    # Un regime de volatilite qui s'emballe merite une decote.
    regime = _band(metrics.vol_regime, 0.7, 1.2, 0.8)
    drawdown = _ramp(metrics.current_drawdown, -0.60, -0.05)

    if np.isfinite(metrics.vol_regime) and metrics.vol_regime > 1.4:
        warnings.append(
            f"volatilite en acceleration (x{metrics.vol_regime:.2f} vs regime normal) : reduire la taille"
        )
    if np.isfinite(metrics.sortino) and metrics.sortino > 1.5:
        reasons.append(f"Sortino eleve ({metrics.sortino:.2f}) : le risque paye")
    if np.isfinite(metrics.current_drawdown) and metrics.current_drawdown < -0.35:
        warnings.append(f"titre a {metrics.current_drawdown:.0%} de son plus haut : rebond non confirme")

    return float(0.30 * sortino + 0.25 * calmar + 0.25 * regime + 0.20 * drawdown)


def evaluate(ticker: str, frame: pd.DataFrame, metrics: RiskMetrics) -> Signal:
    """Produit le score d'achat et l'action recommandee pour un titre."""
    state = technical_state(frame)
    reasons: list[str] = []
    warnings: list[str] = []

    components = {
        "tendance": _trend_score(state, reasons, warnings),
        "momentum": _momentum_score(state, reasons),
        "entree": _entry_score(state, reasons, warnings),
        "risque": _risk_score(metrics, reasons, warnings),
    }
    score = float(sum(components[key] * WEIGHTS[key] for key in WEIGHTS))

    if score >= 68:
        action = ACTION_BUY
    elif score >= 55:
        action = ACTION_PARTIAL
    elif score >= 42:
        action = ACTION_WATCH
    else:
        action = ACTION_AVOID

    # Vetos : conditions ou aucun score ne justifie une entree pleine.
    below_200 = np.isfinite(state.sma200) and state.price < state.sma200
    if below_200 and np.isfinite(state.momentum_6m) and state.momentum_6m < -0.20:
        action = ACTION_AVOID
        warnings.append("veto : sous la MM200 avec un momentum 6 mois negatif")
    elif below_200 and action == ACTION_BUY:
        action = ACTION_PARTIAL
        warnings.append("taille limitee a une demi-position : tendance de fond non confirmee")
    if np.isfinite(metrics.vol_regime) and metrics.vol_regime > 1.8 and action == ACTION_BUY:
        action = ACTION_PARTIAL
        warnings.append("taille limitee : choc de volatilite en cours")
    if action == ACTION_BUY and components["momentum"] < 45:
        # Une position pleine se justifie par une dynamique, pas par la seule
        # absence de mauvaises nouvelles : sans momentum, on n'engage qu'a moitie.
        action = ACTION_PARTIAL
        warnings.append("taille limitee : momentum trop faible pour une position pleine")

    return Signal(
        ticker=ticker.upper(),
        score=score,
        action=action,
        components=components,
        state=state,
        reasons=reasons,
        warnings=warnings,
    )
