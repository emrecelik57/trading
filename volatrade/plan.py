"""Construction du plan de trade : ou acheter, ou placer le stop, quand vendre.

Un plan est entierement chiffre a l'avance. C'est ce qui distingue une
decision d'un pari : le point de sortie est decide avant l'entree, en unites
de risque (R = distance entre le prix d'entree et le stop initial).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from .metrics import RiskMetrics
from .risk import RiskSettings, Sizing
from .signals import ACTION_AVOID, ACTION_WATCH, Signal

#: Objectifs exprimes en multiples de R et fraction de la position a solder.
TARGET_LADDER = ((1.5, 0.33), (3.0, 0.33))
#: Multiple d'ATR pour le stop suiveur (chandelier exit).
TRAILING_ATR_MULTIPLE = 3.0
#: Fenetre du plus haut utilise par le stop suiveur.
TRAILING_LOOKBACK = 22
#: Nombre de seances au bout desquelles un trade qui n'avance pas est solde.
TIME_STOP_DAYS = 25
#: Marge, en ATR, sous la moyenne 50 jours avant de declarer la tendance cassee.
#: Sans marge, le bruit d'un titre a 6 % d'amplitude quotidienne declenche des
#: sorties permanentes (verifie par backtest : sortie tous les 3 jours).
TREND_BREAK_ATR_BUFFER = 1.0


@dataclass(frozen=True)
class ExitRule:
    """Une regle de sortie, avec son declencheur et l'action associee."""

    nom: str
    declencheur: str
    action: str
    niveau: float | None = None

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class TradePlan:
    """Plan complet d'un achat : entree, risque, objectifs et sorties."""

    ticker: str
    action: str
    score: float
    theme: str
    price: float
    entry_low: float
    entry_high: float
    shares: int
    notional: float
    weight: float
    stop_price: float
    stop_pct: float
    risk_amount: float
    risk_pct: float
    unit_risk: float
    target1: float
    target2: float
    gain_target1: float
    gain_target2: float
    trailing_rule: str
    time_stop_days: int
    horizon_days: int
    binding_constraint: str
    exits: list[ExitRule] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        data = asdict(self)
        data["exits"] = [rule.as_dict() for rule in self.exits]
        return data


def _horizon_days(unit_risk: float, price: float, daily_volatility: float) -> int:
    """Estimation du temps pour parcourir 3 R, via une marche aleatoire.

    Une marche aleatoire parcourt en moyenne sigma * racine(t) ; le temps
    necessaire pour couvrir une distance d vaut donc (d / sigma)^2 seances.
    C'est un ordre de grandeur, pas une prevision.
    """
    if not np.isfinite(daily_volatility) or daily_volatility <= 0 or price <= 0:
        return TIME_STOP_DAYS
    distance = TARGET_LADDER[-1][0] * unit_risk / price
    days = (distance / daily_volatility) ** 2
    return int(np.clip(round(days), 5, 180))


def build_plan(
    signal: Signal,
    metrics: RiskMetrics,
    sizing: Sizing,
    theme: str = "autre",
    settings: RiskSettings | None = None,
) -> TradePlan:
    """Assemble le plan de trade a partir du signal, du risque et de la taille."""
    settings = settings or RiskSettings()
    state = signal.state
    price = sizing.price
    unit_risk = sizing.stop_distance
    atr_value = state.atr14 if np.isfinite(state.atr14) else unit_risk / 2.5

    # Zone d'achat : on paie au maximum un demi-ATR au-dessus du cours actuel,
    # et on accepte de rater le titre s'il s'envole avant d'etre achete.
    entry_low = price - 0.5 * atr_value
    entry_high = price + 0.5 * atr_value

    target1 = price + TARGET_LADDER[0][0] * unit_risk
    target2 = price + TARGET_LADDER[1][0] * unit_risk

    trailing = (
        f"apres l'objectif 1, remonter le stop sous le plus haut des "
        f"{TRAILING_LOOKBACK} seances moins {TRAILING_ATR_MULTIPLE:.0f} x ATR"
    )

    exits = [
        ExitRule(
            "stop initial",
            f"cloture ou meche sous {sizing.stop_price:.2f}",
            "vendre 100 % de la ligne, sans discussion",
            sizing.stop_price,
        ),
        ExitRule(
            "objectif 1 (+1,5 R)",
            f"cours >= {target1:.2f}",
            f"vendre {TARGET_LADDER[0][1]:.0%} et remonter le stop au point mort ({price:.2f})",
            target1,
        ),
        ExitRule(
            "objectif 2 (+3 R)",
            f"cours >= {target2:.2f}",
            f"vendre {TARGET_LADDER[1][1]:.0%} et laisser courir le solde en stop suiveur",
            target2,
        ),
        ExitRule(
            "stop suiveur",
            trailing,
            "vendre le solde si le stop suiveur est touche",
            None,
        ),
        ExitRule(
            "rupture de tendance",
            f"deux clotures consecutives a plus de {TREND_BREAK_ATR_BUFFER:g} ATR "
            "sous la moyenne 50 jours"
            + (
                f" (soit {state.sma50 - TREND_BREAK_ATR_BUFFER * atr_value:.2f})"
                if np.isfinite(state.sma50)
                else ""
            ),
            "vendre la totalite du solde",
            (
                float(state.sma50 - TREND_BREAK_ATR_BUFFER * atr_value)
                if np.isfinite(state.sma50)
                else None
            ),
        ),
        ExitRule(
            "stop temporel",
            f"apres {TIME_STOP_DAYS} seances sans avoir atteint +0,5 R",
            "solder : le capital travaille mieux ailleurs",
            None,
        ),
        ExitRule(
            "choc de volatilite",
            "volatilite 1 mois > 1,8 x volatilite 6 mois",
            "reduire la ligne de moitie meme sans signal de prix",
            None,
        ),
    ]

    warnings = list(signal.warnings)
    if sizing.shares == 0 and signal.conviction > 0:
        if sizing.binding_constraint == "risque par trade" and settings.risk_per_trade > 0:
            needed = unit_risk / settings.risk_per_trade
            warnings.append(
                "signal d'achat valide mais un seul titre ferait deja courir plus de "
                f"{settings.risk_per_trade:.1%} du capital : il faudrait environ "
                f"{needed:,.0f} de capital pour cette ligne".replace(",", " ")
            )
        else:
            warnings.append(
                f"signal d'achat valide mais aucune quantite allouee : {sizing.binding_constraint}"
            )

    return TradePlan(
        ticker=signal.ticker,
        action=signal.action,
        score=signal.score,
        theme=theme,
        price=price,
        entry_low=float(entry_low),
        entry_high=float(entry_high),
        shares=sizing.shares,
        notional=sizing.notional,
        weight=sizing.weight,
        stop_price=sizing.stop_price,
        stop_pct=float(unit_risk / price) if price else float("nan"),
        risk_amount=sizing.risk_amount,
        risk_pct=sizing.risk_pct,
        unit_risk=float(unit_risk),
        target1=float(target1),
        target2=float(target2),
        gain_target1=float(sizing.shares * TARGET_LADDER[0][1] * (target1 - price)),
        gain_target2=float(sizing.shares * TARGET_LADDER[1][1] * (target2 - price)),
        trailing_rule=trailing,
        time_stop_days=TIME_STOP_DAYS,
        horizon_days=_horizon_days(unit_risk, price, metrics.expected_daily_move),
        binding_constraint=sizing.binding_constraint,
        exits=exits,
        reasons=list(signal.reasons),
        warnings=warnings,
    )


def is_actionable(plan: TradePlan) -> bool:
    """Vrai si le plan debouche sur un ordre a passer aujourd'hui."""
    return plan.shares > 0 and plan.action not in (ACTION_AVOID, ACTION_WATCH)


def chandelier_stop(
    frame: pd.DataFrame, atr_value: float, lookback: int = TRAILING_LOOKBACK
) -> float:
    """Stop suiveur : plus haut recent moins un multiple d'ATR."""
    highs = frame["high"].dropna().iloc[-lookback:]
    if highs.empty or not np.isfinite(atr_value):
        return float("nan")
    return float(highs.max() - TRAILING_ATR_MULTIPLE * atr_value)


def expected_annual_move(metrics: RiskMetrics) -> float:
    """Amplitude annuelle a un ecart-type, pour situer l'ordre de grandeur."""
    volatility = metrics.vol_ewma if np.isfinite(metrics.vol_ewma) else metrics.vol_ann_3m
    return float(volatility) if np.isfinite(volatility) else float("nan")


def summarize_basket(plans: list[TradePlan]) -> dict[str, float]:
    """Agrege les plans actionnables : capital engage, risque, gain vise."""
    actionable = [plan for plan in plans if is_actionable(plan)]
    return {
        "lignes": float(len(actionable)),
        "capital_engage": float(sum(plan.notional for plan in actionable)),
        "risque_total": float(sum(plan.risk_amount for plan in actionable)),
        "gain_si_objectifs": float(
            sum(plan.gain_target1 + plan.gain_target2 for plan in actionable)
        ),
        "horizon_median": float(
            np.median([plan.horizon_days for plan in actionable]) if actionable else float("nan")
        ),
    }
