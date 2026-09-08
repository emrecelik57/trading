"""Mise en forme des resultats pour le terminal (tableaux et fiches)."""

from __future__ import annotations

import math

import numpy as np

from .backtest import BacktestReport
from .plan import TradePlan, is_actionable
from .positions import (
    PositionReview,
    VERDICT_HOLD,
    VERDICT_SELL,
    VERDICT_TIGHTEN,
    VERDICT_TRIM,
)
from .metrics import RiskMetrics
from .signals import ACTION_AVOID, ACTION_BUY, ACTION_PARTIAL, ACTION_WATCH
from .universe import ScreenResult

DISCLAIMER = (
    "Outil quantitatif a but educatif. Aucun conseil en investissement : les actions\n"
    "  tres volatiles peuvent perdre plus de la moitie de leur valeur en quelques\n"
    "  semaines. Les tailles de position supposent que vous respectez les stops."
)


# --------------------------------------------------------------------------
# Formatage elementaire
# --------------------------------------------------------------------------
def pct(value: float, digits: int = 1) -> str:
    if value is None or not np.isfinite(value):
        return "n/d"
    return f"{value * 100:.{digits}f} %"


def num(value: float, digits: int = 2) -> str:
    if value is None:
        return "n/d"
    if np.isposinf(value):
        return "+inf"
    if np.isneginf(value):
        return "-inf"
    if not np.isfinite(value):
        return "n/d"
    return f"{value:.{digits}f}"


def money(value: float, currency: str = "$", digits: int = 2) -> str:
    if value is None or not np.isfinite(value):
        return "n/d"
    return f"{value:,.{digits}f} {currency}".replace(",", " ")


def compact(value: float) -> str:
    """Nombre abrege : 1 234 567 -> 1.2 M."""
    if value is None or not np.isfinite(value):
        return "n/d"
    for limit, suffix in ((1e9, " Md"), (1e6, " M"), (1e3, " k")):
        if abs(value) >= limit:
            return f"{value / limit:.1f}{suffix}"
    return f"{value:.0f}"


def table(headers: list[str], rows: list[list[str]], aligns: str | None = None) -> str:
    """Rend un tableau texte a largeur fixe."""
    if not rows:
        return "  (aucune ligne)"
    widths = [len(header) for header in headers]
    for row in rows:
        for column, cell in enumerate(row):
            widths[column] = max(widths[column], len(cell))
    aligns = aligns or "l" * len(headers)

    def render(cells: list[str]) -> str:
        parts = []
        for column, cell in enumerate(cells):
            width = widths[column]
            parts.append(cell.ljust(width) if aligns[column] == "l" else cell.rjust(width))
        return "  " + " | ".join(parts)

    separator = "  " + "-+-".join("-" * width for width in widths)
    return "\n".join([render(headers), separator] + [render(row) for row in rows])


def title(text: str) -> str:
    return f"\n{text}\n{'=' * len(text)}"


def _action_tag(action: str) -> str:
    return {
        ACTION_BUY: "[ACHAT]",
        ACTION_PARTIAL: "[ACHAT PARTIEL]",
        ACTION_WATCH: "[SURVEILLER]",
        ACTION_AVOID: "[EVITER]",
    }.get(action, action)


# --------------------------------------------------------------------------
# Selection et metriques
# --------------------------------------------------------------------------
def format_screen(results: list[ScreenResult], show_rejected: bool = False) -> str:
    """Tableau de selection : volatilite, liquidite et motif de retenue."""
    rows = []
    for rank, item in enumerate(results, start=1):
        if not item.retained and not show_rejected:
            continue
        metrics = item.metrics
        rows.append(
            [
                str(rank) if item.retained else "-",
                item.ticker,
                item.theme,
                pct(item.volatility_score, 0),
                pct(metrics.vol_ann_1m, 0),
                pct(metrics.atr_pct),
                num(metrics.beta),
                compact(metrics.dollar_volume),
                item.reason,
            ]
        )
    return table(
        ["#", "Titre", "Theme", "Vol. sel.", "Vol. 1m", "ATR/j", "Beta", "Volume/j", "Statut"],
        rows,
        aligns="llrrrrrrl",
    )


def format_risk_table(metrics_list: list[RiskMetrics]) -> str:
    """Tableau detaille des mesures de risque."""
    rows = []
    for metrics in metrics_list:
        rows.append(
            [
                metrics.ticker,
                money(metrics.last_price),
                pct(metrics.vol_ewma, 0),
                pct(metrics.max_drawdown, 0),
                pct(metrics.current_drawdown, 0),
                pct(metrics.var_95_1d),
                pct(metrics.cvar_95_1d),
                pct(metrics.var_95_10d, 0),
                num(metrics.sharpe),
                num(metrics.sortino),
                num(metrics.calmar),
                num(metrics.vol_regime),
            ]
        )
    return table(
        [
            "Titre", "Cours", "Vol. EWMA", "Perte max", "Ecart au haut",
            "VaR 95 %", "CVaR 95 %", "VaR 10j", "Sharpe", "Sortino", "Calmar", "Regime",
        ],
        rows,
        aligns="lrrrrrrrrrrr",
    )


def explain_metrics() -> str:
    """Rappel de la lecture des mesures principales."""
    lignes = [
        "Vol. EWMA      volatilite annualisee reactive : 80 % = un ecart-type de 80 % sur un an.",
        "ATR/j          amplitude typique d'une seance, en % du cours (sert a calibrer le stop).",
        "Beta           sensibilite au S&P 500 : 2.0 = amplifie de 100 % les mouvements de l'indice.",
        "Perte max      pire chute pic-creux sur 12 mois : le scenario deja vecu par le titre.",
        "VaR 95 %       perte quotidienne depassee dans 5 % des seances les pires.",
        "CVaR 95 %      perte moyenne le jour ou la VaR est depassee (le vrai cout des queues).",
        "Sharpe         rendement par unite de risque total ; Sortino ne compte que la baisse.",
        "Calmar         rendement annualise divise par la perte maximale.",
        "Regime         vol. 1 mois / vol. 6 mois : > 1.3 = le risque s'emballe, reduire la taille.",
    ]
    return "\n".join("  " + ligne for ligne in lignes)


# --------------------------------------------------------------------------
# Plans d'achat
# --------------------------------------------------------------------------
def format_plan_table(plans: list[TradePlan], currency: str = "$") -> str:
    """Vue synthetique : quoi acheter, combien, ou sortir."""
    rows = []
    for plan in plans:
        tag = _action_tag(plan.action)
        if plan.shares == 0 and plan.action in (ACTION_BUY, ACTION_PARTIAL):
            tag += " (non execute)"
        rows.append(
            [
                plan.ticker,
                tag,
                num(plan.score, 0),
                f"{num(plan.entry_low)} - {num(plan.entry_high)}",
                str(plan.shares),
                money(plan.notional, currency, 0),
                pct(plan.weight, 0),
                money(plan.stop_price, currency),
                money(plan.target1, currency),
                money(plan.target2, currency),
                money(plan.risk_amount, currency, 0),
                f"{plan.horizon_days} j",
            ]
        )
    return table(
        [
            "Titre", "Action", "Score", "Zone d'achat", "Qte", "Montant",
            "Poids", "Stop", "Objectif 1", "Objectif 2", "Risque", "Horizon",
        ],
        rows,
        aligns="llrlrrrrrrrr",
    )


def format_plan_detail(plan: TradePlan, currency: str = "$") -> str:
    """Fiche complete d'un titre : contexte, ordre a passer, regles de sortie."""
    lines = [
        f"\n{plan.ticker} — {_action_tag(plan.action)} (score {plan.score:.0f}/100, theme {plan.theme})",
        "-" * 78,
        f"  Cours            {money(plan.price, currency)}"
        f"   |  zone d'achat {money(plan.entry_low, currency)} - {money(plan.entry_high, currency)}",
        f"  Quantite         {plan.shares} titres  ({money(plan.notional, currency, 0)}, "
        f"{pct(plan.weight, 1)} du capital, contrainte : {plan.binding_constraint})",
        f"  Stop initial     {money(plan.stop_price, currency)}  "
        f"({pct(-plan.stop_pct)} sous le cours, 1 R = {money(plan.unit_risk, currency)})",
        f"  Risque assume    {money(plan.risk_amount, currency, 0)} = {pct(plan.risk_pct)} du capital",
        f"  Objectif 1       {money(plan.target1, currency)} (+1,5 R, gain {money(plan.gain_target1, currency, 0)} "
        f"sur le tiers vendu)",
        f"  Objectif 2       {money(plan.target2, currency)} (+3 R, gain {money(plan.gain_target2, currency, 0)} "
        f"sur le tiers vendu)",
        f"  Horizon estime   {plan.horizon_days} seances pour atteindre l'objectif 2",
    ]
    if plan.reasons:
        lines.append("  Pourquoi :")
        lines.extend(f"    + {reason}" for reason in plan.reasons)
    if plan.warnings:
        lines.append("  Points de vigilance :")
        lines.extend(f"    ! {warning}" for warning in plan.warnings)
    lines.append("  Quand vendre :")
    for rule in plan.exits:
        lines.append(f"    - {rule.nom} : si {rule.declencheur} -> {rule.action}")
    return "\n".join(lines)


def format_basket_summary(summary: dict, risk: dict, currency: str = "$") -> str:
    """Bilan du panier : capital engage, risque agrege, VaR."""
    lines = [
        f"  Lignes a acheter        {int(summary['lignes'])}",
        f"  Capital engage          {money(summary['capital_engage'], currency, 0)} "
        f"({pct(risk.get('exposition', float('nan')), 0)} du capital)",
        f"  Risque total (chaleur)  {money(summary['risque_total'], currency, 0)} "
        f"({pct(risk.get('chaleur', float('nan')))} du capital si tous les stops sautent)",
        f"  Gain si objectifs       {money(summary['gain_si_objectifs'], currency, 0)} "
        "(deux tiers de chaque ligne, solde en stop suiveur)",
        f"  Volatilite du panier    {pct(risk.get('volatilite_portefeuille', float('nan')), 0)} annualisee",
        f"  VaR 95 % a 1 jour       {money(risk.get('var_95_1j', float('nan')), currency, 0)}",
    ]
    horizon = summary.get("horizon_median", float("nan"))
    if isinstance(horizon, float) and math.isfinite(horizon):
        lines.append(f"  Horizon median          {horizon:.0f} seances")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Suivi des positions
# --------------------------------------------------------------------------
def format_reviews(reviews: list[PositionReview], currency: str = "$") -> str:
    """Tableau des verdicts sur les positions ouvertes."""
    order = {VERDICT_SELL: 0, VERDICT_TRIM: 1, VERDICT_TIGHTEN: 2, VERDICT_HOLD: 3}
    ordered = sorted(reviews, key=lambda review: order.get(review.verdict, 9))
    rows = []
    for review in ordered:
        rows.append(
            [
                review.ticker,
                review.verdict,
                money(review.entry_price, currency),
                money(review.price, currency),
                pct(review.pnl_pct),
                f"{review.r_multiple:+.2f} R",
                money(review.pnl, currency, 0),
                money(review.stop_price, currency),
                money(review.suggested_stop, currency),
                f"{review.days_held} j",
            ]
        )
    header = table(
        ["Titre", "Verdict", "Achat", "Cours", "Perf.", "R", "P&L", "Stop", "Stop suggere", "Duree"],
        rows,
        aligns="llrrrrrrrr",
    )
    details = []
    for review in ordered:
        if not review.actions and review.verdict == VERDICT_HOLD:
            continue
        details.append(f"\n  {review.ticker} — {review.verdict}")
        details.extend(f"    -> {action}" for action in review.actions)
        details.extend(f"       {note}" for note in review.notes)
    return header + ("\n" + "\n".join(details) if details else "")


# --------------------------------------------------------------------------
# Backtest
# --------------------------------------------------------------------------
def format_backtest(report: BacktestReport, per_ticker: dict[str, BacktestReport]) -> str:
    """Statistiques globales puis par titre."""
    if report.n_trades == 0:
        return "  Aucun trade declenche sur la periode."
    lines = [
        f"  Trades               {report.n_trades}",
        f"  Taux de reussite     {pct(report.win_rate, 0)}",
        f"  Gain moyen           {report.avg_r:+.2f} R (median {report.median_r:+.2f} R)",
        f"  Facteur de profit    {num(report.profit_factor)}",
        f"  Resultat cumule      {report.total_r:+.1f} R",
        f"  Pire serie           {report.max_drawdown_r:.1f} R",
        f"  Duree moyenne        {report.avg_days:.0f} seances",
        f"  Meilleur / pire      {report.best_r:+.2f} R / {report.worst_r:+.2f} R",
        f"  Performance capital  {pct(report.capital_return, 0)} "
        "(en risquant le meme montant a chaque trade)",
        f"  Acheter-conserver    {pct(report.buy_hold, 0)} en moyenne sur les memes titres",
    ]
    rows = []
    for ticker, item in sorted(per_ticker.items(), key=lambda kv: -(kv[1].total_r if kv[1].n_trades else 0)):
        if item.n_trades == 0:
            continue
        rows.append(
            [
                ticker,
                str(item.n_trades),
                pct(item.win_rate, 0),
                f"{item.avg_r:+.2f}",
                f"{item.total_r:+.1f}",
                f"{item.avg_days:.0f}",
            ]
        )
    return "\n".join(lines) + "\n\n" + table(
        ["Titre", "Trades", "Reussite", "R moyen", "R cumule", "Duree"], rows, aligns="lrrrrr"
    )


def format_actionable_count(plans: list[TradePlan]) -> str:
    actionable = [plan for plan in plans if is_actionable(plan)]
    if not actionable:
        return "  Aucun achat recommande aujourd'hui : le marche ne paie pas le risque pris."
    return f"  {len(actionable)} achat(s) recommande(s) aujourd'hui."
