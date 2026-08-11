"""Mise en forme des sorties pour le terminal."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .orders import OrderPlan
from .portfolio import PortfolioTarget
from .state import PortfolioState


def table(frame: pd.DataFrame, formats: dict[str, str] | None = None, indent: str = "  ") -> str:
    """Rend un DataFrame en tableau texte aligne."""
    if frame is None or frame.empty:
        return f"{indent}(vide)"
    formats = formats or {}
    rendered = frame.copy()
    for column in rendered.columns:
        spec = formats.get(column)
        if spec:
            rendered[column] = rendered[column].map(
                lambda v, spec=spec: "n/a"
                if v is None or (isinstance(v, float) and not np.isfinite(v))
                else format(v, spec)
            )
        else:
            rendered[column] = rendered[column].astype(str)

    headers = [str(c) for c in rendered.columns]
    widths = [
        max(len(headers[i]), *(len(v) for v in rendered.iloc[:, i])) for i in range(len(headers))
    ]
    lines = [indent + "  ".join(h.ljust(widths[i]) for i, h in enumerate(headers))]
    lines.append(indent + "  ".join("-" * w for w in widths))
    for _, row in rendered.iterrows():
        lines.append(indent + "  ".join(str(v).ljust(widths[i]) for i, v in enumerate(row)))
    return "\n".join(lines)


def title(text: str, char: str = "=") -> str:
    return f"\n{text}\n{char * len(text)}"


def format_orders(plan: OrderPlan, currency: str = "USD") -> str:
    """Affiche les ordres du jour."""
    lines = [title("ORDRES A PASSER")]
    if not plan.orders:
        lines.append("  Aucun ordre : le portefeuille est deja conforme a la cible.")
    else:
        frame = plan.to_frame()[
            ["side", "ticker", "shares", "price", "notional", "current_weight", "target_weight", "score", "reason"]
        ]
        frame = frame.rename(
            columns={
                "side": "sens",
                "shares": "qte",
                "price": "cours",
                "notional": "montant",
                "current_weight": "poids act.",
                "target_weight": "poids cible",
                "reason": "motif",
            }
        )
        frame["sens"] = frame["sens"].map({"BUY": "ACHAT", "SELL": "VENTE"}).fillna(frame["sens"])
        lines.append(
            table(
                frame,
                {
                    "qte": ",.4g",
                    "cours": ",.2f",
                    "montant": ",.0f",
                    "poids act.": ".1%",
                    "poids cible": ".1%",
                    "score": "+.2f",
                },
            )
        )
        buys = sum(o.notional for o in plan.buys)
        sells = sum(o.notional for o in plan.sells)
        lines.append(
            f"\n  {len(plan.buys)} achat(s) pour {buys:,.0f} {currency} | "
            f"{len(plan.sells)} vente(s) pour {sells:,.0f} {currency} | "
            f"frais estimes {plan.estimated_costs:,.0f} {currency}"
        )
    lines.append(
        f"  Capital {plan.equity:,.0f} {currency} | "
        f"liquidites {plan.cash_before:,.0f} -> {plan.cash_after:,.0f} {currency} | "
        f"rotation {plan.turnover:.1%}"
    )
    for warning in plan.warnings:
        lines.append(f"  /!\\ {warning}")
    return "\n".join(lines)


def format_target(target: PortfolioTarget, scores: pd.Series | None = None, top: int = 15) -> str:
    """Affiche le portefeuille cible et le classement des titres."""
    lines = [title("PORTEFEUILLE CIBLE")]
    exposure = target.exposure
    lines.append(
        f"  Exposition visee {target.invested:.1%} | liquidites {target.cash_weight:.1%} | "
        f"{exposure.reason}"
    )
    if not exposure.regime_on:
        lines.append("  Regime de marche defavorable : exposition volontairement reduite.")

    if target.detail.empty:
        lines.append("  Aucun titre ne passe le filtre : rester en liquidites.")
    else:
        detail = target.detail.reset_index().rename(
            columns={"index": "ticker", "score": "score", "vol": "vol ann.", "weight": "poids"}
        )
        detail = detail[["ticker", "score", "vol ann.", "poids"]]
        lines.append(table(detail, {"score": "+.2f", "vol ann.": ".1%", "poids": ".1%"}))

    if scores is not None and not scores.dropna().empty:
        ranked = scores.dropna().sort_values(ascending=False)
        lines.append(title("CLASSEMENT DU JOUR", "-"))
        best = ranked.head(top).rename("score").reset_index()
        best.columns = ["ticker", "score"]
        best.insert(0, "rang", range(1, len(best) + 1))
        lines.append(table(best, {"score": "+.3f"}))
        if len(ranked) > top:
            worst = ranked.tail(3).rename("score").reset_index()
            worst.columns = ["ticker", "score"]
            lines.append("  Bas de classement (a eviter / vendre) :")
            lines.append(table(worst, {"score": "+.3f"}, indent="    "))
    return "\n".join(lines)


def format_positions(state: PortfolioState, prices: pd.Series | None = None) -> str:
    """Affiche le portefeuille detenu."""
    lines = [title("PORTEFEUILLE ACTUEL")]
    frame = state.to_frame(prices)
    if frame.empty:
        lines.append("  Aucune position ouverte.")
    else:
        display = frame.rename(
            columns={
                "shares": "qte",
                "avg_price": "PRU",
                "last_price": "cours",
                "value": "valeur",
                "pnl": "P&L",
                "pnl_pct": "P&L %",
            }
        )
        lines.append(
            table(
                display,
                {"qte": ",.4g", "PRU": ",.2f", "cours": ",.2f", "valeur": ",.0f",
                 "P&L": "+,.0f", "P&L %": "+.1%"},
            )
        )
    equity = state.equity(prices) if prices is not None else state.cash
    lines.append(
        f"\n  Liquidites {state.cash:,.0f} {state.currency} | "
        f"valeur totale {equity:,.0f} {state.currency} | "
        f"P&L realise {state.realized_pnl:+,.0f} | frais cumules {state.total_costs:,.0f}"
    )
    if state.updated_at:
        lines.append(f"  Derniere mise a jour : {state.updated_at}")
    return "\n".join(lines)
