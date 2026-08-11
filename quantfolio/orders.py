"""Traduction du portefeuille cible en ordres concrets (acheter / vendre)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import PortfolioConfig
from .portfolio import apply_no_trade_band, current_weights
from .state import PortfolioState

EPSILON = 1e-9


@dataclass
class Order:
    ticker: str
    side: str  # BUY | SELL
    shares: float
    price: float
    notional: float
    current_weight: float = 0.0
    target_weight: float = 0.0
    score: float = float("nan")
    reason: str = ""

    def as_dict(self) -> dict:
        return {
            "ticker": self.ticker,
            "side": self.side,
            "shares": round(self.shares, 6),
            "price": round(self.price, 4),
            "notional": round(self.notional, 2),
            "current_weight": round(self.current_weight, 4),
            "target_weight": round(self.target_weight, 4),
            "score": None if not np.isfinite(self.score) else round(float(self.score), 4),
            "reason": self.reason,
        }


@dataclass
class OrderPlan:
    """Ensemble d'ordres a passer, plus le contexte qui les explique."""

    orders: list[Order] = field(default_factory=list)
    equity: float = 0.0
    cash_before: float = 0.0
    cash_after: float = 0.0
    estimated_costs: float = 0.0
    turnover: float = 0.0
    date: pd.Timestamp | None = None
    warnings: list[str] = field(default_factory=list)

    @property
    def buys(self) -> list[Order]:
        return [o for o in self.orders if o.side == "BUY"]

    @property
    def sells(self) -> list[Order]:
        return [o for o in self.orders if o.side == "SELL"]

    def to_frame(self) -> pd.DataFrame:
        if not self.orders:
            return pd.DataFrame(
                columns=[
                    "ticker", "side", "shares", "price", "notional",
                    "current_weight", "target_weight", "score", "reason",
                ]
            )
        return pd.DataFrame([o.as_dict() for o in self.orders])

    def as_dict(self) -> dict:
        return {
            "date": None if self.date is None else pd.Timestamp(self.date).strftime("%Y-%m-%d"),
            "equity": round(self.equity, 2),
            "cash_before": round(self.cash_before, 2),
            "cash_after": round(self.cash_after, 2),
            "estimated_costs": round(self.estimated_costs, 2),
            "turnover": round(self.turnover, 4),
            "orders": [o.as_dict() for o in self.orders],
            "warnings": self.warnings,
        }


def _round_shares(quantity: float, fractional: bool) -> float:
    """Arrondit vers zero : on ne depasse jamais la taille visee."""
    if fractional:
        return round(quantity, 6)
    return float(math.floor(quantity)) if quantity > 0 else float(math.ceil(quantity))


def generate_orders(
    target_weights: pd.Series,
    prices: pd.Series,
    state: PortfolioState,
    cfg: PortfolioConfig,
    scores: pd.Series | None = None,
    cost_bps: float = 0.0,
    date: pd.Timestamp | None = None,
    apply_band: bool = True,
) -> OrderPlan:
    """Compare le portefeuille detenu au portefeuille cible et en deduit les ordres.

    Les ventes sont traitees avant les achats afin que le produit des cessions
    finance les acquisitions du meme jour.
    """
    prices = prices.astype("float64")
    prices = prices[prices.notna() & (prices > 0)]

    plan = OrderPlan(date=date, cash_before=state.cash)

    unpriced = state.unpriced(prices)
    if unpriced:
        plan.warnings.append(
            "Positions sans cours disponible, ignorees dans le calcul : "
            + ", ".join(unpriced)
        )

    equity = state.equity(prices)
    plan.equity = equity
    if equity <= 0:
        plan.warnings.append("Capital nul ou negatif : aucun ordre genere.")
        plan.cash_after = state.cash
        return plan

    # Univers d'action = titres cibles + positions detenues et cotees.
    held = [t for t in state.as_shares_dict() if t in prices.index]
    universe = sorted(set(target_weights.index) | set(held))
    targets = target_weights.reindex(universe).fillna(0.0)
    targets = targets[targets.index.isin(prices.index)]
    universe = list(targets.index)

    current = current_weights(state.as_shares_dict(), prices.reindex(universe), state.cash)
    if apply_band:
        targets = apply_no_trade_band(targets, current, cfg.no_trade_band)

    delta_notional = (targets - current) * equity
    scores = scores if scores is not None else pd.Series(dtype="float64")

    # --- ventes ------------------------------------------------------------
    cash = state.cash
    costs = 0.0
    traded_notional = 0.0

    for ticker in delta_notional[delta_notional < 0].sort_values().index:
        price = float(prices[ticker])
        held_shares = state.shares(ticker)
        if held_shares <= EPSILON:
            continue  # pas de vente a decouvert
        wanted = -float(delta_notional[ticker])
        # Sortie complete demandee : on solde la ligne, sans reliquat.
        if targets[ticker] <= EPSILON:
            shares = held_shares
        else:
            shares = min(_round_shares(wanted / price, cfg.fractional_shares), held_shares)
        if shares <= EPSILON:
            continue
        notional = shares * price
        if notional < cfg.min_order_notional and targets[ticker] > EPSILON:
            continue  # trop petit pour valoir les frais (sauf sortie totale)

        cost = notional * cost_bps / 10_000.0
        cash += notional - cost
        costs += cost
        traded_notional += notional
        plan.orders.append(
            Order(
                ticker=ticker,
                side="SELL",
                shares=shares,
                price=price,
                notional=notional,
                current_weight=float(current.get(ticker, 0.0)),
                target_weight=float(targets[ticker]),
                score=float(scores.get(ticker, float("nan"))),
                reason="sortie complete" if targets[ticker] <= EPSILON else "allegement",
            )
        )

    # --- achats ------------------------------------------------------------
    buy_candidates = delta_notional[delta_notional > 0].sort_values(ascending=False)
    # On garde toujours le coussin de liquidites demande.
    spendable = max(0.0, cash - cfg.cash_buffer * equity)

    for ticker in buy_candidates.index:
        price = float(prices[ticker])
        wanted = float(delta_notional[ticker])
        budget = min(wanted, spendable / (1.0 + cost_bps / 10_000.0))
        if budget <= 0:
            continue
        shares = _round_shares(budget / price, cfg.fractional_shares)
        if shares <= EPSILON:
            continue
        notional = shares * price
        if notional < cfg.min_order_notional:
            continue

        cost = notional * cost_bps / 10_000.0
        if notional + cost > spendable + EPSILON:
            continue
        cash -= notional + cost
        spendable -= notional + cost
        costs += cost
        traded_notional += notional
        plan.orders.append(
            Order(
                ticker=ticker,
                side="BUY",
                shares=shares,
                price=price,
                notional=notional,
                current_weight=float(current.get(ticker, 0.0)),
                target_weight=float(targets[ticker]),
                score=float(scores.get(ticker, float("nan"))),
                reason="ouverture" if current.get(ticker, 0.0) <= EPSILON else "renforcement",
            )
        )

    if buy_candidates.sum() > 0 and spendable <= 0 and not plan.buys:
        plan.warnings.append(
            "Liquidites insuffisantes pour executer les achats vises."
        )

    plan.cash_after = cash
    plan.estimated_costs = costs
    plan.turnover = traded_notional / equity if equity > 0 else 0.0
    plan.orders.sort(key=lambda o: (o.side != "SELL", -o.notional))
    return plan


def execute_plan(
    plan: OrderPlan,
    state: PortfolioState,
    cost_bps: float = 0.0,
    date: pd.Timestamp | None = None,
) -> PortfolioState:
    """Applique les ordres a l'etat du portefeuille (backtest ou execution reelle).

    `cost_bps` doit reprendre la valeur utilisee lors de la generation du plan,
    sans quoi les liquidites resultantes ne correspondront pas au plan affiche.
    """
    stamp = pd.Timestamp(date or plan.date or pd.Timestamp.now()).strftime("%Y-%m-%d")
    for order in plan.orders:
        signed = order.shares if order.side == "BUY" else -order.shares
        cost = order.notional * cost_bps / 10_000.0
        state.trade(order.ticker, signed, order.price, cost=cost, date=stamp, note=order.reason)
    return state
