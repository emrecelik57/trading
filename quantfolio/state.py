"""Etat du portefeuille : positions detenues, liquidites, historique."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass
class Position:
    shares: float = 0.0
    avg_price: float = 0.0

    def value(self, price: float) -> float:
        return self.shares * price

    def unrealized(self, price: float) -> float:
        return (price - self.avg_price) * self.shares


@dataclass
class PortfolioState:
    """Photographie du portefeuille, persistee en JSON."""

    cash: float = 0.0
    currency: str = "USD"
    positions: dict[str, Position] = field(default_factory=dict)
    updated_at: str | None = None
    realized_pnl: float = 0.0
    total_costs: float = 0.0
    history: list[dict[str, Any]] = field(default_factory=list)

    # -- lecture ------------------------------------------------------------
    def shares(self, ticker: str) -> float:
        position = self.positions.get(ticker)
        return position.shares if position else 0.0

    def as_shares_dict(self) -> dict[str, float]:
        return {t: p.shares for t, p in self.positions.items() if abs(p.shares) > 1e-12}

    def market_value(self, prices: pd.Series) -> float:
        total = 0.0
        for ticker, position in self.positions.items():
            price = prices.get(ticker)
            if price is not None and pd.notna(price):
                total += position.value(float(price))
        return total

    def equity(self, prices: pd.Series) -> float:
        return self.cash + self.market_value(prices)

    def unpriced(self, prices: pd.Series) -> list[str]:
        """Positions detenues dont on n'a pas de prix (titre hors univers...)."""
        return sorted(
            ticker
            for ticker, position in self.positions.items()
            if abs(position.shares) > 1e-12
            and (ticker not in prices.index or pd.isna(prices.get(ticker)))
        )

    def to_frame(self, prices: pd.Series | None = None) -> pd.DataFrame:
        rows = []
        for ticker, position in sorted(self.positions.items()):
            if abs(position.shares) <= 1e-12:
                continue
            price = float(prices.get(ticker)) if prices is not None and ticker in prices.index else float("nan")
            rows.append(
                {
                    "ticker": ticker,
                    "shares": position.shares,
                    "avg_price": position.avg_price,
                    "last_price": price,
                    "value": position.shares * price,
                    "pnl": position.unrealized(price),
                    "pnl_pct": (price / position.avg_price - 1.0) if position.avg_price else float("nan"),
                }
            )
        return pd.DataFrame(rows)

    # -- ecriture -----------------------------------------------------------
    def trade(self, ticker: str, shares: float, price: float, cost: float = 0.0,
              date: str | None = None, note: str = "") -> None:
        """Enregistre une transaction. `shares` > 0 achat, < 0 vente."""
        if shares == 0:
            return
        position = self.positions.setdefault(ticker, Position())
        notional = shares * price

        if shares > 0:
            new_shares = position.shares + shares
            if new_shares > 0:
                # Prix de revient moyen pondere (uniquement sur les achats).
                position.avg_price = (
                    position.avg_price * max(position.shares, 0.0) + notional
                ) / new_shares
            position.shares = new_shares
        else:
            sold = min(-shares, position.shares)
            self.realized_pnl += (price - position.avg_price) * sold
            position.shares += shares
            if abs(position.shares) <= 1e-9:
                position.shares = 0.0
                position.avg_price = 0.0

        self.cash -= notional + cost
        self.total_costs += cost
        if position.shares <= 1e-12:
            self.positions.pop(ticker, None)

        self.history.append(
            {
                "date": date or pd.Timestamp.now().strftime("%Y-%m-%d"),
                "ticker": ticker,
                "side": "BUY" if shares > 0 else "SELL",
                "shares": abs(shares),
                "price": price,
                "notional": abs(notional),
                "cost": cost,
                "note": note,
            }
        )
        self.updated_at = pd.Timestamp.now().isoformat(timespec="seconds")

    def set_position(self, ticker: str, shares: float, avg_price: float) -> None:
        """Force une position (initialisation depuis un portefeuille existant)."""
        if shares <= 0:
            self.positions.pop(ticker, None)
        else:
            self.positions[ticker] = Position(shares=float(shares), avg_price=float(avg_price))
        self.updated_at = pd.Timestamp.now().isoformat(timespec="seconds")

    # -- persistance --------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["positions"] = {t: asdict(p) for t, p in self.positions.items()}
        return payload

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Ecriture atomique : on ne veut pas d'un etat tronque en cas de crash.
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2, ensure_ascii=False)
        temporary.replace(path)
        return path

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PortfolioState":
        payload = dict(payload or {})
        positions = {
            ticker: Position(**values)
            for ticker, values in (payload.pop("positions", {}) or {}).items()
        }
        known = {"cash", "currency", "updated_at", "realized_pnl", "total_costs", "history"}
        kwargs = {k: v for k, v in payload.items() if k in known}
        return cls(positions=positions, **kwargs)

    @classmethod
    def load(cls, path: str | Path) -> "PortfolioState":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"Aucun portefeuille enregistre dans {path}. "
                "Lancez `quantfolio init-portfolio --cash MONTANT` d'abord."
            )
        with path.open("r", encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))

    @classmethod
    def load_or_create(cls, path: str | Path, cash: float, currency: str = "USD") -> "PortfolioState":
        try:
            return cls.load(path)
        except FileNotFoundError:
            return cls(cash=float(cash), currency=currency)
