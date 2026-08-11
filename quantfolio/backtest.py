"""Backtest walk-forward.

Regles du jeu, pour que le resultat veuille dire quelque chose :
  - le modele n'est jamais entraine sur des donnees posterieures a la decision ;
  - les labels a cheval sur la date de decision sont purges (horizon + embargo) ;
  - un ordre decide le jour J est execute au jour J+1 avec les prix de J+1 ;
  - frais et slippage sont preleves a chaque transaction.

Un backtest reste une simulation optimiste par nature : il ignore la liquidite
reelle, les fenetres de negociation et le biais du survivant de votre univers.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import Config
from .engine import Engine
from .metrics import summarize
from .model import Ranker, daily_information_coefficient
from .orders import OrderPlan
from .state import PortfolioState


@dataclass
class BacktestResult:
    equity: pd.Series
    benchmark: pd.Series | None = None
    turnover: pd.Series = field(default_factory=lambda: pd.Series(dtype="float64"))
    exposure: pd.Series = field(default_factory=lambda: pd.Series(dtype="float64"))
    weights: pd.DataFrame = field(default_factory=pd.DataFrame)
    trades: pd.DataFrame = field(default_factory=pd.DataFrame)
    scores: pd.Series = field(default_factory=lambda: pd.Series(dtype="float64"))
    ic: pd.Series = field(default_factory=lambda: pd.Series(dtype="float64"))
    summary: dict[str, float] = field(default_factory=dict)
    state: PortfolioState | None = None
    n_rebalances: int = 0
    n_trainings: int = 0

    def describe_ic(self) -> str:
        if self.ic.empty:
            return "IC indisponible"
        mean = float(self.ic.mean())
        std = float(self.ic.std(ddof=1)) if len(self.ic) > 1 else float("nan")
        t_stat = mean / (std / np.sqrt(len(self.ic))) if std and np.isfinite(std) and std > 0 else float("nan")
        return (
            f"IC moyen {mean:+.4f} | ecart-type {std:.4f} | "
            f"t-stat {t_stat:+.2f} | {len(self.ic)} dates"
        )


def _execution_prices(engine: Engine, date: pd.Timestamp, mode: str) -> pd.Series:
    panel = engine.load().panel
    frame = panel.open if mode == "next_open" else panel.close
    return frame.loc[date]


def _apply_execution(
    plan: OrderPlan,
    state: PortfolioState,
    prices: pd.Series,
    cost_bps: float,
    slippage_bps: float,
    date: pd.Timestamp,
    fractional_shares: bool = False,
) -> tuple[float, list[dict]]:
    """Execute un plan aux prix reels du jour d'execution.

    Les quantites ont ete decidees la veille sur la base de la derniere cloture
    connue : le prix d'execution differe, exactement comme dans la vraie vie.
    """
    traded = 0.0
    fills: list[dict] = []
    slip = slippage_bps / 10_000.0

    for order in plan.orders:
        reference = prices.get(order.ticker)
        if reference is None or not np.isfinite(reference) or reference <= 0:
            continue  # titre non cote ce jour-la : ordre abandonne
        price = float(reference) * (1.0 + slip if order.side == "BUY" else 1.0 - slip)

        if order.side == "SELL":
            shares = min(order.shares, state.shares(order.ticker))
        else:
            # Le prix d'execution differe de celui qui a servi a dimensionner
            # l'ordre : la veille, l'achat pouvait tenir dans la tresorerie et
            # ne plus y tenir aujourd'hui. On reduit alors la quantite.
            affordable = max(0.0, state.cash) / (price * (1.0 + cost_bps / 10_000.0))
            shares = min(order.shares, affordable)
            if shares < order.shares and not fractional_shares:
                shares = float(np.floor(shares))
        if shares <= 1e-9:
            continue

        notional = shares * price
        cost = notional * cost_bps / 10_000.0
        state.trade(
            order.ticker,
            shares if order.side == "BUY" else -shares,
            price,
            cost=cost,
            date=pd.Timestamp(date).strftime("%Y-%m-%d"),
            note=order.reason,
        )
        traded += notional
        fills.append(
            {
                "date": pd.Timestamp(date),
                "ticker": order.ticker,
                "side": order.side,
                "shares": shares,
                "price": price,
                "notional": notional,
                "cost": cost,
            }
        )
    return traded, fills


def run_backtest(
    engine: Engine, cfg: Config | None = None, verbose: bool = False, progress_every: int = 25
) -> BacktestResult:
    """Deroule la strategie jour par jour sur l'historique."""
    cfg = cfg or engine.cfg
    engine.ensure_prepared()
    market_data = engine.load()
    panel = market_data.panel
    dates = panel.dates

    backtest_cfg = cfg.backtest
    start = pd.Timestamp(backtest_cfg.start) if backtest_cfg.start else None
    end = pd.Timestamp(backtest_cfg.end) if backtest_cfg.end else None

    first = max(backtest_cfg.warmup_days, cfg.data.min_history)
    if start is not None:
        first = max(first, int(dates.searchsorted(start, side="left")))
    last = len(dates) - 1 if end is None else int(dates.searchsorted(end, side="right")) - 1
    if last <= first:
        raise ValueError(
            "Periode de backtest trop courte apres la phase de chauffe. "
            f"{len(dates)} seances disponibles, {first} consommees par le warmup "
            "(reduisez backtest.warmup_days / data.min_history ou allongez l'historique)."
        )

    state = PortfolioState(cash=float(cfg.portfolio.capital), currency=cfg.portfolio.currency)
    ranker: Ranker | None = None
    last_training_position = -10**9

    equity_records: dict[pd.Timestamp, float] = {}
    turnover_records: dict[pd.Timestamp, float] = {}
    exposure_records: dict[pd.Timestamp, float] = {}
    weight_records: dict[pd.Timestamp, pd.Series] = {}
    score_records: list[pd.Series] = []
    all_fills: list[dict] = []
    pending: tuple[pd.Timestamp, OrderPlan] | None = None
    n_rebalances = 0
    n_trainings = 0

    immediate = backtest_cfg.execution == "close"

    for position in range(first, last + 1):
        date = dates[position]
        close_prices = panel.close.loc[date]

        # 1. Execution des ordres decides lors d'une seance precedente.
        if pending is not None and pending[0] == date:
            prices = _execution_prices(engine, date, backtest_cfg.execution)
            traded, fills = _apply_execution(
                pending[1], state, prices, backtest_cfg.cost_bps,
                backtest_cfg.slippage_bps, date, cfg.portfolio.fractional_shares,
            )
            all_fills.extend(fills)
            equity_before = equity_records.get(dates[position - 1], state.equity(close_prices))
            turnover_records[date] = traded / equity_before if equity_before > 0 else 0.0
            pending = None

        # 2. Decision de rebalancement.
        is_rebalance = (position - first) % backtest_cfg.rebalance_days == 0
        if is_rebalance and position < last:
            if (position - last_training_position) >= cfg.model.retrain_every:
                candidate = engine.fit(as_of=date)
                if candidate.is_fitted:
                    ranker = candidate
                    n_trainings += 1
                last_training_position = position

            scores = engine.score_at(date, ranker)
            if not scores.empty:
                indexed = scores.dropna()
                if not indexed.empty:
                    score_records.append(
                        pd.Series(
                            indexed.to_numpy(),
                            index=pd.MultiIndex.from_product(
                                [[date], indexed.index], names=["date", "ticker"]
                            ),
                        )
                    )

                target = engine.target_at(
                    date, ranker, scores=scores, holdings=set(state.as_shares_dict())
                )
                exposure_records[date] = target.exposure.exposure
                weight_records[date] = target.weights

                from .orders import generate_orders  # import local : evite un cycle

                plan = generate_orders(
                    target_weights=target.weights,
                    prices=close_prices,
                    state=state,
                    cfg=cfg.portfolio,
                    scores=scores,
                    cost_bps=backtest_cfg.cost_bps,
                    date=date,
                )
                if plan.orders:
                    execution_date = date if immediate else dates[position + 1]
                    if immediate:
                        prices = panel.close.loc[date]
                        traded, fills = _apply_execution(
                            plan, state, prices, backtest_cfg.cost_bps,
                            backtest_cfg.slippage_bps, date,
                            cfg.portfolio.fractional_shares,
                        )
                        all_fills.extend(fills)
                        equity_now = state.equity(close_prices)
                        turnover_records[date] = traded / equity_now if equity_now > 0 else 0.0
                    else:
                        pending = (execution_date, plan)
                n_rebalances += 1
                if verbose and progress_every and n_rebalances % progress_every == 0:
                    print(
                        f"  {date:%Y-%m-%d} | capital {state.equity(close_prices):,.0f} "
                        f"| {len(target.holdings)} lignes"
                    )

        # 3. Valorisation de fin de seance.
        equity_records[date] = state.equity(close_prices)

    equity = pd.Series(equity_records).sort_index()
    equity.index.name = "date"

    benchmark = _benchmark_series(engine, equity)
    turnover = pd.Series(turnover_records).sort_index()
    exposure = pd.Series(exposure_records).sort_index()
    weights = pd.DataFrame(weight_records).T.sort_index() if weight_records else pd.DataFrame()
    trades = pd.DataFrame(all_fills)

    scores_long = pd.concat(score_records) if score_records else pd.Series(dtype="float64")
    ic = pd.Series(dtype="float64")
    if not scores_long.empty and engine.dataset is not None:
        ic = daily_information_coefficient(scores_long, engine.dataset.y)

    summary = summarize(equity, benchmark=benchmark, turnover=turnover, exposure=exposure)
    summary["nb_rebalancements"] = float(n_rebalances)
    summary["nb_transactions"] = float(len(trades))
    if not ic.empty:
        summary["ic_moyen"] = float(ic.mean())

    return BacktestResult(
        equity=equity,
        benchmark=benchmark,
        turnover=turnover,
        exposure=exposure,
        weights=weights,
        trades=trades,
        scores=scores_long,
        ic=ic,
        summary=summary,
        state=state,
        n_rebalances=n_rebalances,
        n_trainings=n_trainings,
    )


def _benchmark_series(engine: Engine, equity: pd.Series) -> pd.Series | None:
    """Reference de comparaison, rebasee sur le capital initial."""
    if equity.empty:
        return None
    market_data = engine.load()
    if market_data.benchmark is not None:
        series = market_data.benchmark.reindex(equity.index).ffill()
        name = market_data.benchmark_name or "benchmark"
    else:
        from .risk import market_index

        series = market_index(market_data.panel.close, market_data.panel.available)
        series = series.reindex(equity.index).ffill()
        name = "univers equipondere"
    series = series.dropna()
    if series.empty or series.iloc[0] <= 0:
        return None
    rebased = series / series.iloc[0] * float(equity.iloc[0])
    rebased.name = name
    return rebased.reindex(equity.index).ffill()
