"""Interface en ligne de commande."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from .backtest import run_backtest
from .config import Config, ConfigError, load_config
from .data.base import DataError
from .engine import Engine
from .metrics import format_summary
from .model import Ranker, daily_information_coefficient, feature_importance
from .orders import generate_orders
from .report import format_orders, format_positions, format_target, table, title
from .state import PortfolioState
from .templates import DEFAULT_CONFIG_YAML

DISCLAIMER = (
    "Outil d'aide a la decision : les scores sont des estimations statistiques, "
    "pas des certitudes. Aucun ordre n'est transmis a un courtier."
)


# --------------------------------------------------------------------------
# Commandes
# --------------------------------------------------------------------------
def cmd_init(args: argparse.Namespace) -> int:
    target = Path(args.output)
    if target.exists() and not args.force:
        print(f"{target} existe deja. Utilisez --force pour l'ecraser.")
        return 1
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(DEFAULT_CONFIG_YAML, encoding="utf-8")
    print(f"Configuration ecrite dans {target}")
    print("Editez l'univers et data.provider, puis lancez : quantfolio backtest")
    return 0


def cmd_fetch(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    engine = Engine(cfg)
    market_data = engine.load(refresh=args.refresh, verbose=True)
    panel = market_data.panel
    print(
        f"{len(panel.tickers)} titres, {len(panel)} seances "
        f"du {panel.dates.min():%Y-%m-%d} au {panel.dates.max():%Y-%m-%d}."
    )
    coverage = panel.available.sum().sort_values()
    thin = coverage[coverage < cfg.data.min_history]
    if not thin.empty:
        print(
            "Historique insuffisant (ces titres seront ignores) : "
            + ", ".join(f"{t} ({int(n)}j)" for t, n in thin.items())
        )
    if market_data.benchmark is not None:
        print(f"Benchmark {market_data.benchmark_name} : {len(market_data.benchmark.dropna())} points.")
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    if cfg.model.kind == "rules":
        print("model.kind = rules : aucun apprentissage necessaire.")
        return 0

    engine = Engine(cfg)
    engine.prepare(refresh=args.refresh, verbose=True)
    as_of = pd.Timestamp(args.date) if args.date else engine.last_date()
    ranker = engine.fit(as_of=as_of, verbose=True)

    if not ranker.is_fitted:
        print(f"Echec de l'entrainement : {ranker.report.reason}")
        print(
            "Allongez l'historique (data.start), elargissez l'univers, "
            "ou baissez model.min_train_rows."
        )
        return 1

    path = ranker.save(cfg.model_path)
    print(f"Modele sauvegarde dans {path}")

    if args.importance:
        X, y = engine.training_sample(as_of)
        importance = feature_importance(ranker, X, y)
        if not importance.empty:
            print(title("IMPORTANCE DES INDICATEURS (permutation)", "-"))
            frame = importance.head(12).rename("importance").reset_index()
            frame.columns = ["indicateur", "importance"]
            print(table(frame, {"importance": ".5f"}))
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    """Mesure la qualite de classement du modele hors echantillon."""
    cfg = load_config(args.config)
    engine = Engine(cfg)
    engine.prepare(verbose=True)
    dataset = engine.dataset
    assert dataset is not None

    dates = engine.load().panel.dates
    first = max(cfg.backtest.warmup_days, cfg.data.min_history)
    if cfg.backtest.start:
        first = max(first, int(dates.searchsorted(pd.Timestamp(cfg.backtest.start), side="left")))
    step = max(1, args.step)
    evaluation_dates = dates[first : len(dates) - cfg.model.horizon : step]
    if len(evaluation_dates) == 0:
        print("Pas assez d'historique pour evaluer le modele.")
        return 1

    print(f"Evaluation sur {len(evaluation_dates)} dates (pas de {step} seances)...")
    ranker: Ranker | None = None
    last_train = -10**9
    records = []
    for position, date in enumerate(evaluation_dates):
        absolute = int(dates.searchsorted(date))
        if (absolute - last_train) >= cfg.model.retrain_every:
            candidate = engine.fit(as_of=date)
            if candidate.is_fitted:
                ranker = candidate
            last_train = absolute
        scores = engine.score_at(date, ranker).dropna()
        if scores.empty:
            continue
        records.append(
            pd.Series(
                scores.to_numpy(),
                index=pd.MultiIndex.from_product([[date], scores.index], names=["date", "ticker"]),
            )
        )
        if args.verbose and position % 25 == 0:
            print(f"  {date:%Y-%m-%d}")

    if not records:
        print("Aucun score produit : verifiez l'univers et l'historique.")
        return 1

    scores_long = pd.concat(records)
    ic = daily_information_coefficient(scores_long, dataset.y)
    print(title("QUALITE DU CLASSEMENT (hors echantillon)"))
    if ic.empty:
        print("  IC non calculable.")
        return 1
    mean, std = float(ic.mean()), float(ic.std(ddof=1))
    t_stat = mean / (std / len(ic) ** 0.5) if std > 0 else float("nan")
    print(f"  IC moyen        {mean:+.4f}")
    print(f"  Ecart-type      {std:.4f}")
    print(f"  t-stat          {t_stat:+.2f}")
    print(f"  IC > 0          {float((ic > 0).mean()):.1%} des dates")
    print(f"  Dates evaluees  {len(ic)}")
    print(
        "\n  Reperes : IC moyen de 0.02-0.05 avec t-stat > 2 = signal exploitable. "
        "\n  Un IC > 0.15 sur donnees reelles doit faire suspecter une fuite de donnees."
    )
    return 0


def cmd_backtest(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    if args.start:
        cfg.backtest.start = args.start
    if args.end:
        cfg.backtest.end = args.end

    engine = Engine(cfg)
    engine.prepare(refresh=args.refresh, verbose=True)
    print(
        f"Backtest du {cfg.backtest.start or 'debut'} au {cfg.backtest.end or 'fin'} "
        f"| rebalancement tous les {cfg.backtest.rebalance_days} jours "
        f"| execution {cfg.backtest.execution}"
    )
    result = run_backtest(engine, cfg, verbose=True)

    print(title("PERFORMANCE"))
    print(format_summary(result.summary))
    print(f"\n  {result.describe_ic()}")
    print(f"  {result.n_trainings} entrainements, {result.n_rebalances} rebalancements")

    if result.benchmark is not None and not result.benchmark.empty:
        final = float(result.equity.iloc[-1])
        reference = float(result.benchmark.iloc[-1])
        verdict = "au-dessus" if final > reference else "en-dessous"
        print(
            f"  Capital final {final:,.0f} contre {reference:,.0f} pour "
            f"{result.benchmark.name} ({verdict})."
        )

    output_dir = cfg.output_path
    output_dir.mkdir(parents=True, exist_ok=True)
    curve = pd.DataFrame({"equity": result.equity})
    if result.benchmark is not None:
        curve["benchmark"] = result.benchmark
    curve.to_csv(output_dir / "backtest_equity.csv")
    if not result.trades.empty:
        result.trades.to_csv(output_dir / "backtest_trades.csv", index=False)
    if not result.weights.empty:
        result.weights.to_csv(output_dir / "backtest_weights.csv")
    with (output_dir / "backtest_summary.json").open("w", encoding="utf-8") as fh:
        json.dump({k: _jsonable(v) for k, v in result.summary.items()}, fh, indent=2)
    print(f"\n  Details ecrits dans {output_dir}/")
    print(f"\n{DISCLAIMER}")
    return 0


def cmd_daily(args: argparse.Namespace) -> int:
    """Recommandations du jour : quoi acheter, quoi vendre, en quelle quantite."""
    cfg = load_config(args.config)
    engine = Engine(cfg)
    engine.prepare(refresh=args.refresh, verbose=args.verbose)

    date = pd.Timestamp(args.date) if args.date else engine.last_date()
    dates = engine.load().panel.dates
    if date not in dates:
        position = int(dates.searchsorted(date, side="right")) - 1
        if position < 0:
            print(f"Aucune seance disponible avant le {date:%Y-%m-%d}.")
            return 1
        date = dates[position]

    ranker = _load_or_train(cfg, engine, date, retrain=args.retrain)
    scores = engine.score_at(date, ranker)
    if scores.dropna().empty:
        print(
            "Aucun score disponible a cette date : historique insuffisant "
            f"(il faut au moins {cfg.data.min_history} seances par titre)."
        )
        return 1

    prices = engine.load().panel.close.loc[date]
    state = PortfolioState.load_or_create(
        cfg.state_path, cfg.portfolio.capital, cfg.portfolio.currency
    )
    target = engine.target_at(
        date, ranker, scores=scores, holdings=set(state.as_shares_dict())
    )

    plan = generate_orders(
        target_weights=target.weights,
        prices=prices,
        state=state,
        cfg=cfg.portfolio,
        scores=scores,
        cost_bps=cfg.backtest.cost_bps,
        date=date,
    )

    print(title(f"SEANCE DU {date:%Y-%m-%d}"))
    if ranker is not None and ranker.is_fitted:
        print(f"  Modele : {cfg.model.kind}, {ranker.report.describe()}")
    else:
        print(f"  Modele : score a base de regles uniquement ({cfg.model.kind}).")
    print(format_positions(state, prices))
    print(format_target(target, scores))
    print(format_orders(plan, cfg.portfolio.currency))

    output_dir = cfg.output_path
    output_dir.mkdir(parents=True, exist_ok=True)
    orders_file = output_dir / f"orders_{date:%Y-%m-%d}.json"
    with orders_file.open("w", encoding="utf-8") as fh:
        json.dump(plan.as_dict(), fh, indent=2, ensure_ascii=False)
    print(f"\n  Ordres enregistres dans {orders_file}")
    if plan.orders:
        print(
            "  Une fois passes chez votre courtier, enregistrez-les avec :\n"
            f"    quantfolio apply --file {orders_file}"
        )
    print(f"\n{DISCLAIMER}")
    return 0


def cmd_apply(args: argparse.Namespace) -> int:
    """Enregistre l'execution d'un plan d'ordres dans le portefeuille."""
    cfg = load_config(args.config)
    path = Path(args.file)
    if not path.exists():
        print(f"Fichier introuvable : {path}")
        return 1
    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)

    orders = payload.get("orders", [])
    if not orders:
        print("Aucun ordre dans ce fichier.")
        return 0

    state = PortfolioState.load_or_create(
        cfg.state_path, cfg.portfolio.capital, cfg.portfolio.currency
    )
    date = payload.get("date") or pd.Timestamp.now().strftime("%Y-%m-%d")
    cost_bps = args.cost_bps if args.cost_bps is not None else cfg.backtest.cost_bps

    print(f"Application de {len(orders)} ordre(s) du {date} :")
    for order in orders:
        side = order["side"]
        shares = float(order["shares"])
        price = float(order["price"])
        signed = shares if side == "BUY" else -shares
        cost = abs(shares * price) * cost_bps / 10_000.0
        state.trade(order["ticker"], signed, price, cost=cost, date=date, note="execute")
        print(f"  {side:<4} {order['ticker']:<6} {shares:>10,.4g} @ {price:,.2f}")

    if args.dry_run:
        print("\n--dry-run : aucun changement enregistre.")
        return 0
    state.save(cfg.state_path)
    print(f"\nPortefeuille mis a jour : {cfg.state_path}")
    print(
        "  Si vos prix d'execution reels different, editez le JSON avant "
        "d'appliquer : ce sont eux qui font foi pour le prix de revient."
    )
    return 0


def cmd_portfolio(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    try:
        state = PortfolioState.load(cfg.state_path)
    except FileNotFoundError as exc:
        print(exc)
        return 1

    prices = None
    if not args.offline:
        try:
            engine = Engine(cfg)
            prices = engine.load().panel.last_prices()
        except DataError as exc:
            print(f"Cours indisponibles ({exc}), affichage sans valorisation.")

    print(format_positions(state, prices))
    if args.history and state.history:
        print(title("HISTORIQUE DES TRANSACTIONS", "-"))
        frame = pd.DataFrame(state.history).tail(args.history)
        print(table(frame, {"shares": ",.4g", "price": ",.2f", "notional": ",.0f", "cost": ",.2f"}))
    return 0


def cmd_init_portfolio(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    path = cfg.state_path
    if path.exists() and not args.force:
        print(f"{path} existe deja. Utilisez --force pour repartir de zero.")
        return 1
    cash = args.cash if args.cash is not None else cfg.portfolio.capital
    state = PortfolioState(cash=float(cash), currency=cfg.portfolio.currency)
    state.updated_at = pd.Timestamp.now().isoformat(timespec="seconds")
    state.save(path)
    print(f"Portefeuille initialise avec {cash:,.0f} {cfg.portfolio.currency} dans {path}")
    print("Ajoutez vos positions existantes avec : quantfolio set-position TICKER --shares N --price P")
    return 0


def cmd_set_position(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    state = PortfolioState.load_or_create(
        cfg.state_path, cfg.portfolio.capital, cfg.portfolio.currency
    )
    state.set_position(args.ticker, args.shares, args.price)
    if args.cash is not None:
        state.cash = float(args.cash)
    state.save(cfg.state_path)
    print(f"Position {args.ticker} : {args.shares:g} titres au PRU {args.price:,.2f}")
    print(f"Liquidites : {state.cash:,.2f} {state.currency}")
    return 0


# --------------------------------------------------------------------------
# Utilitaires
# --------------------------------------------------------------------------
def _load_or_train(cfg: Config, engine: Engine, date: pd.Timestamp, retrain: bool) -> Ranker | None:
    """Charge le modele sauvegarde, ou l'entraine si besoin."""
    if cfg.model.kind == "rules":
        return None

    path = cfg.model_path
    if path.exists() and not retrain:
        try:
            ranker = Ranker.load(path, cfg.model)
            trained_at = ranker.report.train_end
            if trained_at is not None:
                age = (pd.Timestamp(date) - pd.Timestamp(trained_at)).days
                if age > cfg.model.retrain_every * 2:
                    print(
                        f"  Modele entraine il y a {age} jours : "
                        "relancez `quantfolio train` (ou --retrain) pour le rafraichir."
                    )
            return ranker
        except Exception as exc:
            print(f"  Modele illisible ({exc}), reentrainement...")

    ranker = engine.fit(as_of=date, verbose=True)
    if ranker.is_fitted:
        ranker.save(path)
    else:
        print(
            f"  Modele non entraine ({ranker.report.reason}) : "
            "repli sur le score a base de regles."
        )
    return ranker


def _jsonable(value):
    if isinstance(value, float):
        return None if pd.isna(value) else round(value, 6)
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="quantfolio",
        description=(
            "Modele de scoring d'actions et gestion de portefeuille : "
            "recommandations d'achat/vente quotidiennes."
        ),
    )
    parser.add_argument("--config", help="Chemin du fichier de configuration YAML")
    subparsers = parser.add_subparsers(dest="command", required=True)

    init = subparsers.add_parser("init", help="Cree un fichier de configuration")
    init.add_argument("--output", default="config/config.yaml")
    init.add_argument("--force", action="store_true")
    init.set_defaults(func=cmd_init)

    fetch = subparsers.add_parser("fetch", help="Telecharge et met en cache les donnees")
    fetch.add_argument("--refresh", action="store_true", help="Ignore le cache")
    fetch.set_defaults(func=cmd_fetch)

    train = subparsers.add_parser("train", help="Entraine le modele et le sauvegarde")
    train.add_argument("--date", help="Entrainer avec l'information disponible a cette date")
    train.add_argument("--refresh", action="store_true")
    train.add_argument("--importance", action="store_true", help="Affiche l'importance des indicateurs")
    train.set_defaults(func=cmd_train)

    evaluate = subparsers.add_parser("evaluate", help="Mesure la qualite du classement (IC)")
    evaluate.add_argument("--step", type=int, default=5, help="Pas d'evaluation, en seances")
    evaluate.add_argument("--verbose", action="store_true")
    evaluate.set_defaults(func=cmd_evaluate)

    backtest = subparsers.add_parser("backtest", help="Simule la strategie sur l'historique")
    backtest.add_argument("--start")
    backtest.add_argument("--end")
    backtest.add_argument("--refresh", action="store_true")
    backtest.set_defaults(func=cmd_backtest)

    daily = subparsers.add_parser("daily", help="Recommandations du jour")
    daily.add_argument("--date", help="Se placer a une date passee (format AAAA-MM-JJ)")
    daily.add_argument("--retrain", action="store_true", help="Force le reentrainement")
    daily.add_argument("--refresh", action="store_true", help="Ignore le cache de donnees")
    daily.add_argument("--verbose", action="store_true")
    daily.set_defaults(func=cmd_daily)

    apply_cmd = subparsers.add_parser("apply", help="Enregistre l'execution d'ordres")
    apply_cmd.add_argument("--file", required=True, help="Fichier JSON produit par `daily`")
    apply_cmd.add_argument("--cost-bps", type=float, default=None)
    apply_cmd.add_argument("--dry-run", action="store_true")
    apply_cmd.set_defaults(func=cmd_apply)

    portfolio = subparsers.add_parser("portfolio", help="Affiche le portefeuille detenu")
    portfolio.add_argument("--history", type=int, default=0, help="Affiche les N dernieres transactions")
    portfolio.add_argument("--offline", action="store_true", help="Sans valorisation (pas de donnees)")
    portfolio.set_defaults(func=cmd_portfolio)

    init_portfolio = subparsers.add_parser("init-portfolio", help="Cree le portefeuille")
    init_portfolio.add_argument("--cash", type=float, default=None)
    init_portfolio.add_argument("--force", action="store_true")
    init_portfolio.set_defaults(func=cmd_init_portfolio)

    set_position = subparsers.add_parser("set-position", help="Declare une position existante")
    set_position.add_argument("ticker")
    set_position.add_argument("--shares", type=float, required=True)
    set_position.add_argument("--price", type=float, required=True, help="Prix de revient unitaire")
    set_position.add_argument("--cash", type=float, default=None, help="Ajuste aussi les liquidites")
    set_position.set_defaults(func=cmd_set_position)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except ConfigError as exc:
        print(f"Erreur de configuration : {exc}", file=sys.stderr)
        return 2
    except DataError as exc:
        print(f"Erreur de donnees : {exc}", file=sys.stderr)
        return 3
    except KeyboardInterrupt:
        print("\nInterrompu.", file=sys.stderr)
        return 130


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
