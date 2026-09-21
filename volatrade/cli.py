"""Interface en ligne de commande de volatrade."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import report
from .backtest import run_backtest
from .config import AppConfig, ConfigError, load_config
from .data import DataError, fetch_history, fetch_many
from .engine import analyse, load_market, run
from .metrics import compute_metrics, log_returns
from .plan import build_plan
from .positions import PortfolioError, load_portfolio, review_portfolio
from .risk import size_position
from .signals import evaluate
from .universe import theme_of

EPILOGUE = """exemples :
  volatrade selection                          classe le vivier par volatilite
  volatrade risque                             mesures de risque des 10 titres retenus
  volatrade plan --capital 25000               plans d'achat chiffres + regles de vente
  volatrade plan --json rapport.json           exporte tout le rapport
  volatrade suivi -p portefeuille.json         que faire des positions deja ouvertes
  volatrade backtest                           verifie les regles sur l'historique
  volatrade analyse TSLA NVDA                  fiche detaillee de titres precis
"""


#: Valeurs par defaut des options globales (elles acceptent la position avant
#: OU apres la sous-commande, d'ou le recours a argparse.SUPPRESS).
GLOBAL_DEFAULTS = {
    "config": None,
    "capital": None,
    "risque_par_trade": None,
    "nombre_titres": None,
    "univers": None,
    "historique": None,
    "devise": None,
    "sans_cache": False,
    "json_path": None,
}


def _global_options() -> argparse.ArgumentParser:
    """Options communes, utilisables avant ou apres la sous-commande."""
    parser = argparse.ArgumentParser(add_help=False)
    add = parser.add_argument
    add("--config", type=Path, default=argparse.SUPPRESS,
        help="fichier de configuration JSON")
    add("--capital", type=float, default=argparse.SUPPRESS,
        help="capital total du portefeuille")
    add("--risque", type=float, dest="risque_par_trade", default=argparse.SUPPRESS,
        help="risque par trade en fraction du capital (0.01 = 1 %%)")
    add("--top", type=int, dest="nombre_titres", default=argparse.SUPPRESS,
        help="nombre de titres retenus")
    add("--univers", type=str, default=argparse.SUPPRESS,
        help="liste de tickers separes par des virgules (remplace le vivier)")
    add("--historique", type=str, default=argparse.SUPPRESS,
        help="profondeur d'historique (1y, 2y, 5y...)")
    add("--devise", type=str, default=argparse.SUPPRESS, help="symbole monetaire affiche")
    add("--sans-cache", action="store_true", dest="sans_cache", default=argparse.SUPPRESS,
        help="force le retelechargement des cours")
    add("--json", type=Path, dest="json_path", default=argparse.SUPPRESS,
        help="exporte le resultat en JSON")
    return parser


def build_parser() -> argparse.ArgumentParser:
    common = _global_options()
    parser = argparse.ArgumentParser(
        prog="volatrade",
        parents=[common],
        description=(
            "Selectionne les actions les plus volatiles d'un vivier liquide, mesure leur "
            "risque et produit des plans d'achat avec les regles de vente associees."
        ),
        epilog=EPILOGUE,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="commande")

    selection = sub.add_parser("selection", parents=[common],
                               help="classe le vivier par volatilite et affiche les retenus")
    selection.add_argument("--tout", action="store_true", help="affiche aussi les titres ecartes")

    sub.add_parser("risque", parents=[common],
                   help="tableau des mesures de risque des titres retenus")

    plan = sub.add_parser("plan", parents=[common],
                          help="plans d'achat et regles de vente (commande par defaut)")
    plan.add_argument("--tout", action="store_true",
                      help="affiche la fiche detaillee de tous les titres, pas seulement des achats")

    suivi = sub.add_parser("suivi", parents=[common],
                           help="verdict du jour sur les positions ouvertes")
    suivi.add_argument("-p", "--portefeuille", type=Path, default=Path("portefeuille.json"),
                       help="fichier JSON des positions (defaut : portefeuille.json)")

    backtest = sub.add_parser("backtest", parents=[common], help="rejoue les regles sur l'historique")
    backtest.add_argument("--cadence", type=int, default=5,
                          help="nombre de seances entre deux evaluations d'entree (defaut : 5)")

    analyse_cmd = sub.add_parser("analyse", parents=[common], help="fiche detaillee de titres precis")
    analyse_cmd.add_argument("tickers", nargs="+", help="tickers a analyser")
    return parser


def _config_from_args(args: argparse.Namespace) -> AppConfig:
    overrides = {
        "capital": args.capital,
        "risque_par_trade": args.risque_par_trade,
        "nombre_titres": args.nombre_titres,
        "univers": args.univers,
        "historique": args.historique,
        "devise": args.devise,
    }
    if args.sans_cache:
        overrides["cache_ttl_heures"] = 0.0
    return load_config(args.config, **overrides)


def _export(path: Path | None, payload: dict) -> None:
    if path is None:
        return
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str))
    print(f"\n  Rapport exporte : {path}")


def _print_errors(errors: list) -> None:
    if errors:
        print("\n  Titres ignores (donnees indisponibles) :")
        for ticker, message in errors:
            print(f"    - {ticker} : {message}")


def cmd_selection(args: argparse.Namespace, config: AppConfig) -> int:
    view = run(config)
    print(report.title(f"Selection : {len(view.selection)} actions les plus volatiles du vivier"))
    print(report.format_screen(view.screen, show_rejected=getattr(args, "tout", False)))
    _print_errors(view.errors)
    _export(args.json_path, view.as_dict())
    return 0


def cmd_risque(args: argparse.Namespace, config: AppConfig) -> int:
    view = run(config)
    print(report.title("Mesures de risque des titres retenus"))
    print(report.format_risk_table([item.metrics for item in view.selection]))
    print("\n  Lecture :")
    print(report.explain_metrics())
    if not view.correlations.empty:
        print(report.title("Correlations des rendements quotidiens"))
        print(view.correlations.round(2).to_string())
        average = (view.correlations.sum().sum() - len(view.correlations)) / max(
            len(view.correlations) ** 2 - len(view.correlations), 1
        )
        print(f"\n  Correlation moyenne du panier : {average:.2f} "
              "(au-dela de 0,60 la diversification est illusoire)")
    _print_errors(view.errors)
    _export(args.json_path, view.as_dict())
    return 0


def cmd_plan(args: argparse.Namespace, config: AppConfig) -> int:
    view = run(config)
    currency = config.devise
    print(report.title(f"Univers retenu ({len(view.selection)} titres les plus volatils)"))
    print(report.format_screen(view.screen))

    print(report.title("Risque mesure"))
    print(report.format_risk_table([item.metrics for item in view.selection]))

    print(report.title("Plans d'achat"))
    print(report.format_plan_table(view.plans, currency))
    print()
    print(report.format_actionable_count(view.plans))

    print(report.title("Fiches detaillees"))
    shown = view.plans if getattr(args, "tout", False) else view.actionable
    if not shown:
        print("  Aucun titre ne remplit les conditions d'achat. "
              "Relancer --tout pour voir le detail des titres ecartes.")
    for plan in shown:
        print(report.format_plan_detail(plan, currency))

    print(report.title("Bilan du panier"))
    print(report.format_basket_summary(view.basket, view.risk, currency))
    print(f"\n  Capital de reference : {report.money(config.capital, currency, 0)}, "
          f"risque par trade {config.risque_par_trade:.1%}, "
          f"stop a {config.multiple_atr_stop:g} ATR.")
    _print_errors(view.errors)
    print(f"\n  {report.DISCLAIMER}")
    _export(args.json_path, view.as_dict())
    return 0


def cmd_suivi(args: argparse.Namespace, config: AppConfig) -> int:
    try:
        positions = load_portfolio(args.portefeuille)
    except PortfolioError as exc:
        print(f"erreur : {exc}", file=sys.stderr)
        print(
            "\n  Format attendu :\n"
            '  [{"ticker": "TSLA", "shares": 12, "entry_price": 250.0,\n'
            '    "entry_date": "2026-06-02", "stop_price": 221.5, "trimmed": []}]',
            file=sys.stderr,
        )
        return 2
    if not positions:
        print("  Portefeuille vide.")
        return 0

    errors: list[tuple[str, str]] = []
    quotes = fetch_many(
        [position.ticker for position in positions],
        config.historique,
        cache_dir=config.dossier_cache,
        cache_ttl=config.cache_ttl_heures * 3600,
        on_error=lambda ticker, exc: errors.append((ticker, str(exc))),
    )
    reviews = review_portfolio(positions, quotes, config.risk_settings())

    print(report.title(f"Suivi de {len(reviews)} position(s) ouverte(s)"))
    print(report.format_reviews(reviews, config.devise))
    total = sum(review.pnl for review in reviews)
    print(f"\n  Gain/perte latent total : {report.money(total, config.devise, 0)}")
    _print_errors(errors)
    print(f"\n  {report.DISCLAIMER}")
    _export(args.json_path, {"positions": [review.as_dict() for review in reviews]})
    return 0


def cmd_backtest(args: argparse.Namespace, config: AppConfig) -> int:
    quotes, benchmark, errors = load_market(config)
    view = analyse(config, quotes, benchmark, errors)
    selected = {item.ticker: quotes[item.ticker] for item in view.selection}
    print(report.title(f"Backtest des regles sur {len(selected)} titres ({config.historique})"))
    print("  Entree evaluee toutes les "
          f"{args.cadence} seances, execution a l'ouverture suivante, hors frais.\n")
    global_report, per_ticker = run_backtest(
        selected, benchmark, config.risk_settings(), args.cadence
    )
    print(report.format_backtest(global_report, per_ticker))
    print(f"\n  {report.DISCLAIMER}")
    _export(args.json_path, {"global": global_report.as_dict(),
                             "par_titre": {k: v.as_dict() for k, v in per_ticker.items()}})
    return 0


def cmd_analyse(args: argparse.Namespace, config: AppConfig) -> int:
    tickers = [ticker.upper() for ticker in args.tickers]
    errors: list[tuple[str, str]] = []
    quotes = fetch_many(
        tickers,
        config.historique,
        cache_dir=config.dossier_cache,
        cache_ttl=config.cache_ttl_heures * 3600,
        on_error=lambda ticker, exc: errors.append((ticker, str(exc))),
    )
    if not quotes:
        _print_errors(errors)
        return 1

    try:
        benchmark = fetch_history(
            config.indice_reference, config.historique,
            cache_dir=config.dossier_cache, cache_ttl=config.cache_ttl_heures * 3600,
        )
    except DataError:
        benchmark = None
    bench_returns = log_returns(benchmark.close) if benchmark is not None else None

    settings = config.risk_settings()
    payload = []
    for ticker, quote in quotes.items():
        metrics = compute_metrics(ticker, quote.frame, bench_returns, config.taux_sans_risque)
        signal = evaluate(ticker, quote.frame, metrics)
        sizing = size_position(
            ticker, metrics.last_price, signal.state.atr14, metrics.vol_ewma,
            signal.conviction, settings,
        )
        plan = build_plan(signal, metrics, sizing, theme_of(ticker), settings)
        print(report.title(f"{ticker} ({quote.exchange}, {quote.currency})"))
        print(report.format_risk_table([metrics]))
        print("\n  Composantes du score :")
        for name, value in signal.components.items():
            print(f"    {name:<10} {value:5.0f}/100")
        print(report.format_plan_detail(plan, config.devise))
        payload.append({"metriques": metrics.as_dict(), "plan": plan.as_dict()})

    _print_errors(errors)
    print(f"\n  {report.DISCLAIMER}")
    _export(args.json_path, {"titres": payload})
    return 0


COMMANDS = {
    "selection": cmd_selection,
    "risque": cmd_risque,
    "plan": cmd_plan,
    "suivi": cmd_suivi,
    "backtest": cmd_backtest,
    "analyse": cmd_analyse,
}


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    for name, value in GLOBAL_DEFAULTS.items():
        if not hasattr(args, name):
            setattr(args, name, value)
    commande = args.commande or "plan"
    if not hasattr(args, "tout"):
        args.tout = False

    try:
        config = _config_from_args(args)
    except ConfigError as exc:
        print(f"erreur de configuration : {exc}", file=sys.stderr)
        return 2

    try:
        return COMMANDS[commande](args, config)
    except DataError as exc:
        print(f"erreur de donnees : {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\ninterrompu", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
