"""quantfolio : scoring d'actions et gestion de portefeuille.

Pipeline complet :

    donnees -> indicateurs -> score cross-sectionnel -> portefeuille cible
            -> ordres d'achat/vente -> suivi du portefeuille

Exemple minimal :

    from quantfolio import Config, Engine

    cfg = Config.from_yaml("config/config.yaml")
    engine = Engine(cfg)
    engine.prepare()
    ranker = engine.fit()
    scores = engine.score_at(engine.last_date(), ranker)
    target = engine.target_at(engine.last_date(), ranker, scores=scores)
"""

from .backtest import BacktestResult, run_backtest
from .config import Config, ConfigError, load_config
from .data import DataProvider, MarketData, Panel, load_market_data
from .engine import Dataset, Engine
from .model import Ranker
from .orders import Order, OrderPlan, execute_plan, generate_orders
from .portfolio import PortfolioTarget, build_target
from .state import PortfolioState, Position

__version__ = "0.1.0"

__all__ = [
    "BacktestResult",
    "Config",
    "ConfigError",
    "Dataset",
    "DataProvider",
    "Engine",
    "MarketData",
    "Order",
    "OrderPlan",
    "Panel",
    "PortfolioState",
    "PortfolioTarget",
    "Position",
    "Ranker",
    "build_target",
    "execute_plan",
    "generate_orders",
    "load_config",
    "load_market_data",
    "run_backtest",
    "__version__",
]
