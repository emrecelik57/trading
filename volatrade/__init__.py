"""volatrade — selection d'actions volatiles, mesure du risque, plans d'achat/vente."""

from .config import AppConfig, load_config
from .engine import MarketView, analyse, run
from .metrics import RiskMetrics, compute_metrics
from .plan import TradePlan, build_plan
from .positions import Position, PositionReview, review_portfolio
from .risk import RiskSettings, allocate
from .signals import Signal, evaluate
from .universe import CANDIDATE_POOL, screen

__version__ = "1.0.0"

__all__ = [
    "AppConfig",
    "CANDIDATE_POOL",
    "MarketView",
    "Position",
    "PositionReview",
    "RiskMetrics",
    "RiskSettings",
    "Signal",
    "TradePlan",
    "allocate",
    "analyse",
    "build_plan",
    "compute_metrics",
    "evaluate",
    "load_config",
    "review_portfolio",
    "run",
    "screen",
    "__version__",
]
