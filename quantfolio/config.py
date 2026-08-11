"""Configuration de l'outil, chargee depuis un fichier YAML."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Erreur de configuration (cle inconnue, valeur invalide...)."""


@dataclass
class UniverseConfig:
    """Univers d'actions analyse."""

    tickers: list[str] = field(default_factory=list)
    # Ticker de reference pour le filtre de regime de marche (ex: SPY).
    # Si None, un indice equipondere de l'univers est utilise.
    benchmark: str | None = None


@dataclass
class DataConfig:
    """Source des donnees de prix."""

    # csv | yahoo | synthetic | custom
    provider: str = "csv"
    # Repertoire des CSV pour le provider "csv"
    path: str = "data/prices"
    # "module.chemin:MaClasse" pour brancher votre propre API
    custom: str | None = None
    # Arguments passes au constructeur du fournisseur (custom et synthetic)
    provider_args: dict[str, Any] = field(default_factory=dict)
    start: str = "2015-01-01"
    end: str | None = None
    cache_dir: str = "data/cache"
    use_cache: bool = True
    # Nombre minimum de seances disponibles pour qu'un titre soit eligible
    min_history: int = 260
    # Un titre est considere comme non negociable si sa derniere cotation
    # date de plus de N seances (delisting, suspension...)
    max_staleness: int = 5


@dataclass
class FeaturesConfig:
    momentum_windows: list[int] = field(default_factory=lambda: [21, 63, 126, 252])
    vol_windows: list[int] = field(default_factory=lambda: [21, 63])
    rsi_period: int = 14
    atr_period: int = 14
    bollinger_window: int = 20
    reversal_window: int = 5
    drawdown_window: int = 63
    # Nombre minimum de titres cotes un jour donne pour normaliser en coupe
    min_cross_section: int = 5


@dataclass
class ModelConfig:
    """Modele de prevision du rendement relatif."""

    # gbm | ridge | rules
    kind: str = "gbm"
    # Horizon de prevision, en seances
    horizon: int = 5
    # rank : on apprend le rang cross-sectionnel (robuste, defaut)
    # excess : on apprend le rendement en exces de la moyenne de l'univers
    label: str = "rank"
    # Poids du score a base de regles dans le melange final (0 = 100% ML)
    rules_weight: float = 0.35
    # Nombre de seances sur lesquelles le signal est moyenne avant d'etre
    # classe (1 = aucun lissage). Un score qui change tous les jours alors que
    # l'horizon de prevision est de plusieurs jours est surtout du bruit :
    # le lisser reduit fortement la rotation pour une perte de signal minime.
    score_smoothing: int = 5
    # Nombre minimum de lignes (titre x date) pour entrainer le ML
    min_train_rows: int = 1500
    # Re-entrainement tous les N jours de bourse en backtest
    retrain_every: int = 21
    # Jours mis en quarantaine entre train et predict, en plus de l'horizon
    embargo_days: int = 3
    # Fenetre d'apprentissage glissante (0 = tout l'historique)
    max_train_years: float = 8.0
    # Demi-vie de la ponderation des observations anciennes (0 = pas de decote)
    sample_half_life_years: float = 3.0
    # Hyperparametres passes a l'estimateur scikit-learn
    params: dict[str, Any] = field(default_factory=dict)
    random_state: int = 7


@dataclass
class PortfolioConfig:
    """Regles de construction du portefeuille."""

    capital: float = 100_000.0
    currency: str = "USD"
    max_positions: int = 10
    max_weight: float = 0.20
    min_weight: float = 0.02
    # On n'achete que les titres dont le score depasse ce seuil
    score_threshold: float = 0.0
    # Hysterese : un titre deja detenu n'est vendu que si son score passe sous
    # ce seuil, plus bas que le seuil d'achat. Evite d'entrer et sortir sans
    # cesse d'une ligne qui oscille autour du seuil.
    exit_threshold: float = -0.25
    # Bonus de rang accorde aux titres deja detenus lors de la selection :
    # on ne remplace pas une ligne pour un candidat a peine mieux classe.
    hold_bonus: float = 0.15
    # Volatilite annualisee ciblee pour le portefeuille
    vol_target: float = 0.15
    # Exposition brute maximale (1.0 = pas de levier)
    max_gross: float = 1.0
    # Reduction d'exposition quand le marche est sous sa moyenne mobile
    regime_filter: bool = True
    regime_ma: int = 200
    regime_risk_off: float = 0.40
    # On ne rebalance une ligne que si l'ecart de poids depasse cette bande
    no_trade_band: float = 0.02
    min_order_notional: float = 100.0
    fractional_shares: bool = False
    # Part du capital laissee en liquidites en permanence
    cash_buffer: float = 0.02
    # Fenetre pour estimer la volatilite / covariance du portefeuille
    vol_lookback: int = 63
    # Shrinkage de la matrice de covariance vers sa diagonale
    covariance_shrinkage: float = 0.30


@dataclass
class BacktestConfig:
    start: str | None = None
    end: str | None = None
    # Rebalancement tous les N jours de bourse (1 = quotidien)
    rebalance_days: int = 5
    cost_bps: float = 5.0
    slippage_bps: float = 5.0
    # next_open | next_close | close (close = execution au cours du signal)
    execution: str = "next_open"
    # Seances ignorees au demarrage, le temps que les indicateurs se forment
    warmup_days: int = 260


@dataclass
class PathsConfig:
    state_file: str = "state/portfolio.json"
    model_file: str = "models/model.joblib"
    output_dir: str = "output"


@dataclass
class Config:
    universe: UniverseConfig = field(default_factory=UniverseConfig)
    data: DataConfig = field(default_factory=DataConfig)
    features: FeaturesConfig = field(default_factory=FeaturesConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    portfolio: PortfolioConfig = field(default_factory=PortfolioConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    # Repertoire de base pour resoudre les chemins relatifs
    root: str = "."

    # -- helpers de chemins -------------------------------------------------
    def resolve(self, path: str) -> Path:
        p = Path(path)
        return p if p.is_absolute() else Path(self.root) / p

    @property
    def state_path(self) -> Path:
        return self.resolve(self.paths.state_file)

    @property
    def model_path(self) -> Path:
        return self.resolve(self.paths.model_file)

    @property
    def output_path(self) -> Path:
        return self.resolve(self.paths.output_dir)

    # -- chargement ---------------------------------------------------------
    @classmethod
    def from_dict(cls, raw: dict[str, Any], root: str = ".") -> "Config":
        raw = dict(raw or {})
        raw.pop("root", None)
        sections: dict[str, Any] = {}
        known = {f.name: f for f in fields(cls) if f.name != "root"}
        unknown = set(raw) - set(known)
        if unknown:
            raise ConfigError(
                f"Section(s) inconnue(s) dans la config : {sorted(unknown)}. "
                f"Sections valides : {sorted(known)}"
            )
        section_classes = {
            "universe": UniverseConfig,
            "data": DataConfig,
            "features": FeaturesConfig,
            "model": ModelConfig,
            "portfolio": PortfolioConfig,
            "backtest": BacktestConfig,
            "paths": PathsConfig,
        }
        for name in known:
            sections[name] = _build_section(section_classes[name], raw.get(name) or {})
        cfg = cls(root=root, **sections)
        cfg.validate()
        return cfg

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        path = Path(path)
        if not path.exists():
            raise ConfigError(f"Fichier de configuration introuvable : {path}")
        with path.open("r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
        if not isinstance(raw, dict):
            raise ConfigError(f"Le fichier {path} doit contenir un mapping YAML.")
        return cls.from_dict(raw, root=str(path.parent.parent))

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    # -- validation ---------------------------------------------------------
    def validate(self) -> None:
        p, m, b = self.portfolio, self.model, self.backtest
        if not self.universe.tickers:
            raise ConfigError("universe.tickers est vide : indiquez au moins un titre.")
        if len(set(self.universe.tickers)) != len(self.universe.tickers):
            raise ConfigError("universe.tickers contient des doublons.")
        if m.kind not in {"gbm", "ridge", "rules"}:
            raise ConfigError(f"model.kind invalide : {m.kind!r} (gbm | ridge | rules)")
        if m.label not in {"rank", "excess"}:
            raise ConfigError(f"model.label invalide : {m.label!r} (rank | excess)")
        if m.horizon < 1:
            raise ConfigError("model.horizon doit valoir au moins 1.")
        if not 0.0 <= m.rules_weight <= 1.0:
            raise ConfigError("model.rules_weight doit etre entre 0 et 1.")
        if m.score_smoothing < 1:
            raise ConfigError("model.score_smoothing doit valoir au moins 1.")
        if p.max_positions < 1:
            raise ConfigError("portfolio.max_positions doit valoir au moins 1.")
        if not 0.0 < p.max_weight <= 1.0:
            raise ConfigError("portfolio.max_weight doit etre dans ]0, 1].")
        if p.max_weight * p.max_positions < 1.0 and p.max_gross >= 1.0:
            raise ConfigError(
                "portfolio.max_weight x max_positions < 1 : impossible d'atteindre "
                "l'exposition cible. Augmentez max_weight ou max_positions."
            )
        if not 0.0 <= p.min_weight < p.max_weight:
            raise ConfigError("portfolio.min_weight doit etre dans [0, max_weight[.")
        if p.exit_threshold > p.score_threshold:
            raise ConfigError(
                "portfolio.exit_threshold doit etre inferieur ou egal a "
                "score_threshold (sinon un titre serait vendu des son achat)."
            )
        if p.hold_bonus < 0:
            raise ConfigError("portfolio.hold_bonus ne peut pas etre negatif.")
        if p.capital <= 0:
            raise ConfigError("portfolio.capital doit etre strictement positif.")
        if not 0.0 <= p.cash_buffer < 1.0:
            raise ConfigError("portfolio.cash_buffer doit etre dans [0, 1[.")
        if not 0.0 <= p.covariance_shrinkage <= 1.0:
            raise ConfigError("portfolio.covariance_shrinkage doit etre entre 0 et 1.")
        if p.vol_target <= 0:
            raise ConfigError("portfolio.vol_target doit etre strictement positif.")
        if b.rebalance_days < 1:
            raise ConfigError("backtest.rebalance_days doit valoir au moins 1.")
        if b.execution not in {"next_open", "next_close", "close"}:
            raise ConfigError(
                f"backtest.execution invalide : {b.execution!r} "
                "(next_open | next_close | close)"
            )
        if self.data.provider not in {"csv", "yahoo", "synthetic", "custom"}:
            raise ConfigError(
                f"data.provider invalide : {self.data.provider!r} "
                "(csv | yahoo | synthetic | custom)"
            )
        if self.data.provider == "custom" and not self.data.custom:
            raise ConfigError(
                "data.provider = custom mais data.custom n'est pas renseigne "
                '(format attendu : "mon_module:MaClasse").'
            )


def _build_section(section_cls: type, raw: Any) -> Any:
    if isinstance(raw, section_cls):
        return raw
    if not isinstance(raw, dict):
        raise ConfigError(
            f"La section {section_cls.__name__} doit etre un mapping, "
            f"recu : {type(raw).__name__}"
        )
    valid = {f.name for f in fields(section_cls)}
    unknown = set(raw) - valid
    if unknown:
        raise ConfigError(
            f"Cle(s) inconnue(s) dans {section_cls.__name__} : {sorted(unknown)}. "
            f"Cles valides : {sorted(valid)}"
        )
    return section_cls(**raw)


def load_config(path: str | Path | None) -> Config:
    """Charge la config depuis `path`, ou depuis config/config.yaml par defaut."""
    if path is not None:
        return Config.from_yaml(path)
    for candidate in (Path("config/config.yaml"), Path("config/config.example.yaml")):
        if candidate.exists():
            return Config.from_yaml(candidate)
    raise ConfigError(
        "Aucune configuration trouvee. Creez config/config.yaml "
        "(copiez config/config.example.yaml) ou passez --config CHEMIN."
    )
