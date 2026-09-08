"""Configuration de l'outil : valeurs par defaut, fichier JSON, options CLI."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

from .risk import RiskSettings
from .universe import DEFAULT_BENCHMARK, default_pool


class ConfigError(ValueError):
    """Configuration invalide."""


@dataclass
class AppConfig:
    """Parametres complets d'une execution."""

    # Capital et risque
    capital: float = 10_000.0
    devise: str = "$"
    risque_par_trade: float = 0.01
    multiple_atr_stop: float = 2.5
    poids_max: float = 0.15
    volatilite_cible: float = 0.45
    chaleur_max: float = 0.06
    exposition_max: float = 1.00
    max_positions_par_theme: int = 2

    # Selection
    nombre_titres: int = 10
    volume_min: float = 30e6
    prix_min: float = 3.0
    volatilite_max: float = 2.5
    univers: list[str] = field(default_factory=default_pool)
    indice_reference: str = DEFAULT_BENCHMARK

    # Donnees
    historique: str = "2y"
    taux_sans_risque: float = 0.03
    cache_ttl_heures: float = 6.0
    dossier_cache: str = ".volatrade_cache"

    def risk_settings(self) -> RiskSettings:
        """Traduit la configuration en parametres de dimensionnement."""
        return RiskSettings(
            capital=self.capital,
            risk_per_trade=self.risque_par_trade,
            atr_stop_multiple=self.multiple_atr_stop,
            max_weight=self.poids_max,
            target_volatility=self.volatilite_cible,
            max_portfolio_heat=self.chaleur_max,
            max_exposure=self.exposition_max,
            max_positions_per_theme=self.max_positions_par_theme,
        )

    def validate(self) -> "AppConfig":
        """Verifie les bornes des parametres sensibles."""
        if self.capital <= 0:
            raise ConfigError("le capital doit etre strictement positif")
        if not 0 < self.risque_par_trade <= 0.10:
            raise ConfigError("risque_par_trade doit etre dans ]0, 0.10] (1 % = 0.01)")
        if not 0 < self.poids_max <= 1:
            raise ConfigError("poids_max doit etre dans ]0, 1]")
        if self.nombre_titres < 1:
            raise ConfigError("nombre_titres doit valoir au moins 1")
        if self.multiple_atr_stop <= 0:
            raise ConfigError("multiple_atr_stop doit etre positif")
        if not self.univers:
            raise ConfigError("l'univers de depart est vide")
        return self

    def as_dict(self) -> dict:
        return asdict(self)


def load_config(path: Path | str | None = None, **overrides) -> AppConfig:
    """Charge la configuration : defauts, puis fichier JSON, puis options CLI."""
    data: dict = {}
    if path is not None:
        file_path = Path(path)
        if not file_path.exists():
            raise ConfigError(f"fichier de configuration introuvable : {file_path}")
        try:
            data = json.loads(file_path.read_text())
        except json.JSONDecodeError as exc:
            raise ConfigError(f"JSON invalide dans {file_path} : {exc}") from exc
        if not isinstance(data, dict):
            raise ConfigError("le fichier de configuration doit contenir un objet JSON")

    known = {item.name for item in fields(AppConfig)}
    unknown = set(data) - known
    if unknown:
        raise ConfigError(f"cles inconnues dans la configuration : {', '.join(sorted(unknown))}")

    merged = {**data, **{key: value for key, value in overrides.items() if value is not None}}
    if "univers" in merged and isinstance(merged["univers"], str):
        merged["univers"] = [item.strip().upper() for item in merged["univers"].split(",") if item.strip()]
    return AppConfig(**merged).validate()
