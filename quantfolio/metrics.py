"""Mesures de performance et de risque."""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252

# Un ecart-type inferieur a ce seuil est du bruit d'arrondi, pas de la
# dispersion : diviser par lui produirait des ratios astronomiques.
MIN_STD = 1e-12


def to_returns(equity: pd.Series) -> pd.Series:
    return (equity.div(equity.shift(1)) - 1.0).dropna()


def total_return(equity: pd.Series) -> float:
    if len(equity) < 2 or equity.iloc[0] <= 0:
        return float("nan")
    return float(equity.iloc[-1] / equity.iloc[0] - 1.0)


def cagr(equity: pd.Series) -> float:
    if len(equity) < 2 or equity.iloc[0] <= 0:
        return float("nan")
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    if years <= 0:
        return float("nan")
    return float((equity.iloc[-1] / equity.iloc[0]) ** (1.0 / years) - 1.0)


def annual_volatility(returns: pd.Series) -> float:
    if len(returns) < 2:
        return float("nan")
    return float(returns.std(ddof=1) * np.sqrt(TRADING_DAYS))


def sharpe_ratio(returns: pd.Series, risk_free: float = 0.0) -> float:
    if len(returns) < 2:
        return float("nan")
    excess = returns - risk_free / TRADING_DAYS
    std = excess.std(ddof=1)
    if not np.isfinite(std) or std < MIN_STD:
        return float("nan")
    return float(excess.mean() / std * np.sqrt(TRADING_DAYS))


def sortino_ratio(returns: pd.Series, risk_free: float = 0.0) -> float:
    if len(returns) < 2:
        return float("nan")
    excess = returns - risk_free / TRADING_DAYS
    downside = excess.clip(upper=0.0)
    denominator = float(np.sqrt((downside**2).mean()))
    if not np.isfinite(denominator) or denominator < MIN_STD:
        return float("nan")
    return float(excess.mean() / denominator * np.sqrt(TRADING_DAYS))


def drawdown_series(equity: pd.Series) -> pd.Series:
    peak = equity.cummax()
    return equity.div(peak) - 1.0


def max_drawdown(equity: pd.Series) -> float:
    if len(equity) < 2:
        return float("nan")
    return float(drawdown_series(equity).min())


def calmar_ratio(equity: pd.Series) -> float:
    drawdown = max_drawdown(equity)
    if not np.isfinite(drawdown) or drawdown == 0:
        return float("nan")
    return float(cagr(equity) / abs(drawdown))


def hit_rate(returns: pd.Series) -> float:
    if returns.empty:
        return float("nan")
    return float((returns > 0).mean())


def monthly_returns(equity: pd.Series) -> pd.Series:
    if equity.empty:
        return pd.Series(dtype="float64")
    monthly = equity.resample("ME").last()
    return (monthly.div(monthly.shift(1)) - 1.0).dropna()


def beta_alpha(returns: pd.Series, benchmark_returns: pd.Series) -> tuple[float, float]:
    """Beta et alpha annualise par rapport a un indice de reference."""
    joined = pd.concat([returns, benchmark_returns], axis=1, join="inner").dropna()
    if len(joined) < 20:
        return float("nan"), float("nan")
    portfolio, market = joined.iloc[:, 0], joined.iloc[:, 1]
    variance = float(market.var(ddof=1))
    if not np.isfinite(variance) or variance < MIN_STD**2:
        return float("nan"), float("nan")
    beta = float(portfolio.cov(market) / variance)
    alpha = float((portfolio.mean() - beta * market.mean()) * TRADING_DAYS)
    return beta, alpha


def information_ratio(returns: pd.Series, benchmark_returns: pd.Series) -> float:
    joined = pd.concat([returns, benchmark_returns], axis=1, join="inner").dropna()
    if len(joined) < 20:
        return float("nan")
    active = joined.iloc[:, 0] - joined.iloc[:, 1]
    std = active.std(ddof=1)
    if not np.isfinite(std) or std < MIN_STD:
        return float("nan")
    return float(active.mean() / std * np.sqrt(TRADING_DAYS))


def summarize(
    equity: pd.Series,
    benchmark: pd.Series | None = None,
    turnover: pd.Series | None = None,
    exposure: pd.Series | None = None,
    risk_free: float = 0.0,
) -> dict[str, float]:
    """Tableau de bord complet d'une courbe de capital."""
    equity = equity.dropna()
    returns = to_returns(equity)
    summary: dict[str, float] = {
        "rendement_total": total_return(equity),
        "cagr": cagr(equity),
        "volatilite": annual_volatility(returns),
        "sharpe": sharpe_ratio(returns, risk_free),
        "sortino": sortino_ratio(returns, risk_free),
        "max_drawdown": max_drawdown(equity),
        "calmar": calmar_ratio(equity),
        "jours_gagnants": hit_rate(returns),
        "mois_gagnants": hit_rate(monthly_returns(equity)),
        "meilleur_jour": float(returns.max()) if len(returns) else float("nan"),
        "pire_jour": float(returns.min()) if len(returns) else float("nan"),
        "nb_seances": float(len(equity)),
    }
    if turnover is not None and len(turnover):
        # Rotation annualisee : somme des echanges rapportee a une annee.
        years = max((equity.index[-1] - equity.index[0]).days / 365.25, 1e-9)
        summary["rotation_annuelle"] = float(turnover.sum() / years)
    if exposure is not None and len(exposure):
        summary["exposition_moyenne"] = float(exposure.mean())
    if benchmark is not None and len(benchmark.dropna()) > 1:
        benchmark_returns = to_returns(benchmark.dropna())
        beta, alpha = beta_alpha(returns, benchmark_returns)
        summary.update(
            {
                "benchmark_cagr": cagr(benchmark.dropna()),
                "benchmark_vol": annual_volatility(benchmark_returns),
                "benchmark_sharpe": sharpe_ratio(benchmark_returns, risk_free),
                "benchmark_max_drawdown": max_drawdown(benchmark.dropna()),
                "beta": beta,
                "alpha": alpha,
                "information_ratio": information_ratio(returns, benchmark_returns),
            }
        )
    return summary


PERCENT_METRICS = {
    "rendement_total", "cagr", "volatilite", "max_drawdown", "jours_gagnants",
    "mois_gagnants", "meilleur_jour", "pire_jour", "exposition_moyenne",
    "benchmark_cagr", "benchmark_vol", "benchmark_max_drawdown", "alpha",
}


def format_summary(summary: dict[str, float]) -> str:
    """Rend le tableau de bord lisible dans un terminal."""
    if not summary:
        return "(aucune metrique)"
    width = max(len(k) for k in summary)
    lines = []
    for key, value in summary.items():
        if value is None or (isinstance(value, float) and not np.isfinite(value)):
            rendered = "n/a"
        elif key in PERCENT_METRICS:
            rendered = f"{value:>8.2%}"
        elif key == "nb_seances":
            rendered = f"{value:>8.0f}"
        else:
            rendered = f"{value:>8.2f}"
        lines.append(f"  {key.replace('_', ' '):<{width}}  {rendered}")
    return "\n".join(lines)
