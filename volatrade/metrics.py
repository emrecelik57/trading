"""Mesures de risque et de performance calculees sur des series de cours.

Toutes les fonctions acceptent des `pandas.Series` / `DataFrame` et renvoient
des scalaires ou des series. Les rendements utilises sont logarithmiques pour
la volatilite (additifs dans le temps) et arithmetiques pour la VaR et les
pertes exprimees en euros/dollars.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# --------------------------------------------------------------------------
# Rendements
# --------------------------------------------------------------------------
def log_returns(prices: pd.Series) -> pd.Series:
    """Rendements logarithmiques quotidiens."""
    return np.log(prices.astype(float)).diff().dropna()


def simple_returns(prices: pd.Series) -> pd.Series:
    """Rendements arithmetiques quotidiens."""
    return prices.astype(float).pct_change().dropna()


def total_return(prices: pd.Series, window: int | None = None) -> float:
    """Performance cumulee sur `window` seances (toute la serie si None)."""
    series = prices.dropna()
    if window is not None:
        series = series.iloc[-(window + 1):]
    if len(series) < 2:
        return float("nan")
    return float(series.iloc[-1] / series.iloc[0] - 1.0)


def cagr(prices: pd.Series, periods: int = TRADING_DAYS) -> float:
    """Rendement annualise compose."""
    series = prices.dropna()
    if len(series) < 2:
        return float("nan")
    years = (len(series) - 1) / periods
    if years <= 0:
        return float("nan")
    growth = series.iloc[-1] / series.iloc[0]
    if growth <= 0:
        return float("nan")
    return float(growth ** (1 / years) - 1.0)


# --------------------------------------------------------------------------
# Volatilite
# --------------------------------------------------------------------------
def annualized_volatility(
    returns: pd.Series, window: int | None = None, periods: int = TRADING_DAYS
) -> float:
    """Volatilite annualisee (ecart-type des rendements x racine(252))."""
    series = returns.dropna()
    if window is not None:
        series = series.iloc[-window:]
    if len(series) < 3:
        return float("nan")
    return float(series.std(ddof=1) * np.sqrt(periods))


def ewma_volatility(
    returns: pd.Series, lam: float = 0.94, periods: int = TRADING_DAYS
) -> float:
    """Volatilite annualisee ponderee exponentiellement (RiskMetrics).

    Reagit plus vite qu'un ecart-type glissant a un changement de regime, ce
    qui compte sur des titres tres volatils.
    """
    series = returns.dropna()
    if len(series) < 3:
        return float("nan")
    weights = lam ** np.arange(len(series) - 1, -1, -1)
    weights = weights / weights.sum()
    variance = float(np.sum(weights * (series.to_numpy() - series.mean()) ** 2))
    return float(np.sqrt(variance * periods))


def downside_deviation(
    returns: pd.Series, mar: float = 0.0, periods: int = TRADING_DAYS
) -> float:
    """Ecart-type des seuls rendements sous le seuil `mar` (annualise)."""
    series = returns.dropna()
    if len(series) < 3:
        return float("nan")
    shortfall = np.minimum(series - mar / periods, 0.0)
    return float(np.sqrt((shortfall ** 2).mean() * periods))


def volatility_regime(returns: pd.Series, short: int = 21, long: int = 126) -> float:
    """Ratio volatilite courte / volatilite longue.

    > 1.3 signale une acceleration du risque (reduire la taille de position),
    < 0.8 un calme relatif.
    """
    fast = annualized_volatility(returns, window=short)
    slow = annualized_volatility(returns, window=long)
    if not np.isfinite(fast) or not np.isfinite(slow) or slow == 0:
        return float("nan")
    return float(fast / slow)


# --------------------------------------------------------------------------
# Pertes extremes
# --------------------------------------------------------------------------
def drawdown_series(prices: pd.Series) -> pd.Series:
    """Serie des pertes depuis le plus haut glissant (valeurs <= 0)."""
    series = prices.dropna().astype(float)
    return series / series.cummax() - 1.0


def max_drawdown(prices: pd.Series) -> float:
    """Perte maximale historique, en fraction negative (-0.42 = -42 %)."""
    series = drawdown_series(prices)
    if series.empty:
        return float("nan")
    return float(series.min())


def current_drawdown(prices: pd.Series) -> float:
    """Ecart actuel au plus haut de la periode."""
    series = drawdown_series(prices)
    if series.empty:
        return float("nan")
    return float(series.iloc[-1])


def ulcer_index(prices: pd.Series) -> float:
    """Indice d'ulcere : moyenne quadratique des drawdowns.

    Penalise les baisses profondes ET longues, contrairement au max drawdown
    qui ne retient qu'un instant.
    """
    series = drawdown_series(prices)
    if series.empty:
        return float("nan")
    return float(np.sqrt((series ** 2).mean()))


def historical_var(returns: pd.Series, level: float = 0.95, horizon: int = 1) -> float:
    """Value at Risk historique, exprimee en perte positive.

    0.087 signifie : dans 5 % des cas les pires, la perte a depasse 8,7 %.
    L'echelle multi-jours suit la racine du temps (approximation usuelle).
    """
    series = returns.dropna()
    if len(series) < 20:
        return float("nan")
    quantile = float(np.quantile(series, 1.0 - level))
    return float(max(0.0, -quantile) * np.sqrt(horizon))


def conditional_var(returns: pd.Series, level: float = 0.95, horizon: int = 1) -> float:
    """CVaR / Expected Shortfall : perte moyenne au-dela de la VaR."""
    series = returns.dropna()
    if len(series) < 20:
        return float("nan")
    threshold = float(np.quantile(series, 1.0 - level))
    tail = series[series <= threshold]
    if tail.empty:
        return float("nan")
    return float(max(0.0, -tail.mean()) * np.sqrt(horizon))


# --------------------------------------------------------------------------
# Ratios rendement / risque
# --------------------------------------------------------------------------
def sharpe_ratio(
    returns: pd.Series, risk_free: float = 0.0, periods: int = TRADING_DAYS
) -> float:
    """Sharpe annualise, `risk_free` etant un taux annuel (0.03 = 3 %)."""
    series = returns.dropna()
    if len(series) < 20:
        return float("nan")
    excess = series - risk_free / periods
    std = excess.std(ddof=1)
    if std == 0:
        return float("nan")
    return float(excess.mean() / std * np.sqrt(periods))


def sortino_ratio(
    returns: pd.Series, risk_free: float = 0.0, periods: int = TRADING_DAYS
) -> float:
    """Sortino annualise : ne penalise que la volatilite baissiere."""
    series = returns.dropna()
    if len(series) < 20:
        return float("nan")
    downside = downside_deviation(series, mar=risk_free, periods=periods)
    excess = (series.mean() - risk_free / periods) * periods
    if not np.isfinite(downside):
        return float("nan")
    if downside == 0:
        # Aucune seance sous le seuil : le ratio n'est pas indefini, il est infini.
        return float(np.sign(excess) * np.inf) if excess != 0 else 0.0
    return float(excess / downside)


def calmar_ratio(prices: pd.Series, periods: int = TRADING_DAYS) -> float:
    """Rendement annualise rapporte a la perte maximale."""
    drawdown = max_drawdown(prices)
    growth = cagr(prices, periods)
    if not np.isfinite(drawdown) or not np.isfinite(growth):
        return float("nan")
    if drawdown == 0:
        # Serie sans aucun repli : ratio infini plutot qu'indefini.
        return float(np.sign(growth) * np.inf) if growth != 0 else 0.0
    return float(growth / abs(drawdown))


# --------------------------------------------------------------------------
# Relation au marche
# --------------------------------------------------------------------------
def beta(returns: pd.Series, benchmark: pd.Series) -> float:
    """Sensibilite au marche : 1.8 = amplifie de 80 % les mouvements de l'indice."""
    aligned = pd.concat([returns, benchmark], axis=1, join="inner").dropna()
    if len(aligned) < 20:
        return float("nan")
    asset, index = aligned.iloc[:, 0], aligned.iloc[:, 1]
    variance = index.var(ddof=1)
    if variance == 0:
        return float("nan")
    return float(asset.cov(index) / variance)


def correlation(returns: pd.Series, benchmark: pd.Series) -> float:
    """Correlation des rendements avec une reference."""
    aligned = pd.concat([returns, benchmark], axis=1, join="inner").dropna()
    if len(aligned) < 20:
        return float("nan")
    return float(aligned.iloc[:, 0].corr(aligned.iloc[:, 1]))


# --------------------------------------------------------------------------
# Indicateurs techniques
# --------------------------------------------------------------------------
def true_range(frame: pd.DataFrame) -> pd.Series:
    """True Range de Wilder : max(h-b, |h-c_prec|, |b-c_prec|)."""
    high, low, close = frame["high"], frame["low"], frame["close"]
    previous_close = close.shift(1)
    return pd.concat(
        [high - low, (high - previous_close).abs(), (low - previous_close).abs()],
        axis=1,
    ).max(axis=1)


def atr(frame: pd.DataFrame, window: int = 14) -> pd.Series:
    """Average True Range (lissage de Wilder)."""
    return true_range(frame).ewm(alpha=1 / window, adjust=False, min_periods=window).mean()


def atr_percent(frame: pd.DataFrame, window: int = 14) -> float:
    """ATR rapporte au dernier cours : amplitude quotidienne typique en %."""
    series = atr(frame, window).dropna()
    if series.empty:
        return float("nan")
    last_close = float(frame["close"].iloc[-1])
    if last_close == 0:
        return float("nan")
    return float(series.iloc[-1] / last_close)


def rsi(prices: pd.Series, window: int = 14) -> pd.Series:
    """RSI de Wilder sur `window` seances."""
    delta = prices.astype(float).diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    result = 100.0 - 100.0 / (1.0 + rs)
    return result.where(avg_loss != 0.0, 100.0)


def sma(prices: pd.Series, window: int) -> pd.Series:
    """Moyenne mobile simple."""
    return prices.astype(float).rolling(window, min_periods=window).mean()


def average_dollar_volume(frame: pd.DataFrame, window: int = 21) -> float:
    """Volume median echange en devise sur `window` seances (filtre de liquidite)."""
    if "volume" not in frame or len(frame) < 5:
        return float("nan")
    turnover = (frame["close"] * frame["volume"]).dropna().iloc[-window:]
    if turnover.empty:
        return float("nan")
    return float(turnover.median())


# --------------------------------------------------------------------------
# Agregation
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class RiskMetrics:
    """Tableau de bord de risque d'un titre."""

    ticker: str
    last_price: float
    vol_ann_1m: float
    vol_ann_3m: float
    vol_ann_1y: float
    vol_ewma: float
    vol_regime: float
    atr_pct: float
    beta: float
    correlation_market: float
    max_drawdown: float
    current_drawdown: float
    ulcer_index: float
    var_95_1d: float
    cvar_95_1d: float
    var_95_10d: float
    downside_deviation: float
    sharpe: float
    sortino: float
    calmar: float
    cagr: float
    return_1m: float
    return_3m: float
    return_6m: float
    return_12m: float
    skew: float
    kurtosis: float
    dollar_volume: float
    observations: int

    def as_dict(self) -> dict:
        return asdict(self)

    @property
    def expected_daily_move(self) -> float:
        """Amplitude journaliere attendue (1 ecart-type), en fraction du cours."""
        if np.isfinite(self.vol_ewma):
            return float(self.vol_ewma / np.sqrt(TRADING_DAYS))
        return float(self.vol_ann_3m / np.sqrt(TRADING_DAYS))


def compute_metrics(
    ticker: str,
    frame: pd.DataFrame,
    benchmark_returns: pd.Series | None = None,
    risk_free: float = 0.0,
) -> RiskMetrics:
    """Calcule l'ensemble des mesures de risque pour un titre."""
    prices = frame["close"].dropna()
    returns = log_returns(prices)
    arithmetic = simple_returns(prices)
    empty = pd.Series(dtype=float)
    bench = benchmark_returns if benchmark_returns is not None else empty

    return RiskMetrics(
        ticker=ticker.upper(),
        last_price=float(prices.iloc[-1]) if len(prices) else float("nan"),
        vol_ann_1m=annualized_volatility(returns, window=21),
        vol_ann_3m=annualized_volatility(returns, window=63),
        vol_ann_1y=annualized_volatility(returns, window=252),
        vol_ewma=ewma_volatility(returns.iloc[-252:]),
        vol_regime=volatility_regime(returns),
        atr_pct=atr_percent(frame),
        beta=beta(returns.iloc[-252:], bench),
        correlation_market=correlation(returns.iloc[-252:], bench),
        max_drawdown=max_drawdown(prices.iloc[-252:]),
        current_drawdown=current_drawdown(prices.iloc[-252:]),
        ulcer_index=ulcer_index(prices.iloc[-252:]),
        var_95_1d=historical_var(arithmetic.iloc[-252:], 0.95),
        cvar_95_1d=conditional_var(arithmetic.iloc[-252:], 0.95),
        var_95_10d=historical_var(arithmetic.iloc[-252:], 0.95, horizon=10),
        downside_deviation=downside_deviation(returns.iloc[-252:]),
        sharpe=sharpe_ratio(returns.iloc[-252:], risk_free),
        sortino=sortino_ratio(returns.iloc[-252:], risk_free),
        calmar=calmar_ratio(prices.iloc[-252:]),
        cagr=cagr(prices),
        return_1m=total_return(prices, 21),
        return_3m=total_return(prices, 63),
        return_6m=total_return(prices, 126),
        return_12m=total_return(prices, 252),
        skew=float(arithmetic.iloc[-252:].skew()) if len(arithmetic) > 20 else float("nan"),
        kurtosis=float(arithmetic.iloc[-252:].kurt()) if len(arithmetic) > 20 else float("nan"),
        dollar_volume=average_dollar_volume(frame),
        observations=int(len(prices)),
    )
