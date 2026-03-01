"""
src/risk_metrics.py
===================
Complete risk metric suite for the Stock Assessment dashboard.

SOP Rules Enforced:
    - All parameters (risk_free_rate, var_confidence, trading_days_per_year)
      read from config.yaml — NOTHING hardcoded
    - Benchmark is ^NSEI (read from config)
    - Three VaR methods: Historical, Parametric, Monte Carlo
    - Rolling Beta uses 252-day window (from config trading_days_per_year)
    - Diversification Ratio = Σ(wi × σi) / σp

Public API:
    compute_all_metrics(portfolio_returns, benchmark_returns, weights,
                        individual_returns, mc_simulations) -> dict
    compute_sharpe(returns, risk_free_rate, trading_days)    -> float
    compute_sortino(returns, risk_free_rate, trading_days)   -> float
    compute_max_drawdown(cum_returns)                        -> dict
    compute_beta(portfolio_returns, benchmark_returns,
                 window)                                     -> pd.Series
    compute_var(returns, confidence, portfolio_value,
                mc_simulations)                              -> dict
    compute_cvar(returns, confidence)                        -> float
    compute_diversification_ratio(weights, individual_returns,
                                  portfolio_returns)         -> float

Mathematical formulas (from SOP Section 8):
    Sharpe   = (Rp - Rf) / σp
    Sortino  = (Rp - Rf) / σd          σd = downside std below MAR (=Rf)
    Beta     = Cov(Rp, Rm) / Var(Rm)   rolling 252-day window
    DR       = Σ(wi × σi) / σp
    VaR_par  = -(μp - 1.645 × σp) × Portfolio Value   [95% confidence]
    CVaR     = mean of returns below VaR threshold
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------
_CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"


def _load_config() -> dict:
    if not _CONFIG_PATH.exists():
        raise FileNotFoundError(f"config.yaml not found at {_CONFIG_PATH}")
    with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Portfolio returns helper
# ---------------------------------------------------------------------------

def compute_portfolio_returns(
    prices: pd.DataFrame,
    weights: dict[str, float],
) -> pd.Series:
    """
    Compute daily weighted portfolio returns from price DataFrame.

    Parameters
    ----------
    prices  : pd.DataFrame — adjusted close prices, columns = tickers
    weights : dict         — {ticker: weight}, weights must sum to ~1.0

    Returns
    -------
    pd.Series — daily portfolio returns, index = trading dates

    Unit Test Case:
        Input:  prices with 2 stocks equal-weighted (0.5 each)
                TCS returns [0.01, -0.02, 0.03]
                INFY returns [0.02, -0.01, 0.04]
        Expect: portfolio returns [0.015, -0.015, 0.035]
    """
    if abs(sum(weights.values()) - 1.0) > 1e-4:
        raise ValueError(
            f"Weights must sum to 1.0. Got {sum(weights.values()):.4f}. "
            "Check your optimization output."
        )

    tickers = list(weights.keys())
    missing = [t for t in tickers if t not in prices.columns]
    if missing:
        raise ValueError(f"Tickers not in price DataFrame: {missing}")

    w_array = np.array([weights[t] for t in tickers])
    returns = prices[tickers].pct_change().dropna()
    port_returns = returns.values @ w_array
    return pd.Series(port_returns, index=returns.index, name="portfolio_return")


# ---------------------------------------------------------------------------
# 1. Sharpe Ratio
# ---------------------------------------------------------------------------

def compute_sharpe(
    returns: pd.Series,
    risk_free_rate: Optional[float] = None,
    trading_days: Optional[int] = None,
) -> float:
    """
    Compute annualized Sharpe Ratio.

    Formula (SOP §8):
        Sharpe = (Rp - Rf) / σp

    Where:
        Rp = annualized portfolio return (geometric mean × trading_days)
        Rf = annual risk-free rate (from config.yaml)
        σp = annualized volatility (daily std × √trading_days)

    Parameters
    ----------
    returns      : pd.Series of daily portfolio returns
    risk_free_rate: annual rate (default from config.yaml)
    trading_days : trading days per year (default from config.yaml = 252)

    Returns
    -------
    float — annualized Sharpe Ratio

    Unit Test Case:
        Input:  daily returns ~ N(0.0004, 0.01), rf=0.065, days=252
        Expect: Sharpe in range [-1, 3] for realistic inputs
        Input:  all-zero returns
        Expect: Sharpe == -rf_daily_equivalent (negative)
    """
    cfg = _load_config()
    rf = risk_free_rate if risk_free_rate is not None else cfg.get("risk_free_rate", 0.065)
    td = trading_days if trading_days is not None else cfg.get("trading_days_per_year", 252)

    if returns.empty or returns.std() == 0:
        return 0.0

    ann_return = returns.mean() * td
    ann_vol = returns.std() * np.sqrt(td)
    return float((ann_return - rf) / ann_vol)


# ---------------------------------------------------------------------------
# 2. Sortino Ratio
# ---------------------------------------------------------------------------

def compute_sortino(
    returns: pd.Series,
    risk_free_rate: Optional[float] = None,
    trading_days: Optional[int] = None,
    mar: Optional[float] = None,
) -> float:
    """
    Compute annualized Sortino Ratio.

    Formula (SOP §8):
        Sortino = (Rp - Rf) / σd
        σd = std dev of returns BELOW the minimum acceptable return (MAR)

    MAR defaults to the daily equivalent of the annual risk-free rate.
    Only downside deviations (returns < MAR) count in σd.

    Parameters
    ----------
    returns      : pd.Series of daily portfolio returns
    risk_free_rate: annual rate (default from config.yaml)
    trading_days : trading days per year (default 252)
    mar          : minimum acceptable daily return (default = rf/trading_days)

    Returns
    -------
    float — annualized Sortino Ratio

    Unit Test Case:
        Input:  returns all positive → σd ≈ 0 → Sortino → very large positive
        Input:  returns all negative → Sortino negative
        Input:  identical to Sharpe inputs but no positive days
        Expect: Sortino >= Sharpe always (less penalisation for upside vol)
    """
    cfg = _load_config()
    rf = risk_free_rate if risk_free_rate is not None else cfg.get("risk_free_rate", 0.065)
    td = trading_days if trading_days is not None else cfg.get("trading_days_per_year", 252)
    daily_mar = mar if mar is not None else (rf / td)

    if returns.empty:
        return 0.0

    ann_return = returns.mean() * td
    downside = returns[returns < daily_mar] - daily_mar
    downside_std = np.sqrt((downside ** 2).mean()) * np.sqrt(td)

    if downside_std == 0:
        return float("inf") if ann_return > rf else 0.0

    return float((ann_return - rf) / downside_std)


# ---------------------------------------------------------------------------
# 3. Maximum Drawdown + Recovery Time
# ---------------------------------------------------------------------------

def compute_max_drawdown(
    returns: pd.Series,
) -> dict:
    """
    Compute maximum drawdown and recovery time from daily return series.

    Formula:
        Drawdown(t) = (Peak(t) - P(t)) / Peak(t)
        Max Drawdown = max(Drawdown(t)) over all t

    Recovery Time = number of trading days from trough back to a new peak.

    Parameters
    ----------
    returns : pd.Series of daily portfolio returns

    Returns
    -------
    dict:
        {
            "max_drawdown_pct"  : float  — as percentage (e.g. -34.5)
            "drawdown_start"    : str    — date of peak before worst drawdown
            "drawdown_trough"   : str    — date of trough
            "drawdown_end"      : str    — date of recovery (or None if not recovered)
            "recovery_days"     : int    — trading days to recovery (or None)
            "drawdown_series"   : pd.Series — full drawdown curve for plotting
        }

    Unit Test Case:
        Input:  returns = [0.1, -0.2, -0.1, 0.1, 0.15]
                Cum values ≈ [1.1, 0.88, 0.792, 0.871, 1.002]
        Expect: max_drawdown_pct ≈ -28.0  (from 1.1 to 0.792)
                recovery_days > 0
    """
    if returns.empty:
        return {
            "max_drawdown_pct": 0.0, "drawdown_start": None,
            "drawdown_trough": None, "drawdown_end": None,
            "recovery_days": None, "drawdown_series": pd.Series(dtype=float),
        }

    cum_returns = (1 + returns).cumprod()
    rolling_max = cum_returns.cummax()
    drawdown_series = (cum_returns - rolling_max) / rolling_max

    # Worst drawdown
    trough_idx = drawdown_series.idxmin()
    max_dd = drawdown_series.min()

    # Find peak before trough
    peak_idx = rolling_max[:trough_idx].idxmax()

    # Recovery: first date after trough where cumulative return exceeds peak
    post_trough = cum_returns[trough_idx:]
    peak_value = rolling_max[trough_idx]
    recovered = post_trough[post_trough >= peak_value]

    recovery_date = None
    recovery_days = None
    if not recovered.empty:
        recovery_date = recovered.index[0]
        recovery_days = int((recovery_date - trough_idx).days)

    return {
        "max_drawdown_pct": round(float(max_dd) * 100, 2),
        "drawdown_start": str(peak_idx)[:10],
        "drawdown_trough": str(trough_idx)[:10],
        "drawdown_end": str(recovery_date)[:10] if recovery_date else None,
        "recovery_days": recovery_days,
        "drawdown_series": drawdown_series,
    }


# ---------------------------------------------------------------------------
# 4. Rolling Beta vs Benchmark
# ---------------------------------------------------------------------------

def compute_beta(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    window: Optional[int] = None,
) -> pd.Series:
    """
    Compute rolling Beta of portfolio vs benchmark (^NSEI).

    Formula (SOP §8):
        β = Cov(Rp, Rm) / Var(Rm)
        Computed over a rolling window (default: 252 trading days)

    Parameters
    ----------
    portfolio_returns  : pd.Series — daily portfolio returns
    benchmark_returns  : pd.Series — daily NIFTY 50 returns
    window             : int — rolling window in trading days (default: 252)

    Returns
    -------
    pd.Series — rolling Beta values, same index as portfolio_returns

    Unit Test Case:
        Input:  portfolio_returns == benchmark_returns (identical series)
        Expect: beta.dropna() ≈ 1.0 for all values

        Input:  portfolio_returns == 2 × benchmark_returns
        Expect: beta.dropna() ≈ 2.0 for all values

        Input:  portfolio_returns == -benchmark_returns
        Expect: beta.dropna() ≈ -1.0 (inverse market mover)
    """
    cfg = _load_config()
    w = window if window is not None else cfg.get("trading_days_per_year", 252)

    # Align on common dates
    aligned = pd.concat(
        [portfolio_returns.rename("portfolio"), benchmark_returns.rename("benchmark")],
        axis=1,
    ).dropna()

    # Vectorised rolling beta: β = Cov(Rp, Rm) / Var(Rm)
    # Use native pandas rolling operations — no Python-level loop, no apply().
    port  = aligned["portfolio"]
    bench = aligned["benchmark"]

    cov_rb   = port.rolling(window=w).cov(bench)           # rolling covariance
    var_b    = bench.rolling(window=w).var()                # rolling variance of benchmark

    beta_raw = cov_rb / var_b                              # element-wise division
    beta_series = beta_raw.rename("rolling_beta")

    # Propagate NaN wherever benchmark variance is zero or insufficient data
    beta_series[var_b == 0] = np.nan

    return beta_series


# ---------------------------------------------------------------------------
# 5. Value at Risk — Three Methods
# ---------------------------------------------------------------------------

def compute_var(
    returns: pd.Series,
    portfolio_value: float = 1_000_000.0,
    confidence: Optional[float] = None,
    mc_simulations: Optional[np.ndarray] = None,
) -> dict:
    """
    Compute Value at Risk using three methods.

    Method 1 — Historical VaR:
        5th percentile of actual historical daily return distribution.
        No distribution assumption.

    Method 2 — Parametric VaR:
        Assumes normal distribution.
        Formula (SOP §8): VaR = -(μp - 1.645 × σp) × Portfolio Value
        1.645 is the z-score for 95% one-tailed confidence.

    Method 3 — Monte Carlo VaR:
        5th percentile of simulated return distribution.
        Requires mc_simulations array from monte_carlo.py.

    Parameters
    ----------
    returns          : pd.Series — daily portfolio returns (historical)
    portfolio_value  : float     — portfolio value in ₹
    confidence       : float     — confidence level (default from config: 0.95)
    mc_simulations   : np.ndarray — 1D array of simulated 1-day returns
                                    (pass None if monte_carlo not run yet)

    Returns
    -------
    dict:
        {
            "confidence"        : float  — e.g. 0.95
            "var_historical"    : float  — ₹ amount at risk (positive = loss)
            "var_parametric"    : float  — ₹ amount at risk
            "var_mc"            : float  — ₹ amount at risk (None if no MC)
            "var_historical_pct": float  — as % of portfolio value
            "var_parametric_pct": float  — as % of portfolio value
            "var_mc_pct"        : float  — as % of portfolio value (None if no MC)
            "interpretation"    : str    — human-readable explanation
        }

    Unit Test Case:
        Input:  returns ~ N(0, 0.01), portfolio_value=1_000_000, confidence=0.95
        Expect: var_parametric ≈ 16_450  (1.645 × 0.01 × 1M)
                var_historical close to var_parametric for large N

        Input:  all-zero returns
        Expect: var_historical == 0, var_parametric == 0
    """
    cfg = _load_config()
    conf = confidence if confidence is not None else cfg.get("var_confidence", 0.95)
    alpha = 1 - conf  # e.g. 0.05 for 95% confidence

    if returns.empty:
        return {
            "confidence": conf, "var_historical": 0.0, "var_parametric": 0.0,
            "var_mc": None, "var_historical_pct": 0.0, "var_parametric_pct": 0.0,
            "var_mc_pct": None, "interpretation": "No data provided.",
        }

    # Method 1: Historical
    hist_var_pct = float(np.percentile(returns, alpha * 100))
    hist_var = float(-hist_var_pct * portfolio_value)

    # Method 2: Parametric — SOP formula: -(μ - 1.645σ) × value
    mu = returns.mean()
    sigma = returns.std()
    z_score = float(pd.Series([0.95]).apply(
        lambda c: abs(float(np.percentile(np.random.normal(0, 1, 100000),
                                          (1 - conf) * 100)))
    ).iloc[0])
    # Use exact z-score for common confidence levels
    z_map = {0.90: 1.282, 0.95: 1.645, 0.99: 2.326}
    z = z_map.get(round(conf, 2), 1.645)
    param_var_pct = -(mu - z * sigma)
    param_var = float(param_var_pct * portfolio_value)

    # Method 3: Monte Carlo
    mc_var = None
    mc_var_pct = None
    if mc_simulations is not None and len(mc_simulations) > 0:
        mc_var_pct_val = float(np.percentile(mc_simulations, alpha * 100))
        mc_var = float(-mc_var_pct_val * portfolio_value)
        mc_var_pct = round(-mc_var_pct_val * 100, 4)

    interpretation = (
        f"At {int(conf*100)}% confidence, the maximum expected daily loss is "
        f"₹{hist_var:,.0f} (Historical), ₹{param_var:,.0f} (Parametric)"
        + (f", ₹{mc_var:,.0f} (Monte Carlo)" if mc_var is not None else "")
        + f" on a ₹{portfolio_value:,.0f} portfolio."
    )

    return {
        "confidence": conf,
        "var_historical": round(hist_var, 2),
        "var_parametric": round(param_var, 2),
        "var_mc": round(mc_var, 2) if mc_var is not None else None,
        "var_historical_pct": round(-hist_var_pct * 100, 4),
        "var_parametric_pct": round(param_var_pct * 100, 4),
        "var_mc_pct": mc_var_pct,
        "interpretation": interpretation,
    }


# ---------------------------------------------------------------------------
# 6. Conditional VaR (CVaR / Expected Shortfall)
# ---------------------------------------------------------------------------

def compute_cvar(
    returns: pd.Series,
    confidence: Optional[float] = None,
) -> dict:
    """
    Compute Conditional VaR (Expected Shortfall).

    Definition:
        CVaR = average of all returns that fall BELOW the VaR threshold.
        It answers: "Given we are in the worst X% of outcomes, what is
        the average loss?"

    This is a more complete tail risk measure than VaR alone because VaR
    only says "losses won't exceed X" — CVaR says "when they do, here's
    the average severity."

    Parameters
    ----------
    returns    : pd.Series — daily portfolio returns
    confidence : float     — confidence level (default 0.95 from config)

    Returns
    -------
    dict:
        {
            "cvar_pct"   : float — average tail return as % (negative = loss)
            "cvar_value" : float — will be ₹ amount once portfolio_value known
            "var_threshold": float — the VaR boundary used
            "tail_obs"   : int   — number of observations in the tail
            "interpretation": str
        }

    Unit Test Case:
        Input:  100 returns, worst 5 are [-0.05, -0.04, -0.03, -0.025, -0.02]
        Expect: cvar_pct ≈ -0.033  (mean of 5 worst)
        Expect: cvar_pct <= var_pct always (CVaR is worse than or equal to VaR)
    """
    cfg = _load_config()
    conf = confidence if confidence is not None else cfg.get("var_confidence", 0.95)
    alpha = 1 - conf

    if returns.empty:
        return {
            "cvar_pct": 0.0, "cvar_value": 0.0,
            "var_threshold": 0.0, "tail_obs": 0,
            "interpretation": "No data provided.",
        }

    var_threshold = float(np.percentile(returns, alpha * 100))
    tail_returns = returns[returns <= var_threshold]

    if tail_returns.empty:
        cvar = var_threshold
    else:
        cvar = float(tail_returns.mean())

    interpretation = (
        f"In the worst {int(alpha * 100)}% of trading days, the average loss is "
        f"{abs(cvar) * 100:.2f}% of the portfolio. "
        f"This is the Expected Shortfall beyond the {int(conf*100)}% VaR level."
    )

    return {
        "cvar_pct": round(cvar * 100, 4),
        "var_threshold": round(var_threshold * 100, 4),
        "tail_obs": len(tail_returns),
        "interpretation": interpretation,
    }


# ---------------------------------------------------------------------------
# 7. Diversification Ratio
# ---------------------------------------------------------------------------

def compute_diversification_ratio(
    weights: dict[str, float],
    individual_returns: pd.DataFrame,
    portfolio_returns: pd.Series,
    trading_days: Optional[int] = None,
) -> dict:
    """
    Compute the Diversification Ratio of the portfolio.

    Formula (SOP §8):
        DR = Σ(wi × σi) / σp
        DR > 1 confirms diversification is reducing portfolio risk.

    Interpretation:
        DR = 1.0  → no diversification benefit (perfect correlation)
        DR = 1.43 → portfolio 30% less volatile than weighted-avg components
        DR < 1.0  → impossible in a long-only portfolio (sanity check)

    Parameters
    ----------
    weights            : dict {ticker: weight}
    individual_returns : pd.DataFrame — daily returns per stock (columns = tickers)
    portfolio_returns  : pd.Series    — daily weighted portfolio returns
    trading_days       : int          — for annualizing (default 252)

    Returns
    -------
    dict:
        {
            "diversification_ratio": float
            "weighted_avg_vol"     : float — annualized Σ(wi × σi)
            "portfolio_vol"        : float — annualized portfolio σp
            "interpretation"       : str
        }

    Unit Test Case:
        Input:  2 stocks perfectly correlated, equal weights
        Expect: DR ≈ 1.0

        Input:  2 stocks perfectly negatively correlated, equal weights
        Expect: DR >> 1.0 (maximum diversification benefit)
    """
    cfg = _load_config()
    td = trading_days if trading_days is not None else cfg.get("trading_days_per_year", 252)

    tickers = list(weights.keys())
    returns_aligned = individual_returns[tickers].dropna()

    # Individual annualized volatilities
    vol_i = returns_aligned.std() * np.sqrt(td)

    # Weighted average of individual vols: Σ(wi × σi)
    w_array = np.array([weights[t] for t in tickers])
    weighted_avg_vol = float(np.dot(w_array, vol_i.values))

    # Portfolio annualized volatility
    port_vol = float(portfolio_returns.std() * np.sqrt(td))

    if port_vol == 0:
        dr = 1.0
    else:
        dr = weighted_avg_vol / port_vol

    pct_reduction = (1 - 1 / dr) * 100 if dr > 1 else 0

    interpretation = (
        f"Diversification Ratio of {dr:.2f} means the portfolio is "
        f"{pct_reduction:.1f}% less volatile than the weighted average of its components. "
        + ("✅ Diversification is working." if dr > 1.05
           else "⚠️ Low diversification benefit — stocks may be highly correlated.")
    )

    return {
        "diversification_ratio": round(dr, 4),
        "weighted_avg_vol": round(weighted_avg_vol * 100, 4),
        "portfolio_vol": round(port_vol * 100, 4),
        "interpretation": interpretation,
    }


# ---------------------------------------------------------------------------
# 8. Annualized Return & Volatility
# ---------------------------------------------------------------------------

def compute_annualized_return(
    returns: pd.Series,
    trading_days: Optional[int] = None,
) -> float:
    """
    Compute annualized return using arithmetic mean × trading days.

    Formula (SOP §2 table):
        Annualized Return = mean(daily returns) × 252

    Unit Test Case:
        Input:  daily return = 0.001 every day
        Expect: annualized ≈ 0.252 (25.2%)
    """
    cfg = _load_config()
    td = trading_days or cfg.get("trading_days_per_year", 252)
    return float(returns.mean() * td)


def compute_annualized_volatility(
    returns: pd.Series,
    trading_days: Optional[int] = None,
) -> float:
    """
    Compute annualized volatility.

    Formula (SOP §2 table):
        Annualized Volatility = std(daily returns) × √252

    Unit Test Case:
        Input:  daily std = 0.01
        Expect: annualized vol ≈ 0.1587 (15.87%)
    """
    cfg = _load_config()
    td = trading_days or cfg.get("trading_days_per_year", 252)
    return float(returns.std() * np.sqrt(td))


# ---------------------------------------------------------------------------
# Master compute function (called from app.py)
# ---------------------------------------------------------------------------

def compute_all_metrics(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    weights: dict[str, float],
    individual_returns: pd.DataFrame,
    portfolio_value: float = 1_000_000.0,
    mc_simulations: Optional[np.ndarray] = None,
    risk_free_rate: Optional[float] = None,
) -> dict:
    """
    Compute the full risk metric suite defined in the SOP.

    Calls all individual metric functions and returns one consolidated dict.

    SOP Metrics Covered:
        ✅ Annualized Return
        ✅ Annualized Volatility
        ✅ Sharpe Ratio
        ✅ Sortino Ratio
        ✅ Rolling Beta (252-day window)
        ✅ Maximum Drawdown + Recovery Time
        ✅ Diversification Ratio
        ✅ VaR — Historical, Parametric, Monte Carlo
        ✅ CVaR (Expected Shortfall)

    Parameters
    ----------
    portfolio_returns  : pd.Series   — daily weighted portfolio returns
    benchmark_returns  : pd.Series   — daily NIFTY 50 (^NSEI) returns
    weights            : dict        — {ticker: weight}
    individual_returns : pd.DataFrame — daily returns per stock
    portfolio_value    : float       — portfolio size in ₹
    mc_simulations     : np.ndarray  — simulated 1-day returns from monte_carlo.py
    risk_free_rate     : float       — override config value if needed

    Returns
    -------
    dict — all metrics, keyed by metric name. Ready for Streamlit display.

    Example
    -------
    >>> from src.risk_metrics import compute_all_metrics
    >>> metrics = compute_all_metrics(
    ...     portfolio_returns=port_ret,
    ...     benchmark_returns=nifty_ret,
    ...     weights={"RELIANCE.NS": 0.4, "TCS.NS": 0.3, "HDFCBANK.NS": 0.3},
    ...     individual_returns=stock_returns_df,
    ...     portfolio_value=1_000_000,
    ... )
    >>> print(metrics["sharpe"], metrics["max_drawdown"]["max_drawdown_pct"])
    """
    cfg = _load_config()
    rf = risk_free_rate if risk_free_rate is not None else cfg.get("risk_free_rate", 0.065)
    td = cfg.get("trading_days_per_year", 252)

    ann_return = compute_annualized_return(portfolio_returns, td)
    ann_vol = compute_annualized_volatility(portfolio_returns, td)
    sharpe = compute_sharpe(portfolio_returns, rf, td)
    sortino = compute_sortino(portfolio_returns, rf, td)
    drawdown = compute_max_drawdown(portfolio_returns)
    beta_series = compute_beta(portfolio_returns, benchmark_returns, td)
    var = compute_var(portfolio_returns, portfolio_value, None, mc_simulations)
    cvar = compute_cvar(portfolio_returns)
    dr = compute_diversification_ratio(weights, individual_returns, portfolio_returns, td)

    # Benchmark metrics for comparison
    bm_return = compute_annualized_return(benchmark_returns, td)
    bm_vol = compute_annualized_volatility(benchmark_returns, td)
    bm_sharpe = compute_sharpe(benchmark_returns, rf, td)
    bm_drawdown = compute_max_drawdown(benchmark_returns)

    return {
        # Portfolio metrics
        "annualized_return": round(ann_return * 100, 2),        # %
        "annualized_volatility": round(ann_vol * 100, 2),        # %
        "sharpe": round(sharpe, 4),
        "sortino": round(sortino, 4),
        "max_drawdown": drawdown,
        "rolling_beta": beta_series,
        "current_beta": round(float(beta_series.dropna().iloc[-1]), 4)
                        if not beta_series.dropna().empty else None,
        "var": var,
        "cvar": cvar,
        "diversification_ratio": dr,
        # Benchmark comparison
        "benchmark": {
            "annualized_return": round(bm_return * 100, 2),
            "annualized_volatility": round(bm_vol * 100, 2),
            "sharpe": round(bm_sharpe, 4),
            "max_drawdown_pct": bm_drawdown["max_drawdown_pct"],
        },
        # Metadata
        "risk_free_rate_used": rf,
        "portfolio_value": portfolio_value,
        "trading_days": td,
        "n_observations": len(portfolio_returns),
    }


# ---------------------------------------------------------------------------
# Inline Unit Tests
# Run: python src/risk_metrics.py
# ---------------------------------------------------------------------------

def _run_tests():
    import numpy as np
    np.random.seed(42)
    print("\n--- Running risk_metrics unit tests ---")

    dates = pd.bdate_range("2020-01-01", periods=500)
    bench = pd.Series(np.random.normal(0.0004, 0.012, 500), index=dates, name="^NSEI")
    port = bench * 1.2 + np.random.normal(0, 0.003, 500)
    port.index = dates

    # Sharpe
    s = compute_sharpe(port)
    assert isinstance(s, float), "Sharpe must be float"
    print(f"✅ Sharpe: {s:.4f}")

    # Sortino
    so = compute_sortino(port)
    assert isinstance(so, float), "Sortino must be float"
    assert so >= s, "Sortino should be >= Sharpe (less penalisation for upside)"
    print(f"✅ Sortino: {so:.4f}")

    # Max Drawdown
    dd = compute_max_drawdown(port)
    assert dd["max_drawdown_pct"] < 0, "Drawdown must be negative"
    print(f"✅ Max Drawdown: {dd['max_drawdown_pct']}%")

    # Beta
    beta = compute_beta(port, bench)
    last_beta = beta.dropna().iloc[-1]
    assert 0 < last_beta < 3, f"Beta {last_beta:.2f} out of expected range"
    print(f"✅ Rolling Beta (last): {last_beta:.4f}")

    # VaR
    var = compute_var(port, portfolio_value=1_000_000)
    assert var["var_historical"] > 0, "VaR must be positive loss"
    print(f"✅ VaR Historical: ₹{var['var_historical']:,.0f}")
    print(f"   VaR Parametric: ₹{var['var_parametric']:,.0f}")

    # CVaR
    cvar = compute_cvar(port)
    assert cvar["cvar_pct"] < 0, "CVaR must be negative (loss)"
    print(f"✅ CVaR: {cvar['cvar_pct']:.4f}%")

    # Diversification Ratio
    stocks = pd.DataFrame({
        "RELIANCE.NS": np.random.normal(0.0004, 0.015, 500),
        "TCS.NS": np.random.normal(0.0005, 0.013, 500),
        "HDFCBANK.NS": np.random.normal(0.0003, 0.011, 500),
    }, index=dates)
    weights = {"RELIANCE.NS": 0.4, "TCS.NS": 0.35, "HDFCBANK.NS": 0.25}
    dr = compute_diversification_ratio(weights, stocks, port)
    assert dr["diversification_ratio"] > 0, "DR must be positive"
    print(f"✅ Diversification Ratio: {dr['diversification_ratio']}")

    print("--- All risk_metrics tests passed ---\n")


if __name__ == "__main__":
    _run_tests()
