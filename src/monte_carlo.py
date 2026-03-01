"""
src/monte_carlo.py
==================
Cholesky-decomposition-based correlated Monte Carlo simulation.

SOP Rules Enforced (CRITICAL — non-negotiable):
    - Cholesky decomposition MUST be used to preserve cross-stock correlations
      Formula: R_correlated = L × Z  where L = chol(Σ), Z ~ N(0,1)
    - Plain independent simulation UNDERESTIMATES tail risk — never use it
    - All parameters read from config.yaml (paths, horizon, confidence)
    - Three drift modes supported: "historical", "user_defined", "zero"

Config parameters consumed:
    monte_carlo_paths         (default 10000)
    monte_carlo_horizon_years (default 3)
    trading_days_per_year     (default 252)
    var_confidence            (default 0.95)
    risk_free_rate            (default 0.065)

Public API:
    run_simulation(weights, returns, n_paths, horizon_days, drift,
                   user_drift, initial_value)
        -> SimulationResult (dataclass)

    extract_1day_returns(simulation_result)
        -> np.ndarray  [for passing to risk_metrics.compute_var()]

    build_simulation_summary(simulation_result, initial_value)
        -> dict         [ready for Streamlit display]

Cholesky motivation (SOP Section 5):
    Without Cholesky: stocks simulated independently — correlation structure
        destroyed, portfolio tail risks (crashes) massively underestimated.
    With Cholesky: correlated shocks ensure all stocks fall together during
        simulated crash scenarios, matching observed market behaviour.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

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
# Simulation result dataclass
# ---------------------------------------------------------------------------

@dataclass
class SimulationResult:
    """
    Container for all Monte Carlo simulation outputs.

    Attributes
    ----------
    paths           : np.ndarray, shape (n_paths, horizon_days + 1)
                      Portfolio VALUE (not return) over time.
                      Column 0 = initial_value. Column t = value at day t.
    daily_returns   : np.ndarray, shape (n_paths, horizon_days)
                      Simulated daily returns for the portfolio.
    weights         : dict[str, float]
                      The portfolio weights used.
    tickers         : list[str]
                      Ticker order matching the simulation.
    n_paths         : int
    horizon_days    : int
    drift_mode      : str  — "historical", "user_defined", or "zero"
    initial_value   : float — starting portfolio value in ₹
    daily_mu        : float — daily drift used in simulation
    daily_sigma     : float — portfolio daily volatility
    trading_days    : int   — from config
    """
    paths: np.ndarray
    daily_returns: np.ndarray
    weights: dict[str, float]
    tickers: list[str]
    n_paths: int
    horizon_days: int
    drift_mode: str
    initial_value: float
    daily_mu: float
    daily_sigma: float
    trading_days: int
    cholesky_L: np.ndarray = field(repr=False)   # preserve for diagnostics


# ---------------------------------------------------------------------------
# Internal: covariance + Cholesky from daily returns
# ---------------------------------------------------------------------------

def _compute_cholesky(returns: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute daily mean returns and Cholesky factor of the covariance matrix.

    SOP Formula:
        If Σ = covariance matrix of daily returns, then
        L = chol(Σ)   (lower triangular Cholesky factor)
        Correlated shocks: L × Z  where Z ~ N(0, I)

    Regularisation: Add a small diagonal (1e-8 × I) if Σ is not
    positive-definite due to numerical noise — common with short history
    or highly correlated Indian stocks.

    Returns
    -------
    mu_daily  : np.ndarray shape (n,) — mean daily return per stock
    cov_daily : np.ndarray shape (n,n) — daily covariance matrix
    L         : np.ndarray shape (n,n) — Cholesky lower-triangular factor

    Unit Test Cases:
        Input:  perfectly correlated 2 stocks
        Expect: L shape (2,2), L @ L.T ≈ Σ  (reconstruction check)

        Input:  cov Σ with tiny negative eigenvalue (numerical noise)
        Expect: regularisation applied, no LinAlgError raised
    """
    clean = returns.dropna()
    mu_daily = clean.mean().values
    cov_daily = clean.cov().values

    # Symmetrise (guard against floating point asymmetry)
    cov_daily = (cov_daily + cov_daily.T) / 2.0

    # Regularise if needed
    min_eig = np.min(np.linalg.eigvalsh(cov_daily))
    if min_eig < 1e-10:
        eps = max(abs(min_eig) + 1e-8, 1e-8)
        cov_daily += eps * np.eye(len(mu_daily))
        logger.info(f"Covariance matrix regularised by {eps:.2e} for Cholesky stability.")

    try:
        L = np.linalg.cholesky(cov_daily)
    except np.linalg.LinAlgError as e:
        raise ValueError(
            f"Cholesky decomposition failed even after regularisation: {e}. "
            "Check for tickers with zero or constant price history."
        ) from e

    return mu_daily, cov_daily, L


def _resolve_drift(
    mode: Literal["historical", "user_defined", "zero"],
    weights_arr: np.ndarray,
    mu_daily: np.ndarray,
    user_drift: Optional[float],
) -> float:
    """
    Resolve the daily portfolio drift (μ) based on the selected mode.

    Modes (SOP Section 4):
        "historical"   — use historical mean daily portfolio return
                         μ_p = Σ(wi × μi)
        "user_defined" — use the annual return supplied by the user
                         converted to daily: μ = (1 + annual)^(1/252) - 1
        "zero"         — conservative; no drift (μ = 0)
                         equivalent to assuming no real return above cash

    Unit Test Cases:
        mode="zero"           → returns exactly 0.0
        mode="historical"     → returns weighted sum of mu_daily
        mode="user_defined", user_drift=0.12  → returns (1.12)^(1/252) - 1 ≈ 0.000449
    """
    cfg = _load_config()
    td = cfg.get("trading_days_per_year", 252)

    if mode == "zero":
        return 0.0
    elif mode == "historical":
        return float(weights_arr @ mu_daily)
    elif mode == "user_defined":
        if user_drift is None:
            raise ValueError(
                "user_drift (annual return) must be provided when drift='user_defined'. "
                "Example: user_drift=0.12 for 12% annual expected return."
            )
        return float((1 + user_drift) ** (1 / td) - 1)
    else:
        raise ValueError(
            f"Invalid drift mode '{mode}'. "
            "Choose one of: 'historical', 'user_defined', 'zero'."
        )


# ---------------------------------------------------------------------------
# Core simulation engine
# ---------------------------------------------------------------------------

def run_simulation(
    weights: dict[str, float],
    returns: pd.DataFrame,
    n_paths: Optional[int] = None,
    horizon_days: Optional[int] = None,
    drift: Literal["historical", "user_defined", "zero"] = "historical",
    user_drift: Optional[float] = None,
    initial_value: float = 1_000_000.0,
    random_seed: Optional[int] = 42,
) -> SimulationResult:
    """
    Run correlated Monte Carlo simulation using Cholesky decomposition.

    SOP Formula (Section 4 & 8):
        L = chol(Σ)                     — Cholesky of daily cov matrix
        Z ~ N(0, I)  shape (n, T)       — independent standard normals
        R_correlated = (L @ Z).T       — correlated daily stock returns shape (T, n)
        R_portfolio  = R_correlated @ w — weighted portfolio return per day

    This ensures stock returns move together the way they do in reality.
    During crash simulations, correlated stocks ALL fall together, so
    portfolio tail losses are realistically large (not averaged away).

    Parameters
    ----------
    weights       : dict[str, float] — {ticker: weight}, must sum to 1.0
    returns       : pd.DataFrame     — historical daily returns (columns = tickers)
    n_paths       : int              — number of simulated paths (default from config)
    horizon_days  : int              — simulation horizon in trading days
                                       (default = monte_carlo_horizon_years × 252)
    drift         : str              — "historical" | "user_defined" | "zero"
    user_drift    : float            — annual drift (only for drift="user_defined")
    initial_value : float            — starting portfolio value in ₹
    random_seed   : int or None      — for reproducibility (None = random)

    Returns
    -------
    SimulationResult dataclass — see class definition above

    Output shape:
        result.paths         shape (n_paths, horizon_days + 1)  — VALUE over time
        result.daily_returns shape (n_paths, horizon_days)       — daily return per path

    Unit Test Cases:
        Input:  n_paths=100, horizon_days=252, drift="zero"
        Expect: result.paths.shape == (100, 253)
                result.paths[:, 0] == initial_value (all paths start at same value)
                result.daily_mu == 0.0

        Input:  drift="historical"
        Expect: result.daily_mu == weighted mean of historical daily returns

        Input:  n_paths=1000, drift="zero"
        Expect: result.paths[:, -1].mean() ≈ initial_value (no drift, centered)
    """
    cfg = _load_config()
    td = cfg.get("trading_days_per_year", 252)
    n_paths = n_paths or cfg.get("monte_carlo_paths", 10000)
    horizon_years = cfg.get("monte_carlo_horizon_years", 3)
    horizon_days = horizon_days or int(horizon_years * td)

    # Validate weights
    tickers = list(weights.keys())
    total_w = sum(weights.values())
    if abs(total_w - 1.0) > 1e-4:
        raise ValueError(f"Weights must sum to 1.0 (got {total_w:.4f}).")

    missing = [t for t in tickers if t not in returns.columns]
    if missing:
        raise ValueError(f"Tickers not in returns DataFrame: {missing}")

    weights_arr = np.array([weights[t] for t in tickers])
    stock_returns = returns[tickers].dropna()

    # --- Cholesky decomposition (SOP core requirement) ---
    mu_daily, cov_daily, L = _compute_cholesky(stock_returns)

    # --- Resolve drift ---
    daily_mu = _resolve_drift(drift, weights_arr, mu_daily, user_drift)
    daily_sigma = float(np.sqrt(weights_arr @ cov_daily @ weights_arr))

    logger.info(
        f"Monte Carlo: {n_paths:,} paths × {horizon_days} days | "
        f"drift={drift} ({daily_mu*252*100:.2f}% ann.) | "
        f"σ_daily={daily_sigma*100:.3f}%"
    )

    # --- Simulate ---
    if random_seed is not None:
        np.random.seed(random_seed)

    n_stocks = len(tickers)

    # Z shape: (n_stocks, horizon_days × n_paths) — independent standard normals
    # Reshaped to (n_paths, horizon_days, n_stocks) after correlation injection
    Z = np.random.standard_normal((n_paths, horizon_days, n_stocks))

    # Apply Cholesky: for each day and path, R_corr = (L @ z)
    # Vectorised: Z @ L.T gives shape (n_paths, horizon_days, n_stocks)
    R_correlated = Z @ L.T          # shape: (n_paths, horizon_days, n_stocks)

    # Add drift to each stock's return
    R_correlated += mu_daily         # broadcast: mu_daily shape (n_stocks,)

    # Weighted portfolio return per day: shape (n_paths, horizon_days)
    port_daily_returns = R_correlated @ weights_arr

    # Replace drift with selected mode drift for the portfolio
    # (mu_daily applied above was per-stock; adjust to use resolved portfolio drift)
    port_daily_returns = port_daily_returns - (weights_arr @ mu_daily) + daily_mu

    # Build cumulative value paths: shape (n_paths, horizon_days + 1)
    # Column 0 = initial_value; column t = value at end of day t
    cum_factor = np.cumprod(1 + port_daily_returns, axis=1)  # (n_paths, horizon)
    paths = np.empty((n_paths, horizon_days + 1))
    paths[:, 0] = initial_value
    paths[:, 1:] = initial_value * cum_factor

    return SimulationResult(
        paths=paths,
        daily_returns=port_daily_returns,
        weights=weights,
        tickers=tickers,
        n_paths=n_paths,
        horizon_days=horizon_days,
        drift_mode=drift,
        initial_value=initial_value,
        daily_mu=daily_mu,
        daily_sigma=daily_sigma,
        trading_days=td,
        cholesky_L=L,
    )


# ---------------------------------------------------------------------------
# Extract 1-day simulated returns (for risk_metrics.compute_var)
# ---------------------------------------------------------------------------

def extract_1day_returns(result: SimulationResult) -> np.ndarray:
    """
    Extract the first simulated day's return from each path.

    Used to pass Monte Carlo returns into risk_metrics.compute_var()
    as the mc_simulations parameter for Monte Carlo VaR.

    Returns
    -------
    np.ndarray shape (n_paths,) — one 1-day return per simulated path

    Unit Test Case:
        Input:  result from run_simulation(n_paths=1000, horizon_days=252)
        Expect: len(extract_1day_returns(result)) == 1000
    """
    return result.daily_returns[:, 0]


# ---------------------------------------------------------------------------
# Summary report builder
# ---------------------------------------------------------------------------

def build_simulation_summary(
    result: SimulationResult,
    initial_value: Optional[float] = None,
) -> dict:
    """
    Build a structured summary report from a SimulationResult.

    Computes expected values, percentile bands, and loss probabilities
    at 1, 2, and 3-year horizons (and the configured horizon if different).

    Parameters
    ----------
    result        : SimulationResult — output of run_simulation()
    initial_value : float            — portfolio start value (default: result.initial_value)

    Returns
    -------
    dict ready for Streamlit display:
    {
        "drift_mode"    : str,
        "n_paths"       : int,
        "horizon_days"  : int,
        "initial_value" : float,
        "daily_mu"      : float,
        "annual_mu_pct" : float,
        "daily_sigma"   : float,
        "annual_sigma_pct": float,
        "horizons": {
            "1yr": { "day": int, "expected": float, "p5": float, "p95": float,
                     "p_loss": float, "expected_return_pct": float },
            "2yr": { ... },
            "3yr": { ... },
            "end": { ... }    <- configured horizon (may equal 3yr)
        },
        "final_distribution": {
            "mean": float, "median": float, "std": float,
            "p5": float, "p25": float, "p75": float, "p95": float,
        },
        "interpretation": [str, ...]   <- human-readable insights
    }

    Unit Test Cases:
        Input:  1000 paths, horizon=252 (1 year), drift="zero"
        Expect: summary["horizons"]["1yr"]["p_loss"] ≈ 0.50  (50/50 for zero drift)
                summary["horizons"]["1yr"]["expected"] ≈ initial_value (zero drift)

        Input:  drift="historical" with strongly positive mu
        Expect: summary["horizons"]["3yr"]["expected"] > initial_value
                summary["horizons"]["3yr"]["p_loss"] < 0.30
    """
    cfg = _load_config()
    td = result.trading_days
    iv = initial_value or result.initial_value
    paths = result.paths       # shape (n_paths, horizon_days + 1)

    ann_mu_pct = result.daily_mu * td * 100
    ann_sigma_pct = result.daily_sigma * np.sqrt(td) * 100

    def _horizon_stats(day_idx: int) -> dict:
        """Compute stats at a given day column index."""
        if day_idx > result.horizon_days:
            day_idx = result.horizon_days
        vals = paths[:, day_idx]
        expected = float(np.mean(vals))
        p5 = float(np.percentile(vals, 5))
        p95 = float(np.percentile(vals, 95))
        p_loss = float(np.mean(vals < iv))
        exp_ret = (expected - iv) / iv * 100
        return {
            "day": day_idx,
            "expected": round(expected, 2),
            "p5": round(p5, 2),
            "p95": round(p95, 2),
            "p_loss": round(p_loss, 4),
            "p_loss_pct": round(p_loss * 100, 2),
            "expected_return_pct": round(exp_ret, 2),
        }

    # Compute at standard horizons
    horizons = {}
    for label, years in [("1yr", 1), ("2yr", 2), ("3yr", 3)]:
        day_idx = min(int(years * td), result.horizon_days)
        horizons[label] = _horizon_stats(day_idx)

    # Also compute at the actual simulation end
    horizons["end"] = _horizon_stats(result.horizon_days)
    horizons["end"]["label"] = f"{result.horizon_days // td:.1f}yr"

    # Final day distribution
    final_vals = paths[:, -1]
    final_dist = {
        "mean":   round(float(np.mean(final_vals)), 2),
        "median": round(float(np.median(final_vals)), 2),
        "std":    round(float(np.std(final_vals)), 2),
        "p5":     round(float(np.percentile(final_vals, 5)), 2),
        "p25":    round(float(np.percentile(final_vals, 25)), 2),
        "p75":    round(float(np.percentile(final_vals, 75)), 2),
        "p95":    round(float(np.percentile(final_vals, 95)), 2),
    }

    # Human-readable interpretations
    insights = []
    h3 = horizons.get("3yr") or horizons["end"]
    h1 = horizons["1yr"]

    insights.append(
        f"Expected portfolio value in 3 years: ₹{h3['expected']:,.0f} "
        f"(+{h3['expected_return_pct']:.1f}% from today's ₹{iv:,.0f})."
    )
    insights.append(
        f"Worst-case (5th percentile) in 3 years: ₹{h3['p5']:,.0f} | "
        f"Best-case (95th percentile): ₹{h3['p95']:,.0f}."
    )
    insights.append(
        f"Probability of capital loss after 1 year: {h1['p_loss_pct']:.1f}% "
        f"| After 3 years: {h3['p_loss_pct']:.1f}%."
    )
    if result.drift_mode == "zero":
        insights.append(
            "⚠️ Conservative mode: zero drift assumes no real return above inflation. "
            "This is the most cautious projection."
        )
    elif result.drift_mode == "historical":
        insights.append(
            f"Using historical drift of {ann_mu_pct:.2f}% annualised. "
            "Past returns may not repeat — treat as reference, not forecast."
        )

    return {
        "drift_mode": result.drift_mode,
        "n_paths": result.n_paths,
        "horizon_days": result.horizon_days,
        "initial_value": iv,
        "daily_mu": round(result.daily_mu, 8),
        "annual_mu_pct": round(ann_mu_pct, 4),
        "daily_sigma": round(result.daily_sigma, 8),
        "annual_sigma_pct": round(ann_sigma_pct, 4),
        "horizons": horizons,
        "final_distribution": final_dist,
        "interpretation": insights,
    }


# ---------------------------------------------------------------------------
# Plotly-ready path sampler (called from app.py)
# ---------------------------------------------------------------------------

def sample_paths_for_plot(
    result: SimulationResult,
    n_display: int = 200,
    percentiles: tuple[int, ...] = (5, 25, 50, 75, 95),
    random_seed: int = 0,
) -> dict:
    """
    Sample a small subset of paths and compute percentile bands for plotting.

    Plotting all 10,000 paths in Plotly would be too slow.
    This returns a thin sample + smooth percentile bands.

    Parameters
    ----------
    result       : SimulationResult
    n_display    : int  — number of individual paths to show (default 200)
    percentiles  : tuple — which bands to compute (default: 5/25/50/75/95)
    random_seed  : int  — for reproducible sampling

    Returns
    -------
    dict:
        {
            "days"          : list[int]         — x-axis (day 0 to horizon)
            "sampled_paths" : np.ndarray (n_display, horizon+1) — thin sample
            "bands"         : {str: np.ndarray} — e.g. {"p5": array, "p50": array}
            "initial_value" : float
        }
    """
    rng = np.random.default_rng(random_seed)
    idx = rng.choice(result.n_paths, size=min(n_display, result.n_paths), replace=False)
    sampled = result.paths[idx]

    bands = {}
    for p in percentiles:
        bands[f"p{p}"] = np.percentile(result.paths, p, axis=0)

    days = list(range(result.horizon_days + 1))
    return {
        "days": days,
        "sampled_paths": sampled,
        "bands": bands,
        "initial_value": result.initial_value,
    }


# ---------------------------------------------------------------------------
# Inline Unit Tests
# Run: python src/monte_carlo.py
# ---------------------------------------------------------------------------

def _run_tests():
    import numpy as np
    print("\n--- Running monte_carlo unit tests ---")

    # Build 4 correlated synthetic stocks
    dates = pd.bdate_range("2018-01-01", periods=1000)
    corr = np.array([
        [1.00, 0.65, 0.55, 0.50],
        [0.65, 1.00, 0.60, 0.45],
        [0.55, 0.60, 1.00, 0.40],
        [0.50, 0.45, 0.40, 1.00],
    ])
    vols = np.array([0.015, 0.013, 0.011, 0.012])
    cov = np.outer(vols, vols) * corr
    L_true = np.linalg.cholesky(cov)
    np.random.seed(42)
    Z = np.random.randn(4, 1000)
    raw = (L_true @ Z).T + np.array([0.0005, 0.0004, 0.0003, 0.0004])
    tickers = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS"]
    returns = pd.DataFrame(raw, index=dates, columns=tickers)
    weights = {"RELIANCE.NS": 0.30, "TCS.NS": 0.30, "HDFCBANK.NS": 0.25, "INFY.NS": 0.15}

    # 1. Cholesky reconstruction check
    _, cov_est, L_est = _compute_cholesky(returns)
    reconstruction_err = np.max(np.abs(L_est @ L_est.T - cov_est))
    assert reconstruction_err < 1e-12, f"Cholesky reconstruction error: {reconstruction_err:.2e}"
    print(f"✅ Cholesky: L @ L.T ≈ Σ (max error {reconstruction_err:.2e})")

    # 2. Drift modes
    mu_daily = returns.mean().values
    w_arr = np.array([0.30, 0.30, 0.25, 0.15])
    assert _resolve_drift("zero", w_arr, mu_daily, None) == 0.0
    assert abs(_resolve_drift("historical", w_arr, mu_daily, None) - float(w_arr @ mu_daily)) < 1e-10
    exp_ud = (1.12) ** (1/252) - 1
    assert abs(_resolve_drift("user_defined", w_arr, mu_daily, 0.12) - exp_ud) < 1e-10
    print("✅ All 3 drift modes resolve correctly")

    # 3. Output shape
    result = run_simulation(weights, returns, n_paths=500, horizon_days=252,
                            drift="historical", random_seed=42)
    assert result.paths.shape == (500, 253),  f"Expected (500,253), got {result.paths.shape}"
    assert result.daily_returns.shape == (500, 252), f"Expected (500,252)"
    print(f"✅ Output shape: paths={result.paths.shape}, returns={result.daily_returns.shape}")

    # 4. All paths start at initial_value
    assert np.all(result.paths[:, 0] == 1_000_000.0), "All paths must start at initial_value"
    print("✅ All 500 paths start at ₹1,000,000")

    # 5. Weights sum check
    assert abs(sum(weights.values()) - 1.0) < 1e-10
    print("✅ Weights sum to 1.0")

    # 6. Zero drift: expected final value ≈ initial (within 2 std of mean over paths)
    r_zero = run_simulation(weights, returns, n_paths=5000, horizon_days=252,
                            drift="zero", random_seed=99)
    final_mean = r_zero.paths[:, -1].mean()
    assert abs(final_mean - 1_000_000) < 50_000, \
        f"Zero drift: final mean={final_mean:.0f}, expected ≈ 1,000,000"
    print(f"✅ Zero drift: final mean ₹{final_mean:,.0f} (expected ≈ ₹1,000,000)")

    # 7. Summary report
    summary = build_simulation_summary(result)
    assert "horizons" in summary
    assert "1yr" in summary["horizons"]
    assert 0 <= summary["horizons"]["1yr"]["p_loss"] <= 1
    print(f"✅ Summary: P(loss 1yr)={summary['horizons']['1yr']['p_loss_pct']:.1f}% "
          f"| Expected 3yr=₹{summary['horizons']['3yr']['expected']:,.0f}")

    # 8. 1-day return extraction (for MC VaR)
    one_day = extract_1day_returns(result)
    assert len(one_day) == 500, f"Expected 500 returns, got {len(one_day)}"
    print(f"✅ 1-day returns extracted: shape {one_day.shape}, std={one_day.std():.4f}")

    # 9. Path sampler
    plot_data = sample_paths_for_plot(result, n_display=50)
    assert plot_data["sampled_paths"].shape == (50, 253)
    assert "p5" in plot_data["bands"] and "p95" in plot_data["bands"]
    print(f"✅ Plot sampler: {plot_data['sampled_paths'].shape[0]} paths, "
          f"{len(plot_data['bands'])} percentile bands")

    print("--- All monte_carlo tests passed ---\n")


if __name__ == "__main__":
    _run_tests()
