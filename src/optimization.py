"""
src/optimization.py
===================
CVXPY-based mean-variance portfolio optimization.

SOP Rules Enforced (CRITICAL — never change these):
    - Efficient frontier uses EXACT quadratic programming (CVXPY) — NEVER random simulation
    - Constraints always enforced: w ≥ 0, Σw = 1, w ≤ max_weight_per_stock
    - max_weight_per_stock read from config.yaml (default: 0.40)
    - risk_free_rate read from config.yaml — never hardcoded
    - Random portfolio scatter is a VISUAL REFERENCE ONLY — computed separately,
      never replaces the QP frontier

QP Formulation (SOP Section 8):
    Minimize:   wᵀΣw
    Subject to: wᵀμ = target_return
                Σw = 1
                w ≥ 0
                w ≤ max_weight_per_stock

Public API:
    compute_efficient_frontier(returns, n_points)       -> pd.DataFrame
    max_sharpe_portfolio(returns, risk_free_rate)        -> tuple[np.ndarray, dict]
    min_variance_portfolio(returns)                      -> tuple[np.ndarray, dict]
    solve_target_return(returns, target_return)          -> tuple[np.ndarray, dict]
    random_portfolio_scatter(returns, n_portfolios)      -> pd.DataFrame  [visual only]
    portfolio_metrics_from_weights(weights_arr, tickers,
                                   returns, risk_free_rate) -> dict
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Optional

import cvxpy as cp
import numpy as np
import pandas as pd
import yaml
from scipy.optimize import minimize

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=UserWarning)

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
# Internal helpers
# ---------------------------------------------------------------------------

def _prepare_inputs(returns: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Compute annualized mean returns and covariance matrix from daily returns.

    Returns
    -------
    mu      : np.ndarray shape (n,) — annualized expected returns
    sigma   : np.ndarray shape (n,n) — annualized covariance matrix
    tickers : list[str]
    """
    cfg = _load_config()
    td = cfg.get("trading_days_per_year", 252)
    tickers = list(returns.columns)
    clean = returns.dropna()

    mu = clean.mean().values * td                       # Annualized
    sigma = clean.cov().values * td                     # Annualized

    # Ensure covariance matrix is positive semi-definite (numerical stability)
    min_eig = np.min(np.linalg.eigvalsh(sigma))
    if min_eig < 0:
        sigma -= 1.1 * min_eig * np.eye(len(tickers))    # Regularize

    return mu, sigma, tickers


def _build_constraints(
    w: cp.Variable,
    n: int,
    max_weight: float,
    target_return: Optional[float] = None,
    mu: Optional[np.ndarray] = None,
) -> list:
    """
    Build the standard CVXPY constraint set per SOP.

    Constraints always applied:
        1. Σw = 1  (fully invested)
        2. w ≥ 0   (no short selling)
        3. w ≤ max_weight_per_stock  (concentration limit)

    Optional:
        4. wᵀμ = target_return  (used when solving frontier points)
    """
    constraints = [
        cp.sum(w) == 1,                     # Fully invested
        w >= 0,                             # No short selling
        w <= max_weight,                    # Concentration cap from config
    ]
    if target_return is not None and mu is not None:
        constraints.append(mu @ w == target_return)

    return constraints


def portfolio_metrics_from_weights(
    weights_arr: np.ndarray,
    tickers: list[str],
    returns: pd.DataFrame,
    risk_free_rate: Optional[float] = None,
) -> dict:
    """
    Compute annualized return, volatility, and Sharpe for a given weight vector.

    Parameters
    ----------
    weights_arr   : np.ndarray — weight vector, shape (n,)
    tickers       : list[str]  — matching order of weights_arr
    returns       : pd.DataFrame — daily returns (columns = tickers)
    risk_free_rate: float — annual rate (default from config)

    Returns
    -------
    dict: {return_pct, vol_pct, sharpe, weights_dict}
    """
    cfg = _load_config()
    rf = risk_free_rate if risk_free_rate is not None else cfg.get("risk_free_rate", 0.065)
    td = cfg.get("trading_days_per_year", 252)

    mu, sigma, _ = _prepare_inputs(returns[tickers])
    port_return = float(mu @ weights_arr)
    port_vol = float(np.sqrt(weights_arr @ sigma @ weights_arr))
    sharpe = (port_return - rf) / port_vol if port_vol > 0 else 0.0

    return {
        "return_pct": round(port_return * 100, 4),
        "vol_pct": round(port_vol * 100, 4),
        "sharpe": round(sharpe, 4),
        "weights_dict": {t: round(float(w), 6) for t, w in zip(tickers, weights_arr)},
    }


# ---------------------------------------------------------------------------
# 1. Solve at a specific target return (building block for the frontier)
# ---------------------------------------------------------------------------

def solve_target_return(
    returns: pd.DataFrame,
    target_return: float,
    max_weight: Optional[float] = None,
) -> tuple[np.ndarray | None, dict]:
    """
    Solve the QP: minimize portfolio variance at a specific target return.

    SOP QP Formulation:
        Minimize:   wᵀΣw
        Subject to: wᵀμ = target_return
                    Σw = 1
                    w ≥ 0
                    w ≤ max_weight_per_stock

    Parameters
    ----------
    returns       : pd.DataFrame — daily returns (columns = tickers)
    target_return : float        — annualized target return (e.g. 0.15 = 15%)
    max_weight    : float        — max single-stock weight (default from config)

    Returns
    -------
    weights : np.ndarray or None (None if problem infeasible)
    info    : dict — {feasible, return_pct, vol_pct, sharpe, solver_status}

    Unit Test Case:
        Input:  3 stocks, target_return = achievable value (between min and max mu)
        Expect: weights.sum() ≈ 1.0
                all(weights >= 0)
                all(weights <= max_weight)
                info["feasible"] == True

        Input:  target_return > max(mu) of any stock
        Expect: info["feasible"] == False, weights == None
    """
    cfg = _load_config()
    mw = max_weight if max_weight is not None else cfg.get("max_weight_per_stock", 0.40)
    rf = cfg.get("risk_free_rate", 0.065)

    mu, sigma, tickers = _prepare_inputs(returns)
    n = len(tickers)

    w = cp.Variable(n)
    portfolio_variance = cp.quad_form(w, sigma)

    constraints = _build_constraints(w, n, mw, target_return, mu)
    problem = cp.Problem(cp.Minimize(portfolio_variance), constraints)

    try:
        problem.solve(solver=cp.CLARABEL, warm_start=True)
    except Exception:
        try:
            problem.solve(solver=cp.SCS)
        except Exception as e:
            logger.warning(f"QP solver failed for target_return={target_return:.4f}: {e}")
            return None, {"feasible": False, "solver_status": str(e)}

    if problem.status not in ("optimal", "optimal_inaccurate") or w.value is None:
        return None, {"feasible": False, "solver_status": problem.status}

    weights_arr = np.clip(w.value, 0, 1)           # Clip numerical noise
    weights_arr /= weights_arr.sum()               # Re-normalise to exactly 1

    port_vol = float(np.sqrt(weights_arr @ sigma @ weights_arr))
    port_ret = float(mu @ weights_arr)
    sharpe = (port_ret - rf) / port_vol if port_vol > 0 else 0.0

    return weights_arr, {
        "feasible": True,
        "return_pct": round(port_ret * 100, 4),
        "vol_pct": round(port_vol * 100, 4),
        "sharpe": round(sharpe, 4),
        "solver_status": problem.status,
        "weights_dict": {t: round(float(ww), 6) for t, ww in zip(tickers, weights_arr)},
    }


# ---------------------------------------------------------------------------
# 2. Efficient Frontier (exact QP — SOP core differentiator)
# ---------------------------------------------------------------------------

def compute_efficient_frontier(
    returns: pd.DataFrame,
    n_points: int = 100,
    max_weight: Optional[float] = None,
) -> pd.DataFrame:
    """
    Compute the exact mean-variance efficient frontier via CVXPY.

    SOP Rule: This is the CORE DIFFERENTIATOR of this project.
    NEVER replace this with random portfolio simulation.
    Every point on the returned DataFrame is an exactly solved QP.

    Algorithm:
        1. Find the minimum-variance portfolio return (lower bound)
        2. Find the maximum achievable return (upper bound = max stock return
           subject to constraints)
        3. Iterate over n_points evenly spaced target returns in [lb, ub]
        4. Solve QP at each target return
        5. Return all feasible solutions as a DataFrame

    Parameters
    ----------
    returns   : pd.DataFrame — daily returns (columns = NSE tickers with .NS)
    n_points  : int          — number of frontier points (default: 100)
    max_weight: float        — concentration cap (default from config.yaml)

    Returns
    -------
    pd.DataFrame with columns:
        return_pct  : float — annualized portfolio return (%)
        vol_pct     : float — annualized portfolio volatility (%)
        sharpe      : float — Sharpe ratio
        weights_dict: dict  — {ticker: weight} for each frontier point

    Unit Test Case:
        Input:  3 stocks with valid return history, n_points=20
        Expect: len(frontier) > 0 (at least some feasible points)
                frontier["vol_pct"].is_monotonic_increasing after sorting by return
                frontier["return_pct"].between(0, 200).all()  (realistic range)
    """
    cfg = _load_config()
    mw = max_weight if max_weight is not None else cfg.get("max_weight_per_stock", 0.40)

    mu, sigma, tickers = _prepare_inputs(returns)
    n = len(tickers)

    # --- Bounds for target returns ---
    # Lower bound: minimum variance portfolio return
    min_var_weights, min_var_info = min_variance_portfolio(returns, mw)
    if min_var_weights is None:
        raise ValueError("Could not solve min-variance portfolio. Check your input data.")
    min_ret = mu @ min_var_weights

    # Upper bound: max achievable return under weight constraints
    # = weighted portfolio where all weight goes to the highest-return stock(s)
    # Solve max return QP: maximize wᵀμ subject to constraints
    w_max = cp.Variable(n)
    max_ret_problem = cp.Problem(
        cp.Maximize(mu @ w_max),
        _build_constraints(w_max, n, mw),
    )
    try:
        max_ret_problem.solve(solver=cp.CLARABEL)
        max_ret = float(mu @ w_max.value) if w_max.value is not None else float(np.max(mu))
    except Exception:
        max_ret = float(np.max(mu))

    if max_ret <= min_ret:
        raise ValueError(
            f"No return range for the frontier: min={min_ret:.4f}, max={max_ret:.4f}. "
            "Select more stocks with varied return profiles."
        )

    # --- Solve QP at each target return ---
    target_returns = np.linspace(min_ret, max_ret, n_points)
    frontier_points = []

    logger.info(f"Computing efficient frontier: {n_points} points, {n} stocks ...")

    for target in target_returns:
        weights_arr, info = solve_target_return(returns, target, mw)
        if info["feasible"] and weights_arr is not None:
            frontier_points.append({
                "return_pct": info["return_pct"],
                "vol_pct": info["vol_pct"],
                "sharpe": info["sharpe"],
                "weights_dict": info["weights_dict"],
            })

    if not frontier_points:
        raise ValueError("Efficient frontier computation returned no feasible points.")

    frontier_df = pd.DataFrame(frontier_points).sort_values("return_pct").reset_index(drop=True)
    logger.info(f"Frontier computed: {len(frontier_df)} feasible points out of {n_points} attempted.")
    return frontier_df


# ---------------------------------------------------------------------------
# 3. Maximum Sharpe Ratio Portfolio
# ---------------------------------------------------------------------------

def max_sharpe_portfolio(
    returns: pd.DataFrame,
    risk_free_rate: Optional[float] = None,
    max_weight: Optional[float] = None,
) -> tuple[np.ndarray, dict]:
    """
    Find the portfolio with the maximum Sharpe Ratio.

    Method:
        Uses SciPy minimize (SLSQP) to directly maximize Sharpe.
        CVXPY cannot maximize Sharpe directly (it's non-convex in weights)
        so we use numerical optimization with the same constraints.
        The efficient frontier itself remains CVXPY-only.

    Parameters
    ----------
    returns        : pd.DataFrame — daily returns
    risk_free_rate : float        — annual rate (default from config)
    max_weight     : float        — concentration cap (default from config)

    Returns
    -------
    weights : np.ndarray — optimal weight vector, shape (n,)
    metrics : dict       — {return_pct, vol_pct, sharpe, weights_dict}

    Unit Test Case:
        Input:  3 stocks with diverse risk/return profiles
        Expect: weights.sum() ≈ 1.0
                all(weights >= -1e-6)  (non-negative within tolerance)
                all(weights <= max_weight + 1e-6)
                metrics["sharpe"] > min_var_metrics["sharpe"]  (better than min-var)
    """
    cfg = _load_config()
    rf = risk_free_rate if risk_free_rate is not None else cfg.get("risk_free_rate", 0.065)
    mw = max_weight if max_weight is not None else cfg.get("max_weight_per_stock", 0.40)

    mu, sigma, tickers = _prepare_inputs(returns)
    n = len(tickers)

    def neg_sharpe(w: np.ndarray) -> float:
        port_ret = float(mu @ w)
        port_vol = float(np.sqrt(w @ sigma @ w))
        if port_vol < 1e-10:
            return 0.0
        return -(port_ret - rf) / port_vol

    def neg_sharpe_grad(w: np.ndarray) -> np.ndarray:
        port_ret = float(mu @ w)
        port_vol = float(np.sqrt(w @ sigma @ w))
        if port_vol < 1e-10:
            return np.zeros(n)
        sr = (port_ret - rf) / port_vol
        d_ret = mu
        d_vol = (sigma @ w) / port_vol
        return -(d_ret * port_vol - (port_ret - rf) * d_vol) / (port_vol ** 2)

    # Constraints: Σw = 1, w ≥ 0, w ≤ mw
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    bounds = [(0.0, mw)] * n

    # Multiple random starts to avoid local optima
    best_result = None
    best_sharpe = -np.inf
    np.random.seed(42)

    for _ in range(20):
        # Random feasible starting point
        w0 = np.random.dirichlet(np.ones(n))
        w0 = np.clip(w0, 0, mw)
        w0 /= w0.sum()

        result = minimize(
            neg_sharpe,
            w0,
            jac=neg_sharpe_grad,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol": 1e-12, "maxiter": 1000},
        )
        if result.success and -result.fun > best_sharpe:
            best_sharpe = -result.fun
            best_result = result

    if best_result is None or not best_result.success:
        logger.warning("Max-Sharpe optimisation did not converge cleanly. Returning best found.")
        # Fallback: pick point from frontier with highest Sharpe
        frontier = compute_efficient_frontier(returns, n_points=50, max_weight=mw)
        best_row = frontier.loc[frontier["sharpe"].idxmax()]
        w_best = np.array([best_row["weights_dict"][t] for t in tickers])
        metrics = portfolio_metrics_from_weights(w_best, tickers, returns, rf)
        return w_best, metrics

    w_best = np.clip(best_result.x, 0, 1)
    w_best /= w_best.sum()

    metrics = portfolio_metrics_from_weights(w_best, tickers, returns, rf)
    logger.info(f"Max-Sharpe portfolio: Sharpe={metrics['sharpe']:.4f}, "
                f"Return={metrics['return_pct']:.2f}%, Vol={metrics['vol_pct']:.2f}%")
    return w_best, metrics


# ---------------------------------------------------------------------------
# 4. Minimum Variance Portfolio
# ---------------------------------------------------------------------------

def min_variance_portfolio(
    returns: pd.DataFrame,
    max_weight: Optional[float] = None,
) -> tuple[np.ndarray | None, dict]:
    """
    Find the global minimum variance portfolio using CVXPY.

    SOP QP (no return constraint — pure variance minimization):
        Minimize:   wᵀΣw
        Subject to: Σw = 1
                    w ≥ 0
                    w ≤ max_weight_per_stock

    Parameters
    ----------
    returns    : pd.DataFrame — daily returns
    max_weight : float        — concentration cap (default from config)

    Returns
    -------
    weights : np.ndarray or None
    metrics : dict — {return_pct, vol_pct, sharpe, weights_dict, solver_status}

    Unit Test Case:
        Input:  3 stocks
        Expect: portfolio vol <= any individual stock vol × any weight combo
                weights.sum() ≈ 1.0, all(weights >= 0)
                metrics["vol_pct"] < max(individual annual vols)
    """
    cfg = _load_config()
    mw = max_weight if max_weight is not None else cfg.get("max_weight_per_stock", 0.40)
    rf = cfg.get("risk_free_rate", 0.065)

    mu, sigma, tickers = _prepare_inputs(returns)
    n = len(tickers)

    w = cp.Variable(n)
    objective = cp.Minimize(cp.quad_form(w, sigma))
    constraints = _build_constraints(w, n, mw)          # No return target
    problem = cp.Problem(objective, constraints)

    try:
        problem.solve(solver=cp.CLARABEL, warm_start=True)
    except Exception:
        try:
            problem.solve(solver=cp.SCS)
        except Exception as e:
            logger.error(f"Min-variance solver failed: {e}")
            return None, {"feasible": False, "solver_status": str(e)}

    if problem.status not in ("optimal", "optimal_inaccurate") or w.value is None:
        return None, {"feasible": False, "solver_status": problem.status}

    weights_arr = np.clip(w.value, 0, 1)
    weights_arr /= weights_arr.sum()

    metrics = portfolio_metrics_from_weights(weights_arr, tickers, returns, rf)
    metrics["solver_status"] = problem.status
    logger.info(f"Min-variance portfolio: Vol={metrics['vol_pct']:.2f}%, "
                f"Return={metrics['return_pct']:.2f}%")
    return weights_arr, metrics


# ---------------------------------------------------------------------------
# 5. Target Return Portfolio
# ---------------------------------------------------------------------------

def target_return_portfolio(
    returns: pd.DataFrame,
    target_return_pct: float,
    max_weight: Optional[float] = None,
) -> tuple[np.ndarray | None, dict]:
    """
    Find minimum-variance portfolio achieving a user-specified return.

    This is the per-point QP used to build the frontier — exposed publicly
    so Streamlit can call it when the user selects "Target Return" mode.

    Parameters
    ----------
    returns            : pd.DataFrame — daily returns
    target_return_pct  : float — desired annualized return as % (e.g. 15.0 = 15%)
    max_weight         : float — concentration cap (default from config)

    Returns
    -------
    weights : np.ndarray or None (None if infeasible — target too high)
    metrics : dict

    Unit Test Case:
        Input:  target_return_pct = impossible high value (e.g. 200%)
        Expect: weights == None, metrics["feasible"] == False

        Input:  target_return_pct within achievable range
        Expect: metrics["return_pct"] ≈ target_return_pct (within solver tolerance)
    """
    target_return = target_return_pct / 100.0
    return solve_target_return(returns, target_return, max_weight)


# ---------------------------------------------------------------------------
# 6. Random Portfolio Scatter — VISUAL REFERENCE ONLY
# ---------------------------------------------------------------------------

def random_portfolio_scatter(
    returns: pd.DataFrame,
    n_portfolios: int = 3000,
    max_weight: Optional[float] = None,
) -> pd.DataFrame:
    """
    Generate random feasible portfolios for background scatter visualization.

    ⚠️  SOP Rule: This output is VISUAL REFERENCE ONLY.
        It must NEVER replace the CVXPY efficient frontier.
        It is rendered BEHIND the frontier in the Efficient Frontier tab
        to give the user a sense of the feasible space.

    Parameters
    ----------
    returns      : pd.DataFrame — daily returns
    n_portfolios : int          — number of random portfolios (default: 3000)
    max_weight   : float        — weight cap applied to random samples too

    Returns
    -------
    pd.DataFrame with columns: return_pct, vol_pct, sharpe

    Unit Test Case:
        Input:  n_portfolios=100
        Expect: len(scatter) == 100
                scatter["vol_pct"].min() > 0
                scatter["return_pct"] varies (not all same)
    """
    cfg = _load_config()
    mw = max_weight if max_weight is not None else cfg.get("max_weight_per_stock", 0.40)
    rf = cfg.get("risk_free_rate", 0.065)

    mu, sigma, tickers = _prepare_inputs(returns)
    n = len(tickers)
    records = []
    np.random.seed(123)

    for _ in range(n_portfolios):
        # Dirichlet-sampled weights — automatically sum to 1
        w = np.random.dirichlet(np.ones(n))
        # Clip to max_weight and re-normalise
        w = np.clip(w, 0, mw)
        if w.sum() == 0:
            continue
        w /= w.sum()

        port_ret = float(mu @ w)
        port_vol = float(np.sqrt(w @ sigma @ w))
        sharpe = (port_ret - rf) / port_vol if port_vol > 0 else 0.0

        records.append({
            "return_pct": round(port_ret * 100, 4),
            "vol_pct": round(port_vol * 100, 4),
            "sharpe": round(sharpe, 4),
        })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Inline Unit Tests
# Run: python src/optimization.py
# ---------------------------------------------------------------------------

def _run_tests():
    import numpy as np
    np.random.seed(42)
    print("\n--- Running optimization unit tests ---")

    # Build synthetic return series for 4 stocks
    dates = pd.bdate_range("2018-01-01", periods=1000)
    mus_daily = [0.0006, 0.0004, 0.0003, 0.0005]
    vols_daily = [0.015,  0.012,  0.010,  0.013]
    corr = np.array([
        [1.00, 0.60, 0.55, 0.50],
        [0.60, 1.00, 0.65, 0.45],
        [0.55, 0.65, 1.00, 0.40],
        [0.50, 0.45, 0.40, 1.00],
    ])
    cov = np.outer(vols_daily, vols_daily) * corr
    L = np.linalg.cholesky(cov)
    raw = (L @ np.random.randn(4, 1000)).T
    raw += np.array(mus_daily)
    tickers = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS"]
    returns = pd.DataFrame(raw, index=dates, columns=tickers)

    # 1. Min-variance portfolio
    w_mv, m_mv = min_variance_portfolio(returns)
    assert w_mv is not None,            "Min-variance should solve"
    assert abs(w_mv.sum() - 1.0) < 1e-4, f"Weights sum={w_mv.sum():.6f}, expected 1.0"
    assert all(w_mv >= -1e-5),           "All weights should be non-negative"
    print(f"✅ Min-variance: Vol={m_mv['vol_pct']:.2f}% Return={m_mv['return_pct']:.2f}%")

    # 2. Max-Sharpe portfolio
    w_ms, m_ms = max_sharpe_portfolio(returns)
    assert w_ms is not None,             "Max-Sharpe should solve"
    assert abs(w_ms.sum() - 1.0) < 1e-4, f"Weights sum={w_ms.sum():.6f}, expected 1.0"
    assert m_ms["sharpe"] >= m_mv["sharpe"] - 0.01, \
        f"Max-Sharpe ({m_ms['sharpe']:.4f}) should be >= Min-Var ({m_mv['sharpe']:.4f})"
    print(f"✅ Max-Sharpe: Sharpe={m_ms['sharpe']:.4f} Return={m_ms['return_pct']:.2f}%")

    # 3. Efficient frontier
    frontier = compute_efficient_frontier(returns, n_points=30)
    assert len(frontier) > 5,            "Frontier should have meaningful points"
    assert "return_pct" in frontier.columns
    assert "vol_pct" in frontier.columns
    assert "sharpe" in frontier.columns
    print(f"✅ Efficient frontier: {len(frontier)} points computed")

    # 4. Constraint enforcement — no weight exceeds max_weight
    cfg = _load_config()
    mw = cfg.get("max_weight_per_stock", 0.40)
    for _, row in frontier.iterrows():
        weights_dict = row["weights_dict"]
        for t, w in weights_dict.items():
            assert w <= mw + 1e-4, f"Weight {w:.4f} exceeds max {mw} for {t}"
    print(f"✅ Weight constraint: all weights <= {mw}")

    # 5. Target return — feasible
    mid_return = frontier["return_pct"].median()
    w_tr, m_tr = target_return_portfolio(returns, mid_return)
    assert w_tr is not None,            f"Target return {mid_return:.2f}% should be feasible"
    assert abs(m_tr["return_pct"] - mid_return) < 1.0, \
        f"Solved return {m_tr['return_pct']:.2f}% too far from target {mid_return:.2f}%"
    print(f"✅ Target return portfolio: {m_tr['return_pct']:.2f}% (target {mid_return:.2f}%)")

    # 6. Random scatter — visual only, not frontier
    scatter = random_portfolio_scatter(returns, n_portfolios=200)
    assert len(scatter) == 200,         "Scatter should have exactly n_portfolios rows"
    assert scatter["vol_pct"].min() > 0, "All scatter vols must be positive"
    print(f"✅ Random scatter: {len(scatter)} portfolios [visual reference only]")

    # 7. Infeasible target return
    w_inf, m_inf = target_return_portfolio(returns, target_return_pct=999.0)
    assert w_inf is None,               "Infeasible target should return None weights"
    assert m_inf["feasible"] is False,  "Infeasible target should set feasible=False"
    print(f"✅ Infeasible target correctly rejected: {m_inf['solver_status']}")

    print("--- All optimization tests passed ---\n")


if __name__ == "__main__":
    _run_tests()
