"""
src/stress_test.py
==================
Historical stress scenario analysis for the Stock Assessment dashboard.

SOP Rules Enforced (CRITICAL):
    - ALL scenario date ranges loaded from data/stress_periods.json
    - NEVER hardcode scenario dates anywhere in this file
    - Correlation delta is the KEY output — always compute and return it
    - Compare optimized portfolio vs equal-weight portfolio per scenario

For each scenario the module computes:
    1. Portfolio drawdown (%) during the stress window
    2. Recovery days (days from trough to return to pre-stress level)
    3. Normal period correlation matrix (mirror-length window before stress)
    4. Stress period correlation matrix (during the scenario window)
    5. Correlation delta = stress_corr − normal_corr  ← KEY OUTPUT
    6. Optimized vs equal-weight portfolio performance comparison

Public API:
    load_stress_scenarios()             -> dict
    run_stress_analysis(portfolio_weights, prices, scenario) -> dict
    run_all_scenarios(portfolio_weights, prices)             -> dict[str, dict]
    get_equal_weight_portfolio(tickers)                      -> dict[str, float]
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"
_SCENARIOS_PATH = Path(__file__).parent.parent / "data" / "stress_periods.json"


def _load_config() -> dict:
    if not _CONFIG_PATH.exists():
        raise FileNotFoundError(f"config.yaml not found at {_CONFIG_PATH}")
    with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# 1. Load scenario definitions — always from JSON, never hardcoded
# ---------------------------------------------------------------------------

def load_stress_scenarios() -> dict:
    """
    Load all stress scenario definitions from data/stress_periods.json.

    SOP Rule: ALL date ranges come from this file.
    Never hardcode any scenario dates in Python source code.

    Returns
    -------
    dict keyed by scenario_id, each containing:
        {start, end, label, description, key_risk}

    Unit Test Case:
        Expect: "covid_2020" in scenarios
                scenarios["covid_2020"]["start"] == "2020-01-20"
                scenarios["covid_2020"]["end"]   == "2020-03-23"
    """
    if not _SCENARIOS_PATH.exists():
        raise FileNotFoundError(
            f"stress_periods.json not found at {_SCENARIOS_PATH}. "
            "Ensure data/ directory contains this file."
        )
    with open(_SCENARIOS_PATH, "r") as f:
        scenarios = json.load(f)

    # Validate each scenario has required fields
    required = {"start", "end"}
    for sid, sdata in scenarios.items():
        missing = required - set(sdata.keys())
        if missing:
            raise ValueError(
                f"Scenario '{sid}' missing required fields: {missing}. "
                f"Check data/stress_periods.json."
            )
    return scenarios


# ---------------------------------------------------------------------------
# 2. Portfolio return series helpers
# ---------------------------------------------------------------------------

def _portfolio_returns(
    prices: pd.DataFrame,
    weights: dict[str, float],
) -> pd.Series:
    """
    Compute weighted daily portfolio returns from adjusted price DataFrame.

    Returns
    -------
    pd.Series — daily portfolio returns indexed by date
    """
    tickers = [t for t in weights if t in prices.columns]
    if not tickers:
        raise ValueError("No weight tickers found in prices DataFrame.")

    total_w = sum(weights[t] for t in tickers)
    if abs(total_w - 1.0) > 0.01:
        logger.warning(f"Weights sum to {total_w:.4f} — normalising for stress analysis.")
    normalised = {t: weights[t] / total_w for t in tickers}

    w_arr = np.array([normalised[t] for t in tickers])
    ret = prices[tickers].pct_change().dropna()
    port_ret = ret.values @ w_arr
    return pd.Series(port_ret, index=ret.index, name="portfolio")


def get_equal_weight_portfolio(tickers: list[str]) -> dict[str, float]:
    """
    Build an equal-weight portfolio dict for comparison against optimised weights.

    Parameters
    ----------
    tickers : list[str] — must all have .NS suffix

    Returns
    -------
    dict[str, float] — {ticker: 1/n} for each ticker

    Unit Test Case:
        Input:  tickers = ["A.NS", "B.NS", "C.NS"]
        Expect: all values == 1/3, sum == 1.0
    """
    n = len(tickers)
    if n == 0:
        raise ValueError("Ticker list is empty.")
    return {t: round(1.0 / n, 10) for t in tickers}


# ---------------------------------------------------------------------------
# 3. Drawdown and recovery computation
# ---------------------------------------------------------------------------

def _compute_drawdown_and_recovery(
    returns: pd.Series,
    pre_crash_prices: pd.Series,
    post_crash_prices: pd.Series,
) -> dict:
    """
    Compute max drawdown during stress window and recovery time after it.

    Parameters
    ----------
    returns          : pd.Series — daily returns during the stress window
    pre_crash_prices : pd.Series — prices in the pre-stress normal window
    post_crash_prices: pd.Series — prices after the stress window (for recovery)

    Returns
    -------
    dict:
        max_drawdown_pct    : float  — worst peak-to-trough during stress (%)
        stress_start_value  : float  — normalised portfolio value at stress start (1.0)
        stress_end_value    : float  — normalised portfolio value at stress end
        total_return_pct    : float  — total return during stress window (%)
        recovery_days       : int    — trading days after stress until recovery (None if not recovered)
        recovered           : bool

    Unit Test Case:
        Input:  returns = [-0.03, -0.05, -0.02, -0.01] (all down days)
        Expect: max_drawdown_pct < 0
                stress_end_value < 1.0
    """
    if returns.empty:
        return {
            "max_drawdown_pct": 0.0, "stress_start_value": 1.0,
            "stress_end_value": 1.0, "total_return_pct": 0.0,
            "recovery_days": None, "recovered": False,
        }

    # Build cumulative return path starting at 1.0
    cum = (1 + returns).cumprod()
    cum = pd.concat([pd.Series([1.0], index=[returns.index[0]]), cum])

    rolling_max = cum.cummax()
    drawdown_series = (cum - rolling_max) / rolling_max
    max_dd = float(drawdown_series.min())
    stress_end_val = float(cum.iloc[-1])
    total_ret = (stress_end_val - 1.0) * 100

    # Recovery: find first post-stress day where cumulative return recovers to 1.0
    recovery_days = None
    recovered = False
    if not post_crash_prices.empty and stress_end_val < 1.0:
        entry_price = pre_crash_prices.iloc[-1] if not pre_crash_prices.empty else None
        if entry_price is not None and entry_price > 0:
            post_ret = (post_crash_prices / entry_price)
            recovered_days = post_ret[post_ret >= 1.0]
            if not recovered_days.empty:
                first_recovery = recovered_days.index[0]
                stress_end_date = returns.index[-1]
                recovery_days = int(
                    len(post_crash_prices.loc[:first_recovery]) - 1
                )
                recovered = True

    return {
        "max_drawdown_pct": round(max_dd * 100, 2),
        "stress_start_value": 1.0,
        "stress_end_value": round(stress_end_val, 6),
        "total_return_pct": round(total_ret, 2),
        "recovery_days": recovery_days,
        "recovered": recovered,
    }


# ---------------------------------------------------------------------------
# 4. Correlation matrices (normal + stress + delta)
# ---------------------------------------------------------------------------

def _compute_correlation_matrices(
    prices: pd.DataFrame,
    tickers: list[str],
    stress_start: str,
    stress_end: str,
) -> dict:
    """
    Compute correlation matrices for the normal and stress periods.
    Return the correlation delta — the KEY output (SOP Section 4 & 6).

    Normal window: same number of trading days as the stress window,
    immediately preceding the stress period start date.

    Parameters
    ----------
    prices       : pd.DataFrame — full adjusted price history
    tickers      : list[str]    — tickers to include
    stress_start : str          — "YYYY-MM-DD"
    stress_end   : str          — "YYYY-MM-DD"

    Returns
    -------
    dict:
        normal_corr  : pd.DataFrame — correlation matrix in normal period
        stress_corr  : pd.DataFrame — correlation matrix in stress period
        delta_corr   : pd.DataFrame — stress_corr − normal_corr  ← KEY OUTPUT
        avg_normal_pairwise  : float — average off-diagonal correlation (normal)
        avg_stress_pairwise  : float — average off-diagonal correlation (stress)
        avg_delta_pairwise   : float — average change in pairwise correlation
        interpretation       : str   — plain-English insight

    Unit Test Case:
        Input:  perfectly uncorrelated stocks in normal window,
                perfectly correlated during stress window
        Expect: avg_normal_pairwise ≈ 0.0
                avg_stress_pairwise ≈ 1.0
                avg_delta_pairwise  ≈ 1.0
                interpretation contains "collapsed"
    """
    avail = [t for t in tickers if t in prices.columns]
    if len(avail) < 2:
        empty_corr = pd.DataFrame()
        return {
            "normal_corr": empty_corr, "stress_corr": empty_corr,
            "delta_corr": empty_corr, "avg_normal_pairwise": None,
            "avg_stress_pairwise": None, "avg_delta_pairwise": None,
            "interpretation": "Insufficient tickers for correlation analysis.",
        }

    stress_ret = prices[avail].pct_change().dropna()
    stress_window = stress_ret.loc[stress_start:stress_end]
    n_stress_days = len(stress_window)

    # Normal window: equal-length window immediately before stress start
    pre_stress = stress_ret.loc[:stress_start].iloc[:-1]  # exclude stress_start itself
    normal_window = pre_stress.iloc[-n_stress_days:] if len(pre_stress) >= n_stress_days \
        else pre_stress

    if len(normal_window) < 5:
        logger.warning(
            f"Normal window has only {len(normal_window)} days "
            f"(stress window: {n_stress_days} days). Results may be unreliable."
        )

    normal_corr = normal_window.corr() if len(normal_window) >= 2 else pd.DataFrame()
    stress_corr = stress_window.corr() if len(stress_window) >= 2 else pd.DataFrame()

    if normal_corr.empty or stress_corr.empty:
        return {
            "normal_corr": normal_corr, "stress_corr": stress_corr,
            "delta_corr": pd.DataFrame(), "avg_normal_pairwise": None,
            "avg_stress_pairwise": None, "avg_delta_pairwise": None,
            "interpretation": "Insufficient data for correlation comparison.",
        }

    # Delta correlation: stress − normal (KEY OUTPUT)
    shared_tickers = [t for t in avail if t in normal_corr.columns and t in stress_corr.columns]
    delta_corr = stress_corr.loc[shared_tickers, shared_tickers] - \
                 normal_corr.loc[shared_tickers, shared_tickers]

    def _avg_offdiag(corr_df: pd.DataFrame) -> float:
        """Average of upper-triangle off-diagonal elements."""
        arr = corr_df.values
        n = len(arr)
        if n < 2:
            return 0.0
        upper = arr[np.triu_indices(n, k=1)]
        return float(np.nanmean(upper))

    avg_normal = _avg_offdiag(normal_corr.loc[shared_tickers, shared_tickers])
    avg_stress = _avg_offdiag(stress_corr.loc[shared_tickers, shared_tickers])
    avg_delta  = avg_stress - avg_normal

    # Generate insight
    if avg_delta > 0.2:
        direction = "rose sharply"
        diversif_note = "Diversification largely collapsed during this crisis."
    elif avg_delta > 0.05:
        direction = "increased moderately"
        diversif_note = "Diversification was partially reduced during stress."
    elif avg_delta < -0.05:
        direction = "fell"
        diversif_note = "Diversification benefit actually improved during this period."
    else:
        direction = "remained stable"
        diversif_note = "Diversification benefit held up during this period."

    interpretation = (
        f"Average pairwise correlation {direction} from {avg_normal:.2f} (normal) "
        f"to {avg_stress:.2f} (stress), a Δ of {avg_delta:+.2f}. {diversif_note}"
    )

    return {
        "normal_corr": normal_corr,
        "stress_corr": stress_corr,
        "delta_corr": delta_corr,
        "normal_window_days": len(normal_window),
        "stress_window_days": n_stress_days,
        "avg_normal_pairwise": round(avg_normal, 4),
        "avg_stress_pairwise": round(avg_stress, 4),
        "avg_delta_pairwise": round(avg_delta, 4),
        "interpretation": interpretation,
    }


# ---------------------------------------------------------------------------
# 5. Core: run_stress_analysis — SOP-specified function signature
# ---------------------------------------------------------------------------

def run_stress_analysis(
    portfolio_weights: dict[str, float],
    prices: pd.DataFrame,
    scenario: str,
) -> dict:
    """
    Evaluate portfolio behaviour during a historical market stress scenario.

    SOP Rule: Scenario dates always loaded from data/stress_periods.json.
    Never call this with hardcoded date strings — use the scenario key.

    Parameters
    ----------
    portfolio_weights : dict[str, float]
        Optimised portfolio weights {ticker: weight}, e.g. from optimization.py.
    prices : pd.DataFrame
        Full adjusted price history — must cover the scenario period.
        Columns = NSE tickers with .NS suffix.
    scenario : str
        Key from data/stress_periods.json, e.g. "covid_2020", "gfc_2008".

    Returns
    -------
    dict:
    {
        "scenario_id"      : str,
        "scenario_label"   : str,
        "scenario_description": str,
        "stress_start"     : str,
        "stress_end"       : str,
        "tickers_analysed" : list[str],

        "optimized": {
            "max_drawdown_pct"  : float,
            "total_return_pct"  : float,
            "stress_end_value"  : float,
            "recovery_days"     : int | None,
            "recovered"         : bool,
        },
        "equal_weight": {
            "max_drawdown_pct"  : float,
            "total_return_pct"  : float,
            "stress_end_value"  : float,
            "recovery_days"     : int | None,
            "recovered"         : bool,
        },
        "comparison": {
            "drawdown_improvement_pct" : float,  # positive = optimized did better
            "return_improvement_pct"   : float,
            "insight"                  : str,
        },

        "correlation": {
            "normal_corr"          : pd.DataFrame,
            "stress_corr"          : pd.DataFrame,
            "delta_corr"           : pd.DataFrame,  ← KEY OUTPUT
            "avg_normal_pairwise"  : float,
            "avg_stress_pairwise"  : float,
            "avg_delta_pairwise"   : float,
            "interpretation"       : str,
        },

        "data_available"   : bool,
        "warnings"         : list[str],
    }

    Raises
    ------
    KeyError    — if scenario key not found in stress_periods.json
    ValueError  — if prices DataFrame doesn't cover the scenario period

    Example
    -------
    >>> from src.stress_test import run_stress_analysis
    >>> results = run_stress_analysis(
    ...     portfolio_weights={"RELIANCE.NS": 0.4, "TCS.NS": 0.3, "HDFCBANK.NS": 0.3},
    ...     prices=prices_df,
    ...     scenario="covid_2020"
    ... )
    >>> print(results["correlation"]["interpretation"])
    >>> print(results["comparison"]["insight"])
    """
    scenarios = load_stress_scenarios()            # Always from JSON — never hardcoded
    warnings_list: list[str] = []

    if scenario not in scenarios:
        available = list(scenarios.keys())
        raise KeyError(
            f"Scenario '{scenario}' not found in stress_periods.json. "
            f"Available: {available}"
        )

    sdata = scenarios[scenario]
    stress_start = sdata["start"]
    stress_end   = sdata["end"]
    label        = sdata.get("label", scenario)
    description  = sdata.get("description", "")

    tickers = [t for t in portfolio_weights if t in prices.columns]
    if not tickers:
        raise ValueError("None of the portfolio tickers found in the prices DataFrame.")

    missing_tickers = [t for t in portfolio_weights if t not in prices.columns]
    if missing_tickers:
        warnings_list.append(
            f"Tickers not in price data (excluded from stress analysis): {missing_tickers}"
        )

    # --- Check price data covers the scenario window ---
    price_start = str(prices.index[0])[:10]
    price_end   = str(prices.index[-1])[:10]
    data_available = (price_start <= stress_start) and (price_end >= stress_end)

    if not data_available:
        warnings_list.append(
            f"Price data ({price_start} to {price_end}) does not fully cover "
            f"scenario window ({stress_start} to {stress_end}). "
            "Extend data_start in config.yaml to include this period."
        )
        return {
            "scenario_id": scenario, "scenario_label": label,
            "scenario_description": description,
            "stress_start": stress_start, "stress_end": stress_end,
            "tickers_analysed": tickers, "data_available": False,
            "warnings": warnings_list,
            "optimized": None, "equal_weight": None,
            "comparison": None, "correlation": None,
        }

    # --- Build return series ---
    all_returns = prices[tickers].pct_change().dropna()
    stress_returns = all_returns.loc[stress_start:stress_end]

    if stress_returns.empty:
        warnings_list.append(
            f"No trading days found between {stress_start} and {stress_end}. "
            "Check dates in stress_periods.json."
        )
        return {
            "scenario_id": scenario, "scenario_label": label,
            "scenario_description": description,
            "stress_start": stress_start, "stress_end": stress_end,
            "tickers_analysed": tickers, "data_available": False,
            "warnings": warnings_list,
            "optimized": None, "equal_weight": None,
            "comparison": None, "correlation": None,
        }

    pre_stress_prices  = prices[tickers].loc[:stress_start].iloc[:-1]
    post_stress_prices = prices[tickers].loc[stress_end:]

    # Normalised price series (start = 1.0) for recovery detection
    entry_price = pre_stress_prices.iloc[-1] if not pre_stress_prices.empty \
        else prices[tickers].iloc[0]

    def _normed_post(w: dict[str, float]) -> pd.Series:
        """Build normalised equal-weight of tickers post-stress."""
        ew_tickers = [t for t in w if t in post_stress_prices.columns]
        if not ew_tickers:
            return pd.Series(dtype=float)
        w_arr = np.array([w[t] for t in ew_tickers])
        w_arr /= w_arr.sum()
        post_ret = post_stress_prices[ew_tickers].pct_change().dropna()
        port_post = post_ret.values @ w_arr
        entry = entry_price[ew_tickers].values @ w_arr
        stress_end_price_port = (
            prices[ew_tickers].loc[:stress_end].iloc[-1].values @ w_arr
        )
        # Normalise to pre-crash entry price
        cumpost = stress_end_price_port * (1 + port_post).cumprod()
        return pd.Series(cumpost / entry, index=post_ret.index)

    # --- Optimised portfolio stress metrics ---
    opt_w_norm = {t: portfolio_weights[t] for t in tickers}
    total_opt = sum(opt_w_norm.values())
    opt_w_norm = {t: v / total_opt for t, v in opt_w_norm.items()}

    opt_stress_ret = _portfolio_returns(prices[tickers].loc[stress_start:stress_end], opt_w_norm)
    opt_post_normed = _normed_post(opt_w_norm)

    opt_metrics = _compute_drawdown_and_recovery(
        returns=opt_stress_ret,
        pre_crash_prices=pre_stress_prices[[tickers[0]]].squeeze(),
        post_crash_prices=opt_post_normed,
    )

    # --- Equal-weight portfolio stress metrics ---
    ew_weights = get_equal_weight_portfolio(tickers)
    ew_stress_ret = _portfolio_returns(prices[tickers].loc[stress_start:stress_end], ew_weights)
    ew_post_normed = _normed_post(ew_weights)

    ew_metrics = _compute_drawdown_and_recovery(
        returns=ew_stress_ret,
        pre_crash_prices=pre_stress_prices[[tickers[0]]].squeeze(),
        post_crash_prices=ew_post_normed,
    )

    # --- Comparison ---
    dd_improvement = opt_metrics["max_drawdown_pct"] - ew_metrics["max_drawdown_pct"]
    ret_improvement = opt_metrics["total_return_pct"] - ew_metrics["total_return_pct"]

    if dd_improvement > 2:
        comp_insight = (
            f"Optimised portfolio reduced maximum drawdown by {abs(dd_improvement):.1f}pp "
            f"vs equal-weight during {label} "
            f"({opt_metrics['max_drawdown_pct']:.1f}% vs {ew_metrics['max_drawdown_pct']:.1f}%)."
        )
    elif dd_improvement < -2:
        comp_insight = (
            f"Equal-weight portfolio outperformed optimised by {abs(dd_improvement):.1f}pp "
            f"during {label} — optimised portfolio was more concentrated in affected stocks."
        )
    else:
        comp_insight = (
            f"Optimised and equal-weight portfolios had similar drawdowns during {label} "
            f"({opt_metrics['max_drawdown_pct']:.1f}% vs {ew_metrics['max_drawdown_pct']:.1f}%)."
        )

    # --- Correlation matrices and delta ---
    corr_result = _compute_correlation_matrices(
        prices=prices,
        tickers=tickers,
        stress_start=stress_start,
        stress_end=stress_end,
    )

    return {
        "scenario_id": scenario,
        "scenario_label": label,
        "scenario_description": description,
        "stress_start": stress_start,
        "stress_end": stress_end,
        "tickers_analysed": tickers,
        "data_available": True,
        "warnings": warnings_list,

        "optimized": opt_metrics,
        "equal_weight": ew_metrics,

        "comparison": {
            "drawdown_improvement_pct": round(dd_improvement, 2),
            "return_improvement_pct": round(ret_improvement, 2),
            "insight": comp_insight,
        },

        "correlation": corr_result,
    }


# ---------------------------------------------------------------------------
# 6. Run all scenarios at once
# ---------------------------------------------------------------------------

def run_all_scenarios(
    portfolio_weights: dict[str, float],
    prices: pd.DataFrame,
) -> dict[str, dict]:
    """
    Run stress analysis for every scenario defined in stress_periods.json.

    Parameters
    ----------
    portfolio_weights : dict[str, float] — optimised weights
    prices            : pd.DataFrame     — full price history

    Returns
    -------
    dict[scenario_id, stress_result_dict]
        Scenario results that had insufficient data will have data_available=False.

    Example
    -------
    >>> all_results = run_all_scenarios(weights, prices_df)
    >>> for sid, res in all_results.items():
    ...     if res["data_available"]:
    ...         print(sid, res["correlation"]["interpretation"])
    """
    scenarios = load_stress_scenarios()
    results = {}
    for sid in scenarios:
        try:
            results[sid] = run_stress_analysis(portfolio_weights, prices, sid)
            status = "✅ data available" if results[sid]["data_available"] else "⚠️ no data"
            logger.info(f"Stress scenario '{sid}': {status}")
        except Exception as e:
            logger.error(f"Stress analysis failed for scenario '{sid}': {e}")
            results[sid] = {
                "scenario_id": sid,
                "scenario_label": scenarios[sid].get("label", sid),
                "data_available": False,
                "warnings": [str(e)],
                "optimized": None, "equal_weight": None,
                "comparison": None, "correlation": None,
            }
    return results


# ---------------------------------------------------------------------------
# Inline Unit Tests
# Run: python src/stress_test.py
# ---------------------------------------------------------------------------

def _run_tests():
    import numpy as np
    print("\n--- Running stress_test unit tests ---")

    # 1. Scenario loading — always from JSON, never hardcoded
    scenarios = load_stress_scenarios()
    assert "covid_2020" in scenarios,   "covid_2020 must be in stress_periods.json"
    assert "start" in scenarios["covid_2020"]
    assert "end"   in scenarios["covid_2020"]
    assert scenarios["covid_2020"]["start"] == "2020-01-20"
    assert scenarios["covid_2020"]["end"]   == "2020-03-23"
    print(f"✅ Loaded {len(scenarios)} scenarios from stress_periods.json")

    # 2. Equal-weight portfolio
    tickers = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS"]
    ew = get_equal_weight_portfolio(tickers)
    assert abs(sum(ew.values()) - 1.0) < 1e-9
    assert all(abs(v - 1/3) < 1e-9 for v in ew.values())
    print("✅ Equal-weight portfolio: all weights = 1/3")

    # 3. Correlation delta with synthetic data
    dates = pd.bdate_range("2019-01-01", periods=300)
    np.random.seed(42)

    # Normal period: low correlation
    normal_raw = np.random.randn(300, 3) * 0.01 + 0.0004
    # Stress period (last 40 days): high correlation via shared crash factor
    crash_factor = np.random.randn(40) * 0.025
    stress_raw = normal_raw[-40:].copy()
    stress_raw[:, 0] += crash_factor
    stress_raw[:, 1] += crash_factor * 0.9
    stress_raw[:, 2] += crash_factor * 0.85
    normal_raw[-40:] = stress_raw

    prices_sim = pd.DataFrame(
        np.cumprod(1 + normal_raw, axis=0) * 1000,
        index=dates,
        columns=["A.NS", "B.NS", "C.NS"],
    )

    stress_start_sim = str(dates[-40])[:10]
    stress_end_sim   = str(dates[-1])[:10]

    corr_result = _compute_correlation_matrices(
        prices=prices_sim,
        tickers=["A.NS", "B.NS", "C.NS"],
        stress_start=stress_start_sim,
        stress_end=stress_end_sim,
    )

    assert not corr_result["delta_corr"].empty, "Delta correlation should be computed"
    avg_delta = corr_result["avg_delta_pairwise"]
    assert avg_delta is not None
    print(f"✅ Correlation delta: normal={corr_result['avg_normal_pairwise']:.3f} "
          f"→ stress={corr_result['avg_stress_pairwise']:.3f} | Δ={avg_delta:+.3f}")

    # 4. Drawdown with all-down returns
    down_returns = pd.Series(
        [-0.03, -0.05, -0.02, -0.04, -0.01],
        index=pd.bdate_range("2020-03-09", periods=5),
    )
    dd = _compute_drawdown_and_recovery(
        returns=down_returns,
        pre_crash_prices=pd.Series(dtype=float),
        post_crash_prices=pd.Series(dtype=float),
    )
    assert dd["max_drawdown_pct"] < 0,    "Drawdown must be negative"
    assert dd["stress_end_value"] < 1.0,  "End value must be below start"
    print(f"✅ Drawdown: {dd['max_drawdown_pct']:.2f}% | End value: {dd['stress_end_value']:.4f}")

    # 5. Missing scenario key raises KeyError
    try:
        weights_test = {"A.NS": 0.5, "B.NS": 0.3, "C.NS": 0.2}
        run_stress_analysis(weights_test, prices_sim, "nonexistent_scenario_xyz")
        assert False, "Should have raised KeyError"
    except KeyError as e:
        print(f"✅ Missing scenario key correctly raises KeyError: {e}")

    # 6. Data unavailable returns data_available=False
    small_prices = prices_sim.iloc[:10]   # only 10 rows — won't cover covid_2020
    weights_test = {"A.NS": 0.4, "B.NS": 0.3, "C.NS": 0.3}
    result = run_stress_analysis(weights_test, small_prices, "covid_2020")
    assert result["data_available"] is False
    print("✅ Insufficient data: data_available=False correctly returned")

    print("--- All stress_test tests passed ---\n")


if __name__ == "__main__":
    _run_tests()
