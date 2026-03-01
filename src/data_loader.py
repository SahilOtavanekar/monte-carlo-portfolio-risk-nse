"""
src/data_loader.py
==================
Data fetching and quality validation for the Stock Assessment dashboard.

SOP Rules Enforced:
    - auto_adjust=True on ALL yfinance calls (never omit)
    - Forward-fill gaps ONLY within config.max_forward_fill_days (default: 2)
    - Flag stocks with >config.max_missing_data_pct missing data (default: 2%)
    - Flag stocks with same price repeating >config.stale_price_threshold consecutive days (default: 3)
    - All parameters read from config.yaml — NOTHING hardcoded here

Public API:
    load_portfolio_data(tickers, start, end) -> (prices_df, quality_report)
    load_benchmark_data(start, end)          -> pd.Series
    get_ticker_metadata()                    -> pd.DataFrame

Unit Test Cases (inline — run manually or with pytest):
    See bottom of file for test stubs with expected behavior.
"""

from __future__ import annotations

import logging
import warnings
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml
import yfinance as yf

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Config loader — reads config.yaml from project root (one level above src/)
# ---------------------------------------------------------------------------
_CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"


def _load_config() -> dict:
    """Load and return the global config.yaml. Cached after first call."""
    if not _CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"config.yaml not found at {_CONFIG_PATH}. "
            "Ensure it exists in the project root."
        )
    with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Ticker metadata helper
# ---------------------------------------------------------------------------

def get_ticker_metadata() -> pd.DataFrame:
    """
    Load tickers.csv and return as a DataFrame.

    Returns
    -------
    pd.DataFrame
        Columns: ticker, company_name, sector
    """
    tickers_path = Path(__file__).parent.parent / "data" / "tickers.csv"
    if not tickers_path.exists():
        raise FileNotFoundError(f"tickers.csv not found at {tickers_path}")
    df = pd.read_csv(tickers_path)
    required_cols = {"ticker", "company_name", "sector"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"tickers.csv must have columns: {required_cols}")
    return df


# ---------------------------------------------------------------------------
# Core fetch helper
# ---------------------------------------------------------------------------

def _fetch_raw_prices(
    tickers: list[str],
    start: str,
    end: str,
    cfg: dict,
) -> pd.DataFrame:
    """
    Fetch adjusted closing prices from Yahoo Finance.

    SOP Rule: auto_adjust=True ALWAYS. This corrects for:
        - Stock splits
        - Bonus issues
        - Rights issues
    Omitting this causes incorrect return calculations on Indian stocks.

    Parameters
    ----------
    tickers : list of NSE tickers with .NS suffix (e.g. "RELIANCE.NS")
    start   : ISO date string "YYYY-MM-DD"
    end     : ISO date string "YYYY-MM-DD"
    cfg     : loaded config dict

    Returns
    -------
    pd.DataFrame  — raw Close prices indexed by trading date
    """
    if not tickers:
        raise ValueError("Ticker list cannot be empty.")

    # Validate .NS suffix — SOP requirement
    invalid = [t for t in tickers if not t.endswith(".NS")]
    if invalid:
        raise ValueError(
            f"All NSE tickers must use .NS suffix. Invalid tickers: {invalid}"
        )

    logger.info(f"Fetching {len(tickers)} tickers from {start} to {end} ...")

    raw = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        auto_adjust=True,       # SOP: NEVER change this to False
        progress=False,
        threads=False,
    )

    if raw.empty:
        raise ValueError(
            f"yfinance returned no data for tickers {tickers} "
            f"between {start} and {end}. Check date range and ticker validity."
        )

    # Extract Close prices — after auto_adjust=True these are adjusted closes
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        # Single ticker — yfinance returns flat DataFrame
        prices = raw[["Close"]].rename(columns={"Close": tickers[0]})

    # Keep only requested tickers (yfinance may silently drop invalid ones)
    prices = prices[[t for t in tickers if t in prices.columns]]

    # Identify any tickers yfinance silently dropped (likely delisted)
    dropped = [t for t in tickers if t not in prices.columns]
    if dropped:
        logger.warning(f"Tickers not returned by yfinance (possibly delisted): {dropped}")

    return prices, dropped


# ---------------------------------------------------------------------------
# Data Quality Checks
# ---------------------------------------------------------------------------

def _check_missing_data(
    prices: pd.DataFrame,
    cfg: dict,
) -> dict[str, dict]:
    """
    Check each stock for missing data percentage.

    SOP Rule: Flag any stock with >max_missing_data_pct (default 2%) missing rows.

    Unit Test Case:
        Input:  100-row DataFrame where HDFCBANK.NS has 5 NaN rows
        Expect: quality["HDFCBANK.NS"]["missing_pct"] == 5.0
                quality["HDFCBANK.NS"]["missing_flag"] == True

        Input:  100-row DataFrame where TCS.NS has 1 NaN row
        Expect: quality["TCS.NS"]["missing_pct"] == 1.0
                quality["TCS.NS"]["missing_flag"] == False  (below 2% threshold)

    Returns
    -------
    dict: {ticker: {"total_rows": int, "missing_rows": int,
                    "missing_pct": float, "missing_flag": bool}}
    """
    threshold = cfg.get("max_missing_data_pct", 0.02)
    report = {}
    total_rows = len(prices)

    for ticker in prices.columns:
        missing_rows = prices[ticker].isna().sum()
        missing_pct = (missing_rows / total_rows) * 100 if total_rows > 0 else 0.0
        report[ticker] = {
            "total_rows": total_rows,
            "missing_rows": int(missing_rows),
            "missing_pct": round(missing_pct, 2),
            "missing_flag": (missing_rows / total_rows) > threshold if total_rows > 0 else False,
        }
    return report


def _check_stale_prices(
    prices: pd.DataFrame,
    cfg: dict,
) -> dict[str, dict]:
    """
    Detect frozen/stale prices — same closing price repeated consecutively.

    SOP Rule: Flag if same price repeats >stale_price_threshold (default 3) days.

    This catches:
        - Yahoo Finance data errors (frozen feed on Indian holidays)
        - Suspended stocks that were not properly excluded
        - Circuit-breaker events where the last traded price is repeated

    Unit Test Case:
        Input:  prices["SBIN.NS"] = [100, 100, 100, 100, 105] (4 consecutive same)
        Expect: quality["SBIN.NS"]["max_consecutive_repeats"] == 4
                quality["SBIN.NS"]["stale_flag"] == True  (>3 threshold)

        Input:  prices["TCS.NS"] = [100, 100, 100, 105, 110] (3 consecutive same)
        Expect: quality["TCS.NS"]["max_consecutive_repeats"] == 3
                quality["TCS.NS"]["stale_flag"] == False  (equal to, not exceeding 3)

    Returns
    -------
    dict: {ticker: {"max_consecutive_repeats": int, "stale_flag": bool,
                    "stale_periods": list[str]}}
    """
    threshold = cfg.get("stale_price_threshold", 3)
    report = {}

    for ticker in prices.columns:
        series = prices[ticker].dropna()
        if series.empty:
            report[ticker] = {
                "max_consecutive_repeats": 0,
                "stale_flag": False,
                "stale_periods": [],
            }
            continue

        # Compute run lengths of consecutive equal values
        diff = series.diff().ne(0)          # True where value changes
        group_ids = diff.cumsum()           # Each run gets a unique group id
        run_lengths = series.groupby(group_ids).transform("count")
        max_run = int(run_lengths.max())

        # Collect the start dates of stale periods
        stale_periods = []
        run_start = None
        prev_val = None
        count = 0
        for idx, val in series.items():
            if val == prev_val:
                count += 1
                if count == threshold + 1:  # Just breached threshold
                    stale_periods.append(str(run_start)[:10])
            else:
                count = 1
                run_start = idx
                prev_val = val

        report[ticker] = {
            "max_consecutive_repeats": max_run,
            "stale_flag": max_run > threshold,
            "stale_periods": stale_periods,
        }
    return report


def _check_forward_fill_gaps(
    prices: pd.DataFrame,
    cfg: dict,
) -> tuple[pd.DataFrame, dict[str, list]]:
    """
    Forward-fill missing data ONLY within allowed window; flag larger gaps.

    SOP Rule: Fill gaps <= max_forward_fill_days (default: 2 trading days).
              Gaps wider than this are flagged — NOT filled.

    Unit Test Case:
        Input:  prices["INFY.NS"] has NaN at index positions [5, 6]  (2-day gap)
        Expect: Gap filled. No flag raised for INFY.NS.

        Input:  prices["WIPRO.NS"] has NaN at index positions [10, 11, 12]  (3-day gap)
        Expect: Gap NOT filled for those 3 rows. Flag raised for WIPRO.NS.
                filled_df["WIPRO.NS"].iloc[10:13].isna().all() == True

    Returns
    -------
    tuple:
        filled_df : pd.DataFrame — prices with short gaps filled
        gap_report: dict {ticker: [list of gap start dates that were too large]}
    """
    max_fill = cfg.get("max_forward_fill_days", 2)
    filled = prices.copy()
    gap_report: dict[str, list] = {t: [] for t in prices.columns}

    for ticker in prices.columns:
        series = prices[ticker].copy()

        # Identify NaN run lengths
        is_nan = series.isna()
        if not is_nan.any():
            continue

        # Label consecutive NaN groups
        nan_groups = (~is_nan).cumsum()
        nan_run_lengths = is_nan.groupby(nan_groups).transform("sum")

        # Positions where gap is too large — keep as NaN
        too_large = is_nan & (nan_run_lengths > max_fill)

        # Forward-fill short gaps only
        short_gap_mask = is_nan & ~too_large
        temp = series.copy()
        temp[too_large] = np.nan        # Preserve large gaps
        filled_series = temp.ffill()    # Fill only reachable NaNs
        # Restore large gaps that ffill may have overwritten
        filled_series[too_large] = np.nan
        filled[ticker] = filled_series

        # Record large gap start dates
        in_gap = False
        gap_start = None
        for idx, flag in too_large.items():
            if flag and not in_gap:
                in_gap = True
                gap_start = str(idx)[:10]
                gap_report[ticker].append(gap_start)
            elif not flag:
                in_gap = False

    return filled, gap_report


def _check_delisted(
    tickers_requested: list[str],
    tickers_returned: list[str],
    dropped: list[str],
) -> dict[str, str]:
    """
    Identify tickers that yfinance silently dropped (likely delisted/invalid).

    Unit Test Case:
        Input:  requested=["RELIANCE.NS", "FAKECO.NS"], returned=["RELIANCE.NS"]
        Expect: delisted_report == {"FAKECO.NS": "Not returned by yfinance — possibly delisted or invalid ticker"}
    """
    report = {}
    for t in dropped:
        report[t] = "Not returned by yfinance — possibly delisted or invalid ticker"
    return report


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------

def load_portfolio_data(
    tickers: list[str],
    start: str,
    end: Optional[str] = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Fetch adjusted price data and run all SOP data quality checks.

    Parameters
    ----------
    tickers : list[str]
        NSE tickers with .NS suffix (e.g. ["RELIANCE.NS", "TCS.NS"]).
        Min 3, max 15 per config.
    start : str
        Start date as "YYYY-MM-DD".
    end : str, optional
        End date as "YYYY-MM-DD". Defaults to today if None.

    Returns
    -------
    prices_df : pd.DataFrame
        Adjusted daily closing prices. Index = trading dates. Columns = tickers.
        Short gaps (<=2 days) forward-filled. Large gaps remain NaN.
    quality_report : dict
        Structured report ready for Streamlit sidebar rendering. Contains:
        {
            "summary": {"total_stocks": int, "total_trading_days": int,
                        "date_range": str, "flags_raised": int},
            "per_stock": {
                ticker: {
                    "company_name": str,
                    "sector": str,
                    "missing_rows": int,
                    "missing_pct": float,
                    "missing_flag": bool,
                    "max_consecutive_repeats": int,
                    "stale_flag": bool,
                    "stale_periods": list,
                    "large_gap_dates": list,
                    "status": "✅ OK" | "⚠️ Warning"
                }
            },
            "delisted": {ticker: str},
            "warnings": [str],           # Human-readable warning messages
        }

    Raises
    ------
    ValueError
        If tickers list is empty, tickers lack .NS suffix,
        or yfinance returns no data.
    FileNotFoundError
        If config.yaml or tickers.csv are missing.

    Example
    -------
    >>> prices, report = load_portfolio_data(
    ...     tickers=["RELIANCE.NS", "HDFCBANK.NS", "TCS.NS"],
    ...     start="2020-01-01",
    ...     end="2024-12-31"
    ... )
    >>> print(report["summary"])
    >>> print(prices.tail())
    """
    cfg = _load_config()
    end = end or date.today().strftime("%Y-%m-%d")

    # Validate ticker count against config
    min_stocks = cfg.get("min_stocks", 3)
    max_stocks = cfg.get("max_stocks", 15)
    if len(tickers) < min_stocks:
        raise ValueError(f"Select at least {min_stocks} stocks (got {len(tickers)}).")
    if len(tickers) > max_stocks:
        raise ValueError(f"Select at most {max_stocks} stocks (got {len(tickers)}).")

    # Load ticker metadata for enriching the report
    try:
        meta_df = get_ticker_metadata()
        meta = meta_df.set_index("ticker")[["company_name", "sector"]].to_dict("index")
    except Exception:
        meta = {}

    # --- Step 1: Fetch raw data ---
    raw_prices, dropped = _fetch_raw_prices(tickers, start, end, cfg)

    # Tickers actually returned
    valid_tickers = list(raw_prices.columns)

    # --- Step 2: Missing data check (pre-fill) ---
    missing_report = _check_missing_data(raw_prices, cfg)

    # --- Step 3: Stale price check (pre-fill, on raw data) ---
    stale_report = _check_stale_prices(raw_prices, cfg)

    # --- Step 4: Forward-fill short gaps; flag large gaps ---
    prices_filled, gap_report = _check_forward_fill_gaps(raw_prices, cfg)

    # --- Step 5: Delisted check ---
    delisted_report = _check_delisted(tickers, valid_tickers, dropped)

    # --- Step 6: Build structured quality report ---
    warnings_list: list[str] = []
    flags_raised = 0
    per_stock: dict[str, dict] = {}

    for ticker in valid_tickers:
        ticker_meta = meta.get(ticker, {"company_name": ticker, "sector": "Unknown"})

        missing_flag = missing_report[ticker]["missing_flag"]
        stale_flag = stale_report[ticker]["stale_flag"]
        large_gaps = gap_report.get(ticker, [])
        has_large_gap = len(large_gaps) > 0
        has_any_flag = missing_flag or stale_flag or has_large_gap

        if has_any_flag:
            flags_raised += 1

        if missing_flag:
            warnings_list.append(
                f"⚠️ {ticker}: {missing_report[ticker]['missing_pct']:.1f}% missing data "
                f"({missing_report[ticker]['missing_rows']} rows) — exceeds {cfg.get('max_missing_data_pct', 0.02)*100:.0f}% threshold."
            )
        if stale_flag:
            warnings_list.append(
                f"⚠️ {ticker}: Stale/frozen price detected — same price repeated "
                f"{stale_report[ticker]['max_consecutive_repeats']} consecutive days."
            )
        if has_large_gap:
            warnings_list.append(
                f"⚠️ {ticker}: Large data gap(s) detected at: {', '.join(large_gaps)} "
                f"(>{cfg.get('max_forward_fill_days', 2)}-day gaps left as NaN)."
            )

        per_stock[ticker] = {
            "company_name": ticker_meta.get("company_name", ticker),
            "sector": ticker_meta.get("sector", "Unknown"),
            "missing_rows": missing_report[ticker]["missing_rows"],
            "missing_pct": missing_report[ticker]["missing_pct"],
            "missing_flag": missing_flag,
            "max_consecutive_repeats": stale_report[ticker]["max_consecutive_repeats"],
            "stale_flag": stale_flag,
            "stale_periods": stale_report[ticker]["stale_periods"],
            "large_gap_dates": large_gaps,
            "status": "⚠️ Warning" if has_any_flag else "✅ OK",
        }

    for ticker, reason in delisted_report.items():
        warnings_list.append(f"🚫 {ticker}: {reason}")

    quality_report = {
        "summary": {
            "total_stocks": len(valid_tickers),
            "total_trading_days": len(prices_filled),
            "date_range": f"{start} → {end}",
            "data_fetched_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "flags_raised": flags_raised,
            "delisted_count": len(delisted_report),
        },
        "per_stock": per_stock,
        "delisted": delisted_report,
        "warnings": warnings_list,
    }

    logger.info(
        f"Data loaded: {len(valid_tickers)} stocks, {len(prices_filled)} rows. "
        f"Flags: {flags_raised}. Delisted/dropped: {len(delisted_report)}."
    )

    return prices_filled, quality_report


# ---------------------------------------------------------------------------
# Benchmark loader
# ---------------------------------------------------------------------------

def load_benchmark_data(
    start: str,
    end: Optional[str] = None,
) -> pd.Series:
    """
    Fetch NIFTY 50 Index (^NSEI) adjusted closing prices.
    Benchmark ticker is read from config.yaml — never hardcoded.

    SOP Rule: Benchmark is ^NSEI. auto_adjust=True enforced.

    Parameters
    ----------
    start : str — "YYYY-MM-DD"
    end   : str — "YYYY-MM-DD" (default: today)

    Returns
    -------
    pd.Series — benchmark adjusted close, indexed by trading date

    Unit Test Case:
        Input:  start="2020-01-01"
        Expect: Series.name == "^NSEI"
                Series.index is a DatetimeIndex
                Series.isna().sum() < len(Series) * 0.05  (< 5% missing)
    """
    cfg = _load_config()
    end = end or date.today().strftime("%Y-%m-%d")
    benchmark = cfg.get("benchmark_ticker", "^NSEI")

    logger.info(f"Fetching benchmark {benchmark} from {start} to {end} ...")

    raw = yf.download(
        tickers=benchmark,
        start=start,
        end=end,
        auto_adjust=True,       # SOP: NEVER change this to False
        progress=False,
    )

    if raw.empty:
        raise ValueError(
            f"yfinance returned no data for benchmark {benchmark}. "
            f"Check your internet connection and date range."
        )

    # Robustly extract a 1-D price Series from yfinance output.
    # yfinance >= 0.2.x returns MultiIndex columns (Price, Ticker) even for
    # single tickers when auto_adjust=True — handle both old and new formats.
    if isinstance(raw.columns, pd.MultiIndex):
        # New yfinance: columns = [("Close", "^NSEI"), ("High", "^NSEI"), ...]
        if "Close" in raw.columns.get_level_values(0):
            close_data = raw["Close"]
        else:
            close_data = raw.iloc[:, 0]
        # close_data may still be a DataFrame with one column
        if isinstance(close_data, pd.DataFrame):
            close_data = close_data.iloc[:, 0]
    else:
        # Old yfinance: flat columns
        if "Close" in raw.columns:
            close_data = raw["Close"]
        else:
            close_data = raw.iloc[:, 0]

    series = pd.Series(
        np.array(close_data).flatten(),
        index=raw.index,
        name=benchmark,
        dtype=float,
    ).dropna()
    return series


# ---------------------------------------------------------------------------
# Sidebar render helper (called from app.py)
# ---------------------------------------------------------------------------

def render_quality_report_sidebar(quality_report: dict) -> None:
    """
    Render the data quality report in the Streamlit sidebar.

    Call this from app.py after loading data:
        from src.data_loader import render_quality_report_sidebar
        render_quality_report_sidebar(quality_report)

    Displays:
        - Summary stats (total stocks, trading days, date range)
        - Per-stock status (✅ OK / ⚠️ Warning)
        - Full warning messages in an expander

    SOP Rule: Never suppress this report. Always show it to the user.
    """
    try:
        import streamlit as st
    except ImportError:
        logger.warning("Streamlit not installed — cannot render sidebar report.")
        return

    summary = quality_report.get("summary", {})
    per_stock = quality_report.get("per_stock", {})
    warnings = quality_report.get("warnings", [])
    delisted = quality_report.get("delisted", {})

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔍 Data Quality Report")

    # Summary bar
    col1, col2 = st.sidebar.columns(2)
    col1.metric("Stocks Loaded", summary.get("total_stocks", 0))
    col2.metric("Trading Days", summary.get("total_trading_days", 0))
    st.sidebar.caption(f"📅 {summary.get('date_range', 'N/A')}")
    st.sidebar.caption(f"🕐 Fetched: {summary.get('data_fetched_at', 'N/A')}")

    flags = summary.get("flags_raised", 0)
    if flags == 0 and not delisted:
        st.sidebar.success("✅ All stocks passed quality checks.")
    else:
        st.sidebar.warning(f"⚠️ {flags} stock(s) flagged. {len(delisted)} possibly delisted.")

    # Per-stock status table
    with st.sidebar.expander("📋 Per-Stock Status", expanded=flags > 0):
        for ticker, info in per_stock.items():
            status = info["status"]
            company = info["company_name"]
            missing_pct = info["missing_pct"]
            st.markdown(
                f"{status} **{ticker}** — {company}  \n"
                f"&nbsp;&nbsp;&nbsp;Missing: `{missing_pct}%` | "
                f"Stale: `{'Yes' if info['stale_flag'] else 'No'}` | "
                f"Large gaps: `{len(info['large_gap_dates'])}`"
            )

    # Warning detail expander
    if warnings:
        with st.sidebar.expander("⚠️ Warnings Detail", expanded=False):
            for w in warnings:
                st.markdown(w)

    # Delisted notification
    if delisted:
        with st.sidebar.expander("🚫 Delisted / Invalid Tickers", expanded=True):
            for ticker, reason in delisted.items():
                st.error(f"**{ticker}**: {reason}")


# ---------------------------------------------------------------------------
# Inline Unit Test Stubs
# Run: pytest src/data_loader.py  (requires pytest)
# ---------------------------------------------------------------------------

def _make_test_prices(n_rows: int = 100) -> pd.DataFrame:
    """Helper: Make a clean price DataFrame for testing."""
    import numpy as np
    dates = pd.bdate_range("2023-01-01", periods=n_rows)
    np.random.seed(42)
    data = {
        "RELIANCE.NS": 2400 + np.random.randn(n_rows).cumsum(),
        "TCS.NS": 3500 + np.random.randn(n_rows).cumsum(),
        "HDFCBANK.NS": 1600 + np.random.randn(n_rows).cumsum(),
    }
    return pd.DataFrame(data, index=dates)


def test_missing_data_flag_above_threshold():
    """
    Stocks with >2% missing should be flagged.
    Expected: HDFCBANK.NS flagged, TCS.NS not flagged.
    """
    prices = _make_test_prices(100)
    prices.loc[prices.index[:5], "HDFCBANK.NS"] = np.nan  # 5% missing → flag
    prices.loc[prices.index[:1], "TCS.NS"] = np.nan       # 1% missing → no flag
    cfg = {"max_missing_data_pct": 0.02}
    report = _check_missing_data(prices, cfg)
    assert report["HDFCBANK.NS"]["missing_flag"] is True,  "HDFCBANK.NS should be flagged at 5%"
    assert report["TCS.NS"]["missing_flag"] is False,      "TCS.NS should NOT be flagged at 1%"
    print("✅ test_missing_data_flag_above_threshold PASSED")


def test_stale_price_detection():
    """
    Stocks with same price >3 consecutive days should be flagged.
    Expected: SBIN.NS flagged (4 repeats), TCS.NS not flagged (3 repeats).
    """
    prices = _make_test_prices(20)
    prices.loc[prices.index[0:4], "RELIANCE.NS"] = 2400.0  # 4 repeats → flag
    prices.loc[prices.index[5:8], "TCS.NS"] = 3500.0       # 3 repeats → NOT flagged (threshold = >3)
    cfg = {"stale_price_threshold": 3}
    report = _check_stale_prices(prices, cfg)
    assert report["RELIANCE.NS"]["stale_flag"] is True, "RELIANCE.NS should be flagged (4 repeats > 3)"
    assert report["TCS.NS"]["stale_flag"] is False,     "TCS.NS should NOT be flagged (3 repeats == threshold)"
    print("✅ test_stale_price_detection PASSED")


def test_forward_fill_respects_window():
    """
    Gaps <= 2 days should be filled. Gaps > 2 days should remain NaN.
    """
    prices = _make_test_prices(20)
    # 2-day gap → should be filled
    prices.loc[prices.index[3:5], "TCS.NS"] = np.nan
    # 4-day gap → should remain NaN
    prices.loc[prices.index[10:14], "HDFCBANK.NS"] = np.nan
    cfg = {"max_forward_fill_days": 2}
    filled, gap_report = _check_forward_fill_gaps(prices, cfg)
    assert filled["TCS.NS"].iloc[3:5].isna().sum() == 0,    "2-day gap should be filled"
    assert filled["HDFCBANK.NS"].iloc[10:14].isna().sum() > 0, "4-day gap should NOT be filled"
    assert len(gap_report["HDFCBANK.NS"]) > 0,               "Large gap should be reported"
    print("✅ test_forward_fill_respects_window PASSED")


def test_delisted_ticker_caught():
    """
    Tickers not returned by yfinance should appear in delisted report.
    """
    requested = ["RELIANCE.NS", "FAKECO.NS"]
    returned = ["RELIANCE.NS"]
    dropped = ["FAKECO.NS"]
    report = _check_delisted(requested, returned, dropped)
    assert "FAKECO.NS" in report, "FAKECO.NS should be in delisted report"
    print("✅ test_delisted_ticker_caught PASSED")


if __name__ == "__main__":
    # Run inline tests
    import numpy as np
    print("\n--- Running data_loader unit tests ---")
    test_missing_data_flag_above_threshold()
    test_stale_price_detection()
    test_forward_fill_respects_window()
    test_delisted_ticker_caught()
    print("--- All inline tests passed ---\n")
