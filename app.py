"""
app.py
======
Streamlit entry point — Interactive Risk Analysis & Portfolio Optimization Dashboard.

SOP Rules Enforced:
    - All config read from config.yaml via _load_config()
    - Data quality report from data_loader always shown — never suppressed
    - CVXPY frontier never replaced with random simulation
    - Cholesky Monte Carlo always used
    - All chart rendering via Plotly
    - All NSE tickers validate .NS suffix
    - Benchmark always ^NSEI (from config)

Run:
    streamlit run app.py
"""

from __future__ import annotations

import warnings
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yaml

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Page config — MUST be first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Indian Portfolio Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------
_CONFIG_PATH = Path(__file__).parent / "config.yaml"


@st.cache_data(ttl=300)
def _load_config() -> dict:
    if not _CONFIG_PATH.exists():
        st.error("❌ config.yaml not found. Ensure it is in the project root.")
        st.stop()
    with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Source module imports
# ---------------------------------------------------------------------------
from src.data_loader import (
    get_ticker_metadata,
    load_benchmark_data,
    load_portfolio_data,
    render_quality_report_sidebar,
)
from src.monte_carlo import (
    build_simulation_summary,
    extract_1day_returns,
    run_simulation,
    sample_paths_for_plot,
)
from src.optimization import (
    compute_efficient_frontier,
    max_sharpe_portfolio,
    min_variance_portfolio,
    random_portfolio_scatter,
)
from src.rebalancing import build_rebalancing_summary, simulate_rebalancing
from src.risk_metrics import (
    compute_all_metrics,
    compute_portfolio_returns,
)
from src.stress_test import load_stress_scenarios, run_all_scenarios

# ---------------------------------------------------------------------------
# Global CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    .metric-card {
        background: #1e1e2e;
        border-radius: 10px;
        padding: 16px 20px;
        border-left: 4px solid #7c3aed;
    }
    .kpi-label { font-size: 0.78rem; color: #a0aec0; margin-bottom: 4px; }
    .kpi-value { font-size: 1.6rem; font-weight: 700; color: #f0f4ff; }
    .kpi-delta { font-size: 0.82rem; }
    .section-header {
        font-size: 1.1rem; font-weight: 600;
        color: #c4b5fd; margin-top: 1rem; margin-bottom: 0.4rem;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 8px 20px;
        font-weight: 600;
    }
    div[data-testid="stSidebarContent"] { background-color: #13131f; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt_inr(val: float) -> str:
    """Format a float as ₹ with Indian comma notation."""
    if val >= 1e7:
        return f"₹{val/1e7:.2f} Cr"
    elif val >= 1e5:
        return f"₹{val/1e5:.2f} L"
    return f"₹{val:,.0f}"


def _color(val: float, good_positive: bool = True) -> str:
    if (val > 0) == good_positive:
        return "#22c55e"
    return "#ef4444"


@st.cache_data(ttl=600, show_spinner=False)
def _cached_load_data(tickers: tuple, start: str, end: str):
    return load_portfolio_data(list(tickers), start, end)


@st.cache_data(ttl=600, show_spinner=False)
def _cached_benchmark(start: str, end: str):
    return load_benchmark_data(start, end)


# ---------------------------------------------------------------------------
# ═══════════════════════════  SIDEBAR  ════════════════════════════════════
# ---------------------------------------------------------------------------

def render_sidebar(cfg: dict) -> dict | None:
    """Render the sidebar and return selections, or None if not ready."""

    st.sidebar.markdown(
        """
        <div style="
            background: linear-gradient(135deg, #1e3a5f 0%, #0f2442 100%);
            border-radius: 10px;
            padding: 14px 16px;
            margin-bottom: 8px;
            text-align: center;
            border: 1px solid #2d5491;
        ">
            <div style="font-size: 1.5rem; font-weight: 800; color: #ffffff; letter-spacing: 0.05em;">
                📈 NSE INDIA
            </div>
            <div style="font-size: 0.72rem; color: #93b4d8; margin-top: 2px; letter-spacing: 0.12em;">
                PORTFOLIO ANALYTICS
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.sidebar.markdown("## 📊 Portfolio Dashboard")
    st.sidebar.markdown("---")

    # ── Risk-free rate freshness warning ──────────────────────────────────
    rf = cfg.get("risk_free_rate", 0.065)
    st.sidebar.info(
        f"📌 **Risk-Free Rate:** {rf*100:.2f}%  \n"
        "Update in `config.yaml` when RBI changes the repo rate.",
        icon="⚠️",
    )

    # ── Stock selector ────────────────────────────────────────────────────
    st.sidebar.markdown("### 🏦 Stock Selection")
    try:
        meta_df = get_ticker_metadata()
    except Exception as e:
        st.sidebar.error(f"Could not load tickers.csv: {e}")
        return None

    sector_options = ["All Sectors"] + sorted(meta_df["sector"].unique().tolist())
    selected_sector = st.sidebar.selectbox("Filter by sector", sector_options, key="sector_filter")

    filtered = meta_df if selected_sector == "All Sectors" \
        else meta_df[meta_df["sector"] == selected_sector]

    ticker_options = [
        f"{row['ticker']} — {row['company_name']}"
        for _, row in filtered.iterrows()
    ]
    
    if "persistent_stocks" not in st.session_state:
        all_options = [f"{row['ticker']} — {row['company_name']}" for _, row in meta_df.iterrows()]
        defaults = ["RELIANCE.NS — Reliance Industries Ltd",
                    "HDFCBANK.NS — HDFC Bank Ltd",
                    "TCS.NS — Tata Consultancy Services Ltd",
                    "INFY.NS — Infosys Ltd",
                    "ICICIBANK.NS — ICICI Bank Ltd"]
        st.session_state.persistent_stocks = [d for d in defaults if d in all_options][:5]

    def update_stocks():
        if "stock_selector" in st.session_state:
            st.session_state.persistent_stocks = st.session_state.stock_selector

    current_selection = st.session_state.persistent_stocks
    ticker_options = list(dict.fromkeys(current_selection + ticker_options))

    ticker_map = {
        f"{row['ticker']} — {row['company_name']}": row['ticker']
        for _, row in meta_df.iterrows()
    }

    selected_labels = st.sidebar.multiselect(
        "Select stocks (3–15)",
        options=ticker_options,
        default=current_selection,
        key="stock_selector",
        on_change=update_stocks
    )
    selected_tickers = [ticker_map[lbl] for lbl in selected_labels]

    min_stocks = cfg.get("min_stocks", 3)
    max_stocks = cfg.get("max_stocks", 15)
    if len(selected_tickers) < min_stocks:
        st.sidebar.warning(f"Select at least {min_stocks} stocks.")
        return None
    if len(selected_tickers) > max_stocks:
        st.sidebar.error(f"Select at most {max_stocks} stocks.")
        return None

    # ── Date range ────────────────────────────────────────────────────────
    st.sidebar.markdown("### 📅 Date Range")
    cfg_start = cfg.get("data_start", "2015-01-01")
    default_start = datetime.strptime(cfg_start, "%Y-%m-%d").date()
    today = date.today()

    col1, col2 = st.sidebar.columns(2)
    start_date = col1.date_input("Start", value=default_start,
                                 min_value=date(2000, 1, 1), max_value=today - timedelta(days=365))
    end_date = col2.date_input("End", value=today,
                               min_value=start_date + timedelta(days=365), max_value=today)

    if (end_date - start_date).days < 365:
        st.sidebar.warning("Select at least 1 year of data for reliable results.")
        return None

    # ── Optimization objective ────────────────────────────────────────────
    st.sidebar.markdown("### ⚙️ Optimization")
    obj = st.sidebar.radio(
        "Objective",
        ["Maximum Sharpe", "Minimum Volatility", "Target Return"],
        key="opt_objective",
    )
    target_ret = None
    if obj == "Target Return":
        target_ret = st.sidebar.slider(
            "Target annual return (%)", min_value=5, max_value=50, value=15, step=1
        ) / 100.0

    max_weight = st.sidebar.slider(
        "Max weight per stock (%)",
        min_value=10, max_value=100,
        value=int(cfg.get("max_weight_per_stock", 0.40) * 100),
        step=5,
        key="max_weight_slider",
    ) / 100.0

    capital = st.sidebar.number_input(
        "Investment Capital (₹)", min_value=100_000,
        value=1_000_000, step=100_000, format="%d", key="capital_input"
    )

    # ── Monte Carlo drift ─────────────────────────────────────────────────
    st.sidebar.markdown("### 🎲 Monte Carlo")
    drift_mode = st.sidebar.selectbox(
        "Drift mode",
        ["historical", "zero", "user_defined"],
        key="drift_mode",
    )
    user_drift = None
    if drift_mode == "user_defined":
        user_drift = st.sidebar.slider(
            "Expected annual return (%)", 0, 30, 12, 1, key="user_drift"
        ) / 100.0

    run_btn = st.sidebar.button("🚀 Run Analysis", type="primary", use_container_width=True)

    return {
        "tickers": selected_tickers,
        "start": str(start_date),
        "end": str(end_date),
        "objective": obj,
        "target_return": target_ret,
        "max_weight": max_weight,
        "capital": float(capital),
        "drift_mode": drift_mode,
        "user_drift": user_drift,
        "run": run_btn,
    }


# ---------------------------------------------------------------------------
# ══════════════════════════  TAB 1: OVERVIEW  ═════════════════════════════
# ---------------------------------------------------------------------------

def render_overview(
    metrics: dict,
    weights: dict[str, float],
    prices: pd.DataFrame,
    benchmark: pd.Series,
    portfolio_returns: pd.Series,
    capital: float,
):
    st.markdown("### 📋 Portfolio Overview")

    bm = metrics["benchmark"]
    col1, col2, col3, col4, col5, col6 = st.columns(6)

    def _kpi(col, label, value, delta=None, pct=True, good_pos=True):
        suffix = "%" if pct else ""
        delta_html = ""
        if delta is not None:
            color = _color(delta, good_pos)
            sign = "▲" if delta > 0 else "▼"
            delta_html = (f'<div class="kpi-delta" style="color:{color}">'
                          f'{sign} {abs(delta):.2f}{suffix} vs benchmark</div>')
        col.markdown(
            f'<div class="metric-card">'
            f'<div class="kpi-label">{label}</div>'
            f'<div class="kpi-value">{value:.2f}{suffix}</div>'
            f'{delta_html}</div>',
            unsafe_allow_html=True,
        )

    _kpi(col1, "Annualised Return", metrics["annualized_return"],
         metrics["annualized_return"] - bm["annualized_return"])
    _kpi(col2, "Annualised Volatility", metrics["annualized_volatility"],
         metrics["annualized_volatility"] - bm["annualized_volatility"], good_pos=False)
    _kpi(col3, "Sharpe Ratio", metrics["sharpe"],
         metrics["sharpe"] - bm["sharpe"], pct=False)
    _kpi(col4, "Max Drawdown", metrics["max_drawdown"]["max_drawdown_pct"],
         metrics["max_drawdown"]["max_drawdown_pct"] - bm["max_drawdown_pct"], good_pos=False)
    _kpi(col5, "Current Beta", metrics["current_beta"] or 0, pct=False)
    _kpi(col6, "Div. Ratio", metrics["diversification_ratio"]["diversification_ratio"], pct=False)

    st.markdown("<br>", unsafe_allow_html=True)
    left, right = st.columns([1, 2])

    # Allocation pie chart
    with left:
        st.markdown('<div class="section-header">Portfolio Allocation</div>',
                    unsafe_allow_html=True)
        labels = list(weights.keys())
        vals   = [weights[t] * 100 for t in labels]
        short  = [t.replace(".NS", "") for t in labels]
        fig_pie = go.Figure(go.Pie(
            labels=short, values=vals,
            hole=0.45,
            textinfo="label+percent",
            hovertemplate="%{label}<br>Weight: %{value:.1f}%<extra></extra>",
        ))
        fig_pie.update_layout(
            showlegend=False, height=340,
            margin=dict(t=10, b=10, l=10, r=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    # Cumulative return vs benchmark
    with right:
        st.markdown('<div class="section-header">Portfolio vs NIFTY 50</div>',
                    unsafe_allow_html=True)
        # Fix: Compute cumulative product of (1 + daily_return), then subtract 1
        port_ret_copy = pd.Series(portfolio_returns.copy(), dtype=float)
        cum_port = (1 + port_ret_copy).cumprod() * 100 - 100
        bench_ret = pd.Series(benchmark.pct_change().dropna(), dtype=float)
        bench_aligned = bench_ret.reindex(port_ret_copy.index).fillna(0)
        cum_bench = (1 + bench_aligned).cumprod() * 100 - 100

        fig_cum = go.Figure()
        fig_cum.add_trace(go.Scatter(
            x=list(cum_port.index), y=list(cum_port.values),
            name="Portfolio", line=dict(color="#7c3aed", width=2),
        ))
        fig_cum.add_trace(go.Scatter(
            x=list(cum_bench.index), y=list(cum_bench.values),
            name="NIFTY 50", line=dict(color="#64748b", width=1.5, dash="dot"),
        ))
        fig_cum.update_layout(
            yaxis_title="Cumulative Return (%)", xaxis_title="",
            height=340, legend=dict(orientation="h", y=1.05),
            margin=dict(t=30, b=20, l=40, r=20),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(gridcolor="#2d2d3d"), yaxis=dict(gridcolor="#2d2d3d"),
        )
        st.plotly_chart(fig_cum, use_container_width=True)

    # Drawdown curve
    st.markdown('<div class="section-header">Portfolio Drawdown</div>', unsafe_allow_html=True)
    
    # Retrieve the correctly calculated drawdown series from the risk metrics and invert to negative percentage for charting
    dd_series = pd.Series(metrics["max_drawdown"]["drawdown_series"], dtype=float) * -100
    
    fig_dd = go.Figure(go.Scatter(
        x=list(dd_series.index), y=list(dd_series.values),
        fill="tozeroy", fillcolor="rgba(239,68,68,0.15)",
        line=dict(color="#ef4444", width=1.5),
        name="Drawdown %",
    ))
    fig_dd.update_layout(
        yaxis_title="Drawdown (%)", height=220,
        margin=dict(t=10, b=20, l=40, r=20),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(gridcolor="#2d2d3d"), yaxis=dict(gridcolor="#2d2d3d"),
    )
    st.plotly_chart(fig_dd, use_container_width=True)


# ---------------------------------------------------------------------------
# ════════════════════  TAB 2: EFFICIENT FRONTIER  ═════════════════════════
# ---------------------------------------------------------------------------

def render_frontier(
    returns: pd.DataFrame,
    user_weights: dict[str, float],
    user_metrics: dict,
    max_weight: float,
    cfg: dict,
):
    st.markdown("### 🎯 Efficient Frontier (Exact QP via CVXPY)")
    st.caption(
        "Every point on the frontier is an exactly solved quadratic program — "
        "not random simulation. Random scatter is shown behind for reference only."
    )

    with st.spinner("Computing efficient frontier (CVXPY)..."):
        try:
            frontier = compute_efficient_frontier(returns, n_points=80, max_weight=max_weight)
            scatter  = random_portfolio_scatter(returns, n_portfolios=2000, max_weight=max_weight)
            w_mv, m_mv = min_variance_portfolio(returns, max_weight)
            w_ms, m_ms = max_sharpe_portfolio(returns, cfg.get("risk_free_rate", 0.065), max_weight)
        except Exception as e:
            st.error(f"Optimization error: {e}")
            return

    fig = go.Figure()

    # Background scatter (visual reference only — NOT the frontier)
    fig.add_trace(go.Scatter(
        x=list(scatter["vol_pct"]), y=list(scatter["return_pct"]),
        mode="markers",
        marker=dict(color=list(scatter["sharpe"]), colorscale="Viridis",
                    size=4, opacity=0.35,
                    colorbar=dict(title="Sharpe", thickness=12, len=0.6)),
        name="Random Portfolios (reference)",
        hovertemplate="Vol: %{x:.2f}%<br>Return: %{y:.2f}%<extra></extra>",
    ))

    # CVXPY exact frontier
    fig.add_trace(go.Scatter(
        x=list(frontier["vol_pct"]), y=list(frontier["return_pct"]),
        mode="lines", line=dict(color="#7c3aed", width=3),
        name="Efficient Frontier (CVXPY QP)",
        hovertemplate="Vol: %{x:.2f}%<br>Return: %{y:.2f}%<extra></extra>",
    ))

    # Min-variance portfolio
    if m_mv.get("vol_pct"):
        fig.add_trace(go.Scatter(
            x=[m_mv["vol_pct"]], y=[m_mv["return_pct"]],
            mode="markers+text",
            marker=dict(color="#3b82f6", size=16, symbol="diamond"),
            text=["Min-Var"], textposition="top right",
            name=f"Min Variance (Sharpe {m_mv['sharpe']:.2f})",
            hovertemplate=f"Min Variance<br>Vol: {m_mv['vol_pct']:.2f}%<br>"
                          f"Return: {m_mv['return_pct']:.2f}%<extra></extra>",
        ))

    # Max-Sharpe portfolio
    if m_ms.get("vol_pct"):
        fig.add_trace(go.Scatter(
            x=[m_ms["vol_pct"]], y=[m_ms["return_pct"]],
            mode="markers+text",
            marker=dict(color="#f59e0b", size=16, symbol="star"),
            text=["Max Sharpe"], textposition="top right",
            name=f"Max Sharpe ({m_ms['sharpe']:.2f})",
            hovertemplate=f"Max Sharpe<br>Vol: {m_ms['vol_pct']:.2f}%<br>"
                          f"Return: {m_ms['return_pct']:.2f}%<extra></extra>",
        ))

    # User portfolio
    fig.add_trace(go.Scatter(
        x=[user_metrics["vol_pct"]], y=[user_metrics["return_pct"]],
        mode="markers+text",
        marker=dict(color="#22c55e", size=18, symbol="circle",
                    line=dict(color="white", width=2)),
        text=["Your Portfolio"], textposition="top center",
        name=f"Your Portfolio (Sharpe {user_metrics['sharpe']:.2f})",
        hovertemplate=f"Your Portfolio<br>Vol: {user_metrics['vol_pct']:.2f}%<br>"
                      f"Return: {user_metrics['return_pct']:.2f}%<extra></extra>",
    ))

    fig.update_layout(
        xaxis_title="Annualised Volatility (%)",
        yaxis_title="Annualised Return (%)",
        height=580,
        legend=dict(orientation="h", y=-0.15),
        margin=dict(t=20, b=60, l=60, r=20),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(gridcolor="#2d2d3d"), yaxis=dict(gridcolor="#2d2d3d"),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Weights table
    st.markdown('<div class="section-header">Portfolio Weights Comparison</div>',
                unsafe_allow_html=True)
    tickers = list(user_weights.keys())
    weights_df = pd.DataFrame({
        "Ticker": [t.replace(".NS", "") for t in tickers],
        "Your Portfolio": [f"{user_weights[t]*100:.1f}%" for t in tickers],
        "Max Sharpe": [f"{m_ms['weights_dict'].get(t, 0)*100:.1f}%" for t in tickers],
        "Min Variance": [f"{m_mv['weights_dict'].get(t, 0)*100:.1f}%" for t in tickers],
    })
    st.dataframe(weights_df, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# ══════════════════════  TAB 3: RISK METRICS  ═════════════════════════════
# ---------------------------------------------------------------------------

def render_risk_metrics(metrics: dict, portfolio_returns: pd.Series, capital: float):
    st.markdown("### ⚠️ Advanced Risk Metrics")

    var = metrics["var"]
    cvar = metrics["cvar"]
    dr   = metrics["diversification_ratio"]

    # VaR panel
    st.markdown('<div class="section-header">Value at Risk (95% Confidence)</div>',
                unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("📉 Historical VaR", _fmt_inr(var["var_historical"]),
              f"{var['var_historical_pct']:.2f}% of portfolio")
    c2.metric("📐 Parametric VaR", _fmt_inr(var["var_parametric"]),
              f"{var['var_parametric_pct']:.2f}% of portfolio")
    mc_var_val = var.get("var_mc")
    c3.metric("🎲 Monte Carlo VaR",
              _fmt_inr(mc_var_val) if mc_var_val else "Run Monte Carlo first",
              f"{var.get('var_mc_pct', 0):.2f}% of portfolio" if mc_var_val else "")
    st.caption(var["interpretation"])

    # CVaR
    st.markdown('<div class="section-header">Conditional VaR (Expected Shortfall)</div>',
                unsafe_allow_html=True)
    st.info(f"**CVaR:** {abs(cvar['cvar_pct']):.2f}% average loss in worst 5% of days.  \n"
            + cvar["interpretation"])

    # Return distribution histogram
    st.markdown('<div class="section-header">Daily Return Distribution</div>',
                unsafe_allow_html=True)
    rf_daily = metrics["risk_free_rate_used"] / metrics["trading_days"]
    var_hist_line = np.percentile(portfolio_returns.astype(float), 5)

    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(
        x=list(np.array(portfolio_returns.astype(float).values) * 100),
        nbinsx=80,
        name="Daily Returns",
        marker_color="#7c3aed",
        opacity=0.75,
    ))
    fig_hist.add_vline(x=var_hist_line * 100, line_color="#ef4444", line_dash="dash",
                       annotation_text=f"Historical VaR ({var_hist_line*100:.2f}%)")
    fig_hist.update_layout(
        xaxis_title="Daily Return (%)", yaxis_title="Frequency",
        height=320, margin=dict(t=20, b=40, l=40, r=20),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(gridcolor="#2d2d3d"), yaxis=dict(gridcolor="#2d2d3d"),
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    # Rolling volatility
    st.markdown('<div class="section-header">Rolling Volatility (60-day & 252-day)</div>',
                unsafe_allow_html=True)
    roll60  = portfolio_returns.rolling(60).std()  * np.sqrt(252) * 100
    roll252 = portfolio_returns.rolling(252).std() * np.sqrt(252) * 100
    fig_vol = go.Figure()
    fig_vol.add_trace(go.Scatter(x=list(roll60.index),  y=list(roll60.values),
                                 name="60-day rolling vol",
                                 line=dict(color="#a78bfa", width=1.5)))
    fig_vol.add_trace(go.Scatter(x=list(roll252.index), y=list(roll252.values),
                                 name="252-day rolling vol",
                                 line=dict(color="#f59e0b", width=2)))
    fig_vol.update_layout(
        yaxis_title="Annualised Volatility (%)", height=280,
        margin=dict(t=10, b=20, l=40, r=20),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(gridcolor="#2d2d3d"), yaxis=dict(gridcolor="#2d2d3d"),
    )
    st.plotly_chart(fig_vol, use_container_width=True)

    # Sortino and DR
    st.markdown('<div class="section-header">Additional Metrics</div>',
                unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("Sortino Ratio", f"{metrics['sortino']:.4f}",
              help="Penalises only downside volatility. Higher = better.")
    c2.metric("Diversification Ratio", f"{dr['diversification_ratio']:.4f}",
              delta=f"Portfolio {dr['portfolio_vol']:.1f}% vs avg component {dr['weighted_avg_vol']:.1f}%")
    c3.metric("Current Beta (252d)", f"{metrics['current_beta']:.4f}" if metrics["current_beta"] else "N/A",
              help=">1 amplifies market moves; <1 dampens them.")
    st.caption(dr["interpretation"])


# ---------------------------------------------------------------------------
# ═══════════════════════  TAB 4: MONTE CARLO  ═════════════════════════════
# ---------------------------------------------------------------------------

def render_monte_carlo(
    weights: dict,
    returns: pd.DataFrame,
    capital: float,
    drift_mode: str,
    user_drift: float | None,
    cfg: dict,
):
    st.markdown("### 🎲 Monte Carlo Simulation (Cholesky-Correlated)")
    st.caption(
        "Uses Cholesky decomposition of the covariance matrix to preserve "
        "cross-stock correlations. **L × Z where L = chol(Σ), Z ~ N(0,1)**"
    )

    n_paths = cfg.get("monte_carlo_paths", 10000)

    with st.spinner(f"Running {n_paths:,} correlated Monte Carlo paths..."):
        try:
            sim = run_simulation(
                weights=weights,
                returns=returns,
                drift=drift_mode,
                user_drift=user_drift,
                initial_value=capital,
            )
        except Exception as e:
            st.error(f"Monte Carlo simulation error: {e}")
            return

    summary = build_simulation_summary(sim, capital)
    plot_data = sample_paths_for_plot(sim, n_display=150)

    # KPI row
    h1 = summary["horizons"]["1yr"]
    h3 = summary["horizons"]["3yr"] if summary["horizons"]["3yr"]["day"] <= sim.horizon_days \
        else summary["horizons"]["end"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Expected Value (3yr)", _fmt_inr(h3["expected"]))
    c2.metric("Worst Case 5th pct (3yr)", _fmt_inr(h3["p5"]))
    c3.metric("Best Case 95th pct (3yr)", _fmt_inr(h3["p95"]))
    c4.metric("P(Capital Loss) 3yr", f"{h3['p_loss_pct']:.1f}%")

    # Simulation paths chart
    fig_mc = go.Figure()

    # Thin sample of paths (light, background)
    for path in plot_data["sampled_paths"][:100]:
        fig_mc.add_trace(go.Scatter(
            x=list(plot_data["days"]), y=list(path),
            mode="lines", line=dict(color="#7c3aed", width=1.0),
            showlegend=False, opacity=0.4,
            hoverinfo="skip",
        ))

    # Percentile bands
    band_styles = {
        "p5":  ("#ef4444", "5th pct (Worst case)"),
        "p25": ("#f59e0b", "25th pct"),
        "p50": ("#22c55e", "Median"),
        "p75": ("#3b82f6", "75th pct"),
        "p95": ("#a78bfa", "95th pct (Best case)"),
    }
    for key, (color, name) in band_styles.items():
        if key in plot_data["bands"]:
            fig_mc.add_trace(go.Scatter(
                x=list(plot_data["days"]),
                y=list(plot_data["bands"][key]),
                mode="lines",
                line=dict(color=color, width=2.5),
                name=name,
            ))

    fig_mc.add_hline(y=capital, line_color="white", line_dash="dot",
                     annotation_text="Initial Capital", annotation_position="left")

    year_markers = [252, 504, 756]
    for d in year_markers:
        if d <= sim.horizon_days:
            fig_mc.add_vline(x=d, line_color="#475569", line_dash="dash",
                             annotation_text=f"Yr {d//252}", annotation_position="top")

    fig_mc.update_layout(
        xaxis_title="Trading Days",
        yaxis_title="Portfolio Value (₹)",
        height=500,
        legend=dict(orientation="h", y=-0.18),
        margin=dict(t=20, b=80, l=60, r=20),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(gridcolor="#2d2d3d"), yaxis=dict(gridcolor="#2d2d3d"),
    )
    st.plotly_chart(fig_mc, use_container_width=True)

    # Horizon table
    st.markdown('<div class="section-header">Horizon Summary</div>', unsafe_allow_html=True)
    horizon_rows = []
    for label in ["1yr", "2yr", "3yr"]:
        h = summary["horizons"][label]
        if h["day"] <= sim.horizon_days:
            horizon_rows.append({
                "Horizon": label,
                "Expected Value": _fmt_inr(h["expected"]),
                "Expected Return": f"{h['expected_return_pct']:.1f}%",
                "5th Pct (Worst)": _fmt_inr(h["p5"]),
                "95th Pct (Best)": _fmt_inr(h["p95"]),
                "P(Loss)": f"{h['p_loss_pct']:.1f}%",
            })
    if horizon_rows:
        st.dataframe(pd.DataFrame(horizon_rows), use_container_width=True, hide_index=True)

    for insight in summary["interpretation"]:
        st.markdown(f"💡 {insight}")


# ---------------------------------------------------------------------------
# ══════════════════════  TAB 5: STRESS TEST  ══════════════════════════════
# ---------------------------------------------------------------------------

def render_stress_test(weights: dict, prices: pd.DataFrame, cfg: dict):
    st.markdown("### 🔴 Stress Test — Historical Crash Scenarios")
    st.caption(
        "All scenario date ranges loaded from `data/stress_periods.json`. "
        "Correlation delta (stress − normal) is the key output."
    )

    with st.spinner("Running all stress scenarios..."):
        try:
            all_results = run_all_scenarios(weights, prices)
        except Exception as e:
            st.error(f"Stress test error: {e}")
            return

    available = {sid: r for sid, r in all_results.items() if r.get("data_available")}
    unavailable = {sid: r for sid, r in all_results.items() if not r.get("data_available")}

    if unavailable:
        with st.expander(f"⚠️ {len(unavailable)} scenarios skipped (insufficient historical data)"):
            for sid, r in unavailable.items():
                st.write(f"**{r.get('scenario_label', sid)}**: " + "; ".join(r.get("warnings", [])))

    if not available:
        st.warning("No scenarios have data coverage. Extend `data_start` in config.yaml.")
        return

    scenario_tabs = st.tabs([r["scenario_label"] for r in available.values()])

    for tab, (sid, res) in zip(scenario_tabs, available.items()):
        with tab:
            st.markdown(f"**{res['stress_start']}  →  {res['stress_end']}**")
            st.caption(res.get("scenario_description", ""))

            # Drawdown comparison
            opt = res["optimized"]
            ew  = res["equal_weight"]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Optimised Drawdown", f"{opt['max_drawdown_pct']:.1f}%")
            c2.metric("Equal-Weight Drawdown", f"{ew['max_drawdown_pct']:.1f}%",
                      delta=f"{res['comparison']['drawdown_improvement_pct']:+.1f}pp",
                      delta_color="inverse")
            c3.metric("Optimised Recovery",
                      f"{opt['recovery_days']}d" if opt["recovery_days"] else "Not recovered")
            c4.metric("Total Return (Stress)",
                      f"{opt['total_return_pct']:.1f}%",
                      delta=f"{res['comparison']['return_improvement_pct']:+.1f}pp vs EW")

            st.info(res["comparison"]["insight"])

            # Correlation heatmaps
            corr_data = res["correlation"]
            st.markdown('<div class="section-header">Correlation: Normal vs Stress Period</div>',
                        unsafe_allow_html=True)

            tickers_short = [t.replace(".NS", "") for t in res["tickers_analysed"]]

            col_n, col_s, col_d = st.columns(3)

            def _heatmap(df: pd.DataFrame, title: str, colorscale: str, zmin=-1, zmax=1):
                if df.empty:
                    return go.Figure()
                short = [c.replace(".NS", "") for c in df.columns]
                fig = go.Figure(go.Heatmap(
                    z=df.values.tolist(), x=short, y=short,
                    colorscale=colorscale, zmin=zmin, zmax=zmax,
                    text=np.round(df.values, 2).tolist(), texttemplate="%{text}",
                    hovertemplate="%{y} × %{x}: %{z:.3f}<extra></extra>",
                ))
                fig.update_layout(
                    title=title, height=320,
                    margin=dict(t=40, b=20, l=20, r=20),
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                )
                return fig

            with col_n:
                st.plotly_chart(
                    _heatmap(corr_data["normal_corr"], "Normal Period", "Blues"),
                    use_container_width=True,
                )
            with col_s:
                st.plotly_chart(
                    _heatmap(corr_data["stress_corr"], "Stress Period", "Reds"),
                    use_container_width=True,
                )
            with col_d:
                if not corr_data["delta_corr"].empty:
                    dmax = corr_data["delta_corr"].values[
                        ~np.eye(len(corr_data["delta_corr"]), dtype=bool)
                    ].max()
                    st.plotly_chart(
                        _heatmap(
                            corr_data["delta_corr"], "Δ Correlation (Stress − Normal)",
                            "RdYlGn_r", zmin=-dmax, zmax=dmax,
                        ),
                        use_container_width=True,
                    )

            st.markdown(f"**📊 Correlation insight:** {corr_data['interpretation']}")
            st.markdown(
                f"Normal window: {corr_data.get('normal_window_days', '?')} days | "
                f"Avg correlation: **{corr_data.get('avg_normal_pairwise', 'N/A')}** → "
                f"Stress: **{corr_data.get('avg_stress_pairwise', 'N/A')}** | "
                f"Δ = **{corr_data.get('avg_delta_pairwise', 'N/A')}**"
            )


# ---------------------------------------------------------------------------
# ════════════════════════  TAB 6: REBALANCING  ════════════════════════════
# ---------------------------------------------------------------------------

def render_rebalancing(weights: dict, prices: pd.DataFrame, capital: float, cfg: dict):
    st.markdown("### ♻️ Rebalancing vs Buy-and-Hold")

    freq = cfg.get("rebalancing_frequency", "Q")
    tc   = cfg.get("transaction_cost", 0.001)
    freq_label = {"Q": "Quarterly", "M": "Monthly", None: "None (Buy-and-Hold)"}.get(freq, freq)

    st.caption(
        f"Frequency: **{freq_label}** | Transaction cost: **{tc*100:.2f}%** round-trip "
        f"(from `config.yaml`)"
    )

    with st.spinner("Simulating rebalancing..."):
        try:
            bah, rebal, log = simulate_rebalancing(
                prices=prices,
                target_weights=weights,
                frequency=freq,
                initial_capital=capital,
                transaction_cost=tc,
            )
        except Exception as e:
            st.error(f"Rebalancing simulation error: {e}")
            return

    summary = build_rebalancing_summary(bah, rebal, log, capital)
    bah_s   = summary["buy_and_hold"]
    reb_s   = summary["rebalanced"]
    comp    = summary["comparison"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("BAH Final Value",    _fmt_inr(bah_s["final_value"]),
              f"{bah_s['total_return_pct']:.1f}% total")
    c2.metric("Rebalanced Final",   _fmt_inr(reb_s["final_value"]),
              f"{comp['return_difference_pp']:+.1f}pp vs BAH",
              delta_color="normal")
    c3.metric("Rebalancing Events", str(reb_s["n_rebalances"]))
    c4.metric("Total Costs Paid",   _fmt_inr(reb_s["total_cost_inr"]),
              f"{reb_s['total_cost_pct']:.3f}% of capital")

    st.info(comp["insight"])

    # Value comparison chart
    fig_reb = go.Figure()
    fig_reb.add_trace(go.Scatter(
        x=bah.index, y=bah.values,
        name="Buy & Hold", line=dict(color="#64748b", width=2, dash="dot"),
    ))
    fig_reb.add_trace(go.Scatter(
        x=rebal.index, y=rebal.values,
        name=f"Rebalanced ({freq_label})", line=dict(color="#7c3aed", width=2.5),
    ))
    # Add rebalancing event markers
    if log:
        event_dates = [pd.Timestamp(e["date"]) for e in log]
        event_vals  = [rebal.loc[rebal.index.asof(d)] for d in event_dates if d >= rebal.index[0]]
        fig_reb.add_trace(go.Scatter(
            x=event_dates[:len(event_vals)], y=event_vals,
            mode="markers", marker=dict(color="#f59e0b", size=8, symbol="triangle-up"),
            name="Rebalancing Event",
        ))

    fig_reb.add_hline(y=capital, line_dash="dot", line_color="#475569",
                      annotation_text="Initial Capital")
    fig_reb.update_layout(
        yaxis_title="Portfolio Value (₹)", height=400,
        legend=dict(orientation="h", y=-0.15),
        margin=dict(t=10, b=60, l=60, r=20),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(gridcolor="#2d2d3d"), yaxis=dict(gridcolor="#2d2d3d"),
    )
    st.plotly_chart(fig_reb, use_container_width=True)

    # Stats comparison table
    st.markdown('<div class="section-header">Performance Comparison</div>',
                unsafe_allow_html=True)
    comp_df = pd.DataFrame({
        "Metric": ["Final Value", "Total Return", "Ann. Return", "Max Drawdown", "Volatility"],
        "Buy & Hold": [
            _fmt_inr(bah_s["final_value"]),
            f"{bah_s['total_return_pct']:.2f}%",
            f"{bah_s['annualized_return']:.2f}%",
            f"{bah_s['max_drawdown_pct']:.2f}%",
            f"{bah_s['volatility_pct']:.2f}%",
        ],
        "Rebalanced": [
            _fmt_inr(reb_s["final_value"]),
            f"{reb_s['total_return_pct']:.2f}%",
            f"{reb_s['annualized_return']:.2f}%",
            f"{reb_s['max_drawdown_pct']:.2f}%",
            f"{reb_s['volatility_pct']:.2f}%",
        ],
    })
    st.dataframe(comp_df, use_container_width=True, hide_index=True)

    # Transaction log
    if log:
        with st.expander(f"📋 Transaction Log ({len(log)} events)"):
            log_df = pd.DataFrame([{
                "Date": e["date"],
                "Value Pre-Rebal": _fmt_inr(e["portfolio_value_pre"]),
                "Turnover": f"{e['turnover_fraction']*100:.1f}%",
                "Cost": _fmt_inr(e["cost_paid_inr"]),
                "Cost %": f"{e['cost_pct']:.3f}%",
                "Value Post-Rebal": _fmt_inr(e["portfolio_value_post"]),
            } for e in log])
            st.dataframe(log_df, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# ═══════════════════════  TAB 7: INSIGHTS  ════════════════════════════════
# ---------------------------------------------------------------------------

def render_insights(
    metrics: dict,
    weights: dict,
    prices: pd.DataFrame,
    quality_report: dict,
    cfg: dict,
):
    st.markdown("### 🧩 Auto-Generated Portfolio Insights")

    rf = cfg.get("risk_free_rate", 0.065)
    dr = metrics["diversification_ratio"]
    var = metrics["var"]
    dd  = metrics["max_drawdown"]
    bm  = metrics["benchmark"]

    insights = []

    # Return vs benchmark
    ret_diff = metrics["annualized_return"] - bm["annualized_return"]
    if ret_diff > 0:
        insights.append((
            "📈 Return Premium",
            f"Your portfolio delivered **{metrics['annualized_return']:.1f}%** annualised return, "
            f"**{ret_diff:.1f}pp above** the NIFTY 50 benchmark ({bm['annualized_return']:.1f}%). "
            "Active selection is adding value.",
            "success",
        ))
    else:
        insights.append((
            "📉 Return Lag",
            f"Your portfolio returned **{metrics['annualized_return']:.1f}%** annualised, "
            f"**{abs(ret_diff):.1f}pp below** NIFTY 50 ({bm['annualized_return']:.1f}%). "
            "Consider revisiting the optimization objective.",
            "warning",
        ))

    # Sharpe
    if metrics["sharpe"] > 1.0:
        insights.append((
            "⭐ Strong Risk-Adjusted Return",
            f"Sharpe Ratio of **{metrics['sharpe']:.2f}** (>1.0) indicates excellent "
            "risk-adjusted performance. The portfolio earns well per unit of volatility taken.",
            "success",
        ))
    elif metrics["sharpe"] < 0:
        insights.append((
            "⚠️ Negative Sharpe Ratio",
            f"Sharpe Ratio of **{metrics['sharpe']:.2f}** means returns are below the "
            f"risk-free rate ({rf*100:.1f}%). Consider a different optimization objective.",
            "error",
        ))

    # Diversification
    insights.append((
        "🔗 Diversification",
        dr["interpretation"],
        "info",
    ))

    # Drawdown
    insights.append((
        "📉 Maximum Drawdown",
        f"The portfolio experienced a maximum drawdown of **{dd['max_drawdown_pct']:.1f}%** "
        f"(from {dd['drawdown_start']} to {dd['drawdown_trough']}). "
        + (f"Recovered in **{dd['recovery_days']} trading days**."
           if dd["recovery_days"] else "Has not yet recovered to the pre-drawdown peak."),
        "warning" if abs(dd["max_drawdown_pct"]) > 20 else "info",
    ))

    # VaR
    insights.append((
        "🎯 Daily Risk Exposure",
        var["interpretation"],
        "info",
    ))

    # Beta
    beta = metrics["current_beta"]
    if beta is not None:
        if beta > 1.3:
            insights.append((
                "⚡ High Market Sensitivity",
                f"Beta of **{beta:.2f}** means the portfolio amplifies NIFTY moves by {beta:.2f}×. "
                "In a 10% market crash, expect ~" + f"{beta*10:.0f}% portfolio decline.",
                "warning",
            ))
        elif beta < 0.7:
            insights.append((
                "🛡️ Defensive Portfolio",
                f"Beta of **{beta:.2f}** indicates the portfolio is defensive — "
                "it moves less than the market in both directions.",
                "success",
            ))

    # Data quality summary
    flags = quality_report["summary"].get("flags_raised", 0)
    if flags > 0:
        insights.append((
            "🔍 Data Quality Alert",
            f"**{flags}** stock(s) triggered data quality flags during loading. "
            "Results for flagged stocks may be less reliable. Review the sidebar Quality Report.",
            "warning",
        ))

    # Render insights
    for title, body, level in insights:
        if level == "success":
            st.success(f"**{title}**  \n{body}")
        elif level == "error":
            st.error(f"**{title}**  \n{body}")
        elif level == "warning":
            st.warning(f"**{title}**  \n{body}")
        else:
            st.info(f"**{title}**  \n{body}")

    # Top weights
    st.markdown('<div class="section-header">Portfolio Composition</div>',
                unsafe_allow_html=True)
    try:
        meta_df = get_ticker_metadata().set_index("ticker")
    except Exception:
        meta_df = pd.DataFrame()

    rows = []
    for ticker, w in sorted(weights.items(), key=lambda x: -x[1]):
        company = meta_df.loc[ticker, "company_name"] if ticker in meta_df.index else ticker
        sector  = meta_df.loc[ticker, "sector"]       if ticker in meta_df.index else "N/A"
        rows.append({
            "Ticker": ticker.replace(".NS", ""),
            "Company": company,
            "Sector": sector,
            "Weight": f"{w*100:.1f}%",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# ═══════════════════════════  MAIN APP  ═══════════════════════════════════
# ---------------------------------------------------------------------------

def main():
    cfg = _load_config()
    sel = render_sidebar(cfg)

    if sel is None:
        st.markdown("## 👈 Configure your portfolio in the sidebar")
        st.markdown(
            "Select **3–15 NIFTY 50 stocks**, choose a date range and optimization objective, "
            "then click **🚀 Run Analysis**."
        )
        st.stop()

    if not sel["run"]:
        st.markdown("## 👈 Click **🚀 Run Analysis** to begin")
        st.stop()

    # ── Load data ─────────────────────────────────────────────────────────
    with st.spinner("📡 Fetching price data from Yahoo Finance..."):
        try:
            prices, quality_report = _cached_load_data(
                tuple(sel["tickers"]), sel["start"], sel["end"]
            )
        except Exception as e:
            st.error(f"❌ Data loading failed: {e}")
            st.stop()

    # Always render the quality report in the sidebar — SOP Rule: never suppress
    render_quality_report_sidebar(quality_report)

    with st.spinner("📡 Fetching NIFTY 50 benchmark..."):
        try:
            benchmark = _cached_benchmark(sel["start"], sel["end"])
        except Exception as e:
            st.warning(f"Benchmark data unavailable: {e}")
            benchmark = pd.Series(dtype=float)

    valid_tickers = [t for t in sel["tickers"] if t in prices.columns]
    if len(valid_tickers) < cfg.get("min_stocks", 3):
        st.error("Insufficient valid price data. Try a different date range or stocks.")
        st.stop()

    prices = prices[valid_tickers].dropna(how="all")

    # ── Optimise portfolio ────────────────────────────────────────────────
    with st.spinner("🔧 Optimising portfolio (CVXPY)..."):
        try:
            returns_df = prices.pct_change().dropna().astype(float)
            obj = sel["objective"]
            mw  = sel["max_weight"]
            rf  = cfg.get("risk_free_rate", 0.065)

            if obj == "Maximum Sharpe":
                w_arr, opt_metrics = max_sharpe_portfolio(returns_df, rf, mw)
            elif obj == "Minimum Volatility":
                w_arr, opt_metrics = min_variance_portfolio(returns_df, mw)
            else:  # Target Return
                from src.optimization import target_return_portfolio
                w_arr, opt_metrics = target_return_portfolio(
                    returns_df, sel["target_return"] * 100, mw
                )
                if w_arr is None:
                    st.error(
                        f"Target return of {sel['target_return']*100:.1f}% is not achievable "
                        "with the selected stocks and weight constraints. "
                        "Try a lower target or different stocks."
                    )
                    st.stop()

            weights = opt_metrics["weights_dict"]

        except Exception as e:
            st.error(f"❌ Optimization failed: {e}")
            st.stop()

    # ── Compute risk metrics ──────────────────────────────────────────────
    with st.spinner("📐 Computing risk metrics..."):
        try:
            port_returns = pd.Series(compute_portfolio_returns(prices, weights), dtype=float)

            # Defensive benchmark handling — yfinance may return a MultiIndex
            # DataFrame for single tickers in newer versions; force to 1-D Series
            bm = benchmark
            if isinstance(bm, pd.DataFrame):
                bm = bm.iloc[:, 0]          # take first column
            bm = bm.squeeze()               # drop any remaining single-dim axes
            if not isinstance(bm, pd.Series):
                bm = pd.Series(bm, dtype=float)
            bm = bm.dropna()

            if bm.empty:
                # Fallback: zero benchmark returns so metrics still compute
                bench_series = pd.Series(
                    np.zeros(len(port_returns)), index=port_returns.index, name="^NSEI"
                )
            else:
                bench_ret = bm.pct_change().dropna()
                bench_ret = bench_ret.reindex(port_returns.index).fillna(0)
                # Guarantee 1-D even after reindex (can add a length-1 axis)
                bench_series = pd.Series(
                    np.array(bench_ret).flatten(), index=port_returns.index, name="^NSEI"
                )

            metrics = compute_all_metrics(
                portfolio_returns=port_returns,
                benchmark_returns=bench_series,
                weights=weights,
                individual_returns=returns_df,
                portfolio_value=sel["capital"],
                risk_free_rate=rf,
            )
        except Exception as e:
            st.error(f"❌ Risk metrics computation failed: {e}")
            st.stop()

    # ── Tab header ────────────────────────────────────────────────────────
    st.markdown(
        f"## 📊 Portfolio Analysis  "
        f"<span style='font-size:0.85rem;color:#94a3b8;font-weight:400'>"
        f"{sel['start']} → {sel['end']} | {len(valid_tickers)} stocks | "
        f"{sel['objective']} | Capital: {_fmt_inr(sel['capital'])}"
        f"</span>",
        unsafe_allow_html=True,
    )

    # ── 7 Tabs ────────────────────────────────────────────────────────────
    tabs = st.tabs([
        "📋 Overview",
        "🎯 Efficient Frontier",
        "⚠️ Risk Metrics",
        "🎲 Monte Carlo",
        "🔴 Stress Test",
        "♻️ Rebalancing",
        "🧩 Insights",
    ])

    with tabs[0]:
        render_overview(metrics, weights, prices, benchmark, port_returns, sel["capital"])

    with tabs[1]:
        render_frontier(returns_df, weights, opt_metrics, mw, cfg)

    with tabs[2]:
        render_risk_metrics(metrics, port_returns, sel["capital"])

    with tabs[3]:
        render_monte_carlo(
            weights, returns_df, sel["capital"],
            sel["drift_mode"], sel["user_drift"], cfg,
        )

    with tabs[4]:
        render_stress_test(weights, prices, cfg)

    with tabs[5]:
        render_rebalancing(weights, prices, sel["capital"], cfg)

    with tabs[6]:
        render_insights(metrics, weights, prices, quality_report, cfg)


if __name__ == "__main__":
    main()
