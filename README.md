# 📊 Interactive Risk Analysis & Portfolio Optimization Dashboard (Indian Stocks)

> A production-grade, risk-aware portfolio analysis dashboard for Indian equity markets — built for analysts, not just academics.

---

## 📌 Table of Contents

- [Project Overview](#-project-overview)
- [Problem Statement](#-problem-statement)
- [Market Scope](#-market-scope)
- [Key Features](#-key-features)
- [Mathematical Foundation](#-mathematical-foundation)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Installation & Setup](#-installation--setup)
- [Configuration](#-configuration)
- [Module Breakdown](#-module-breakdown)
- [Risk Metrics Reference](#-risk-metrics-reference)
- [Stress Test Scenarios](#-stress-test-scenarios)
- [Data Quality & Limitations](#-data-quality--limitations)
- [Usage Guide](#-usage-guide)
- [Roadmap](#-roadmap)

---

## 🔍 Project Overview

This project is an **interactive, risk-aware portfolio analysis dashboard** for Indian equity markets. It allows users to construct portfolios using NIFTY 50 stocks, optimize allocations using Modern Portfolio Theory, and evaluate downside risk under normal and stressed market conditions.

Unlike traditional "stock prediction" projects, this focuses on **risk management, portfolio construction, and decision-making** — which is how real-world analysts and fund managers operate.

**What makes this project different:**

- Uses **exact quadratic programming** (CVXPY) for the efficient frontier — not random simulation
- Preserves **cross-stock correlation structure** in Monte Carlo via Cholesky decomposition
- Shows how **diversification collapses during stress** with side-by-side correlation heatmaps
- Includes **India-specific stress scenarios** (Demonetization, IL&FS, COVID, Rate Hike Cycle)
- Enforces **data quality checks** on Yahoo Finance `.NS` tickers before any computation
- Uses a **configurable, auditable risk-free rate** — not a hardcoded assumption

---

## 🎯 Problem Statement

> How can an investor construct a diversified Indian equity portfolio that maximizes risk-adjusted returns while controlling downside risk during volatile market conditions?

This dashboard answers that question by combining portfolio optimization theory with rigorous risk analysis and India-relevant stress testing — translating complex mathematics into actionable investor intelligence.

---

## 🇮🇳 Market Scope

| Parameter | Detail |
|---|---|
| Universe | NIFTY 50 constituent stocks |
| Benchmark | NIFTY 50 Index (`^NSEI`) |
| Market | Indian National Stock Exchange (NSE) |
| Ticker Format | Yahoo Finance `.NS` suffix (e.g., `RELIANCE.NS`) |
| Risk-Free Rate | Configurable — 91-Day T-Bill / RBI Repo Rate (see `config.yaml`) |
| Data Frequency | Daily adjusted closing prices |
| Time Horizon | 3-5 years historical; 1–3 year Monte Carlo projection |

---

## 🧠 Key Features

### 1. 📐 Interactive Portfolio Construction

Users can select multiple Indian stocks from the NIFTY 50 universe, define capital investment in ₹, and choose an optimization objective:

- **Maximum Sharpe Ratio** — best risk-adjusted return
- **Minimum Volatility** — lowest possible portfolio variance
- **Target Return** — meet a user-defined expected return at minimum risk

Constraints applied:
- No short selling (all weights ≥ 0)
- Maximum allocation cap per stock (default: 40%)
- Weights must sum to 1

---

### 2. 📈 Portfolio Risk & Return Analysis

Metrics computed for the selected portfolio vs the NIFTY 50 benchmark:

| Metric | Description |
|---|---|
| Annualized Return | Geometric mean of daily returns × 252 |
| Annualized Volatility | Std dev of daily returns × √252 |
| Sharpe Ratio | (Return − Rf) / Volatility |
| Sortino Ratio | (Return − Rf) / Downside Deviation |
| Beta vs NIFTY | Rolling 252-day Cov(portfolio, NIFTY) / Var(NIFTY) |
| Maximum Drawdown | Largest peak-to-trough decline |
| Diversification Ratio | Weighted avg individual vol / Portfolio vol |
| Recovery Time | Days taken to recover from maximum drawdown |

Visualizations:
- Portfolio allocation pie chart
- Portfolio vs benchmark cumulative return
- Rolling 60-day and 252-day volatility
- Drawdown curve

---

### 3. 🎯 Efficient Frontier (Exact QP Solution)

The efficient frontier is computed using **exact quadratic programming via CVXPY**, not random portfolio simulation. This means:

- The frontier line represents the **true set of optimal portfolios**
- Every point on the curve is a solved optimization problem at a specific target return
- Random portfolio scatter is shown *behind* the frontier as a visual reference only

Highlighted portfolios:
- ✅ User-selected portfolio
- 🔵 Minimum variance portfolio
- 🟠 Maximum Sharpe ratio portfolio

---

### 4. ⚠️ Advanced Risk Metrics

**Value at Risk (VaR) at 95% confidence:**

| Method | Description |
|---|---|
| Historical VaR | 5th percentile of historical daily return distribution |
| Parametric VaR | Assumes normal distribution: μ − 1.645σ |
| Monte Carlo VaR | 5th percentile of simulated return distribution |

**Conditional VaR (CVaR / Expected Shortfall):**
Average loss in the worst 5% of outcomes — measures tail risk beyond VaR.

---

### 5. 🎲 Monte Carlo Simulation (Correlation-Preserving)

Simulates 10,000 future portfolio paths using **Cholesky decomposition** of the covariance matrix to preserve realistic cross-stock correlations.

**Without Cholesky:** Each stock's path is simulated independently — correlations are destroyed, tail risks are underestimated.

**With Cholesky:** Correlated random shocks ensure the simulation reflects how stocks actually move together.

Outputs:
- Expected portfolio value at 1, 2, and 3 years
- Worst-case (5th percentile) and best-case (95th percentile) paths
- Probability of capital loss at each horizon
- User-selectable drift: historical mean / user-defined / zero (conservative)

---

### 6. 🔴 Stress Testing (India-Specific Scenarios)

Portfolio performance is evaluated across historically significant Indian market events:

| Scenario | Period | Key Risk Factor |
|---|---|---|
| Demonetization | Nov–Dec 2016 | Sudden liquidity shock, uniquely Indian |
| IL&FS Credit Crisis | Sep–Oct 2018 | NBFC and financial sector contagion |
| COVID-19 Crash | Feb–Mar 2020 | Global systemic shock, fastest bear market |
| FII Selloff / Rate Hike | Jan–Jun 2022 | Foreign outflows, rising global rates |

For each scenario, the dashboard shows:
- Maximum drawdown of the optimized vs equal-weight portfolio
- Recovery time (days to return to pre-crash level)
- **Correlation heatmap: normal period vs stress period** — the critical insight that diversification collapses during crashes

---

### 7. 🧩 Dynamic Insights Panel

Auto-generated human-readable insights, for example:

- *"Optimized portfolio reduced maximum drawdown by 18.4% vs equal-weight during the COVID crash"*
- *"Average pairwise correlation rose from 0.38 to 0.81 during the IL&FS crisis — diversification collapsed"*
- *"Your portfolio's Diversification Ratio of 1.43 means it is 30% less volatile than its average component"*
- *"High-beta stocks (TCS, Infosys) account for 62% of downside VaR despite 35% portfolio weight"*

---

## 🧮 Mathematical Foundation

### Sharpe Ratio
```
Sharpe = (Rp - Rf) / σp
```

### Sortino Ratio
```
Sortino = (Rp - Rf) / σd
where σd = std dev of returns below the minimum acceptable return (MAR)
```

### Beta (Rolling)
```
β = Cov(Rp, Rm) / Var(Rm)
Computed over a rolling 252-day window
```

### Diversification Ratio
```
DR = (Σ wi × σi) / σp
DR > 1 confirms diversification is reducing portfolio risk
```

### Portfolio VaR (Parametric)
```
VaR = -(μp - 1.645 × σp) × Portfolio Value
```

### Cholesky Decomposition (Monte Carlo)
```
If Σ = covariance matrix, then L = chol(Σ)
Correlated returns: R_correlated = L × Z  where Z ~ N(0,1)
```

### Efficient Frontier (QP Formulation)
```
Minimize:   wᵀΣw
Subject to: wᵀμ = target_return
            Σw = 1
            w ≥ 0
            w ≤ max_weight
```

---

## 🛠 Tech Stack

| Component | Library |
|---|---|
| Language | Python 3.10+ |
| Dashboard | Streamlit |
| Data Handling | Pandas, NumPy |
| Optimization | CVXPY (exact QP), SciPy |
| Monte Carlo | NumPy (Cholesky), SciPy |
| Visualization | Plotly (interactive), Matplotlib (static) |
| Market Data | `yfinance` with `auto_adjust=True` |
| Configuration | PyYAML (`config.yaml`) |

---

## 📁 Project Structure

```
Indian-Portfolio-Dashboard/
│
├── data/
│   ├── tickers.csv                  # NIFTY 50 ticker list with sector info
│   ├── stress_periods.json          # Stress scenario date windows
│   └── nifty_index.csv              # Cached benchmark data (optional)
│
├── src/
│   ├── data_loader.py               # Data fetching + quality checks
│   ├── risk_metrics.py              # Sharpe, Sortino, VaR, CVaR, Beta, DR
│   ├── optimization.py              # CVXPY exact frontier + weight solver
│   ├── monte_carlo.py               # Cholesky-based simulation
│   ├── stress_test.py               # Stress scenario analysis + correlation
│   └── rebalancing.py               # Rebalancing simulation (monthly/quarterly)
│
├── app.py                           # Streamlit entry point
├── config.yaml                      # Risk-free rate + global assumptions
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/Indian-Portfolio-Dashboard.git
cd Indian-Portfolio-Dashboard
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Dashboard

```bash
streamlit run app.py
```

---

## ⚙️ Configuration

All global assumptions are defined in `config.yaml` — never hardcoded in source files.

```yaml
# config.yaml

risk_free_rate: 0.065          # Annual rate — update to current RBI repo rate
benchmark_ticker: "^NSEI"      # NIFTY 50 index
data_start: "2015-01-01"       # Historical data start date
max_weight_per_stock: 0.40     # Maximum single-stock allocation
rebalancing_frequency: "Q"     # Q = quarterly, M = monthly, None = buy-and-hold
transaction_cost: 0.001        # Round-trip cost: brokerage + STT (~0.1%)
monte_carlo_paths: 10000
monte_carlo_horizon_years: 3
var_confidence: 0.95
```

**Why this matters:** Every Sharpe Ratio, Sortino Ratio, and optimization output uses the risk-free rate. A stale hardcoded value silently corrupts all results. Update `risk_free_rate` whenever RBI changes the repo rate.

---

## 🔬 Module Breakdown

### `data_loader.py`

Fetches and validates data from Yahoo Finance before any computation.

**Key checks performed:**
- Enforces `auto_adjust=True` to account for splits and bonus issues
- Flags any stock with >2% missing data in the selected period
- Forward-fills gaps only within a 2-day window (larger gaps are flagged)
- Aligns all price series to NSE trading days (removes weekends + holidays)
- Logs a data quality report visible in the sidebar

```python
# Example usage
from src.data_loader import load_portfolio_data

prices, quality_report = load_portfolio_data(
    tickers=["RELIANCE.NS", "HDFCBANK.NS", "TCS.NS"],
    start="2015-01-01",
    end="2024-12-31"
)
```

---

### `optimization.py`

Solves the mean-variance optimization problem exactly using CVXPY.

**Efficient frontier construction:**
- Iterates over a range of target return values
- Solves the QP at each point: minimize variance subject to target return, weight constraints
- Returns exact frontier points — not approximations

```python
from src.optimization import compute_efficient_frontier, max_sharpe_portfolio

frontier = compute_efficient_frontier(returns, n_points=100)
weights, metrics = max_sharpe_portfolio(returns, risk_free_rate=0.065)
```

---

### `monte_carlo.py`

Generates correlated simulation paths using Cholesky decomposition.

```python
from src.monte_carlo import run_simulation

results = run_simulation(
    weights=weights,
    returns=returns,
    n_paths=10000,
    horizon_days=756,       # 3 years
    drift="historical"      # Options: "historical", "user_defined", "zero"
)
```

**Returns:** Array of shape `(n_paths, horizon_days)` — portfolio value over time for each simulated path.

---

### `risk_metrics.py`

Computes the complete risk metric suite.

```python
from src.risk_metrics import compute_all_metrics

metrics = compute_all_metrics(
    portfolio_returns=port_returns,
    benchmark_returns=nifty_returns,
    weights=weights,
    individual_returns=stock_returns,
    risk_free_rate=0.065
)

# metrics includes: sharpe, sortino, beta (rolling), max_drawdown,
# recovery_days, var_hist, var_param, var_mc, cvar, diversification_ratio
```

---

### `stress_test.py`

Evaluates portfolio behavior during historical market crises.

```python
from src.stress_test import run_stress_analysis

results = run_stress_analysis(
    portfolio_weights=weights,
    prices=prices,
    scenario="covid_2020"   # From stress_periods.json
)

# results includes: drawdown, recovery_days,
# normal_corr_matrix, stress_corr_matrix, correlation_delta
```

The **correlation delta** is the most important output — it shows how much pairwise correlations increased during the crash, quantifying exactly how much diversification collapsed.

---

### `rebalancing.py`

Compares buy-and-hold vs periodic rebalancing (net of transaction costs).

```python
from src.rebalancing import simulate_rebalancing

bah_value, rebal_value = simulate_rebalancing(
    prices=prices,
    target_weights=weights,
    frequency="Q",              # Quarterly
    initial_capital=1000000,    # ₹10 lakhs
    transaction_cost=0.001
)
```

---

## 📊 Risk Metrics Reference

| Metric | Formula | Interpretation |
|---|---|---|
| Sharpe Ratio | (Rp − Rf) / σp | Higher = better risk-adjusted return |
| Sortino Ratio | (Rp − Rf) / σd | Higher = better downside-adjusted return |
| Max Drawdown | Min(Pt/Ppeak − 1) | Closer to 0 = less severe losses |
| Recovery Time | Days from trough to new high | Shorter = faster recovery |
| Diversification Ratio | Σwσ / σp | Higher = more diversification benefit |
| Historical VaR (95%) | 5th percentile of return dist. | Max expected daily loss 95% of days |
| CVaR (95%) | Mean of worst 5% returns | Average loss in tail scenarios |
| Rolling Beta | Cov(Rp,Rm)/Var(Rm) — 252-day | >1 = amplifies market moves |

---

## 🔴 Stress Test Scenarios

Defined in `data/stress_periods.json`:

```json
{
  "demonetization_2016": {
    "start": "2016-11-08",
    "end": "2016-12-31",
    "description": "Currency demonetization — sudden domestic liquidity shock"
  },
  "ilfs_crisis_2018": {
    "start": "2018-09-21",
    "end": "2018-10-19",
    "description": "IL&FS default — NBFC and credit market contagion"
  },
  "covid_crash_2020": {
    "start": "2020-02-19",
    "end": "2020-03-23",
    "description": "COVID-19 — fastest 30% drawdown in NSE history"
  },
  "rate_hike_fii_2022": {
    "start": "2022-01-01",
    "end": "2022-06-17",
    "description": "Global rate hike cycle — FII outflows, rising cost of capital"
  }
}
```

Each scenario reports: drawdown (%), recovery time (days), correlation matrix comparison, and optimized vs equal-weight performance delta.

---

## 🧹 Data Quality & Limitations

### Known Yahoo Finance `.NS` Issues

| Issue | How We Handle It |
|---|---|
| Corporate actions (splits, bonuses) | `auto_adjust=True` enforced on all fetches |
| NSE holidays causing missing rows | Forward-fill within 2-day window only |
| Stale or frozen prices | Flagged if same price repeats >3 consecutive days |
| >2% missing data in period | Stock flagged in sidebar quality report |
| Delisted stocks | Graceful error with user notification |

### Assumptions & Limitations

- Returns are computed from adjusted closing prices only — intraday dynamics are not captured
- Optimization assumes historical return and covariance estimates are valid forward-looking inputs — they are not perfect predictors
- Monte Carlo assumes returns follow a multivariate normal distribution — real returns exhibit skewness and fat tails
- Transaction cost model is simplified — actual costs vary by broker, order size, and market conditions
- Tax implications (LTCG, STCG, STT) are estimated, not calculated precisely — consult a tax advisor for actual tax liability

---

## 🖥️ Usage Guide

### Step 1 — Select Stocks
Use the sidebar multiselect to choose 3–15 stocks from the NIFTY 50 universe. The data quality report will appear automatically.

### Step 2 — Set Parameters
- Enter your investment capital in ₹
- Set the analysis period (minimum 3 years recommended)
- Confirm the risk-free rate in `config.yaml` is current

### Step 3 — Choose Optimization Objective
Select Maximum Sharpe, Minimum Volatility, or Target Return. Set maximum per-stock weight constraint.

### Step 4 — Explore the Dashboard
Navigate through tabs:
- **Overview** — allocation, return, risk summary
- **Efficient Frontier** — exact QP frontier with your portfolio marked
- **Risk Metrics** — VaR, CVaR, Sortino, Diversification Ratio
- **Monte Carlo** — 10,000 simulated paths with confidence bands
- **Stress Test** — historical scenario analysis + correlation breakdown
- **Rebalancing** — buy-and-hold vs rebalanced comparison
- **Insights** — auto-generated plain-language analysis

---

## 🗺 Roadmap

- [ ] Live intraday data via NSE API or Zerodha Kite Connect
- [ ] Factor exposure analysis (Market, Size, Value — Fama-French style)
- [ ] CVaR-optimized portfolio (in addition to Sharpe and min-vol)
- [ ] PDF report export of full analysis
- [ ] Multi-asset extension: include gold, bonds (G-Sec ETFs), REITs
- [ ] Portfolio Health Score — single composite metric (1–100)

---

## 📄 License

MIT License. See `LICENSE` for details.

---

## 🙏 Acknowledgements

- Yahoo Finance via `yfinance` for historical price data
- NSE India for market structure reference
- Modern Portfolio Theory — Harry Markowitz (1952)
- Conditional Value at Risk — Rockafellar & Uryasev (2000)

---

*Built with Python · Streamlit · CVXPY · Plotly*
