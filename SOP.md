# 🤖 AI Assistant SOP — Stock Assessment (Indian Portfolio Dashboard)

> Standard Operating Procedure for AI-assisted development, debugging, and feature expansion on this project.

---

## 1. Project Identity

| Field | Value |
|---|---|
| Project | Interactive Risk Analysis & Portfolio Optimization Dashboard |
| Market | Indian Equity (NIFTY 50 / NSE) |
| Entry Point | `app.py` (Streamlit) |
| Config File | `config.yaml` |
| Language | Python 3.10+ |
| Root Path | `d:\Data Science\Projects\Stock_Market\Stock_Assessment` |

---

## 2. Directory Map

```
Stock_Assessment/
├── data/
│   ├── tickers.csv              # NIFTY 50 universe with sector info
│   ├── stress_periods.json      # Stress scenario date windows (DO NOT hardcode dates)
│   └── nifty_index.csv          # Cached benchmark (optional)
│
├── src/
│   ├── data_loader.py           # Data fetching + quality validation
│   ├── risk_metrics.py          # All risk metric computations
│   ├── optimization.py          # CVXPY-based efficient frontier + weight solver
│   ├── monte_carlo.py           # Cholesky-based correlated simulations
│   ├── stress_test.py           # Historical scenario analysis
│   └── rebalancing.py           # Buy-and-hold vs rebalancing simulation
│
├── app.py                       # Streamlit UI entry point
├── config.yaml                  # Global assumptions (risk-free rate, etc.)
├── requirements.txt
├── SOP.md                       # This file
└── README.md
```

> **Rule:** Never hardcode assumptions. All parameters live in `config.yaml`.

---

## 3. Core Design Principles (Never Violate These)

| Principle | Why It Matters |
|---|---|
| No hardcoded risk-free rate | Sharpe/Sortino/optimization all depend on it — stale value = silent corruption |
| Use `auto_adjust=True` on all yfinance calls | Adjusts for splits/bonuses; omitting causes return calculation errors |
| Efficient frontier via CVXPY (exact QP), not random simulation | Core differentiator — never switch to random sampling |
| Monte Carlo via Cholesky decomposition | Preserves cross-stock correlations; plain simulation underestimates tail risk |
| No short selling (`w ≥ 0`) | Hard constraint — enforce in all optimization |
| Max weight per stock from `config.yaml` | Default 40%; always read from config, never hardcode |
| NSE tickers must use `.NS` suffix | e.g., `RELIANCE.NS`, `HDFCBANK.NS` |
| Benchmark is `^NSEI` (NIFTY 50 Index) | Used for Beta, rolling correlation, and benchmark comparison |

---

## 4. Module-by-Module Rules

### `data_loader.py`
- Always enforce `auto_adjust=True`
- Forward-fill missing data **only within a 2-day window** — larger gaps must be **flagged**, not silently filled
- Flag any stock with **>2% missing data** in the period
- Flag stocks where the same price repeats **>3 consecutive days** (stale/frozen data)
- Log a data quality sidebar report — never suppress warnings
- Align all price series to NSE trading days

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
- Use **CVXPY** for all efficient frontier and weight computations
- Constraints: `w ≥ 0`, `Σw = 1`, `w ≤ max_weight_per_stock` (from config)
- Frontier computed by iterating over target returns and solving QP at each point
- Random portfolio scatter is shown **behind** the frontier as visual reference only — never replace the frontier with it

```python
from src.optimization import compute_efficient_frontier, max_sharpe_portfolio

frontier = compute_efficient_frontier(returns, n_points=100)
weights, metrics = max_sharpe_portfolio(returns, risk_free_rate=0.065)
```

---

### `monte_carlo.py`
- Use **Cholesky decomposition** of the covariance matrix for correlated random shocks
- Default: `n_paths = 10,000`, `horizon = 3 years` (756 trading days) — from `config.yaml`
- Support three drift modes: `"historical"`, `"user_defined"`, `"zero"` (conservative)
- Output shape: `(n_paths, horizon_days)` — portfolio value over time per path
- Report: Expected value at 1/2/3 years, 5th/95th percentile bounds, probability of loss

```python
from src.monte_carlo import run_simulation

results = run_simulation(
    weights=weights,
    returns=returns,
    n_paths=10000,
    horizon_days=756,       # 3 years
    drift="historical"
)
```

---

### `risk_metrics.py`
- Always compute: Sharpe, Sortino, Beta (rolling 252-day), Max Drawdown, Recovery Time, Diversification Ratio
- Compute **three VaR methods**: Historical (5th percentile), Parametric (μ − 1.645σ), Monte Carlo (5th percentile of simulated paths)
- CVaR = mean of worst 5% returns
- Diversification Ratio: `DR = (Σ wi × σi) / σp`
- Beta: `Cov(Rp, Rm) / Var(Rm)` on a rolling 252-day window

```python
from src.risk_metrics import compute_all_metrics

metrics = compute_all_metrics(
    portfolio_returns=port_returns,
    benchmark_returns=nifty_returns,
    weights=weights,
    individual_returns=stock_returns,
    risk_free_rate=0.065
)
# Returns: sharpe, sortino, beta, max_drawdown, recovery_days,
#          var_hist, var_param, var_mc, cvar, diversification_ratio
```

---

### `stress_test.py`
- Scenarios defined in `data/stress_periods.json` — **never hardcode date ranges**
- For each scenario compute: drawdown (%), recovery days, normal vs stress correlation matrix
- **Correlation delta** (stress corr − normal corr) is the key output — always include it
- Compare optimized portfolio vs equal-weight portfolio during the scenario

```python
from src.stress_test import run_stress_analysis

results = run_stress_analysis(
    portfolio_weights=weights,
    prices=prices,
    scenario="covid_2020"
)
# Returns: drawdown, recovery_days, normal_corr_matrix, stress_corr_matrix, correlation_delta
```

---

### `rebalancing.py`
- Compare buy-and-hold vs periodic rebalancing (monthly/quarterly)
- Read frequency from `config.yaml` (`rebalancing_frequency`)
- Apply `transaction_cost` (from config) at every rebalancing event, round-trip

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

## 5. Streamlit Dashboard Tab Structure

| Tab | Contents |
|---|---|
| **Overview** | Allocation pie chart, return/risk summary, benchmark comparison |
| **Efficient Frontier** | Exact QP frontier + user portfolio + min-var + max-Sharpe highlighted |
| **Risk Metrics** | VaR (3 methods), CVaR, Sortino, Diversification Ratio |
| **Monte Carlo** | 10,000 simulated paths with confidence bands, probability of loss |
| **Stress Test** | Historical scenario drawdowns + normal vs stress correlation heatmaps |
| **Rebalancing** | Buy-and-hold vs rebalanced performance, net of transaction costs |
| **Insights** | Auto-generated plain-English portfolio interpretation |

---

## 6. `config.yaml` Reference

```yaml
risk_free_rate: 0.065           # Annual — update to current RBI repo rate
benchmark_ticker: "^NSEI"       # NIFTY 50 index
data_start: "2015-01-01"        # Historical data start date
max_weight_per_stock: 0.40      # Maximum single-stock allocation
rebalancing_frequency: "Q"      # Q = quarterly, M = monthly, None = buy-and-hold
transaction_cost: 0.001         # Round-trip cost: brokerage + STT (~0.1%)
monte_carlo_paths: 10000
monte_carlo_horizon_years: 3
var_confidence: 0.95
```

> ⚠️ **Always remind the user to update `risk_free_rate` when RBI changes the repo rate.** A stale value silently corrupts all Sharpe/Sortino outputs.

---

## 7. Known Data Quality Issues

| Issue | Handling |
|---|---|
| Corporate actions (splits/bonuses) | `auto_adjust=True` — never omit |
| NSE holidays → missing rows | Forward-fill ≤ 2-day gaps only; flag larger gaps |
| Stale/frozen prices | Flag if same price repeats >3 consecutive days |
| >2% missing data in period | Flag stock in sidebar quality report |
| Delisted stocks | Raise graceful error with user notification |

---

## 8. Mathematical Formulas (Quick Reference)

```
Sharpe Ratio        = (Rp - Rf) / σp
Sortino Ratio       = (Rp - Rf) / σd        # σd = downside std dev below MAR
Rolling Beta        = Cov(Rp, Rm) / Var(Rm)  # 252-day rolling window
Diversif. Ratio     = Σ(wi × σi) / σp        # >1 confirms diversification benefit
Parametric VaR      = -(μp - 1.645 × σp) × Portfolio Value
Cholesky MC         = L × Z  where L = chol(Σ), Z ~ N(0,1)

QP Efficient Frontier:
  Minimize:   wᵀΣw
  Subject to: wᵀμ = target_return
              Σw = 1
              w ≥ 0
              w ≤ max_weight_per_stock
```

---

## 9. Roadmap (Planned — Not Yet Implemented)

- [ ] Live intraday data via NSE API / Zerodha Kite Connect
- [ ] Factor exposure analysis (Fama-French style)
- [ ] CVaR-optimized portfolio (alongside Sharpe and min-vol)
- [ ] PDF export of full analysis
- [ ] Multi-asset extension (gold, G-Sec ETFs, REITs)
- [ ] Portfolio Health Score (1–100 composite)

> When implementing roadmap items, integrate with the existing module structure and read all parameters from `config.yaml`.

---

## 10. AI Behavior Rules for This Project

1. **Read `config.yaml` first** — never assume any parameter value (risk-free rate, max weight, MC settings).
2. **Never replace CVXPY** with random simulation for the efficient frontier.
3. **Never remove Cholesky decomposition** from Monte Carlo — it's a core differentiator.
4. **Always validate data quality** before computation using checks in `data_loader.py`.
5. **Read scenarios from `stress_periods.json`** — do not hardcode date ranges.
6. **Keep financial math consistent** with the formulas in Section 8.
7. **Maintain the sidebar data quality report** — never suppress it.
8. **Run `streamlit run app.py`** to verify the dashboard renders after any change.
9. **New features** → create a new module in `src/` (load config → compute → return results for Streamlit).
10. **Start debugging at `data_loader.py`** — most errors originate from upstream data quality issues.

---

*SOP generated from `README.md` · Stock_Assessment Project · 2026-02-23*
