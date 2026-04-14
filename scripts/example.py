"""
scripts/example.py
------------------
Example script demonstrating finportfolio capabilities.
Runs end-to-end without a notebook — suitable for Docker.
"""

import pandas as pd
import numpy as np

print("=" * 60)
print("  finportfolio — Example Script")
print("=" * 60)

# ---------------------------------------------------------------------------
# 1. Generate synthetic data (no internet required)
# ---------------------------------------------------------------------------
print("\n[1] Generating synthetic price data...")

np.random.seed(42)
dates = pd.date_range("2020-01-01", periods=500, freq="B")
prices = pd.DataFrame(
    np.cumprod(1 + np.random.normal(0.0008, 0.018, (500, 4)), axis=0) * 100,
    index=dates,
    columns=["AAPL", "MSFT", "GOOGL", "NVDA"]
)
benchmark_prices = pd.DataFrame(
    np.cumprod(1 + np.random.normal(0.0005, 0.012, (500, 1)), axis=0) * 100,
    index=dates,
    columns=["SPY"]
)
print(f"  Price data shape: {prices.shape}")

# ---------------------------------------------------------------------------
# 2. Compute returns
# ---------------------------------------------------------------------------
print("\n[2] Computing returns...")

from finportfolio.returns import compute_returns, summary_stats, annualize_returns

returns = compute_returns(prices, method="simple")
benchmark_returns = compute_returns(benchmark_prices, method="simple")

stats = summary_stats(returns, rf=0.0)
print("\n  Summary Statistics (daily):")
print(stats[["mean", "std", "sharpe_ratio"]].to_string())

ann = annualize_returns(returns, periods_per_year=252)
print("\n  Annualized Expected Returns:")
for ticker, r in ann.items():
    print(f"    {ticker}: {r:.2%}")

# ---------------------------------------------------------------------------
# 3. Markowitz optimization
# ---------------------------------------------------------------------------
print("\n[3] Markowitz Optimization...")

from finportfolio.optimization import Markowitz, Tobin

m = Markowitz(returns, annualize=True, periods_per_year=252)
gmv_weights = m.min_variance()
gmv_perf = m.portfolio_performance(gmv_weights, rf=0.05)

print("\n  Global Minimum Variance Portfolio:")
for ticker, w in gmv_weights.items():
    print(f"    {ticker}: {w:.2%}")
print(f"\n  Expected Return: {gmv_perf['expected_return']:.2%}")
print(f"  Risk (std):      {gmv_perf['risk']:.2%}")
print(f"  Sharpe Ratio:    {gmv_perf['sharpe_ratio']:.4f}")

# ---------------------------------------------------------------------------
# 4. Tobin — Tangency Portfolio
# ---------------------------------------------------------------------------
print("\n[4] Tobin — Tangency Portfolio...")

t = Tobin(returns, rf=0.05, annualize=True, periods_per_year=252)
tang_weights = t.tangency_portfolio()
tang_perf = t.portfolio_performance(tang_weights)

print("\n  Tangency Portfolio Weights:")
for ticker, w in tang_weights.items():
    print(f"    {ticker}: {w:.2%}")
print(f"\n  Expected Return: {tang_perf['expected_return']:.2%}")
print(f"  Risk (std):      {tang_perf['risk']:.2%}")
print(f"  Sharpe Ratio:    {tang_perf['sharpe_ratio']:.4f}")

# ---------------------------------------------------------------------------
# 5. CAPM
# ---------------------------------------------------------------------------
print("\n[5] CAPM — Beta Estimation...")

from finportfolio.equilibrium import estimate_beta, capm_expected_return

spy_returns = benchmark_returns["SPY"]
rf_daily = 0.05 / 252
market_return = annualize_returns(benchmark_returns, periods_per_year=252)["SPY"]

betas = {}
for ticker in prices.columns:
    betas[ticker] = estimate_beta(
        returns[ticker] - rf_daily,
        spy_returns - rf_daily
    )

print("\n  Estimated Betas vs SPY:")
for ticker, beta in betas.items():
    capm_r = capm_expected_return(0.05, beta, market_return)
    print(f"    {ticker}: beta={beta:.4f}, CAPM E[R]={capm_r:.2%}")

# ---------------------------------------------------------------------------
# 6. Gordon Growth Model
# ---------------------------------------------------------------------------
print("\n[6] Gordon Growth Model...")

from finportfolio.equilibrium import gordon_model, gordon_model_implied_return

price = gordon_model(D1=2.50, g=0.04, r=0.09)
implied_r = gordon_model_implied_return(P=45.0, D1=2.50, g=0.04)

print(f"\n  Intrinsic value (D1=2.50, g=4%, r=9%): ${price:.2f}")
print(f"  Implied return at P=$45.00:             {implied_r:.2%}")

# ---------------------------------------------------------------------------
# 7. Single Index Model
# ---------------------------------------------------------------------------
print("\n[7] Single Index Model — AAPL vs SPY...")

from finportfolio.factors import single_index_model

sim = single_index_model(returns["AAPL"], spy_returns)
print(f"\n  Alpha:               {sim['alpha']:.6f}")
print(f"  Beta:                {sim['beta']:.4f}")
print(f"  R-squared:           {sim['r_squared']:.4f}")
print(f"  Residual Variance:   {sim['residual_variance']:.8f}")
print(f"  Systematic Variance: {sim['systematic_variance']:.8f}")

# ---------------------------------------------------------------------------
# 8. Portfolio Performance Metrics
# ---------------------------------------------------------------------------
print("\n[8] Portfolio Performance Metrics...")

from finportfolio.performance import RiskReport

weights = pd.Series([0.25] * 4, index=["AAPL", "MSFT", "GOOGL", "NVDA"])
portfolio_returns = returns @ weights

report = RiskReport(portfolio_returns, spy_returns, rf=rf_daily)
summary = report.summary()

print("\n  Equal-Weight Portfolio vs SPY:")
for metric, value in summary.items():
    print(f"    {metric}: {value:.6f}")

print("\n" + "=" * 60)
print("  Example script completed successfully.")
print("=" * 60)
