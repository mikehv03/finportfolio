# finportfolio
**Portfolio Theory and Asset Pricing Library**
---
A Python library for portfolio optimization and asset pricing, designed to support investment analysis and decision-making.
 
[![PyPI version](https://badge.fury.io/py/finportfolio.svg)](https://badge.fury.io/py/finportfolio)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
 
---

## Why finportfolio?

This library provides a unified and educational implementation of core concepts in modern portfolio theory and asset pricing.

Unlike fragmented tools, **finportfolio** integrates in a single interface:

- Portfolio optimization (Markowitz, Tobin)
- Asset pricing models (CAPM, APT, Gordon)
- Factor models (Fama-French)
- Performance analytics (Sharpe, Alpha, VaR, CVaR)

It is designed both for learning and practical investment analysis.

---
 
## Tutorial
 
| Language | Link |
|---|---|
| English | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mikehv03/finportfolio/blob/main/notebooks/tutorial.ipynb) |
| Español | [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mikehv03/finportfolio/blob/main/notebooks/tutorial_es.ipynb) |
 
---
 
## Installation
 
```bash
pip install finportfolio
```
 
**Requirements:** Python 3.11+, pandas, numpy, scipy, matplotlib, yfinance
 
---
 
## Quick Start
 
```python
from finportfolio.data import get_prices
from finportfolio.returns import compute_returns, summary_stats
from finportfolio.optimization import Markowitz
 
# Load data
prices = get_prices(["AAPL", "MSFT", "GOOGL"], start_date="2020-01-01", end_date="2024-12-31")
 
# Compute returns
returns = compute_returns(prices, method="simple")
 
# Optimize portfolio
m = Markowitz(returns, annualize=True, periods_per_year=252)
gmv_weights = m.min_variance()
print(gmv_weights)
```
 
---
 
## Features
 
- **Market Data** — Download adjusted closing prices and Fama-French factors
- **Return Analysis** — Simple and log returns, summary statistics, annualization
- **Markowitz Optimization** — Efficient frontier, global minimum variance portfolio
- **Tobin Separation** — Capital Allocation Line, tangency portfolio
- **CAPM** — Beta estimation, equilibrium expected returns, Security Market Line
- **Gordon Growth Model** — Intrinsic value and implied required return
- **APT** — Multifactor expected return estimation
- **Factor Models** — Single Index Model, Fama-French 3-Factor, general multifactor regression
- **Performance Metrics** — Sharpe, Treynor, Jensen's alpha, Information Ratio, Tracking Error, M², VaR, CVaR, Max Drawdown
 
---
 
## Project Structure
 
```
finportfolio/
├── data/
│   └── loader.py
├── equilibrium/
│   └── capm.py
├── factors/
│   └── models.py
├── optimization/
│   └── markowitz.py
├── performance/
│   └── metrics.py
└── returns/
    └── stats.py
```
 
---
 
## Modules
 
### `finportfolio.data`
Load historical market data from external sources.
 
| Function | Description |
|---|---|
| `get_prices(tickers, start_date, end_date)` | Download adjusted closing prices via yfinance |
| `get_ff_factors(start_date, end_date)` | Download Fama-French 3-Factor daily data from Kenneth French's library |
 
---
 
### `finportfolio.returns`
Compute and analyze asset returns.
 
| Function | Description |
|---|---|
| `compute_returns(prices, method)` | Compute simple or log returns from price data |
| `summary_stats(returns, rf)` | Mean, std, skewness, kurtosis, Sharpe ratio |
| `annualize_returns(returns, periods_per_year)` | Scale sample mean to annual frequency |
 
---
 
### `finportfolio.optimization`
Mean-variance portfolio optimization.
 
| Class | Description |
|---|---|
| `Markowitz(returns)` | Efficient frontier — `min_variance()`, `optimal_portfolio()`, `plot_frontier()` |
| `Tobin(returns, rf)` | Extends Markowitz with a risk-free asset — `tangency_portfolio()`, `cal()`, `plot_frontier()` |
 
---
 
### `finportfolio.equilibrium`
Equilibrium asset pricing models.
 
| Function | Description |
|---|---|
| `estimate_beta(asset_returns, market_returns)` | OLS beta estimation |
| `capm_expected_return(rf, beta, market_return)` | CAPM equilibrium expected return |
| `security_market_line(betas, rf, market_return)` | Construct the SML |
| `plot_security_market_line(betas, rf, market_return, ...)` | Plot SML with optional asset overlay |
| `gordon_model(D1, g, r)` | Intrinsic stock value via Gordon Growth Model |
| `gordon_model_implied_return(P, D1, g)` | Implied required return from market price |
| `apt_expected_return(rf, factor_betas, factor_premia)` | APT expected return |
 
---
 
### `finportfolio.factors`
Factor model estimation.
 
| Function | Description |
|---|---|
| `single_index_model(asset_returns, market_returns)` | Alpha, beta, R², systematic and residual variance |
| `fama_french_3factor(asset_returns, ff_factors)` | Fama-French 3-Factor Model |
| `multifactor_model(asset_returns, factors)` | General multifactor OLS regression |
| `factor_exposure_report(returns, ff_factors)` | Factor exposures for all assets in a portfolio |
 
---
 
### `finportfolio.performance`
Portfolio performance and risk metrics.
 
| Class | Description |
|---|---|
| `RiskReport(returns_portfolio, returns_benchmark, rf)` | Sharpe, Treynor, Jensen's alpha, Information Ratio, Tracking Error, M², VaR, CVaR, Max Drawdown |
 
---
 
## Development
 
```bash
git clone https://github.com/mikehv03/finportfolio.git
cd finportfolio
pip install -e ".[dev]"
```
 
---

## Docker

You can run the library without installing dependencies locally using Docker.

### Run with Docker

**Build the image:**

```bash
docker build -t finportfolio .
```

**Run the example script:**

```bash
docker run --rm finportfolio python scripts/example.py
```

**Run the tests:**

```bash
docker run --rm finportfolio pytest tests/ -v
```

### Run with Docker Compose

```bash
git clone https://github.com/mikehv03/finportfolio.git
cd finportfolio
docker compose up --build
```

This will:
- Build the Docker image with all dependencies
- Run the **32 unit tests** with pytest
- Run the **example script** demonstrating all modules

To run them separately:

```bash
docker compose up finportfolio-tests    # tests only
docker compose up finportfolio-example  # example only
```
---

## License
 
MIT License — see [LICENSE](LICENSE) for details.
 
---
 
## Author
 
**Miguel Herrera** — [GitHub](https://github.com/mikehv03)
 
---
 
## References
 
### Books
- Ang, A. (2014). *Asset Management: A Systematic Approach to Factor Investing*. Oxford University Press.
- Bodie, Z., Kane, A., & Marcus, A. J. (2020). *Investments* (12th ed.). McGraw Hill.
- Elton, E. J., Gruber, M. J., Brown, S. J., & Goetzmann, W. N. (2014). *Modern Portfolio Theory and Investment Analysis* (9th ed.). John Wiley and Sons.
- Fama, E. F. (1976). *Foundations of Finance: Portfolio Decisions and Securities Prices*. Basic Books.
 
### Acknowledgement
This library was developed with the assistance of AI tools, including **Claude** (Anthropic) and **ChatGPT** (OpenAI) for code review, documentation, and theoretical validation.
