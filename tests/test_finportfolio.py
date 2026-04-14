"""
tests/test_finportfolio.py
--------------------------
Test suite for finportfolio library.
"""

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_prices():
    """Sample price DataFrame for testing."""
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=100, freq="B")
    prices = pd.DataFrame(
        np.cumprod(1 + np.random.normal(0.001, 0.02, (100, 3)), axis=0) * 100,
        index=dates,
        columns=["AAPL", "MSFT", "GOOGL"]
    )
    return prices


@pytest.fixture
def sample_returns(sample_prices):
    """Sample returns DataFrame for testing."""
    from finportfolio.returns import compute_returns
    return compute_returns(sample_prices, method="simple")


@pytest.fixture
def sample_benchmark_returns():
    """Sample benchmark returns for testing."""
    np.random.seed(99)
    dates = pd.date_range("2020-01-02", periods=99, freq="B")
    return pd.Series(np.random.normal(0.0005, 0.015, 99), index=dates, name="SPY")


# ---------------------------------------------------------------------------
# returns/stats.py
# ---------------------------------------------------------------------------

class TestComputeReturns:

    def test_simple_returns_shape(self, sample_prices):
        from finportfolio.returns import compute_returns
        returns = compute_returns(sample_prices, method="simple")
        assert isinstance(returns, pd.DataFrame)
        assert returns.shape == (99, 3)

    def test_log_returns_shape(self, sample_prices):
        from finportfolio.returns import compute_returns
        returns = compute_returns(sample_prices, method="log")
        assert isinstance(returns, pd.DataFrame)
        assert returns.shape == (99, 3)

    def test_invalid_method_raises(self, sample_prices):
        from finportfolio.returns import compute_returns
        with pytest.raises(ValueError):
            compute_returns(sample_prices, method="invalid")

    def test_empty_dataframe_raises(self):
        from finportfolio.returns import compute_returns
        with pytest.raises(ValueError):
            compute_returns(pd.DataFrame())


class TestSummaryStats:

    def test_returns_dataframe(self, sample_returns):
        from finportfolio.returns import summary_stats
        stats = summary_stats(sample_returns, rf=0.0)
        assert isinstance(stats, pd.DataFrame)
        assert "mean" in stats.columns
        assert "std" in stats.columns
        assert "sharpe_ratio" in stats.columns

    def test_correct_shape(self, sample_returns):
        from finportfolio.returns import summary_stats
        stats = summary_stats(sample_returns, rf=0.0)
        assert stats.shape[0] == 3  # 3 assets

    def test_empty_raises(self):
        from finportfolio.returns import summary_stats
        with pytest.raises(ValueError):
            summary_stats(pd.DataFrame(), rf=0.0)


class TestAnnualizeReturns:

    def test_returns_series(self, sample_returns):
        from finportfolio.returns import annualize_returns
        ann = annualize_returns(sample_returns, periods_per_year=252)
        assert isinstance(ann, pd.Series)
        assert len(ann) == 3

    def test_invalid_periods_raises(self, sample_returns):
        from finportfolio.returns import annualize_returns
        with pytest.raises(ValueError):
            annualize_returns(sample_returns, periods_per_year=0)


# ---------------------------------------------------------------------------
# optimization/markowitz.py
# ---------------------------------------------------------------------------

class TestMarkowitz:

    def test_min_variance_weights_sum_to_one(self, sample_returns):
        from finportfolio.optimization import Markowitz
        m = Markowitz(sample_returns, annualize=False)
        weights = m.min_variance()
        assert abs(weights.sum() - 1.0) < 1e-6

    def test_min_variance_returns_series(self, sample_returns):
        from finportfolio.optimization import Markowitz
        m = Markowitz(sample_returns, annualize=False)
        weights = m.min_variance()
        assert isinstance(weights, pd.Series)
        assert list(weights.index) == ["AAPL", "MSFT", "GOOGL"]

    def test_optimal_portfolio_weights_sum_to_one(self, sample_returns):
        from finportfolio.optimization import Markowitz
        m = Markowitz(sample_returns, annualize=True, periods_per_year=252)
        mu = m.expected_returns.mean()
        weights = m.optimal_portfolio(return_target=float(mu))
        assert abs(weights.sum() - 1.0) < 1e-6

    def test_frontier_returns_dataframe(self, sample_returns):
        from finportfolio.optimization import Markowitz
        m = Markowitz(sample_returns, annualize=True, periods_per_year=252)
        frontier = m.frontier(minimum_return=0.05, maximum_return=0.30, n_portfolios=10)
        assert isinstance(frontier, pd.DataFrame)
        assert len(frontier) == 10

    def test_empty_returns_raises(self):
        from finportfolio.optimization import Markowitz
        with pytest.raises(ValueError):
            Markowitz(pd.DataFrame())


class TestTobin:

    def test_tangency_weights_sum_to_one(self, sample_returns):
        from finportfolio.optimization import Tobin
        t = Tobin(sample_returns, rf=0.02, annualize=True, periods_per_year=252)
        weights = t.tangency_portfolio()
        assert abs(weights.sum() - 1.0) < 1e-6

    def test_optimal_portfolio_includes_risk_free(self, sample_returns):
        from finportfolio.optimization import Tobin
        t = Tobin(sample_returns, rf=0.02, annualize=True, periods_per_year=252)
        weights = t.optimal_portfolio(return_target=0.10)
        assert "Risk-Free Asset" in weights.index

    def test_cal_returns_dataframe(self, sample_returns):
        from finportfolio.optimization import Tobin
        t = Tobin(sample_returns, rf=0.02, annualize=True, periods_per_year=252)
        cal = t.cal(minimum_return=0.05, maximum_return=0.30, n_portfolios=10)
        assert isinstance(cal, pd.DataFrame)
        assert len(cal) == 10


# ---------------------------------------------------------------------------
# equilibrium/capm.py
# ---------------------------------------------------------------------------

class TestEstimateBeta:

    def test_beta_is_float(self, sample_returns, sample_benchmark_returns):
        from finportfolio.equilibrium import estimate_beta
        beta = estimate_beta(sample_returns["AAPL"], sample_benchmark_returns)
        assert isinstance(beta, float)

    def test_mismatched_lengths_raises(self, sample_returns):
        from finportfolio.equilibrium import estimate_beta
        short = pd.Series([0.01, 0.02, -0.01])
        with pytest.raises(ValueError):
            estimate_beta(sample_returns["AAPL"], short)

    def test_empty_series_raises(self):
        from finportfolio.equilibrium import estimate_beta
        with pytest.raises(ValueError):
            estimate_beta(pd.Series([], dtype=float), pd.Series([], dtype=float))


class TestCapmExpectedReturn:

    def test_correct_formula(self):
        from finportfolio.equilibrium import capm_expected_return
        result = capm_expected_return(rf=0.03, beta=1.0, market_return=0.10)
        assert abs(result - 0.10) < 1e-10

    def test_risk_free_asset(self):
        from finportfolio.equilibrium import capm_expected_return
        result = capm_expected_return(rf=0.03, beta=0.0, market_return=0.10)
        assert abs(result - 0.03) < 1e-10


class TestGordonModel:

    def test_intrinsic_value(self):
        from finportfolio.equilibrium import gordon_model
        price = gordon_model(D1=2.50, g=0.04, r=0.09)
        assert abs(price - 50.0) < 1e-6

    def test_r_less_than_g_raises(self):
        from finportfolio.equilibrium import gordon_model
        with pytest.raises(ValueError):
            gordon_model(D1=2.50, g=0.09, r=0.04)

    def test_implied_return(self):
        from finportfolio.equilibrium import gordon_model_implied_return
        r = gordon_model_implied_return(P=50.0, D1=2.50, g=0.04)
        assert abs(r - 0.09) < 1e-6


# ---------------------------------------------------------------------------
# factors/models.py
# ---------------------------------------------------------------------------

class TestSingleIndexModel:

    def test_returns_correct_keys(self, sample_returns, sample_benchmark_returns):
        from finportfolio.factors import single_index_model
        result = single_index_model(sample_returns["AAPL"], sample_benchmark_returns)
        assert "alpha" in result.index
        assert "beta" in result.index
        assert "r_squared" in result.index
        assert "residual_variance" in result.index
        assert "systematic_variance" in result.index

    def test_r_squared_between_0_and_1(self, sample_returns, sample_benchmark_returns):
        from finportfolio.factors import single_index_model
        result = single_index_model(sample_returns["AAPL"], sample_benchmark_returns)
        assert 0.0 <= result["r_squared"] <= 1.0

    def test_empty_series_raises(self):
        from finportfolio.factors import single_index_model
        with pytest.raises(ValueError):
            single_index_model(pd.Series([], dtype=float), pd.Series([], dtype=float))


# ---------------------------------------------------------------------------
# performance/metrics.py
# ---------------------------------------------------------------------------

class TestRiskReport:

    def test_summary_returns_series(self, sample_returns, sample_benchmark_returns):
        from finportfolio.performance import RiskReport
        weights = pd.Series([1/3] * 3, index=["AAPL", "MSFT", "GOOGL"])
        portfolio_returns = sample_returns @ weights
        report = RiskReport(portfolio_returns, sample_benchmark_returns, rf=0.0)
        summary = report.summary()
        assert isinstance(summary, pd.Series)
        assert "Sharpe Ratio" in summary.index
        assert "Jensen's Alpha" in summary.index

    def test_mismatched_index_raises(self, sample_returns, sample_benchmark_returns):
        from finportfolio.performance import RiskReport
        weights = pd.Series([1/3] * 3, index=["AAPL", "MSFT", "GOOGL"])
        portfolio_returns = sample_returns @ weights
        short_benchmark = sample_benchmark_returns.iloc[:50]
        with pytest.raises(ValueError):
            RiskReport(portfolio_returns, short_benchmark, rf=0.0)

    def test_var_is_positive(self, sample_returns, sample_benchmark_returns):
        from finportfolio.performance import RiskReport
        weights = pd.Series([1/3] * 3, index=["AAPL", "MSFT", "GOOGL"])
        portfolio_returns = sample_returns @ weights
        report = RiskReport(portfolio_returns, sample_benchmark_returns, rf=0.0)
        var = report.var(confidence_level=0.95, method="historical")
        assert var >= 0

    def test_max_drawdown_is_negative(self, sample_returns, sample_benchmark_returns):
        from finportfolio.performance import RiskReport
        weights = pd.Series([1/3] * 3, index=["AAPL", "MSFT", "GOOGL"])
        portfolio_returns = sample_returns @ weights
        report = RiskReport(portfolio_returns, sample_benchmark_returns, rf=0.0)
        dd = report.max_drawdown()
        assert dd <= 0
