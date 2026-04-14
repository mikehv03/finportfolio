"""
finportfolio.performance.metrics
-----------------------------------
Portfolio performance evaluation and risk metrics.

Classes:
- RiskReport: A class for computing portfolio performance, risk-adjusted metrics, and downside risk measures such as VaR, CVaR, and drawdowns.
"""


import numpy as np
import pandas as pd
from scipy import stats


class RiskReport:
    """
    Compute classical portfolio performance and risk-adjusted metrics.

    This class compares a portfolio returns series against the returns of a benchmark and a risk-free rate to calculate performance and risk metrics such as:
    - Sharpe ratio
    - Treynor ratio
    - Jensen's alpha
    - Tracking error
    - Information ratio
    - Value at Risk (VaR)
    - Conditional Value at Risk (CVaR)
    - Maximum drawdown
    - M-squared

    Args:
        returns_portfolio (pd.Series): A pandas Series of portfolio returns.
        returns_benchmark (pd.Series): A pandas Series of benchmark returns, aligned with the portfolio returns.
        rf (float): The risk-free rate, expressed as a decimal and in the same frequency as the returns.

    Raises:
        TypeError: If returns_portfolio or returns_benchmark is not a pandas Series.
        TypeError: If rf is not a numeric value.
        ValueError: If returns_portfolio or returns_benchmark is empty.
        ValueError: If returns_portfolio or returns_benchmark contains null values.
        ValueError: If returns_portfolio and returns_benchmark do not have the same index or length.
        ValueError: If returns_benchmark has zero variance (which would make beta calculation impossible).

    Example:
        >>> from finportfolio.performance import RiskReport
        >>> portfolio_returns = pd.Series([0.01, 0.02, -0.005, 0.015])
        >>> benchmark_returns = pd.Series([0.005, 0.01, -0.002, 0.01])
        >>> rf = 0.01
        >>> report = RiskReport(portfolio_returns, benchmark_returns, rf)
        >>> report.summary()
    """

    def __init__(self, returns_portfolio: pd.Series, returns_benchmark: pd.Series, rf: float):
        if not isinstance(returns_portfolio, pd.Series):
            raise TypeError("Portfolio returns must be a pandas Series.")
        if not isinstance(returns_benchmark, pd.Series):
            raise TypeError("Benchmark returns must be a pandas Series.")
        if not isinstance(rf, (int, float)):
            raise TypeError("Risk-free rate must be a numeric value.")
        if returns_portfolio.empty:
            raise ValueError("returns_portfolio cannot be empty.")
        if returns_benchmark.empty:
            raise ValueError("returns_benchmark cannot be empty.")
        if returns_portfolio.isnull().any() or returns_benchmark.isnull().any():
            raise ValueError("Returns series cannot contain null values.")
        if not returns_portfolio.index.equals(returns_benchmark.index):
            raise ValueError("Portfolio and benchmark returns must have the same index.")
        if np.isclose(returns_benchmark.var(ddof=1), 0.0):
            raise ValueError("Benchmark returns must have non-zero variance for beta calculation.")

        
        self.returns_portfolio = returns_portfolio
        self.returns_benchmark = returns_benchmark
        self.rf = rf

        X = np.column_stack([np.ones(len(self.returns_benchmark)), self.returns_benchmark.values])
        coeffs = np.linalg.lstsq(X, self.returns_portfolio.values, rcond=None)[0]
        self._beta = float(coeffs[1])

    def sharpe_ratio(self) -> float:
        """
        Calculate the Sharpe ratio of the portfolio.
        
        Returns:
            float: The Sharpe ratio of the portfolio. If it is undefined, it returns NaN.
        """
        excess_return = self.returns_portfolio.mean() - self.rf
        volatility = self.returns_portfolio.std(ddof=1)
        if np.isclose(volatility, 0.0):
            return np.nan
        return float(excess_return / volatility)
    
    def treynor_ratio(self) -> float:
        """
        Calculate the Treynor ratio of the portfolio.
        
        Returns:
            float: The Treynor ratio of the portfolio. If it is undefined, it returns NaN.
        """
        excess_return = self.returns_portfolio.mean() - self.rf
        if np.isclose(self._beta, 0.0):
            return np.nan
        return float(excess_return / self._beta)
    
    def jensen_alpha(self) -> float:
        """
        Calculate Jensen's alpha of the portfolio.
        
        Returns:
            float: Jensen's alpha of the portfolio.
        """
        return float(self.returns_portfolio.mean() - (self.rf + self._beta * (self.returns_benchmark.mean() - self.rf)))
    
    def max_drawdown(self) -> float:
        """
        Calculate the maximum drawdown of the portfolio.
        
        Returns:
            float: The maximum drawdown of the portfolio.

        Note: Assumes simple (arithmetic) returns. For log returns, convert to simple returns before computing drawdown.
        """
        cumulative_returns = (1 + self.returns_portfolio).cumprod()
        running_max = cumulative_returns.cummax()
        drawdown = (cumulative_returns/running_max) - 1
        return float(drawdown.min())
    
    def var(self, confidence_level: float = 0.95, method: str = "historical") -> float:
        """
        Calculate the Value at Risk (VaR) of the portfolio at a given confidence level.

        Args:
            confidence_level (float): The confidence level for VaR calculation (default 0.95).
            method (str): 'historical' or 'parametric' (default 'historical').

        Returns:
            float: The Value at Risk of the portfolio.

        Raises:
            ValueError: If confidence_level is not between 0 and 1.
            ValueError: If method is not 'historical' or 'parametric'.

        Notes:
            The parametric method assumes normally distributed returns.
        """
        if not (0 < confidence_level < 1):
            raise ValueError("confidence_level must be between 0 and 1.")
        
        method = method.lower()
        if method == "historical":
            return float(-self.returns_portfolio.quantile(1 - confidence_level))
        elif method == "parametric":
            mean = self.returns_portfolio.mean()
            std = self.returns_portfolio.std(ddof=1)

            if np.isclose(std, 0.0):
                return float(-mean)
                        
            z_score = stats.norm.ppf(1 - confidence_level)
            return float(-(mean + z_score * std))
        else:
            raise ValueError("Method must be 'historical' or 'parametric'.")

    def cvar(self, confidence_level: float = 0.95) -> float:
        """
        Calculate the historical Conditional Value at Risk (CVaR) of the portfolio.

        Args:
            confidence_level (float): Confidence level (default 0.95).

        Returns:
            float: The CVaR of the portfolio.

        Raises:
            ValueError: If confidence_level is not between 0 and 1.
        """
        if not (0 < confidence_level < 1):
            raise ValueError("confidence_level must be between 0 and 1.")

        var_threshold = self.returns_portfolio.quantile(1 - confidence_level)
        tail = self.returns_portfolio[self.returns_portfolio <= var_threshold]

        if tail.empty:
            return float(-var_threshold)

        return float(-tail.mean()) 
    
    def tracking_error(self) -> float:
        """
        Calculate the tracking error of the portfolio.

        Returns:
            float: The tracking error of the portfolio.
        """
        return float((self.returns_portfolio - self.returns_benchmark).std(ddof=1))
    
    def information_ratio(self) -> float:
        """
        Calculate the information ratio of the portfolio.

        Returns:
            float: The information ratio of the portfolio. If it is undefined, it returns NaN.
        """
        active_return = (self.returns_portfolio - self.returns_benchmark).mean()
        tracking_error = self.tracking_error()
        if np.isclose(tracking_error, 0.0):
            return np.nan
        return float(active_return / tracking_error)
    
    def m_squared(self) -> float:
        """
        Calculate the Modigliani-Modigliani (M^2) measure of the portfolio.

        Returns:
            float: The M^2 measure of the portfolio.
        """
        sharpe = self.sharpe_ratio()
        if np.isnan(sharpe):
            return np.nan
        benchmark_std = self.returns_benchmark.std(ddof=1)
        return float(sharpe * benchmark_std + self.rf)
    
    def summary(self) -> pd.Series:
        """
        Generate a summary of all performance metrics.

        Returns:
            pd.Series: A Series containing all calculated performance metrics.
        """
        return pd.Series({
            "Sharpe Ratio": self.sharpe_ratio(),
            "Treynor Ratio": self.treynor_ratio(),
            "Jensen's Alpha": self.jensen_alpha(),
            "Beta": self._beta,
            "Max Drawdown": self.max_drawdown(),
            "VaR (95%)": self.var(confidence_level=0.95),
            "CVaR (95%)": self.cvar(confidence_level=0.95),
            "Tracking Error": self.tracking_error(),
            "Information Ratio": self.information_ratio(),
            "M^2 Measure": self.m_squared()
        })