"""
finportfolio.optimization.markowitz
-----------------------------------
Portfolio optimization using Markowitz's Modern Portfolio Theory.

Classes:
- Markowitz: A class for performing Markowitz portfolio optimization.
- Tobin: Extends Markowitz by adding a risk-free asset (CAL).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings

class Markowitz:
    """
    A class for performing Markowitz portfolio optimization, based on Markowitz (1952) Modern Portfolio Theory.

    This class implements mean-variance optimization, computing the minimum variance portfolio, efficient frontier, and optimal portfolios for a given target return.

    Args:
        returns (pd.DataFrame): Asset returns with dates as index and tickers as columns, expressed in a consistent frequency (e.g., daily or monthly).
        annualize (bool): Whether to annualize returns and covariance matrix. Defaults to False.
        periods_per_year (int): Number of periods per year for annualization. Defaults to 252.

    Raises:
        TypeError: If returns is not a pandas DataFrame.
        ValueError: If returns is empty.
        TypeError: If annualize is not a bool.
        ValueError: If periods_per_year is not a positive integer.
        ValueError: If the covariance matrix is not positive semi-definite.
        ValueError: If the covariance matrix is ill-conditioned.
        ValueError: If the covariance matrix is singular (not invertible).

    Example:
        >>> from finportfolio.optimization import Markowitz
        >>> returns = pd.DataFrame({
        ...     "A": [0.01, 0.02, -0.01, 0.015],
        ...     "B": [0.005, 0.01, -0.002, 0.01]
        ... })
        >>> m = Markowitz(returns, annualize=False, periods_per_year=1)
        >>> m.min_variance()
        >>> m.optimal_portfolio(return_target=0.1)
        >>> m.plot_frontier(minimum_return=0.05, maximum_return=0.15)
        """

    def __init__(self, returns: pd.DataFrame, annualize: bool = False, periods_per_year: int = 252):
        if not isinstance(returns, pd.DataFrame):
            raise TypeError("returns must be a pandas DataFrame.")
        if returns.empty:
            raise ValueError("returns cannot be empty.")
        if not isinstance(annualize, bool):
            raise TypeError("annualize must be a bool.")
        if not isinstance(periods_per_year, int) or periods_per_year <= 0:
            raise ValueError("periods_per_year must be a positive integer.")
        
        self.returns = returns

        if annualize:
            self.expected_returns = self.returns.mean() * periods_per_year
            self.cov_matrix = self.returns.cov() * periods_per_year
        else:
            self.expected_returns = self.returns.mean()
            self.cov_matrix = self.returns.cov()

        self.n_assets = self.returns.shape[1]
        self.tickers = self.returns.columns.tolist()
        self.ones = np.ones(self.n_assets)
        self.annualize = annualize
        self.periods_per_year = periods_per_year

        eigvals = np.linalg.eigvals(self.cov_matrix.values)
        if np.any(eigvals < -1e-10):
            raise ValueError("Covariance matrix is not positive semi-definite.")
        cond_number = np.linalg.cond(self.cov_matrix.values)
        if cond_number > 1e10:
            warnings.warn("Covariance matrix is ill-conditioned.")
        try:
            self.inv_cov = np.linalg.inv(self.cov_matrix.values)
        except np.linalg.LinAlgError:
            raise ValueError("Covariance matrix is singular.")
    
    def min_variance(self) -> pd.Series:
        """
        Calculate the global minimum variance portfolio weights. 

        Returns:
            pd.Series: Global minimum variance portfolio weights
        
        Raises:
            ValueError: If the minimum variance portfolio is degenerate for the given inputs.
        """
        e = self.ones
        denom = float(e.T @ self.inv_cov @ e)

        if np.isclose(denom, 0.0):
            raise ValueError("Minimum variance portfolio is degenerate for the given inputs.")
        
        w = (self.inv_cov @ e) / denom
        return pd.Series(w, index=self.tickers)
    
    def optimal_portfolio(self, return_target: float | int | np.floating) -> pd.Series:
        """
        Calculate the optimal portfolio weights for a given target return.

        Args:
            return_target (float | int | np.floating): The target return for the portfolio. Must be in the same frequency as the expected returns (e.g., annualized if expected returns are annualized).

        Returns: 
            pd.Series: Optimal portfolio weights
        
        Raises:
            TypeError: If return_target is not a float.
            ValueError: If the efficient frontier is degenerate for the given inputs (i.e., if the denominator in the weight calculation is close to zero).
        """
        if not isinstance(return_target, (float, int, np.floating)):
            raise TypeError("return_target is not numeric.")

        mu = self.expected_returns.values
        e = self.ones

        A = float(e.T @ self.inv_cov @ e)
        B = float(e.T @ self.inv_cov @ mu)
        C = float(mu.T @ self.inv_cov @ mu)

        denom = A * C - B ** 2
        if np.isclose(denom, 0.0):
            raise ValueError("Efficient frontier is degenerate for the given inputs.")
        w = self.inv_cov @ (((C - B * return_target) / denom) * e + ((A * return_target - B) / denom) * mu)
        return pd.Series(w, index=self.tickers)
    
    def portfolio_performance(self, weights: pd.Series, rf: float = 0.0) -> pd.Series:
        """
        Calculate the expected return, risk (standard deviation) and Sharpe ratio of a portfolio given its weights.
        
        Args:
            weights (pd.Series): The portfolio weights.
            rf (float): The risk-free rate. Must be in the same frequency as the asset returns. Default to 0.0. 
        
        Returns:
            pd.Series: A Series containing the expected return, risk and Sharpe ratio of the portfolio.

        Raises:
            TypeError: If weights is not a pandas Series.
            ValueError: If weights do not sum to 1.
            ValueError: If weights do not include all asset tickers.
        """
        if not isinstance(weights, pd.Series):
            raise TypeError("weights must be a pandas Series.")
        if not np.isclose(weights.sum(), 1.0):
            raise ValueError("weights must sum to 1.")
        weights = weights.reindex(self.tickers)
        if weights.isna().any():
            raise ValueError("Weights must include all asset tickers.")
        
        expected_return = np.dot(weights.values, self.expected_returns.values)
        variance = weights.values.T @ self.cov_matrix.values @ weights.values
        variance = max(variance, 0.0)
        risk = np.sqrt(variance)
        
        if np.isclose(risk, 0.0):
            sharpe_ratio = np.nan
        else:
            sharpe_ratio = (expected_return - rf) / risk
        return pd.Series({
            "expected_return": expected_return,
            "risk": risk,
            "sharpe_ratio": sharpe_ratio
        })
    
    def frontier(self, minimum_return: float, maximum_return: float, n_portfolios: int = 100) -> pd.DataFrame:
        """
        Calculate the efficient frontier for a range of target returns.

        Args:
            minimum_return (float): The minimum target return for the frontier.
            maximum_return (float): The maximum target return for the frontier.
            n_portfolios (int): The number of portfolios to calculate on the frontier. Defaults to 100.
        
        Returns: 
            pd.DataFrame: A DataFrame containing the target returns and corresponding portfolio risks.

        Raises:
            ValueError: If n_portfolios is not a positive integer.
            ValueError: If minimum_return is greater than maximum_return.
        """
        if not isinstance(n_portfolios, int) or n_portfolios <= 0:
            raise ValueError("n_portfolios must be a positive integer.")
        if minimum_return > maximum_return:
            raise ValueError("minimum_return must be less than or equal to maximum_return.")
        
        target_returns = np.linspace(minimum_return, maximum_return, n_portfolios)
        risks = []
        for r in target_returns:
            weights = self.optimal_portfolio(r)
            performance = self.portfolio_performance(weights)
            risks.append(performance["risk"])
        return pd.DataFrame({"target_return": target_returns, "risk": risks})
    
    def plot_frontier(self, minimum_return: float, maximum_return: float, n_portfolios: int = 100) -> None:
        """
        Plot the efficient frontier.

        Args:
            minimum_return (float): The minimum target return for the frontier.
            maximum_return (float): The maximum target return for the frontier.
            n_portfolios (int): The number of portfolios to calculate on the frontier. Defaults to 100.
        """
        frontier = self.frontier(minimum_return, maximum_return, n_portfolios)
        plt.figure(figsize=(10, 6))
        plt.plot(frontier["risk"], frontier["target_return"], label="Efficient Frontier", color="blue")
        plt.xlabel("Risk (Standard Deviation)")
        plt.ylabel("Expected Return")
        plt.title("Markowitz Efficient Frontier")
        plt.legend()
        plt.grid()
        plt.show()

        
class Tobin(Markowitz):
    """
    Portfolio optimization using Tobin's separation theorem.

    Based on Tobin (1958) "Liquidity Preference as Behavior Towards Risk". Extends Markowitz by introducing a risk-free asset.

    This class computes the tangency portfolio and the Capital Allocation Line (CAL), allowing investors to optimally combine a risky portfolio with a risk-free asset.
    Under this model, all investors hold the same optimal risky portfolio and differ only in their allocation between the risky portfolio and the risk-free asset.

    Args:
        returns (pd.DataFrame): Asset returns with dates as index and tickers as columns, expressed in a consistent frequency (e.g., daily or monthly).
        rf (float): Risk-free rate expressed in the same frequency as the returns (e.g., daily if returns are daily, annual if returns are annualized).
        annualize (bool): Whether to annualize returns and covariance matrix. Defaults to False.
        periods_per_year (int): Number of periods per year for annualization. Defaults to 252.

    Raises:
        TypeError: If returns is not a pandas DataFrame.
        TypeError: If rf is not numeric.
        ValueError: If returns is empty.
        ValueError: If the covariance matrix is singular or ill-conditioned.

    Example:
        >>> import pandas as pd
        >>> from finportfolio.optimization import Tobin
        >>> returns = pd.DataFrame({
        ...     "A": [0.01, 0.02, -0.01, 0.015],
        ...     "B": [0.005, 0.01, -0.002, 0.01]
        ... })
        >>> t = Tobin(returns, rf=0.01, annualize=False, periods_per_year=1)
        >>> t.tangency_portfolio()
        >>> t.plot_frontier(minimum_return=0.0, maximum_return=0.02)
    """

    def __init__(self, returns: pd.DataFrame, rf: float, annualize: bool = False, periods_per_year: int = 252):
        super().__init__(returns, annualize, periods_per_year)
        self.rf = rf
    
    def tangency_portfolio(self) -> pd.Series:
        """
        Calculate the tangency portfolio weights for the risky assets.

        Returns:
            pd.Series: Tangency portfolio weights for the risky assets.

        Raises:
            ValueError: If the tangency portfolio is degenerate for the given inputs (i.e., if the denominator in the weight calculation is close to zero).
        """
        mu = self.expected_returns.values
        e = self.ones

        numerator = self.inv_cov @ (mu - self.rf * e)
        denominator = float(e.T @ numerator)
        if np.isclose(denominator, 0.0):
            raise ValueError("Tangency portfolio is degenerate for the given inputs.")

        w = numerator / denominator
        return pd.Series(w, index=self.tickers)
    
    def portfolio_performance(self, weights: pd.Series) -> pd.Series:
        """
        Calculate the expected return, risk, and Sharpe ratio of a portfolio, allowing either:
            - weights for risky assets only (risk-free weight assumed to be 0).
            - weights including the risk-free asset. You should put "Risk-Free Asset" as the index for the risk-free weight.

        Args:
            weights (pd.Series): Portfolio weights. It may contain only risky assets
                or risky assets plus "Risk-Free Asset".

        Returns:
            pd.Series: A Series containing the expected return, risk, and Sharpe ratio
                of the portfolio.

        Raises:
            TypeError: If weights is not a pandas Series.
            ValueError: If weights do not include all required risky asset tickers.
        """
        if not isinstance(weights, pd.Series):
            raise TypeError("weights must be a pandas Series.")

        if "Risk-Free Asset" in weights.index:
            weights = weights.reindex(self.tickers + ["Risk-Free Asset"])
            if weights.isna().any():
                raise ValueError("Weights must include all asset tickers and 'Risk-Free Asset'.")
            w_rf = float(weights["Risk-Free Asset"])
            w_risky = weights[self.tickers]
        else:
            weights = weights.reindex(self.tickers)
            if weights.isna().any():
                raise ValueError("Weights must include all asset tickers.")
            w_risky = weights
            w_rf = 0.0

        expected_return = np.dot(w_risky.values, self.expected_returns.values) + w_rf * self.rf
        variance = w_risky.values.T @ self.cov_matrix.values @ w_risky.values
        variance = max(variance, 0.0)
        risk = np.sqrt(variance)
        if np.isclose(risk,0.0):
            sharpe_ratio = np.nan
        else:
            sharpe_ratio = (expected_return - self.rf) / risk
        
        return pd.Series({
            "expected_return": expected_return,
            "risk": risk,
            "sharpe_ratio": sharpe_ratio
        })
    
    def optimal_portfolio(self, return_target: float) -> pd.Series:
        """
        Calculate the optimal portfolio weights for a given target return,
        including the risk-free asset.

        Args:
            return_target (float): Target return for the portfolio. Must be in the
                same frequency as expected returns.

        Returns:
            pd.Series: Optimal portfolio weights including the risk-free asset.

        Raises:
            TypeError: If return_target is not numeric.
            ValueError: If the optimal portfolio is degenerate for the given inputs.
            ValueError: If the tangency portfolio return equals the risk-free rate.
        """
        if not isinstance(return_target, (int, float)):
            raise TypeError("return_target must be a numeric value.")

        mu = self.expected_returns.values
        e = self.ones

        numerator = self.inv_cov @ (mu - self.rf * e)
        denominator = float(e.T @ numerator)

        if np.isclose(denominator, 0.0):
            raise ValueError("Optimal portfolio is degenerate for the given inputs.")

        tangency_weights = numerator / denominator
        tangency_return = float(mu @ tangency_weights)

        if np.isclose(tangency_return - self.rf, 0.0):
            raise ValueError(
                "Optimal portfolio is degenerate because tangency return equals the risk-free rate."
            )

        weight_tangency = (return_target - self.rf) / (tangency_return - self.rf)
        weight_rf = 1.0 - weight_tangency

        w_risky = weight_tangency * tangency_weights

        return pd.Series(np.append(w_risky, weight_rf),index=self.tickers + ["Risk-Free Asset"],)
    
    def cal(self, minimum_return: float, maximum_return: float, n_portfolios: int = 100) -> pd.DataFrame:
        """
        Calculate the Capital Allocation Line (CAL) for a range of target returns.

        Args:
            minimum_return (float): The minimum target return for the CAL.
            maximum_return (float): The maximum target return for the CAL.
            n_portfolios (int): The number of portfolios to calculate on the CAL. Defaults to 100.

        Returns:
            pd.DataFrame: A DataFrame containing the target returns and corresponding portfolio risks on the CAL

        Raises:
            ValueError: If n_portfolios is not a positive integer.
            ValueError: If minimum_return is greater than maximum_return.
        """
        if not isinstance(n_portfolios, int) or n_portfolios <= 0:
            raise ValueError("n_portfolios must be a positive integer.")
        if minimum_return > maximum_return:
            raise ValueError("minimum_return must be less than or equal to maximum_return.")
        
        target_returns = np.linspace(minimum_return, maximum_return, n_portfolios)
        risks = []
        for r in target_returns:
            weights = self.optimal_portfolio(r)
            performance = self.portfolio_performance(weights)
            risks.append(performance["risk"])
        return pd.DataFrame({"target_return": target_returns, "risk": risks})
    
    def plot_frontier(self, minimum_return: float, maximum_return: float, n_portfolios: int = 100, include_markowitz: bool = False) -> None:
        """
        Plot the Capital Allocation Line (CAL), with the option of including the Markowitz efficient frontier.

        Args:
            minimum_return (float): The minimum target return for the graph.
            maximum_return (float): The maximum target return for the graph.
            n_portfolios (int): The number of portfolios to calculate on the graph. Defaults to 100.
            include_markowitz (bool): Whether to include the Markowitz efficient frontier. Defaults to False.
        """
        cal = self.cal(minimum_return, maximum_return, n_portfolios)

        tang_w = self.tangency_portfolio()
        tang_perf = self.portfolio_performance(tang_w)

        plt.figure(figsize=(10, 6))
        plt.plot(cal["risk"], cal["target_return"], label="Capital Allocation Line", color="red")
        plt.scatter(0, self.rf, color="green", label="Risk-Free Asset", zorder=5)
        plt.scatter(tang_perf["risk"], tang_perf["expected_return"],
                    marker="*", s=200, color="orange", label="Tangency Portfolio", zorder=5)

        if include_markowitz:
            m = Markowitz(self.returns, self.annualize, self.periods_per_year)
            frontier = m.frontier(minimum_return, maximum_return, n_portfolios)
            plt.plot(frontier["risk"], frontier["target_return"], label="Markowitz Efficient Frontier", color="blue")
            plt.title("Tobin's CAL with Markowitz Efficient Frontier")
        else:
            plt.title("Tobin's Capital Allocation Line")

        plt.xlabel("Risk (Standard Deviation)")
        plt.ylabel("Expected Return")
        plt.legend()
        plt.grid()
        plt.show()