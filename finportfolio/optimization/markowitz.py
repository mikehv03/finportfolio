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

class Markowitz:
    """
    A class for performing Markowitz portfolio optimization, based on Markowitz (1952) Modern Portfolio Theory.
    
    Args:
        returns (pd.DataFrame): Asset returns with dates as index and tickers as columns.
        annualize (bool): Whether to annualize returns and covariance matrix. Defaults to False.
        periods_per_year (int): Number of periods per year for annualization. Defaults to 252.

    Raises:
        ValueError: If the covariance matrix is ill-conditioned.
        ValueError: If the covariance matrix is singular (not invertible).

    Example:
        >>> m = Markowitz(returns, annualize=False, periods_per_year=1)
        >>> m.min_variance()
        >>> m.optimal_portfolio(return_target=0.1)
        >>> m.plot_frontier(minimum_return=0.05, maximum_return=0.15)
    """
    def __init__(self, returns: pd.DataFrame, annualize: bool = False, periods_per_year: int = 252):
        self.returns = returns
        if annualize:
            self.expected_returns = returns.mean() * periods_per_year
            self.cov_matrix = returns.cov() * periods_per_year
        else:
            self.expected_returns = returns.mean()
            self.cov_matrix = returns.cov()
        self.n_assets = returns.shape[1]
        self.tickers = returns.columns.tolist()
        self.ones = np.ones(self.n_assets)
        self.annualize = annualize
        self.periods_per_year = periods_per_year

        cond_number = np.linalg.cond(self.cov_matrix.values)
        if cond_number > 1e10:
            raise ValueError("Covariance matrix is ill-conditioned.")
        try:
            self.inv_cov = np.linalg.inv(self.cov_matrix.values)
        except np.linalg.LinAlgError:
            raise ValueError("Covariance matrix is singular.")
    
    def min_variance(self) -> pd.Series:
        """
        Calculate the minimum variance portfolio weights. 

        Returns:
            pd.Series: Minimum variance portfolio weights
        
        Raises:
            ValueError: If the minimum variance portfolio is degenerate for the given inputs.
        """
        e = self.ones
        denom = e.T @ self.inv_cov @ e
        if np.isclose(denom, 0.0):
            raise ValueError("Minimum variance portfolio is degenerate for the given inputs.")
        w = (self.inv_cov @ e) / (e.T @ self.inv_cov @ e)
        return pd.Series(w, index=self.tickers)
    
    def optimal_portfolio(self, return_target: float) -> pd.Series:
        """
        Calculate the optimal portfolio weights for a given target return.

        Args:
            return_target (float): The target return for the portfolio. Must be in the same frequency as the expected returns (e.g., annualized if expected returns are annualized).

        Returns: 
            pd.Series: Optimal portfolio weights
        
        Raises:
            ValueError: If the efficient frontier is degenerate for the given inputs (i.e., if the denominator in the weight calculation is close to zero).
        """
        mu = self.expected_returns.values
        e = self.ones

        A = e.T @ self.inv_cov @ e
        B = e.T @ self.inv_cov @ mu
        C = mu.T @ self.inv_cov @ mu

        denom = A*C - B**2
        if np.isclose(denom, 0.0):
            raise ValueError("Efficient frontier is degenerate for the given inputs.")
        w = self.inv_cov @ (((C - B * return_target) / denom) * e + ((A * return_target - B) / denom) * mu)
        return pd.Series(w, index=self.tickers)
    
    def portfolio_performance(self, weights: pd.Series, rf: float = 0.0) -> pd.Series:
        """
        Calculate the expected return, risk and Sharpe ratio of a portfolio given its weights.
        
        Args:
            weights (pd.Series): The portfolio weights.
            rf (float): The risk-free rate. Must be in the same frequency as the asset returns. Default to 0.0. 
        
        Returns:
            pd.Series: A Series containing the expected return, risk and Sharpe ratio of the portfolio.

        Raises:
            ValueError: If weights do not include all asset tickers.
        """
        weights = weights.reindex(self.tickers)
        if weights.isna().any():
            raise ValueError("Weights must include all asset tickers.")
        expected_return = np.dot(weights.values, self.expected_returns.values)
        variance = weights.values.T @ self.cov_matrix.values @ weights.values
        variance = max(variance, 0)
        risk = np.sqrt(variance)
        
        if risk == 0:
            sharpe_ratio = 0
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
        """
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

    Based on Tobin (1958) Liquidity Preference as Behavior Towards Risk.
    Extends Markowitz by adding a risk-free asset.

    Args:
        returns (pd.DataFrame): Asset returns with dates as index and tickers as columns.
        rf (float): The risk-free rate. Must be in the same frequency as the asset returns (e.g., annualized if asset returns are annualized).
        annualize (bool): Whether to annualize returns and covariance matrix. Defaults to False.
        periods_per_year (int): Number of periods per year for annualization. Defaults to 252.

    Example:
        >>> t = Tobin(returns, rf=0.02, annualize=False, periods_per_year=1)
        >>> t.tangency_portfolio()
        >>> t.plot_frontier(minimum_return=0.05, maximum_return=0.15)
    """
    def __init__(self, returns: pd.DataFrame, rf: float, annualize: bool = False, periods_per_year: int = 252):
        super().__init__(returns, annualize, periods_per_year)
        self.rf = rf
    
    def tangency_portfolio(self) -> pd.Series:
        """
        Calculate the tangency portfolio weights.

        Returns:
            pd.Series: Tangency portfolio weights

        Raises:
            ValueError: If the tangency portfolio is degenerate for the given inputs (i.e., if the denominator in the weight calculation is close to zero).
        """
        mu = self.expected_returns.values
        e = self.ones

        numerator = self.inv_cov @ (mu - self.rf * e)
        denominator = e.T @ numerator
        if np.isclose(denominator, 0.0):
            raise ValueError("Tangency portfolio is degenerate for the given inputs.")

        w = numerator / denominator
        wf = 1 - np.sum(w)
        return pd.Series(np.append(w, wf), index=self.tickers + ["Risk-Free Asset"])
    
    def portfolio_performance(self, weights: pd.Series) -> pd.Series:
        """
        Calculate the expected return, risk and Sharpe ratio of a portfolio given its weights, including the risk-free asset.

        Args:
            weights (pd.Series): The portfolio weights including the risk-free asset.
        
        Returns:
            pd.Series: A Series containing the expected return, risk and Sharpe ratio of the portfolio.

        Raises:
            ValueError: If weights do not include all asset tickers.
        """
        weights = weights.reindex(self.tickers + ["Risk-Free Asset"])
        if weights.isna().any():
            raise ValueError("Weights must include all asset tickers and 'Risk-Free Asset'.")
        w_risky = weights.iloc[:-1]
        w_rf = weights.iloc[-1]

        expected_return = np.dot(w_risky.values, self.expected_returns.values) + w_rf * self.rf
        variance = w_risky.values.T @ self.cov_matrix.values @ w_risky.values
        variance = max(variance, 0)
        risk = np.sqrt(variance)
        sharpe_ratio = (expected_return - self.rf) / risk if risk > 0 else 0
        
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
            return_target (float): The target return for the portfolio. Must be in the same frequency as the expected returns (e.g., annualized if expected returns are annualized).

        Returns:
            pd.Series: Optimal portfolio weights including risk-free asset.

        Raises:
            ValueError: If the optimal portfolio is degenerate for the given inputs.
            ValueError: If the optimal portfolio is degenerate because the tangency return equals the risk-free rate.
        """
        mu = self.expected_returns.values
        e = self.ones

        numerator = self.inv_cov @ (mu - self.rf * e)
        denominator = e.T @ numerator
        if np.isclose(denominator, 0.0):
            raise ValueError("Optimal portfolio is degenerate for the given inputs.")
        gamma_tau = numerator / denominator

        mu_tau = mu @ gamma_tau
        if np.isclose(mu_tau - self.rf, 0.0):
            raise ValueError(
                "Optimal portfolio is degenerate because tangency return equals the risk-free rate."
            )
        gamma_z = (mu_tau - return_target) / (mu_tau - self.rf)
        w = (1 - gamma_z) * gamma_tau
        return pd.Series(np.append(w, gamma_z), index=self.tickers + ["Risk-Free Asset"])
    
    def cal(self, minimum_return: float, maximum_return: float, n_portfolios: int = 100) -> pd.DataFrame:
        """
        Calculate the Capital Allocation Line (CAL) for a range of target returns.

        Args:
            minimum_return (float): The minimum target return for the CAL.
            maximum_return (float): The maximum target return for the CAL.
            n_portfolios (int): The number of portfolios to calculate on the CAL. Defaults to 100.

        Returns:
            pd.DataFrame: A DataFrame containing the target returns and corresponding portfolio risks on the CAL
        """
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