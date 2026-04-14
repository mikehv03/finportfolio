"""
finportfolio.returns.stats
--------------------------
Functions for computing and analyzing asset returns

Functions:
- compute_returns: Compute simple or logarithmic returns from price data
- summary_stats: Compute summary statistics for returns
- annualize_returns: Annualize returns from periodic returns data
"""

import numpy as np
import pandas as pd


def compute_returns(prices: pd.DataFrame, method: str = "simple") -> pd.DataFrame:
    """
    Compute returns from price data
    
    Args: 
        prices (pd.DataFrame): DataFrame of asset prices with dates as index and assets as columns
        method (str): "simple" for arithmetic returns, "log" for logarithmic returns

    Returns: 
        pd.DataFrame: Returns of each asset

    Raises:
        TypeError: If prices is not a pandas DataFrame
        ValueError: If prices DataFrame is empty
        ValueError: If log returns are requested but prices contain non-positive values
        ValueError: If method is not "simple" or "log"
    """
    if not isinstance(prices, pd.DataFrame):
        raise TypeError("prices must be a pandas DataFrame.")
    if prices.empty:
        raise ValueError("prices DataFrame cannot be empty.")
    method = method.lower()

    if method == "simple":
        return prices.pct_change().dropna()
    elif method == "log":
        if (prices <= 0).any().any():
            raise ValueError("Log returns require strictly positive prices.")
        return np.log(prices / prices.shift(1)).dropna()
    else: 
        raise ValueError("Method must be 'simple' or 'log'")  


def summary_stats(returns: pd.DataFrame, rf: float) -> pd.DataFrame:
    """
    Compute summary statistics for returns

    Args:
        returns (pd.DataFrame): DataFrame of asset returns with dates as index and assets as columns
        rf (float): Risk-free rate for Sharpe ratio calculation. It must be expressed at the same frequency as returns.

    Returns:
        pd.DataFrame: Summary statistics (mean, std, skewness, kurtosis, min, max, Sharpe ratio) for each asset
    
    Raises:
        TypeError: If returns is not a pandas DataFrame
        ValueError: If returns DataFrame is empty

    Notes:
        Sharpe ratio is calculated as (mean - rf) / std. It is set to NaN for assets with near-zero volatility, since the ratio is not defined when the standard deviation is zero.
    """
    if not isinstance(returns, pd.DataFrame):
        raise TypeError("returns must be a pandas DataFrame.")
    if returns.empty:
        raise ValueError("returns DataFrame cannot be empty.")

    std = returns.std(ddof=1)
    mean = returns.mean()
    sharpe = (mean - rf) / std
    sharpe = sharpe.mask(np.isclose(std, 0), np.nan)

    stats = pd.DataFrame({
        "mean": mean,
        "std": std,
        "skewness": returns.skew(),
        "excess_kurtosis": returns.kurt(),
        "min": returns.min(),
        "max": returns.max(),
        "sharpe_ratio": sharpe
    })
    return stats 


def annualize_returns(returns: pd.DataFrame, periods_per_year: int = 252, method: str = "simple") -> pd.Series:
    """
    This function annualizes the expected return by scaling the sample mean of periodic returns. 
    It does not compute the realized compounded return.

    Args:
        returns (pd.DataFrame): DataFrame of asset returns with dates as index and assets as columns
        periods_per_year (int): Number of periods in a year (default is 252 for daily returns)
        method (str): "simple" for arithmetic returns, "log" for logarithmic returns

    Returns:
        pd.Series: Annualized returns for each asset

    Raises:
        TypeError: If returns is not a pandas DataFrame
        ValueError: If returns DataFrame is empty
        ValueError: If periods_per_year is not a positive integer
        ValueError: If method is not "simple" or "log"    
    """
    if not isinstance(returns, pd.DataFrame):
        raise TypeError("returns must be a pandas DataFrame.")
    if returns.empty:
        raise ValueError("returns DataFrame cannot be empty.")
    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be a positive integer.")
    method = method.lower()
    
    if method == "simple":
        return (1 + returns.mean()) ** periods_per_year - 1
    elif method == "log":
        return np.exp(returns.mean() * periods_per_year) - 1
    else:
        raise ValueError("Method must be 'simple' or 'log'")