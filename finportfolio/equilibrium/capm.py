"""
finportfolio.equilibrium.capm
-----------------------------
Capital Asset Pricing Model (CAPM) and related equilibrium models.

Functions:
- estimate_beta: Estimate the beta of an asset given its returns and market returns.
- capm_expected_return: Calculate the expected return of an asset using the CAPM formula.
- security_market_line: Calculate the Security Market Line (SML) for a range of betas.
- gordon_model: Calculate the intrinsic value of a stock using the Gordon Growth Model.
- apt_expected_return: Calculate the expected return of an asset using the Arbitrage Pricing Theory (APT).
"""

import numpy as np
import pandas as pd


def estimate_beta(asset_returns: pd.Series, market_returns: pd.Series) -> float:
    """
    Estimate the beta of an asset using OLS regression.
    Beta is estimated as the slope coefficient in the OLS regression:
        asset_returns = alpha + beta * market_returns + error

    Args:
        asset_returns (pd.Series): Series of returns for the asset.
        market_returns (pd.Series): Series of returns for the market.

    Returns:
        float: The beta of the asset.

    Raises:
        TypeError: If asset_returns or market_returns is not a pandas Series.
        ValueError: If the input series are empty.
        ValueError: If the input series have different lengths.
        ValueError: If there are fewer than 2 observations in the input series.
        ValueError: If the variance of market_returns is zero.
    """
    if not isinstance(asset_returns, pd.Series):
        raise TypeError("asset_returns must be a pandas Series.")
    if not isinstance(market_returns, pd.Series):
        raise TypeError("market_returns must be a pandas Series.")
    if asset_returns.empty:
        raise ValueError("asset_returns cannot be empty.")
    if market_returns.empty:
        raise ValueError("market_returns cannot be empty.")
    if len(asset_returns) != len(market_returns):
        raise ValueError("asset_returns and market_returns must have the same length.")
    if len(asset_returns) < 2:
        raise ValueError("At least 2 observations are required to estimate beta.")
    data = pd.concat([asset_returns, market_returns], axis=1).dropna()
    if len(data) < 2:
        raise ValueError("At least two valid paired observations are required to estimate beta.")

    y = data.iloc[:, 0].to_numpy(dtype=float)
    x = data.iloc[:, 1].to_numpy(dtype=float)
    
    if np.isclose(np.var(x, ddof=1), 0.0):
        raise ValueError("market_returns have zero variance; beta is undefined.")
    X = np.column_stack([np.ones(len(x)), x])
    coefficients = np.linalg.lstsq(X, y, rcond=None)[0]

    return float(coefficients[1])


def capm_expected_return(rf: float, beta: float, market_return: float) -> float:
    """
    Calculate the expected return of an asset using the CAPM formula

    Args:
        rf (float): The risk-free rate.
        beta (float): The beta of the asset.
        market_return (float): The expected return of the market.

    Returns:
        float: The expected return of the asset.
    """
    return rf + beta * (market_return - rf)


def security_market_line(betas: np.ndarray | list | pd.Series, rf: float, market_return: float) -> pd.DataFrame:
    """
    Construct the Security Market Line (SML), representing the CAPM equilibrium relationship between systematic risk (beta) and expected return.

    Args:
        betas (array-like): Array-like of beta values.
        rf (float): The risk-free rate.
        market_return (float): The expected return of the market.

    Returns:
        pd.DataFrame: A DataFrame containing betas and corresponding expected returns.

    Raises:
        TypeError: If betas is not array-like.
        ValueError: If betas is empty.
    """
    if not isinstance(betas, (np.ndarray, list, pd.Series)):
        raise TypeError("betas must be array-like (numpy array, list, or pandas Series).")
    
    betas = np.asarray(betas, dtype=float)
    if betas.size == 0:
        raise ValueError("betas cannot be empty.")
    
    expected_returns = rf + betas * (market_return - rf)

    return pd.DataFrame({"beta": betas, "expected_return": expected_returns})


def gordon_model(D1: float, g: float, r: float) -> float:
    """
    Calculate the intrinsic value of a stock using the Gordon Growth Model.

    Args:
        D1 (float): The expected dividend in the next period.
        g (float): The constant growth rate of dividends.
        r (float): The required rate of return.

    Returns:
        float: The intrinsic value of the stock.
    
    Raises: 
        ValueError: If the required rate of return is not greater than the growth rate.
    """
    if r <= g:
        raise ValueError("Required rate of return must be greater than growth rate.")
    
    return D1 / (r - g)


def apt_expected_return(rf: float, factor_betas: pd.Series, factor_premia: pd.Series) -> float:
    """
    Calculate the expected return of an asset using the Arbitrage Pricing Theory (APT).

    Args:
        rf (float): The risk-free rate.
        factor_betas (pd.Series): Factor betas of the asset (from multifactor_model).
        factor_premia (pd.Series): Expected risk premium for each factor.

    Returns:
        float: The expected return of the asset.

    Raises:
        TypeError: If factor_betas or factor_premia is not a pandas Series.
        ValueError: If factor_betas and factor_premia have different lengths.
        ValueError: If factor_betas and factor_premia have different indices.
    """
    if not isinstance(factor_betas, pd.Series):
        raise TypeError("factor_betas must be a pandas Series.")
    if not isinstance(factor_premia, pd.Series):
        raise TypeError("factor_premia must be a pandas Series.")
    if len(factor_betas) != len(factor_premia):
        raise ValueError("factor_betas and factor_premia must have the same length.")
    if not factor_betas.index.equals(factor_premia.index):
        raise ValueError("factor_betas and factor_premia must have the same indices.")
    
    return rf + np.dot(factor_betas.values, factor_premia.values)