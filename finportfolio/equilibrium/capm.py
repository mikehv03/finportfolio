"""
finportfolio.equilibrium.capm
-----------------------------
Capital Asset Pricing Model (CAPM) and related equilibrium models.

Function:
- estimate_beta: Estimate the beta of an asset given its returns and market returns.
- capm_expected_return: Calculate the expected return of an asset using the CAPM formula.
- security_market_line: Calculate the Security Market Line (SML) for a range of betas.
- gordon_model: Calculate the intrinsic value of a stock using the Gordon Growth Model.
"""

import numpy as np
import pandas as pd

def estimate_beta(asset_returns: pd.Series, market_returns: pd.Series) -> float:
    """
    Estimate the beta of an asset

    Args:
    asset_returns (pd.Series): A series of returns for the asset.
    market_returns (pd.Series): A series of returns for the market.

    Returns:
    float: The beta of the asset.
    """
    covariance = np.cov(asset_returns, market_returns, ddof=1)[0][1]
    variance = np.var(market_returns, ddof=1)
    beta = covariance / variance
    return beta

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

def security_market_line(betas: np.ndarray, rf: float, market_return: float) -> pd.DataFrame:
    """
    Calculate the Security Market Line (SML) for a range of betas.

    Args:
        betas (np.ndarray): Array of beta values.
        rf (float): The risk-free rate.
        market_return (float): The expected return of the market.

    Returns:
        pd.DataFrame: A DataFrame containing betas and corresponding expected returns.
    """
    expected_returns = [capm_expected_return(rf, beta, market_return) for beta in betas]
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
        ValueError: If factor_betas and factor_premia have different lengths.
        ValueError: If factor_betas and factor_premia have different indices.
    """
    if len(factor_betas) != len(factor_premia):
        raise ValueError("factor_betas and factor_premia must have the same length.")
    if not factor_betas.index.equals(factor_premia.index):
        raise ValueError("factor_betas and factor_premia must have the same indices.")
    
    return rf + np.dot(factor_betas.values, factor_premia.values)