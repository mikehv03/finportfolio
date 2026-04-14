"""
finportfolio.factors.models
---------------------------
Factor models for estimating asset exposures using OLS regression.

This module provides implementations of classical asset pricing models, including the Single Index Model and the Fama-French 3-Factor Model, as well as a general multifactor regression framework.

Functions:
- single_index_model: Estimate the Single Index Model for an asset using OLS regression
- fama_french_3factor: Estimate the Fama-French 3-Factor Model
- multifactor_model: Estimate a multifactor model for an asset using OLS regression
- factor_exposure_report: Generate a factor exposure report for all assets in a portfolio
"""

import numpy as np
import pandas as pd
import warnings
    
    
def single_index_model(asset_returns: pd.Series, market_returns: pd.Series) -> pd.Series:
    """
    Estimate the Single Index Model for an asset using OLS regression.

    The model is estimated as:

        asset_returns = alpha + beta * market_returns + error

    Args:
        asset_returns (pd.Series): Returns of the asset.
        market_returns (pd.Series): Returns of the market.

    Returns:
        pd.Series: Alpha, beta, R-squared, and residual variance of the asset.

    Raises:
        TypeError: If asset_returns or market_returns is not a pandas Series.
        ValueError: If the input series are empty.
        ValueError: If the input series have different lengths.
        ValueError: If there are fewer than three valid paired observations.
        ValueError: If market_returns have zero variance.
        ValueError: If asset_returns have zero total variance.

    Notes:
        This alpha is the intercept of the regression and does not correspond to Jensen's alpha.
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

    data = pd.concat([asset_returns, market_returns], axis=1).dropna()
    if len(data) < 3:
        raise ValueError("At least three valid paired observations are required to estimate the Single Index Model.")

    y = data.iloc[:, 0].to_numpy(dtype=float)
    x = data.iloc[:, 1].to_numpy(dtype=float)

    market_variance = np.var(x, ddof=1)
    if np.isclose(market_variance, 0.0):
        raise ValueError("market_returns have zero variance; beta is undefined.")

    X = np.column_stack([np.ones(len(x)), x])
    coefficients = np.linalg.lstsq(X, y, rcond=None)[0]

    alpha = float(coefficients[0])
    beta = float(coefficients[1])

    y_hat = X @ coefficients
    residuals = y - y_hat

    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)

    if np.isclose(ss_tot, 0.0):
        raise ValueError("asset_returns have zero total variance; R-squared is undefined.")

    r_squared = 1 - ss_res / ss_tot
    residual_variance = ss_res/ (len(y) - 2)
    systematic_variance = beta ** 2 * market_variance

    return pd.Series(
        {
            "alpha": alpha,
            "beta": beta,
            "r_squared": float(r_squared),
            "residual_variance": float(residual_variance),
            "systematic_variance": float(systematic_variance),
        }
    )


def fama_french_3factor(asset_returns: pd.Series, ff_factors: pd.DataFrame) -> pd.Series:
    """
    Estimate the Fama-French 3-Factor Model for an asset.

    The model is estimated as:
        asset_returns - RF = alpha + beta_market * (Mkt-RF) + beta_smb * SMB + beta_hml * HML + error

    Args:
        asset_returns (pd.Series): Returns of the asset.
        ff_factors (pd.DataFrame): A DataFrame containing the market excess return, SMB and HML factors.

    Returns:
        pd.Series: Alpha, market beta, SMB beta, HML beta and R-squared.

    Raises:
        TypeError: If asset_returns is not a pandas Series.
        TypeError: If ff_factors is not a pandas DataFrame.
        ValueError: If ff_factors is missing required columns.
        ValueError: If there are fewer than five valid overlapping observations.
        ValueError: If asset_returns have zero total variance.
    """
    if not isinstance(asset_returns, pd.Series):
        raise TypeError("asset_returns must be a pandas Series.")
    if not isinstance(ff_factors, pd.DataFrame):
        raise TypeError("ff_factors must be a pandas DataFrame.")
    if ff_factors.empty:
        raise ValueError("ff_factors cannot be empty.")

    required_columns = ["Mkt-RF", "SMB", "HML", "RF"]
    missing = [col for col in required_columns if col not in ff_factors.columns]
    if missing:
        raise ValueError(f"ff_factors is missing columns: {missing}")
    
    aligned = pd.concat([asset_returns, ff_factors], axis=1).dropna()
    if len(aligned) < 5:
        raise ValueError("At least five valid overlapping observations are required to estimate the Fama-French 3-Factor Model.")

    y = aligned.iloc[:, 0].values - aligned["RF"].values
    X = aligned[["Mkt-RF", "SMB", "HML"]].values
    X = np.column_stack([np.ones(len(X)), X])

    coeffs = np.linalg.lstsq(X, y, rcond=None)[0]

    y_hat = X @ coeffs
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    if np.isclose(ss_tot, 0.0):
        raise ValueError("asset_returns have zero total variance; R-squared is undefined.")
    r_squared = 1 - ss_res / ss_tot

    return pd.Series(
        {
            "alpha": float(coeffs[0]),
            "beta_market": float(coeffs[1]),
            "beta_smb": float(coeffs[2]),
            "beta_hml": float(coeffs[3]),
            "r_squared": float(r_squared),
        }
    )


def multifactor_model(asset_returns: pd.Series, factors: pd.DataFrame, returns_are_excess: bool = True, rf: float = None) -> pd.Series:
    """
    Estimate a multifactor model for an asset using OLS regression.

    If returns_are_excess is True, the model is estimated directly on excess returns:

        asset_returns = alpha + beta_1 * factor_1 + ... + beta_k * factor_k + error

    If returns_are_excess is False, raw asset returns are converted to excess returns
    using a constant risk-free rate rf:

        asset_returns - rf = alpha + beta_1 * factor_1 + ... + beta_k * factor_k + error

    Args:
        asset_returns (pd.Series): Returns of the asset.
        factors (pd.DataFrame): Factor returns with dates as index and factor names as columns.
        returns_are_excess (bool): Whether asset_returns already represent excess returns.
            Defaults to True.
        rf (float | None): Constant risk-free rate used only when returns_are_excess is False.

    Returns:
        pd.Series: Alpha, factor betas, and R-squared.

    Raises:
        TypeError: If asset_returns is not a pandas Series.
        TypeError: If factors is not a pandas DataFrame.
        TypeError: If returns_are_excess is not a bool.
        TypeError: If rf is provided and is not a float or int.
        ValueError: If factors is empty.
        ValueError: If rf is not provided when returns_are_excess is False.
        ValueError: If there are too few valid overlapping observations.
        ValueError: If asset_returns have zero total variance.
    """
    if not isinstance(asset_returns, pd.Series):
        raise TypeError("asset_returns must be a pandas Series.")
    if not isinstance(factors, pd.DataFrame):
        raise TypeError("factors must be a pandas DataFrame.")
    if not isinstance(returns_are_excess, bool):
        raise TypeError("returns_are_excess must be a bool.")
    if rf is not None and not isinstance(rf, (int, float)):
        raise TypeError("rf must be a float when provided.")
    if factors.empty:
        raise ValueError("factors cannot be empty.")

    aligned = pd.concat([asset_returns, factors], axis=1).dropna()

    n_factors = factors.shape[1]
    min_obs = n_factors + 2
    if len(aligned) < min_obs:
        raise ValueError(
            f"At least {min_obs} valid overlapping observations are required to estimate the multifactor model."
        )

    y = aligned.iloc[:, 0].to_numpy(dtype=float)
    if not returns_are_excess:
        if rf is None:
            raise ValueError("rf must be provided when returns_are_excess is False.")
        y = y - float(rf)

    X = aligned.iloc[:, 1:].to_numpy(dtype=float)
    X = np.column_stack([np.ones(len(X)), X])

    coeffs = np.linalg.lstsq(X, y, rcond=None)[0]

    y_hat = X @ coeffs
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)

    if np.isclose(ss_tot, 0.0):
        raise ValueError("asset_returns have zero total variance; R-squared is undefined.")

    r_squared = 1 - ss_res / ss_tot

    factor_names = factors.columns.tolist()
    result = {"alpha": float(coeffs[0])}
    for i, name in enumerate(factor_names):
        result[f"beta_{name}"] = float(coeffs[i + 1])
    result["r_squared"] = float(r_squared)

    return pd.Series(result)


def factor_exposure_report(returns: pd.DataFrame, ff_factors: pd.DataFrame) -> pd.DataFrame:
    """
    Generate a factor exposure report for all assets in a portfolio.

    Args:
        returns (pd.DataFrame): Asset returns with dates as index and tickers as columns.
        ff_factors (pd.DataFrame): Fama-French factors with columns ['Mkt-RF', 'SMB', 'HML', 'RF'].

    Returns:
        pd.DataFrame: Factor exposures (alpha, betas, R-squared) for each asset.

    Raises:
        TypeError: If returns is not a pandas DataFrame.
        TypeError: If ff_factors is not a pandas DataFrame.
        ValueError: If returns is empty.
        ValueError: If returns has no asset columns.
        ValueError: If ff_factors is missing required columns.
        ValueError: If any asset has insufficient overlapping observations with ff_factors.
        Warning: If the model cannot be estimated for an asset, a warning is issued and an empty Series is returned for that asset.
    """
    if not isinstance(returns, pd.DataFrame):
        raise TypeError("returns must be a pandas DataFrame.")
    if not isinstance(ff_factors, pd.DataFrame):
        raise TypeError("ff_factors must be a pandas DataFrame.")
    if returns.shape[0] == 0:
        raise ValueError("returns cannot be empty.")
    if returns.shape[1] == 0:
        raise ValueError("returns must have at least one asset column.")

    required_columns = ["Mkt-RF", "SMB", "HML", "RF"]
    missing = [c for c in required_columns if c not in ff_factors.columns]
    if missing:
        raise ValueError(f"ff_factors is missing columns: {missing}")

    report = {}
    for ticker in returns.columns:
        try:
            report[ticker] = fama_french_3factor(returns[ticker], ff_factors)
        except ValueError as e:
            warnings.warn(f"Could not estimate model for {ticker}: {e}")
            report[ticker] = pd.Series(dtype=float)

    return pd.DataFrame(report).T