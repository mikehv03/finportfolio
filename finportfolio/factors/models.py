import numpy as np
import pandas as pd

def _validate_series(asset_returns, market_returns):
    """
    Validate the input series for beta estimation.
    
    Args:
        asset_returns (pd.Series): A series of returns for the asset.
        market_returns (pd.Series): A series of returns for the market.
    
    Raises:
        ValueError: If the lengths of the series do not match, if they are empty, or if they contain NaN values.
    """
    if len(asset_returns) != len(market_returns):
        raise ValueError("asset_returns and market_returns must have the same length.")
    if asset_returns.empty or market_returns.empty:
        raise ValueError("Returns cannot be empty.")
    if asset_returns.isna().any() or market_returns.isna().any():
        raise ValueError("Returns cannot contain NaN values.")
    
def single_index_model(asset_returns: pd.Series, market_returns: pd.Series) -> float:
    """
     Estimate the Single Index Model for an asset.

    Args:
        asset_returns (pd.Series): Returns of the asset.
        market_returns (pd.Series): Returns of the market.

    Returns:
        pd.Series: Alpha, beta, R-squared and residual variance of the asset.

    Raises:
        ValueError: If the series have different lengths.
        ValueError: If the series are empty.
        ValueError: If the series contain NaN values.
        ValueError: If market variance is zero.
        ValueError: If asset variance is zero.
    """
    _validate_series(asset_returns, market_returns)
    
    covariance = np.cov(asset_returns, market_returns, ddof=1)[0][1]
    market_variance = np.var(market_returns, ddof=1)
    if np.close(market_variance, 0.0):
        raise ValueError("Market returns variance cannot be zero.")
    
    beta = covariance / market_variance
    alpha = asset_returns.mean() - beta * market_returns.mean()
    asset_variance = np.var(asset_returns, ddof=1)
    if np.close(asset_variance, 0.0):
        raise ValueError("Asset returns variance cannot be zero.")
    r_squared = (beta ** 2 * market_variance) / asset_variance
    residual_variance = asset_variance - beta ** 2 * market_variance

    return pd.Series({"alpha": alpha, "beta": beta, "r_squared": r_squared, "residual_variance": residual_variance})

def fama_french_3factor(asset_returns: pd.Series, ff_factors: pd.DataFrame) -> pd.Series:
    """
    Estimate the Fama-French 3-Factor Model for an asset.

    Args:
        asset_returns (pd.Series): Returns of the asset.
        ff_factors (pd.DataFrame): A DataFrame containing the market excess return, SMB and HML factors.

    Returns:
        pd.Series: Alpha, market beta, SMB beta, HML beta and R-squared.

    Raises:
        ValueError: If ff_factors is missing required columns.
        ValueError: If there are no overlapping dates.
        ValueError: If the data is empty.
        ValueError: If the data contains NaN values.
    """
    required_columns = ["Mkt-RF", "SMB", "HML", "RF"]
    missing = [col for col in required_columns if col not in ff_factors.columns]
    if missing:
        raise ValueError(f"ff_factors is missing columns: {missing}")
    
    aligned = pd.concat([asset_returns, ff_factors], axis=1).dropna()
    if aligned.empty:
        raise ValueError("No overlapping data between asset returns and Fama-French factors.")

    y = aligned.iloc[:, 0].values - aligned["RF"].values
    X = aligned[["Mkt-RF", "SMB", "HML"]].values
    X = np.column_stack([np.ones(len(X)), X])

    coeffs = np.linalg.inv(X.T @ X) @ X.T @ y

    y_hat = X @ coeffs
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r_squared = 1 - ss_res / ss_tot

    return pd.Series({
        "alpha": coeffs[0],
        "beta_market": coeffs[1],
        "beta_smb": coeffs[2],
        "beta_hml": coeffs[3],
        "r_squared": r_squared
    })

def factor_exposure_report(returns: pd.DataFrame, ff_factors: pd.DataFrame) -> pd.DataFrame:
    """
    Generate a factor exposure report for all assets in a portfolio.

    Args:
        returns (pd.DataFrame): Asset returns with dates as index and tickers as columns.
        ff_factors (pd.DataFrame): Fama-French factors with columns
                                   ['Mkt-RF', 'SMB', 'HML', 'RF'].

    Returns:
        pd.DataFrame: Factor exposures (alpha, betas, R-squared) for each asset.

    Raises:
        ValueError: If returns is empty.
        ValueError: If ff_factors is missing required columns.
        ValueError: If there are no overlapping dates.
    """
    if returns.empty:
        raise ValueError("returns cannot be empty.")

    required_cols = ["Mkt-RF", "SMB", "HML", "RF"]
    missing = [c for c in required_cols if c not in ff_factors.columns]
    if missing:
        raise ValueError(f"ff_factors is missing columns: {missing}")

    report = {}
    for ticker in returns.columns:
        report[ticker] = fama_french_3factor(returns[ticker], ff_factors)

    return pd.DataFrame(report).T   

def multifactor_model(asset_returns: pd.Series, factors: pd.DataFrame, rf: pd.Series = None) -> pd.Series:
    """
    Estimate a multifactor model for an asset using any set of factors.

    Args:
        asset_returns (pd.Series): Returns of the asset.
        factors (pd.DataFrame): Factor returns with dates as index and factor names as columns.
        rf (pd.Series, optional): Risk-free rate series. If provided, excess returns are used.

    Returns:
        pd.Series: Alpha, factor betas and R-squared.

    Raises:
        ValueError: If factors is empty.
        ValueError: If there are no overlapping dates.
        ValueError: If the data contains NaN values.
    """
    if factors.empty:
        raise ValueError("factors cannot be empty.")

    if rf is not None:
        aligned = pd.concat([asset_returns, factors, rf], axis=1).dropna()
        y = aligned.iloc[:, 0].values - aligned.iloc[:, -1].values
        X = aligned.iloc[:, 1:-1].values
    else:
        aligned = pd.concat([asset_returns, factors], axis=1).dropna()
        y = aligned.iloc[:, 0].values
        X = aligned.iloc[:, 1:].values

    if len(aligned) == 0:
        raise ValueError("No overlapping dates between asset_returns and factors.")

    X = np.column_stack([np.ones(len(X)), X])

    coeffs = np.linalg.inv(X.T @ X) @ X.T @ y

    y_hat = X @ coeffs
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r_squared = 1 - ss_res / ss_tot

    factor_names = factors.columns.tolist()
    result = {"alpha": coeffs[0]}
    for i, name in enumerate(factor_names):
        result[f"beta_{name}"] = coeffs[i + 1]
    result["r_squared"] = r_squared

    return pd.Series(result)
