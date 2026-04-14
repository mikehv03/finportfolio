"""
finportfolio.data.loader
-------------------------
Utilities for loading market data from external sources.

Functions:
- get_prices: Load historical adjusted closing prices for a list of tickers.
- get_ff_factors: Download Fama-French factor data from Kenneth French's data library.
"""


import pandas as pd
import yfinance as yf
import io
import zipfile
import requests


def get_prices(tickers: list[str], start_date: str, end_date: str, source: str = "yfinance") -> pd.DataFrame:
    """
    Load historical adjusted closing prices for a list of financial instruments.

    Args:
        tickers (list[str]): A list of ticker symbols.
        start_date (str): The start date for the historical data.
        end_date (str): The end date for the historical data.
        source (str): The data source to use. Only "yfinance" is supported.

    Returns:
        pd.DataFrame: A DataFrame with dates as index and tickers as columns, containing adjusted closing prices.

    Raises:
        NotImplementedError: If source is not "yfinance".
        ValueError: If tickers is empty.
        ValueError: If a ticker is not found or has no data available.
        ValueError: If start_date or end_date is not a valid date.
    """
    if source != "yfinance":
        raise NotImplementedError(f"Data source '{source}' is not supported. Currently, only 'yfinance' is supported.")
    if not tickers:
        raise ValueError("The list of tickers cannot be empty.")
    try:
        pd.Timestamp(start_date)
        pd.Timestamp(end_date)
    except Exception:
        raise ValueError("start_date and end_date must be valid dates.")
    
    series_list = []
    for ticker in tickers:
        data = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=True)
        if data.empty:
            raise ValueError(f"Ticker '{ticker}' not found or no data available.")
        prices = data["Close"]
        if isinstance(prices, pd.DataFrame):
            prices = prices.iloc[:, 0]
        prices.name = ticker
        if prices.dropna().empty:
            raise ValueError(f"No valid data for ticker '{ticker}'.")
        series_list.append(prices)

    df = pd.concat(series_list, axis=1).sort_index()

    df = df.dropna(how="all")

    return df


def get_ff_factors(start_date: str = None, end_date: str = None) -> pd.DataFrame:
    """
    Download Fama-French 3-Factor daily data from Kenneth French's data library.

    Args:
        start_date (str): Start date in 'YYYY-MM-DD' format. Defaults to None.
        end_date (str): End date in 'YYYY-MM-DD' format. Defaults to None.

    Returns:
        pd.DataFrame: DataFrame with columns ['Mkt-RF', 'SMB', 'HML', 'RF'],
            returns expressed as decimals and DatetimeIndex.

    Raises:
        ValueError: If the data cannot be downloaded.
    """

    url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_daily_CSV.zip"
    
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
    except Exception as e:
        raise ValueError(f"Could not download Fama-French data: {e}")

    z = zipfile.ZipFile(io.BytesIO(r.content))
    with z.open(z.namelist()[0]) as f:
        raw = pd.read_csv(f, skiprows=4, index_col=0)

    raw = raw[raw.index.astype(str).str.len() == 8]
    raw.index = pd.to_datetime(raw.index.astype(str))
    raw = raw / 100
    raw.columns = ["Mkt-RF", "SMB", "HML", "RF"]
    raw.index.name = "Date"

    if start_date:
        raw = raw.loc[start_date:]
    if end_date:
        raw = raw.loc[:end_date]

    return raw