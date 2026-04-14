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
import pandas_datareader.data as web


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