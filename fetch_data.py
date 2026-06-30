"""Script to fetch historical data and populate the database for the MVP."""

import pandas as pd
import yfinance as yf
from db import ingest_m5

def seed_market_data(ticker: str, db_name: str):
    print(f"Fetching 5-minute data for {ticker} from Yahoo Finance...")
    
    # Yahoo Finance allows retrieving max 60 days of 5m data
    df = yf.download(tickers=ticker, period="60d", interval="5m")
    
    if df.empty:
        print("Failed to fetch data. Verify network or ticker.")
        return

    # Flatten multi-index columns if present in newer yfinance versions
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.reset_index()
    
    # Rename columns to match the strict schema in db.py
    df = df.rename(columns={
        "Datetime": "timestamp",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume"
    })

    # Drop any extra columns not needed by the schema
    df = df[["timestamp", "open", "high", "low", "close", "volume"]]

    # Run the strict ingestion layer from your db.py
    inserted = ingest_m5(db_name, df)
    print(f"Successfully inserted {inserted} new candles into {db_name}!")

if __name__ == "__main__":
    # Example 1: NIFTY 50 Index
    seed_market_data("^NSEI", "NIFTY_50.db")
    
    # Example 2: Bitcoin
    seed_market_data("BTC-USD", "BTC_USD.db")