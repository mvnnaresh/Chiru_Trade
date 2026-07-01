"""Temporary Yahoo Finance adapter for refreshing canonical M5 test databases.

Yahoo is deliberately isolated here: the analytical pipeline continues to read
only the provider-neutral SQLite schema defined in :mod:`db`.
"""

from __future__ import annotations

import argparse
import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yfinance as yf

from db import get_asset_profile, ingest_m5, set_asset_profile, setup_database


@dataclass(frozen=True)
class MarketSymbol:
    symbol: str
    database_name: str
    profile: str
    display_name: str


DEFAULT_UNIVERSE: tuple[MarketSymbol, ...] = (
    MarketSymbol("^NSEI", "NIFTY_50.db", "NSE", "NIFTY 50"),
    MarketSymbol("^NSEBANK", "NIFTY_BANK.db", "NSE", "NIFTY Bank"),
    MarketSymbol("RELIANCE.NS", "RELIANCE.db", "NSE", "Reliance Industries"),
    MarketSymbol("TCS.NS", "TCS.db", "NSE", "Tata Consultancy Services"),
    MarketSymbol("HDFCBANK.NS", "HDFC_BANK.db", "NSE", "HDFC Bank"),
    MarketSymbol("ICICIBANK.NS", "ICICI_BANK.db", "NSE", "ICICI Bank"),
    MarketSymbol("INFY.NS", "INFOSYS.db", "NSE", "Infosys"),
    MarketSymbol("SBIN.NS", "SBI.db", "NSE", "State Bank of India"),
    MarketSymbol("BHARTIARTL.NS", "BHARTI_AIRTEL.db", "NSE", "Bharti Airtel"),
    MarketSymbol("LT.NS", "LARSEN_TOUBRO.db", "NSE", "Larsen & Toubro"),
    MarketSymbol("BTC-USD", "BTC_USD.db", "24_7", "Bitcoin / US Dollar"),
    MarketSymbol("ETH-USD", "ETH_USD.db", "24_7", "Ether / US Dollar"),
    MarketSymbol("SOL-USD", "SOL_USD.db", "24_7", "Solana / US Dollar"),
    MarketSymbol("BNB-USD", "BNB_USD.db", "24_7", "BNB / US Dollar"),
    MarketSymbol("XRP-USD", "XRP_USD.db", "24_7", "XRP / US Dollar"),
    MarketSymbol("^GSPC", "SP500.db", "US_EQUITY", "S&P 500"),
    MarketSymbol("^NDX", "NASDAQ_100.db", "US_EQUITY", "Nasdaq 100"),
    MarketSymbol("^DJI", "DOW_JONES.db", "US_EQUITY", "Dow Jones"),
    MarketSymbol("^RUT", "RUSSELL_2000.db", "US_EQUITY", "Russell 2000"),
    MarketSymbol("GC=F", "GOLD.db", "FUTURES", "Gold Futures"),
    MarketSymbol("SI=F", "SILVER.db", "FUTURES", "Silver Futures"),
    MarketSymbol("CL=F", "WTI_CRUDE.db", "FUTURES", "WTI Crude Oil"),
    MarketSymbol("EURUSD=X", "EUR_USD.db", "FOREX", "EUR / USD"),
    MarketSymbol("GBPUSD=X", "GBP_USD.db", "FOREX", "GBP / USD"),
    MarketSymbol("USDINR=X", "USD_INR.db", "FOREX", "USD / INR"),
)

UNIVERSE_BY_DATABASE: dict[str, MarketSymbol] = {
    market.database_name.lower(): market for market in DEFAULT_UNIVERSE
}


def download_m5(symbol: str, *, period: str = "60d") -> pd.DataFrame:
    """Download and normalize the currently available five-minute history."""
    raw = yf.download(
        symbol,
        period=period,
        interval="5m",
        auto_adjust=False,
        actions=False,
        prepost=False,
        progress=False,
        threads=False,
        timeout=30,
    )
    if raw.empty:
        raise RuntimeError(f"Yahoo Finance returned no M5 candles for {symbol}")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    normalized = raw.rename(columns=str.lower)
    normalized = normalized.loc[:, ["open", "high", "low", "close", "volume"]]
    normalized.index = pd.to_datetime(normalized.index, utc=True)
    normalized.index = normalized.index.floor("5min")
    normalized = normalized[~normalized.index.duplicated(keep="last")]
    normalized = normalized.dropna(subset=["open", "high", "low", "close"])
    normalized["volume"] = normalized["volume"].fillna(0.0)

    # Never persist a still-forming five-minute candle.
    current_bucket = pd.Timestamp.now(tz="UTC").floor("5min")
    normalized = normalized.loc[normalized.index < current_bucket]
    if normalized.empty:
        raise RuntimeError(f"Yahoo Finance returned no completed M5 candles for {symbol}")
    return normalized


def _read_metadata(database: str | Path, key: str) -> str | None:
    setup_database(database)
    connection = sqlite3.connect(database)
    try:
        row = connection.execute(
            "SELECT value FROM metadata WHERE key = ?",
            (key,),
        ).fetchone()
    finally:
        connection.close()
    return None if row is None else str(row[0])


def resolve_market_symbol(database: str | Path) -> MarketSymbol | None:
    """Resolve a database back to its Yahoo/market registry metadata."""
    path = Path(database)
    matched = UNIVERSE_BY_DATABASE.get(path.name.lower())
    if matched is not None:
        return matched
    symbol = _read_metadata(path, "symbol")
    if not symbol:
        return None
    display_name = _read_metadata(path, "display_name") or path.stem
    profile = _read_metadata(path, "asset_profile") or get_asset_profile(path).label
    return MarketSymbol(symbol, path.name, profile, display_name)


def append_latest_m5(
    database: str | Path,
    *,
    period: str = "5d",
) -> tuple[int, pd.Timestamp]:
    """Append the latest completed Yahoo M5 candles to an existing database."""
    path = Path(database)
    market = resolve_market_symbol(path)
    if market is None:
        raise ValueError(f"no Yahoo market symbol is registered for {path.name}")
    candles = download_m5(market.symbol, period=period)
    set_asset_profile(path, market.profile)
    inserted = ingest_m5(path, candles)
    connection = sqlite3.connect(path)
    try:
        connection.executemany(
            "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
            (
                ("symbol", market.symbol),
                ("display_name", market.display_name),
                ("data_source", "Yahoo Finance"),
                ("last_refresh_utc", pd.Timestamp.now(tz="UTC").isoformat()),
            ),
        )
        connection.commit()
    finally:
        connection.close()
    return inserted, candles.index[-1]


def replace_database(
    market: MarketSymbol,
    directory: str | Path = ".",
    *,
    period: str = "60d",
) -> tuple[Path, int, pd.Timestamp, pd.Timestamp]:
    """Atomically replace one test database with freshly downloaded candles."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / market.database_name
    temporary = destination.with_suffix(destination.suffix + ".refreshing")
    if temporary.exists():
        temporary.unlink()

    candles = download_m5(market.symbol, period=period)
    try:
        set_asset_profile(temporary, market.profile)
        inserted = ingest_m5(temporary, candles)
        connection = sqlite3.connect(temporary)
        try:
            connection.executemany(
                "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
                (
                    ("symbol", market.symbol),
                    ("display_name", market.display_name),
                    ("data_source", "Yahoo Finance"),
                    ("last_refresh_utc", pd.Timestamp.now(tz="UTC").isoformat()),
                ),
            )
            connection.commit()
        finally:
            connection.close()
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination, inserted, candles.index[0], candles.index[-1]


def refresh_universe(
    directory: str | Path = ".",
    universe: tuple[MarketSymbol, ...] = DEFAULT_UNIVERSE,
    *,
    period: str = "60d",
) -> tuple[list[tuple[Path, int, pd.Timestamp, pd.Timestamp]], list[tuple[str, str]]]:
    """Refresh all symbols, retaining successful assets if another fails."""
    successes = []
    failures = []
    for market in universe:
        try:
            successes.append(replace_database(market, directory, period=period))
        except Exception as error:  # provider failures must not abort the batch
            failures.append((market.symbol, str(error)))
    return successes, failures


def main() -> int:
    parser = argparse.ArgumentParser(description="Refresh local Yahoo M5 test databases")
    parser.add_argument("--directory", default=".")
    parser.add_argument("--period", default="60d")
    arguments = parser.parse_args()
    successes, failures = refresh_universe(arguments.directory, period=arguments.period)
    for database, rows, start, end in successes:
        print(f"OK {database.name}: {rows:,} rows, {start} through {end}")
    for symbol, error in failures:
        print(f"FAILED {symbol}: {error}")
    print(f"Refreshed {len(successes)}/{len(DEFAULT_UNIVERSE)} markets")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
