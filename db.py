"""SQLite storage and deterministic aggregation for canonical M5 OHLCV data."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Literal

import pandas as pd

OHLCV_COLUMNS = ("open", "high", "low", "close", "volume")
TIMEFRAME_RULES = {"1H": ("1h", 12), "4H": ("4h", 48)}


def setup_database(database: str | Path) -> None:
    """Create the canonical M5 candle table and its schema metadata."""
    with sqlite3.connect(database) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS m5_candles (
                timestamp TEXT PRIMARY KEY,
                open      REAL NOT NULL,
                high      REAL NOT NULL,
                low       REAL NOT NULL,
                close     REAL NOT NULL,
                volume    REAL NOT NULL CHECK (volume >= 0),
                CHECK (high >= low),
                CHECK (high >= open AND high >= close),
                CHECK (low <= open AND low <= close)
            ) WITHOUT ROWID
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS metadata (
                key   TEXT PRIMARY KEY,
                value TEXT NOT NULL
            ) WITHOUT ROWID
            """
        )
        connection.execute(
            "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
            ("base_timeframe", "5min"),
        )


def ingest_m5(
    database: str | Path,
    candles: pd.DataFrame | Iterable[dict[str, Any]],
) -> int:
    """Append valid M5 candles, ignoring timestamps already stored.

    Timestamps are normalized to UTC and must lie exactly on a five-minute
    boundary. The function is atomic and returns the number of inserted rows.
    Existing rows are deliberately immutable to keep the raw feed auditable.
    """
    frame = _normalize_candles(candles)
    if frame.empty:
        return 0

    setup_database(database)
    records = [
        (
            timestamp.isoformat().replace("+00:00", "Z"),
            float(row.open),
            float(row.high),
            float(row.low),
            float(row.close),
            float(row.volume),
        )
        for timestamp, row in frame.iterrows()
    ]

    with sqlite3.connect(database) as connection:
        before = connection.total_changes
        connection.executemany(
            """
            INSERT OR IGNORE INTO m5_candles
                (timestamp, open, high, low, close, volume)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            records,
        )
        return connection.total_changes - before


def load_m5(
    database: str | Path,
    start: Any | None = None,
    end: Any | None = None,
) -> pd.DataFrame:
    """Load canonical candles in UTC, optionally bounded inclusively."""
    setup_database(database)
    clauses: list[str] = []
    parameters: list[str] = []
    if start is not None:
        clauses.append("timestamp >= ?")
        parameters.append(_utc_iso(start))
    if end is not None:
        clauses.append("timestamp <= ?")
        parameters.append(_utc_iso(end))

    where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
    query = (
        "SELECT timestamp, open, high, low, close, volume "
        f"FROM m5_candles{where} ORDER BY timestamp"
    )
    with sqlite3.connect(database) as connection:
        frame = pd.read_sql_query(query, connection, params=parameters)

    if frame.empty:
        return pd.DataFrame(
            columns=OHLCV_COLUMNS,
            index=pd.DatetimeIndex([], name="timestamp", tz="UTC"),
        )
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    return frame.set_index("timestamp")


def resample_ohlcv(
    database: str | Path,
    timeframe: Literal["1H", "4H"],
    start: Any | None = None,
    end: Any | None = None,
    *,
    complete_only: bool = True,
) -> pd.DataFrame:
    """Generate aligned 1H or 4H candles dynamically from canonical M5 data.

    Bars are UTC epoch-aligned and left-labelled. With ``complete_only=True``,
    a bar is returned only when every expected M5 timestamp is present.
    """
    normalized_timeframe = timeframe.upper()
    if normalized_timeframe not in TIMEFRAME_RULES:
        raise ValueError("timeframe must be '1H' or '4H'")
    rule, expected_count = TIMEFRAME_RULES[normalized_timeframe]
    m5 = load_m5(database, start=start, end=end)
    if m5.empty:
        return m5

    aggregation = m5.resample(
        rule, label="left", closed="left", origin="epoch"
    ).agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
    )
    counts = m5["close"].resample(
        rule, label="left", closed="left", origin="epoch"
    ).count()
    aggregation = aggregation.dropna(subset=["open", "high", "low", "close"])
    if complete_only:
        aggregation = aggregation.loc[counts == expected_count]
    aggregation.index.name = "timestamp"
    return aggregation.astype(float)


def _normalize_candles(
    candles: pd.DataFrame | Iterable[dict[str, Any]],
) -> pd.DataFrame:
    frame = candles.copy() if isinstance(candles, pd.DataFrame) else pd.DataFrame(candles)
    if frame.empty:
        return pd.DataFrame(columns=OHLCV_COLUMNS)

    if "timestamp" in frame.columns:
        timestamps = pd.to_datetime(frame.pop("timestamp"), utc=True, errors="raise")
    elif isinstance(frame.index, pd.DatetimeIndex):
        timestamps = pd.to_datetime(frame.index, utc=True, errors="raise")
    else:
        raise ValueError("candles require a 'timestamp' column or DatetimeIndex")

    missing = set(OHLCV_COLUMNS).difference(frame.columns)
    if missing:
        raise ValueError(f"missing candle columns: {', '.join(sorted(missing))}")
    frame = frame.loc[:, OHLCV_COLUMNS].apply(pd.to_numeric, errors="raise")
    frame.index = pd.DatetimeIndex(timestamps, name="timestamp")

    if frame.index.hasnans:
        raise ValueError("timestamps must not be null")
    if frame.index.duplicated().any():
        raise ValueError("input contains duplicate timestamps")
    if ((frame.index.minute % 5) != 0).any() or (
        (frame.index.second != 0) | (frame.index.microsecond != 0)
    ).any():
        raise ValueError("timestamps must align exactly to five-minute boundaries")
    if frame.isna().any().any():
        raise ValueError("OHLCV values must not be null")
    if (frame["volume"] < 0).any():
        raise ValueError("volume must be non-negative")
    if (
        (frame["high"] < frame[["open", "close", "low"]].max(axis=1))
        | (frame["low"] > frame[["open", "close", "high"]].min(axis=1))
    ).any():
        raise ValueError("OHLC prices are internally inconsistent")
    return frame.sort_index()


def _utc_iso(value: Any) -> str:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    return timestamp.isoformat().replace("+00:00", "Z")
