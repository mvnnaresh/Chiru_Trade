"""SQLite storage and deterministic aggregation for canonical M5 OHLCV data."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable
from contextlib import closing
from dataclasses import dataclass
from datetime import time
from pathlib import Path
from typing import Any, Literal

import pandas as pd

OHLCV_COLUMNS = ("open", "high", "low", "close", "volume")


@dataclass(frozen=True)
class TimeframeSpec:
    label: str
    pandas_rule: str
    kind: Literal["fixed", "calendar"]
    expected_m5_count: int | None


@dataclass(frozen=True)
class AssetProfile:
    """Trading-session rules required for calendar aggregation."""

    label: str
    timezone: str
    expected_daily_m5_count: int
    days_per_week: int
    session_open: time | None = None
    session_close: time | None = None


ASSET_PROFILES: dict[str, AssetProfile] = {
    "24_7": AssetProfile("24_7", "UTC", 288, 7),
    "NSE": AssetProfile(
        "NSE",
        "Asia/Kolkata",
        75,
        5,
        session_open=time(9, 15),
        session_close=time(15, 30),
    ),
    "US_EQUITY": AssetProfile(
        "US_EQUITY",
        "America/New_York",
        78,
        5,
        session_open=time(9, 30),
        session_close=time(16, 0),
    ),
    "FOREX": AssetProfile("FOREX", "UTC", 288, 5),
    # CME/NYMEX instruments normally pause for one hour each trading day.
    "FUTURES": AssetProfile("FUTURES", "America/New_York", 276, 5),
}


TIMEFRAMES: dict[str, TimeframeSpec] = {
    "15M": TimeframeSpec("15M", "15min", "fixed", 3),
    "30M": TimeframeSpec("30M", "30min", "fixed", 6),
    "1H": TimeframeSpec("1H", "1h", "fixed", 12),
    "2H": TimeframeSpec("2H", "2h", "fixed", 24),
    "4H": TimeframeSpec("4H", "4h", "fixed", 48),
    "1D": TimeframeSpec("1D", "1D", "calendar", None),
    "1W": TimeframeSpec("1W", "W-MON", "calendar", None),
}


def setup_database(database: str | Path) -> None:
    """Create the canonical M5 candle table and its schema metadata."""
    with closing(sqlite3.connect(database)) as connection, connection:
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


def set_asset_profile(database: str | Path, profile: str) -> None:
    """Attach a registered market-session profile to an asset database."""
    normalized = profile.upper()
    profile_key = "24_7" if normalized == "24_7" else normalized
    if profile_key not in ASSET_PROFILES:
        supported = "', '".join(ASSET_PROFILES)
        raise ValueError(f"asset profile must be one of '{supported}'")
    setup_database(database)
    with closing(sqlite3.connect(database)) as connection, connection:
        connection.execute(
            "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
            ("asset_profile", profile_key),
        )


def get_asset_profile(database: str | Path) -> AssetProfile:
    """Resolve explicit profile metadata, with compatibility filename fallbacks."""
    setup_database(database)
    with closing(sqlite3.connect(database)) as connection, connection:
        row = connection.execute(
            "SELECT value FROM metadata WHERE key = 'asset_profile'"
        ).fetchone()
    if row is not None:
        return ASSET_PROFILES[row[0]]

    asset_name = Path(database).stem.upper()
    if any(token in asset_name for token in ("NIFTY", "BANKNIFTY", "NSE")):
        return ASSET_PROFILES["NSE"]
    return ASSET_PROFILES["24_7"]


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

    with closing(sqlite3.connect(database)) as connection, connection:
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
    with closing(sqlite3.connect(database)) as connection, connection:
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
    timeframe: str,
    start: Any | None = None,
    end: Any | None = None,
    *,
    complete_only: bool = True,
) -> pd.DataFrame:
    """Generate a registered timeframe dynamically from canonical M5 data.

    Bars are UTC epoch-aligned and left-labelled. With ``complete_only=True``,
    fixed bars are returned only when every expected M5 timestamp is present.
    """
    normalized_timeframe = timeframe.upper()
    if normalized_timeframe not in TIMEFRAMES:
        supported = "', '".join(TIMEFRAMES)
        raise ValueError(f"timeframe must be one of '{supported}'")
    spec = TIMEFRAMES[normalized_timeframe]
    m5 = load_m5(database, start=start, end=end)
    if m5.empty:
        return m5

    profile = get_asset_profile(database) if spec.kind == "calendar" else None
    source = m5.tz_convert(profile.timezone) if profile is not None else m5
    if (
        profile is not None
        and profile.session_open is not None
        and profile.session_close is not None
    ):
        source = source.between_time(
            profile.session_open,
            profile.session_close,
            inclusive="left",
        )
    aggregation = source.resample(
        spec.pandas_rule, label="left", closed="left", origin="epoch"
    ).agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
    )
    counts = source["close"].resample(
        spec.pandas_rule, label="left", closed="left", origin="epoch"
    ).count()
    aggregation = aggregation.dropna(subset=["open", "high", "low", "close"])
    if complete_only and spec.kind == "fixed":
        assert spec.expected_m5_count is not None
        aggregation = aggregation.loc[counts == spec.expected_m5_count]
    elif complete_only and profile is not None:
        expected_count = profile.expected_daily_m5_count
        if normalized_timeframe == "1W":
            expected_count *= profile.days_per_week
        aggregation = aggregation.loc[counts == expected_count]
    aggregation.index = aggregation.index.tz_convert("UTC")
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
