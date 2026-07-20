"""Backfill canonical M5 SQLite databases from Upstox historical APIs.

This loader is intentionally separate from the live ingestor:
- historical backfill populates older completed M5 candles
- live ingestor appends newly completed M5 candles going forward
"""

from __future__ import annotations

import argparse
import sqlite3
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import quote

import pandas as pd
import requests

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from db import ingest_m5, set_asset_profile, setup_database
from live.instrument_resolver import UpstoxInstrumentMap, resolve_instruments
from live.token_loader import load_upstox_access_token
from providers.upstox_provider import DEFAULT_HEADERS

HISTORICAL_V3_URL = "https://api.upstox.com/v3/historical-candle"
INTRADAY_V3_URL = "https://api.upstox.com/v3/historical-candle/intraday"
DEFAULT_MONTHS = 6
SAFE_5M_CHUNK_DAYS = 28


@dataclass(frozen=True, slots=True)
class BackfillResult:
    database_name: str
    instrument_key: str
    inserted_rows: int
    start: pd.Timestamp | None
    end: pd.Timestamp | None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Backfill Upstox M5 history")
    parser.add_argument("--months", type=int, default=DEFAULT_MONTHS)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--cache", type=Path, default=Path("upstox_instruments_cache.json"))
    parser.add_argument("--symbols", nargs="*", metavar="DATABASE")
    parser.add_argument("--include-today", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    token = load_upstox_access_token(root=args.root)
    if not token:
        print("UPSTOX_ACCESS_TOKEN not found in env or .streamlit/secrets.toml")
        return 2

    resolved, unresolved = resolve_target_instruments(
        root=args.root,
        cache_path=args.cache,
        symbols=args.symbols,
    )
    if unresolved:
        print(
            "Skipping unresolved symbols: "
            + ", ".join(item.database_name for item in unresolved)
        )
    if not resolved:
        print("No resolved Upstox instruments available for backfill.")
        return 2

    results: list[BackfillResult] = []
    for item in resolved:
        try:
            result = backfill_market(
                item,
                token=token,
                root=args.root,
                months=args.months,
                include_today=args.include_today,
            )
            results.append(result)
            print(
                f"OK {result.database_name}: +{result.inserted_rows} rows | "
                f"{result.start or 'n/a'} -> {result.end or 'n/a'}"
            )
        except Exception as error:
            print(f"FAILED {item.database_name}: {error}")
    return 0 if results else 1


def resolve_target_instruments(
    *,
    root: Path,
    cache_path: Path,
    symbols: list[str] | None,
) -> tuple[tuple[UpstoxInstrumentMap, ...], tuple[UpstoxInstrumentMap, ...]]:
    requested = None if not symbols else {Path(item).name.lower() for item in symbols}
    resolved, unresolved = resolve_instruments(
        cache_path=cache_path,
        output_path=root / "upstox_instrument_map.json",
    )
    if requested is None:
        return resolved, unresolved
    resolved = tuple(item for item in resolved if item.database_name.lower() in requested)
    unresolved = tuple(item for item in unresolved if item.database_name.lower() in requested)
    return resolved, unresolved


def backfill_market(
    item: UpstoxInstrumentMap,
    *,
    token: str,
    root: Path,
    months: int = DEFAULT_MONTHS,
    include_today: bool = False,
) -> BackfillResult:
    if not item.instrument_key:
        raise ValueError(f"{item.database_name} does not have an instrument_key")
    database = root / item.database_name
    setup_database(database)
    set_asset_profile(database, _profile_for_mapping(item))

    end_date = pd.Timestamp.now(tz="UTC").date()
    start_date = (pd.Timestamp.now(tz="UTC") - pd.DateOffset(months=months)).date()
    historical_end = end_date if include_today else (end_date - timedelta(days=1))

    frames: list[pd.DataFrame] = []
    if start_date <= historical_end:
        for window_start, window_end in chunk_date_range(start_date, historical_end):
            payload = fetch_historical_candles(
                item.instrument_key,
                token=token,
                from_date=window_start,
                to_date=window_end,
            )
            frame = candles_to_m5_frame(payload)
            if not frame.empty:
                frames.append(frame)

    if include_today:
        intraday = fetch_intraday_candles(item.instrument_key, token=token)
        intraday_frame = candles_to_m5_frame(intraday, drop_forming=True)
        if not intraday_frame.empty:
            frames.append(intraday_frame)

    combined = (
        pd.concat(frames).sort_index()
        if frames
        else pd.DataFrame(
            columns=["open", "high", "low", "close", "volume"],
            index=pd.DatetimeIndex([], tz="UTC", name="timestamp"),
        )
    )
    if not combined.empty:
        combined = combined[~combined.index.duplicated(keep="last")]
    inserted = ingest_m5(database, combined)
    _write_metadata(database, item)
    return BackfillResult(
        database_name=item.database_name,
        instrument_key=item.instrument_key,
        inserted_rows=inserted,
        start=None if combined.empty else combined.index[0],
        end=None if combined.empty else combined.index[-1],
    )


def chunk_date_range(
    start_date: date,
    end_date: date,
    *,
    chunk_days: int = SAFE_5M_CHUNK_DAYS,
) -> tuple[tuple[date, date], ...]:
    windows: list[tuple[date, date]] = []
    cursor = start_date
    while cursor <= end_date:
        window_end = min(cursor + timedelta(days=chunk_days - 1), end_date)
        windows.append((cursor, window_end))
        cursor = window_end + timedelta(days=1)
    return tuple(windows)


def fetch_historical_candles(
    instrument_key: str,
    *,
    token: str,
    from_date: date,
    to_date: date,
) -> dict[str, Any]:
    url = (
        f"{HISTORICAL_V3_URL}/{quote(instrument_key, safe='')}/minutes/5/"
        f"{to_date.isoformat()}/{from_date.isoformat()}"
    )
    return _get_json(url, token)


def fetch_intraday_candles(
    instrument_key: str,
    *,
    token: str,
) -> dict[str, Any]:
    url = f"{INTRADAY_V3_URL}/{quote(instrument_key, safe='')}/minutes/5"
    return _get_json(url, token)


def candles_to_m5_frame(
    payload: dict[str, Any],
    *,
    drop_forming: bool = False,
) -> pd.DataFrame:
    candles = payload.get("data", {}).get("candles", [])
    if not candles:
        return pd.DataFrame(
            columns=["open", "high", "low", "close", "volume"],
            index=pd.DatetimeIndex([], tz="UTC", name="timestamp"),
        )
    records: list[dict[str, Any]] = []
    for candle in candles:
        if not isinstance(candle, list) or len(candle) < 6:
            continue
        timestamp = pd.Timestamp(candle[0], tz="UTC").floor("5min")
        records.append(
            {
                "timestamp": timestamp,
                "open": float(candle[1]),
                "high": float(candle[2]),
                "low": float(candle[3]),
                "close": float(candle[4]),
                "volume": float(candle[5] or 0.0),
            }
        )
    if not records:
        return pd.DataFrame(
            columns=["open", "high", "low", "close", "volume"],
            index=pd.DatetimeIndex([], tz="UTC", name="timestamp"),
        )
    frame = pd.DataFrame.from_records(records).set_index("timestamp").sort_index()
    frame = frame[~frame.index.duplicated(keep="last")]
    if drop_forming and not frame.empty:
        current_bucket = pd.Timestamp.now(tz="UTC").floor("5min")
        frame = frame.loc[frame.index < current_bucket]
    return frame


def _get_json(url: str, token: str) -> dict[str, Any]:
    response = requests.get(
        url,
        headers={
            **DEFAULT_HEADERS,
            "Authorization": f"Bearer {token.strip()}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        },
        timeout=30,
        allow_redirects=True,
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise RuntimeError("Upstox returned a non-object JSON payload")
    return payload


def _profile_for_mapping(item: UpstoxInstrumentMap) -> str:
    segment = item.exchange_segment.upper()
    if segment.startswith("NSE") or segment.startswith("BSE"):
        return "NSE"
    if segment.startswith("MCX"):
        return "FUTURES"
    if segment.startswith("NCD") or segment.startswith("FOREX"):
        return "FOREX"
    if segment.startswith("GLOBAL"):
        return "24_7"
    return "24_7"


def _write_metadata(database: Path, item: UpstoxInstrumentMap) -> None:
    with sqlite3.connect(database) as connection:
        connection.executemany(
            "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
            (
                ("display_name", item.display_name),
                ("data_source", "Upstox"),
                ("upstox_instrument_key", str(item.instrument_key or "")),
                ("last_backfill_utc", pd.Timestamp.now(tz="UTC").isoformat()),
            ),
        )
        connection.commit()


if __name__ == "__main__":
    raise SystemExit(main())
