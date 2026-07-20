import json
import sqlite3
from datetime import date

import pandas as pd

from live.candle_builder import M5CandleBuilder
from live.upstox_backfill import candles_to_m5_frame, chunk_date_range
from live.instrument_resolver import (
    DEFAULT_UPSTOX_UNIVERSE,
    UpstoxInstrumentMap,
    resolve_instruments,
)
from live.token_loader import load_upstox_access_token
from live.upstox_live_ingestor import main, select_universe


def test_default_mapping_covers_existing_universe():
    assert len(DEFAULT_UPSTOX_UNIVERSE) == 25
    by_database = {item.database_name: item for item in DEFAULT_UPSTOX_UNIVERSE}
    assert by_database["NIFTY_50.db"].trading_symbol == "NIFTY"
    assert by_database["HDFC_BANK.db"].trading_symbol == "HDFCBANK"
    assert by_database["RELIANCE.db"].exchange_segment == "NSE_EQ"


def test_resolver_matches_exact_segment_and_reports_unresolved(tmp_path):
    cache = tmp_path / "cache.json"
    output = tmp_path / "map.json"
    cache.write_text(
        json.dumps(
            [
                {
                    "segment": "NSE_EQ",
                    "trading_symbol": "RELIANCE",
                    "instrument_key": "NSE_EQ|INE002A01018",
                }
            ]
        ),
        encoding="utf-8",
    )
    universe = (
        UpstoxInstrumentMap("RELIANCE.db", "Reliance", "NSE_EQ", "RELIANCE"),
        UpstoxInstrumentMap("TCS.db", "TCS", "NSE_EQ", "TCS"),
    )

    resolved, unresolved = resolve_instruments(
        universe, cache_path=cache, output_path=output
    )

    assert resolved[0].instrument_key == "NSE_EQ|INE002A01018"
    assert [item.database_name for item in unresolved] == ["TCS.db"]
    assert output.exists()


def test_candle_builder_writes_only_completed_aligned_m5(tmp_path):
    database = tmp_path / "RELIANCE.db"
    builder = M5CandleBuilder({"key": database})

    builder.process_tick(
        {
            "instrument_key": "key",
            "timestamp": pd.Timestamp("2026-01-01 10:01:00+00:00"),
            "ltp": 100,
            "volume": 2,
        }
    )
    builder.process_tick(
        {
            "instrument_key": "key",
            "timestamp": pd.Timestamp("2026-01-01 10:04:59+00:00"),
            "ltp": 103,
            "volume": 3,
        }
    )
    assert not database.exists()

    inserted = builder.process_tick(
        {
            "instrument_key": "key",
            "timestamp": pd.Timestamp("2026-01-01 10:05:00+00:00"),
            "ltp": 101,
            "volume": None,
        }
    )

    assert inserted == 1
    with sqlite3.connect(database) as connection:
        row = connection.execute(
            "SELECT timestamp, open, high, low, close, volume FROM m5_candles"
        ).fetchone()
    assert row == ("2026-01-01T10:00:00Z", 100.0, 103.0, 100.0, 103.0, 5.0)
    assert pd.Timestamp(row[0]).minute % 5 == 0


def test_flush_does_not_write_still_forming_bucket(tmp_path):
    database = tmp_path / "TCS.db"
    builder = M5CandleBuilder({"key": database})
    builder.process_tick(
        {
            "instrument_key": "key",
            "timestamp": pd.Timestamp("2026-01-01 10:02:00+00:00"),
            "ltp": 50,
        }
    )

    assert builder.flush_completed(pd.Timestamp("2026-01-01 10:04:00+00:00")) == 0
    assert not database.exists()
    assert builder.flush_completed(pd.Timestamp("2026-01-01 10:05:00+00:00")) == 1


def test_dry_run_and_missing_token_are_clean(tmp_path, monkeypatch):
    cache = tmp_path / "cache.json"
    cache.write_text("[]", encoding="utf-8")
    monkeypatch.delenv("UPSTOX_ACCESS_TOKEN", raising=False)

    assert main(["--dry-run", "--root", str(tmp_path), "--cache", str(cache)]) == 0
    assert main(["--root", str(tmp_path), "--cache", str(cache)]) == 2


def test_symbol_subset_does_not_require_manual_subscription():
    selected = select_universe(["HDFC_BANK.db", "RELIANCE.db"])
    assert {item.database_name for item in selected} == {
        "HDFC_BANK.db",
        "RELIANCE.db",
    }


def test_token_loader_reads_streamlit_secrets(tmp_path, monkeypatch):
    monkeypatch.delenv("UPSTOX_ACCESS_TOKEN", raising=False)
    secrets_dir = tmp_path / ".streamlit"
    secrets_dir.mkdir()
    (secrets_dir / "secrets.toml").write_text(
        'UPSTOX_ACCESS_TOKEN = "secret-token"\n',
        encoding="utf-8",
    )

    assert load_upstox_access_token(root=tmp_path) == "secret-token"


def test_chunk_date_range_splits_six_month_backfill_into_safe_windows():
    windows = chunk_date_range(date(2026, 1, 1), date(2026, 2, 15), chunk_days=28)

    assert windows == (
        (date(2026, 1, 1), date(2026, 1, 28)),
        (date(2026, 1, 29), date(2026, 2, 15)),
    )


def test_candles_to_m5_frame_normalizes_and_sorts(monkeypatch):
    payload = {
        "data": {
            "candles": [
                ["2026-07-20T09:20:00+05:30", 101, 103, 100, 102, 12, 0],
                ["2026-07-20T09:15:00+05:30", 100, 102, 99, 101, 10, 0],
            ]
        }
    }

    frame = candles_to_m5_frame(payload)

    assert list(frame.columns) == ["open", "high", "low", "close", "volume"]
    assert list(frame.index) == [
        pd.Timestamp("2026-07-20T03:45:00Z"),
        pd.Timestamp("2026-07-20T03:50:00Z"),
    ]
    assert frame.iloc[0]["close"] == 101.0
    assert frame.iloc[1]["volume"] == 12.0


def test_candles_to_m5_frame_drops_forming_bucket(monkeypatch):
    class FrozenTimestamp(pd.Timestamp):
        @classmethod
        def now(cls, tz=None):
            return pd.Timestamp("2026-07-20T10:32:00Z")

    monkeypatch.setattr(pd, "Timestamp", FrozenTimestamp)
    payload = {
        "data": {
            "candles": [
                ["2026-07-20T10:30:00+00:00", 101, 103, 100, 102, 12, 0],
                ["2026-07-20T10:25:00+00:00", 100, 102, 99, 101, 10, 0],
            ]
        }
    }

    frame = candles_to_m5_frame(payload, drop_forming=True)

    assert list(frame.index) == [pd.Timestamp("2026-07-20T10:25:00Z")]
