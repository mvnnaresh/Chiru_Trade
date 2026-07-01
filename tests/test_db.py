import sqlite3

import pandas as pd
import pytest

from db import (
    get_asset_profile,
    ingest_m5,
    load_m5,
    resample_ohlcv,
    set_asset_profile,
    setup_database,
)


def make_candles(periods=48, start="2026-01-01 00:00:00+00:00"):
    timestamps = pd.date_range(start, periods=periods, freq="5min")
    sequence = pd.Series(range(periods), dtype=float)
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": 100.0 + sequence,
            "high": 101.0 + sequence,
            "low": 99.0 + sequence,
            "close": 100.5 + sequence,
            "volume": 10.0 + sequence,
        }
    )


def test_setup_database_is_idempotent_and_records_base_timeframe(tmp_path):
    database = tmp_path / "market.sqlite"
    setup_database(database)
    setup_database(database)

    with sqlite3.connect(database) as connection:
        value = connection.execute(
            "SELECT value FROM metadata WHERE key = 'base_timeframe'"
        ).fetchone()
        table = connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='m5_candles'"
        ).fetchone()

    assert value == ("5min",)
    assert table == ("m5_candles",)


def test_ingestion_round_trip_is_sorted_utc_and_idempotent(tmp_path):
    database = tmp_path / "market.sqlite"
    candles = make_candles(periods=3).iloc[::-1]

    assert ingest_m5(database, candles) == 3
    assert ingest_m5(database, candles) == 0
    loaded = load_m5(database)

    assert list(loaded.columns) == ["open", "high", "low", "close", "volume"]
    assert loaded.index.is_monotonic_increasing
    assert str(loaded.index.tz) == "UTC"
    assert loaded.iloc[0].to_dict() == {
        "open": 100.0,
        "high": 101.0,
        "low": 99.0,
        "close": 100.5,
        "volume": 10.0,
    }


def test_existing_raw_candle_is_immutable(tmp_path):
    database = tmp_path / "market.sqlite"
    candle = make_candles(periods=1)
    ingest_m5(database, candle)
    revised = candle.assign(close=100.75)

    assert ingest_m5(database, revised) == 0
    assert load_m5(database).iloc[0]["close"] == 100.5


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda frame: frame.drop(columns="volume"), "missing candle columns"),
        (lambda frame: frame.assign(volume=-1), "volume must be non-negative"),
        (lambda frame: frame.assign(high=50), "internally inconsistent"),
        (
            lambda frame: frame.assign(
                timestamp=frame["timestamp"] + pd.Timedelta(minutes=1)
            ),
            "five-minute boundaries",
        ),
    ],
)
def test_ingestion_rejects_invalid_candles(tmp_path, mutation, message):
    with pytest.raises(ValueError, match=message):
        ingest_m5(tmp_path / "market.sqlite", mutation(make_candles(periods=1)))


def test_load_m5_applies_inclusive_time_bounds(tmp_path):
    database = tmp_path / "market.sqlite"
    ingest_m5(database, make_candles(periods=4))

    loaded = load_m5(
        database,
        start="2026-01-01T00:05:00Z",
        end="2026-01-01T00:10:00Z",
    )

    assert list(loaded.index) == list(
        pd.date_range("2026-01-01 00:05:00+00:00", periods=2, freq="5min")
    )


def test_resample_one_hour_uses_ohlcv_semantics(tmp_path):
    database = tmp_path / "market.sqlite"
    candles = make_candles(periods=12)
    ingest_m5(database, candles)

    result = resample_ohlcv(database, "1H")

    assert len(result) == 1
    assert result.index[0] == pd.Timestamp("2026-01-01T00:00:00Z")
    assert result.iloc[0].to_dict() == {
        "open": 100.0,
        "high": 112.0,
        "low": 99.0,
        "close": 111.5,
        "volume": candles["volume"].sum(),
    }


def test_resample_four_hour_is_epoch_aligned(tmp_path):
    database = tmp_path / "market.sqlite"
    ingest_m5(database, make_candles(periods=48, start="2026-01-01T04:00:00Z"))

    result = resample_ohlcv(database, "4h")

    assert len(result) == 1
    assert result.index[0] == pd.Timestamp("2026-01-01T04:00:00Z")


def test_resample_excludes_incomplete_or_gapped_bars_by_default(tmp_path):
    database = tmp_path / "market.sqlite"
    candles = make_candles(periods=24).drop(index=5)
    ingest_m5(database, candles)

    complete = resample_ohlcv(database, "1H")
    including_partial = resample_ohlcv(database, "1H", complete_only=False)

    assert list(complete.index) == [pd.Timestamp("2026-01-01T01:00:00Z")]
    assert len(including_partial) == 2


@pytest.mark.parametrize(
    ("timeframe", "complete_count", "partial_count"),
    (("15M", 3, 2), ("30M", 6, 5), ("2H", 24, 23)),
)
def test_new_fixed_timeframes_drop_partial_periods(
    tmp_path, timeframe, complete_count, partial_count
):
    database = tmp_path / f"{timeframe}.sqlite"
    ingest_m5(database, make_candles(periods=complete_count + partial_count))

    complete = resample_ohlcv(database, timeframe)
    including_partial = resample_ohlcv(database, timeframe, complete_only=False)

    assert list(complete.index) == [pd.Timestamp("2026-01-01T00:00:00Z")]
    assert len(including_partial) == 2


def test_daily_crypto_profile_requires_complete_24_hour_period(tmp_path):
    database = tmp_path / "BTC.sqlite"
    set_asset_profile(database, "24_7")
    ingest_m5(database, make_candles(periods=288 + 20))

    complete = resample_ohlcv(database, "1D")
    including_partial = resample_ohlcv(database, "1D", complete_only=False)

    assert get_asset_profile(database).label == "24_7"
    assert list(complete.index) == [pd.Timestamp("2026-01-01T00:00:00Z")]
    assert len(including_partial) == 2


def test_daily_nse_profile_uses_local_session_candle_count(tmp_path):
    database = tmp_path / "NIFTY.sqlite"
    set_asset_profile(database, "NSE")
    first_session = make_candles(
        periods=75, start="2026-01-01T03:45:00Z"
    )
    partial_session = make_candles(
        periods=20, start="2026-01-02T03:45:00Z"
    )
    ingest_m5(database, pd.concat((first_session, partial_session)))

    complete = resample_ohlcv(database, "1D")
    including_partial = resample_ohlcv(database, "1D", complete_only=False)

    assert get_asset_profile(database).label == "NSE"
    assert list(complete.index) == [pd.Timestamp("2025-12-31T18:30:00Z")]
    assert len(including_partial) == 2


def test_weekly_crypto_profile_requires_seven_complete_days(tmp_path):
    database = tmp_path / "BTC.sqlite"
    set_asset_profile(database, "24_7")
    ingest_m5(
        database,
        make_candles(periods=(7 * 288) + 20, start="2026-01-05T00:00:00Z"),
    )

    complete = resample_ohlcv(database, "1W")
    including_partial = resample_ohlcv(database, "1W", complete_only=False)

    assert list(complete.index) == [pd.Timestamp("2026-01-05T00:00:00Z")]
    assert len(including_partial) == 2


def test_weekly_nse_profile_requires_five_complete_sessions(tmp_path):
    database = tmp_path / "NIFTY.sqlite"
    set_asset_profile(database, "NSE")
    complete_sessions = [
        make_candles(periods=75, start=f"2026-01-{day:02d}T03:45:00Z")
        for day in range(5, 10)
    ]
    partial_next_week = make_candles(
        periods=20, start="2026-01-12T03:45:00Z"
    )
    ingest_m5(database, pd.concat((*complete_sessions, partial_next_week)))

    complete = resample_ohlcv(database, "1W")
    including_partial = resample_ohlcv(database, "1W", complete_only=False)

    assert list(complete.index) == [pd.Timestamp("2026-01-04T18:30:00Z")]
    assert len(including_partial) == 2


def test_resample_rejects_unsupported_timeframe(tmp_path):
    with pytest.raises(ValueError, match="1H.*4H"):
        resample_ohlcv(tmp_path / "market.sqlite", "1M")
