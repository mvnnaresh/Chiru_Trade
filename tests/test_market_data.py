from pathlib import Path

import pandas as pd

from db import ASSET_PROFILES, get_asset_profile, load_m5
from market_data import (
    MarketSymbol,
    append_latest_m5,
    download_m5,
    replace_database,
    resolve_market_symbol,
)


def _provider_frame() -> pd.DataFrame:
    index = pd.date_range("2026-01-01", periods=3, freq="5min", tz="UTC")
    columns = pd.MultiIndex.from_product(
        [["Open", "High", "Low", "Close", "Adj Close", "Volume"], ["TEST"]]
    )
    return pd.DataFrame(
        [
            [100, 102, 99, 101, 101, 10],
            [101, 103, 100, 102, 102, 11],
            [102, 104, 101, 103, 103, 12],
        ],
        index=index,
        columns=columns,
    )


def test_download_m5_normalizes_yahoo_multiindex(monkeypatch):
    monkeypatch.setattr("market_data.yf.download", lambda *args, **kwargs: _provider_frame())
    result = download_m5("TEST")
    assert list(result.columns) == ["open", "high", "low", "close", "volume"]
    assert str(result.index.tz) == "UTC"
    assert len(result) == 3


def test_replace_database_is_canonical_and_records_profile(tmp_path, monkeypatch):
    monkeypatch.setattr("market_data.download_m5", lambda *args, **kwargs: _provider_frame().droplevel(1, axis=1).rename(columns=str.lower).drop(columns="adj close"))
    market = MarketSymbol("TEST.NS", "TEST.db", "NSE", "Test Asset")
    destination, rows, start, end = replace_database(market, tmp_path)
    assert destination == Path(tmp_path) / "TEST.db"
    assert rows == 3
    assert len(load_m5(destination)) == 3
    assert get_asset_profile(destination).label == "NSE"
    assert start < end


def test_cross_asset_profiles_have_distinct_calendar_expectations():
    assert ASSET_PROFILES["US_EQUITY"].expected_daily_m5_count == 78
    assert ASSET_PROFILES["FOREX"].days_per_week == 5
    assert ASSET_PROFILES["FUTURES"].expected_daily_m5_count == 276


def test_resolve_market_symbol_uses_registry_and_metadata(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "market_data.download_m5",
        lambda *args, **kwargs: _provider_frame()
        .droplevel(1, axis=1)
        .rename(columns=str.lower)
        .drop(columns="adj close"),
    )
    market = MarketSymbol("TEST.NS", "TEST.db", "NSE", "Test Asset")
    destination, *_ = replace_database(market, tmp_path)

    assert resolve_market_symbol(destination) == market


def test_append_latest_m5_appends_without_replacing_existing_rows(
    tmp_path, monkeypatch
):
    first = _provider_frame().droplevel(1, axis=1).rename(columns=str.lower).drop(
        columns="adj close"
    )
    second = pd.concat(
        [
            first,
            pd.DataFrame(
                [[103, 105, 102, 104, 13]],
                index=pd.DatetimeIndex(
                    [pd.Timestamp("2026-01-01 00:15:00+00:00")]
                ),
                columns=["open", "high", "low", "close", "volume"],
            ),
        ]
    )
    calls = iter((first, second))
    monkeypatch.setattr(
        "market_data.download_m5", lambda *args, **kwargs: next(calls)
    )
    market = MarketSymbol("TEST.NS", "TEST.db", "NSE", "Test Asset")
    destination, *_ = replace_database(market, tmp_path)

    inserted, last_timestamp = append_latest_m5(destination)

    assert inserted == 1
    assert last_timestamp == pd.Timestamp("2026-01-01 00:15:00+00:00")
    assert len(load_m5(destination)) == 4
