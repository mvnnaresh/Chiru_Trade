import sqlite3

import pandas as pd

from app import (
    DashboardResult,
    actionable_rankings,
    build_lightweight_charts,
    candidate_lifecycle,
    discover_databases,
    focus_dashboard,
    format_setup_alert,
    pattern_rankings,
    recent_rankings,
    target_zone,
)
from engine import RuleState, WaveCandidate
from pivots import Pivot
from scoring import ConfidenceScore


def candidate(terminal_label="4"):
    index = pd.date_range("2026-01-01", periods=5, freq="5min", tz="UTC")
    prices = (100, 110, 104, 120, 112)
    kinds = ("Low", "High", "Low", "High", "Low")
    pivots = tuple(
        Pivot(timestamp, float(price), kind, 1.0)
        for timestamp, price, kind in zip(index, prices, kinds)
    )
    labels = ("Start", "1", "2", "3", terminal_label)
    return WaveCandidate(
        "Impulse",
        "Bullish",
        pivots,
        labels,
        (RuleState("partial", True, "valid"),),
        104.0,
        "below",
    )


def score(total=80):
    return ConfidenceScore(40, 25, 15, total, ())


def test_database_discovery_is_filtered_and_sorted(tmp_path):
    for name in ("z.sqlite", "A.db", "ignore.txt", "b.sqlite3"):
        (tmp_path / name).touch()

    result = discover_databases(tmp_path)

    assert [path.name for path in result] == ["A.db", "b.sqlite3", "z.sqlite"]


def test_target_zone_uses_wave4_and_wave1_projection():
    low, high = target_zone(candidate())

    assert low == 122.0
    assert high == 128.18


def test_alert_is_only_created_for_threshold_crossing_at_wave4_or_b():
    active = candidate()

    message = format_setup_alert(active, score(80), 75, chat_id="123")

    assert message is not None
    assert "Confidence Score=80.00/100" in message
    assert "chat=123" in message
    assert format_setup_alert(active, score(70), 75, chat_id="123") is None
    assert format_setup_alert(candidate("5"), score(80), 75, chat_id="123") is None


def test_dashboard_result_is_safe_for_empty_sequences():
    candles = pd.DataFrame(
        columns=["open", "high", "low", "close", "volume"],
        index=pd.DatetimeIndex([], tz="UTC"),
    )
    result = DashboardResult(
        candles,
        (),
        (),
        pd.Series(index=candles.index, dtype=float, name="rsi"),
    )

    assert result.candles.empty
    assert result.pivots == ()
    assert result.rankings == ()


def test_lightweight_chart_specs_include_synced_price_rsi_and_wave_markers():
    active = candidate()
    index = pd.date_range("2026-01-01", periods=5, freq="5min", tz="UTC")
    candles = pd.DataFrame(
        {
            "open": [100, 108, 105, 115, 113],
            "high": [102, 111, 107, 121, 114],
            "low": [99, 107, 103, 114, 111],
            "close": [101, 110, 104, 120, 112],
            "volume": 1,
        },
        index=index,
    )
    result = DashboardResult(
        candles,
        active.pivots,
        ((active, score()),),
        pd.Series([40, 50, 60, 55, 45], index=index, name="rsi"),
    )

    charts = build_lightweight_charts(result, (True, False, False))

    assert len(charts) == 2
    assert charts[0]["series"][0]["type"] == "Candlestick"
    wave_series = charts[0]["series"][1]
    assert wave_series["type"] == "Line"
    assert [marker["text"] for marker in wave_series["markers"]] == [
        "Start",
        "1",
        "2",
        "3",
        "4",
    ]
    assert charts[1]["series"][0]["options"]["title"] == "RSI (14)"
    assert [series["data"][0]["value"] for series in charts[1]["series"][1:]] == [
        30.0,
        70.0,
    ]
    assert charts[0]["chart"]["timeScale"]["visible"] is False
    assert charts[1]["chart"]["timeScale"]["visible"] is True
    reference_lines = [
        series
        for series in charts[0]["series"]
        if series["type"] == "Line" and len(series["data"]) == 2
    ]
    assert len(reference_lines) == 3
    assert all(line["options"]["lastValueVisible"] is False for line in reference_lines)
    assert all(
        line["data"][0]["time"] == int(active.pivots[4].timestamp.timestamp())
        for line in reference_lines
    )


def test_recent_rankings_filters_by_candidate_completion_time():
    active = candidate()
    ranked = ((active, score()),)

    assert recent_rankings(ranked, active.pivots[-1].timestamp, days=1) == ranked
    assert recent_rankings(
        ranked, active.pivots[-1].timestamp + pd.Timedelta(days=31), days=30
    ) == ()


def test_pattern_views_surface_impulse_and_zigzag_families():
    impulse = candidate()
    zigzag = WaveCandidate(
        "ZigZag",
        impulse.direction,
        impulse.pivots[:4],
        ("Start", "A", "B", "C"),
        impulse.rule_states,
        impulse.invalidation_level,
        impulse.invalidation_side,
    )
    impulse_item = (impulse, score(60))
    zigzag_item = (zigzag, score(90))
    rankings = (zigzag_item, impulse_item)

    assert pattern_rankings(rankings, "Balanced · 1–5 + ABC")[:2] == (
        impulse_item,
        zigzag_item,
    )
    assert pattern_rankings(rankings, "Impulse · 1–5") == (impulse_item,)
    assert pattern_rankings(rankings, "ZigZag · ABC") == (zigzag_item,)


def test_candidate_lifecycle_and_actionable_filtering():
    active = candidate()
    index = pd.date_range("2026-01-01", periods=7, freq="5min", tz="UTC")
    base = pd.DataFrame(
        {
            "open": [100] * 7,
            "high": [101] * 7,
            "low": [105] * 7,
            "close": [100] * 7,
            "volume": 1,
        },
        index=index,
    )
    ranked = ((active, score()),)

    assert candidate_lifecycle(active, base) == "Active"
    assert actionable_rankings(ranked, base) == ranked

    target_hit = base.copy()
    target_hit.loc[index[-1], "high"] = 130
    assert candidate_lifecycle(active, target_hit) == "Target hit"
    assert actionable_rankings(ranked, target_hit) == ()

    invalidated = base.copy()
    invalidated.loc[index[-1], "low"] = 103
    assert candidate_lifecycle(active, invalidated) == "Invalidated"


def test_focus_dashboard_slices_candles_and_rsi_around_selected_path():
    active = candidate()
    index = pd.date_range(
        active.pivots[0].timestamp - pd.Timedelta(minutes=20),
        periods=15,
        freq="5min",
    )
    candles = pd.DataFrame(
        {
            "open": 100,
            "high": 101,
            "low": 99,
            "close": 100,
            "volume": 1,
        },
        index=index,
    )
    result = DashboardResult(
        candles, active.pivots, ((active, score()),), pd.Series(50, index=index)
    )

    focused = focus_dashboard(result, active, padding_bars=1)

    assert focused.candles.index[0] == index[3]
    assert focused.candles.index[-1] == index[9]
    assert focused.rsi.index.equals(focused.candles.index)
