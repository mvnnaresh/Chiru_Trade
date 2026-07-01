import sqlite3

from dataclasses import replace

import pandas as pd
import app as app_module
from db import TIMEFRAMES

from app import (
    DashboardResult,
    actionable_rankings,
    build_lightweight_charts,
    candidate_lifecycle,
    discover_databases,
    focus_dashboard,
    format_setup_alert,
    marker_status,
    pattern_rankings,
    recent_rankings,
    scan_global_markets,
    system_hints,
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


def test_trade_setup_markers_apply_overlay_threshold_and_direction():
    bullish = candidate()
    bearish = replace(bullish, direction="Bearish")
    index = pd.date_range("2026-01-01", periods=5, freq="5min", tz="UTC")
    candles = pd.DataFrame(
        {
            "open": [100] * 5,
            "high": [121] * 5,
            "low": [99] * 5,
            "close": [112] * 5,
            "volume": [1] * 5,
        },
        index=index,
    )
    result = DashboardResult(
        candles,
        bullish.pivots,
        ((bullish, score(85)), (bearish, score(90))),
        pd.Series(50, index=index, name="rsi"),
    )

    charts = build_lightweight_charts(
        result, (True, True, False), alert_threshold=80
    )
    markers = charts[0]["series"][0]["markers"]

    assert markers == [
        {
            "time": int(bullish.pivots[-1].timestamp.timestamp()),
            "position": "belowBar",
            "shape": "arrowUp",
            "color": "#18C98B",
            "text": "#1 Impulse (85.0)",
        },
        {
            "time": int(bearish.pivots[-1].timestamp.timestamp()),
            "position": "aboveBar",
            "shape": "arrowDown",
            "color": "#F05D68",
            "text": "#2 Impulse (90.0)",
        },
    ]


def test_trade_setup_markers_clear_when_disabled_below_gate_or_not_terminal():
    active = candidate()
    completed = replace(active, labels=("Start", "1", "2", "3", "5"))
    index = pd.date_range("2026-01-01", periods=5, freq="5min", tz="UTC")
    candles = pd.DataFrame(
        {
            "open": [100] * 5,
            "high": [121] * 5,
            "low": [99] * 5,
            "close": [112] * 5,
            "volume": [1] * 5,
        },
        index=index,
    )
    result = DashboardResult(
        candles,
        active.pivots,
        ((active, score(70)), (completed, score(95))),
        pd.Series(50, index=index, name="rsi"),
    )

    charts = build_lightweight_charts(
        result, (True, True, False), alert_threshold=75
    )

    assert charts[0]["series"][0]["markers"] == []


def test_marker_status_explains_threshold_and_direction():
    bullish = candidate()

    assert marker_status(((bullish, score(85)),), (True, False, False), 80) == (
        "80.0",
        "112.00",
        "No bearish setup",
    )
    assert marker_status(((bullish, score(70)),), (True, False, False), 80) == (
        "80.0",
        "Below threshold (70.0)",
        "Below threshold (70.0)",
    )


def test_marker_status_explains_absent_tradeable_setup():
    completed = replace(candidate(), labels=("Start", "1", "2", "3", "5"))

    assert marker_status(
        ((completed, score(95)),), (True, False, False), 75
    ) == ("75.0", "Awaiting Wave 4/B", "Awaiting Wave 4/B")
    assert marker_status(
        ((completed, score(95)),), (False, False, False), 75
    ) == ("75.0", "Overlay disabled", "Overlay disabled")


def test_system_hints_distinguish_completed_and_actionable_setups():
    active = candidate()

    actionable = system_hints(active, score(85), 75, "Active")
    completed = system_hints(active, score(85), 75, "Target hit")

    assert "Confidence gate: Passed (85.0 ≥ 75.0)" in actionable
    assert "Entry gate: Passed — terminal Wave 4/B is present" in actionable
    assert "Marker decision: Visible — Buy at 112.00" in actionable
    assert "Invalidation reference: 104.00 floor" in actionable
    assert "Lifecycle: Target hit" in completed
    assert "Entry gate: Failed — lifecycle is target hit" in completed
    assert "Marker decision: Hidden — setup target hit" in completed
    assert (
        "Trading interpretation: Historical structure; not a current trade signal"
        in completed
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


def test_global_scanner_sorts_active_setups_and_skips_failed_inputs(
    tmp_path, monkeypatch
):
    first = tmp_path / "BTC.db"
    second = tmp_path / "NIFTY.db"
    broken = tmp_path / "BROKEN.db"
    for database in (first, second, broken):
        database.touch()
    active = candidate()
    candles = pd.DataFrame(
        {
            "open": [100] * 5,
            "high": [101] * 5,
            "low": [105] * 5,
            "close": [100] * 5,
            "volume": 1,
        },
        index=pd.date_range("2026-01-01", periods=5, freq="5min", tz="UTC"),
    )

    def fake_compute(database, timeframe, _multiplier, _period):
        if database.name == "BROKEN.db":
            raise ValueError("invalid database")
        confidence = 90 if database.name == "NIFTY.db" else 70
        return DashboardResult(
            candles,
            active.pivots,
            ((active, score(confidence)),),
            pd.Series(50, index=candles.index),
        )

    monkeypatch.setattr(app_module, "compute_dashboard", fake_compute)

    frame, errors = scan_global_markets(
        (first, second, broken), atr_multiplier=2.0, atr_period=14
    )

    timeframe_count = len(TIMEFRAMES)
    assert list(frame["Confidence Score"]) == [90] * timeframe_count + [
        70
    ] * timeframe_count
    assert list(frame["Market"][:timeframe_count]) == ["NIFTY"] * timeframe_count
    assert set(frame["Timeframe"]) == set(TIMEFRAMES)
    assert frame.iloc[0]["Pattern"] == "Impulse"
    assert len(errors) == timeframe_count
