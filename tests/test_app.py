import sqlite3

from dataclasses import replace

import pandas as pd
import app as app_module
from db import TIMEFRAMES
from decision import DecisionState
from backtester import Friction

from app import (
    _offset_timestamp,
    _candidate_query_key,
    _resolve_selected_index,
    DashboardResult,
    RecommendationState,
    SIGNAL_STRICTNESS_THRESHOLDS,
    SENSITIVITY_PRESETS,
    actionable_rankings,
    build_lightweight_charts,
    candidate_lifecycle,
    current_wave_label,
    discover_databases,
    decision_panel_markup,
    emit_alert_shell,
    fallback_rankings_for_view,
    focus_dashboard,
    format_setup_alert,
    live_rankings,
    marker_status,
    pattern_rankings,
    refresh_live_database,
    refresh_live_database_state,
    recent_rankings,
    resolve_sensitivity,
    resolve_setup_quality_threshold,
    scan_global_markets,
    scanner_candidate_row,
    system_status,
    system_hints,
    target_zone,
    trader_recommendation,
    wave_three_window,
)
from engine import RuleState, WaveCandidate
from pivots import ActiveLeg, Pivot
from scoring import ConfidenceScore
from risk import RiskPolicy


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
        status="EntryReady" if terminal_label in {"4", "B"} else "Completed",
        as_of=index[-1],
    )


def zigzag_candidate(direction="Bearish", *, as_of=None):
    index = pd.date_range("2026-01-01", periods=3, freq="5min", tz="UTC")
    prices = (120, 100, 110) if direction == "Bearish" else (100, 120, 110)
    kinds = ("High", "Low", "High") if direction == "Bearish" else ("Low", "High", "Low")
    pivots = tuple(
        Pivot(timestamp, float(price), kind, 1.0)
        for timestamp, price, kind in zip(index, prices, kinds)
    )
    return WaveCandidate(
        "ZigZag",
        direction,
        pivots,
        ("Start", "A", "B"),
        (RuleState("fixture", True, "valid"),),
        120.0 if direction == "Bearish" else 100.0,
        "above" if direction == "Bearish" else "below",
        status="EntryReady",
        as_of=as_of,
    )


def score(total=80):
    return ConfidenceScore(40, 25, 15, total, ())


def test_resolve_sensitivity_uses_presets_and_optional_override():
    assert resolve_sensitivity("Tight") == SENSITIVITY_PRESETS["Tight"]
    assert resolve_sensitivity("Balanced") == SENSITIVITY_PRESETS["Balanced"]
    assert resolve_sensitivity("Conservative") == SENSITIVITY_PRESETS["Conservative"]
    assert resolve_sensitivity(
        "Balanced",
        override_enabled=True,
        atr_multiplier=2.4,
        atr_period=18,
    ) == (2.4, 18)


def test_setup_quality_threshold_resolves_strictness_and_override():
    assert resolve_setup_quality_threshold("Balanced") == 70
    assert resolve_setup_quality_threshold("Aggressive") == 65
    assert resolve_setup_quality_threshold("Conservative") == 75
    assert resolve_setup_quality_threshold("Very strict") == 80
    assert SIGNAL_STRICTNESS_THRESHOLDS["Balanced"] == 70
    assert resolve_setup_quality_threshold(
        "Balanced", override_enabled=True, override_value=68.5
    ) == 68.5


def test_system_status_reflects_live_and_offline_states():
    assert system_status(live_enabled=False, live_supported=True) == (
        "LOCAL DATA MODE",
        "#F0B90B",
    )
    assert system_status(live_enabled=True, live_supported=False) == (
        "LOCAL DATA MODE",
        "#F0B90B",
    )
    assert system_status(
        live_enabled=True,
        live_supported=True,
        live_refresh_ok=False,
    ) == ("LIVE REFRESH ERROR", "#F05D68")
    assert system_status(
        live_enabled=True,
        live_supported=True,
        live_refresh_ok=True,
    ) == ("SYSTEM LIVE", "#18C98B")


def test_trader_recommendation_maps_to_buy_watch_and_no_trade_states():
    active = candidate()

    buy = trader_recommendation(active, score(85), 75, "Active")
    assert isinstance(buy, RecommendationState)
    assert buy.action == "BUY"
    assert buy.entry_text == "112.00"

    watch = trader_recommendation(active, score(70), 75, "Active")
    assert watch.action == "WATCH"
    assert "Below confidence threshold" in watch.status_text

    no_trade = trader_recommendation(active, score(85), 75, "Target hit")
    assert no_trade.action == "NO TRADE"
    assert "lifecycle is target hit" in no_trade.status_text.lower()

    forming = replace(
        active,
        pivots=active.pivots[:4],
        labels=("Start", "1", "2", "3"),
        status="Forming",
        active_leg=ActiveLeg(
            active.pivots[-1].timestamp, 112.0, "Low", 1.0, "down"
        ),
        forming_label="4",
    )
    watch_forming = trader_recommendation(forming, score(85), 75, "Forming")
    assert watch_forming.action == "WATCH"
    assert watch_forming.status_text == "Forming Wave 4"


def test_decision_panel_marks_watch_targets_provisional_and_shows_action():
    markup = decision_panel_markup(
        DecisionState(
            status="WATCH",
            direction="Bearish",
            color="#F0B90B",
            reason="Forming.",
            second_reason="Wait.",
            current_price=24131.8,
            target_1=23462.0,
            target_2=23728.3,
            stage="Forming",
        )
    )

    assert "Current price" in markup and "24,131.80" in markup
    assert "Provisional target 1" in markup
    assert "Provisional target 2" in markup
    assert "Action: Do not trade yet. Wait for Wave 4/B confirmation." in markup


def test_offset_timestamp_reports_completed_through_time_for_fixed_bars():
    assert _offset_timestamp(pd.Timestamp("2026-01-01 09:00:00+00:00"), "1H") == (
        pd.Timestamp("2026-01-01 10:00:00+00:00")
    )


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


def test_alert_shell_handles_missing_candidate(monkeypatch):
    warnings = []
    monkeypatch.setattr("streamlit.warning", warnings.append)

    assert emit_alert_shell(None, None) is False
    assert warnings == ["No candidate is selected for an alert."]


def test_refresh_live_database_formats_provider_summary(monkeypatch, tmp_path):
    path = tmp_path / "BTC.db"
    monkeypatch.setattr(
        app_module,
        "append_latest_m5",
        lambda *_args, **_kwargs: (2, pd.Timestamp("2026-01-01 12:35:00+00:00")),
    )

    result = refresh_live_database(path)

    assert "Yahoo live" in result
    assert "01 Jan 12:35 UTC" in result
    assert "+2 rows" in result


def test_refresh_live_database_state_returns_auditable_fields(monkeypatch, tmp_path):
    path = tmp_path / "BTC.db"
    monkeypatch.setattr(
        app_module,
        "append_latest_m5",
        lambda *_args, **_kwargs: (0, pd.Timestamp("2026-01-01 12:35:00+00:00")),
    )

    result = refresh_live_database_state(path)

    assert result.ok is True
    assert result.rows_added == 0
    assert result.last_completed_bar == pd.Timestamp("2026-01-01 12:35:00+00:00")
    assert "Yahoo live" in result.message


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
    assert [line["options"]["title"] for line in reference_lines] == [
        "Invalidation",
        "Target 1.000",
        "Target 1.618",
    ]
    assert all(line["options"]["lastValueVisible"] is True for line in reference_lines)
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


def test_forming_overlay_is_dashed_and_labels_unconfirmed_endpoint():
    ready = candidate()
    active_leg = ActiveLeg(
        ready.pivots[-1].timestamp,
        ready.pivots[-1].price,
        "Low",
        1.0,
        "down",
    )
    forming = replace(
        ready,
        pivots=ready.pivots[:4],
        labels=ready.labels[:4],
        status="Forming",
        active_leg=active_leg,
        forming_label="4",
    )
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
        forming.pivots,
        ((forming, score(85)),),
        pd.Series(50, index=index, name="rsi"),
    )

    charts = build_lightweight_charts(
        result, (True, False, False), alert_threshold=75
    )
    wave = charts[0]["series"][1]

    assert wave["options"]["lineStyle"] == 2
    assert wave["options"]["title"] == "#1 Impulse · Forming"
    assert wave["data"][-1] == {
        "time": int(active_leg.timestamp.timestamp()),
        "value": active_leg.price,
    }
    assert wave["markers"][-1]["text"] == "4?"
    assert charts[0]["series"][0]["markers"] == []


def test_current_wave_and_wave_three_window_are_exposed():
    active = candidate()

    wave_three = wave_three_window(active)

    assert current_wave_label(active) == "EntryReady at Wave 4"
    assert wave_three is not None
    assert wave_three[0][0] == "Wave 3 start"
    assert wave_three[0][1] == active.pivots[2]
    assert wave_three[1][0] == "Wave 3 end"
    assert wave_three[1][1] == active.pivots[3]


def test_selected_candidate_query_key_round_trips_to_index_resolution():
    first = candidate()
    second = replace(
        candidate(),
        pattern="ZigZag",
        pivots=candidate().pivots[:4],
        labels=("Start", "A", "B", "C"),
    )
    rankings = ((first, score(80)), (second, score(90)))

    selected_key = _candidate_query_key(second)

    assert _resolve_selected_index(rankings, selected_key) == 1
    assert _resolve_selected_index(rankings, "missing") == 0


def test_live_rankings_prioritize_current_market_edge_over_raw_score():
    ready = candidate()
    forming = replace(
        ready,
        pivots=ready.pivots[:4],
        labels=("Start", "1", "2", "3"),
        status="Forming",
        active_leg=ActiveLeg(
            ready.pivots[-1].timestamp + pd.Timedelta(minutes=5),
            111.0,
            "Low",
            1.0,
            "down",
        ),
        forming_label="4",
        as_of=ready.pivots[-1].timestamp + pd.Timedelta(minutes=5),
    )
    completed = replace(
        ready,
        status="Completed",
        labels=("Start", "1", "2", "3", "4"),
        as_of=None,
    )
    rankings = (
        (completed, score(95)),
        (ready, score(80)),
        (forming, score(70)),
    )

    ordered = live_rankings(rankings)

    assert [item[0].status for item in ordered] == [
        "Forming",
        "EntryReady",
        "Completed",
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
        "No bearish setup",
    )


def test_marker_status_explains_absent_tradeable_setup():
    completed = replace(candidate(), labels=("Start", "1", "2", "3", "5"))

    assert marker_status(
        ((completed, score(95)),), (True, False, False), 75
    ) == ("75.0", "Awaiting Wave 4/B", "No bearish setup")
    assert marker_status(
        ((completed, score(95)),), (False, False, False), 75
    ) == ("75.0", "Overlay disabled", "Overlay disabled")

    forming = replace(
        candidate(),
        pivots=candidate().pivots[:4],
        labels=("Start", "1", "2", "3"),
        status="Forming",
        active_leg=ActiveLeg(
            candidate().pivots[-1].timestamp, 112, "Low", 1, "down"
        ),
        forming_label="4",
    )
    assert marker_status(
        ((forming, score(95)),), (True, False, False), 75
    ) == ("75.0", "Forming Wave 4", "No bearish setup")

    bearish_forming = replace(forming, direction="Bearish", forming_label="B")
    assert marker_status(
        ((bearish_forming, score(70)),), (True, False, False), 75
    ) == ("75.0", "No bullish setup", "Forming Wave B")


def test_system_hints_distinguish_completed_and_actionable_setups():
    active = candidate()

    actionable = system_hints(active, score(85), 75, "Active")
    completed = system_hints(active, score(85), 75, "Target hit")
    assert "Current wave: EntryReady at Wave 4" in actionable
    assert any(item.startswith("Wave 3 start:") for item in actionable)
    assert any(item.startswith("Wave 3 end:") for item in actionable)

    assert "Setup quality gate: Passed (85.0 >= 75.0)" in actionable
    assert "Entry gate: Passed - terminal Wave 4/B is present" in actionable
    assert "Marker decision: BUY setup present" in actionable
    assert "BUY setup pivot: 112.00" in actionable
    assert "Invalidation reference: 104.00 floor" in actionable
    assert "Lifecycle: Target hit" in completed
    assert "Entry gate: Failed - lifecycle is target hit" in completed
    assert "Marker decision: Hidden - setup target hit" in completed
    assert (
        "Trading interpretation: Historical structure; not a current trade signal"
        in completed
    )


def test_system_hints_match_blocked_and_trade_ready_decisions():
    active = candidate()
    blocked = DecisionState(
        status="BLOCKED",
        direction="Bullish",
        color="#F05D68",
        reason="Reward-to-risk below policy minimum",
        second_reason="Risk rejected.",
        risk_reason="Reward-to-risk below policy minimum",
        stage="EntryReady",
    )
    blocked_hints = system_hints(
        active, score(85), 75, "Active", decision_state=blocked
    )
    blocked_text = " | ".join(blocked_hints)
    assert "Risk gate: Failed" in blocked_text
    assert "trade blocked by risk policy" in blocked_text
    assert "Actionable buy setup" not in blocked_text
    assert "Buy at 112.00" not in blocked_text

    ready = replace(blocked, status="TRADE READY", risk_reason="Approved")
    ready_hints = system_hints(
        active, score(85), 75, "Active", decision_state=ready
    )
    assert "Risk gate: Passed" in ready_hints
    assert "Marker decision: BUY setup ready" in ready_hints


def test_system_hints_use_forming_pattern_state_for_provisional_zigzag():
    bearish = WaveCandidate(
        "ZigZag",
        "Bearish",
        candidate().pivots[:2],
        ("Start", "A"),
        (RuleState("partial", True, "valid"),),
        120.0,
        "above",
        status="Forming",
        active_leg=ActiveLeg(
            candidate().pivots[-1].timestamp, 110.0, "High", 1.0, "up"
        ),
        forming_label="B",
        as_of=candidate().pivots[-1].timestamp,
    )

    hints = system_hints(bearish, score(68), 75, "Forming")

    assert (
        "Pattern state: Bearish ZigZag is currently forming Wave B" in hints
    )
    assert "Current wave: Forming Wave B" in hints


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


def test_entryready_zigzag_lifecycle_uses_b_pivot_not_latest_as_of():
    for direction in ("Bearish", "Bullish"):
        provisional = zigzag_candidate(direction)
        future_time = provisional.pivots[-1].timestamp + pd.Timedelta(minutes=5)
        item = replace(provisional, as_of=future_time)
        if direction == "Bearish":
            row = {"open": 100, "high": 105, "low": 89, "close": 90, "volume": 1}
        else:
            row = {"open": 120, "high": 131, "low": 115, "close": 130, "volume": 1}
        candles = pd.DataFrame([row], index=[future_time])

        assert candidate_lifecycle(item, candles) == "Target hit"


def test_actionable_scope_prioritizes_entry_ready_then_forming():
    ready = candidate()
    completed = replace(
        ready,
        status="Completed",
        labels=("Start", "1", "2", "3", "5"),
    )
    forming = replace(
        ready,
        pivots=ready.pivots[:4],
        labels=("Start", "1", "2", "3"),
        status="Forming",
        active_leg=ActiveLeg(
            ready.pivots[-1].timestamp, 112, "Low", 1, "down"
        ),
        forming_label="4",
    )
    candles = pd.DataFrame(
        {
            "open": [110],
            "high": [111],
            "low": [109],
            "close": [110],
            "volume": [1],
        },
        index=pd.DatetimeIndex([ready.pivots[-1].timestamp]),
    )

    ordered = actionable_rankings(
        (
            (completed, score(99)),
            (forming, score(70)),
            (ready, score(80)),
        ),
        candles,
    )

    assert [item[0].status for item in ordered] == [
        "EntryReady",
        "Forming",
        "Completed",
    ]


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
        (first, second, broken),
        atr_multiplier=2.0,
        atr_period=14,
        evaluate_risk=False,
    )

    timeframe_count = len(TIMEFRAMES)
    assert list(frame["Setup Quality Score"]) == [90] * timeframe_count + [
        70
    ] * timeframe_count
    assert set(frame["Trade Decision"]) == {"WATCH"}
    assert "TRADE READY" not in set(frame["Trade Decision"])

    strict_frame, _ = scan_global_markets(
        (first, second), 2.0, 14,
        setup_quality_threshold=75,
        evaluate_risk=False,
    )
    low_quality = strict_frame[strict_frame["Setup Quality Score"] == 70]
    high_quality = strict_frame[strict_frame["Setup Quality Score"] == 90]
    assert set(low_quality["Structure Status"]) == {"BUY SETUP"}
    assert set(high_quality["Structure Status"]) == {"BUY SETUP"}
    assert list(frame["Market"][:timeframe_count]) == ["NIFTY"] * timeframe_count
    assert set(frame["Timeframe"]) == set(TIMEFRAMES)
    assert frame.iloc[0]["Pattern"] == "Impulse"
    assert frame.iloc[0]["Current Wave"] == "EntryReady at Wave 4"
    assert len(errors) == timeframe_count


def test_scanner_candidate_states_share_terminal_quality_and_risk_gates():
    active = candidate()
    candles = pd.DataFrame(
        {"open": [110], "high": [111], "low": [109], "close": [110], "volume": [1]},
        index=[active.pivots[-1].timestamp],
    )
    common = {
        "market": "TEST",
        "timeframe": "1H",
        "candles": candles,
        "setup_quality_threshold": 70,
        "risk_policy": RiskPolicy(100_000),
        "friction": Friction(),
    }

    forming = replace(
        active,
        pivots=active.pivots[:4],
        labels=active.labels[:4],
        status="Forming",
        active_leg=ActiveLeg(active.pivots[-1].timestamp, 112, "Low", 1, "down"),
        forming_label="4",
    )
    completed = replace(active, status="Completed")
    forming_row = scanner_candidate_row(
        candidate=forming, score=score(90), **common
    )
    completed_row = scanner_candidate_row(
        candidate=completed, score=score(90), **common
    )
    low_row = scanner_candidate_row(candidate=active, score=score(68.6), **common)
    blocked_row = scanner_candidate_row(
        candidate=active,
        score=score(80),
        **{**common, "risk_policy": RiskPolicy(100_000, minimum_reward_risk=3)},
    )
    ready_row = scanner_candidate_row(candidate=active, score=score(80), **common)
    unchecked_row = scanner_candidate_row(
        candidate=active, score=score(80), evaluate_risk=False, **common
    )

    assert (forming_row["Trade Decision"], forming_row["Structure Status"]) == (
        "WATCH", "WATCHLIST"
    )
    assert (completed_row["Trade Decision"], completed_row["Structure Status"]) == (
        "NO TRADE", "HISTORICAL"
    )
    assert low_row["Trade Decision"] == "WATCH"
    assert blocked_row["Trade Decision"] == "BLOCKED"
    assert ready_row["Trade Decision"] == "TRADE READY"
    assert isinstance(ready_row["Risk/Reward"], float)
    assert unchecked_row["Trade Decision"] == "WATCH"
    assert unchecked_row["Reason"] == "Needs risk check"
    assert unchecked_row["Risk/Reward"] == "Not evaluated"
    assert "SETUP" not in completed_row["Structure Status"]


def test_scanner_target_hit_is_historical():
    active = candidate()
    future_time = active.pivots[-1].timestamp + pd.Timedelta(minutes=5)
    candles = pd.DataFrame(
        {"open": [112], "high": [123], "low": [110], "close": [122], "volume": [1]},
        index=[future_time],
    )
    row = scanner_candidate_row(
        market="TEST",
        timeframe="1H",
        candidate=active,
        score=score(80),
        candles=candles,
        setup_quality_threshold=70,
        risk_policy=RiskPolicy(100_000),
        friction=Friction(),
    )

    assert row["Trade Decision"] == "NO TRADE"
    assert row["Structure Status"] == "HISTORICAL"
    assert row["Reason"] == "Target hit"


def test_scanner_sorts_by_decision_then_quality(tmp_path, monkeypatch):
    database = tmp_path / "TEST.db"
    database.touch()
    active = candidate()
    blocked = replace(
        active,
        pivots=active.pivots[:-1] + (replace(active.pivots[-1], price=105.0),),
    )
    forming = replace(
        active,
        pivots=active.pivots[:4],
        labels=active.labels[:4],
        status="Forming",
        active_leg=ActiveLeg(active.pivots[-1].timestamp, 112, "Low", 1, "down"),
        forming_label="4",
    )
    completed = replace(active, status="Completed")
    candles = pd.DataFrame(
        {"open": [110], "high": [111], "low": [109], "close": [110], "volume": [1]},
        index=[active.pivots[-1].timestamp],
    )

    monkeypatch.setattr(
        app_module,
        "compute_dashboard",
        lambda *_args: DashboardResult(
            candles,
            active.pivots,
            (
                (completed, score(99)),
                (forming, score(95)),
                (blocked, score(85)),
                (active, score(80)),
            ),
            pd.Series(50, index=candles.index),
        ),
    )
    frame, _ = scan_global_markets(
        (database,),
        2.0,
        14,
        70,
        RiskPolicy(100_000),
        Friction(),
    )

    order = {"TRADE READY": 0, "BLOCKED": 1, "WATCH": 2, "NO TRADE": 3}
    ranks = [order[value] for value in frame["Trade Decision"]]
    assert ranks == sorted(ranks)
