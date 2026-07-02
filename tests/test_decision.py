from dataclasses import replace

import pandas as pd

from backtester import Friction
from decision import build_decision_state
from engine import RuleState, WaveCandidate
from pivots import ActiveLeg, Pivot
from risk import RiskPolicy
from scoring import ConfidenceScore


def candidate(direction="Bullish", status="EntryReady"):
    index = pd.date_range("2026-01-01", periods=5, freq="5min", tz="UTC")
    prices = (100, 110, 104, 120, 112) if direction == "Bullish" else (120, 110, 116, 100, 108)
    kinds = ("Low", "High", "Low", "High", "Low") if direction == "Bullish" else ("High", "Low", "High", "Low", "High")
    return WaveCandidate(
        "Impulse", direction,
        tuple(Pivot(t, float(p), k, 1.0) for t, p, k in zip(index, prices, kinds)),
        ("Start", "1", "2", "3", "4"),
        (RuleState("fixture", True, "valid"),),
        104.0 if direction == "Bullish" else 116.0,
        "below" if direction == "Bullish" else "above",
        status=status,
    )


def score(total=80):
    return ConfidenceScore(40, 25, 15, total, ())


def candles(close):
    return pd.DataFrame({"close": [close]}, index=[pd.Timestamp("2026-01-02", tz="UTC")])


def decide(item=None, quality=80, lifecycle="Active", close=110, policy=None):
    return build_decision_state(
        item, score(quality) if item else None, 75, lifecycle, candles(close),
        policy or RiskPolicy(100_000), Friction(),
    )


def test_no_candidate_is_no_trade():
    assert decide().status == "NO TRADE"


def test_forming_candidate_is_watch():
    item = candidate()
    item = replace(
        item, pivots=item.pivots[:4], labels=item.labels[:4], status="Forming",
        active_leg=ActiveLeg(item.pivots[-1].timestamp, 112, "Low", 1, "down"),
        forming_label="4",
    )
    result = decide(item, lifecycle="Forming", close=123.45)
    assert result.status == "WATCH"
    assert result.current_price == 123.45


def test_inactive_and_low_quality_states():
    assert decide(candidate(), lifecycle="Invalidated").status == "NO TRADE"
    assert decide(candidate(), quality=70).status == "WATCH"


def test_low_reward_risk_is_blocked():
    result = decide(candidate(), close=110, policy=RiskPolicy(100_000, minimum_reward_risk=10))
    assert result.status == "BLOCKED"
    assert "Reward-to-risk" in result.reason


def test_approved_bullish_and_bearish_geometry():
    bullish = decide(candidate(), close=110)
    bearish = decide(candidate("Bearish"), close=110)
    assert bullish.status == bearish.status == "TRADE READY"
    assert bullish.stop < bullish.entry_reference < bullish.target_1
    assert bearish.target_2 < bearish.entry_reference < bearish.stop
    assert bullish.entry_reference == 110
    assert bullish.setup_reference == 112


def test_invalid_risk_settings_block_without_crashing():
    result = build_decision_state(
        candidate(), score(), 75, "Active", candles(110), None, None
    )
    assert result.status == "BLOCKED"
    assert result.risk_reason == "Invalid risk settings"


def test_configurable_threshold_controls_risk_evaluation():
    item = candidate()
    strict = build_decision_state(
        item, score(68.6), 75, "Active", candles(110),
        RiskPolicy(100_000), Friction(),
    )
    aggressive = build_decision_state(
        item, score(68.6), 65, "Active", candles(110),
        RiskPolicy(100_000), Friction(),
    )
    forming = replace(
        item, pivots=item.pivots[:4], labels=item.labels[:4], status="Forming",
        active_leg=ActiveLeg(item.pivots[-1].timestamp, 112, "Low", 1, "down"),
        forming_label="4",
    )
    forming_result = build_decision_state(
        forming, score(68.6), 65, "Forming", candles(110),
        RiskPolicy(100_000), Friction(),
    )

    assert strict.status == "WATCH"
    assert strict.required_threshold == 75
    assert strict.quality_gate_result == "Below threshold"
    assert aggressive.status == "TRADE READY"
    assert aggressive.quality_gate_result == "Passed"
    assert forming_result.status == "WATCH"


def test_target_hit_precedes_risk_and_reports_no_trade():
    result = build_decision_state(
        candidate(), score(80), 70, "Target hit", candles(130),
        RiskPolicy(100_000), Friction(),
    )

    assert result.status == "NO TRADE"
    assert result.reason == "Target zone already reached."
    assert result.reward_risk is None


def test_no_valid_directional_target_remains():
    bullish = build_decision_state(
        candidate(), score(80), 70, "Active", candles(130),
        RiskPolicy(100_000), Friction(),
    )
    bearish = build_decision_state(
        candidate("Bearish"), score(80), 70, "Active", candles(90),
        RiskPolicy(100_000), Friction(),
    )

    assert bullish.status == "NO TRADE"
    assert bullish.reason == "No valid upside target remains"
    assert bearish.status == "NO TRADE"
    assert bearish.reason == "No valid downside target remains"
