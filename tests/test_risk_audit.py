from dataclasses import replace

import pandas as pd

from audit import append_signal_event, load_signal_events
from backtester import Friction
from engine import RuleState, WaveCandidate
from pivots import Pivot
from risk import RiskPolicy, size_trade


def candidate(status="Forming"):
    index = pd.date_range("2026-01-01", periods=4, freq="5min", tz="UTC")
    pivots = tuple(
        Pivot(timestamp, price, kind, 1)
        for timestamp, price, kind in zip(
            index,
            (100, 110, 104, 120),
            ("Low", "High", "Low", "High"),
        )
    )
    return WaveCandidate(
        "Impulse",
        "Bullish",
        pivots,
        ("Start", "1", "2", "3"),
        (RuleState("fixture", True, "valid"),),
        110,
        "below",
        status=status,
    )


def test_position_size_obeys_lot_risk_and_reward_gates():
    policy = RiskPolicy(100_000, lot_size=25, minimum_reward_risk=1.5)

    plan = size_trade(
        direction="Bullish",
        entry=100,
        stop=95,
        target=112,
        policy=policy,
        friction=Friction(spread=0.2, slippage=0.1, commission=0.05),
    )

    assert plan.approved
    assert plan.units % 25 == 0
    assert plan.total_risk <= 1_000
    assert plan.reward_risk >= 1.5


def test_position_size_rejects_daily_loss_and_bad_geometry():
    policy = RiskPolicy(100_000)

    daily = size_trade(
        direction="Bullish",
        entry=100,
        stop=95,
        target=115,
        policy=policy,
        realized_daily_loss=3_000,
    )
    geometry = size_trade(
        direction="Bullish",
        entry=100,
        stop=105,
        target=115,
        policy=policy,
    )

    assert not daily.approved and "daily loss" in daily.reason
    assert not geometry.approved and "geometry" in geometry.reason


def test_signal_audit_is_append_only_and_transition_validated(tmp_path):
    database = tmp_path / "signals.sqlite"
    forming = candidate()
    ready = replace(forming, status="EntryReady")

    first = append_signal_event(
        database, forming, 65, observed_at=pd.Timestamp("2026-01-01T01:00Z")
    )
    second = append_signal_event(
        database, ready, 80, observed_at=pd.Timestamp("2026-01-01T01:05Z")
    )

    events = load_signal_events(database)
    assert events == (first, second)
    try:
        append_signal_event(
            database,
            forming,
            70,
            observed_at=pd.Timestamp("2026-01-01T01:10Z"),
        )
    except ValueError as error:
        assert "invalid signal transition" in str(error)
    else:
        raise AssertionError("backward signal transition was accepted")
