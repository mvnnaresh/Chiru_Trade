from dataclasses import FrozenInstanceError

import pandas as pd
import pytest

from backtester import (
    CandidateRanking,
    Friction,
    build_causal_rankings,
    run_backtest,
)
from engine import RuleState, WaveCandidate
from pivots import ActiveLeg, Pivot


def market(highs, lows, closes):
    index = pd.date_range("2026-01-01", periods=len(highs), freq="5min", tz="UTC")
    return pd.DataFrame(
        {
            "open": closes,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": 1,
        },
        index=index,
    )


def wave4_candidate(index):
    prices = (100, 110, 104, 120, 112)
    kinds = ("Low", "High", "Low", "High", "Low")
    pivots = tuple(
        Pivot(index[i], float(price), kind, 1.0)
        for i, (price, kind) in enumerate(zip(prices, kinds))
    )
    return WaveCandidate(
        pattern="Impulse",
        direction="Bullish",
        pivots=pivots,
        labels=("Start", "1", "2", "3", "4"),
        rule_states=(RuleState("partial_structure", True, "valid through Wave 4"),),
        invalidation_level=104.0,
        invalidation_side="below",
        status="EntryReady",
        as_of=index[4],
    )


def test_causal_ranking_builder_passes_only_each_prefix_to_ranker():
    frame = market([10] * 5, [9] * 5, [9.5] * 5)
    observed_lengths = []
    observed_last_times = []

    def spy_ranker(prefix):
        observed_lengths.append(len(prefix))
        observed_last_times.append(prefix.index[-1])
        return []

    result = build_causal_rankings(frame, multiplier=2, ranker=spy_ranker)

    assert result == ()
    assert observed_lengths == [1, 2, 3, 4, 5]
    assert observed_last_times == list(frame.index)


def test_causal_ranking_builder_rejects_future_pivot():
    frame = market([10] * 5, [9] * 5, [9.5] * 5)
    candidate = wave4_candidate(frame.index)

    def leaking_ranker(_prefix):
        return [(candidate, 80)]

    with pytest.raises(ValueError, match="after detected_at"):
        build_causal_rankings(frame, multiplier=2, ranker=leaking_ranker)


def test_take_profit_trade_applies_all_friction():
    frame = market(
        [101, 111, 105, 121, 113, 130],
        [99, 103, 103, 111, 111, 111],
        [100, 110, 104, 120, 112, 129],
    )
    candidate = wave4_candidate(frame.index)
    ranking = CandidateRanking(frame.index[4], candidate, 80)

    summary = run_backtest(
        frame,
        [ranking],
        minimum_confidence=70,
        friction=Friction(spread=2, slippage=0.5, commission=1),
    )

    assert summary.total_trades == 1
    assert summary.win_rate == 100.0
    trade = summary.ledger[0]
    assert trade.entry_price == 113.5
    assert trade.target_price == pytest.approx(128.18)
    assert trade.exit_price == pytest.approx(126.68)
    assert trade.friction_cost == 5.0
    assert trade.net_pnl == pytest.approx(11.18)
    assert trade.exit_reason == "take_profit"


def test_stop_has_priority_if_stop_and_target_touch_same_bar():
    frame = market(
        [101, 111, 105, 121, 113, 130],
        [99, 103, 103, 111, 111, 103],
        [100, 110, 104, 120, 112, 110],
    )
    candidate = wave4_candidate(frame.index)
    ranking = CandidateRanking(frame.index[4], candidate, 80)

    summary = run_backtest(frame, [ranking], minimum_confidence=70)

    assert summary.ledger[0].exit_reason == "stop_loss"
    assert summary.ledger[0].exit_price == 104.0
    assert summary.net_profit_loss == -8.0
    assert summary.maximum_drawdown == 8.0


def test_threshold_filters_setup_and_returns_empty_immutable_summary():
    frame = market([101] * 5, [99] * 5, [100] * 5)
    candidate = wave4_candidate(frame.index)
    ranking = CandidateRanking(frame.index[4], candidate, 69.99)

    summary = run_backtest(frame, [ranking], minimum_confidence=70)

    assert summary.total_trades == 0
    assert summary.win_rate == 0
    assert summary.ledger == ()
    with pytest.raises(FrozenInstanceError):
        summary.total_trades = 1  # type: ignore[misc]


def test_completed_wave_5_path_is_not_treated_as_fresh_wave_4_setup():
    frame = market(
        [101, 111, 105, 121, 113, 130],
        [99, 103, 103, 111, 111, 120],
        [100, 110, 104, 120, 112, 128],
    )
    partial = wave4_candidate(frame.index)
    wave5 = Pivot(frame.index[5], 128.0, "High", 1.0)
    completed = WaveCandidate(
        partial.pattern,
        partial.direction,
        partial.pivots + (wave5,),
        partial.labels + ("5",),
        partial.rule_states,
        partial.invalidation_level,
        partial.invalidation_side,
    )

    summary = run_backtest(
        frame,
        [CandidateRanking(frame.index[5], completed, 90)],
        minimum_confidence=70,
    )

    assert summary.total_trades == 0


def test_forming_candidate_is_observed_but_never_traded():
    frame = market(
        [101, 111, 105, 121, 115, 116],
        [99, 103, 103, 111, 112, 113],
        [100, 110, 104, 120, 114, 115],
    )
    ready = wave4_candidate(frame.index)
    forming = WaveCandidate(
        pattern=ready.pattern,
        direction=ready.direction,
        pivots=ready.pivots[:4],
        labels=("Start", "1", "2", "3"),
        rule_states=ready.rule_states,
        invalidation_level=ready.invalidation_level,
        invalidation_side=ready.invalidation_side,
        status="Forming",
        active_leg=ActiveLeg(
            frame.index[4], 112.0, "Low", 1.0, "down"
        ),
        forming_label="4",
        as_of=frame.index[4],
    )

    summary = run_backtest(
        frame,
        [CandidateRanking(frame.index[4], forming, 95)],
        minimum_confidence=70,
    )

    assert summary.total_trades == 0


def test_causal_rankings_record_forming_revisions_once_and_entry_transition_once():
    frame = market([121] * 7, [99] * 7, [110] * 7)
    ready = wave4_candidate(frame.index)

    def staged_ranker(prefix):
        position = len(prefix) - 1
        if position < 4:
            return []
        if position in {4, 5}:
            active = ActiveLeg(
                prefix.index[-1],
                114.0 - position,
                "Low",
                1.0,
                "down",
            )
            forming = WaveCandidate(
                ready.pattern,
                ready.direction,
                ready.pivots[:4],
                ("Start", "1", "2", "3"),
                ready.rule_states,
                ready.invalidation_level,
                ready.invalidation_side,
                status="Forming",
                active_leg=active,
                forming_label="4",
                as_of=prefix.index[-1],
            )
            return [(forming, 75 + position)]
        repeated_ready = WaveCandidate(
            ready.pattern,
            ready.direction,
            ready.pivots,
            ready.labels,
            ready.rule_states,
            ready.invalidation_level,
            ready.invalidation_side,
            status="EntryReady",
            as_of=prefix.index[-1],
        )
        return [(repeated_ready, 90)]

    rankings = build_causal_rankings(
        frame, multiplier=2, ranker=staged_ranker
    )

    assert [item.candidate.status for item in rankings] == [
        "Forming",
        "Forming",
        "EntryReady",
    ]
    assert rankings[-1].detected_at == frame.index[6]


def test_open_trade_is_closed_at_end_of_data():
    frame = market(
        [101, 111, 105, 121, 113],
        [99, 103, 103, 111, 111],
        [100, 110, 104, 120, 112],
    )
    candidate = wave4_candidate(frame.index)
    ranking = CandidateRanking(frame.index[4], candidate, 80)

    summary = run_backtest(frame, [ranking], minimum_confidence=70)

    assert summary.ledger[0].exit_reason == "end_of_data"
    assert summary.ledger[0].entry_time == summary.ledger[0].exit_time


def test_future_rankings_and_non_chronological_rankings_are_rejected():
    frame = market([101] * 6, [99] * 6, [100] * 6)
    candidate = wave4_candidate(frame.index)
    future_pivots = list(candidate.pivots)
    future_pivots[-1] = Pivot(
        frame.index[5], future_pivots[-1].price, future_pivots[-1].type, 1
    )
    future_candidate = WaveCandidate(
        candidate.pattern,
        candidate.direction,
        tuple(future_pivots),
        candidate.labels,
        candidate.rule_states,
        candidate.invalidation_level,
        candidate.invalidation_side,
    )
    bad = CandidateRanking(frame.index[4], future_candidate, 80)

    with pytest.raises(ValueError, match="after detected_at"):
        run_backtest(frame, [bad], minimum_confidence=70)

    first = CandidateRanking(frame.index[4], candidate, 80)
    second = CandidateRanking(frame.index[5], candidate, 81)
    with pytest.raises(ValueError, match="chronological"):
        run_backtest(frame, [second, first], minimum_confidence=70)


def test_friction_values_and_score_bounds_are_validated():
    with pytest.raises(ValueError, match="non-negative"):
        Friction(spread=-1)
    frame = market([101] * 5, [99] * 5, [100] * 5)
    with pytest.raises(ValueError, match="between"):
        CandidateRanking(frame.index[4], wave4_candidate(frame.index), 101)
