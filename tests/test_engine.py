from dataclasses import FrozenInstanceError

import pandas as pd
import pytest

from engine import (
    WaveDAG,
    build_candidates,
    evaluate_flat,
    evaluate_impulse,
    evaluate_triangle,
    evaluate_zigzag,
    find_provisional_candidates,
)
from pivots import ActiveLeg, Pivot, PivotState


def pivot(minute, price, kind):
    return Pivot(
        timestamp=pd.Timestamp("2026-01-01T00:00:00Z")
        + pd.Timedelta(minutes=minute),
        price=float(price),
        type=kind,
        atr=1.0,
    )


def bullish_impulse(prices=(100, 110, 104, 120, 112, 125)):
    kinds = ("Low", "High", "Low", "High", "Low", "High")
    return tuple(pivot(index * 5, price, kind) for index, (price, kind) in enumerate(zip(prices, kinds)))


def bearish_impulse(prices=(125, 115, 121, 105, 113, 100)):
    kinds = ("High", "Low", "High", "Low", "High", "Low")
    return tuple(pivot(index * 5, price, kind) for index, (price, kind) in enumerate(zip(prices, kinds)))


def test_dag_has_only_forward_edges_between_adjacent_alternating_pivots():
    pivots = bullish_impulse()

    graph = WaveDAG.from_pivots(pivots)

    assert graph.nodes == pivots
    assert graph.edges == ((0, 1), (1, 2), (2, 3), (3, 4), (4, 5))
    assert all(source < target for source, target in graph.edges)


@pytest.mark.parametrize(
    "prices",
    [
        (100, 110, 99, 120, 112, 125),   # Wave 2 beyond Wave 1 origin
        (100, 110, 104, 112, 111, 125),  # Wave 3 is shortest
        (100, 110, 104, 120, 109, 125),  # Wave 4 enters Wave 1 territory
    ],
)
def test_each_absolute_rule_prunes_bullish_impulse(prices):
    assert evaluate_impulse(bullish_impulse(prices)) is None


@pytest.mark.parametrize(
    "prices",
    [
        (125, 115, 126, 105, 113, 100),
        (125, 115, 121, 113, 112, 100),
        (125, 115, 121, 105, 116, 100),
    ],
)
def test_each_absolute_rule_prunes_bearish_impulse(prices):
    assert evaluate_impulse(bearish_impulse(prices)) is None


def test_valid_bullish_impulse_has_labels_rules_and_exact_floor():
    candidate = evaluate_impulse(bullish_impulse())

    assert candidate is not None
    assert candidate.pattern == "Impulse"
    assert candidate.direction == "Bullish"
    assert candidate.labels == ("Start", "1", "2", "3", "4", "5")
    assert all(state.passed for state in candidate.rule_states)
    assert candidate.invalidation_level == 110.0
    assert candidate.invalidation_side == "below"
    assert candidate.labeled_waves[4][0] == "4"
    assert candidate.status == "Completed"


def test_forming_impulse_keeps_active_wave4_outside_confirmed_pivots():
    confirmed = bullish_impulse()[:4]
    as_of = confirmed[-1].timestamp + pd.Timedelta(minutes=5)
    active = ActiveLeg(as_of, 114.0, "Low", 2.0, "down")

    candidates = find_provisional_candidates(
        PivotState(as_of, confirmed, active),
        include_zigzags=False,
    )

    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.status == "Forming"
    assert candidate.labels == ("Start", "1", "2", "3")
    assert candidate.forming_label == "4"
    assert candidate.active_leg == active
    assert active not in candidate.pivots
    assert candidate.invalidation_level == 110.0
    assert candidate.invalidation_side == "below"
    assert candidate.as_of == as_of


def test_forming_impulse_is_pruned_on_wave1_territory_overlap():
    confirmed = bullish_impulse()[:4]
    as_of = confirmed[-1].timestamp + pd.Timedelta(minutes=5)
    active = ActiveLeg(as_of, 109.0, "Low", 2.0, "down")

    candidates = find_provisional_candidates(
        PivotState(as_of, confirmed, active),
        include_zigzags=False,
    )

    assert candidates == []


def test_confirmed_wave4_creates_entry_ready_impulse():
    confirmed = bullish_impulse()[:5]
    as_of = confirmed[-1].timestamp + pd.Timedelta(minutes=5)

    candidates = find_provisional_candidates(
        PivotState(as_of, confirmed, None),
        include_zigzags=False,
    )

    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.status == "EntryReady"
    assert candidate.labels[-1] == "4"
    assert candidate.active_leg is None
    assert candidate.as_of == as_of


def test_forming_and_entry_ready_zigzag_are_causal():
    start = pivot(0, 120, "High")
    wave_a = pivot(5, 100, "Low")
    forming_as_of = wave_a.timestamp + pd.Timedelta(minutes=5)
    active_b = ActiveLeg(forming_as_of, 110.0, "High", 2.0, "up")

    forming = find_provisional_candidates(
        PivotState(forming_as_of, (start, wave_a), active_b),
        include_impulses=False,
    )
    confirmed_b = pivot(10, 110, "High")
    ready_as_of = confirmed_b.timestamp + pd.Timedelta(minutes=5)
    ready = find_provisional_candidates(
        PivotState(ready_as_of, (start, wave_a, confirmed_b), None),
        include_impulses=False,
    )

    assert len(forming) == 1
    assert forming[0].status == "Forming"
    assert forming[0].forming_label == "B"
    assert forming[0].active_leg == active_b
    assert len(ready) == 1
    assert ready[0].status == "EntryReady"
    assert ready[0].labels == ("Start", "A", "B")


def test_provisional_zigzag_is_pruned_when_wave_b_reaches_origin():
    start = pivot(0, 120, "High")
    wave_a = pivot(5, 100, "Low")
    as_of = wave_a.timestamp + pd.Timedelta(minutes=5)
    active_b = ActiveLeg(as_of, 120.0, "High", 2.0, "up")

    candidates = find_provisional_candidates(
        PivotState(as_of, (start, wave_a), active_b),
        include_impulses=False,
    )

    assert candidates == []


def test_provisional_state_rejects_noncausal_active_leg_timestamp():
    confirmed = bullish_impulse()[:4]
    as_of = confirmed[-1].timestamp + pd.Timedelta(minutes=5)
    stale = ActiveLeg(confirmed[-1].timestamp, 114.0, "Low", 2.0, "down")
    future = ActiveLeg(
        as_of + pd.Timedelta(minutes=5), 114.0, "Low", 2.0, "down"
    )

    with pytest.raises(ValueError, match="must follow"):
        find_provisional_candidates(
            PivotState(as_of, confirmed, stale),
            include_zigzags=False,
        )
    with pytest.raises(ValueError, match="after state as_of"):
        find_provisional_candidates(
            PivotState(as_of, confirmed, future),
            include_zigzags=False,
        )


def test_valid_bearish_impulse_has_exact_ceiling():
    candidate = evaluate_impulse(bearish_impulse())

    assert candidate is not None
    assert candidate.direction == "Bearish"
    assert candidate.invalidation_level == 115.0
    assert candidate.invalidation_side == "above"


def test_wave_3_tied_for_shortest_is_allowed():
    candidate = evaluate_impulse(
        bullish_impulse((100, 110, 104, 114, 112, 122))
    )

    assert candidate is not None


def test_valid_bearish_abc_zigzag_has_origin_invalidation_ceiling():
    path = (
        pivot(0, 120, "High"),
        pivot(5, 100, "Low"),
        pivot(10, 112, "High"),
        pivot(15, 90, "Low"),
    )

    candidate = evaluate_zigzag(path)

    assert candidate is not None
    assert candidate.labels == ("Start", "A", "B", "C")
    assert candidate.direction == "Bearish"
    assert candidate.invalidation_level == 120.0
    assert candidate.invalidation_side == "above"


def test_abc_is_pruned_when_b_passes_a_origin():
    path = (
        pivot(0, 120, "High"),
        pivot(5, 100, "Low"),
        pivot(10, 121, "High"),
        pivot(15, 90, "Low"),
    )

    assert evaluate_zigzag(path) is None


def test_valid_expanded_flat_has_labels_rules_and_b_invalidation():
    path = (
        pivot(0, 120, "High"),
        pivot(5, 100, "Low"),
        pivot(10, 121, "High"),
        pivot(15, 101, "Low"),
    )

    candidate = evaluate_flat(path)

    assert candidate is not None
    assert candidate.pattern == "Flat"
    assert candidate.labels == ("Start", "A", "B", "C")
    assert all(state.passed for state in candidate.rule_states)
    assert candidate.invalidation_level == 121
    assert candidate.invalidation_side == "above"


@pytest.mark.parametrize(
    "prices",
    [
        (120, 100, 117, 101),  # B retraces only 85% of A
        (120, 100, 121, 110),  # C terminates too far from A's extreme
    ],
)
def test_flat_rules_prune_invalid_paths(prices):
    kinds = ("High", "Low", "High", "Low")
    path = tuple(
        pivot(index * 5, price, kind)
        for index, (price, kind) in enumerate(zip(prices, kinds))
    )

    assert evaluate_flat(path) is None


def test_valid_contracting_triangle_has_abcde_labels_and_invalidation():
    path = (
        pivot(0, 120, "High"),
        pivot(5, 100, "Low"),
        pivot(10, 114, "High"),
        pivot(15, 104, "Low"),
        pivot(20, 111, "High"),
        pivot(25, 106, "Low"),
    )

    candidate = evaluate_triangle(path)

    assert candidate is not None
    assert candidate.pattern == "Triangle"
    assert candidate.labels == ("Start", "A", "B", "C", "D", "E")
    assert len(candidate.rule_states) == 4
    assert candidate.invalidation_level == 120
    assert candidate.invalidation_side == "above"


def test_triangle_is_pruned_when_any_leg_stops_contracting():
    path = (
        pivot(0, 120, "High"),
        pivot(5, 100, "Low"),
        pivot(10, 114, "High"),
        pivot(15, 104, "Low"),
        pivot(20, 111, "High"),
        pivot(25, 103, "Low"),  # E=8 exceeds D=7
    )

    assert evaluate_triangle(path) is None


def test_builder_integrates_flat_and_triangle_patterns():
    triangle = [
        pivot(0, 120, "High"),
        pivot(5, 100, "Low"),
        pivot(10, 114, "High"),
        pivot(15, 104, "Low"),
        pivot(20, 111, "High"),
        pivot(25, 106, "Low"),
    ]
    patterns = {candidate.pattern for candidate in build_candidates(triangle)}

    assert "Triangle" in patterns


def test_builder_returns_both_pattern_types_from_graph_paths():
    pivots = list(bullish_impulse())

    candidates = build_candidates(pivots)

    assert any(candidate.pattern == "Impulse" for candidate in candidates)
    assert any(candidate.pattern == "ZigZag" for candidate in candidates)


def test_non_alternating_nodes_break_paths_instead_of_being_skipped():
    pivots = list(bullish_impulse())
    pivots[2] = pivot(10, 104, "High")

    graph = WaveDAG.from_pivots(pivots)

    assert (1, 2) not in graph.edges
    assert graph.candidates() == []


def test_outputs_are_immutable():
    mutable_input = list(bullish_impulse())
    candidate = evaluate_impulse(mutable_input)
    assert candidate is not None
    mutable_input.clear()

    assert len(candidate.pivots) == 6
    with pytest.raises(FrozenInstanceError):
        candidate.invalidation_level = 0  # type: ignore[misc]
    with pytest.raises(TypeError):
        candidate.pivots[0] = pivot(0, 0, "Low")  # type: ignore[index]


def test_rejects_non_chronological_input():
    path = list(bullish_impulse())
    path[1], path[2] = path[2], path[1]

    with pytest.raises(ValueError, match="chronological"):
        WaveDAG.from_pivots(path)
