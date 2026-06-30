from dataclasses import FrozenInstanceError

import pandas as pd
import pytest

from engine import WaveDAG, build_candidates, evaluate_impulse, evaluate_zigzag
from pivots import Pivot


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
