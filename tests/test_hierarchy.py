from dataclasses import replace

import pandas as pd

from engine import evaluate_impulse
from hierarchy import build_wave_degree_hierarchy
from pivots import Pivot


def bullish_impulse():
    prices = (100, 110, 104, 120, 112, 125)
    kinds = ("Low", "High", "Low", "High", "Low", "High")
    index = pd.date_range("2026-01-01", periods=6, freq="5min", tz="UTC")
    return tuple(
        Pivot(timestamp, float(price), kind, 1.0)
        for timestamp, price, kind in zip(index, prices, kinds)
    )


def test_higher_timeframe_candidate_parents_contained_lower_structure():
    child = evaluate_impulse(bullish_impulse())
    assert child is not None
    parent = replace(child, variant="parent")

    nodes = build_wave_degree_hierarchy({"1H": [child], "4H": [parent]})

    child_index = next(i for i, node in enumerate(nodes) if node.timeframe == "1H")
    parent_index = next(i for i, node in enumerate(nodes) if node.timeframe == "4H")
    assert nodes[child_index].parent_index == parent_index
    assert nodes[parent_index].child_indices == (child_index,)
    assert nodes[parent_index].degree_rank > nodes[child_index].degree_rank


def test_hierarchy_rejects_unknown_timeframe():
    candidate = evaluate_impulse(bullish_impulse())
    assert candidate is not None

    try:
        build_wave_degree_hierarchy({"3H": [candidate]})
    except ValueError as error:
        assert "unknown timeframes" in str(error)
    else:
        raise AssertionError("unknown timeframe was accepted")
