"""Independent golden and adversarial checks for the normative model."""

import pandas as pd

from engine import evaluate_flat, evaluate_impulse, evaluate_triangle, evaluate_zigzag
from pivots import Pivot


def path(prices, first_type="Low"):
    types = ("Low", "High") if first_type == "Low" else ("High", "Low")
    index = pd.date_range("2026-01-01", periods=len(prices), freq="5min", tz="UTC")
    return tuple(
        Pivot(timestamp, float(price), types[i % 2], 1.0)
        for i, (timestamp, price) in enumerate(zip(index, prices))
    )


def mirror(prices, center=200):
    return tuple(center - price for price in prices)


def test_impulse_rules_are_directionally_symmetric():
    bullish = (100, 110, 104, 120, 112, 125)
    bearish = mirror(bullish)

    bull = evaluate_impulse(path(bullish, "Low"))
    bear = evaluate_impulse(path(bearish, "High"))

    assert bull is not None and bear is not None
    assert bull.direction == "Bullish"
    assert bear.direction == "Bearish"
    assert [state.name for state in bull.rule_states] == [
        state.name for state in bear.rule_states
    ]


def test_single_point_mutations_cannot_cross_hard_impulse_boundaries():
    invalid_mutations = (
        (100, 110, 100, 120, 112, 125),
        (100, 110, 104, 110, 109, 125),
        (100, 110, 104, 120, 109, 125),
        (100, 110, 104, 120, 112, 119),
    )

    assert all(
        evaluate_impulse(path(prices, "Low")) is None
        for prices in invalid_mutations
    )


def test_correction_classifiers_are_mutually_separated_by_b_ratio():
    zigzag = evaluate_zigzag(path((120, 100, 112, 90), "High"))
    flat = evaluate_flat(path((120, 100, 124, 95), "High"))

    assert zigzag is not None
    assert evaluate_flat(path((120, 100, 112, 90), "High")) is None
    assert flat is not None and flat.variant == "Expanded"
    assert evaluate_zigzag(path((120, 100, 124, 95), "High")) is None


def test_triangle_near_miss_with_expanding_boundary_is_rejected():
    valid = evaluate_triangle(path((120, 100, 114, 104, 111, 106), "High"))
    expanding = evaluate_triangle(path((120, 100, 114, 98, 116, 96), "High"))

    assert valid is not None and valid.variant == "Contracting"
    assert expanding is None
