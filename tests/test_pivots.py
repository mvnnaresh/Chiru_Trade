import pandas as pd
import pandas.testing as pdt
import pytest

from pivots import ActiveLeg, Pivot, calculate_atr, extract_pivot_state, extract_pivots


def make_ohlc(highs, lows, closes=None):
    if closes is None:
        closes = [(high + low) / 2 for high, low in zip(highs, lows)]
    index = pd.date_range("2026-01-01", periods=len(highs), freq="5min", tz="UTC")
    return pd.DataFrame(
        {"open": closes, "high": highs, "low": lows, "close": closes, "volume": 1},
        index=index,
    )


def test_atr_uses_true_range_and_rolling_simple_average():
    frame = make_ohlc(
        highs=[10, 13, 12, 15],
        lows=[8, 11, 9, 13],
        closes=[9, 12, 10, 14],
    )

    result = calculate_atr(frame, period=3)

    expected = pd.Series(
        [float("nan"), float("nan"), 3.0, 4.0],
        index=frame.index,
        name="atr",
    )
    pdt.assert_series_equal(result, expected)


def test_default_atr_requires_fourteen_observations():
    frame = make_ohlc(list(range(20, 34)), list(range(18, 32)))

    result = calculate_atr(frame)

    assert result.iloc[:13].isna().all()
    assert result.iloc[13] == pytest.approx(2.0)


def test_zigzag_emits_alternating_confirmed_pivots_with_atr_at_extreme():
    frame = make_ohlc(
        highs=[100, 101, 102, 104, 105, 103, 101, 104],
        lows=[98, 99, 100, 102, 103, 100, 98, 101],
    )

    pivots = extract_pivots(frame, multiplier=1.5, atr_period=3)

    assert [(p.timestamp, p.price, p.type) for p in pivots] == [
        (frame.index[2], 100.0, "Low"),
        (frame.index[4], 105.0, "High"),
        (frame.index[6], 98.0, "Low"),
    ]
    assert [p.atr for p in pivots] == pytest.approx([2.0, 7 / 3, 19 / 6])


def test_threshold_scales_with_atr_instead_of_price_percentage():
    # Both regimes move by four price units. The low-ATR regime confirms a
    # reversal; the later high-ATR regime does not under the same multiplier.
    low_volatility = make_ohlc(
        highs=[101, 101, 101, 103, 105, 103],
        lows=[99, 99, 99, 101, 103, 100],
    )
    high_volatility = make_ohlc(
        highs=[105, 105, 105, 107, 109, 107],
        lows=[95, 95, 95, 97, 99, 97],
    )

    low_pivots = extract_pivots(low_volatility, multiplier=1.5, atr_period=3)
    high_pivots = extract_pivots(high_volatility, multiplier=1.5, atr_period=3)

    assert [pivot.type for pivot in low_pivots] == ["Low", "High"]
    assert high_pivots == []


def test_unconfirmed_last_extreme_is_not_returned():
    frame = make_ohlc(
        highs=[100, 101, 102, 104, 106],
        lows=[98, 99, 100, 102, 104],
    )

    pivots = extract_pivots(frame, multiplier=1.5, atr_period=3)

    assert pivots == [Pivot(frame.index[2], 100.0, "Low", 2.0)]
    assert all(pivot.price != 106 for pivot in pivots)


def test_active_leg_is_separate_from_confirmed_pivots():
    frame = make_ohlc(
        highs=[100, 101, 102, 104, 106],
        lows=[98, 99, 100, 102, 104],
    )

    state = extract_pivot_state(frame, multiplier=1.5, atr_period=3)

    assert state.confirmed == (Pivot(frame.index[2], 100.0, "Low", 2.0),)
    assert state.active_leg == ActiveLeg(
        frame.index[4], 106.0, "High", 8 / 3, "up"
    )
    assert state.as_of == frame.index[-1]
    assert state.active_leg.timestamp > state.confirmed[-1].timestamp


def test_active_leg_revision_is_causal_and_does_not_rewrite_confirmed_history():
    frame = make_ohlc(
        highs=[100, 101, 102, 104, 106, 108],
        lows=[98, 99, 100, 102, 104, 106],
    )

    earlier = extract_pivot_state(frame.iloc[:-1], multiplier=1.5, atr_period=3)
    later = extract_pivot_state(frame, multiplier=1.5, atr_period=3)

    assert earlier.confirmed == later.confirmed
    assert earlier.active_leg is not None and later.active_leg is not None
    assert earlier.active_leg.price == 106.0
    assert later.active_leg.price == 108.0
    assert earlier.as_of < later.as_of


def test_active_leg_is_absent_until_direction_is_established():
    frame = make_ohlc(highs=[101, 101, 101], lows=[99, 99, 99])

    state = extract_pivot_state(frame, multiplier=2.0, atr_period=3)

    assert state.confirmed == ()
    assert state.active_leg is None


@pytest.mark.parametrize(
    ("opening", "closing", "expected_type"),
    [
        (100, 110, "Low"),
        (110, 100, "High"),
    ],
)
def test_same_candle_extremes_use_body_direction_without_duplicate_timestamps(
    opening, closing, expected_type
):
    index = pd.date_range("2026-01-01", periods=4, freq="5min", tz="UTC")
    frame = pd.DataFrame(
        {
            "open": [opening, 105, 104, 106],
            "high": [115, 108, 107, 109],
            "low": [95, 102, 101, 103],
            "close": [closing, 104, 106, 105],
        },
        index=index,
    )

    pivots = extract_pivots(frame, multiplier=1.0, atr_period=1)

    assert pivots[0].timestamp == index[0]
    assert pivots[0].type == expected_type
    assert len({pivot.timestamp for pivot in pivots}) == len(pivots)
    assert all(
        earlier.timestamp < later.timestamp
        for earlier, later in zip(pivots, pivots[1:])
    )


def test_later_reversal_candle_cannot_emit_both_extremes_at_one_timestamp():
    index = pd.date_range("2026-01-01", periods=5, freq="5min", tz="UTC")
    frame = pd.DataFrame(
        {
            "open": [100, 102, 103, 99, 101],
            "high": [105, 110, 108, 106, 109],
            "low": [95, 100, 98, 96, 99],
            "close": [104, 101, 99, 104, 100],
        },
        index=index,
    )

    pivots = extract_pivots(frame, multiplier=1.0, atr_period=1)

    assert len({pivot.timestamp for pivot in pivots}) == len(pivots)
    assert all(
        earlier.timestamp < later.timestamp
        for earlier, later in zip(pivots, pivots[1:])
    )


def test_prefix_results_never_change_when_future_candles_are_appended():
    frame = make_ohlc(
        highs=[100, 101, 102, 104, 105, 103, 101, 104, 106],
        lows=[98, 99, 100, 102, 103, 100, 98, 101, 104],
    )
    prefix = frame.iloc[:6]

    prefix_pivots = extract_pivots(prefix, multiplier=1.5, atr_period=3)
    full_pivots = extract_pivots(frame, multiplier=1.5, atr_period=3)

    assert full_pivots[: len(prefix_pivots)] == prefix_pivots


@pytest.mark.parametrize("multiplier", [0, -1, True, "2"])
def test_zigzag_rejects_invalid_multiplier(multiplier):
    with pytest.raises(ValueError, match="multiplier"):
        extract_pivots(
            make_ohlc([2, 3, 4], [1, 2, 3]),
            multiplier=multiplier,
            atr_period=2,
        )


def test_validation_rejects_non_chronological_or_malformed_data():
    frame = make_ohlc([2, 3, 4], [1, 2, 3])

    with pytest.raises(ValueError, match="chronological"):
        calculate_atr(frame.iloc[::-1])
    with pytest.raises(ValueError, match="missing OHLC"):
        calculate_atr(frame.drop(columns="close"))
    with pytest.raises(ValueError, match="high"):
        calculate_atr(frame.assign(high=0))
