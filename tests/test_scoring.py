from dataclasses import FrozenInstanceError

import pandas as pd
import pandas.testing as pdt
import pytest

import scoring
from engine import (
    evaluate_flat,
    evaluate_impulse,
    evaluate_triangle,
    evaluate_zigzag,
)
from pivots import Pivot
from scoring import calculate_rsi, score_candidates


def pivot(minute, price, kind):
    return Pivot(
        pd.Timestamp("2026-01-01T00:00:00Z") + pd.Timedelta(minutes=minute),
        float(price),
        kind,
        1.0,
    )


def ideal_impulse():
    points = (
        pivot(0, 100, "Low"),
        pivot(5, 110, "High"),
        pivot(10, 104, "Low"),
        pivot(15, 120.18, "High"),
        pivot(20, 114, "Low"),
        pivot(25, 122, "High"),
    )
    candidate = evaluate_impulse(points)
    assert candidate is not None
    return candidate


def market_frame(periods=30):
    index = pd.date_range("2026-01-01", periods=periods, freq="1min", tz="UTC")
    return pd.DataFrame(
        {
            "open": range(periods),
            "high": range(1, periods + 1),
            "low": range(periods),
            "close": range(periods),
            "volume": 1,
        },
        index=index,
    )


def test_wilder_rsi_is_causal_and_bounded():
    frame = market_frame(20)

    result = calculate_rsi(frame, period=3)

    assert result.iloc[:3].isna().all()
    assert (result.dropna() == 100.0).all()
    changed_future = frame.copy()
    changed_future.iloc[-1, changed_future.columns.get_loc("close")] = -100
    pdt.assert_series_equal(
        result.iloc[:-1], calculate_rsi(changed_future, period=3).iloc[:-1]
    )


def test_ideal_fibonacci_and_momentum_receive_full_80_points(monkeypatch):
    candidate = ideal_impulse()
    frame = market_frame()
    rsi = pd.Series(50.0, index=frame.index, name="rsi")
    rsi.loc[candidate.pivots[0].timestamp : candidate.pivots[1].timestamp] = 60
    rsi.loc[candidate.pivots[2].timestamp : candidate.pivots[3].timestamp] = 80
    rsi.loc[candidate.pivots[4].timestamp : candidate.pivots[5].timestamp] = 70
    monkeypatch.setattr(scoring, "calculate_rsi", lambda *_args, **_kwargs: rsi)

    score = score_candidates([candidate], frame)[candidate]

    assert score.fibonacci == 50.0
    assert score.momentum == 30.0
    assert score.total <= 100.0
    assert sum(item.maximum for item in score.items) == 100


def test_fibonacci_points_decay_deterministically_away_from_targets(monkeypatch):
    candidate = ideal_impulse()
    frame = market_frame()
    monkeypatch.setattr(
        scoring,
        "calculate_rsi",
        lambda *_args, **_kwargs: pd.Series(float("nan"), index=frame.index),
    )
    score = score_candidates((candidate,), frame)[candidate]

    fib_items = [item for item in score.items if item.category == "Fibonacci Alignment"]
    assert fib_items[0].points == 25.0
    assert fib_items[1].points == pytest.approx(25.0)
    assert score.momentum == 0.0


def test_missing_rsi_history_awards_no_momentum_points():
    candidate = ideal_impulse()
    frame = market_frame()

    score = score_candidates([candidate], frame, rsi_period=100)[candidate]

    assert score.momentum == 0.0


def test_score_mapping_and_score_objects_are_immutable():
    candidate = ideal_impulse()
    scores = score_candidates([candidate], market_frame())
    score = scores[candidate]

    with pytest.raises(TypeError):
        scores[candidate] = score  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        score.total = 0  # type: ignore[misc]


def test_bearish_zigzag_receives_bounded_auditable_score():
    path = (
        pivot(0, 120, "High"),
        pivot(5, 100, "Low"),
        pivot(10, 112, "High"),
        pivot(15, 80, "Low"),
    )
    candidate = evaluate_zigzag(path)
    assert candidate is not None

    score = score_candidates([candidate], market_frame())[candidate]

    assert 0 <= score.total <= 100
    assert score.fibonacci <= 50
    assert score.momentum <= 30
    assert score.channeling_alternation <= 20
    assert all(item.reason for item in score.items)


def test_flat_scores_maximum_fibonacci_at_expanded_b_and_regular_c(monkeypatch):
    path = (
        pivot(0, 120, "High"),
        pivot(5, 100, "Low"),
        pivot(10, 120, "High"),
        pivot(15, 100, "Low"),
    )
    candidate = evaluate_flat(path)
    assert candidate is not None
    frame = market_frame()
    rsi = pd.Series(float("nan"), index=frame.index, name="rsi")
    rsi.loc[frame.index[1:5]] = 30
    rsi.loc[frame.index[11:15]] = 20
    monkeypatch.setattr(scoring, "calculate_rsi", lambda *_args, **_kwargs: rsi)

    result = score_candidates([candidate], frame)[candidate]

    assert result.fibonacci == 50
    assert result.momentum == 20
    assert result.channeling_alternation == 20
    assert result.total == 90
    assert any("expanded range" in item.reason for item in result.items)


def test_triangle_scores_clean_contraction_wedge_and_declining_rsi(monkeypatch):
    path = (
        pivot(0, 120, "High"),
        pivot(10, 100, "Low"),
        pivot(20, 114, "High"),
        pivot(30, 104.2, "Low"),
        pivot(40, 111.06, "High"),
        pivot(50, 106.258, "Low"),
    )
    candidate = evaluate_triangle(path)
    assert candidate is not None
    frame = market_frame(periods=60)
    rsi = pd.Series(float("nan"), index=frame.index, name="rsi")
    for start, end, value in (
        (1, 9, 20),
        (11, 19, 70),
        (21, 29, 40),
        (31, 39, 50),
        (41, 49, 60),
    ):
        rsi.loc[frame.index[start:end]] = value
    monkeypatch.setattr(scoring, "calculate_rsi", lambda *_args, **_kwargs: rsi)

    result = score_candidates([candidate], frame)[candidate]

    assert result.fibonacci == 50
    assert result.momentum == 30
    assert result.channeling_alternation == 20
    assert result.total == 100
    assert any("contracting wedge" in item.reason for item in result.items)


def test_rejects_invalid_market_data_or_candidate_types():
    frame = market_frame()

    with pytest.raises(ValueError, match="close"):
        score_candidates([], frame.drop(columns="close"))
    with pytest.raises(ValueError, match="chronological"):
        score_candidates([], frame.iloc[::-1])
    with pytest.raises(TypeError, match="WaveCandidate"):
        score_candidates([object()], frame)  # type: ignore[list-item]
