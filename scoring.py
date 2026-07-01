"""Deterministic guideline scoring for structurally valid wave candidates."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import pandas as pd

from engine import WaveCandidate


@dataclass(frozen=True, slots=True)
class ScoreItem:
    category: str
    points: float
    maximum: float
    reason: str


@dataclass(frozen=True, slots=True)
class ConfidenceScore:
    """Immutable point score and its auditable category breakdown."""

    fibonacci: float
    momentum: float
    channeling_alternation: float
    total: float
    items: tuple[ScoreItem, ...]


def calculate_rsi(ohlcv: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate causal Wilder RSI from close prices."""
    _validate_market_data(ohlcv)
    if not isinstance(period, int) or isinstance(period, bool) or period <= 0:
        raise ValueError("period must be a positive integer")

    change = ohlcv["close"].astype(float).diff()
    gains = change.clip(lower=0)
    losses = -change.clip(upper=0)
    average_gain = gains.ewm(
        alpha=1 / period, adjust=False, min_periods=period
    ).mean()
    average_loss = losses.ewm(
        alpha=1 / period, adjust=False, min_periods=period
    ).mean()
    relative_strength = average_gain / average_loss
    rsi = 100 - (100 / (1 + relative_strength))
    rsi = rsi.mask((average_loss == 0) & (average_gain > 0), 100.0)
    rsi = rsi.mask((average_loss == 0) & (average_gain == 0), 50.0)
    rsi.name = "rsi"
    return rsi


def score_candidates(
    candidates: list[WaveCandidate] | tuple[WaveCandidate, ...],
    ohlcv: pd.DataFrame,
    *,
    rsi_period: int = 14,
) -> Mapping[WaveCandidate, ConfidenceScore]:
    """Return an immutable candidate-to-Confidence-Score mapping."""
    _validate_market_data(ohlcv)
    if any(not isinstance(candidate, WaveCandidate) for candidate in candidates):
        raise TypeError("all candidates must be WaveCandidate objects")

    rsi = calculate_rsi(ohlcv, period=rsi_period)
    scorers = {
        "Impulse": _score_impulse,
        "ZigZag": _score_zigzag,
        "Flat": _score_flat,
        "Triangle": _score_triangle,
    }
    scores = {
        candidate: scorers[candidate.pattern](candidate, rsi)
        for candidate in candidates
    }
    return MappingProxyType(scores)


def _score_impulse(candidate: WaveCandidate, rsi: pd.Series) -> ConfidenceScore:
    pivots = candidate.pivots
    sign = 1.0 if candidate.direction == "Bullish" else -1.0
    wave_1 = sign * (pivots[1].price - pivots[0].price)
    wave_2 = sign * (pivots[1].price - pivots[2].price)
    wave_3 = sign * (pivots[3].price - pivots[2].price)
    wave_4 = sign * (pivots[3].price - pivots[4].price)

    retracement = wave_2 / wave_1
    extension = wave_3 / wave_1
    wave_2_points = _range_alignment(retracement, 0.5, 0.618, 0.25, 25)
    wave_3_points = _target_alignment(extension, (1.618, 2.618), 0.382, 25)

    strengths = tuple(
        _wave_momentum(rsi, pivots[start].timestamp, pivots[start + 1].timestamp, sign)
        for start in (0, 2, 4)
    )
    wave_3_strongest = (
        all(value is not None for value in strengths)
        and strengths[1] >= strengths[0]
        and strengths[1] >= strengths[2]
    )
    momentum_peak_points = 20.0 if wave_3_strongest else 0.0
    new_wave_5_extreme = sign * (pivots[5].price - pivots[3].price) > 0
    divergence = (
        strengths[1] is not None
        and strengths[2] is not None
        and new_wave_5_extreme
        and strengths[2] < strengths[1]
    )
    divergence_points = 10.0 if divergence else 0.0

    wave_2_depth = wave_2 / wave_1
    wave_4_depth = wave_4 / wave_3
    depth_alternates = (
        (wave_2_depth >= 0.5 and wave_4_depth <= 0.382)
        or (wave_4_depth >= 0.5 and wave_2_depth <= 0.382)
    )
    alternation_points = 12.0 if depth_alternates else 0.0
    channel_error = _channel_error(candidate)
    channel_points = max(0.0, 8.0 * (1.0 - channel_error / 0.25))

    items = (
        ScoreItem(
            "Fibonacci Alignment",
            wave_2_points,
            25,
            f"Wave 2 retracement={retracement:.4f}; ideal range=0.500-0.618",
        ),
        ScoreItem(
            "Fibonacci Alignment",
            wave_3_points,
            25,
            f"Wave 3 extension={extension:.4f}; targets=1.618 or 2.618",
        ),
        ScoreItem(
            "Momentum Verification",
            momentum_peak_points,
            20,
            "Wave 3 has the strongest motive-wave RSI"
            if wave_3_strongest
            else "Wave 3 does not have the strongest available motive-wave RSI",
        ),
        ScoreItem(
            "Momentum Verification",
            divergence_points,
            10,
            "Wave 5 makes a price extreme with weaker RSI than Wave 3"
            if divergence
            else "No qualifying Wave 5 RSI divergence",
        ),
        ScoreItem(
            "Channeling & Alternation",
            alternation_points,
            12,
            f"corrective depths: Wave 2={wave_2_depth:.4f}, Wave 4={wave_4_depth:.4f}",
        ),
        ScoreItem(
            "Channeling & Alternation",
            channel_points,
            8,
            f"Wave 4 normalized distance from projected 1-3 channel={channel_error:.4f}",
        ),
    )
    return _assemble_score(items)


def _score_zigzag(candidate: WaveCandidate, rsi: pd.Series) -> ConfidenceScore:
    """Score ABC paths with the same capped matrix using applicable guidelines."""
    pivots = candidate.pivots
    sign = 1.0 if candidate.direction == "Bullish" else -1.0
    wave_a = sign * (pivots[1].price - pivots[0].price)
    wave_b = sign * (pivots[1].price - pivots[2].price)
    wave_c = sign * (pivots[3].price - pivots[2].price)
    b_ratio = wave_b / wave_a
    c_ratio = wave_c / wave_a
    b_points = _range_alignment(b_ratio, 0.382, 0.786, 0.236, 25)
    c_points = _target_alignment(c_ratio, (1.0, 1.618), 0.382, 25)

    a_strength = _wave_momentum(
        rsi, pivots[0].timestamp, pivots[1].timestamp, sign
    )
    c_strength = _wave_momentum(
        rsi, pivots[2].timestamp, pivots[3].timestamp, sign
    )
    c_momentum_points = (
        30.0
        if a_strength is not None and c_strength is not None and c_strength >= a_strength
        else 0.0
    )
    duration_a = (pivots[1].timestamp - pivots[0].timestamp).total_seconds()
    duration_c = (pivots[3].timestamp - pivots[2].timestamp).total_seconds()
    time_ratio = min(duration_a, duration_c) / max(duration_a, duration_c)
    proportion_points = 20.0 * time_ratio

    items = (
        ScoreItem("Fibonacci Alignment", b_points, 25, f"Wave B/A={b_ratio:.4f}"),
        ScoreItem("Fibonacci Alignment", c_points, 25, f"Wave C/A={c_ratio:.4f}"),
        ScoreItem(
            "Momentum Verification",
            c_momentum_points,
            30,
            "Wave C RSI strength equals or exceeds Wave A"
            if c_momentum_points
            else "Wave C lacks confirming RSI strength",
        ),
        ScoreItem(
            "Channeling & Alternation",
            proportion_points,
            20,
            f"Wave A/C duration symmetry={time_ratio:.4f}",
        ),
    )
    return _assemble_score(items)


def _score_flat(candidate: WaveCandidate, rsi: pd.Series) -> ConfidenceScore:
    pivots = candidate.pivots
    sign = 1.0 if candidate.direction == "Bullish" else -1.0
    wave_a = sign * (pivots[1].price - pivots[0].price)
    wave_b = sign * (pivots[1].price - pivots[2].price)
    wave_c = sign * (pivots[3].price - pivots[2].price)
    b_ratio = wave_b / wave_a
    c_ratio = wave_c / wave_a
    b_points = _range_alignment(b_ratio, 1.0, 1.05, 0.1, 30)
    c_points = _target_alignment(c_ratio, (1.0,), 0.25, 20)

    a_strength = _wave_momentum(
        rsi, pivots[0].timestamp, pivots[1].timestamp, sign
    )
    c_strength = _wave_momentum(
        rsi, pivots[2].timestamp, pivots[3].timestamp, sign
    )
    c_confirms = (
        a_strength is not None
        and c_strength is not None
        and c_strength >= a_strength
    )
    c_extends = sign * (pivots[3].price - pivots[1].price) > 0
    c_diverges = (
        a_strength is not None
        and c_strength is not None
        and c_extends
        and c_strength < a_strength
    )
    confirmation_points = 20.0 if c_confirms else 0.0
    divergence_points = 10.0 if c_diverges else 0.0

    duration_a = (pivots[1].timestamp - pivots[0].timestamp).total_seconds()
    duration_c = (pivots[3].timestamp - pivots[2].timestamp).total_seconds()
    duration_symmetry = min(duration_a, duration_c) / max(duration_a, duration_c)
    duration_points = 10.0 * duration_symmetry
    leg_balance = min(wave_b, wave_c) / max(wave_b, wave_c)
    balance_points = 10.0 * leg_balance
    items = (
        ScoreItem(
            "Fibonacci Alignment",
            b_points,
            30,
            f"Flat Wave B/A={b_ratio:.4f}; ideal expanded range=1.000-1.050",
        ),
        ScoreItem(
            "Fibonacci Alignment",
            c_points,
            20,
            f"Flat Wave C/A={c_ratio:.4f}; regular-flat target=1.000",
        ),
        ScoreItem(
            "Momentum Verification",
            confirmation_points,
            20,
            "Wave C RSI strength confirms Wave A"
            if c_confirms
            else "Wave C lacks RSI confirmation versus Wave A",
        ),
        ScoreItem(
            "Momentum Verification",
            divergence_points,
            10,
            "Extended Wave C shows RSI divergence"
            if c_diverges
            else "No qualifying extended Wave C divergence",
        ),
        ScoreItem(
            "Channeling & Alternation",
            duration_points,
            10,
            f"Wave A/C duration symmetry={duration_symmetry:.4f}",
        ),
        ScoreItem(
            "Channeling & Alternation",
            balance_points,
            10,
            f"Wave B/C price-leg balance={leg_balance:.4f}",
        ),
    )
    return _assemble_score(items)


def _score_triangle(candidate: WaveCandidate, rsi: pd.Series) -> ConfidenceScore:
    pivots = candidate.pivots
    lengths = tuple(
        abs(right.price - left.price)
        for left, right in zip(pivots, pivots[1:])
    )
    ratios = tuple(
        current / previous
        for previous, current in zip(lengths, lengths[1:])
    )
    contraction_items = tuple(
        ScoreItem(
            "Fibonacci Alignment",
            _range_alignment(ratio, 0.5, 0.8, 0.2, 12.5),
            12.5,
            f"Triangle {label} contraction ratio={ratio:.4f}",
        )
        for ratio, label in zip(ratios, ("B/A", "C/B", "D/C", "E/D"))
    )

    strengths = tuple(
        _wave_momentum(
            rsi,
            left.timestamp,
            right.timestamp,
            1.0 if right.price > left.price else -1.0,
        )
        for left, right in zip(pivots, pivots[1:])
    )
    momentum_points = sum(
        7.5
        for previous, current in zip(strengths, strengths[1:])
        if previous is not None and current is not None and current < previous
    )

    highs = [pivot.price for pivot in pivots if pivot.type == "High"]
    lows = [pivot.price for pivot in pivots if pivot.type == "Low"]
    highs_contract = all(
        current < previous for previous, current in zip(highs, highs[1:])
    )
    lows_contract = all(
        current > previous for previous, current in zip(lows, lows[1:])
    )
    wedge_points = 15.0 if highs_contract and lows_contract else 0.0
    mean_ratio = sum(ratios) / len(ratios)
    mean_deviation = sum(abs(ratio - mean_ratio) for ratio in ratios) / len(ratios)
    symmetry_points = max(0.0, 5.0 * (1.0 - mean_deviation / 0.2))
    items = contraction_items + (
        ScoreItem(
            "Momentum Verification",
            momentum_points,
            30,
            f"{int(momentum_points / 7.5)}/4 consecutive legs show declining RSI",
        ),
        ScoreItem(
            "Channeling & Alternation",
            wedge_points,
            15,
            "Upper highs descend and lower lows ascend in a contracting wedge"
            if wedge_points
            else "Pivot boundaries do not form a clean contracting wedge",
        ),
        ScoreItem(
            "Channeling & Alternation",
            symmetry_points,
            5,
            f"Contraction-ratio mean deviation={mean_deviation:.4f}",
        ),
    )
    return _assemble_score(items)


def _wave_momentum(
    rsi: pd.Series,
    start: pd.Timestamp,
    end: pd.Timestamp,
    sign: float,
) -> float | None:
    window = rsi.loc[(rsi.index >= start) & (rsi.index <= end)].dropna()
    if window.empty:
        return None
    strength = window if sign > 0 else 100.0 - window
    return float(strength.max())


def _range_alignment(
    value: float, lower: float, upper: float, tolerance: float, maximum: float
) -> float:
    distance = max(lower - value, value - upper, 0.0)
    return max(0.0, maximum * (1.0 - distance / tolerance))


def _target_alignment(
    value: float, targets: tuple[float, ...], tolerance: float, maximum: float
) -> float:
    distance = min(abs(value - target) for target in targets)
    return max(0.0, maximum * (1.0 - distance / tolerance))


def _channel_error(candidate: WaveCandidate) -> float:
    pivots = candidate.pivots
    t1 = pivots[1].timestamp.value
    t2 = pivots[2].timestamp.value
    t3 = pivots[3].timestamp.value
    t4 = pivots[4].timestamp.value
    if t3 == t1:
        return 1.0
    slope = (pivots[3].price - pivots[1].price) / (t3 - t1)
    projected = pivots[2].price + slope * (t4 - t2)
    scale = abs(pivots[1].price - pivots[0].price)
    return abs(pivots[4].price - projected) / scale


def _assemble_score(items: tuple[ScoreItem, ...]) -> ConfidenceScore:
    fibonacci = min(50.0, sum(i.points for i in items if i.category == "Fibonacci Alignment"))
    momentum = min(30.0, sum(i.points for i in items if i.category == "Momentum Verification"))
    channeling = min(
        20.0, sum(i.points for i in items if i.category == "Channeling & Alternation")
    )
    total = min(100.0, fibonacci + momentum + channeling)
    return ConfidenceScore(
        fibonacci=round(fibonacci, 4),
        momentum=round(momentum, 4),
        channeling_alternation=round(channeling, 4),
        total=round(total, 4),
        items=items,
    )


def _validate_market_data(ohlcv: pd.DataFrame) -> None:
    if not isinstance(ohlcv, pd.DataFrame):
        raise TypeError("ohlcv must be a pandas DataFrame")
    if "close" not in ohlcv.columns:
        raise ValueError("ohlcv requires a close column")
    if not isinstance(ohlcv.index, pd.DatetimeIndex):
        raise ValueError("ohlcv must use a DatetimeIndex")
    if not ohlcv.index.is_monotonic_increasing or ohlcv.index.has_duplicates:
        raise ValueError("ohlcv index must be unique and chronological")
    if ohlcv["close"].isna().any():
        raise ValueError("close values must not be null")
