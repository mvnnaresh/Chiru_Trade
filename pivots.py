"""Causal ATR calculation and volatility-adjusted ZigZag pivot extraction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd

PivotType = Literal["High", "Low"]


@dataclass(frozen=True, slots=True)
class Pivot:
    """A confirmed swing extreme and the ATR observed at that extreme."""

    timestamp: pd.Timestamp
    price: float
    type: PivotType
    atr: float


def calculate_atr(ohlc: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate a causal rolling simple-average True Range.

    True Range at time ``t`` uses the current high/low and the close at
    ``t - 1``. ATR is unavailable until ``period`` observations exist.
    """
    _validate_ohlc(ohlc)
    if not isinstance(period, int) or isinstance(period, bool) or period <= 0:
        raise ValueError("period must be a positive integer")

    previous_close = ohlc["close"].shift(1)
    true_range = pd.concat(
        (
            ohlc["high"] - ohlc["low"],
            (ohlc["high"] - previous_close).abs(),
            (ohlc["low"] - previous_close).abs(),
        ),
        axis=1,
    ).max(axis=1)
    atr = true_range.rolling(window=period, min_periods=period).mean()
    atr.name = "atr"
    return atr


def extract_pivots(
    ohlc: pd.DataFrame,
    multiplier: float,
    *,
    atr_period: int = 14,
) -> list[Pivot]:
    """Return confirmed ZigZag pivots using ``multiplier * current ATR``.

    Processing is strictly chronological. During an upswing the highest high
    seen so far is retained; it becomes a confirmed High only when a subsequent
    low reverses by the threshold calculated at that subsequent candle. The
    inverse applies during a downswing. No provisional final pivot is emitted.
    """
    _validate_ohlc(ohlc)
    if not isinstance(multiplier, (int, float)) or isinstance(multiplier, bool):
        raise ValueError("multiplier must be a positive number")
    if multiplier <= 0:
        raise ValueError("multiplier must be a positive number")
    if ohlc.empty:
        return []

    atr = calculate_atr(ohlc, atr_period)
    ready = atr.notna() & (atr > 0)
    if not ready.any():
        return []

    pivots: list[Pivot] = []
    direction: Literal["up", "down"] | None = None
    candidate_high: tuple[pd.Timestamp, float, float] | None = None
    candidate_low: tuple[pd.Timestamp, float, float] | None = None

    for position in range(len(ohlc)):
        current_atr = atr.iloc[position]
        if pd.isna(current_atr) or current_atr <= 0:
            continue

        timestamp = ohlc.index[position]
        high = float(ohlc["high"].iloc[position])
        low = float(ohlc["low"].iloc[position])
        atr_value = float(current_atr)
        threshold = float(multiplier) * atr_value

        if candidate_high is None or high > candidate_high[1]:
            candidate_high = (timestamp, high, atr_value)
        if candidate_low is None or low < candidate_low[1]:
            candidate_low = (timestamp, low, atr_value)

        if direction is None:
            assert candidate_high is not None and candidate_low is not None
            if candidate_high[1] - candidate_low[1] < threshold:
                continue
            if candidate_low[0] < candidate_high[0]:
                pivots.append(_pivot(candidate_low, "Low"))
                direction = "up"
                candidate_low = None
            elif candidate_high[0] < candidate_low[0]:
                pivots.append(_pivot(candidate_high, "High"))
                direction = "down"
                candidate_high = None
            elif float(ohlc["close"].iloc[position]) >= float(
                ohlc["open"].iloc[position]
            ):
                # Deterministic intrabar assumption for a bullish body: low
                # occurred first. The same-timestamp high cannot be retained
                # as a future pivot because DAG timestamps must be strict.
                pivots.append(_pivot(candidate_low, "Low"))
                direction = "up"
                candidate_low = None
                candidate_high = None
            else:
                # Deterministic intrabar assumption for a bearish body: high
                # occurred first. Discard the same-timestamp low for the same
                # strict chronological invariant.
                pivots.append(_pivot(candidate_high, "High"))
                direction = "down"
                candidate_high = None
                candidate_low = None
            continue

        if direction == "up":
            assert candidate_high is not None
            if candidate_high[1] - low >= threshold:
                extreme_timestamp = candidate_high[0]
                pivots.append(_pivot(candidate_high, "High"))
                direction = "down"
                # If the confirming low belongs to the same candle as the
                # emitted high, retaining it could later produce two pivots
                # with one timestamp. Start opposite tracking next candle.
                candidate_low = (
                    None
                    if extreme_timestamp == timestamp
                    else (timestamp, low, atr_value)
                )
                candidate_high = None
        else:
            assert candidate_low is not None
            if high - candidate_low[1] >= threshold:
                extreme_timestamp = candidate_low[0]
                pivots.append(_pivot(candidate_low, "Low"))
                direction = "up"
                candidate_high = (
                    None
                    if extreme_timestamp == timestamp
                    else (timestamp, high, atr_value)
                )
                candidate_low = None

    return pivots


def _pivot(candidate: tuple[pd.Timestamp, float, float], kind: PivotType) -> Pivot:
    timestamp, price, atr = candidate
    return Pivot(timestamp=timestamp, price=price, type=kind, atr=atr)


def _validate_ohlc(ohlc: pd.DataFrame) -> None:
    if not isinstance(ohlc, pd.DataFrame):
        raise TypeError("ohlc must be a pandas DataFrame")
    missing = {"open", "high", "low", "close"}.difference(ohlc.columns)
    if missing:
        raise ValueError(f"missing OHLC columns: {', '.join(sorted(missing))}")
    if not isinstance(ohlc.index, pd.DatetimeIndex):
        raise ValueError("ohlc must use a DatetimeIndex")
    if not ohlc.index.is_monotonic_increasing or ohlc.index.has_duplicates:
        raise ValueError("ohlc index must be unique and chronological")
    values = ohlc.loc[:, ["high", "low", "close"]]
    if values.isna().any().any():
        raise ValueError("OHLC values must not be null")
    if (values["high"] < values["low"]).any():
        raise ValueError("high must be greater than or equal to low")
