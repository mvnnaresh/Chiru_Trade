"""Walk-forward and parameter-sensitivity validation for ranked signals."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import pandas as pd

from backtester import BacktestSummary, CandidateRanking, Friction, run_backtest


@dataclass(frozen=True, slots=True)
class FoldResult:
    fold: int
    start: pd.Timestamp
    end: pd.Timestamp
    summary: BacktestSummary


@dataclass(frozen=True, slots=True)
class WalkForwardReport:
    folds: tuple[FoldResult, ...]
    total_trades: int
    net_profit_loss: float
    worst_drawdown: float


@dataclass(frozen=True, slots=True)
class SensitivityRow:
    atr_multiplier: float
    atr_period: int
    summary: BacktestSummary


@dataclass(frozen=True, slots=True)
class SensitivityReport:
    rows: tuple[SensitivityRow, ...]
    net_profit_range: float
    trade_count_range: int
    worst_drawdown: float


def walk_forward_validate(
    ohlcv: pd.DataFrame,
    rankings: tuple[CandidateRanking, ...],
    *,
    folds: int,
    minimum_confidence: float,
    friction: Friction = Friction(),
) -> WalkForwardReport:
    """Evaluate chronological, non-overlapping out-of-sample windows."""
    if not isinstance(folds, int) or isinstance(folds, bool) or folds < 2:
        raise ValueError("folds must be an integer of at least 2")
    if len(ohlcv) < folds:
        raise ValueError("ohlcv must contain at least one bar per fold")
    boundaries = [round(i * len(ohlcv) / folds) for i in range(folds + 1)]
    results: list[FoldResult] = []
    for fold in range(folds):
        window = ohlcv.iloc[boundaries[fold] : boundaries[fold + 1]]
        fold_rankings = tuple(
            ranking
            for ranking in rankings
            if window.index[0] <= ranking.detected_at <= window.index[-1]
        )
        summary = run_backtest(
            window,
            fold_rankings,
            minimum_confidence=minimum_confidence,
            friction=friction,
        )
        results.append(
            FoldResult(fold + 1, window.index[0], window.index[-1], summary)
        )
    return WalkForwardReport(
        folds=tuple(results),
        total_trades=sum(item.summary.total_trades for item in results),
        net_profit_loss=round(
            sum(item.summary.net_profit_loss for item in results), 10
        ),
        worst_drawdown=max(
            (item.summary.maximum_drawdown for item in results), default=0.0
        ),
    )


def parameter_sensitivity(
    ohlcv: pd.DataFrame,
    rankings_by_parameter: Mapping[
        tuple[float, int], tuple[CandidateRanking, ...]
    ],
    *,
    minimum_confidence: float,
    friction: Friction = Friction(),
) -> SensitivityReport:
    """Compare precomputed causal rankings across an explicit parameter grid."""
    if not rankings_by_parameter:
        raise ValueError("rankings_by_parameter must not be empty")
    rows = tuple(
        SensitivityRow(
            atr_multiplier=float(multiplier),
            atr_period=int(period),
            summary=run_backtest(
                ohlcv,
                rankings,
                minimum_confidence=minimum_confidence,
                friction=friction,
            ),
        )
        for (multiplier, period), rankings in sorted(
            rankings_by_parameter.items()
        )
    )
    profits = [row.summary.net_profit_loss for row in rows]
    counts = [row.summary.total_trades for row in rows]
    return SensitivityReport(
        rows=rows,
        net_profit_range=round(max(profits) - min(profits), 10),
        trade_count_range=max(counts) - min(counts),
        worst_drawdown=max(row.summary.maximum_drawdown for row in rows),
    )
