import pandas as pd

from backtester import CandidateRanking
from engine import RuleState, WaveCandidate
from pivots import Pivot
from validation import parameter_sensitivity, walk_forward_validate


def fixture():
    index = pd.date_range("2026-01-01", periods=12, freq="5min", tz="UTC")
    frame = pd.DataFrame(
        {
            "open": [100] * 12,
            "high": [101, 111, 105, 121, 113, 129] * 2,
            "low": [99, 103, 103, 111, 111, 111] * 2,
            "close": [100, 110, 104, 120, 112, 128] * 2,
            "volume": 1,
        },
        index=index,
    )
    pivots = tuple(
        Pivot(index[i], price, kind, 1.0)
        for i, (price, kind) in enumerate(
            zip(
                (100, 110, 104, 120, 112),
                ("Low", "High", "Low", "High", "Low"),
            )
        )
    )
    candidate = WaveCandidate(
        "Impulse",
        "Bullish",
        pivots,
        ("Start", "1", "2", "3", "4"),
        (RuleState("fixture", True, "valid"),),
        104,
        "below",
        status="EntryReady",
        as_of=index[4],
    )
    return frame, CandidateRanking(index[4], candidate, 80)


def test_walk_forward_report_uses_non_overlapping_chronological_folds():
    frame, ranking = fixture()

    report = walk_forward_validate(
        frame, (ranking,), folds=2, minimum_confidence=70
    )

    assert len(report.folds) == 2
    assert report.folds[0].end < report.folds[1].start
    assert report.total_trades == 1


def test_parameter_sensitivity_reports_metric_ranges():
    frame, ranking = fixture()

    report = parameter_sensitivity(
        frame,
        {(1.5, 14): (ranking,), (2.0, 14): ()},
        minimum_confidence=70,
    )

    assert len(report.rows) == 2
    assert report.trade_count_range == 1
    assert report.net_profit_range >= 0
