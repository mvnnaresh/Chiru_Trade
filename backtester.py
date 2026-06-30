"""Causal event-driven simulation for ranked Elliott Wave setups."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Mapping

import pandas as pd

from engine import WaveCandidate, build_candidates
from pivots import extract_pivots
from scoring import ConfidenceScore, score_candidates


@dataclass(frozen=True, slots=True)
class Friction:
    """Fixed price-unit costs. Commission is charged on each fill."""

    spread: float = 0.0
    slippage: float = 0.0
    commission: float = 0.0

    def __post_init__(self) -> None:
        if min(self.spread, self.slippage, self.commission) < 0:
            raise ValueError("friction values must be non-negative")


@dataclass(frozen=True, slots=True)
class CandidateRanking:
    """A score that became observable at a specific historical bar."""

    detected_at: pd.Timestamp
    candidate: WaveCandidate
    confidence_score: float

    def __post_init__(self) -> None:
        if not 0 <= self.confidence_score <= 100:
            raise ValueError("confidence_score must be between 0 and 100")


@dataclass(frozen=True, slots=True)
class Trade:
    direction: str
    signal_time: pd.Timestamp
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    target_price: float
    stop_price: float
    confidence_score: float
    exit_reason: str
    gross_pnl: float
    friction_cost: float
    net_pnl: float


@dataclass(frozen=True, slots=True)
class BacktestSummary:
    total_trades: int
    win_rate: float
    net_profit_loss: float
    maximum_drawdown: float
    ledger: tuple[Trade, ...]


Ranker = Callable[
    [pd.DataFrame], Mapping[WaveCandidate, ConfidenceScore] | Iterable[tuple[WaveCandidate, float]]
]


def build_causal_rankings(
    ohlcv: pd.DataFrame,
    *,
    multiplier: float,
    atr_period: int = 14,
    rsi_period: int = 14,
    ranker: Ranker | None = None,
) -> tuple[CandidateRanking, ...]:
    """Build historical rankings by giving the engine only each bar prefix.

    ``ranker`` is injectable for testing or alternate deterministic ranking
    policies. The default invokes pivots, engine, and scoring on every prefix.
    A candidate path is recorded only the first time it becomes observable.
    """
    _validate_ohlcv(ohlcv)

    def default_ranker(prefix: pd.DataFrame) -> Mapping[WaveCandidate, ConfidenceScore]:
        pivots = extract_pivots(prefix, multiplier, atr_period=atr_period)
        candidates = build_candidates(pivots)
        return score_candidates(candidates, prefix, rsi_period=rsi_period)

    selected_ranker = ranker or default_ranker
    seen: set[WaveCandidate] = set()
    rankings: list[CandidateRanking] = []
    for position, timestamp in enumerate(ohlcv.index):
        prefix = ohlcv.iloc[: position + 1].copy()
        ranked = selected_ranker(prefix)
        entries = ranked.items() if isinstance(ranked, Mapping) else ranked
        for candidate, score_value in entries:
            if candidate in seen:
                continue
            score = (
                score_value.total
                if isinstance(score_value, ConfidenceScore)
                else float(score_value)
            )
            _assert_candidate_is_causal(candidate, timestamp)
            rankings.append(CandidateRanking(timestamp, candidate, score))
            seen.add(candidate)
    return tuple(rankings)


def run_backtest(
    ohlcv: pd.DataFrame,
    rankings: list[CandidateRanking] | tuple[CandidateRanking, ...],
    *,
    minimum_confidence: float,
    friction: Friction = Friction(),
) -> BacktestSummary:
    """Simulate ranked Wave-4 or Wave-B setups with one open trade at a time.

    Rankings are consumed only at ``detected_at``. Entry is that bar's close.
    If target and stop are both touched in one bar, the stop is chosen as the
    conservative deterministic outcome.
    """
    _validate_ohlcv(ohlcv)
    if not 0 <= minimum_confidence <= 100:
        raise ValueError("minimum_confidence must be between 0 and 100")
    ordered = sorted(rankings, key=lambda item: item.detected_at)
    if list(rankings) != ordered:
        raise ValueError("rankings must be chronological")

    rankings_by_time: dict[pd.Timestamp, list[CandidateRanking]] = {}
    for ranking in ordered:
        if ranking.detected_at not in ohlcv.index:
            raise ValueError("each detected_at timestamp must exist in OHLCV data")
        _assert_candidate_is_causal(ranking.candidate, ranking.detected_at)
        rankings_by_time.setdefault(ranking.detected_at, []).append(ranking)

    active: dict[str, object] | None = None
    ledger: list[Trade] = []
    for timestamp, bar in ohlcv.iterrows():
        if active is not None and timestamp > active["entry_time"]:
            exit_result = _check_exit(active, timestamp, bar, friction)
            if exit_result is not None:
                ledger.append(exit_result)
                active = None

        if active is None:
            eligible = [
                ranking
                for ranking in rankings_by_time.get(timestamp, ())
                if ranking.confidence_score >= minimum_confidence
                and _is_tradeable_setup(ranking.candidate)
            ]
            if eligible:
                ranking = max(eligible, key=lambda item: item.confidence_score)
                active = _open_trade(ranking, float(bar["close"]), friction)

    if active is not None:
        last_time = ohlcv.index[-1]
        last_close = float(ohlcv.iloc[-1]["close"])
        ledger.append(_close_at_end(active, last_time, last_close, friction))

    return _summary(tuple(ledger))


def _open_trade(
    ranking: CandidateRanking, close: float, friction: Friction
) -> dict[str, object]:
    candidate = ranking.candidate
    sign = 1.0 if candidate.direction == "Bullish" else -1.0
    setup_index = candidate.labels.index("4") if "4" in candidate.labels else candidate.labels.index("B")
    setup_price = candidate.pivots[setup_index].price
    first_leg = abs(candidate.pivots[1].price - candidate.pivots[0].price)
    target = setup_price + sign * 1.618 * first_leg
    adverse_entry = sign * (friction.spread / 2 + friction.slippage)
    return {
        "candidate": candidate,
        "direction": candidate.direction,
        "sign": sign,
        "signal_time": ranking.detected_at,
        "entry_time": ranking.detected_at,
        "entry_price": close + adverse_entry,
        "target": target,
        "stop": candidate.invalidation_level,
        "confidence": ranking.confidence_score,
    }


def _check_exit(
    active: dict[str, object],
    timestamp: pd.Timestamp,
    bar: pd.Series,
    friction: Friction,
) -> Trade | None:
    sign = float(active["sign"])
    target = float(active["target"])
    stop = float(active["stop"])
    if sign > 0:
        stop_hit = float(bar["low"]) <= stop
        target_hit = float(bar["high"]) >= target
    else:
        stop_hit = float(bar["high"]) >= stop
        target_hit = float(bar["low"]) <= target
    if not stop_hit and not target_hit:
        return None
    raw_exit = stop if stop_hit else target
    return _make_trade(
        active,
        timestamp,
        raw_exit,
        "stop_loss" if stop_hit else "take_profit",
        friction,
    )


def _close_at_end(
    active: dict[str, object],
    timestamp: pd.Timestamp,
    close: float,
    friction: Friction,
) -> Trade:
    return _make_trade(active, timestamp, close, "end_of_data", friction)


def _make_trade(
    active: dict[str, object],
    exit_time: pd.Timestamp,
    raw_exit: float,
    reason: str,
    friction: Friction,
) -> Trade:
    sign = float(active["sign"])
    exit_price = raw_exit - sign * (friction.spread / 2 + friction.slippage)
    entry_price = float(active["entry_price"])
    gross = sign * (raw_exit - (entry_price - sign * (friction.spread / 2 + friction.slippage)))
    friction_cost = friction.spread + 2 * friction.slippage + 2 * friction.commission
    net = sign * (exit_price - entry_price) - 2 * friction.commission
    return Trade(
        direction=str(active["direction"]),
        signal_time=active["signal_time"],  # type: ignore[arg-type]
        entry_time=active["entry_time"],  # type: ignore[arg-type]
        exit_time=exit_time,
        entry_price=entry_price,
        exit_price=exit_price,
        target_price=float(active["target"]),
        stop_price=float(active["stop"]),
        confidence_score=float(active["confidence"]),
        exit_reason=reason,
        gross_pnl=round(gross, 10),
        friction_cost=round(friction_cost, 10),
        net_pnl=round(net, 10),
    )


def _summary(ledger: tuple[Trade, ...]) -> BacktestSummary:
    total = len(ledger)
    wins = sum(trade.net_pnl > 0 for trade in ledger)
    net = sum(trade.net_pnl for trade in ledger)
    equity = 0.0
    peak = 0.0
    maximum_drawdown = 0.0
    for trade in sorted(ledger, key=lambda item: item.exit_time):
        equity += trade.net_pnl
        peak = max(peak, equity)
        maximum_drawdown = max(maximum_drawdown, peak - equity)
    return BacktestSummary(
        total_trades=total,
        win_rate=round(100 * wins / total, 4) if total else 0.0,
        net_profit_loss=round(net, 10),
        maximum_drawdown=round(maximum_drawdown, 10),
        ledger=ledger,
    )


def _is_tradeable_setup(candidate: WaveCandidate) -> bool:
    return bool(candidate.labels) and candidate.labels[-1] in {"4", "B"}


def _assert_candidate_is_causal(
    candidate: WaveCandidate, detected_at: pd.Timestamp
) -> None:
    if candidate.pivots and candidate.pivots[-1].timestamp > detected_at:
        raise ValueError("candidate contains a pivot from after detected_at")


def _validate_ohlcv(ohlcv: pd.DataFrame) -> None:
    required = {"high", "low", "close"}
    if not isinstance(ohlcv, pd.DataFrame):
        raise TypeError("ohlcv must be a pandas DataFrame")
    if missing := required.difference(ohlcv.columns):
        raise ValueError(f"missing OHLCV columns: {', '.join(sorted(missing))}")
    if not isinstance(ohlcv.index, pd.DatetimeIndex):
        raise ValueError("ohlcv must use a DatetimeIndex")
    if ohlcv.empty:
        raise ValueError("ohlcv must not be empty")
    if not ohlcv.index.is_monotonic_increasing or ohlcv.index.has_duplicates:
        raise ValueError("ohlcv index must be unique and chronological")
    if ohlcv.loc[:, list(required)].isna().any().any():
        raise ValueError("OHLCV values must not be null")
