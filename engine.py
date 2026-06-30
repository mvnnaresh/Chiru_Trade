"""Deterministic Elliott Wave candidate graph and hard-rule filtering."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pivots import Pivot

Pattern = Literal["Impulse", "ZigZag"]
Direction = Literal["Bullish", "Bearish"]
BreachSide = Literal["below", "above"]


@dataclass(frozen=True, slots=True)
class RuleState:
    """Auditable result of one absolute structural rule."""

    name: str
    passed: bool
    detail: str


@dataclass(frozen=True, slots=True)
class WaveCandidate:
    """An immutable surviving path through the pivot DAG."""

    pattern: Pattern
    direction: Direction
    pivots: tuple[Pivot, ...]
    labels: tuple[str, ...]
    rule_states: tuple[RuleState, ...]
    invalidation_level: float
    invalidation_side: BreachSide

    @property
    def labeled_waves(self) -> tuple[tuple[str, Pivot], ...]:
        return tuple(zip(self.labels, self.pivots))


@dataclass(frozen=True, slots=True)
class WaveDAG:
    """A chronological DAG whose edges join adjacent alternating pivots."""

    nodes: tuple[Pivot, ...]
    edges: tuple[tuple[int, int], ...]

    @classmethod
    def from_pivots(cls, pivots: list[Pivot] | tuple[Pivot, ...]) -> "WaveDAG":
        nodes = tuple(pivots)
        _validate_pivots(nodes)
        edges = tuple(
            (index, index + 1)
            for index in range(len(nodes) - 1)
            if nodes[index].type != nodes[index + 1].type
        )
        return cls(nodes=nodes, edges=edges)

    def candidates(
        self,
        *,
        include_impulses: bool = True,
        include_zigzags: bool = True,
    ) -> list[WaveCandidate]:
        """Return all valid contiguous completed patterns in start-time order."""
        edge_set = set(self.edges)
        candidates: list[WaveCandidate] = []
        if include_impulses:
            for start in range(len(self.nodes) - 5):
                if _is_path(edge_set, start, 6):
                    candidate = evaluate_impulse(self.nodes[start : start + 6])
                    if candidate is not None:
                        candidates.append(candidate)
        if include_zigzags:
            for start in range(len(self.nodes) - 3):
                if _is_path(edge_set, start, 4):
                    candidate = evaluate_zigzag(self.nodes[start : start + 4])
                    if candidate is not None:
                        candidates.append(candidate)
        return sorted(
            candidates,
            key=lambda candidate: (
                candidate.pivots[0].timestamp,
                0 if candidate.pattern == "Impulse" else 1,
            ),
        )


def build_candidates(
    pivots: list[Pivot] | tuple[Pivot, ...],
    *,
    include_impulses: bool = True,
    include_zigzags: bool = True,
) -> list[WaveCandidate]:
    """Convenience API to build the DAG and return all surviving paths."""
    return WaveDAG.from_pivots(pivots).candidates(
        include_impulses=include_impulses,
        include_zigzags=include_zigzags,
    )


def evaluate_impulse(
    pivots: list[Pivot] | tuple[Pivot, ...],
) -> WaveCandidate | None:
    """Apply all three absolute Elliott impulse rules to one six-pivot path."""
    pivots = tuple(pivots)
    if len(pivots) != 6:
        raise ValueError("an Impulse requires an origin and five wave endpoints")
    _validate_pivots(pivots)
    if not _alternates(pivots):
        return None

    bullish = pivots[0].type == "Low"
    direction: Direction = "Bullish" if bullish else "Bearish"
    prices = tuple(pivot.price for pivot in pivots)
    sign = 1.0 if bullish else -1.0

    wave_1 = sign * (prices[1] - prices[0])
    wave_3 = sign * (prices[3] - prices[2])
    wave_5 = sign * (prices[5] - prices[4])
    rule_1_passed = sign * (prices[2] - prices[0]) >= 0
    rule_2_passed = not (
        wave_3 < wave_1 and wave_3 < wave_5
    )
    rule_3_passed = sign * (prices[4] - prices[1]) > 0

    states = (
        RuleState(
            "wave_2_retracement",
            rule_1_passed,
            "Wave 2 must not pass the origin of Wave 1",
        ),
        RuleState(
            "wave_3_not_shortest",
            rule_2_passed,
            "Wave 3 must not be shorter than both Waves 1 and 5",
        ),
        RuleState(
            "wave_4_no_overlap",
            rule_3_passed,
            "Wave 4 must remain outside Wave 1 price territory",
        ),
    )
    if not all(state.passed for state in states):
        return None

    return WaveCandidate(
        pattern="Impulse",
        direction=direction,
        pivots=pivots,
        labels=("Start", "1", "2", "3", "4", "5"),
        rule_states=states,
        invalidation_level=float(prices[1]),
        invalidation_side="below" if bullish else "above",
    )


def evaluate_zigzag(
    pivots: list[Pivot] | tuple[Pivot, ...],
) -> WaveCandidate | None:
    """Validate one simple ABC ZigZag path.

    A simple ZigZag's B wave may reach, but must not pass, the origin of A.
    This provides its structural invalidation boundary.
    """
    pivots = tuple(pivots)
    if len(pivots) != 4:
        raise ValueError("a ZigZag requires an origin and A, B, C endpoints")
    _validate_pivots(pivots)
    if not _alternates(pivots):
        return None

    bearish = pivots[0].type == "High"
    sign = -1.0 if bearish else 1.0
    b_within_origin = sign * (pivots[2].price - pivots[0].price) >= 0
    state = RuleState(
        "wave_b_within_origin",
        b_within_origin,
        "Wave B must not pass the origin of Wave A",
    )
    if not state.passed:
        return None

    return WaveCandidate(
        pattern="ZigZag",
        direction="Bearish" if bearish else "Bullish",
        pivots=pivots,
        labels=("Start", "A", "B", "C"),
        rule_states=(state,),
        invalidation_level=float(pivots[0].price),
        invalidation_side="above" if bearish else "below",
    )


def _validate_pivots(pivots: tuple[Pivot, ...]) -> None:
    if any(not isinstance(pivot, Pivot) for pivot in pivots):
        raise TypeError("all nodes must be Pivot objects")
    timestamps = [pivot.timestamp for pivot in pivots]
    if any(current <= previous for previous, current in zip(timestamps, timestamps[1:])):
        raise ValueError("pivots must be strictly chronological")


def _alternates(pivots: tuple[Pivot, ...]) -> bool:
    return all(left.type != right.type for left, right in zip(pivots, pivots[1:]))


def _is_path(edges: set[tuple[int, int]], start: int, length: int) -> bool:
    return all((index, index + 1) in edges for index in range(start, start + length - 1))
