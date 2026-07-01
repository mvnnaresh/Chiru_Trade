"""Deterministic Elliott Wave candidate graph and hard-rule filtering."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd

from pivots import ActiveLeg, Pivot, PivotState

Pattern = Literal["Impulse", "ZigZag", "Flat", "Triangle"]
Direction = Literal["Bullish", "Bearish"]
BreachSide = Literal["below", "above"]
CandidateStatus = Literal["Forming", "EntryReady", "Completed", "Invalidated"]


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
    status: CandidateStatus = "Completed"
    active_leg: ActiveLeg | None = None
    forming_label: Literal["4", "B"] | None = None
    as_of: pd.Timestamp | None = None

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
        include_flats: bool = True,
        include_triangles: bool = True,
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
        if include_flats:
            for start in range(len(self.nodes) - 3):
                if _is_path(edge_set, start, 4):
                    candidate = evaluate_flat(self.nodes[start : start + 4])
                    if candidate is not None:
                        candidates.append(candidate)
        if include_triangles:
            for start in range(len(self.nodes) - 5):
                if _is_path(edge_set, start, 6):
                    candidate = evaluate_triangle(self.nodes[start : start + 6])
                    if candidate is not None:
                        candidates.append(candidate)
        return sorted(
            candidates,
            key=lambda candidate: (
                candidate.pivots[0].timestamp,
                ("Impulse", "Triangle", "Flat", "ZigZag").index(
                    candidate.pattern
                ),
            ),
        )


def build_candidates(
    pivots: list[Pivot] | tuple[Pivot, ...],
    *,
    include_impulses: bool = True,
    include_zigzags: bool = True,
    include_flats: bool = True,
    include_triangles: bool = True,
) -> list[WaveCandidate]:
    """Convenience API to build the DAG and return all surviving paths."""
    return WaveDAG.from_pivots(pivots).candidates(
        include_impulses=include_impulses,
        include_zigzags=include_zigzags,
        include_flats=include_flats,
        include_triangles=include_triangles,
    )


def find_provisional_candidates(
    state: PivotState,
    *,
    include_impulses: bool = True,
    include_zigzags: bool = True,
) -> list[WaveCandidate]:
    """Return causal forming and entry-ready paths at the market edge.

    Forming paths use ``state.active_leg`` without promoting it to a confirmed
    pivot. Entry-ready paths terminate at a confirmed Wave 4/B and record the
    snapshot's ``as_of`` detection time.
    """
    if not isinstance(state, PivotState):
        raise TypeError("state must be a PivotState")
    nodes = state.confirmed
    _validate_pivots(nodes)
    if nodes and state.as_of is not None and state.as_of < nodes[-1].timestamp:
        raise ValueError("state as_of cannot precede the latest confirmed pivot")
    if state.active_leg is not None:
        if nodes and state.active_leg.timestamp <= nodes[-1].timestamp:
            raise ValueError("active leg must follow the latest confirmed pivot")
        if state.as_of is not None and state.active_leg.timestamp > state.as_of:
            raise ValueError("active leg cannot occur after state as_of")
    candidates: list[WaveCandidate] = []

    if include_impulses:
        if len(nodes) >= 4 and state.active_leg is not None:
            forming_impulse = _evaluate_forming_impulse(
                nodes[-4:], state.active_leg, state.as_of
            )
            if forming_impulse is not None:
                candidates.append(forming_impulse)
        if len(nodes) >= 5:
            entry_impulse = _evaluate_entry_ready_impulse(
                nodes[-5:], state.as_of
            )
            if entry_impulse is not None:
                candidates.append(entry_impulse)

    if include_zigzags:
        if len(nodes) >= 2 and state.active_leg is not None:
            forming_zigzag = _evaluate_forming_zigzag(
                nodes[-2:], state.active_leg, state.as_of
            )
            if forming_zigzag is not None:
                candidates.append(forming_zigzag)
        if len(nodes) >= 3:
            entry_zigzag = _evaluate_entry_ready_zigzag(
                nodes[-3:], state.as_of
            )
            if entry_zigzag is not None:
                candidates.append(entry_zigzag)

    return candidates


def _evaluate_forming_impulse(
    pivots: tuple[Pivot, ...],
    active_leg: ActiveLeg,
    as_of: pd.Timestamp | None,
) -> WaveCandidate | None:
    if len(pivots) != 4:
        raise ValueError("a forming Impulse requires Start and Waves 1-3")
    _validate_pivots(pivots)
    if not _alternates(pivots):
        return None
    bullish = pivots[0].type == "Low"
    sign = 1.0 if bullish else -1.0
    if active_leg.type != ("Low" if bullish else "High"):
        return None
    wave_1 = sign * (pivots[1].price - pivots[0].price)
    wave_3 = sign * (pivots[3].price - pivots[2].price)
    rule_1 = sign * (pivots[2].price - pivots[0].price) >= 0
    rule_3 = sign * (active_leg.price - pivots[1].price) > 0
    states = (
        RuleState(
            "wave_2_retracement",
            rule_1 and wave_1 > 0,
            "Wave 2 must not pass the origin of Wave 1",
        ),
        RuleState(
            "wave_3_direction",
            wave_3 > 0,
            "Wave 3 must advance in the impulse direction",
        ),
        RuleState(
            "forming_wave_4_no_overlap",
            rule_3,
            "Forming Wave 4 must remain outside Wave 1 territory",
        ),
    )
    if not all(item.passed for item in states):
        return None
    return WaveCandidate(
        "Impulse",
        "Bullish" if bullish else "Bearish",
        pivots,
        ("Start", "1", "2", "3"),
        states,
        float(pivots[1].price),
        "below" if bullish else "above",
        status="Forming",
        active_leg=active_leg,
        forming_label="4",
        as_of=as_of,
    )


def _evaluate_entry_ready_impulse(
    pivots: tuple[Pivot, ...],
    as_of: pd.Timestamp | None,
) -> WaveCandidate | None:
    if len(pivots) != 5:
        raise ValueError("an entry-ready Impulse requires Start and Waves 1-4")
    _validate_pivots(pivots)
    if not _alternates(pivots):
        return None
    bullish = pivots[0].type == "Low"
    sign = 1.0 if bullish else -1.0
    states = (
        RuleState(
            "wave_2_retracement",
            sign * (pivots[2].price - pivots[0].price) >= 0,
            "Wave 2 must not pass the origin of Wave 1",
        ),
        RuleState(
            "wave_3_direction",
            sign * (pivots[3].price - pivots[2].price) > 0,
            "Wave 3 must advance in the impulse direction",
        ),
        RuleState(
            "wave_4_no_overlap",
            sign * (pivots[4].price - pivots[1].price) > 0,
            "Wave 4 must remain outside Wave 1 territory",
        ),
    )
    if not all(item.passed for item in states):
        return None
    return WaveCandidate(
        "Impulse",
        "Bullish" if bullish else "Bearish",
        pivots,
        ("Start", "1", "2", "3", "4"),
        states,
        float(pivots[1].price),
        "below" if bullish else "above",
        status="EntryReady",
        as_of=as_of,
    )


def _evaluate_forming_zigzag(
    pivots: tuple[Pivot, ...],
    active_leg: ActiveLeg,
    as_of: pd.Timestamp | None,
) -> WaveCandidate | None:
    if len(pivots) != 2:
        raise ValueError("a forming ZigZag requires an origin and Wave A")
    _validate_pivots(pivots)
    if not _alternates(pivots):
        return None
    bearish = pivots[0].type == "High"
    sign = -1.0 if bearish else 1.0
    if active_leg.type != ("High" if bearish else "Low"):
        return None
    wave_a = sign * (pivots[1].price - pivots[0].price)
    retracement = sign * (pivots[1].price - active_leg.price)
    ratio = retracement / wave_a if wave_a > 0 else -1.0
    state = RuleState(
        "forming_wave_b_within_origin",
        0 <= ratio < 0.9,
        "Forming Wave B must retrace less than 90% of Wave A",
    )
    if not state.passed:
        return None
    return WaveCandidate(
        "ZigZag",
        "Bearish" if bearish else "Bullish",
        pivots,
        ("Start", "A"),
        (state,),
        float(pivots[0].price),
        "above" if bearish else "below",
        status="Forming",
        active_leg=active_leg,
        forming_label="B",
        as_of=as_of,
    )


def _evaluate_entry_ready_zigzag(
    pivots: tuple[Pivot, ...],
    as_of: pd.Timestamp | None,
) -> WaveCandidate | None:
    if len(pivots) != 3:
        raise ValueError("an entry-ready ZigZag requires an origin, A and B")
    _validate_pivots(pivots)
    if not _alternates(pivots):
        return None
    bearish = pivots[0].type == "High"
    sign = -1.0 if bearish else 1.0
    wave_a = sign * (pivots[1].price - pivots[0].price)
    retracement = sign * (pivots[1].price - pivots[2].price)
    ratio = retracement / wave_a if wave_a > 0 else -1.0
    state = RuleState(
        "wave_b_within_origin",
        0 <= ratio < 0.9,
        "Wave B must retrace less than 90% of Wave A",
    )
    if not state.passed:
        return None
    return WaveCandidate(
        "ZigZag",
        "Bearish" if bearish else "Bullish",
        pivots,
        ("Start", "A", "B"),
        (state,),
        float(pivots[0].price),
        "above" if bearish else "below",
        status="EntryReady",
        as_of=as_of,
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
    wave_a = sign * (pivots[1].price - pivots[0].price)
    wave_b = sign * (pivots[1].price - pivots[2].price)
    if wave_a <= 0:
        return None
    b_ratio = wave_b / wave_a
    b_within_origin = 0 <= b_ratio < 0.9
    state = RuleState(
        "wave_b_within_origin",
        b_within_origin,
        "Wave B must retrace less than 90% of Wave A",
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


def evaluate_flat(
    pivots: list[Pivot] | tuple[Pivot, ...],
) -> WaveCandidate | None:
    """Validate a deterministic regular/expanded A-B-C Flat.

    Wave B must retrace at least 90% of A. Wave C's endpoint must finish
    within 20% of Wave A's endpoint, making "near" explicit and auditable.
    """
    pivots = tuple(pivots)
    if len(pivots) != 4:
        raise ValueError("a Flat requires an origin and A, B, C endpoints")
    _validate_pivots(pivots)
    if not _alternates(pivots):
        return None

    bearish = pivots[0].type == "High"
    sign = -1.0 if bearish else 1.0
    wave_a = sign * (pivots[1].price - pivots[0].price)
    wave_b = sign * (pivots[1].price - pivots[2].price)
    wave_c = sign * (pivots[3].price - pivots[2].price)
    if min(wave_a, wave_b, wave_c) <= 0:
        return None
    b_ratio = wave_b / wave_a
    c_endpoint_error = abs(pivots[3].price - pivots[1].price) / wave_a
    states = (
        RuleState(
            "flat_b_minimum_retracement",
            b_ratio >= 0.9,
            "Wave B must retrace at least 90% of Wave A",
        ),
        RuleState(
            "flat_c_near_a_extreme",
            c_endpoint_error <= 0.2,
            "Wave C must terminate within 20% of Wave A's extreme",
        ),
    )
    if not all(state.passed for state in states):
        return None

    return WaveCandidate(
        pattern="Flat",
        direction="Bearish" if bearish else "Bullish",
        pivots=pivots,
        labels=("Start", "A", "B", "C"),
        rule_states=states,
        invalidation_level=float(pivots[2].price),
        invalidation_side="above" if bearish else "below",
    )


def evaluate_triangle(
    pivots: list[Pivot] | tuple[Pivot, ...],
) -> WaveCandidate | None:
    """Validate a five-leg contracting A-B-C-D-E Triangle."""
    pivots = tuple(pivots)
    if len(pivots) != 6:
        raise ValueError(
            "a Triangle requires an origin and A, B, C, D, E endpoints"
        )
    _validate_pivots(pivots)
    if not _alternates(pivots):
        return None

    lengths = tuple(
        abs(right.price - left.price)
        for left, right in zip(pivots, pivots[1:])
    )
    states = tuple(
        RuleState(
            f"triangle_{current_label}_contracts",
            current < previous,
            f"Wave {current_label} must be smaller than Wave {previous_label}",
        )
        for previous, current, previous_label, current_label in zip(
            lengths, lengths[1:], ("A", "B", "C", "D"), ("B", "C", "D", "E")
        )
    )
    if not all(state.passed for state in states):
        return None

    bearish = pivots[0].type == "High"
    return WaveCandidate(
        pattern="Triangle",
        direction="Bearish" if bearish else "Bullish",
        pivots=pivots,
        labels=("Start", "A", "B", "C", "D", "E"),
        rule_states=states,
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
