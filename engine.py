"""Deterministic Elliott Wave candidate graph and hard-rule filtering."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

import pandas as pd

from pivots import ActiveLeg, Pivot, PivotState

Pattern = Literal["Impulse", "ZigZag", "Flat", "Triangle"]
Direction = Literal["Bullish", "Bearish"]
BreachSide = Literal["below", "above"]
CandidateStatus = Literal["Forming", "EntryReady", "Completed", "Invalidated"]
_STATUS_PRIORITY: dict[CandidateStatus, int] = {
    "EntryReady": 0,
    "Forming": 1,
    "Completed": 2,
    "Invalidated": 3,
}
_PATTERN_PRIORITY: dict[Pattern, int] = {
    "Impulse": 0,
    "ZigZag": 1,
    "Flat": 2,
    "Triangle": 3,
}


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
    variant: str | None = None
    node_indices: tuple[int, ...] = ()

    @property
    def labeled_waves(self) -> tuple[tuple[str, Pivot], ...]:
        return tuple(zip(self.labels, self.pivots))

    def pivot_for_label(self, label: str) -> Pivot | None:
        """Return the pivot currently assigned to a wave label, if present."""
        try:
            index = self.labels.index(label)
        except ValueError:
            return None
        return self.pivots[index]

    def wave_span(
        self, start_label: str, end_label: str
    ) -> tuple[Pivot, Pivot] | None:
        """Return the causal start/end pivots for a labeled wave segment."""
        start = self.pivot_for_label(start_label)
        end = self.pivot_for_label(end_label)
        if start is None or end is None:
            return None
        return start, end


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
        max_pivot_skip: int = 0,
        minimum_atr_displacement: float = 0.0,
    ) -> list[WaveCandidate]:
        """Return all valid contiguous completed patterns in start-time order."""
        if (
            not isinstance(max_pivot_skip, int)
            or isinstance(max_pivot_skip, bool)
            or max_pivot_skip < 0
        ):
            raise ValueError("max_pivot_skip must be a non-negative integer")
        if minimum_atr_displacement < 0:
            raise ValueError("minimum_atr_displacement must be non-negative")
        candidates: list[WaveCandidate] = []
        if include_impulses:
            for indices in _candidate_paths(
                self.nodes, 6, max_pivot_skip, minimum_atr_displacement
            ):
                candidate = evaluate_impulse(tuple(self.nodes[i] for i in indices))
                if candidate is not None:
                    candidates.append(replace(candidate, node_indices=indices))
        if include_zigzags:
            for indices in _candidate_paths(
                self.nodes, 4, max_pivot_skip, minimum_atr_displacement
            ):
                candidate = evaluate_zigzag(tuple(self.nodes[i] for i in indices))
                if candidate is not None:
                    candidates.append(replace(candidate, node_indices=indices))
        if include_flats:
            for indices in _candidate_paths(
                self.nodes, 4, max_pivot_skip, minimum_atr_displacement
            ):
                candidate = evaluate_flat(tuple(self.nodes[i] for i in indices))
                if candidate is not None:
                    candidates.append(replace(candidate, node_indices=indices))
        if include_triangles:
            for indices in _candidate_paths(
                self.nodes, 6, max_pivot_skip, minimum_atr_displacement
            ):
                candidate = evaluate_triangle(tuple(self.nodes[i] for i in indices))
                if candidate is not None:
                    candidates.append(replace(candidate, node_indices=indices))
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
    max_pivot_skip: int = 0,
    minimum_atr_displacement: float = 0.0,
) -> list[WaveCandidate]:
    """Convenience API to build the DAG and return all surviving paths."""
    return WaveDAG.from_pivots(pivots).candidates(
        include_impulses=include_impulses,
        include_zigzags=include_zigzags,
        include_flats=include_flats,
        include_triangles=include_triangles,
        max_pivot_skip=max_pivot_skip,
        minimum_atr_displacement=minimum_atr_displacement,
    )


def candidate_observable_time(candidate: WaveCandidate) -> pd.Timestamp:
    """Return the causal market-edge timestamp at which the path is observable."""
    if candidate.status == "Forming" and candidate.active_leg is not None:
        return candidate.active_leg.timestamp
    if candidate.as_of is not None:
        return candidate.as_of
    return candidate.pivots[-1].timestamp


def active_candidate_sort_key(candidate: WaveCandidate) -> tuple[int, int, int, int]:
    """Sort candidates for a live terminal: newest edge first, then structure quality."""
    observable_time = candidate_observable_time(candidate)
    return (
        -observable_time.value,
        _STATUS_PRIORITY[candidate.status],
        _PATTERN_PRIORITY[candidate.pattern],
        -len(candidate.labels),
    )


def select_active_primary(
    candidates: list[WaveCandidate] | tuple[WaveCandidate, ...],
) -> WaveCandidate | None:
    """Select the single structure that best represents the current live market state."""
    pool = tuple(candidates)
    if not pool:
        return None
    return min(pool, key=active_candidate_sort_key)


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
    wave_2 = sign * (pivots[1].price - pivots[2].price)
    wave_3 = sign * (pivots[3].price - pivots[2].price)
    wave_4 = sign * (pivots[3].price - active_leg.price)
    rule_1 = (
        wave_1 > 0
        and wave_2 > 0
        and sign * (pivots[2].price - pivots[0].price) > 0
    )
    rule_3_progress = sign * (pivots[3].price - pivots[1].price) > 0
    rule_4_retracement = wave_4 > 0 and wave_4 < wave_3
    rule_3 = sign * (active_leg.price - pivots[1].price) > 0
    states = (
        RuleState(
            "wave_2_retracement",
            rule_1 and wave_1 > 0,
            "Wave 2 must not pass the origin of Wave 1",
        ),
        RuleState(
            "wave_3_direction",
            wave_3 > 0 and rule_3_progress,
            "Wave 3 must advance beyond the end of Wave 1",
        ),
        RuleState(
            "forming_wave_4_retracement",
            rule_4_retracement,
            "Forming Wave 4 must retrace less than 100% of Wave 3",
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
    wave_1 = sign * (pivots[1].price - pivots[0].price)
    wave_2 = sign * (pivots[1].price - pivots[2].price)
    wave_3 = sign * (pivots[3].price - pivots[2].price)
    wave_4 = sign * (pivots[3].price - pivots[4].price)
    states = (
        RuleState(
            "wave_2_retracement",
            wave_1 > 0
            and wave_2 > 0
            and sign * (pivots[2].price - pivots[0].price) > 0,
            "Wave 2 must not pass the origin of Wave 1",
        ),
        RuleState(
            "wave_3_direction",
            wave_3 > 0 and sign * (pivots[3].price - pivots[1].price) > 0,
            "Wave 3 must advance beyond the end of Wave 1",
        ),
        RuleState(
            "wave_4_retracement",
            wave_4 > 0 and wave_4 < wave_3,
            "Wave 4 must retrace less than 100% of Wave 3",
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
        0 < ratio < 0.9,
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
        0 < ratio < 0.9,
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
    wave_2 = sign * (prices[1] - prices[2])
    wave_3 = sign * (prices[3] - prices[2])
    wave_4 = sign * (prices[3] - prices[4])
    wave_5 = sign * (prices[5] - prices[4])
    rule_1_passed = (
        wave_1 > 0
        and wave_2 > 0
        and sign * (prices[2] - prices[0]) > 0
    )
    rule_3_progressed = (
        wave_3 > 0 and sign * (prices[3] - prices[1]) > 0
    )
    rule_4_retracement = wave_4 > 0 and wave_4 < wave_3
    rule_2_passed = not (
        wave_3 < wave_1 and wave_3 < wave_5
    )
    rule_3_passed = sign * (prices[4] - prices[1]) > 0
    rule_5_progressed = (
        wave_5 > 0 and sign * (prices[5] - prices[3]) > 0
    )

    states = (
        RuleState(
            "wave_2_retracement",
            rule_1_passed,
            "Wave 2 must not pass the origin of Wave 1",
        ),
        RuleState(
            "wave_3_beyond_wave_1",
            rule_3_progressed,
            "Wave 3 must travel beyond the end of Wave 1",
        ),
        RuleState(
            "wave_3_not_shortest",
            rule_2_passed,
            "Wave 3 must not be shorter than both Waves 1 and 5",
        ),
        RuleState(
            "wave_4_retracement",
            rule_4_retracement,
            "Wave 4 must retrace less than 100% of Wave 3",
        ),
        RuleState(
            "wave_4_no_overlap",
            rule_3_passed,
            "Wave 4 must remain outside Wave 1 price territory",
        ),
        RuleState(
            "wave_5_progress",
            rule_5_progressed,
            "Standard Impulse Wave 5 must travel beyond Wave 3",
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
    wave_c = sign * (pivots[3].price - pivots[2].price)
    if wave_a <= 0:
        return None
    b_ratio = wave_b / wave_a
    b_within_origin = 0 <= b_ratio < 0.9
    c_advances = (
        wave_c > 0
        and sign * (pivots[3].price - pivots[1].price) > 0
    )
    states = (
        RuleState(
            "wave_b_within_origin",
            b_within_origin,
            "Wave B must retrace less than 90% of Wave A",
        ),
        RuleState(
            "wave_c_beyond_a",
            c_advances,
            "Wave C must advance beyond the end of Wave A",
        ),
    )
    if not all(state.passed for state in states):
        return None

    return WaveCandidate(
        pattern="ZigZag",
        direction="Bearish" if bearish else "Bullish",
        pivots=pivots,
        labels=("Start", "A", "B", "C"),
        rule_states=states,
        invalidation_level=float(pivots[0].price),
        invalidation_side="above" if bearish else "below",
    )


def evaluate_flat(
    pivots: list[Pivot] | tuple[Pivot, ...],
) -> WaveCandidate | None:
    """Validate and classify regular, expanded, or running A-B-C Flats."""
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
    c_relative_to_a = sign * (pivots[3].price - pivots[1].price)
    regular = 0.9 <= b_ratio <= 1.0 and c_relative_to_a >= 0
    expanded = 1.0 < b_ratio <= 1.382 and c_relative_to_a >= 0
    running = (
        1.0 < b_ratio <= 1.382
        and c_relative_to_a < 0
        and wave_c / wave_a >= 0.618
    )
    variant = (
        "Regular"
        if regular
        else "Expanded"
        if expanded
        else "Running"
        if running
        else None
    )
    states = (
        RuleState(
            "flat_b_minimum_retracement",
            0.9 <= b_ratio <= 1.382,
            "Wave B must retrace 90%-138.2% of Wave A",
        ),
        RuleState(
            "flat_subtype_geometry",
            variant is not None,
            "Wave B/C endpoints must match Regular, Expanded, or Running Flat geometry",
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
        variant=variant,
    )


def evaluate_triangle(
    pivots: list[Pivot] | tuple[Pivot, ...],
) -> WaveCandidate | None:
    """Validate a contracting or barrier A-B-C-D-E Triangle."""
    pivots = tuple(pivots)
    if len(pivots) != 6:
        raise ValueError(
            "a Triangle requires an origin and A, B, C, D, E endpoints"
        )
    _validate_pivots(pivots)
    if not _alternates(pivots):
        return None

    bearish = pivots[0].type == "High"
    prices = tuple(pivot.price for pivot in pivots)
    start, a, b, c, d, e = prices
    c_contained = min(start, a) < c < max(start, a)
    d_contained = min(a, b) < d <= max(a, b)
    widest = abs(b - a)
    tolerance = widest * 0.1
    e_contained = min(b, c) - tolerance <= e <= max(b, c) + tolerance

    if bearish:
        first_boundary_inward = c > a
        second_change = b - d
        e_boundary_direction = e >= c
    else:
        first_boundary_inward = c < a
        second_change = d - b
        e_boundary_direction = e <= c
    second_boundary_inward = second_change > tolerance
    second_boundary_flat = abs(second_change) <= tolerance
    later_width = abs(d - c)
    boundaries_converge = (
        first_boundary_inward
        and (second_boundary_inward or second_boundary_flat)
        and later_width < widest
    )
    variant = (
        "Contracting"
        if boundaries_converge and second_boundary_inward
        else "Barrier"
        if boundaries_converge and second_boundary_flat
        else None
    )

    a_time = pivots[1].timestamp.value
    c_time = pivots[3].timestamp.value
    e_time = pivots[5].timestamp.value
    projected_e = (
        a + (c - a) * (e_time - a_time) / (c_time - a_time)
        if c_time != a_time
        else c
    )
    e_near_boundary = (
        e_boundary_direction and abs(e - projected_e) <= widest * 0.25
    )
    states = (
        RuleState(
            "triangle_c_contained",
            c_contained,
            "Wave C must remain inside the Start-A range",
        ),
        RuleState(
            "triangle_d_contained",
            d_contained,
            "Wave D must remain inside the A-B range",
        ),
        RuleState(
            "triangle_e_contained",
            e_contained and e_near_boundary,
            "Wave E must remain near the A-C boundary within tolerance",
        ),
        RuleState(
            "triangle_boundaries_converge",
            variant is not None,
            "A-C and B-D boundaries must contract or form a barrier",
        ),
    )
    if not all(state.passed for state in states):
        return None

    return WaveCandidate(
        pattern="Triangle",
        direction="Bearish" if bearish else "Bullish",
        pivots=pivots,
        labels=("Start", "A", "B", "C", "D", "E"),
        rule_states=states,
        invalidation_level=float(pivots[0].price),
        invalidation_side="above" if bearish else "below",
        variant=variant,
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


def _candidate_paths(
    nodes: tuple[Pivot, ...],
    length: int,
    max_pivot_skip: int,
    minimum_atr_displacement: float,
) -> tuple[tuple[int, ...], ...]:
    """Enumerate bounded forward paths with exact source-node provenance."""
    paths: list[tuple[int, ...]] = []

    def extend(path: tuple[int, ...]) -> None:
        if len(path) == length:
            paths.append(path)
            return
        left_index = path[-1]
        upper = min(len(nodes), left_index + max_pivot_skip + 2)
        for right_index in range(left_index + 1, upper):
            left, right = nodes[left_index], nodes[right_index]
            displacement = abs(right.price - left.price)
            threshold = minimum_atr_displacement * max(left.atr, right.atr)
            if left.type == right.type or displacement < threshold:
                continue
            extend(path + (right_index,))

    for start in range(len(nodes)):
        extend((start,))
    return tuple(paths)
