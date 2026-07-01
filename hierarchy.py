"""Deterministic cross-timeframe wave-degree hierarchy."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import median
from typing import Mapping, Sequence

from db import TIMEFRAMES
from engine import WaveCandidate


@dataclass(frozen=True, slots=True)
class WaveHierarchyNode:
    candidate: WaveCandidate
    timeframe: str
    degree_rank: int
    parent_index: int | None
    child_indices: tuple[int, ...]


def build_wave_degree_hierarchy(
    candidates_by_timeframe: Mapping[str, Sequence[WaveCandidate]],
) -> tuple[WaveHierarchyNode, ...]:
    """Nest contained structures using timeframe and relative scale.

    Timeframe supplies the primary degree. Duration and displacement split
    structures within one timeframe into lower/base/upper scale buckets.
    Parent links always point to a higher-timeframe candidate that fully
    contains the child in time; the smallest qualifying parent is selected.
    """
    timeframe_order = {label: index for index, label in enumerate(TIMEFRAMES)}
    unknown = set(candidates_by_timeframe).difference(timeframe_order)
    if unknown:
        raise ValueError(f"unknown timeframes: {', '.join(sorted(unknown))}")

    records: list[tuple[WaveCandidate, str, int]] = []
    for timeframe, candidates in candidates_by_timeframe.items():
        if any(not isinstance(candidate, WaveCandidate) for candidate in candidates):
            raise TypeError("all hierarchy entries must be WaveCandidate objects")
        if not candidates:
            continue
        durations = [_duration(candidate) for candidate in candidates]
        displacements = [_displacement(candidate) for candidate in candidates]
        median_duration = median(durations)
        median_displacement = median(displacements)
        for candidate, duration, displacement in zip(
            candidates, durations, displacements
        ):
            duration_ratio = duration / median_duration if median_duration else 1
            displacement_ratio = (
                displacement / median_displacement if median_displacement else 1
            )
            scale = max(duration_ratio, displacement_ratio)
            bucket = -1 if scale < 0.5 else 1 if scale > 2.0 else 0
            records.append(
                (candidate, timeframe, timeframe_order[timeframe] * 3 + bucket)
            )

    records.sort(
        key=lambda item: (
            item[0].pivots[0].timestamp,
            item[0].pivots[-1].timestamp,
            timeframe_order[item[1]],
        )
    )
    parents: list[int | None] = [None] * len(records)
    for child_index, (child, child_tf, _rank) in enumerate(records):
        options: list[tuple[float, int, int]] = []
        for parent_index, (parent, parent_tf, parent_rank) in enumerate(records):
            if timeframe_order[parent_tf] <= timeframe_order[child_tf]:
                continue
            if (
                parent.pivots[0].timestamp <= child.pivots[0].timestamp
                and parent.pivots[-1].timestamp >= child.pivots[-1].timestamp
            ):
                options.append((_duration(parent), parent_rank, parent_index))
        if options:
            parents[child_index] = min(options)[2]

    children: list[list[int]] = [[] for _ in records]
    for child_index, parent_index in enumerate(parents):
        if parent_index is not None:
            children[parent_index].append(child_index)
    return tuple(
        WaveHierarchyNode(
            candidate=candidate,
            timeframe=timeframe,
            degree_rank=rank,
            parent_index=parents[index],
            child_indices=tuple(children[index]),
        )
        for index, (candidate, timeframe, rank) in enumerate(records)
    )


def _duration(candidate: WaveCandidate) -> float:
    return (
        candidate.pivots[-1].timestamp - candidate.pivots[0].timestamp
    ).total_seconds()


def _displacement(candidate: WaveCandidate) -> float:
    prices = [pivot.price for pivot in candidate.pivots]
    return max(prices) - min(prices)
