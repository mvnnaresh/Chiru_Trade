"""Append-only SQLite audit trail for signal lifecycle transitions."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from engine import WaveCandidate


@dataclass(frozen=True, slots=True)
class SignalEvent:
    event_id: int
    signal_key: str
    observed_at: pd.Timestamp
    stage: str
    confidence_score: float
    invalidation_level: float
    payload: str


_TRANSITIONS = {
    None: {"Forming", "EntryReady", "Completed", "Invalidated"},
    "Forming": {"Forming", "EntryReady", "Invalidated"},
    "EntryReady": {"Completed", "Invalidated"},
    "Completed": set(),
    "Invalidated": set(),
}


def setup_signal_audit(database: str | Path) -> None:
    with sqlite3.connect(database) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS signal_events (
                event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                signal_key TEXT NOT NULL,
                observed_at TEXT NOT NULL,
                stage TEXT NOT NULL,
                confidence_score REAL NOT NULL,
                invalidation_level REAL NOT NULL,
                payload TEXT NOT NULL
            )
            """
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS signal_events_key_time "
            "ON signal_events(signal_key, observed_at)"
        )


def append_signal_event(
    database: str | Path,
    candidate: WaveCandidate,
    confidence_score: float,
    *,
    observed_at: pd.Timestamp,
) -> SignalEvent:
    """Append one validated transition; historical events are never updated."""
    if not 0 <= confidence_score <= 100:
        raise ValueError("confidence_score must be between 0 and 100")
    observed = pd.Timestamp(observed_at)
    if observed.tzinfo is None:
        observed = observed.tz_localize("UTC")
    else:
        observed = observed.tz_convert("UTC")
    setup_signal_audit(database)
    key = signal_key(candidate)
    payload = json.dumps(
        {
            "pattern": candidate.pattern,
            "variant": candidate.variant,
            "direction": candidate.direction,
            "labels": candidate.labels,
            "pivots": [
                [pivot.timestamp.isoformat(), pivot.price, pivot.type]
                for pivot in candidate.pivots
            ],
        },
        separators=(",", ":"),
    )
    with sqlite3.connect(database) as connection:
        previous = connection.execute(
            "SELECT stage, observed_at FROM signal_events "
            "WHERE signal_key = ? ORDER BY event_id DESC LIMIT 1",
            (key,),
        ).fetchone()
        previous_stage = previous[0] if previous else None
        if candidate.status not in _TRANSITIONS[previous_stage]:
            raise ValueError(
                f"invalid signal transition {previous_stage} -> {candidate.status}"
            )
        if previous and observed <= pd.Timestamp(previous[1]):
            raise ValueError("signal events must be strictly chronological")
        cursor = connection.execute(
            "INSERT INTO signal_events "
            "(signal_key, observed_at, stage, confidence_score, "
            "invalidation_level, payload) VALUES (?, ?, ?, ?, ?, ?)",
            (
                key,
                observed.isoformat(),
                candidate.status,
                confidence_score,
                candidate.invalidation_level,
                payload,
            ),
        )
        event_id = int(cursor.lastrowid)
    return SignalEvent(
        event_id,
        key,
        observed,
        candidate.status,
        confidence_score,
        candidate.invalidation_level,
        payload,
    )


def load_signal_events(database: str | Path) -> tuple[SignalEvent, ...]:
    setup_signal_audit(database)
    with sqlite3.connect(database) as connection:
        rows = connection.execute(
            "SELECT event_id, signal_key, observed_at, stage, "
            "confidence_score, invalidation_level, payload "
            "FROM signal_events ORDER BY event_id"
        ).fetchall()
    return tuple(
        SignalEvent(
            int(row[0]),
            row[1],
            pd.Timestamp(row[2]),
            row[3],
            float(row[4]),
            float(row[5]),
            row[6],
        )
        for row in rows
    )


def signal_key(candidate: WaveCandidate) -> str:
    """Stable identity excludes mutable stage and active-leg revisions."""
    body = json.dumps(
        {
            "pattern": candidate.pattern,
            "direction": candidate.direction,
            "origin": (
                candidate.pivots[0].timestamp.isoformat(),
                candidate.pivots[0].price,
            ),
            "first_leg": (
                candidate.pivots[1].timestamp.isoformat(),
                candidate.pivots[1].price,
            ),
        },
        sort_keys=True,
    )
    return hashlib.sha256(body.encode("utf-8")).hexdigest()
