"""Streamlit dashboard for deterministic Elliott Wave decision support.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import html
import hashlib
import json
import logging
import os
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pandas as pd

from backtester import Friction, build_causal_rankings, run_backtest
from decision import DecisionState, build_decision_state
from db import TIMEFRAMES, load_m5, resample_m5, resample_ohlcv
from engine import (
    WaveCandidate,
    build_candidates,
    find_provisional_candidates,
    select_active_primary,
)
from live.instrument_resolver import DEFAULT_UPSTOX_UNIVERSE, resolve_instruments
from live.upstox_backfill import backfill_market
from live.token_loader import load_upstox_access_token
from market_data import append_latest_m5, resolve_market_symbol
from pivots import Pivot, PivotState, extract_pivot_state
from scoring import ConfidenceScore, calculate_rsi, score_candidates
from risk import RiskPolicy

LOGGER = logging.getLogger("elliott_dashboard")
COLORS = ("#00D4FF", "#FFB000", "#D65CFF")
SENSITIVITY_PRESETS: dict[str, tuple[float, int]] = {
    "Tight": (1.8, 10),
    "Balanced": (2.0, 14),
    "Conservative": (2.6, 21),
}
SIGNAL_STRICTNESS_THRESHOLDS: dict[str, float] = {
    "Aggressive": 65.0,
    "Balanced": 70.0,
    "Conservative": 75.0,
    "Very strict": 80.0,
}
LIVE_REFRESH_SECONDS: dict[str, int] = {"15s": 15, "30s": 30, "60s": 60}


@dataclass(frozen=True, slots=True)
class DashboardResult:
    candles: pd.DataFrame
    pivots: tuple[Pivot, ...]
    rankings: tuple[tuple[WaveCandidate, ConfidenceScore], ...]
    rsi: pd.Series
    pivot_state: PivotState | None = None


@dataclass(frozen=True, slots=True)
class ScannerRow:
    candidate_key: str
    pivot_signature: str
    database_name: str
    market: str
    timeframe: str
    pattern: str
    direction: str
    trade_decision: str
    structure_status: str
    setup_stage: str
    current_wave: str
    reason: str
    setup_quality_score: float
    risk_reward: float | str
    invalidation: float | None
    target_zone: str | None

    def as_record(self) -> dict[str, object]:
        return {
            "Trade Decision": self.trade_decision,
            "Structure Status": self.structure_status,
            "Direction": self.direction,
            "Reason": self.reason,
            "Setup Quality Score": self.setup_quality_score,
            "Risk/Reward": self.risk_reward,
            "Market": self.market,
            "Timeframe": self.timeframe,
            "Pattern": self.pattern,
            "Current Wave": self.current_wave,
            "Invalidation": self.invalidation,
            "Target Zone": self.target_zone,
            "Database Name": self.database_name,
            "Setup Stage": self.setup_stage,
            "Candidate Key": self.candidate_key,
            "Pivot Signature": self.pivot_signature,
        }


@dataclass(frozen=True, slots=True)
class RecommendationState:
    action: str
    color: str
    entry_text: str
    stop_text: str
    target_text: str
    status_text: str
    rationale: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class LiveRefreshState:
    checked_at: pd.Timestamp
    rows_added: int
    last_completed_bar: pd.Timestamp | None
    ok: bool
    message: str


def resolve_sensitivity(
    preset: str,
    *,
    override_enabled: bool = False,
    atr_multiplier: float | None = None,
    atr_period: int | None = None,
) -> tuple[float, int]:
    """Resolve trader-facing sensitivity into deterministic ATR settings."""
    if preset not in SENSITIVITY_PRESETS:
        raise ValueError(f"unknown sensitivity preset: {preset}")
    base_multiplier, base_period = SENSITIVITY_PRESETS[preset]
    if not override_enabled:
        return base_multiplier, base_period
    resolved_multiplier = (
        float(atr_multiplier) if atr_multiplier is not None else base_multiplier
    )
    resolved_period = int(atr_period) if atr_period is not None else base_period
    return resolved_multiplier, resolved_period


def resolve_setup_quality_threshold(
    strictness: str,
    *,
    override_enabled: bool = False,
    override_value: float | None = None,
) -> float:
    """Resolve the single setup-quality gate used throughout the application."""
    if strictness not in SIGNAL_STRICTNESS_THRESHOLDS:
        raise ValueError(f"unknown signal strictness: {strictness}")
    if not override_enabled:
        return SIGNAL_STRICTNESS_THRESHOLDS[strictness]
    value = (
        SIGNAL_STRICTNESS_THRESHOLDS[strictness]
        if override_value is None
        else float(override_value)
    )
    if not 50 <= value <= 90:
        raise ValueError("setup-quality threshold must be between 50 and 90")
    return value


def system_status(
    *,
    live_enabled: bool,
    live_supported: bool,
    live_refresh_ok: bool | None = None,
) -> tuple[str, str]:
    """Return a deterministic terminal status label and color."""
    if not live_enabled or not live_supported:
        return "LOCAL DATA MODE", "#F0B90B"
    if live_refresh_ok is False:
        return "LIVE REFRESH ERROR", "#F05D68"
    return "SYSTEM LIVE", "#18C98B"


def discover_databases(directory: str | Path = ".") -> tuple[Path, ...]:
    """Return local SQLite candidates in deterministic name order."""
    root = Path(directory)
    files = {
        path.resolve()
        for pattern in ("*.db", "*.sqlite", "*.sqlite3")
        for path in root.glob(pattern)
        if path.is_file()
    }
    return tuple(sorted(files, key=lambda path: path.name.lower()))


def refresh_live_database(
    database: str | Path,
    *,
    lookback_period: str = "5d",
) -> str:
    """Append the latest completed Yahoo M5 candles and summarize the refresh."""
    state = refresh_live_database_state(database, lookback_period=lookback_period)
    return state.message


def refresh_live_database_state(
    database: str | Path,
    *,
    lookback_period: str = "5d",
) -> LiveRefreshState:
    """Append latest completed Yahoo M5 candles and return auditable refresh state."""
    checked_at = pd.Timestamp.now(tz="UTC")
    inserted, last_completed = append_latest_m5(database, period=lookback_period)
    message = (
        f"Yahoo live | completed M5 through "
        f"{last_completed.strftime('%d %b %H:%M UTC')} | +{inserted} rows"
    )
    return LiveRefreshState(
        checked_at=checked_at,
        rows_added=int(inserted),
        last_completed_bar=pd.Timestamp(last_completed),
        ok=True,
        message=message,
    )


def _offset_timestamp(timestamp: pd.Timestamp, timeframe: str) -> pd.Timestamp:
    """Convert a left-labeled candle timestamp to its completed-through time."""
    spec = TIMEFRAMES[timeframe.upper()]
    return pd.Timestamp(timestamp) + pd.tseries.frequencies.to_offset(
        spec.pandas_rule
    )


def compute_dashboard(
    database: str | Path,
    timeframe: str,
    atr_multiplier: float,
    atr_period: int = 14,
) -> DashboardResult:
    """Load, extract, validate, and rank the current chart state."""
    candles = resample_ohlcv(database, timeframe.upper())  # type: ignore[arg-type]
    return compute_dashboard_from_candles(candles, atr_multiplier, atr_period)


def compute_dashboard_from_candles(
    candles: pd.DataFrame,
    atr_multiplier: float,
    atr_period: int = 14,
) -> DashboardResult:
    """Extract and rank structures from an already-loaded/resampled frame."""
    empty_rsi = pd.Series(index=candles.index, dtype=float, name="rsi")
    if candles.empty or len(candles) < atr_period:
        return DashboardResult(candles, (), (), empty_rsi)

    pivot_state = extract_pivot_state(
        candles, atr_multiplier, atr_period=atr_period
    )
    pivots = pivot_state.confirmed
    candidates = build_candidates(pivots)
    candidates.extend(find_provisional_candidates(pivot_state))
    scores = score_candidates(candidates, candles)
    rankings = tuple(
        sorted(scores.items(), key=lambda item: item[1].total, reverse=True)
    )
    return DashboardResult(
        candles=candles,
        pivots=pivots,
        rankings=rankings,
        rsi=calculate_rsi(candles),
        pivot_state=pivot_state,
    )


def target_zone(candidate: WaveCandidate) -> tuple[float, float]:
    """Return deterministic lower/upper Fibonacci target-zone bounds."""
    sign = 1.0 if candidate.direction == "Bullish" else -1.0
    first_leg = abs(candidate.pivots[1].price - candidate.pivots[0].price)
    if candidate.status == "Forming" and candidate.active_leg is not None:
        anchor = candidate.active_leg.price
        projections = (anchor + sign * first_leg, anchor + sign * 1.618 * first_leg)
    elif candidate.pattern == "Impulse":
        anchor = candidate.pivots[4].price
        projections = (anchor + sign * 1.0 * first_leg, anchor + sign * 1.618 * first_leg)
    else:
        anchor = candidate.pivots[2].price
        projections = (anchor + sign * 1.0 * first_leg, anchor + sign * 1.618 * first_leg)
    return min(projections), max(projections)


def _setup_endpoint(candidate: WaveCandidate) -> tuple[pd.Timestamp, float]:
    """Return the currently observable Wave 4/B setup endpoint."""
    if candidate.status == "Forming" and candidate.active_leg is not None:
        return candidate.active_leg.timestamp, candidate.active_leg.price
    setup_label = "4" if candidate.pattern == "Impulse" else "B"
    if setup_label in candidate.labels:
        index = candidate.labels.index(setup_label)
        pivot = candidate.pivots[index]
        return pivot.timestamp, pivot.price
    pivot = candidate.pivots[-1]
    return pivot.timestamp, pivot.price


def format_setup_alert(
    candidate: WaveCandidate,
    score: ConfidenceScore,
    threshold: float,
    *,
    chat_id: str,
) -> str | None:
    """Build an alert only for a newly completed Wave 4 or Wave B setup."""
    if score.total < threshold or not _is_tradeable_setup(candidate):
        return None
    low, high = target_zone(candidate)
    return (
        f"Elliott setup | chat={chat_id or 'unset'} | "
        f"{candidate.pattern} | {candidate.direction} | "
        f"Confidence Score={score.total:.2f}/100 | "
        f"Invalidation={candidate.invalidation_level:.5f} | "
        f"Target zone={low:.5f}-{high:.5f}"
    )


def log_alert_background(message: str) -> threading.Thread:
    """Log a webhook-ready alert on a daemon thread; performs no network I/O."""
    thread = threading.Thread(
        target=LOGGER.warning,
        args=(message,),
        name="telegram-alert-shell",
        daemon=True,
    )
    thread.start()
    return thread


def emit_alert_shell(
    candidate: WaveCandidate | None,
    score: ConfidenceScore | None,
    _bot_token: str = "",
    chat_id: str = "",
) -> bool:
    """Validate and write an eligible setup alert to application logs only."""
    import streamlit as st

    if candidate is None or score is None:
        st.warning("No candidate is selected for an alert.")
        return False
    threshold = float(
        st.session_state.get("resolved_setup_quality_threshold", 70.0)
    )
    if not _is_tradeable_setup(candidate):
        st.warning("The selected candidate is not a confirmed Wave 4/B setup.")
        return False
    if score.total < threshold:
        st.warning("The selected setup is below the alert setup-quality threshold.")
        return False
    message = format_setup_alert(candidate, score, threshold, chat_id=chat_id)
    if message is None:
        st.warning("The selected setup is not eligible for an alert.")
        return False
    log_alert_background(message)
    st.success("Setup written to application logs.")
    return True


def build_lightweight_charts(
    result: DashboardResult,
    overlays: tuple[bool, bool, bool],
    selected_index: int = 0,
    alert_threshold: float = 101.0,
) -> list[dict[str, object]]:
    """Build synchronized TradingView Lightweight Charts specifications."""
    candles = result.candles
    candle_data = [
        {
            "time": _chart_time(timestamp),
            "open": float(row.open),
            "high": float(row.high),
            "low": float(row.low),
            "close": float(row.close),
        }
        for timestamp, row in candles.iterrows()
    ]
    setup_markers = [
        {
            "time": _chart_time(candidate.pivots[-1].timestamp),
            "position": (
                "belowBar" if candidate.direction == "Bullish" else "aboveBar"
            ),
            "shape": (
                "arrowUp" if candidate.direction == "Bullish" else "arrowDown"
            ),
            "color": (
                "#18C98B" if candidate.direction == "Bullish" else "#F05D68"
            ),
            "text": f"#{rank} {candidate.pattern} ({score.total:.1f})",
        }
        for rank, ((candidate, score), enabled) in enumerate(
            zip(result.rankings[:3], overlays), start=1
        )
        if (
            enabled
            and _is_tradeable_setup(candidate)
            and score.total >= alert_threshold
        )
    ]
    price_series: list[dict[str, object]] = [
        {
            "type": "Candlestick",
            "data": candle_data,
            "options": {
                "upColor": "#18C98B",
                "downColor": "#F05D68",
                "borderUpColor": "#18C98B",
                "borderDownColor": "#F05D68",
                "wickUpColor": "#18C98B",
                "wickDownColor": "#F05D68",
                "priceLineVisible": True,
                "lastValueVisible": True,
            },
            "markers": setup_markers,
        }
    ]
    for rank, ((candidate, score), enabled, color) in enumerate(
        zip(result.rankings[:3], overlays, COLORS), start=1
    ):
        if not enabled:
            continue
        wave_data = [
            {"time": _chart_time(pivot.timestamp), "value": pivot.price}
            for pivot in candidate.pivots
        ]
        if candidate.status == "Forming" and candidate.active_leg is not None:
            wave_data.append(
                {
                    "time": _chart_time(candidate.active_leg.timestamp),
                    "value": candidate.active_leg.price,
                }
            )
        wave_markers = [
            {
                "time": _chart_time(pivot.timestamp),
                "position": "aboveBar" if pivot.type == "High" else "belowBar",
                "color": color,
                "shape": "circle",
                "text": label,
            }
            for label, pivot in candidate.labeled_waves
        ]
        if candidate.status == "Forming" and candidate.active_leg is not None:
            wave_markers.append(
                {
                    "time": _chart_time(candidate.active_leg.timestamp),
                    "position": (
                        "aboveBar"
                        if candidate.active_leg.type == "High"
                        else "belowBar"
                    ),
                    "color": color,
                    "shape": "circle",
                    "text": f"{candidate.forming_label}?",
                }
            )
        price_series.append(
            {
                "type": "Line",
                "data": wave_data,
                "options": {
                    "color": color,
                    "lineWidth": 3 if rank - 1 == selected_index else 1,
                    "lineStyle": 2 if candidate.status == "Forming" else 0,
                    "priceLineVisible": False,
                    "lastValueVisible": False,
                    "title": _series_title(rank, candidate),
                },
                "markers": wave_markers if rank - 1 == selected_index else [],
            }
        )

    if result.rankings:
        selected_index = min(selected_index, len(result.rankings) - 1)
        selected_candidate = result.rankings[selected_index][0]
        price_series.extend(_reference_line_specs(selected_candidate, candles))

    rsi_data = [
        {"time": _chart_time(timestamp), "value": float(value)}
        for timestamp, value in result.rsi.dropna().items()
    ]
    boundary_times = (
        [_chart_time(candles.index[0]), _chart_time(candles.index[-1])]
        if not candles.empty
        else []
    )
    rsi_series: list[dict[str, object]] = [
        {
            "type": "Line",
            "data": rsi_data,
            "options": {
                "color": "#7D8CFF",
                "lineWidth": 2,
                "priceLineVisible": False,
                "lastValueVisible": True,
                "title": "RSI (14)",
            },
        }
    ]
    for level in (30.0, 70.0):
        rsi_series.append(
            {
                "type": "Line",
                "data": [
                    {"time": time, "value": level} for time in boundary_times
                ],
                "options": {
                    "color": "rgba(120, 126, 140, 0.75)",
                    "lineWidth": 1,
                    "lineStyle": 2,
                    "priceLineVisible": False,
                    "lastValueVisible": False,
                },
            }
        )

    return [
        {
            "chart": _chart_options(height=450, show_time_axis=False),
            "series": price_series,
        },
        {
            "chart": _chart_options(height=140, show_time_axis=True, rsi=True),
            "series": rsi_series,
        },
    ]


def marker_status(
    rankings: tuple[tuple[WaveCandidate, ConfidenceScore], ...],
    overlays: tuple[bool, bool, bool],
    alert_threshold: float,
) -> tuple[str, str, str]:
    """Return threshold, buy, and sell display values for visible setups."""
    enabled = [
        (candidate, score)
        for (candidate, score), visible in zip(rankings[:3], overlays)
        if visible
    ]
    tradeable = [
        (candidate, score)
        for candidate, score in enabled
        if _is_tradeable_setup(candidate)
    ]
    threshold_text = f"{alert_threshold:.1f}"
    if not enabled:
        return threshold_text, "Overlay disabled", "Overlay disabled"
    if not tradeable:
        candidate = enabled[0][0]
        label = (
            f"Forming Wave {candidate.forming_label}"
            if candidate.status == "Forming"
            else "Awaiting Wave 4/B"
        )
        if candidate.direction == "Bullish":
            return threshold_text, label, "No bearish setup"
        return threshold_text, "No bullish setup", label

    candidate, score = tradeable[0]
    if score.total < alert_threshold:
        message = f"Below threshold ({score.total:.1f})"
        if candidate.direction == "Bullish":
            return threshold_text, message, "No bearish setup"
        return threshold_text, "No bullish setup", message

    entry = f"{candidate.pivots[-1].price:,.2f}"
    if candidate.direction == "Bullish":
        return threshold_text, entry, "No bearish setup"
    return threshold_text, "No bullish setup", entry


def trader_recommendation(
    candidate: WaveCandidate | None,
    score: ConfidenceScore | None,
    alert_threshold: float,
    lifecycle: str | None,
) -> RecommendationState:
    """Return a trader-facing recommendation derived from existing deterministic gates."""
    if candidate is None or score is None or lifecycle is None:
        return RecommendationState(
            action="NO TRADE",
            color="#F0B90B",
            entry_text="n/a",
            stop_text="n/a",
            target_text="n/a",
            status_text="No active candidate selected",
            rationale=(
                "No candidate is currently selected.",
                "Use the inspector to choose a live structure.",
            ),
        )

    target_low, target_high = target_zone(candidate)
    stop_text = f"{candidate.invalidation_level:,.2f}"
    target_text = f"{target_low:,.2f} - {target_high:,.2f}"

    if candidate.status == "Forming":
        return RecommendationState(
            action="WATCH",
            color="#F0B90B",
            entry_text="Await confirmation",
            stop_text=stop_text,
            target_text=target_text,
            status_text=f"Forming Wave {candidate.forming_label}",
            rationale=(
                "The terminal pivot is still forming.",
                "Wait for the Wave 4/B endpoint to confirm.",
            ),
        )
    if lifecycle != "Active":
        return RecommendationState(
            action="NO TRADE",
            color="#F05D68",
            entry_text="n/a",
            stop_text=stop_text,
            target_text=target_text,
            status_text=f"Structure lifecycle is {lifecycle.lower()}",
            rationale=(
                "The selected structure is no longer live.",
                "Wait for a new active Wave 4/B setup.",
            ),
        )
    if not _is_tradeable_setup(candidate):
        required = "Wave 4" if candidate.pattern == "Impulse" else "Wave B"
        return RecommendationState(
            action="NO TRADE",
            color="#F05D68",
            entry_text="n/a",
            stop_text=stop_text,
            target_text=target_text,
            status_text=f"Awaiting {required}",
            rationale=(
                "This structure is valid but not at the tradeable terminal wave.",
                f"Wait for a fresh {required} setup.",
            ),
        )
    if score.total < alert_threshold:
        return RecommendationState(
            action="WATCH",
            color="#F0B90B",
            entry_text=f"{candidate.pivots[-1].price:,.2f}",
            stop_text=stop_text,
            target_text=target_text,
            status_text=f"Below confidence threshold ({score.total:.1f} < {alert_threshold:.1f})",
            rationale=(
                "The structure is tradeable but not ranked high enough.",
                "Wait for stronger confirmation before acting.",
            ),
        )
    side = "BUY" if candidate.direction == "Bullish" else "SELL"
    return RecommendationState(
        action=side,
        color="#18C98B" if side == "BUY" else "#F05D68",
        entry_text=f"{candidate.pivots[-1].price:,.2f}",
        stop_text=stop_text,
        target_text=target_text,
        status_text=f"{candidate.pattern} | {current_wave_label(candidate)}",
        rationale=(
            "Terminal Wave 4/B is present and active.",
            "Setup quality gate is passed.",
        ),
    )


def decision_panel_markup(decision: DecisionState) -> str:
    """Return the decision-first panel without coupling decision logic to Streamlit."""
    def price(value: float | None) -> str:
        return "n/a" if value is None else f"{value:,.2f}"

    provisional = decision.status == "WATCH"
    target_1_label = "Provisional target 1" if provisional else "Target 1"
    target_2_label = "Provisional target 2" if provisional else "Target 2"
    if decision.status == "WATCH" and decision.stage == "Forming":
        action = "Do not trade yet. Wait for Wave 4/B confirmation."
    elif decision.status == "WATCH":
        action = "Do not trade yet. Wait for stronger confirmation."
    elif decision.status == "NO TRADE":
        action = "Do not enter from this setup. Wait for a new Wave 4/B setup."
    elif decision.status == "BLOCKED":
        action = "Do not trade under the current risk policy."
    else:
        action = "Review manually before placing any order."
    structure_status = (
        "WATCHLIST SETUP"
        if decision.stage == "Forming"
        else f"{'BUY' if decision.direction == 'Bullish' else 'SELL'} SETUP"
        if decision.direction in {"Bullish", "Bearish"}
        and decision.stage == "EntryReady"
        else "NO ACTIVE SETUP"
    )
    trade_direction = (
        decision.direction.upper()
        if decision.stage == "Forming" and decision.direction in {"Bullish", "Bearish"}
        else "BUY" if decision.direction == "Bullish"
        else "SELL" if decision.direction == "Bearish"
        else "n/a"
    )

    fields = (
        ("Structure status", structure_status),
        ("Trade decision", decision.status),
        ("Direction", trade_direction),
        ("Stage", decision.stage or "n/a"),
        ("Lifecycle", decision.lifecycle or "n/a"),
        ("Current price", price(decision.current_price)),
        ("Entry reference", price(decision.entry_reference)),
        ("Setup pivot reference", price(decision.setup_reference)),
        ("Stop / invalidation", price(decision.stop)),
        (target_1_label, price(decision.target_1)),
        (target_2_label, price(decision.target_2)),
        ("Reward/risk", "n/a" if decision.reward_risk is None else f"{decision.reward_risk:.2f}"),
        ("Position size", "n/a" if decision.units is None else f"{decision.units:,} units"),
        ("Total risk", price(decision.total_risk)),
        ("Setup Quality Score", "n/a" if decision.setup_quality_score is None else f"{decision.setup_quality_score:.1f} / 100"),
        ("Required Threshold", "n/a" if decision.required_threshold is None else f"{decision.required_threshold:.1f}"),
        ("Quality Gate Result", decision.quality_gate_result),
        ("Risk gate result", decision.risk_reason),
    )
    cells = "".join(
        "<div><div class='terminal-kicker'>"
        f"{html.escape(label)}</div><div class='terminal-value'>{html.escape(value)}</div></div>"
        for label, value in fields
    )
    return (
        "<div class='terminal-panel' style='border-width:2px;margin-bottom:.5rem'>"
        "<div class='terminal-kicker'>Decision status</div>"
        f"<div style='font-size:1.55rem;font-weight:800;color:{decision.color}'>{html.escape(decision.status)}</div>"
        f"<div style='color:#F4F7FA;margin:.3rem 0'>{html.escape(decision.reason)}</div>"
        f"<div style='color:#8c96a5;font-size:.8rem'>{html.escape(decision.second_reason)}</div>"
        f"<div style='color:#F4F7FA;font-weight:700;margin-top:.5rem'>Action: {html.escape(action)}</div>"
        "<div style='display:grid;grid-template-columns:repeat(4,minmax(120px,1fr));gap:.7rem;margin-top:.8rem'>"
        f"{cells}</div></div>"
    )


def pattern_state_text(candidate: WaveCandidate) -> str:
    """Return a trader-facing summary of the structure's current state."""
    if candidate.status == "Forming" and candidate.forming_label is not None:
        return (
            f"{candidate.direction} {candidate.pattern} is currently "
            f"forming Wave {candidate.forming_label}"
        )
    terminal = candidate.labels[-1] if candidate.labels else "unknown"
    return f"{candidate.direction} {candidate.pattern} through Wave {terminal}"


def system_hints(
    candidate: WaveCandidate,
    score: ConfidenceScore,
    alert_threshold: float,
    lifecycle: str,
    decision_state: DecisionState | None = None,
) -> tuple[str, ...]:
    """Explain every gate controlling an actionable chart marker."""
    terminal = candidate.labels[-1] if candidate.labels else "unknown"
    target_low, target_high = target_zone(candidate)
    invalidation_kind = (
        "floor" if candidate.invalidation_side == "below" else "ceiling"
    )
    confidence = (
        f"Setup quality gate: Passed ({score.total:.1f} >= {alert_threshold:.1f})"
        if score.total >= alert_threshold
        else f"Setup quality gate: Below threshold "
        f"({score.total:.1f} < {alert_threshold:.1f})"
    )
    hints = [
        f"Pattern state: {pattern_state_text(candidate)}",
        f"Current wave: {current_wave_label(candidate)}",
        confidence,
        f"Lifecycle: {lifecycle}",
    ]
    wave_three = wave_three_window(candidate)
    if wave_three is not None:
        hints.extend(
            (
                _format_wave_point(wave_three[0]),
                _format_wave_point(wave_three[1]),
            )
        )
    if candidate.status == "Forming":
        hints.extend(
            (
                "Entry gate: Pending - Wave 4/B endpoint is not confirmed",
                "Risk gate: Not evaluated because setup is not confirmed",
                "Marker decision: Hidden - forming structures are watch-only",
                f"Next required event: Confirm the provisional Wave "
                f"{candidate.forming_label} pivot",
                "Trading interpretation: Watchlist setup; "
                "not a current trade signal",
            )
        )
    elif lifecycle != "Active":
        hints.extend(
            (
                f"Entry gate: Failed - lifecycle is {lifecycle.lower()}",
                f"Marker decision: Hidden - setup {lifecycle.lower()}",
                "Next required event: Wait for a new active structure "
                "terminating at Wave 4 or Wave B",
                "Trading interpretation: Historical structure; "
                "not a current trade signal",
            )
        )
    elif not _is_tradeable_setup(candidate):
        required = "4" if candidate.pattern == "Impulse" else "B"
        hints.extend(
            (
                f"Entry gate: Failed - candidate terminates at Wave "
                f"{terminal}, not Wave {required}",
                "Marker decision: Hidden - no entry setup",
                f"Next required event: Wait for a new active Wave {required} "
                "termination",
                "Trading interpretation: Valid analytical structure; "
                "not a current trade signal",
            )
        )
    elif score.total < alert_threshold:
        hints.extend(
            (
                "Entry gate: Passed - terminal Wave 4/B is present",
                "Risk gate: Not evaluated below the setup quality threshold",
                "Marker decision: Hidden - setup quality gate not met",
                f"Next required event: Setup quality must reach "
                f"{alert_threshold:.1f}",
                "Trading interpretation: Structurally actionable but "
                "insufficiently ranked",
            )
        )
    else:
        side = "BUY" if candidate.direction == "Bullish" else "SELL"
        setup_reference = _setup_endpoint(candidate)[1]
        if decision_state is not None and decision_state.status == "BLOCKED":
            hints.extend(
                (
                    "Entry gate: Passed - terminal Wave 4/B is present",
                    f"Risk gate: Failed - {decision_state.risk_reason}",
                    f"Marker decision: {side} setup detected, but trade blocked by risk policy",
                    f"{side} setup pivot: {setup_reference:,.2f}",
                    "Next required event: Risk policy must approve the setup",
                    "Trading interpretation: Valid structure, not tradable under current risk settings",
                )
            )
        elif decision_state is not None and decision_state.status == "TRADE READY":
            hints.extend(
                (
                    "Entry gate: Passed - terminal Wave 4/B is present",
                    "Risk gate: Passed",
                    f"Marker decision: {side} setup ready",
                    f"{side} setup pivot: {setup_reference:,.2f}",
                    "Next required event: Review entry, target, and invalidation",
                    f"Trading interpretation: Trade-ready {side.lower()} setup under current risk policy",
                )
            )
        else:
            hints.extend(
                (
                    "Entry gate: Passed - terminal Wave 4/B is present",
                    "Risk gate: Not evaluated",
                    f"Marker decision: {side} setup present",
                    f"{side} setup pivot: {setup_reference:,.2f}",
                    "Trading interpretation: Structure ready; trade decision not evaluated",
                )
            )
    hints.extend(
        (
            f"Invalidation reference: {candidate.invalidation_level:,.2f} "
            f"{invalidation_kind}",
            f"Target zone: {target_low:,.2f} - {target_high:,.2f}",
        )
    )
    return tuple(hints)

def _format_hint_value(value: str) -> str:
    """Color only explicit decision terms; keep supporting detail white."""
    formatted = html.escape(value)
    negative = (
        "not a current trade signal",
        "insufficiently ranked",
        "Below threshold",
        "Invalidated",
        "Failed",
        "Hidden",
    )
    positive = ("Actionable", "actionable", "Passed", "Visible")
    for term in negative:
        formatted = formatted.replace(
            term,
            f"<strong style='color:#F05D68'>{term}</strong>",
        )
    for term in positive:
        formatted = formatted.replace(
            term,
            f"<strong style='color:#18C98B'>{term}</strong>",
        )
    return formatted


def _chart_options(
    *, height: int, show_time_axis: bool, rsi: bool = False
) -> dict[str, object]:
    options: dict[str, object] = {
        "height": height,
        "layout": {
            "background": {"type": "solid", "color": "#0E1117"},
            "textColor": "#B2B5BE",
            "fontFamily": "Inter, system-ui, sans-serif",
        },
        "grid": {
            "vertLines": {"color": "rgba(42, 46, 57, 0.35)"},
            "horzLines": {"color": "rgba(42, 46, 57, 0.55)"},
        },
        "crosshair": {
            "mode": 0,
            "vertLine": {"color": "#758696", "style": 2, "width": 1},
            "horzLine": {"color": "#758696", "style": 2, "width": 1},
        },
        "rightPriceScale": {
            "borderColor": "#2A2E39",
            "scaleMargins": {"top": 0.08, "bottom": 0.08},
        },
        "timeScale": {
            "visible": show_time_axis,
            "borderColor": "#2A2E39",
            "timeVisible": True,
            "secondsVisible": False,
            "rightOffset": 4,
            "barSpacing": 9,
            "fixLeftEdge": False,
            "lockVisibleTimeRangeOnResize": True,
        },
        "handleScroll": {
            "mouseWheel": True,
            "pressedMouseMove": True,
            "horzTouchDrag": True,
            "vertTouchDrag": True,
        },
        "handleScale": {
            "axisPressedMouseMove": True,
            "mouseWheel": True,
            "pinch": True,
        },
    }
    if rsi:
        options["rightPriceScale"] = {
            "borderColor": "#2A2E39",
            "autoScale": False,
            "scaleMargins": {"top": 0.08, "bottom": 0.08},
        }
    return options


def _chart_time(timestamp: pd.Timestamp) -> int:
    """Convert any timezone-aware candle time to UTC epoch seconds."""
    value = pd.Timestamp(timestamp)
    if value.tzinfo is None:
        value = value.tz_localize("UTC")
    else:
        value = value.tz_convert("UTC")
    return int(value.timestamp())


def _candidate_name(index: int, candidate: WaveCandidate, score: ConfidenceScore) -> str:
    return (
        f"#{index + 1} | {candidate.pattern} | {candidate.status} | "
        f"{candidate.direction} | "
        f"{score.total:.1f}/100"
    )


def _candidate_query_key(candidate: WaveCandidate) -> str:
    """Return a stable query-string key for the currently selected candidate."""
    encoded = json.dumps(
        candidate_signature(candidate), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def scanner_candidate_key(
    database_name: str, timeframe: str, candidate: WaveCandidate
) -> str:
    """Return a stable scanner identity that cannot collide across markets/frames."""
    payload = {
        "database_name": database_name,
        "timeframe": timeframe.upper(),
        **candidate_signature(candidate),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def candidate_signature(candidate: WaveCandidate) -> dict[str, object]:
    """Return stable, refresh-safe structural identity for a candidate."""
    setup_time, setup_price = _setup_endpoint(candidate)
    return {
        "pattern": candidate.pattern,
        "direction": candidate.direction,
        "status": candidate.status,
        "labels": list(candidate.labels),
        "pivots": [
            [
                pivot.timestamp.isoformat(),
                round(float(pivot.price), 8),
                pivot.type,
            ]
            for pivot in candidate.pivots
        ],
        "setup_pivot_timestamp": setup_time.isoformat(),
        "setup_pivot_price": round(float(setup_price), 8),
    }


def scanner_setup_context(row: pd.Series | dict[str, object]) -> dict[str, object]:
    """Convert one scanner row into the exact Single Chart navigation context."""
    return {
        "market": row["Market"],
        "database": row["Database Name"],
        "timeframe": row["Timeframe"],
        "pattern": row["Pattern"],
        "direction": row["Direction"],
        "setup_stage": row["Setup Stage"],
        "current_wave": row["Current Wave"],
        "candidate_key": row["Candidate Key"],
        "pivot_signature": row["Pivot Signature"],
        "trade_decision": row["Trade Decision"],
        "structure_status": row.get("Structure Status", ""),
        "setup_quality_score": float(row["Setup Quality Score"]),
        "risk_reward": row.get("Risk/Reward"),
    }


def store_scanner_setup(
    state: dict[str, object], row: pd.Series | dict[str, object]
) -> dict[str, object]:
    """Store scanner navigation state without coupling it to Streamlit."""
    context = scanner_setup_context(row)
    state["selected_scanner_setup"] = context
    # Streamlit forbids mutating a widget's key after that widget is
    # instantiated in the current run. Apply this request before the tab
    # selector is created on the next rerun.
    state["requested_active_tab"] = "single_chart"
    return context


def scanner_inspect_button_key(
    row: pd.Series | dict[str, object], row_index: int
) -> str:
    """Build a unique Inspect widget key from stable setup identity."""
    return (
        f"inspect_{row['Candidate Key']}_{row['Market']}_"
        f"{row['Timeframe']}_{row_index}"
    )


def save_scanner_results(
    state: dict[str, object],
    frame: pd.DataFrame,
    errors: tuple[str, ...],
    settings: dict[str, object],
    timestamp: pd.Timestamp,
) -> None:
    """Persist one complete scan so tab changes cannot discard it."""
    state["last_scanner_results"] = frame
    state["last_scanner_summary"] = (
        frame["Trade Decision"].value_counts().to_dict() if not frame.empty else {}
    )
    state["last_scanner_settings"] = settings
    state["last_scan_timestamp"] = timestamp.isoformat()
    state["scanner_errors"] = errors
    state["scanner_frame"] = frame


def scanner_cache_is_compatible(
    previous: dict[str, object] | None, current: dict[str, object]
) -> bool:
    """Return whether cached decisions were made with the current settings."""
    return previous is not None and previous == current


def scanner_selector_defaults(context: dict[str, object]) -> dict[str, str]:
    """Translate scanner context into Single Chart widget values."""
    pattern = str(context.get("pattern", ""))
    return {
        "database": str(context.get("database", "")),
        "timeframe": str(context.get("timeframe", "1H")),
        "pattern_view": (
            "Impulse | 1-5"
            if pattern == "Impulse"
            else "ZigZag | ABC"
            if pattern == "ZigZag"
            else "Balanced | 1-5 + ABC"
        ),
        "candidate_scope": "All history",
    }


def match_scanner_candidate(
    rankings: tuple[tuple[WaveCandidate, ConfidenceScore], ...],
    context: dict[str, object],
) -> tuple[int, bool]:
    """Find the exact scanner candidate or the nearest explicitly ranked match."""
    if not rankings:
        return 0, False
    key = str(context.get("candidate_key", ""))
    database_name = str(context.get("database", ""))
    timeframe = str(context.get("timeframe", ""))
    for index, (candidate, _score) in enumerate(rankings):
        if key in {
            scanner_candidate_key(database_name, timeframe, candidate),
            _candidate_query_key(candidate),
        }:
            return index, True
    try:
        signature = json.loads(str(context.get("pivot_signature", "{}")))
    except ValueError:
        signature = {}
    expected_time = pd.Timestamp(signature.get("setup_pivot_timestamp"))
    expected_price = float(signature.get("setup_pivot_price", 0.0))
    expected_direction = str(context.get("direction", "")).title()
    if expected_direction == "Buy":
        expected_direction = "Bullish"
    elif expected_direction == "Sell":
        expected_direction = "Bearish"

    def proximity(item: tuple[WaveCandidate, ConfidenceScore]) -> tuple[object, ...]:
        candidate = item[0]
        setup_time, setup_price = _setup_endpoint(candidate)
        time_distance = (
            abs((setup_time - expected_time).total_seconds())
            if not pd.isna(expected_time)
            else float("inf")
        )
        return (
            candidate.pattern != context.get("pattern"),
            candidate.direction != expected_direction,
            candidate.status != context.get("setup_stage"),
            current_wave_label(candidate) != context.get("current_wave"),
            time_distance,
            abs(float(setup_price) - expected_price),
        )

    nearest = min(range(len(rankings)), key=lambda index: proximity(rankings[index]))
    return nearest, False


def _resolve_selected_index(
    rankings: tuple[tuple[WaveCandidate, ConfidenceScore], ...],
    selected_key: str | None,
) -> int:
    if not rankings:
        return 0
    if not selected_key:
        primary = select_active_primary(tuple(candidate for candidate, _ in rankings))
        if primary is None:
            return 0
        for index, (candidate, _score) in enumerate(rankings):
            if candidate == primary:
                return index
        return 0
    for index, (candidate, _score) in enumerate(rankings):
        if _candidate_query_key(candidate) == selected_key:
            return index
    primary = select_active_primary(tuple(candidate for candidate, _ in rankings))
    if primary is None:
        return 0
    for index, (candidate, _score) in enumerate(rankings):
        if candidate == primary:
            return index
    return 0


def current_wave_label(candidate: WaveCandidate) -> str:
    """Return the current structural wave/stage in trader-facing terms."""
    if candidate.status == "Forming" and candidate.forming_label is not None:
        return f"Forming Wave {candidate.forming_label}"
    if candidate.status == "EntryReady":
        terminal = candidate.labels[-1] if candidate.labels else "?"
        return f"EntryReady at Wave {terminal}"
    terminal = candidate.labels[-1] if candidate.labels else "?"
    return f"Completed through Wave {terminal}"


def wave_three_window(
    candidate: WaveCandidate,
) -> tuple[tuple[str, Pivot], tuple[str, Pivot]] | None:
    """Return exact Wave 3 start/end pivots for impulse structures."""
    if candidate.pattern != "Impulse":
        return None
    span = candidate.wave_span("2", "3")
    if span is None:
        return None
    start, end = span
    return (("Wave 3 start", start), ("Wave 3 end", end))


def _format_wave_point(point: tuple[str, Pivot] | None) -> str:
    if point is None:
        return "n/a"
    label, pivot = point
    timestamp = pivot.timestamp.tz_convert("UTC").strftime("%d %b %Y %H:%M UTC")
    return f"{label}: {timestamp} @ {pivot.price:,.2f}"


def _series_title(rank: int, candidate: WaveCandidate) -> str:
    return f"#{rank} {candidate.pattern} · {candidate.status}"


def _reference_line_specs(
    candidate: WaveCandidate, candles: pd.DataFrame
) -> tuple[dict[str, object], ...]:
    """Build consistent invalidation/target line metadata."""
    setup_timestamp, _setup_price = _setup_endpoint(candidate)
    first_time = _chart_time(setup_timestamp)
    last_time = _chart_time(candles.index[-1])
    target_low, target_high = target_zone(candidate)
    specs = (
        (candidate.invalidation_level, "#F05D68", "Invalidation"),
        (target_low, "#18C98B", "Target 1.000"),
        (target_high, "#00BFA5", "Target 1.618"),
    )
    return tuple(
        {
            "type": "Line",
            "data": [
                {"time": first_time, "value": price},
                {"time": last_time, "value": price},
            ],
            "options": {
                "color": color,
                "lineWidth": 1,
                "lineStyle": 2,
                "priceLineVisible": False,
                "lastValueVisible": True,
                "title": title,
            },
        }
        for price, color, title in specs
    )


def _legend_markup(
    rankings: tuple[tuple[WaveCandidate, ConfidenceScore], ...],
    overlays: tuple[bool, bool, bool],
    selected_candidate: WaveCandidate | None,
) -> str:
    """Render overlay and reference legends from the same metadata as the chart."""
    items = [
        "<span><b style='color:#F4F7FA'>LEGEND</b></span>",
        "<span><i style='display:inline-block;width:22px;border-top:"
        "2px solid #00D4FF;margin-right:5px'></i>Completed</span>",
        "<span><i style='display:inline-block;width:22px;border-top:"
        "2px dashed #FFB000;margin-right:5px'></i>Forming</span>",
        "<span><b style='color:#18C98B;margin-right:5px'>BUY</b>"
        " Marker visible</span>",
        "<span><b style='color:#F05D68;margin-right:5px'>SELL</b>"
        " Marker visible</span>",
    ]
    for rank, ((candidate, score), enabled, color) in enumerate(
        zip(rankings[:3], overlays, COLORS), start=1
    ):
        if not enabled:
            continue
        items.append(
            "<span><i style='display:inline-block;width:22px;border-top:"
            f"2px solid {color};margin-right:5px'></i>"
            f"{html.escape(_series_title(rank, candidate))} ({score.total:.1f})"
            "</span>"
        )
    if selected_candidate is not None:
        for title, color in (
            ("Invalidation", "#F05D68"),
            ("Target 1.000", "#18C98B"),
            ("Target 1.618", "#00BFA5"),
        ):
            items.append(
                "<span><i style='display:inline-block;width:22px;border-top:"
                f"2px dashed {color};margin-right:5px'></i>{title}</span>"
            )
    return (
        "<div style='display:flex;gap:1.1rem;align-items:center;flex-wrap:wrap;"
        "font-size:.68rem;color:#AAB2BF;margin:.05rem 0 .55rem'>"
        + "".join(items)
        + "</div>"
    )


def recent_rankings(
    rankings: tuple[tuple[WaveCandidate, ConfidenceScore], ...],
    latest_timestamp: pd.Timestamp,
    days: int = 30,
) -> tuple[tuple[WaveCandidate, ConfidenceScore], ...]:
    """Limit the working set to paths ending near the current market edge."""
    cutoff = pd.Timestamp(latest_timestamp) - pd.Timedelta(days=days)
    return tuple(
        item
        for item in rankings
        if _setup_endpoint(item[0])[0] >= cutoff
    )


def candidate_lifecycle(
    candidate: WaveCandidate, candles: pd.DataFrame
) -> str:
    """Classify a path using only candles after its causal detection point."""
    if candidate.status == "Forming":
        return "Forming"
    observable_at = tradeable_signal_time(candidate)
    future = candles.loc[candles.index > observable_at]
    if future.empty:
        return "Active"
    target_low, target_high = target_zone(candidate)
    bullish = candidate.direction == "Bullish"
    for _timestamp, candle in future.iterrows():
        invalidated = (
            float(candle["low"]) <= candidate.invalidation_level
            if candidate.invalidation_side == "below"
            else float(candle["high"]) >= candidate.invalidation_level
        )
        target_hit = (
            float(candle["high"]) >= target_low
            if bullish
            else float(candle["low"]) <= target_high
        )
        if invalidated:
            return "Invalidated"
        if target_hit:
            return "Target hit"
    return "Active"


def tradeable_signal_time(candidate: WaveCandidate) -> pd.Timestamp:
    """Return the pivot time from which a tradeable setup must be monitored."""
    if candidate.status == "EntryReady":
        for label in ("4", "B"):
            pivot = candidate.pivot_for_label(label)
            if pivot is not None:
                return pivot.timestamp
    if candidate.status == "Forming" and candidate.active_leg is not None:
        return candidate.active_leg.timestamp
    return candidate.pivots[-1].timestamp


def actionable_rankings(
    rankings: tuple[tuple[WaveCandidate, ConfidenceScore], ...],
    candles: pd.DataFrame,
) -> tuple[tuple[WaveCandidate, ConfidenceScore], ...]:
    active = [
        item
        for item in rankings
        if candidate_lifecycle(item[0], candles) in {"Active", "Forming"}
    ]
    stage_priority = {"EntryReady": 0, "Forming": 1, "Completed": 2}
    return tuple(
        sorted(
            active,
            key=lambda item: (
                stage_priority.get(item[0].status, 3),
                -item[1].total,
            ),
        )
    )


def live_rankings(
    rankings: tuple[tuple[WaveCandidate, ConfidenceScore], ...],
) -> tuple[tuple[WaveCandidate, ConfidenceScore], ...]:
    """Order candidates for a live terminal instead of pure score history."""
    def observable_time(candidate: WaveCandidate) -> pd.Timestamp:
        if candidate.status == "Forming" and candidate.active_leg is not None:
            return candidate.active_leg.timestamp
        if candidate.as_of is not None:
            return candidate.as_of
        return candidate.pivots[-1].timestamp

    status_priority = {
        "EntryReady": 0,
        "Forming": 1,
        "Completed": 2,
        "Invalidated": 3,
    }
    pattern_priority = {
        "Impulse": 0,
        "ZigZag": 1,
        "Flat": 2,
        "Triangle": 3,
    }
    return tuple(
        sorted(
            rankings,
            key=lambda item: (
                -observable_time(item[0]).value,
                status_priority.get(item[0].status, 9),
                pattern_priority.get(item[0].pattern, 9),
                -item[1].total,
            ),
        )
    )


def focus_dashboard(
    result: DashboardResult,
    candidate: WaveCandidate,
    padding_bars: int = 18,
) -> DashboardResult:
    """Return a chart-only window centered on the selected structure."""
    index = result.candles.index
    start_position = int(index.searchsorted(candidate.pivots[0].timestamp))
    endpoint_time, _endpoint_price = _setup_endpoint(candidate)
    end_position = int(index.searchsorted(endpoint_time, side="right"))
    start = max(0, start_position - padding_bars)
    end = min(len(index), end_position + padding_bars)
    candles = result.candles.iloc[start:end]
    return DashboardResult(
        candles=candles,
        pivots=result.pivots,
        rankings=result.rankings,
        rsi=result.rsi.reindex(candles.index),
        pivot_state=result.pivot_state,
    )


def pattern_rankings(
    rankings: tuple[tuple[WaveCandidate, ConfidenceScore], ...],
    view: str,
) -> tuple[tuple[WaveCandidate, ConfidenceScore], ...]:
    """Select one pattern family or surface the best of both families."""
    if view.startswith("Impulse"):
        return tuple(item for item in rankings if item[0].pattern == "Impulse")
    if view.startswith("ZigZag"):
        return tuple(item for item in rankings if item[0].pattern == "ZigZag")

    best_impulse = next(
        (item for item in rankings if item[0].pattern == "Impulse"), None
    )
    best_zigzag = next(
        (item for item in rankings if item[0].pattern == "ZigZag"), None
    )
    featured = tuple(
        item for item in (best_impulse, best_zigzag) if item is not None
    )
    return featured + tuple(item for item in rankings if item not in featured)


def fallback_rankings_for_view(
    rankings: tuple[tuple[WaveCandidate, ConfidenceScore], ...],
    *,
    pattern_view: str,
    latest_timestamp: pd.Timestamp,
    days: int = 180,
) -> tuple[tuple[WaveCandidate, ConfidenceScore], ...]:
    """Return a broader-history fallback only when the scoped view is empty."""
    if not pattern_view.startswith("Impulse"):
        return ()
    impulses = tuple(item for item in rankings if item[0].pattern == "Impulse")
    if not impulses:
        return ()
    recent_impulses = recent_rankings(impulses, latest_timestamp, days=days)
    return recent_impulses if recent_impulses else impulses


def _inject_terminal_css(st) -> None:
    st.markdown(
        """
        <style>
        #MainMenu, footer, header, [data-testid="stToolbar"],
        [data-testid="stDecoration"], [data-testid="stStatusWidget"] {
            visibility: hidden; height: 0;
        }
        [data-testid="stAppViewContainer"] { background: #080b10; }
        .stMainBlockContainer {
            max-width: 100%; padding: 1rem 1.35rem 2rem;
        }
        h1, h2, h3 { letter-spacing: -0.025em; }
        h1 { font-size: 1.55rem !important; margin: 0 !important; }
        [data-testid="stMetric"] {
            background: #10141c; border: 1px solid #202632;
            border-radius: 8px; padding: .65rem .8rem;
        }
        [data-testid="stMetricLabel"] { color: #8792a2; }
        [data-testid="stMetricValue"] { font-size: 1.15rem; }
        [data-testid="stSelectbox"] label,
        [data-testid="stSlider"] label {
            color: #8b95a5; font-size: .7rem; text-transform: uppercase;
            letter-spacing: .06em;
        }
        div[data-baseweb="select"] > div {
            background: #10141c; border-color: #252c38;
        }
        .terminal-panel {
            background: #10141c; border: 1px solid #202632;
            border-radius: 8px; padding: .7rem .8rem; margin-bottom: .55rem;
        }
        .desktop-only { display: block; }
        .mobile-only { display: none; }
        .scanner-mobile-card {
            background: #10141c;
            border: 1px solid #202632;
            border-radius: 12px;
            padding: .9rem .95rem;
            margin-bottom: .8rem;
        }
        .scanner-mobile-topline {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: .75rem;
            margin-bottom: .55rem;
        }
        .scanner-mobile-decision {
            font-size: .9rem;
            font-weight: 800;
            letter-spacing: .03em;
        }
        .scanner-mobile-market {
            font-size: 1rem;
            font-weight: 700;
            color: #f2f5f8;
        }
        .scanner-mobile-meta {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: .6rem .9rem;
            margin-top: .5rem;
        }
        .scanner-mobile-field {
            min-width: 0;
        }
        .scanner-mobile-label {
            color: #7f8a9a;
            font-size: .63rem;
            text-transform: uppercase;
            letter-spacing: .08em;
            margin-bottom: .18rem;
        }
        .scanner-mobile-value {
            color: #f2f5f8;
            font-size: .95rem;
            font-weight: 600;
            line-height: 1.25;
            word-break: break-word;
        }
        .scanner-mobile-reason {
            margin-top: .7rem;
            padding-top: .65rem;
            border-top: 1px solid #202632;
        }
        .terminal-kicker {
            color: #7f8a9a; font-size: .65rem; text-transform: uppercase;
            letter-spacing: .09em;
        }
        .terminal-value {
            color: #f2f5f8; font-size: 1rem; font-weight: 650;
            margin-top: .15rem;
        }
        .positive { color: #18c98b; }
        .negative { color: #f05d68; }
        .status-live-badge {
            display: inline-flex; align-items: center; justify-content: flex-end; gap: .38rem;
        }
        .status-live-dot {
            width: .5rem; height: .5rem; border-radius: 999px; background: #18C98B;
            box-shadow: 0 0 0 rgba(24, 201, 139, 0.65);
            animation: statusPulse .95s ease-out 1;
        }
        @keyframes statusPulse {
            0% { transform: scale(.7); opacity: .55; box-shadow: 0 0 0 0 rgba(24, 201, 139, 0.7); }
            45% { transform: scale(1.15); opacity: 1; box-shadow: 0 0 0 8px rgba(24, 201, 139, 0.0); }
            100% { transform: scale(1); opacity: .95; box-shadow: 0 0 0 0 rgba(24, 201, 139, 0.0); }
        }
        .status-live-dot.offline {
            background: #F0B90B;
            animation: none;
            box-shadow: none;
        }
        iframe { border-radius: 8px; }
        [data-testid="stDataFrame"] {
            border: 1px solid #202632; border-radius: 8px;
        }
        @media (max-width: 768px) {
            .stMainBlockContainer {
                padding: .75rem .75rem 1.35rem;
            }
            .desktop-only { display: none !important; }
            .mobile-only { display: block !important; }
            .scanner-mobile-meta {
                grid-template-columns: 1fr;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_single_chart_fragmented() -> None:
    import streamlit as st
    import streamlit.components.v1 as components
    from streamlit_lightweight_charts import renderLightweightCharts

    components.html(
        """
        <script>
        (() => {
            const parentDocument = window.parent.document;
            const ignorePasswordManagers = (root = parentDocument) => {
                root.querySelectorAll(
                    '[data-testid="stSelectbox"] input, input[type="password"]'
                ).forEach((input) => {
                    input.setAttribute(
                        'autocomplete',
                        input.type === 'password' ? 'new-password' : 'off'
                    );
                    input.setAttribute('data-1p-ignore', 'true');
                    input.setAttribute('data-bwignore', 'true');
                    input.setAttribute('data-lpignore', 'true');
                    input.setAttribute('data-protonpass-ignore', 'true');
                });
            };

            ignorePasswordManagers();
            window.parent.__elliottPasswordManagerObserver?.disconnect();
            window.parent.__elliottPasswordManagerObserver =
                new window.parent.MutationObserver(() => ignorePasswordManagers());
            window.parent.__elliottPasswordManagerObserver.observe(
                parentDocument.body,
                {childList: true, subtree: true}
            );
        })();
        </script>
        """,
        height=0,
    )

    databases = discover_databases()
    query = st.query_params
    scanner_context = st.session_state.get("selected_scanner_setup")
    if isinstance(scanner_context, dict):
        scanner_defaults = scanner_selector_defaults(scanner_context)
        scanner_key = str(scanner_context.get("candidate_key", ""))
        if st.session_state.get("applied_scanner_candidate_key") != scanner_key:
            database_name = scanner_defaults["database"]
            matched_database = next(
                (path for path in databases if path.name == database_name), None
            )
            if matched_database is not None:
                st.session_state["selected_market"] = matched_database
            st.session_state["selected_timeframe"] = scanner_defaults["timeframe"]
            st.session_state["selected_pattern_view"] = scanner_defaults["pattern_view"]
            st.session_state["selected_candidate_scope"] = scanner_defaults[
                "candidate_scope"
            ]
            st.session_state.pop("selected_candidate_label", None)
            st.session_state["applied_scanner_candidate_key"] = scanner_key
    market_names = [path.name for path in databases]
    timeframe_options = tuple(TIMEFRAMES)
    pattern_options = (
        "Balanced | 1-5 + ABC",
        "Impulse | 1-5",
        "ZigZag | ABC",
    )
    scope_options = ("Actionable", "Recent | 30D", "All history")
    sensitivity_options = tuple(SENSITIVITY_PRESETS)

    market_default = (
        scanner_context.get("database")
        if isinstance(scanner_context, dict)
        else query.get("market", market_names[0] if market_names else None)
    )
    timeframe_default = (
        scanner_context.get("timeframe")
        if isinstance(scanner_context, dict)
        else query.get("timeframe", "1H")
    )
    context_pattern = (
        str(scanner_context.get("pattern", ""))
        if isinstance(scanner_context, dict)
        else ""
    )
    pattern_default = (
        "Impulse | 1-5"
        if context_pattern == "Impulse"
        else "ZigZag | ABC"
        if context_pattern == "ZigZag"
        else "Balanced | 1-5 + ABC"
        if context_pattern
        else query.get("pattern_view", pattern_options[0])
    )
    scope_default = (
        "All history"
        if isinstance(scanner_context, dict)
        else query.get("candidate_scope", scope_options[0])
    )
    sensitivity_default = query.get("sensitivity", "Balanced")
    live_default = query.get("live_yahoo", "0") == "1"
    provider_default = query.get(
        "data_provider",
        "Yahoo Test Refresh" if live_default else "Local SQLite",
    )
    live_interval_default = query.get("live_interval", "30s")

    title_col, status_col = st.columns([0.75, 0.25], vertical_alignment="center")
    with title_col:
        st.title("Elliott Wave Terminal")
        st.caption("Deterministic structure | volatility-adjusted pivots | causal scoring")
    with status_col:
        status_placeholder = st.empty()

    (
        selector_col,
        timeframe_col,
        pattern_col,
        sensitivity_col,
        live_col,
        universe_col,
    ) = st.columns([1.95, 0.7, 1.25, 1.0, 1.0, 1.0], vertical_alignment="bottom")

    with selector_col:
        selected_database = st.selectbox(
            "Market",
            databases,
            format_func=lambda path: path.name,
            index=market_names.index(market_default) if market_default in market_names else 0,
            key="selected_market",
            disabled=not databases,
        )
    with timeframe_col:
        timeframe = st.selectbox(
            "Timeframe",
            timeframe_options,
            index=timeframe_options.index(timeframe_default)
            if timeframe_default in timeframe_options
            else timeframe_options.index("1H"),
            key="selected_timeframe",
        )
    with pattern_col:
        pattern_view = st.selectbox(
            "Pattern View",
            pattern_options,
            index=pattern_options.index(pattern_default) if pattern_default in pattern_options else 0,
            key="selected_pattern_view",
        )
    with sensitivity_col:
        sensitivity = st.selectbox(
            "Sensitivity",
            sensitivity_options,
            index=sensitivity_options.index(sensitivity_default)
            if sensitivity_default in sensitivity_options
            else sensitivity_options.index("Balanced"),
            key="sensitivity_preset",
        )

    live_supported = selected_database is not None and resolve_market_symbol(selected_database) is not None
    with live_col:
        provider_options = ("Local SQLite", "Yahoo Test Refresh", "Upstox Live")
        data_provider = st.selectbox(
            "Data Provider",
            provider_options,
            index=(
                provider_options.index(provider_default)
                if provider_default in provider_options
                else 0
            ),
            key="data_provider",
        )
        yahoo_live_mode = data_provider == "Yahoo Test Refresh"
        upstox_live_mode = data_provider == "Upstox Live"
        live_mode = yahoo_live_mode or upstox_live_mode
    with universe_col:
        candidate_scope = st.selectbox(
            "Candidate Scope",
            scope_options,
            index=scope_options.index(scope_default) if scope_default in scope_options else 0,
            key="selected_candidate_scope",
        )
    if isinstance(scanner_context, dict):
        expected = scanner_selector_defaults(scanner_context)
        if (
            selected_database is None
            or selected_database.name != expected["database"]
            or timeframe != expected["timeframe"]
            or pattern_view != expected["pattern_view"]
        ):
            st.session_state.pop("selected_scanner_setup", None)
            st.session_state.pop("applied_scanner_candidate_key", None)
            scanner_context = None

    with st.expander("Advanced Settings", expanded=False):
        advanced_override = st.checkbox(
            "Override ATR settings",
            value=bool(st.session_state.get("advanced_atr_override", False)),
            key="advanced_atr_override",
        )
        override_cols = st.columns(3)
        with override_cols[0]:
            atr_multiplier_input = st.number_input(
                "ATR Multiplier",
                min_value=1.0,
                max_value=6.0,
                value=float(st.session_state.get("atr_multiplier", 2.0)),
                step=0.1,
                key="atr_multiplier",
                disabled=not advanced_override,
            )
        with override_cols[1]:
            atr_period_input = st.number_input(
                "ATR Period",
                min_value=5,
                max_value=50,
                value=int(st.session_state.get("atr_period", 14)),
                step=1,
                key="atr_period",
                disabled=not advanced_override,
            )
        with override_cols[2]:
            st.selectbox(
                "Live refresh",
                tuple(LIVE_REFRESH_SECONDS),
                index=tuple(LIVE_REFRESH_SECONDS).index(live_interval_default)
                if live_interval_default in LIVE_REFRESH_SECONDS
                else tuple(LIVE_REFRESH_SECONDS).index("30s"),
                key="live_refresh_interval",
                disabled=not live_mode,
            )
        atr_multiplier, atr_period = resolve_sensitivity(
            sensitivity,
            override_enabled=advanced_override,
            atr_multiplier=float(atr_multiplier_input),
            atr_period=int(atr_period_input),
        )
        st.caption(f"Resolved engine settings: ATR {atr_period} x {atr_multiplier:.1f}")
        strictness = st.selectbox(
            "Signal Strictness",
            tuple(SIGNAL_STRICTNESS_THRESHOLDS),
            index=tuple(SIGNAL_STRICTNESS_THRESHOLDS).index("Balanced"),
            key="signal_strictness",
        )
        quality_override = st.checkbox(
            "Override setup-quality threshold",
            value=False,
            key="override_setup_quality_threshold",
        )
        quality_override_value: float | None = None
        if quality_override:
            quality_override_value = st.slider(
                "Setup Quality Threshold",
                min_value=50.0,
                max_value=90.0,
                value=float(SIGNAL_STRICTNESS_THRESHOLDS[strictness]),
                step=0.5,
                key="setup_quality_threshold_override",
            )
        resolved_setup_quality_threshold = resolve_setup_quality_threshold(
            strictness,
            override_enabled=quality_override,
            override_value=quality_override_value,
        )
        st.session_state["resolved_setup_quality_threshold"] = (
            resolved_setup_quality_threshold
        )
        st.caption(
            f"Resolved setup-quality threshold: {resolved_setup_quality_threshold:.1f}"
        )
        upstox_token = load_upstox_access_token()
        if data_provider == "Upstox Live":
            if not upstox_token:
                upstox_token = st.text_input(
                    "Upstox Access Token",
                    type="password",
                    key="upstox_access_token",
                    help="Held only in this Streamlit session; credentials are never written.",
                ).strip()
            resolved_upstox, unresolved_upstox = resolve_instruments()
            selected_upstox_mapping = (
                next(
                    (
                        item
                        for item in resolved_upstox
                        if selected_database is not None
                        and item.database_name == selected_database.name
                    ),
                    None,
                )
                if selected_database is not None
                else None
            )
            st.caption(
                f"Selected universe: {len(DEFAULT_UPSTOX_UNIVERSE)} | "
                f"Resolved: {len(resolved_upstox)}"
            )
            if selected_database is not None and selected_upstox_mapping is None:
                unresolved_selected = next(
                    (
                        item
                        for item in unresolved_upstox
                        if item.database_name == selected_database.name
                    ),
                    None,
                )
                if unresolved_selected is not None:
                    st.warning(
                        "Selected market is not resolved for Upstox: "
                        f"{unresolved_selected.database_name} -> "
                        f"{unresolved_selected.exchange_segment}/"
                        f"{unresolved_selected.trading_symbol}"
                    )
            elif unresolved_upstox:
                st.warning(
                    "Some optional universe symbols are unresolved: "
                    + ", ".join(item.database_name for item in unresolved_upstox)
                )
            status_file = Path("upstox_live_status.json")
            status_payload: dict[str, object] = {}
            if status_file.exists():
                try:
                    status_payload = json.loads(status_file.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    status_payload = {"state": "UPSTOX CONNECTION ERROR"}
            st.caption(
                f"Connection: {status_payload.get('state', 'not started')} | "
                f"Last tick: {status_payload.get('last_tick_time') or 'n/a'} | "
                f"Last completed M5: "
                f"{status_payload.get('last_completed_m5') or 'n/a'}"
            )
            backfill_disabled = not bool(upstox_token) or selected_upstox_mapping is None
            if st.button(
                "Backfill last 6 months",
                key="upstox_backfill_6m",
                disabled=backfill_disabled,
                help=(
                    "Load roughly 6 months of completed 5-minute history from Upstox "
                    "into the selected market database."
                ),
            ):
                try:
                    with st.spinner(
                        f"Backfilling 6 months for {selected_database.name} from Upstox..."
                    ):
                        result = backfill_market(
                            selected_upstox_mapping,
                            token=upstox_token,
                            root=Path("."),
                            months=6,
                            include_today=True,
                        )
                    st.success(
                        f"Backfill complete: {result.database_name} | "
                        f"+{result.inserted_rows} rows | "
                        f"{result.start or 'n/a'} -> {result.end or 'n/a'}"
                    )
                except Exception as error:
                    st.error(f"Upstox backfill failed: {error}")
            st.caption(
                "Bootstrap once with `python live/upstox_bootstrap.py`, then run "
                "`python live/upstox_live_ingestor.py --universe default` in a "
                "separate terminal. OAuth refresh/login remains TODO."
            )

    risk_policy: RiskPolicy | None = None
    friction: Friction | None = None
    with st.expander("Risk Settings", expanded=False):
        risk_cols = st.columns(3)
        with risk_cols[0]:
            account_equity = st.number_input("Account equity", value=100000.0, min_value=0.0)
            risk_per_trade = st.number_input("Risk per trade %", value=1.0, min_value=0.0)
            maximum_open_risk = st.number_input("Maximum open risk %", value=3.0, min_value=0.0)
            maximum_daily_loss = st.number_input("Maximum daily loss %", value=3.0, min_value=0.0)
        with risk_cols[1]:
            minimum_reward_risk = st.number_input("Minimum reward/risk", value=1.5, min_value=0.0)
            lot_size = st.number_input("Lot size", value=1, min_value=1, step=1)
            current_open_risk = st.number_input("Current open risk", value=0.0, min_value=0.0)
            realized_daily_loss = st.number_input("Realized daily loss", value=0.0, min_value=0.0)
        with risk_cols[2]:
            spread = st.number_input("Spread", value=0.0, min_value=0.0)
            slippage = st.number_input("Slippage", value=0.0, min_value=0.0)
            commission = st.number_input("Commission", value=0.0, min_value=0.0)
        try:
            risk_policy = RiskPolicy(
                account_equity=float(account_equity),
                risk_per_trade_percent=float(risk_per_trade),
                maximum_open_risk_percent=float(maximum_open_risk),
                maximum_daily_loss_percent=float(maximum_daily_loss),
                minimum_reward_risk=float(minimum_reward_risk),
                lot_size=int(lot_size),
            )
            friction = Friction(
                spread=float(spread),
                slippage=float(slippage),
                commission=float(commission),
            )
            st.session_state["active_risk_policy"] = risk_policy
            st.session_state["active_friction"] = friction
            st.session_state["active_current_open_risk"] = float(current_open_risk)
            st.session_state["active_realized_daily_loss"] = float(realized_daily_loss)
        except (TypeError, ValueError) as error:
            st.warning(f"Invalid risk settings: {error}")

    if selected_database is not None:
        query["market"] = selected_database.name
    query["timeframe"] = timeframe
    query["pattern_view"] = pattern_view
    query["candidate_scope"] = candidate_scope
    query["sensitivity"] = sensitivity
    query["live_yahoo"] = "1" if live_mode else "0"
    query["data_provider"] = data_provider
    query["live_interval"] = st.session_state.get("live_refresh_interval", live_interval_default)

    if not databases or selected_database is None:
        st.info("Place a .db, .sqlite, or .sqlite3 asset database beside app.py.")
        return

    effective_live_mode = bool(live_mode and (live_supported or upstox_live_mode))
    refresh_key = st.session_state.get("live_refresh_interval", live_interval_default)
    refresh_seconds = LIVE_REFRESH_SECONDS.get(refresh_key, LIVE_REFRESH_SECONDS["30s"])

    @st.fragment(run_every=refresh_seconds if effective_live_mode else None)
    def _render_terminal_body() -> None:
        live_refresh_ok: bool | None = None
        live_state: LiveRefreshState | None = None
        upstox_live_active = data_provider == "Upstox Live"
        if effective_live_mode and data_provider == "Yahoo Test Refresh":
            try:
                live_state = refresh_live_database_state(selected_database)
                st.session_state["live_poll_count"] = int(
                    st.session_state.get("live_poll_count", 0)
                ) + 1
                st.session_state["live_last_checked_at"] = live_state.checked_at
                st.session_state["live_rows_added"] = live_state.rows_added
                if live_state.last_completed_bar is not None:
                    st.session_state["live_last_completed_bar"] = live_state.last_completed_bar
                st.caption(live_state.message)
                live_refresh_ok = True
            except Exception as error:
                st.warning(f"Yahoo live refresh failed: {error}")
                live_refresh_ok = False
                st.session_state["live_poll_count"] = int(
                    st.session_state.get("live_poll_count", 0)
                ) + 1
                st.session_state["live_last_checked_at"] = pd.Timestamp.now(tz="UTC")
                st.session_state["live_rows_added"] = 0
        elif effective_live_mode and upstox_live_active:
            st.session_state["live_poll_count"] = int(
                st.session_state.get("live_poll_count", 0)
            ) + 1
            st.session_state["live_last_checked_at"] = pd.Timestamp.now(tz="UTC")
            st.session_state["live_rows_added"] = 0
            last_completed_m5 = status_payload.get("last_completed_m5")
            if last_completed_m5:
                try:
                    st.session_state["live_last_completed_bar"] = pd.Timestamp(
                        last_completed_m5
                    )
                except Exception:
                    pass
            live_refresh_ok = str(
                status_payload.get("state", "")
            ) == "UPSTOX LIVE CONNECTED"
            st.caption(
                "Upstox live | "
                f"state={status_payload.get('state', 'not started')} | "
                f"last tick={status_payload.get('last_tick_time') or 'n/a'} | "
                f"last completed M5={status_payload.get('last_completed_m5') or 'n/a'}"
            )

        status_label, status_color = system_status(
            live_enabled=effective_live_mode,
            live_supported=live_supported,
            live_refresh_ok=live_refresh_ok,
        )
        if data_provider == "Yahoo Test Refresh" and effective_live_mode:
            status_label = (
                "YAHOO TEST REFRESH"
                if live_refresh_ok is not False
                else "LIVE REFRESH ERROR"
            )
        elif data_provider == "Upstox Live":
            if not upstox_token:
                status_label, status_color = "UPSTOX TOKEN MISSING", "#F05D68"
            elif selected_database is not None and selected_upstox_mapping is None:
                status_label, status_color = (
                    "UPSTOX MARKET UNRESOLVED",
                    "#F0B90B",
                )
            else:
                upstox_state = str(
                    status_payload.get("state", "UPSTOX CONNECTION ERROR")
                )
                status_label = (
                    "UPSTOX LIVE CONNECTED"
                    if upstox_state == "UPSTOX LIVE CONNECTED"
                    else "UPSTOX CONNECTION ERROR"
                )
                status_color = (
                    "#18C98B"
                    if status_label == "UPSTOX LIVE CONNECTED"
                    else "#F05D68"
                )
        pulse_id = int(st.session_state.get("live_poll_count", 0))
        status_prefix = (
            f"<span class='status-live-dot' id='live-dot-{pulse_id}'></span>"
            if status_label in {"SYSTEM LIVE", "UPSTOX LIVE CONNECTED"}
            else "<span class='status-live-dot offline'></span>"
        )
        status_placeholder.markdown(
            f"<div class='status-live-badge' style='text-align:right;color:{status_color};font-size:.78rem;font-weight:650'>{status_prefix}<span>{status_label}</span></div>",
            unsafe_allow_html=True,
        )

        if effective_live_mode:
            checked_at = st.session_state.get("live_last_checked_at")
            rows_added = int(st.session_state.get("live_rows_added", 0))
            last_new_bar = st.session_state.get("live_last_completed_bar")
            checked_text = (
                pd.Timestamp(checked_at).strftime("%d %b %H:%M:%S UTC")
                if checked_at is not None
                else "n/a"
            )
            new_bar_text = (
                pd.Timestamp(last_new_bar).strftime("%d %b %H:%M UTC")
                if last_new_bar is not None
                else "n/a"
            )
            heartbeat_text = (
                "Polling active"
                if live_refresh_ok is not False
                else "Polling error"
            )
            heartbeat_color = "#18C98B" if live_refresh_ok is not False else "#F05D68"
            st.markdown(
                "<div class='terminal-panel' style='display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:.8rem'>"
                f"<div><div class='terminal-kicker'>Live heartbeat</div><div class='terminal-value' style='color:{heartbeat_color}'>{heartbeat_text}</div></div>"
                f"<div><div class='terminal-kicker'>Last checked</div><div class='terminal-value'>{checked_text}</div></div>"
                f"<div><div class='terminal-kicker'>Last new bar</div><div class='terminal-value'>{new_bar_text}</div></div>"
                f"<div><div class='terminal-kicker'>Rows added</div><div class='terminal-value'>{rows_added:+d}</div></div>"
                "</div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                """
                <div class='terminal-panel' style='padding:.55rem .8rem'>
                    <div style='color:#F4F7FA;font-size:.8rem'>
                        <strong>Live behavior:</strong> ticks stream continuously, but the Elliott-wave engine
                        recomputes only when a new <strong>completed M5 candle</strong> is written to the
                        canonical SQLite feed.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        try:
            result = compute_dashboard(selected_database, timeframe, atr_multiplier, atr_period)
        except (ValueError, OSError) as error:
            st.error(f"Unable to calculate dashboard: {error}")
            return

        if result.candles.empty:
            st.warning("No complete candles are available for this timeframe.")
            return

        if candidate_scope == "Actionable":
            scoped_rankings = actionable_rankings(result.rankings, result.candles)
        elif candidate_scope.startswith("Recent"):
            scoped_rankings = recent_rankings(result.rankings, result.candles.index[-1])
        else:
            scoped_rankings = result.rankings
        scoped_rankings = live_rankings(pattern_rankings(scoped_rankings, pattern_view))
        display_rankings = scoped_rankings
        fallback_notice: str | None = None
        if not scoped_rankings:
            fallback_rankings = fallback_rankings_for_view(
                result.rankings,
                pattern_view=pattern_view,
                latest_timestamp=result.candles.index[-1],
            )
            if fallback_rankings:
                display_rankings = live_rankings(fallback_rankings)
                fallback_notice = (
                    "No impulse detected in the current scope. "
                    "Showing the nearest recent impulse from broader history "
                    "for Wave 3 inspection."
                )
        scanner_match_exact: bool | None = None
        if isinstance(scanner_context, dict) and display_rankings:
            scanner_index, scanner_match_exact = match_scanner_candidate(
                display_rankings, scanner_context
            )
            if scanner_index:
                display_rankings = (
                    display_rankings[scanner_index],
                    *display_rankings[:scanner_index],
                    *display_rankings[scanner_index + 1 :],
                )
        elif isinstance(scanner_context, dict):
            scanner_match_exact = False

        view_result = DashboardResult(
            result.candles,
            result.pivots,
            display_rankings,
            result.rsi,
            result.pivot_state,
        )

        last_close = float(result.candles.iloc[-1]["close"])
        previous_close = (
            float(result.candles.iloc[-2]["close"]) if len(result.candles) > 1 else last_close
        )
        change = last_close - previous_close
        change_percent = 100 * change / previous_close if previous_close else 0.0
        metric_cols = st.columns(4)
        metric_cols[0].metric("LAST", f"{last_close:,.2f}", f"{change_percent:+.2f}%")
        metric_cols[1].metric("TIMEFRAME", timeframe)
        impulses = sum(candidate.pattern == "Impulse" for candidate, _score in scoped_rankings)
        abc_like = sum(candidate.pattern != "Impulse" for candidate, _score in scoped_rankings)
        metric_cols[2].metric("VALID PATHS", len(scoped_rankings), f"{impulses} impulse | {abc_like} ABC")
        metric_cols[3].metric(
            "DATA THROUGH",
            _offset_timestamp(result.candles.index[-1], timeframe).strftime("%d %b | %H:%M UTC"),
        )

        top_rankings = display_rankings[:3]
        alert_threshold = resolved_setup_quality_threshold
        chart_col, inspector_col = st.columns([4.9, 1.45], gap="medium")

        selected_candidate = top_rankings[0][0] if top_rankings else None
        selected_score = top_rankings[0][1] if top_rankings else None
        selected_index = 0
        overlays_list = [False, False, False]
        focus_selected = True

        with inspector_col:
            st.markdown(
                """
                <div style='font-size:1.18rem;font-weight:700;color:#F4F7FA;margin:.15rem 0 .55rem 0'>
                    Structure Inspector
                </div>
                """,
                unsafe_allow_html=True,
            )
            if fallback_notice:
                st.warning(fallback_notice, icon="⚠️")
            if not top_rankings:
                if pattern_view.startswith("Impulse"):
                    st.info(
                        "No impulse detected in the current scope. "
                        "Wave 3 start/end is unavailable for this filter."
                    )
                else:
                    st.info("No candidates available in the current scope.")
            else:
                labels = [
                    _candidate_name(index, candidate, score)
                    for index, (candidate, score) in enumerate(top_rankings)
                ]
                selected_query_key = query.get("selected_candidate")
                default_selected_index = (
                    0
                    if isinstance(scanner_context, dict)
                    else _resolve_selected_index(top_rankings, selected_query_key)
                )
                selected_label = st.selectbox(
                    "Inspect path",
                    labels,
                    index=default_selected_index,
                    label_visibility="collapsed",
                    key="selected_candidate_label",
                )
                selected_index = labels.index(selected_label)
                selected_candidate, selected_score = top_rankings[selected_index]
                query["selected_candidate"] = _candidate_query_key(selected_candidate)
                overlays_list[selected_index] = True
                for alt_index in range(1, min(3, len(top_rankings))):
                    if alt_index == selected_index:
                        overlays_list[alt_index] = True
                        continue
                    overlays_list[alt_index] = st.checkbox(
                        f"Show alternative #{alt_index + 1}",
                        value=query.get(f"overlay_{alt_index}", "0") == "1",
                        key=f"overlay_{alt_index}",
                    )
                    query[f"overlay_{alt_index}"] = "1" if overlays_list[alt_index] else "0"
                focus_selected = st.checkbox(
                    "Focus chart on selected path",
                    value=query.get("focus_selected", "1") == "1",
                    key="focus_selected_path",
                )
                query["focus_selected"] = "1" if focus_selected else "0"
                lifecycle = candidate_lifecycle(selected_candidate, result.candles)
                inspector_decision = build_decision_state(
                    selected_candidate,
                    selected_score,
                    alert_threshold,
                    lifecycle,
                    result.candles,
                    risk_policy,
                    friction,
                    current_open_risk=float(current_open_risk),
                    realized_daily_loss=float(realized_daily_loss),
                )
                target_low, target_high = target_zone(selected_candidate)
                st.markdown(
                    f"""
                    <div class='terminal-panel'>
                        <div class='terminal-kicker'>Selected structure</div>
                        <div class='terminal-value {'positive' if selected_candidate.direction == 'Bullish' else 'negative'}'>
                            {html.escape(selected_candidate.pattern)} | {html.escape(selected_candidate.direction)}
                        </div>
                        <div style='color:#8c96a5;font-size:.78rem;margin-top:.28rem'>
                            Stage: {html.escape(selected_candidate.status)} | Lifecycle: {html.escape(lifecycle)}
                        </div>
                    </div>
                    <div class='terminal-panel'>
                        <div class='terminal-kicker'>Setup Quality Score</div>
                        <div class='terminal-value'>{selected_score.total:.1f} / 100</div>
                    </div>
                    <div class='terminal-panel'>
                        <div class='terminal-kicker'>{'Floor' if selected_candidate.invalidation_side == 'below' else 'Ceiling'} invalidation</div>
                        <div class='terminal-value negative'>{selected_candidate.invalidation_level:,.2f}</div>
                    </div>
                    <div class='terminal-panel'>
                        <div class='terminal-kicker'>Planning target zone</div>
                        <div class='terminal-value positive'>{target_low:,.2f} - {target_high:,.2f}</div>
                    </div>
                    <div class='terminal-panel'>
                        <div class='terminal-kicker'>Current wave</div>
                        <div class='terminal-value'>{html.escape(current_wave_label(selected_candidate))}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                wave_three = wave_three_window(selected_candidate)
                if wave_three is not None:
                    st.markdown(
                        f"""
                        <div class='terminal-panel'>
                            <div class='terminal-kicker'>Wave 3 window</div>
                            <div style='color:#f2f5f8;font-size:.8rem'>{html.escape(_format_wave_point(wave_three[0]))}</div>
                            <div style='color:#f2f5f8;font-size:.8rem;margin-top:.2rem'>{html.escape(_format_wave_point(wave_three[1]))}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                elif selected_candidate.pattern == "Impulse":
                    st.markdown(
                        """
                        <div class='terminal-panel'>
                            <div class='terminal-kicker'>Wave 3 window</div>
                            <div style='color:#f2f5f8;font-size:.8rem'>Unavailable for the selected impulse candidate.</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                hint_rows = []
                for hint in system_hints(
                    selected_candidate,
                    selected_score,
                    alert_threshold,
                    lifecycle=lifecycle,
                    decision_state=inspector_decision,
                ):
                    label, _, value = hint.partition(": ")
                    if not value:
                        label, value = hint, ""
                    hint_rows.append(
                        "<tr>"
                        f"<td style='padding:.28rem .4rem .28rem 0;color:#F4F7FA;font-weight:650;white-space:nowrap'>{html.escape(label)}</td>"
                        f"<td style='padding:.28rem 0;color:#F4F7FA'>{_format_hint_value(value)}</td>"
                        "</tr>"
                    )
                st.markdown(
                    "<div class='terminal-panel'><div class='terminal-kicker'>System hints</div>"
                    "<table style='width:100%;border-collapse:collapse;margin-top:.35rem;font-size:.78rem'>"
                    + "".join(hint_rows)
                    + "</table></div>",
                    unsafe_allow_html=True,
                )

                with st.expander("Score audit", expanded=False):
                    items = [
                        {
                            "Guideline": item.reason,
                            "Points": f"{item.points:.2f}/{item.maximum:.0f}",
                        }
                        for item in selected_score.items
                    ]
                    if items:
                        st.dataframe(pd.DataFrame(items), hide_index=True, width="stretch")
                    else:
                        st.info("No score audit items available.")

        overlays = tuple(overlays_list[:3])
        with chart_col:
            if isinstance(scanner_context, dict):
                context_text = (
                    f"{scanner_context.get('market', 'n/a')} | "
                    f"{scanner_context.get('timeframe', 'n/a')} | "
                    f"{scanner_context.get('pattern', 'n/a')} | "
                    f"{scanner_context.get('trade_decision', 'n/a')} | "
                    f"{scanner_context.get('structure_status', 'n/a')} | "
                    f"{scanner_context.get('current_wave', 'n/a')}"
                )
                st.info(f"Opened from scanner: {context_text}")
                if scanner_match_exact:
                    st.success("Exact scanner setup matched.")
                else:
                    notice_key = (
                        f"scanner-match-warning:"
                        f"{scanner_context.get('candidate_key', '')}"
                    )
                    if st.session_state.get("last_navigation_notice") != notice_key:
                        st.toast(
                            "Scanner setup changed or is unavailable after refresh. "
                            "Showing the nearest candidate.",
                            icon="⚠️",
                        )
                        st.session_state["last_navigation_notice"] = notice_key
                    st.warning(
                        "Scanner setup could not be matched after data refresh. "
                        "Showing the nearest candidate. Re-run scanner if needed."
                    )
            legend_candidate = selected_candidate if top_rankings else None
            st.markdown(_legend_markup(top_rankings, overlays, legend_candidate), unsafe_allow_html=True)
            selected_lifecycle = (
                candidate_lifecycle(selected_candidate, result.candles)
                if selected_candidate is not None
                else None
            )
            decision = build_decision_state(
                selected_candidate,
                selected_score,
                alert_threshold,
                selected_lifecycle,
                result.candles,
                risk_policy,
                friction,
                current_open_risk=float(current_open_risk),
                realized_daily_loss=float(realized_daily_loss),
            )
            st.markdown(decision_panel_markup(decision), unsafe_allow_html=True)
            if selected_candidate is not None and selected_score is not None:
                threshold_text, buy_text, sell_text = marker_status(
                    top_rankings,
                    overlays,
                    alert_threshold,
                )
            else:
                threshold_text, buy_text, sell_text = ("n/a", "n/a", "n/a")
            st.markdown(
                "<div class='terminal-panel' style='display:grid;grid-template-columns:1fr 1fr 1fr;gap:.8rem'>"
                f"<div><div class='terminal-kicker'>Setup quality threshold</div><div class='terminal-value'>{threshold_text}</div></div>"
                f"<div><div class='terminal-kicker'>Bullish setup ref</div><div class='terminal-value positive'>{html.escape(buy_text)}</div></div>"
                f"<div><div class='terminal-kicker'>Bearish setup ref</div><div class='terminal-value negative'>{html.escape(sell_text)}</div></div>"
                "</div>",
                unsafe_allow_html=True,
            )
            chart_result = (
                focus_dashboard(view_result, selected_candidate)
                if selected_candidate is not None and focus_selected
                else view_result
            )
            renderLightweightCharts(
                build_lightweight_charts(
                    chart_result,
                    overlays,
                    selected_index=selected_index,
                    alert_threshold=alert_threshold,
                ),
                key=(
                    f"elliott-{selected_database.name}-{timeframe}-{candidate_scope}-"
                    f"{selected_index}-{overlays}-{focus_selected}-{alert_threshold}-"
                    f"{pattern_view}-{sensitivity}-{effective_live_mode}"
                ),
            )

        history_col, operations_col = st.columns([0.7, 0.3], gap="medium")
        with history_col:
            with st.expander("Candidate history", expanded=False):
                if not scoped_rankings:
                    st.info("No candidate history in this scope.")
                else:
                    history_rows = [
                        {
                            "Rank": index + 1,
                            "Pattern": candidate.pattern,
                            "Status": candidate.status,
                            "Direction": candidate.direction,
                            "Confidence": round(score.total, 2),
                            "Wave": current_wave_label(candidate),
                        }
                        for index, (candidate, score) in enumerate(scoped_rankings[:25])
                    ]
                    st.dataframe(pd.DataFrame(history_rows), hide_index=True, width="stretch")
            with st.expander("Historical Evidence", expanded=False):
                st.caption(
                    "Causal historical results are descriptive evidence, not a probability of profit."
                )
                if st.button("Run evidence check", key="run_evidence_check"):
                    try:
                        with st.spinner("Building causal historical rankings..."):
                            historical_rankings = build_causal_rankings(
                                result.candles,
                                multiplier=atr_multiplier,
                                atr_period=atr_period,
                            )
                            evidence = run_backtest(
                                result.candles,
                                historical_rankings,
                                minimum_confidence=alert_threshold,
                                friction=friction or Friction(),
                            )
                        st.session_state["historical_evidence"] = evidence
                    except (ValueError, IndexError) as error:
                        st.session_state["historical_evidence"] = None
                        st.warning(f"Insufficient historical evidence: {error}")
                evidence = st.session_state.get("historical_evidence")
                if evidence is not None:
                    evidence_cols = st.columns(4)
                    evidence_cols[0].metric("Total trades", evidence.total_trades)
                    evidence_cols[1].metric("Win rate", f"{evidence.win_rate:.1f}%")
                    evidence_cols[2].metric("Net P/L", f"{evidence.net_profit_loss:,.2f}")
                    evidence_cols[3].metric("Maximum drawdown", f"{evidence.maximum_drawdown:,.2f}")
                    if evidence.ledger:
                        st.dataframe(
                            pd.DataFrame(
                                [
                                    {
                                        "Direction": trade.direction,
                                        "Entry time": trade.entry_time,
                                        "Exit time": trade.exit_time,
                                        "Entry": trade.entry_price,
                                        "Exit": trade.exit_price,
                                        "Result": trade.exit_reason,
                                        "Net P/L": trade.net_pnl,
                                    }
                                    for trade in evidence.ledger[-10:]
                                ]
                            ),
                            hide_index=True,
                            width="stretch",
                        )
                    else:
                        st.info("Insufficient historical evidence")

        with operations_col:
            with st.expander("Alerts & operations", expanded=False):
                st.caption(
                    "Telegram shell remains local-only in this MVP. "
                    "Use the selected candidate, threshold gate, and trader-facing panel "
                    "to decide whether to forward a signal externally."
                )
                configure_telegram = st.checkbox(
                    "Configure Telegram credentials",
                    value=False,
                    key="configure_telegram_credentials",
                    help=(
                        "Credentials are kept out of the page until needed so browsers "
                        "do not mistake the Structure Inspector for a login field."
                    ),
                )
                bot_token = ""
                chat_id = ""
                if configure_telegram:
                    bot_token = st.text_input(
                        "Bot Token",
                        type="password",
                        key="telegram_bot_token",
                        autocomplete="new-password",
                    )
                    chat_id = st.text_input(
                        "Chat ID",
                        key="telegram_chat_id",
                        autocomplete="off",
                    )
                st.caption(
                    f"Setup quality threshold: {alert_threshold:.1f}"
                )
                if st.button(
                    "Emit local alert shell",
                    width="stretch",
                    disabled=not configure_telegram,
                ):
                    emit_alert_shell(selected_candidate, selected_score, bot_token, chat_id)

    _render_terminal_body()


def _render_single_chart() -> None:
    _render_single_chart_fragmented()
    return

    # TODO: Remove this legacy duplicate after downstream UI snapshot users migrate.
    # It is unreachable; the maintained implementation is the fragmented renderer above.
    import streamlit as st
    import streamlit.components.v1 as components
    from streamlit_lightweight_charts import renderLightweightCharts

    databases = discover_databases()
    query = st.query_params
    market_names = [path.name for path in databases]
    timeframe_options = tuple(TIMEFRAMES)
    pattern_options = (
        "Balanced | 1-5 + ABC",
        "Impulse | 1-5",
        "ZigZag | ABC",
    )
    scope_options = ("Actionable", "Recent | 30D", "All history")
    sensitivity_options = tuple(SENSITIVITY_PRESETS)

    market_default = query.get("market", market_names[0] if market_names else None)
    timeframe_default = query.get("timeframe", "1H")
    pattern_default = query.get("pattern_view", pattern_options[0])
    scope_default = query.get("candidate_scope", scope_options[0])
    sensitivity_default = query.get("sensitivity", "Balanced")
    live_default = query.get("live_yahoo", "0") == "1"
    live_interval_default = query.get("live_interval", "30s")

    title_col, status_col = st.columns([0.75, 0.25], vertical_alignment="center")
    with title_col:
        st.title("Elliott Wave Terminal")
        st.caption("Deterministic structure | volatility-adjusted pivots | causal scoring")
    with status_col:
        status_placeholder = st.empty()

    (
        selector_col,
        timeframe_col,
        pattern_col,
        sensitivity_col,
        live_col,
        universe_col,
    ) = st.columns([1.95, 0.7, 1.25, 1.0, 1.0, 1.0], vertical_alignment="bottom")

    with selector_col:
        selected_database = st.selectbox(
            "Market",
            databases,
            format_func=lambda path: path.name,
            index=market_names.index(market_default) if market_default in market_names else 0,
            key="selected_market",
            disabled=not databases,
        )
    with timeframe_col:
        timeframe = st.selectbox(
            "Timeframe",
            timeframe_options,
            index=timeframe_options.index(timeframe_default) if timeframe_default in timeframe_options else timeframe_options.index("1H"),
            key="selected_timeframe",
        )
    with pattern_col:
        pattern_view = st.selectbox(
            "Pattern View",
            pattern_options,
            index=pattern_options.index(pattern_default) if pattern_default in pattern_options else 0,
            key="selected_pattern_view",
        )
    with sensitivity_col:
        sensitivity = st.selectbox(
            "Sensitivity",
            sensitivity_options,
            index=sensitivity_options.index(sensitivity_default) if sensitivity_default in sensitivity_options else sensitivity_options.index("Balanced"),
            key="sensitivity_preset",
        )
    live_supported = selected_database is not None and resolve_market_symbol(selected_database) is not None
    with live_col:
        live_mode = st.toggle(
            "Yahoo Live",
            value=live_default,
            key="live_yahoo_enabled",
            disabled=not live_supported,
        )
    with universe_col:
        candidate_scope = st.selectbox(
            "Candidate Scope",
            scope_options,
            index=scope_options.index(scope_default) if scope_default in scope_options else 0,
            key="selected_candidate_scope",
        )

    with st.expander("Advanced Settings", expanded=False):
        advanced_override = st.checkbox(
            "Override ATR settings",
            value=bool(st.session_state.get("advanced_atr_override", False)),
            key="advanced_atr_override",
        )
        override_cols = st.columns(3)
        with override_cols[0]:
            atr_multiplier_input = st.number_input(
                "ATR Multiplier",
                min_value=1.0,
                max_value=6.0,
                value=float(st.session_state.get("atr_multiplier", 2.0)),
                step=0.1,
                key="atr_multiplier",
                disabled=not advanced_override,
            )
        with override_cols[1]:
            atr_period_input = st.number_input(
                "ATR Period",
                min_value=5,
                max_value=50,
                value=int(st.session_state.get("atr_period", 14)),
                step=1,
                key="atr_period",
                disabled=not advanced_override,
            )
        with override_cols[2]:
            live_interval = st.selectbox(
                "Live refresh",
                tuple(LIVE_REFRESH_SECONDS),
                index=tuple(LIVE_REFRESH_SECONDS).index(live_interval_default)
                if live_interval_default in LIVE_REFRESH_SECONDS
                else tuple(LIVE_REFRESH_SECONDS).index("30s"),
                key="live_refresh_interval",
                disabled=not live_mode,
            )
        atr_multiplier, atr_period = resolve_sensitivity(
            sensitivity,
            override_enabled=advanced_override,
            atr_multiplier=float(atr_multiplier_input),
            atr_period=int(atr_period_input),
        )
        st.caption(f"Resolved engine settings: ATR {atr_period} x {atr_multiplier:.1f}")

    if selected_database is not None:
        query["market"] = selected_database.name
    query["timeframe"] = timeframe
    query["pattern_view"] = pattern_view
    query["candidate_scope"] = candidate_scope
    query["sensitivity"] = sensitivity
    query["live_yahoo"] = "1" if live_mode else "0"
    query["live_interval"] = st.session_state.get("live_refresh_interval", live_interval_default)

    if not databases or selected_database is None:
        st.info("Place a .db, .sqlite, or .sqlite3 asset database beside app.py.")
        return

    live_refresh_ok: bool | None = None
    if not live_supported:
        live_mode = False
    elif live_mode:
        try:
            st.caption(refresh_live_database(selected_database))
            live_refresh_ok = True
        except Exception as error:
            st.warning(f"Yahoo live refresh failed: {error}")
            live_refresh_ok = False
        refresh_seconds = LIVE_REFRESH_SECONDS[st.session_state.get("live_refresh_interval", "30s")]
        components.html(
            f"""
            <script>
            window.parent.clearTimeout(window.__elliottYahooLiveTimer);
            window.__elliottYahooLiveTimer = window.parent.setTimeout(
                function() {{ window.parent.location.reload(); }},
                {refresh_seconds * 1000}
            );
            </script>
            """,
            height=0,
        )

    status_label, status_color = system_status(
        live_enabled=live_mode,
        live_supported=live_supported,
        live_refresh_ok=live_refresh_ok,
    )
    with status_col:
        status_placeholder.markdown(
            f"<div style='text-align:right;color:{status_color};font-size:.78rem;font-weight:650'>{status_label}</div>",
            unsafe_allow_html=True,
        )

    try:
        result = compute_dashboard(selected_database, timeframe, atr_multiplier, atr_period)
    except (ValueError, OSError) as error:
        st.error(f"Unable to calculate dashboard: {error}")
        return

    if result.candles.empty:
        st.warning("No complete candles are available for this timeframe.")
        return

    if candidate_scope == "Actionable":
        scoped_rankings = actionable_rankings(result.rankings, result.candles)
    elif candidate_scope.startswith("Recent"):
        scoped_rankings = recent_rankings(result.rankings, result.candles.index[-1])
    else:
        scoped_rankings = result.rankings
    scoped_rankings = live_rankings(pattern_rankings(scoped_rankings, pattern_view))
    display_rankings = scoped_rankings
    fallback_notice: str | None = None
    if not scoped_rankings:
        fallback_rankings = fallback_rankings_for_view(
            result.rankings,
            pattern_view=pattern_view,
            latest_timestamp=result.candles.index[-1],
        )
        if fallback_rankings:
            display_rankings = live_rankings(fallback_rankings)
            fallback_notice = (
                "No impulse detected in the current scope. "
                "Showing the nearest recent impulse from broader history "
                "for Wave 3 inspection."
            )
    view_result = DashboardResult(
        result.candles,
        result.pivots,
        display_rankings,
        result.rsi,
        result.pivot_state,
    )

    last_close = float(result.candles.iloc[-1]["close"])
    previous_close = float(result.candles.iloc[-2]["close"]) if len(result.candles) > 1 else last_close
    change = last_close - previous_close
    change_percent = 100 * change / previous_close if previous_close else 0.0
    metric_cols = st.columns(4)
    metric_cols[0].metric("LAST", f"{last_close:,.2f}", f"{change_percent:+.2f}%")
    metric_cols[1].metric("TIMEFRAME", timeframe)
    impulses = sum(candidate.pattern == "Impulse" for candidate, _score in scoped_rankings)
    abc_like = sum(candidate.pattern != "Impulse" for candidate, _score in scoped_rankings)
    metric_cols[2].metric("VALID PATHS", len(scoped_rankings), f"{impulses} impulse | {abc_like} ABC")
    metric_cols[3].metric(
        "DATA THROUGH",
        _offset_timestamp(result.candles.index[-1], timeframe).strftime("%d %b | %H:%M UTC"),
    )

    top_rankings = display_rankings[:3]
    alert_threshold = float(st.session_state.get("alert_threshold", 75.0))

    chart_col, inspector_col = st.columns([4.9, 1.45], gap="medium")

    selected_candidate = top_rankings[0][0] if top_rankings else None
    selected_score = top_rankings[0][1] if top_rankings else None
    selected_index = 0
    overlays_list = [False, False, False]
    focus_selected = True

    with inspector_col:
        st.markdown(
            """
            <div style='font-size:1.18rem;font-weight:700;color:#F4F7FA;margin:.15rem 0 .55rem 0'>
                Structure Inspector
            </div>
            """,
            unsafe_allow_html=True,
        )
        if fallback_notice:
            st.warning(fallback_notice, icon="⚠️")
        if not top_rankings:
            if pattern_view.startswith("Impulse"):
                st.info(
                    "No impulse detected in the current scope. "
                    "Wave 3 start/end is unavailable for this filter."
                )
            else:
                st.info("No candidates available in the current scope.")
        else:
            labels = [
                _candidate_name(index, candidate, score)
                for index, (candidate, score) in enumerate(top_rankings)
            ]
            selected_query_key = query.get("selected_candidate")
            default_selected_index = _resolve_selected_index(
                top_rankings,
                selected_query_key,
            )
            selected_label = st.selectbox(
                "Inspect path",
                labels,
                index=default_selected_index,
                label_visibility="collapsed",
                key="selected_candidate_label",
            )
            selected_index = labels.index(selected_label)
            selected_candidate, selected_score = top_rankings[selected_index]
            query["selected_candidate"] = _candidate_query_key(selected_candidate)
            overlays_list[selected_index] = True
            for alt_index in range(1, min(3, len(top_rankings))):
                if alt_index == selected_index:
                    overlays_list[alt_index] = True
                    continue
                overlays_list[alt_index] = st.checkbox(
                    f"Show alternative #{alt_index + 1}",
                    value=query.get(f"overlay_{alt_index}", "0") == "1",
                    key=f"overlay_{alt_index}",
                )
                query[f"overlay_{alt_index}"] = (
                    "1" if overlays_list[alt_index] else "0"
                )
            focus_selected = st.checkbox(
                "Focus chart on selected path",
                value=query.get("focus_selected", "1") == "1",
                key="focus_selected_path",
            )
            query["focus_selected"] = "1" if focus_selected else "0"
            lifecycle = candidate_lifecycle(selected_candidate, result.candles)
            target_low, target_high = target_zone(selected_candidate)
            st.markdown(
                f"""
                <div class='terminal-panel'>
                    <div class='terminal-kicker'>Selected structure</div>
                    <div class='terminal-value {'positive' if selected_candidate.direction == 'Bullish' else 'negative'}'>
                        {html.escape(selected_candidate.pattern)} | {html.escape(selected_candidate.direction)}
                    </div>
                    <div style='color:#8c96a5;font-size:.78rem;margin-top:.28rem'>
                        Stage: {html.escape(selected_candidate.status)} | Lifecycle: {html.escape(lifecycle)}
                    </div>
                </div>
                <div class='terminal-panel'>
                    <div class='terminal-kicker'>Confidence Score</div>
                    <div class='terminal-value'>{selected_score.total:.1f} / 100</div>
                </div>
                <div class='terminal-panel'>
                    <div class='terminal-kicker'>{'Floor' if selected_candidate.invalidation_side == 'below' else 'Ceiling'} invalidation</div>
                    <div class='terminal-value negative'>{selected_candidate.invalidation_level:,.2f}</div>
                </div>
                <div class='terminal-panel'>
                    <div class='terminal-kicker'>Fibonacci target zone</div>
                    <div class='terminal-value positive'>{target_low:,.2f} - {target_high:,.2f}</div>
                </div>
                <div class='terminal-panel'>
                    <div class='terminal-kicker'>Current wave</div>
                    <div class='terminal-value'>{html.escape(current_wave_label(selected_candidate))}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            wave_three = wave_three_window(selected_candidate)
            if wave_three is not None:
                st.markdown(
                    f"""
                    <div class='terminal-panel'>
                        <div class='terminal-kicker'>Wave 3 window</div>
                        <div style='color:#f2f5f8;font-size:.8rem'>{html.escape(_format_wave_point(wave_three[0]))}</div>
                        <div style='color:#f2f5f8;font-size:.8rem;margin-top:.2rem'>{html.escape(_format_wave_point(wave_three[1]))}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            elif selected_candidate.pattern == "Impulse":
                st.markdown(
                    """
                    <div class='terminal-panel'>
                        <div class='terminal-kicker'>Wave 3 window</div>
                        <div style='color:#f2f5f8;font-size:.8rem'>Unavailable for the selected impulse candidate.</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            hint_rows = []
            for hint in system_hints(selected_candidate, selected_score, alert_threshold, lifecycle=lifecycle):
                label, _, value = hint.partition(": ")
                if not value:
                    label, value = hint, ""
                hint_rows.append(
                    "<tr>"
                    f"<td style='padding:.28rem .4rem .28rem 0;color:#F4F7FA;font-weight:650;white-space:nowrap'>{html.escape(label)}</td>"
                    f"<td style='padding:.28rem 0;color:#F4F7FA'>{_format_hint_value(value)}</td>"
                    "</tr>"
                )
            st.markdown(
                "<div class='terminal-panel'><div class='terminal-kicker'>System hints</div>"
                "<table style='width:100%;border-collapse:collapse;margin-top:.35rem;font-size:.78rem'>"
                + "".join(hint_rows)
                + "</table></div>",
                unsafe_allow_html=True,
            )

            with st.expander("Score audit", expanded=False):
                items = [
                    {
                        "Guideline": item.reason,
                        "Points": f"{item.points:.2f}/{item.maximum:.0f}",
                    }
                    for item in selected_score.items
                ]
                if items:
                    st.dataframe(pd.DataFrame(items), hide_index=True, width="stretch")
                else:
                    st.info("No score audit items available.")

    overlays = tuple(overlays_list[:3])
    with chart_col:
        legend_candidate = selected_candidate if top_rankings else None
        st.markdown(_legend_markup(top_rankings, overlays, legend_candidate), unsafe_allow_html=True)
        if selected_candidate is not None and selected_score is not None:
            threshold_text, buy_text, sell_text = marker_status(
                top_rankings,
                overlays,
                alert_threshold,
            )
        else:
            threshold_text, buy_text, sell_text = ("n/a", "n/a", "n/a")
        st.markdown(
            "<div class='terminal-panel' style='display:grid;grid-template-columns:1fr 1fr 1fr;gap:.8rem'>"
            f"<div><div class='terminal-kicker'>Marker threshold</div><div class='terminal-value'>{threshold_text}</div></div>"
            f"<div><div class='terminal-kicker'>Buy at</div><div class='terminal-value positive'>{html.escape(buy_text)}</div></div>"
            f"<div><div class='terminal-kicker'>Sell at</div><div class='terminal-value negative'>{html.escape(sell_text)}</div></div>"
            "</div>",
            unsafe_allow_html=True,
        )
        chart_result = (
            focus_dashboard(view_result, selected_candidate)
            if selected_candidate is not None and focus_selected
            else view_result
        )
        renderLightweightCharts(
            build_lightweight_charts(
                chart_result,
                overlays,
                selected_index=selected_index,
                alert_threshold=alert_threshold,
            ),
            key=(
                f"elliott-{selected_database.name}-{timeframe}-{candidate_scope}-"
                f"{selected_index}-{overlays}-{focus_selected}-{alert_threshold}-"
                f"{pattern_view}-{sensitivity}-{live_mode}"
            ),
        )
        selected_lifecycle = (
            candidate_lifecycle(selected_candidate, result.candles)
            if selected_candidate is not None
            else None
        )
        recommendation = trader_recommendation(
            selected_candidate,
            selected_score,
            alert_threshold,
            selected_lifecycle,
        )
        st.markdown(
            f"""
            <div class='terminal-panel' style='margin-top:.35rem'>
                <div style='display:flex;justify-content:space-between;align-items:flex-start;gap:1rem;flex-wrap:wrap'>
                    <div style='min-width:180px'>
                        <div class='terminal-kicker'>Trader-facing mode</div>
                        <div style='font-size:1.2rem;font-weight:750;color:{recommendation.color};margin-top:.15rem'>{html.escape(recommendation.action)}</div>
                        <div style='color:#F4F7FA;font-size:.82rem;margin-top:.15rem'>{html.escape(recommendation.status_text)}</div>
                    </div>
                    <div style='display:grid;grid-template-columns:repeat(3,minmax(120px,1fr));gap:.85rem;flex:1;min-width:300px'>
                        <div>
                            <div class='terminal-kicker'>Entry</div>
                            <div class='terminal-value'>{html.escape(recommendation.entry_text)}</div>
                        </div>
                        <div>
                            <div class='terminal-kicker'>Stop</div>
                            <div class='terminal-value negative'>{html.escape(recommendation.stop_text)}</div>
                        </div>
                        <div>
                            <div class='terminal-kicker'>Target zone</div>
                            <div class='terminal-value positive'>{html.escape(recommendation.target_text)}</div>
                        </div>
                    </div>
                </div>
                <div style='display:grid;grid-template-columns:1fr 1fr;gap:.55rem;margin-top:.65rem'>
                    <div style='color:#F4F7FA;font-size:.8rem'><strong>Reason 1:</strong> {html.escape(recommendation.rationale[0])}</div>
                    <div style='color:#F4F7FA;font-size:.8rem'><strong>Reason 2:</strong> {html.escape(recommendation.rationale[1])}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    history_col, operations_col = st.columns([0.7, 0.3], gap="medium")
    with history_col:
        with st.expander("Candidate history", expanded=False):
            if not scoped_rankings:
                st.info("No candidate history in this scope.")
            else:
                rows = [
                    {
                        "Rank": index + 1,
                        "Pattern": candidate.pattern,
                        "Stage": candidate.status,
                        "Direction": candidate.direction,
                        "Score": score.total,
                        "Lifecycle": candidate_lifecycle(candidate, result.candles),
                        "Completed": _setup_endpoint(candidate)[0].strftime("%Y-%m-%d %H:%M"),
                    }
                    for index, (candidate, score) in enumerate(scoped_rankings)
                ]
                st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", height=280)

    with operations_col:
        with st.expander("Alerts & operations", expanded=False):
            st.markdown(
                "<div class='terminal-kicker'>Telegram Bot Token</div>"
                "<div style='font-size:.78rem;color:#8c96a5;margin:.25rem 0 .7rem'>"
                "Configure <code>TELEGRAM_BOT_TOKEN</code> in <code>.streamlit/secrets.toml</code>. "
                "Tokens are never entered into the browser.</div>",
                unsafe_allow_html=True,
            )
            chat_id = st.text_input("Telegram Chat ID", autocomplete="off")
            alert_threshold = st.slider("Alert Confidence Score", 0, 100, int(alert_threshold), key="alert_threshold")
            st.caption("Log-only webhook shell. No network request is sent.")
            if scoped_rankings:
                candidate, score = scoped_rankings[0]
                alert = format_setup_alert(candidate, score, alert_threshold, chat_id=chat_id)
                alert_key = (candidate.pivots[-1].timestamp.isoformat(), score.total, alert_threshold)
                if alert and st.session_state.get("last_alert_key") != alert_key:
                    log_alert_background(alert)
                    st.session_state["last_alert_key"] = alert_key
                    st.success("Setup written to application logs.")


def _is_tradeable_setup(candidate: WaveCandidate) -> bool:
    """Whether the terminal pivot is the actionable Wave 4 or Wave B."""
    return (
        candidate.status == "EntryReady"
        and bool(candidate.labels)
        and candidate.labels[-1] in {"4", "B"}
    )


def _has_completed_trade_setup(candidate: WaveCandidate) -> bool:
    """Backward-compatible alias for the shared tradeability predicate."""
    return _is_tradeable_setup(candidate)


def scanner_candidate_row(
    *,
    market: str,
    database_name: str | None = None,
    timeframe: str,
    candidate: WaveCandidate,
    score: ConfidenceScore,
    candles: pd.DataFrame,
    setup_quality_threshold: float,
    risk_policy: RiskPolicy | None,
    friction: Friction | None,
    current_open_risk: float = 0.0,
    realized_daily_loss: float = 0.0,
    evaluate_risk: bool = True,
) -> dict[str, object]:
    """Classify one scanner row with the same structure and risk gates as the terminal."""
    lifecycle = candidate_lifecycle(candidate, candles)
    direction = "BUY" if candidate.direction == "Bullish" else "SELL"
    target_low, target_high = target_zone(candidate)
    risk_reward: float | str = "Not evaluated"

    if candidate.status == "Forming":
        trade_decision, structure_status, reason = (
            "WATCH", "WATCHLIST", "Forming structure"
        )
    elif lifecycle == "Invalidated":
        trade_decision, structure_status, reason = (
            "NO TRADE", "HISTORICAL", "Invalidated"
        )
    elif lifecycle == "Target hit":
        trade_decision, structure_status, reason = (
            "NO TRADE", "HISTORICAL", "Target hit"
        )
    elif candidate.status == "Completed":
        trade_decision, structure_status, reason = (
            "NO TRADE", "HISTORICAL", "Completed historical structure"
        )
    elif not _is_tradeable_setup(candidate):
        trade_decision, structure_status, reason = (
            "NO TRADE", "HISTORICAL", "Not at tradeable Wave 4/B"
        )
    else:
        structure_status = f"{direction} SETUP"
        if score.total < setup_quality_threshold:
            trade_decision, reason = "WATCH", "Quality below threshold"
        elif not evaluate_risk:
            trade_decision, reason = "WATCH", "Needs risk check"
        else:
            decision = build_decision_state(
                candidate,
                score,
                setup_quality_threshold,
                lifecycle,
                candles,
                risk_policy,
                friction,
                current_open_risk=current_open_risk,
                realized_daily_loss=realized_daily_loss,
            )
            risk_reward = (
                decision.reward_risk
                if decision.reward_risk is not None
                else "Not evaluated"
            )
            if decision.status == "TRADE READY":
                trade_decision, reason = (
                    "TRADE READY", "Quality passed and risk approved"
                )
            elif decision.status == "BLOCKED":
                trade_decision = "BLOCKED"
                reason = f"Risk failed: {decision.risk_reason}"
            else:
                trade_decision = decision.status
                reason = decision.reason

    resolved_database = database_name or f"{market}.db"
    signature = candidate_signature(candidate)
    return ScannerRow(
        candidate_key=scanner_candidate_key(resolved_database, timeframe, candidate),
        pivot_signature=json.dumps(
            signature, sort_keys=True, separators=(",", ":")
        ),
        database_name=resolved_database,
        market=market,
        timeframe=timeframe,
        pattern=candidate.pattern,
        direction=direction,
        trade_decision=trade_decision,
        structure_status=structure_status,
        setup_stage=candidate.status,
        current_wave=current_wave_label(candidate),
        reason=reason,
        setup_quality_score=float(score.total),
        risk_reward=risk_reward,
        invalidation=float(candidate.invalidation_level),
        target_zone=f"{target_low:,.5f} - {target_high:,.5f}",
    ).as_record()


def scan_global_markets(
    databases: tuple[Path, ...],
    atr_multiplier: float,
    atr_period: int,
    setup_quality_threshold: float = 70.0,
    risk_policy: RiskPolicy | None = None,
    friction: Friction | None = None,
    current_open_risk: float = 0.0,
    realized_daily_loss: float = 0.0,
    evaluate_risk: bool = True,
    timeframes: tuple[str, ...] | None = None,
    candidate_limit: int = 3,
    progress_callback: Callable[[int, int, str, str, int], None] | None = None,
    dataset_loader: Callable[[Path], pd.DataFrame] | None = None,
    timeframe_resampler: Callable[[pd.DataFrame, str, Path], pd.DataFrame] | None = None,
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    """Load each market once, then scan selected timeframes from local M5 data."""
    rows: list[dict[str, object]] = []
    errors: list[str] = []
    selected_timeframes = tuple(timeframes or TIMEFRAMES)
    cached_datasets_reused = 0
    load_dataset = dataset_loader or load_m5
    resample_dataset = timeframe_resampler or resample_m5
    handled_errors = (
        ValueError,
        OSError,
        sqlite3.DatabaseError,
        pd.errors.DatabaseError,
        ZeroDivisionError,
    )
    for market_index, database in enumerate(databases, start=1):
        try:
            m5 = load_dataset(database)
        except handled_errors as error:
            errors.append(f"{database.name}: {error}")
            continue
        for timeframe_index, timeframe in enumerate(selected_timeframes):
            if progress_callback is not None:
                progress_callback(
                    market_index,
                    len(databases),
                    database.stem,
                    timeframe,
                    cached_datasets_reused,
                )
            try:
                candles = resample_dataset(m5, timeframe, database)
                result = compute_dashboard_from_candles(
                    candles, atr_multiplier, atr_period
                )
                if timeframe_index:
                    cached_datasets_reused += 1
            except handled_errors as error:
                errors.append(f"{database.name} {timeframe}: {error}")
                continue
            for candidate, score in result.rankings[: max(1, candidate_limit)]:
                rows.append(
                    scanner_candidate_row(
                        market=database.stem,
                        database_name=database.name,
                        timeframe=timeframe,
                        candidate=candidate,
                        score=score,
                        candles=result.candles,
                        setup_quality_threshold=setup_quality_threshold,
                        risk_policy=risk_policy,
                        friction=friction,
                        current_open_risk=current_open_risk,
                        realized_daily_loss=realized_daily_loss,
                        evaluate_risk=evaluate_risk,
                    )
                )
    columns = (
        "Trade Decision",
        "Structure Status",
        "Direction",
        "Reason",
        "Setup Quality Score",
        "Risk/Reward",
        "Market",
        "Timeframe",
        "Pattern",
        "Current Wave",
        "Invalidation",
        "Target Zone",
        "Database Name",
        "Setup Stage",
        "Candidate Key",
        "Pivot Signature",
    )
    frame = pd.DataFrame(rows, columns=columns)
    if not frame.empty:
        decision_order = {
            "TRADE READY": 0, "BLOCKED": 1, "WATCH": 2, "NO TRADE": 3
        }
        frame["_decision_order"] = frame["Trade Decision"].map(decision_order)
        frame = (
            frame.sort_values(
                ["_decision_order", "Setup Quality Score"],
                ascending=[True, False],
                kind="stable",
            )
            .drop(columns="_decision_order")
            .reset_index(drop=True)
        )
    return frame, tuple(errors)


def _render_global_scanner_legacy() -> None:
    import streamlit as st

    st.markdown("## Global Market Scanner")
    st.caption(
        "Batch scan of every local market across every registered timeframe. "
        "Shows Forming watch candidates and confirmed EntryReady Wave 4/B setups."
    )
    sensitivity = st.session_state.get("sensitivity_preset", "Balanced")
    override_enabled = bool(st.session_state.get("advanced_atr_override", False))
    atr_multiplier, atr_period = resolve_sensitivity(
        sensitivity,
        override_enabled=override_enabled,
        atr_multiplier=float(st.session_state.get("atr_multiplier", 2.0)),
        atr_period=int(st.session_state.get("atr_period", 14)),
    )
    databases = discover_databases()
    setup_quality_threshold = float(
        st.session_state.get("resolved_setup_quality_threshold", 70.0)
    )
    risk_policy = st.session_state.get(
        "active_risk_policy", RiskPolicy(account_equity=100_000)
    )
    friction = st.session_state.get("active_friction", Friction())
    current_open_risk = float(
        st.session_state.get("active_current_open_risk", 0.0)
    )
    realized_daily_loss = float(
        st.session_state.get("active_realized_daily_loss", 0.0)
    )
    evaluate_risk = st.checkbox(
        "Evaluate risk in scanner",
        value=True,
        help=(
            "Uses the active Risk Settings. Disable to classify quality-passed "
            "setups as WATCH until risk is checked."
        ),
    )

    status_col, action_col = st.columns([0.75, 0.25], vertical_alignment="center")
    with status_col:
        st.markdown(
            f"<div class='terminal-panel'><span class='terminal-kicker'>Scan universe</span>"
            f"<div class='terminal-value'>{len(databases)} markets | {len(TIMEFRAMES)} timeframes | "
            f"ATR {atr_period} x {atr_multiplier:.1f}</div></div>",
            unsafe_allow_html=True,
        )
    with action_col:
        run_scan = st.button("Run market scan", type="primary", width="stretch", disabled=not databases)

    if not databases:
        st.info("No local asset databases were detected.")
        return
    if run_scan:
        with st.spinner("Scanning markets..."):
            frame, errors = scan_global_markets(
                databases,
                atr_multiplier,
                atr_period,
                setup_quality_threshold,
                risk_policy,
                friction,
                current_open_risk,
                realized_daily_loss,
                evaluate_risk,
            )
        st.session_state["scanner_frame"] = frame
        st.session_state["scanner_errors"] = errors

    frame = st.session_state.get("scanner_frame")
    errors = st.session_state.get("scanner_errors", ())
    if frame is None:
        st.info("Run the scanner to calculate the current global opportunity set.")
        return
    if frame.empty:
        st.warning("No active Wave 4 or Wave B trade setups were found.")
    else:
        counts = frame["Trade Decision"].value_counts()
        summary_cols = st.columns(4)
        summary_cols[0].metric("Trade ready", int(counts.get("TRADE READY", 0)))
        summary_cols[1].metric("Blocked", int(counts.get("BLOCKED", 0)))
        summary_cols[2].metric("Watch", int(counts.get("WATCH", 0)))
        summary_cols[3].metric("No trade", int(counts.get("NO TRADE", 0)))
        st.dataframe(
            frame,
            hide_index=True,
            width="stretch",
            height=min(680, 44 + 35 * len(frame)),
            column_config={
                "Inspect": st.column_config.LinkColumn(
                    "Inspect",
                    display_text="▶",
                    help="Open this setup in Single Chart Terminal",
                    width="small",
                ),
                "Setup Quality Score": st.column_config.NumberColumn(format="%.2f"),
                "Invalidation": st.column_config.NumberColumn(format="%.5f"),
                "Database Name": None,
                "Setup Stage": None,
                "Candidate Key": None,
                "Pivot Signature": None,
            },
        )
        st.caption(
            f"{len(frame)} active setups | setup-quality threshold "
            f"{setup_quality_threshold:.1f} | sorted by Setup Quality Score"
        )
    if errors:
        with st.expander(f"Skipped inputs ({len(errors)})", expanded=False):
            for error in errors:
                st.caption(error)


def _render_global_scanner() -> None:
    import streamlit as st

    st.markdown("## Global Market Scanner")
    st.caption("Persistent opportunity list built from local SQLite data.")
    sensitivity = st.session_state.get("sensitivity_preset", "Balanced")
    atr_multiplier, atr_period = resolve_sensitivity(
        sensitivity,
        override_enabled=bool(st.session_state.get("advanced_atr_override", False)),
        atr_multiplier=float(st.session_state.get("atr_multiplier", 2.0)),
        atr_period=int(st.session_state.get("atr_period", 14)),
    )
    databases = discover_databases()
    quality_threshold = float(
        st.session_state.get("resolved_setup_quality_threshold", 70.0)
    )
    risk_policy = st.session_state.get(
        "active_risk_policy", RiskPolicy(account_equity=100_000)
    )
    friction = st.session_state.get("active_friction", Friction())
    current_open_risk = float(st.session_state.get("active_current_open_risk", 0.0))
    realized_daily_loss = float(
        st.session_state.get("active_realized_daily_loss", 0.0)
    )

    control_cols = st.columns([1.1, 1.55, 0.7, 0.95, 1.0])
    with control_cols[0]:
        market_scope = st.selectbox(
            "Markets to scan", ("All", "Indian only", "Selected watchlist")
        )
    with control_cols[1]:
        selected_timeframes = tuple(
            st.multiselect(
                "Timeframes to scan",
                tuple(TIMEFRAMES),
                default=("15M", "1H", "4H", "1D"),
            )
        )
    with control_cols[2]:
        candidate_limit = int(st.number_input("Candidate limit", 1, 10, 3, 1))
    with control_cols[3]:
        evaluate_risk = st.checkbox(
            "Evaluate risk in scanner",
            value=False,
            help="Optional. Uses Risk Settings and takes more time.",
        )
    with control_cols[4]:
        scanner_layout = st.selectbox(
            "Scanner layout",
            ("Desktop table", "Mobile cards"),
            help="Use Mobile cards on phones or whenever the scanner table feels crowded.",
        )

    indian_names = {
        "BHARTI_AIRTEL", "HDFC_BANK", "ICICI_BANK", "INFOSYS",
        "LARSEN_TOUBRO", "NIFTY_50", "NIFTY_BANK", "RELIANCE", "SBI", "TCS",
    }
    if market_scope == "Indian only":
        scan_databases = tuple(db for db in databases if db.stem in indian_names)
    elif market_scope == "Selected watchlist":
        names = [db.stem for db in databases]
        watchlist = st.multiselect(
            "Selected watchlist", names, default=names[: min(5, len(names))]
        )
        scan_databases = tuple(db for db in databases if db.stem in watchlist)
    else:
        scan_databases = databases

    universe_signature = tuple(
        (db.name, db.stat().st_size, db.stat().st_mtime_ns) for db in scan_databases
    )
    scan_settings = {
        "market_scope": market_scope,
        "databases": tuple(db.name for db in scan_databases),
        "universe_signature": universe_signature,
        "timeframes": selected_timeframes,
        "candidate_limit": candidate_limit,
        "evaluate_risk": evaluate_risk,
        "atr_multiplier": atr_multiplier,
        "atr_period": atr_period,
        "setup_quality_threshold": quality_threshold,
        "risk_policy": repr(risk_policy),
        "friction": repr(friction),
        "current_open_risk": current_open_risk,
        "realized_daily_loss": realized_daily_loss,
    }
    previous_settings = st.session_state.get("last_scanner_settings")
    if (
        previous_settings is not None
        and not scanner_cache_is_compatible(previous_settings, scan_settings)
        and "last_scanner_results" in st.session_state
    ):
        for key in ("last_scanner_results", "last_scanner_summary", "last_scan_timestamp"):
            st.session_state.pop(key, None)
        st.info("Scanner settings changed. Run the scan to calculate new results.")

    @st.cache_data(show_spinner=False)
    def cached_market_m5(path: str, size: int, modified: int) -> pd.DataFrame:
        del size, modified
        return load_m5(path)

    @st.cache_data(show_spinner=False)
    def cached_market_timeframe(
        path: str, size: int, modified: int, timeframe: str
    ) -> pd.DataFrame:
        return resample_m5(
            cached_market_m5(path, size, modified), timeframe, path
        )

    def dataset_loader(database: Path) -> pd.DataFrame:
        stat = database.stat()
        return cached_market_m5(str(database), stat.st_size, stat.st_mtime_ns)

    def timeframe_resampler(
        _m5: pd.DataFrame, timeframe: str, database: Path
    ) -> pd.DataFrame:
        stat = database.stat()
        return cached_market_timeframe(
            str(database), stat.st_size, stat.st_mtime_ns, timeframe
        )

    status_col, run_col, clear_col = st.columns(
        [0.66, 0.22, 0.12], vertical_alignment="center"
    )
    with status_col:
        st.markdown(
            f"<div class='terminal-panel'><span class='terminal-kicker'>Scan universe</span>"
            f"<div class='terminal-value'>{len(scan_databases)} markets | "
            f"{len(selected_timeframes)} timeframes | ATR {atr_period} x "
            f"{atr_multiplier:.1f}</div></div>",
            unsafe_allow_html=True,
        )
    with run_col:
        run_scan = st.button(
            "Run market scan",
            type="primary",
            width="stretch",
            disabled=not scan_databases or not selected_timeframes,
        )
    with clear_col:
        clear_scan = st.button(
            "Clear scan",
            width="stretch",
            disabled="last_scanner_results" not in st.session_state,
        )
    if not databases:
        st.info("No local asset databases were detected.")
        return
    if clear_scan:
        for key in (
            "last_scanner_results", "last_scanner_summary",
            "last_scanner_settings", "last_scan_timestamp",
            "scanner_frame", "scanner_errors",
        ):
            st.session_state.pop(key, None)
        st.rerun()

    if run_scan:
        progress = st.empty()
        started = time.perf_counter()

        def update_progress(
            market_index: int,
            market_count: int,
            market: str,
            timeframe: str,
            reused: int,
        ) -> None:
            progress.info(
                f"Scanning {market_index} / {market_count} markets...\n\n"
                f"Current: {market} | {timeframe}\n\n"
                f"Elapsed time: {time.perf_counter() - started:.1f}s | "
                f"Cached datasets reused: {reused}"
            )

        frame, errors = scan_global_markets(
            scan_databases,
            atr_multiplier,
            atr_period,
            quality_threshold,
            risk_policy,
            friction,
            current_open_risk,
            realized_daily_loss,
            evaluate_risk,
            timeframes=selected_timeframes,
            candidate_limit=candidate_limit,
            progress_callback=update_progress,
            dataset_loader=dataset_loader,
            timeframe_resampler=timeframe_resampler,
        )
        progress.success(
            f"Scan complete in {time.perf_counter() - started:.1f} seconds."
        )
        save_scanner_results(
            st.session_state,
            frame,
            errors,
            scan_settings,
            pd.Timestamp.now(tz="Asia/Kolkata"),
        )

    frame = st.session_state.get("last_scanner_results")
    errors = st.session_state.get("scanner_errors", ())
    if frame is None:
        st.info("Run the scanner to calculate the current opportunity list.")
        return
    timestamp = st.session_state.get("last_scan_timestamp")
    if timestamp:
        st.caption(
            f"Showing cached scan from {pd.Timestamp(timestamp).strftime('%H:%M:%S')}. "
            "Click Run market scan to refresh."
        )
    if frame.empty:
        st.warning("No setups were found with the selected settings.")
    else:
        counts = st.session_state.get(
            "last_scanner_summary",
            frame["Trade Decision"].value_counts().to_dict(),
        )
        metrics = st.columns(4)
        for column, label, value in zip(
            metrics,
            ("Trade ready", "Blocked", "Watch", "No trade"),
            ("TRADE READY", "BLOCKED", "WATCH", "NO TRADE"),
        ):
            column.metric(label, int(counts.get(value, 0)))

        if scanner_layout == "Desktop table":
            widths = [1.2, 1.3, 0.6, 0.8, 1.4, 0.6, 0.7, 0.35]
            labels = ("Decision", "Market", "Frame", "Pattern", "Wave", "Score", "R/R", "")
            for column, label in zip(st.columns(widths), labels):
                column.markdown(f"**{label}**")
            for row_index, (_, row) in enumerate(frame.iterrows()):
                columns = st.columns(widths)
                values = (
                    row["Trade Decision"], row["Market"], row["Timeframe"],
                    row["Pattern"], row["Current Wave"],
                    f"{float(row['Setup Quality Score']):.1f}", row["Risk/Reward"],
                )
                for column, value in zip(columns[:7], values):
                    column.write(value)
                inspect_key = scanner_inspect_button_key(row, row_index)
                if columns[7].button(
                    "▶", key=inspect_key,
                    help="Open this exact setup in Single Chart Terminal",
                ):
                    store_scanner_setup(st.session_state, row)
                    st.rerun()
        else:
            for row_index, (_, row) in enumerate(frame.iterrows()):
                decision = str(row["Trade Decision"])
                decision_class = (
                    "positive"
                    if decision == "TRADE READY"
                    else "negative"
                    if decision == "BLOCKED"
                    else ""
                )
                risk_reward = str(row["Risk/Reward"])
                reason = str(row["Reason"])
                st.markdown(
                    f"""
                    <div class="scanner-mobile-card">
                        <div class="scanner-mobile-topline">
                            <div class="scanner-mobile-market">{html.escape(str(row["Market"]))}</div>
                            <div class="scanner-mobile-decision {decision_class}">{html.escape(decision)}</div>
                        </div>
                        <div class="scanner-mobile-meta">
                            <div class="scanner-mobile-field">
                                <div class="scanner-mobile-label">Timeframe</div>
                                <div class="scanner-mobile-value">{html.escape(str(row["Timeframe"]))}</div>
                            </div>
                            <div class="scanner-mobile-field">
                                <div class="scanner-mobile-label">Pattern</div>
                                <div class="scanner-mobile-value">{html.escape(str(row["Pattern"]))}</div>
                            </div>
                            <div class="scanner-mobile-field">
                                <div class="scanner-mobile-label">Current wave</div>
                                <div class="scanner-mobile-value">{html.escape(str(row["Current Wave"]))}</div>
                            </div>
                            <div class="scanner-mobile-field">
                                <div class="scanner-mobile-label">Score</div>
                                <div class="scanner-mobile-value">{float(row["Setup Quality Score"]):.1f}</div>
                            </div>
                            <div class="scanner-mobile-field">
                                <div class="scanner-mobile-label">R/R</div>
                                <div class="scanner-mobile-value">{html.escape(risk_reward if risk_reward != 'Not evaluated' else 'Pending')}</div>
                            </div>
                            <div class="scanner-mobile-field">
                                <div class="scanner-mobile-label">Direction</div>
                                <div class="scanner-mobile-value">{html.escape(str(row["Direction"]))}</div>
                            </div>
                        </div>
                        <div class="scanner-mobile-reason">
                            <div class="scanner-mobile-label">Reason</div>
                            <div class="scanner-mobile-value">{html.escape(reason)}</div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                if st.button(
                    "Open chart",
                    key=f"{scanner_inspect_button_key(row, row_index)}_mobile",
                    help="Open this exact setup in Single Chart Terminal",
                    use_container_width=True,
                ):
                    store_scanner_setup(st.session_state, row)
                    st.rerun()

        st.caption(
            f"{len(frame)} setups | setup-quality threshold "
            f"{quality_threshold:.1f} | sorted by decision and score"
        )
    if errors:
        with st.expander(f"Skipped inputs ({len(errors)})", expanded=False):
            for error in errors:
                st.caption(error)


def main() -> None:
    import streamlit as st

    st.set_page_config(
        page_title="Deterministic Elliott Wave DSS",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    _inject_terminal_css(st)
    inspect_candidate = st.query_params.get("inspect_candidate")
    if inspect_candidate:
        scanner_frame = st.session_state.get("scanner_frame")
        matching_row = None
        if isinstance(scanner_frame, pd.DataFrame) and not scanner_frame.empty:
            matches = scanner_frame[
                scanner_frame["Candidate Key"].astype(str)
                == str(inspect_candidate)
            ]
            if not matches.empty:
                matching_row = matches.iloc[0]
        if matching_row is not None:
            store_scanner_setup(st.session_state, matching_row)
        else:
            st.session_state["navigation_notice"] = (
                "This scanner setup is no longer available. "
                "Re-run the scanner and try again."
            )
        del st.query_params["inspect_candidate"]
    requested_tab = st.session_state.pop("requested_active_tab", None)
    if requested_tab in {"single_chart", "scanner"}:
        st.session_state["active_tab"] = requested_tab
    active_tab = st.radio(
        "Terminal view",
        ("single_chart", "scanner"),
        format_func=lambda value: (
            "Single Chart Terminal"
            if value == "single_chart"
            else "Global Market Scanner"
        ),
        horizontal=True,
        label_visibility="collapsed",
        key="active_tab",
    )
    navigation_notice = st.session_state.pop("navigation_notice", None)
    if navigation_notice:
        st.toast(str(navigation_notice), icon="⚠️")
    if active_tab == "single_chart":
        _render_single_chart()
    else:
        _render_global_scanner()


if __name__ == "__main__":
    main()
