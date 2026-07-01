"""Streamlit dashboard for deterministic Elliott Wave decision support.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import html
import logging
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from db import TIMEFRAMES, resample_ohlcv
from engine import WaveCandidate, build_candidates
from pivots import Pivot, extract_pivots
from scoring import ConfidenceScore, calculate_rsi, score_candidates

LOGGER = logging.getLogger("elliott_dashboard")
COLORS = ("#00D4FF", "#FFB000", "#D65CFF")


@dataclass(frozen=True, slots=True)
class DashboardResult:
    candles: pd.DataFrame
    pivots: tuple[Pivot, ...]
    rankings: tuple[tuple[WaveCandidate, ConfidenceScore], ...]
    rsi: pd.Series


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


def compute_dashboard(
    database: str | Path,
    timeframe: str,
    atr_multiplier: float,
    atr_period: int = 14,
) -> DashboardResult:
    """Load, extract, validate, and rank the current chart state."""
    candles = resample_ohlcv(database, timeframe.upper())  # type: ignore[arg-type]
    empty_rsi = pd.Series(index=candles.index, dtype=float, name="rsi")
    if candles.empty or len(candles) < atr_period:
        return DashboardResult(candles, (), (), empty_rsi)

    pivots = tuple(
        extract_pivots(candles, atr_multiplier, atr_period=atr_period)
    )
    candidates = build_candidates(pivots)
    scores = score_candidates(candidates, candles)
    rankings = tuple(
        sorted(scores.items(), key=lambda item: item[1].total, reverse=True)
    )
    return DashboardResult(
        candles=candles,
        pivots=pivots,
        rankings=rankings,
        rsi=calculate_rsi(candles),
    )


def target_zone(candidate: WaveCandidate) -> tuple[float, float]:
    """Return deterministic lower/upper Fibonacci target-zone bounds."""
    sign = 1.0 if candidate.direction == "Bullish" else -1.0
    first_leg = abs(candidate.pivots[1].price - candidate.pivots[0].price)
    if candidate.pattern == "Impulse":
        anchor = candidate.pivots[4].price
        projections = (anchor + sign * 1.0 * first_leg, anchor + sign * 1.618 * first_leg)
    else:
        anchor = candidate.pivots[2].price
        projections = (anchor + sign * 1.0 * first_leg, anchor + sign * 1.618 * first_leg)
    return min(projections), max(projections)


def format_setup_alert(
    candidate: WaveCandidate,
    score: ConfidenceScore,
    threshold: float,
    *,
    chat_id: str,
) -> str | None:
    """Build an alert only for a newly completed Wave 4 or Wave B setup."""
    if (
        score.total < threshold
        or not candidate.labels
        or candidate.labels[-1] not in {"4", "B"}
    ):
        return None
    low, high = target_zone(candidate)
    return (
        f"Elliott setup | chat={chat_id or 'unset'} | "
        f"{candidate.pattern} {candidate.direction} | "
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
        price_series.append(
            {
                "type": "Line",
                "data": [
                    {"time": _chart_time(pivot.timestamp), "value": pivot.price}
                    for pivot in candidate.pivots
                ],
                "options": {
                    "color": color,
                    "lineWidth": 3 if rank - 1 == selected_index else 1,
                    "priceLineVisible": False,
                    "lastValueVisible": False,
                    "title": "",
                },
                "markers": [
                    {
                        "time": _chart_time(pivot.timestamp),
                        "position": "aboveBar"
                        if pivot.type == "High"
                        else "belowBar",
                        "color": color,
                        "shape": "circle",
                        "text": label,
                    }
                    for label, pivot in candidate.labeled_waves
                ] if rank - 1 == selected_index else [],
            }
        )

    if result.rankings:
        selected_index = min(selected_index, len(result.rankings) - 1)
        selected_candidate = result.rankings[selected_index][0]
        setup_index = 4 if selected_candidate.pattern == "Impulse" else 2
        first_time = _chart_time(selected_candidate.pivots[setup_index].timestamp)
        last_time = _chart_time(candles.index[-1])
        target_low, target_high = target_zone(selected_candidate)
        for price, color, title in (
            (selected_candidate.invalidation_level, "#F05D68", "Invalidation"),
            (target_low, "#18C98B", "Target 1.000"),
            (target_high, "#00BFA5", "Target 1.618"),
        ):
            price_series.append(
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
                        "lastValueVisible": False,
                        "title": "",
                    },
                }
            )

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
        return threshold_text, "Awaiting Wave 4/B", "Awaiting Wave 4/B"

    candidate, score = tradeable[0]
    if score.total < alert_threshold:
        message = f"Below threshold ({score.total:.1f})"
        return threshold_text, message, message

    entry = f"{candidate.pivots[-1].price:,.2f}"
    if candidate.direction == "Bullish":
        return threshold_text, entry, "No bearish setup"
    return threshold_text, "No bullish setup", entry


def system_hints(
    candidate: WaveCandidate,
    score: ConfidenceScore,
    alert_threshold: float,
    lifecycle: str,
) -> tuple[str, ...]:
    """Explain every gate controlling an actionable chart marker."""
    terminal = candidate.labels[-1] if candidate.labels else "unknown"
    target_low, target_high = target_zone(candidate)
    invalidation_kind = (
        "floor" if candidate.invalidation_side == "below" else "ceiling"
    )
    confidence = (
        f"Confidence gate: Passed ({score.total:.1f} ≥ {alert_threshold:.1f})"
        if score.total >= alert_threshold
        else f"Confidence gate: Below threshold "
        f"({score.total:.1f} < {alert_threshold:.1f})"
    )
    hints = [
        f"Pattern state: {candidate.direction} {candidate.pattern} "
        f"terminates at Wave {terminal}",
        confidence,
        f"Lifecycle: {lifecycle}",
    ]
    if lifecycle != "Active":
        hints.extend(
            (
                f"Entry gate: Failed — lifecycle is {lifecycle.lower()}",
                f"Marker decision: Hidden — setup {lifecycle.lower()}",
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
                f"Entry gate: Failed — candidate terminates at Wave "
                f"{terminal}, not Wave {required}",
                "Marker decision: Hidden — no entry setup",
                f"Next required event: Wait for a new active Wave {required} "
                "termination",
                "Trading interpretation: Valid analytical structure; "
                "not a current trade signal",
            )
        )
    elif score.total < alert_threshold:
        hints.extend(
            (
                "Entry gate: Passed — terminal Wave 4/B is present",
                "Marker decision: Hidden — confidence gate not met",
                f"Next required event: Confidence must reach "
                f"{alert_threshold:.1f}",
                "Trading interpretation: Structurally actionable but "
                "insufficiently ranked",
            )
        )
    else:
        side = "Buy" if candidate.direction == "Bullish" else "Sell"
        hints.extend(
            (
                "Entry gate: Passed — terminal Wave 4/B is present",
                f"Marker decision: Visible — {side} at "
                f"{candidate.pivots[-1].price:,.2f}",
                "Next required event: Monitor target and invalidation",
                f"Trading interpretation: Actionable {side.lower()} setup",
            )
        )
    hints.extend(
        (
            f"Invalidation reference: {candidate.invalidation_level:,.2f} "
            f"{invalidation_kind}",
            f"Target zone: {target_low:,.2f}–{target_high:,.2f}",
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
        f"#{index + 1} | {candidate.pattern} | {candidate.direction} | "
        f"{score.total:.1f}/100"
    )


def recent_rankings(
    rankings: tuple[tuple[WaveCandidate, ConfidenceScore], ...],
    latest_timestamp: pd.Timestamp,
    days: int = 30,
) -> tuple[tuple[WaveCandidate, ConfidenceScore], ...]:
    """Limit the working set to paths ending near the current market edge."""
    cutoff = pd.Timestamp(latest_timestamp) - pd.Timedelta(days=days)
    return tuple(
        item for item in rankings if item[0].pivots[-1].timestamp >= cutoff
    )


def candidate_lifecycle(
    candidate: WaveCandidate, candles: pd.DataFrame
) -> str:
    """Classify a completed path using only candles after its final pivot."""
    future = candles.loc[candles.index > candidate.pivots[-1].timestamp]
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


def actionable_rankings(
    rankings: tuple[tuple[WaveCandidate, ConfidenceScore], ...],
    candles: pd.DataFrame,
) -> tuple[tuple[WaveCandidate, ConfidenceScore], ...]:
    return tuple(
        item
        for item in rankings
        if candidate_lifecycle(item[0], candles) == "Active"
    )


def focus_dashboard(
    result: DashboardResult,
    candidate: WaveCandidate,
    padding_bars: int = 18,
) -> DashboardResult:
    """Return a chart-only window centered on the selected structure."""
    index = result.candles.index
    start_position = int(index.searchsorted(candidate.pivots[0].timestamp))
    end_position = int(index.searchsorted(candidate.pivots[-1].timestamp, side="right"))
    start = max(0, start_position - padding_bars)
    end = min(len(index), end_position + padding_bars)
    candles = result.candles.iloc[start:end]
    return DashboardResult(
        candles=candles,
        pivots=result.pivots,
        rankings=result.rankings,
        rsi=result.rsi.reindex(candles.index),
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
        iframe { border-radius: 8px; }
        [data-testid="stDataFrame"] {
            border: 1px solid #202632; border-radius: 8px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_single_chart() -> None:
    import streamlit as st
    from streamlit_lightweight_charts import renderLightweightCharts

    databases = discover_databases()
    title_col, status_col = st.columns([0.75, 0.25], vertical_alignment="center")
    with title_col:
        st.title("Elliott Wave Terminal")
        st.caption("Deterministic structure · volatility-adjusted pivots · causal scoring")
    with status_col:
        st.markdown(
            "<div style='text-align:right;color:#18c98b;font-size:.78rem;"
            "font-weight:650'>● SYSTEM ONLINE</div>",
            unsafe_allow_html=True,
        )

    (
        selector_col,
        timeframe_col,
        pattern_col,
        multiplier_col,
        period_col,
        universe_col,
    ) = st.columns(
        [2.0, 0.7, 1.35, 1.05, 0.9, 1.15], vertical_alignment="bottom"
    )
    with selector_col:
        selected_database = st.selectbox(
            "Market",
            databases,
            format_func=lambda path: path.name,
            disabled=not databases,
        )
    with timeframe_col:
        timeframe_options = tuple(TIMEFRAMES)
        timeframe = st.selectbox(
            "Timeframe",
            timeframe_options,
            index=timeframe_options.index("1H"),
        )
    with pattern_col:
        pattern_view = st.selectbox(
            "Pattern View",
            ("Balanced · 1–5 + ABC", "Impulse · 1–5", "ZigZag · ABC"),
        )
    with multiplier_col:
        atr_multiplier = st.slider(
            "ATR Multiplier", 1.5, 4.0, 2.0, 0.1, key="atr_multiplier"
        )
    with period_col:
        atr_period = st.slider(
            "ATR Period", 5, 50, 14, 1, key="atr_period"
        )
    with universe_col:
        candidate_scope = st.selectbox(
            "Candidate Scope", ("Actionable", "Recent · 30D", "All history")
        )

    if not databases or selected_database is None:
        st.info("Place a .db, .sqlite, or .sqlite3 asset database beside app.py.")
        return

    try:
        result = compute_dashboard(
            selected_database, timeframe, atr_multiplier, atr_period
        )
    except (ValueError, OSError) as error:
        st.error(f"Unable to calculate dashboard: {error}")
        return

    if result.candles.empty:
        st.warning("No complete candles are available for this timeframe.")
        return

    if candidate_scope == "Actionable":
        scoped_rankings = actionable_rankings(result.rankings, result.candles)
    elif candidate_scope.startswith("Recent"):
        scoped_rankings = recent_rankings(
            result.rankings, result.candles.index[-1]
        )
    else:
        scoped_rankings = result.rankings
    scoped_rankings = pattern_rankings(scoped_rankings, pattern_view)
    view_result = DashboardResult(
        result.candles, result.pivots, scoped_rankings, result.rsi
    )

    last_close = float(result.candles.iloc[-1]["close"])
    previous_close = (
        float(result.candles.iloc[-2]["close"])
        if len(result.candles) > 1
        else last_close
    )
    change = last_close - previous_close
    change_percent = 100 * change / previous_close if previous_close else 0.0
    metric_cols = st.columns(4)
    metric_cols[0].metric("LAST", f"{last_close:,.2f}", f"{change_percent:+.2f}%")
    metric_cols[1].metric("TIMEFRAME", timeframe)
    impulses = sum(
        candidate.pattern == "Impulse" for candidate, _score in scoped_rankings
    )
    zigzags = sum(
        candidate.pattern == "ZigZag" for candidate, _score in scoped_rankings
    )
    metric_cols[2].metric(
        "VALID PATHS", len(scoped_rankings), f"{impulses} impulse · {zigzags} ABC"
    )
    metric_cols[3].metric(
        "DATA THROUGH", result.candles.index[-1].strftime("%d %b · %H:%M UTC")
    )

    alert_threshold = float(st.session_state.get("alert_threshold", 75))
    chart_col, inspector_col = st.columns([0.78, 0.22], gap="medium")
    top_rankings = scoped_rankings[:3]
    with inspector_col:
        st.markdown("### Structure Inspector")
        if not top_rankings:
            selected_index = 0
            overlays = (False, False, False)
            focus_selected = False
            st.info(f"No valid {pattern_view.split(' · ')[0]} candidates in this scope.")
        else:
            selected_index = st.selectbox(
                "Focused path",
                range(len(top_rankings)),
                format_func=lambda index: _candidate_name(
                    index, *top_rankings[index]
                ),
                label_visibility="collapsed",
            )
            overlay_values_list: list[bool] = []
            for index, (candidate, _score) in enumerate(top_rankings):
                if index == selected_index:
                    st.caption(f"Focused overlay: #{index + 1} {candidate.pattern}")
                    overlay_values_list.append(True)
                else:
                    overlay_values_list.append(
                        st.checkbox(
                            f"Show alternative #{index + 1}",
                            value=False,
                            key=(
                                f"overlay-{index}-{timeframe}-"
                                f"{selected_database.name}"
                            ),
                        )
                    )
            overlay_values = tuple(overlay_values_list)
            overlays = tuple(
                overlay_values[index] if index < len(overlay_values) else False
                for index in range(3)
            )
            candidate, score = top_rankings[selected_index]
            lifecycle = candidate_lifecycle(candidate, result.candles)
            low_target, high_target = target_zone(candidate)
            boundary = (
                "Floor" if candidate.invalidation_side == "below" else "Ceiling"
            )
            direction_class = (
                "positive" if candidate.direction == "Bullish" else "negative"
            )
            st.markdown(
                f"""
                <div class="terminal-panel">
                  <div class="terminal-kicker">Selected structure</div>
                  <div class="terminal-value {direction_class}">
                    {candidate.pattern} · {candidate.direction}
                  </div>
                  <div style="color:#8c96a5;font-size:.7rem;margin-top:.2rem">
                    Lifecycle: {lifecycle}
                  </div>
                </div>
                <div class="terminal-panel">
                  <div class="terminal-kicker">Confidence Score</div>
                  <div class="terminal-value">{score.total:.1f}
                    <span style="color:#6f7a89;font-size:.8rem"> / 100</span>
                  </div>
                </div>
                <div class="terminal-panel">
                  <div class="terminal-kicker">{boundary} invalidation</div>
                  <div class="terminal-value negative">
                    {candidate.invalidation_level:,.2f}
                  </div>
                </div>
                <div class="terminal-panel">
                  <div class="terminal-kicker">Fibonacci target zone</div>
                  <div class="terminal-value positive">
                    {low_target:,.2f} – {high_target:,.2f}
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            hints = system_hints(
                candidate, score, alert_threshold, lifecycle
            )
            hint_rows = []
            for hint in hints:
                label, separator, value = hint.partition(":")
                if not separator:
                    label, value = "Status", hint
                hint_rows.append(
                    "<tr>"
                    "<td style='color:#F4F7FA;font-weight:700;"
                    "font-size:.71rem;vertical-align:top;padding:.3rem .7rem "
                    f".3rem 0;white-space:nowrap'>{html.escape(label)}</td>"
                    "<td style='color:#F4F7FA;font-size:.71rem;"
                    "line-height:1.35;padding:.3rem 0'>"
                    f"{_format_hint_value(value.strip())}</td>"
                    "</tr>"
                )
            st.markdown(
                "<div class='terminal-panel'>"
                "<div class='terminal-kicker'>System Hints</div>"
                "<table style='width:100%;border-collapse:collapse;"
                "border:0;margin-top:.35rem'><tbody>"
                + "".join(hint_rows)
                + "</tbody></table></div>",
                unsafe_allow_html=True,
            )
            focus_selected = st.checkbox(
                "Focus chart on selected path", value=True
            )
            with st.expander("Score audit", expanded=False):
                for item in score.items:
                    earned = item.points / item.maximum if item.maximum else 0
                    st.markdown(
                        f"<div style='font-size:.75rem;color:#aab2bf;"
                        f"margin-top:.4rem'>{item.reason}</div>"
                        f"<div style='font-size:.73rem;color:#eef1f5'>"
                        f"{item.points:.2f} / {item.maximum:.0f}</div>"
                        f"<div style='height:3px;background:#202632;"
                        f"margin:.2rem 0'><div style='width:{earned * 100:.1f}%;"
                        f"height:3px;background:#00bfa5'></div></div>",
                        unsafe_allow_html=True,
                    )

    with chart_col:
        threshold_text, buy_text, sell_text = marker_status(
            view_result.rankings, overlays, alert_threshold
        )
        st.markdown(
            "<div class='terminal-panel' style='display:grid;"
            "grid-template-columns:repeat(3,1fr);gap:.8rem;margin-bottom:.7rem'>"
            "<div><span class='terminal-kicker'>Marker threshold</span>"
            f"<div class='terminal-value'>{threshold_text}</div></div>"
            "<div><span class='terminal-kicker'>Buy at</span>"
            f"<div class='terminal-value' style='color:#18C98B'>{buy_text}</div></div>"
            "<div><span class='terminal-kicker'>Sell at</span>"
            f"<div class='terminal-value' style='color:#F05D68'>{sell_text}</div></div>"
            "</div>",
            unsafe_allow_html=True,
        )
        chart_result = (
            focus_dashboard(view_result, top_rankings[selected_index][0])
            if top_rankings and focus_selected
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
                f"elliott-{selected_database.name}-{timeframe}-"
                f"{candidate_scope}-{selected_index}-{overlays}-"
                f"{bool(top_rankings and focus_selected)}-{alert_threshold}"
            ),
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
                        "Direction": candidate.direction,
                        "Score": score.total,
                        "Lifecycle": candidate_lifecycle(
                            candidate, result.candles
                        ),
                        "Completed": candidate.pivots[-1].timestamp.strftime(
                            "%Y-%m-%d %H:%M"
                        ),
                    }
                    for index, (candidate, score) in enumerate(scoped_rankings)
                ]
                st.dataframe(
                    pd.DataFrame(rows),
                    hide_index=True,
                    width="stretch",
                    height=280,
                )

    with operations_col:
        with st.expander("Alerts & operations", expanded=False):
            st.markdown(
                "<div class='terminal-kicker'>Telegram Bot Token</div>"
                "<div style='font-size:.78rem;color:#8c96a5;margin:.25rem 0 .7rem'>"
                "Configure <code>TELEGRAM_BOT_TOKEN</code> in "
                "<code>.streamlit/secrets.toml</code>. Tokens are never entered "
                "into the browser.</div>",
                unsafe_allow_html=True,
            )
            chat_id = st.text_input("Telegram Chat ID")
            alert_threshold = st.slider(
                "Alert Confidence Score", 0, 100, 75, key="alert_threshold"
            )
            st.caption("Log-only webhook shell. No network request is sent.")
            if scoped_rankings:
                candidate, score = scoped_rankings[0]
                alert = format_setup_alert(
                    candidate, score, alert_threshold, chat_id=chat_id
                )
                alert_key = (
                    candidate.pivots[-1].timestamp.isoformat(),
                    score.total,
                    alert_threshold,
                )
                if (
                    alert
                    and st.session_state.get("last_alert_key") != alert_key
                ):
                    log_alert_background(alert)
                    st.session_state["last_alert_key"] = alert_key
                    st.success("Setup written to application logs.")


def _is_tradeable_setup(candidate: WaveCandidate) -> bool:
    """Whether the terminal pivot is the actionable Wave 4 or Wave B."""
    return bool(candidate.labels) and candidate.labels[-1] in {"4", "B"}


def _has_completed_trade_setup(candidate: WaveCandidate) -> bool:
    """Backward-compatible alias for the shared tradeability predicate."""
    return _is_tradeable_setup(candidate)


def scan_global_markets(
    databases: tuple[Path, ...],
    atr_multiplier: float,
    atr_period: int,
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    """Scan every asset/timeframe without mutating terminal state."""
    rows: list[dict[str, object]] = []
    errors: list[str] = []
    handled_errors = (
        ValueError,
        OSError,
        sqlite3.DatabaseError,
        pd.errors.DatabaseError,
        ZeroDivisionError,
    )
    for database in databases:
        for timeframe in TIMEFRAMES:
            try:
                result = compute_dashboard(
                    database, timeframe, atr_multiplier, atr_period
                )
            except handled_errors as error:
                errors.append(f"{database.name} {timeframe}: {error}")
                continue
            for candidate, score in result.rankings:
                if not _has_completed_trade_setup(candidate):
                    continue
                if candidate_lifecycle(candidate, result.candles) != "Active":
                    continue
                target_low, target_high = target_zone(candidate)
                rows.append(
                    {
                        "Market": database.stem,
                        "Timeframe": timeframe,
                        "Pattern": candidate.pattern,
                        "Direction": candidate.direction,
                        "Confidence Score": score.total,
                        "Invalidation Price": candidate.invalidation_level,
                        "Fibonacci Target Zone": (
                            f"{target_low:,.5f} – {target_high:,.5f}"
                        ),
                    }
                )
    columns = (
        "Market",
        "Timeframe",
        "Pattern",
        "Direction",
        "Confidence Score",
        "Invalidation Price",
        "Fibonacci Target Zone",
    )
    frame = pd.DataFrame(rows, columns=columns)
    if not frame.empty:
        frame = frame.sort_values(
            "Confidence Score", ascending=False, kind="stable"
        ).reset_index(drop=True)
    return frame, tuple(errors)


def _render_global_scanner() -> None:
    import streamlit as st

    st.markdown("## Global Market Scanner")
    st.caption(
        "Batch scan of every local market across every registered timeframe. "
        "Only active structures with a completed Wave 4 or Wave B are shown."
    )
    atr_multiplier = float(st.session_state.get("atr_multiplier", 2.0))
    atr_period = int(st.session_state.get("atr_period", 14))
    databases = discover_databases()

    status_col, action_col = st.columns([0.75, 0.25], vertical_alignment="center")
    with status_col:
        st.markdown(
            f"<div class='terminal-panel'><span class='terminal-kicker'>"
            f"Scan universe</span><div class='terminal-value'>"
            f"{len(databases)} markets · {len(TIMEFRAMES)} timeframes · ATR "
            f"{atr_period} × {atr_multiplier:.1f}</div></div>",
            unsafe_allow_html=True,
        )
    with action_col:
        run_scan = st.button(
            "Run market scan",
            type="primary",
            width="stretch",
            disabled=not databases,
        )

    if not databases:
        st.info("No local asset databases were detected.")
        return
    if run_scan:
        with st.spinner("Scanning markets..."):
            frame, errors = scan_global_markets(
                databases, atr_multiplier, atr_period
            )
        st.session_state["scanner_frame"] = frame
        st.session_state["scanner_errors"] = errors
        st.session_state["scanner_signature"] = (
            tuple(str(path) for path in databases),
            atr_multiplier,
            atr_period,
        )

    frame = st.session_state.get("scanner_frame")
    errors = st.session_state.get("scanner_errors", ())
    if frame is None:
        st.info("Run the scanner to calculate the current global opportunity set.")
        return
    if frame.empty:
        st.warning("No active Wave 4 or Wave B trade setups were found.")
    else:
        st.dataframe(
            frame,
            hide_index=True,
            width="stretch",
            height=min(680, 44 + 35 * len(frame)),
            column_config={
                "Confidence Score": st.column_config.NumberColumn(
                    format="%.2f"
                ),
                "Invalidation Price": st.column_config.NumberColumn(
                    format="%.5f"
                ),
            },
        )
        st.caption(
            f"{len(frame)} active setups · sorted by Confidence Score"
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
    terminal_tab, scanner_tab = st.tabs(
        ("Single Chart Terminal", "Global Market Scanner"),
        key="dashboard_view",
    )
    with terminal_tab:
        _render_single_chart()
    with scanner_tab:
        _render_global_scanner()


if __name__ == "__main__":
    main()
