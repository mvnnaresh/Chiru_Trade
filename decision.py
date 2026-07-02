"""Decision-first trade filtering built on the deterministic wave and risk engines."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from backtester import Friction
from engine import WaveCandidate
from risk import RiskPolicy, size_trade
from scoring import ConfidenceScore


@dataclass(frozen=True, slots=True)
class DecisionState:
    status: str
    direction: str
    color: str
    reason: str
    second_reason: str
    current_price: float | None = None
    entry_reference: float | None = None
    setup_reference: float | None = None
    stop: float | None = None
    target_1: float | None = None
    target_2: float | None = None
    reward_risk: float | None = None
    units: int | None = None
    total_risk: float | None = None
    risk_approved: bool = False
    risk_reason: str = "Not evaluated"
    setup_quality_score: float | None = None
    required_threshold: float | None = None
    quality_gate_result: str = "Not evaluated"
    lifecycle: str | None = None
    stage: str | None = None


def _tradeable(candidate: WaveCandidate) -> bool:
    terminal = "4" if candidate.pattern == "Impulse" else "B"
    return candidate.status == "EntryReady" and terminal in candidate.labels


def _setup_reference(candidate: WaveCandidate) -> float | None:
    label = "4" if candidate.pattern == "Impulse" else "B"
    pivot = candidate.pivot_for_label(label)
    return float(pivot.price) if pivot is not None else None


def build_decision_state(
    candidate: WaveCandidate | None,
    score: ConfidenceScore | None,
    alert_threshold: float,
    lifecycle: str | None,
    candles: pd.DataFrame | None,
    risk_policy: RiskPolicy | None,
    friction: Friction | None,
    current_open_risk: float = 0.0,
    realized_daily_loss: float = 0.0,
) -> DecisionState:
    """Combine structure, quality, lifecycle, market price, and risk gates."""
    quality = float(score.total) if score is not None else None
    current_price = (
        float(candles.iloc[-1]["close"])
        if candles is not None and not candles.empty and "close" in candles
        else None
    )
    common = {
        "direction": candidate.direction if candidate else "n/a",
        "current_price": current_price,
        "setup_quality_score": quality,
        "required_threshold": float(alert_threshold),
        "quality_gate_result": (
            "Not evaluated"
            if quality is None
            else "Passed" if quality >= alert_threshold
            else "Below threshold"
        ),
        "lifecycle": lifecycle,
        "stage": candidate.status if candidate else None,
    }
    if candidate is None or score is None:
        return DecisionState(
            status="NO TRADE", color="#F0B90B",
            reason="No active candidate selected.",
            second_reason="Select a live structure to evaluate.",
            **common,
        )

    targets = _target_zone(candidate)
    setup = _setup_reference(candidate)
    base = {
        **common,
        "setup_reference": setup,
        "stop": float(candidate.invalidation_level),
        "target_1": targets[0],
        "target_2": targets[1],
    }
    if candidate.status == "Forming":
        return DecisionState(
            status="WATCH", color="#F0B90B",
            reason="The terminal Wave 4/B pivot is still forming.",
            second_reason="Risk is not evaluated until the setup is confirmed.",
            risk_reason="Not evaluated because setup is not confirmed.",
            **base,
        )
    if lifecycle != "Active":
        if lifecycle == "Target hit":
            reason = "Target zone already reached."
            second_reason = "Wait for a new Wave 4/B setup."
        elif lifecycle == "Invalidated":
            reason = "The selected structure has been invalidated."
            second_reason = "Wait for a new Wave 4/B setup."
        else:
            reason = (
                f"The selected structure is no longer active "
                f"({lifecycle or 'unknown'})."
            )
            second_reason = "Wait for a new active Wave 4/B setup."
        return DecisionState(
            status="NO TRADE", color="#F05D68",
            reason=reason,
            second_reason=second_reason,
            risk_reason="Not evaluated for an inactive structure.",
            **base,
        )
    if not _tradeable(candidate):
        return DecisionState(
            status="NO TRADE", color="#F05D68",
            reason="The structure is valid for analysis but not at a tradeable terminal wave.",
            second_reason="Wait for a confirmed Wave 4 or Wave B.",
            risk_reason="Not evaluated for a non-tradeable structure.",
            **base,
        )
    if score.total < alert_threshold:
        return DecisionState(
            status="WATCH", color="#F0B90B",
            reason="The structure is tradeable but below the configured setup-quality threshold.",
            second_reason=f"Setup quality is {score.total:.1f}; {alert_threshold:.1f} is required.",
            risk_reason="Not evaluated below the setup-quality threshold.",
            **base,
        )
    if candles is None or candles.empty or "close" not in candles:
        return DecisionState(
            status="BLOCKED", color="#F05D68",
            reason="A completed candle close is unavailable.",
            second_reason="Risk cannot be evaluated without a current price.",
            risk_reason="Missing current price",
            **base,
        )
    current = current_price
    priced = {**base, "entry_reference": current}
    if risk_policy is None or friction is None:
        return DecisionState(
            status="BLOCKED", color="#F05D68",
            reason="Risk settings are invalid.",
            second_reason="Correct the Risk Settings before evaluating this setup.",
            risk_reason="Invalid risk settings",
            **priced,
        )
    if candidate.direction == "Bullish":
        valid_targets = [target for target in targets if target > current]
        if not valid_targets:
            return DecisionState(
                status="NO TRADE", color="#F05D68",
                reason="No valid upside target remains",
                second_reason="Wait for a new Wave 4/B setup.",
                risk_reason="Not evaluated because no valid target remains",
                **priced,
            )
        conservative_target = min(valid_targets)
    else:
        valid_targets = [target for target in targets if target < current]
        if not valid_targets:
            return DecisionState(
                status="NO TRADE", color="#F05D68",
                reason="No valid downside target remains",
                second_reason="Wait for a new Wave 4/B setup.",
                risk_reason="Not evaluated because no valid target remains",
                **priced,
            )
        conservative_target = max(valid_targets)
    try:
        plan = size_trade(
            direction=candidate.direction,
            entry=current,
            stop=float(candidate.invalidation_level),
            target=conservative_target,
            policy=risk_policy,
            friction=friction,
            current_open_risk=float(current_open_risk),
            realized_daily_loss=float(realized_daily_loss),
        )
    except (TypeError, ValueError, OverflowError) as error:
        return DecisionState(
            status="BLOCKED", color="#F05D68",
            reason=f"Risk evaluation failed: {error}",
            second_reason="Correct the Risk Settings before evaluating this setup.",
            risk_reason="Invalid risk settings",
            **priced,
        )
    if not plan.approved:
        return DecisionState(
            status="BLOCKED", color="#F05D68", reason=plan.reason,
            second_reason="Structure passed, but risk policy rejected the trade.",
            reward_risk=plan.reward_risk, units=0, total_risk=0.0,
            risk_reason=plan.reason, **priced,
        )
    return DecisionState(
        status="TRADE READY", color="#18C98B",
        reason="Structure, quality threshold, lifecycle, and risk gates passed.",
        second_reason="Use this as a decision-support setup, not an automatic order.",
        reward_risk=plan.reward_risk, units=plan.units,
        total_risk=plan.total_risk, risk_approved=True,
        risk_reason=plan.reason, **priced,
    )


def _target_zone(candidate: WaveCandidate) -> tuple[float, float]:
    """Mirror the engine-independent projection used by the existing UI."""
    sign = 1.0 if candidate.direction == "Bullish" else -1.0
    first_leg = abs(candidate.pivots[1].price - candidate.pivots[0].price)
    if candidate.status == "Forming" and candidate.active_leg is not None:
        anchor = candidate.active_leg.price
    elif candidate.pattern == "Impulse":
        anchor = candidate.pivots[4].price
    else:
        anchor = candidate.pivots[2].price
    projections = (anchor + sign * first_leg, anchor + sign * 1.618 * first_leg)
    return float(min(projections)), float(max(projections))
