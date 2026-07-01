"""Deterministic position sizing and portfolio risk gates."""

from __future__ import annotations

import math
from dataclasses import dataclass

from backtester import Friction


@dataclass(frozen=True, slots=True)
class RiskPolicy:
    account_equity: float
    risk_per_trade_percent: float = 1.0
    maximum_open_risk_percent: float = 3.0
    maximum_daily_loss_percent: float = 3.0
    minimum_reward_risk: float = 1.5
    lot_size: int = 1

    def __post_init__(self) -> None:
        if self.account_equity <= 0:
            raise ValueError("account_equity must be positive")
        if not 0 < self.risk_per_trade_percent <= 100:
            raise ValueError("risk_per_trade_percent must be in (0, 100]")
        if not 0 < self.maximum_open_risk_percent <= 100:
            raise ValueError("maximum_open_risk_percent must be in (0, 100]")
        if not 0 < self.maximum_daily_loss_percent <= 100:
            raise ValueError("maximum_daily_loss_percent must be in (0, 100]")
        if self.minimum_reward_risk <= 0:
            raise ValueError("minimum_reward_risk must be positive")
        if not isinstance(self.lot_size, int) or self.lot_size <= 0:
            raise ValueError("lot_size must be a positive integer")


@dataclass(frozen=True, slots=True)
class TradePlan:
    approved: bool
    direction: str
    units: int
    entry: float
    stop: float
    target: float
    risk_per_unit: float
    total_risk: float
    reward_risk: float
    reason: str


def size_trade(
    *,
    direction: str,
    entry: float,
    stop: float,
    target: float,
    policy: RiskPolicy,
    current_open_risk: float = 0.0,
    realized_daily_loss: float = 0.0,
    friction: Friction = Friction(),
) -> TradePlan:
    """Return an immutable plan after direction, R:R and risk-budget gates."""
    if direction not in {"Bullish", "Bearish"}:
        raise ValueError("direction must be Bullish or Bearish")
    if min(entry, stop, target) <= 0:
        raise ValueError("entry, stop and target must be positive")
    correct_geometry = (
        stop < entry < target
        if direction == "Bullish"
        else target < entry < stop
    )
    friction_per_unit = (
        friction.spread + 2 * friction.slippage + 2 * friction.commission
    )
    risk_per_unit = abs(entry - stop) + friction_per_unit
    reward = max(0.0, abs(target - entry) - friction_per_unit)
    reward_risk = reward / risk_per_unit if risk_per_unit else 0.0
    daily_limit = (
        policy.account_equity * policy.maximum_daily_loss_percent / 100
    )
    trade_budget = policy.account_equity * policy.risk_per_trade_percent / 100
    open_limit = (
        policy.account_equity * policy.maximum_open_risk_percent / 100
    )
    available_open = max(0.0, open_limit - current_open_risk)
    raw_units = math.floor(min(trade_budget, available_open) / risk_per_unit)
    units = raw_units - raw_units % policy.lot_size

    reason = "Approved"
    approved = True
    if not correct_geometry:
        approved, reason = False, "Stop/target geometry conflicts with direction"
    elif realized_daily_loss >= daily_limit:
        approved, reason = False, "Maximum daily loss reached"
    elif reward_risk < policy.minimum_reward_risk:
        approved, reason = False, "Reward-to-risk below policy minimum"
    elif units <= 0:
        approved, reason = False, "Insufficient remaining risk budget"
    if not approved:
        units = 0
    return TradePlan(
        approved=approved,
        direction=direction,
        units=units,
        entry=entry,
        stop=stop,
        target=target,
        risk_per_unit=round(risk_per_unit, 10),
        total_risk=round(units * risk_per_unit, 10),
        reward_risk=round(reward_risk, 4),
        reason=reason,
    )
