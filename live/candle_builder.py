"""Build and persist only completed UTC-aligned five-minute candles."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from db import ingest_m5


@dataclass
class _Candle:
    timestamp: pd.Timestamp
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0

    def update(self, price: float, volume: float) -> None:
        self.high = max(self.high, price)
        self.low = min(self.low, price)
        self.close = price
        self.volume += volume


class M5CandleBuilder:
    """Aggregate normalized ticks and append completed M5 candles to SQLite."""

    def __init__(
        self,
        instrument_databases: dict[str, str | Path],
        *,
        volume_is_cumulative: bool = False,
    ) -> None:
        self.instrument_databases = {
            key: Path(database) for key, database in instrument_databases.items()
        }
        self.volume_is_cumulative = volume_is_cumulative
        self._active: dict[str, _Candle] = {}
        self._last_volume: dict[str, float] = {}
        self.last_tick_time: pd.Timestamp | None = None
        self.last_completed_candle_time: pd.Timestamp | None = None

    def process_tick(self, tick: dict[str, Any]) -> int:
        """Consume one normalized tick and return the number of rows inserted."""
        instrument_key = str(tick["instrument_key"])
        if instrument_key not in self.instrument_databases:
            return 0
        timestamp = pd.Timestamp(tick["timestamp"])
        timestamp = (
            timestamp.tz_localize("UTC")
            if timestamp.tzinfo is None
            else timestamp.tz_convert("UTC")
        )
        price = float(tick["ltp"])
        if price <= 0:
            raise ValueError("tick ltp must be positive")
        bucket = timestamp.floor("5min")
        volume = self._tick_volume(instrument_key, tick.get("volume"))
        active = self._active.get(instrument_key)
        inserted = 0
        if active is not None and bucket > active.timestamp:
            inserted = self._persist(instrument_key, active)
            active = None
        if active is not None and bucket < active.timestamp:
            return inserted
        if active is None:
            self._active[instrument_key] = _Candle(
                bucket, price, price, price, price, volume
            )
        else:
            active.update(price, volume)
        self.last_tick_time = timestamp
        return inserted

    def flush_completed(self, now: pd.Timestamp | None = None) -> int:
        """Persist active buckets strictly older than the current UTC M5 bucket."""
        current = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
        current = (
            current.tz_localize("UTC")
            if current.tzinfo is None
            else current.tz_convert("UTC")
        ).floor("5min")
        inserted = 0
        for instrument_key, candle in tuple(self._active.items()):
            if candle.timestamp < current:
                inserted += self._persist(instrument_key, candle)
                del self._active[instrument_key]
        return inserted

    def _tick_volume(self, instrument_key: str, raw: Any) -> float:
        if raw is None:
            return 0.0
        value = max(0.0, float(raw))
        if not self.volume_is_cumulative:
            return value
        previous = self._last_volume.get(instrument_key)
        self._last_volume[instrument_key] = value
        if previous is None:
            return 0.0
        return value - previous if value >= previous else value

    def _persist(self, instrument_key: str, candle: _Candle) -> int:
        frame = pd.DataFrame(
            [
                {
                    "open": candle.open,
                    "high": candle.high,
                    "low": candle.low,
                    "close": candle.close,
                    "volume": candle.volume,
                }
            ],
            index=pd.DatetimeIndex([candle.timestamp], name="timestamp"),
        )
        inserted = ingest_m5(self.instrument_databases[instrument_key], frame)
        self.last_completed_candle_time = candle.timestamp
        return inserted
