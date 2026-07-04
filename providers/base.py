"""Minimal interface shared by live market-data providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from typing import Any

NormalizedTick = dict[str, Any]
TickCallback = Callable[[NormalizedTick], None]


class MarketDataProvider(ABC):
    """Read-only streaming market-data provider."""

    @abstractmethod
    def connect(self) -> None: ...

    @abstractmethod
    def subscribe(self, instrument_keys: Iterable[str]) -> None: ...

    @abstractmethod
    def listen(self, callback: TickCallback) -> None: ...

    @abstractmethod
    def close(self) -> None: ...
