"""Read-only Upstox Market Data Feed V3 WebSocket adapter."""

from __future__ import annotations

import json
import logging
import uuid
from collections.abc import Iterable
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd

from providers.base import MarketDataProvider, TickCallback

LOGGER = logging.getLogger(__name__)
AUTHORIZE_URL = "https://api.upstox.com/v3/feed/market-data-feed/authorize"
DEFAULT_HEADERS = {
    "Accept": "application/json",
    "Accept-Language": "en-US,en;q=0.9",
    "Cache-Control": "no-cache",
    "Pragma": "no-cache",
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/138.0.0.0 Safari/537.36"
    ),
}


class UpstoxProvider(MarketDataProvider):
    """Connect, subscribe, decode, and normalize Upstox V3 market ticks.

    The generated ``MarketDataFeedV3_pb2.py`` module must be importable at
    runtime. Generate it from Upstox's official Market Data Feed V3 proto:
    ``protoc --python_out=. MarketDataFeedV3.proto``.
    """

    def __init__(self, access_token: str, *, mode: str = "ltpc") -> None:
        if not access_token.strip():
            raise ValueError("UPSTOX_ACCESS_TOKEN is required")
        if mode not in {"ltpc", "full"}:
            raise ValueError("mode must be ltpc or full")
        self.access_token = access_token.strip()
        self.mode = mode
        self._socket: Any = None
        self._keys: tuple[str, ...] = ()

    def connect(self) -> None:
        try:
            import websocket
        except ImportError as error:
            raise RuntimeError(
                "Install websocket-client to use Upstox Live"
            ) from error
        payload = authorize_market_feed(self.access_token)
        uri = payload.get("data", {}).get("authorized_redirect_uri")
        if not uri:
            raise RuntimeError("Upstox did not return an authorized WebSocket URL")
        self._socket = websocket.create_connection(uri, timeout=30)

    def subscribe(self, instrument_keys: Iterable[str]) -> None:
        if self._socket is None:
            raise RuntimeError("connect must be called before subscribe")
        self._keys = tuple(dict.fromkeys(key for key in instrument_keys if key))
        if not self._keys:
            raise ValueError("at least one instrument key is required")
        request = {
            "guid": str(uuid.uuid4()),
            "method": "sub",
            "data": {"mode": self.mode, "instrumentKeys": list(self._keys)},
        }
        self._socket.send_binary(json.dumps(request).encode("utf-8"))

    def listen(self, callback: TickCallback) -> None:
        if self._socket is None:
            raise RuntimeError("provider is not connected")
        while self._socket is not None:
            message = self._socket.recv()
            if not isinstance(message, (bytes, bytearray)):
                continue
            try:
                for event in decode_market_feed(bytes(message)):
                    callback(event)
            except Exception:
                LOGGER.exception("Unable to decode an Upstox market-data message")

    def close(self) -> None:
        socket, self._socket = self._socket, None
        if socket is not None:
            socket.close()


def decode_market_feed(message: bytes) -> tuple[dict[str, Any], ...]:
    """Decode official V3 Protobuf bytes into provider-neutral tick dictionaries."""
    try:
        from google.protobuf.json_format import MessageToDict
    except ImportError as error:
        raise RuntimeError(
            "google.protobuf is required for Upstox live decoding"
        ) from error
    try:
        import MarketDataFeedV3_pb2 as feed_pb2
    except ImportError:
        try:
            import MarketDataFeed_pb2 as feed_pb2
        except ImportError as error:
            raise RuntimeError(
                "Upstox Protobuf bindings are missing; generate "
                "MarketDataFeedV3_pb2.py or MarketDataFeed_pb2.py from the official V3 proto"
            ) from error

    response = feed_pb2.FeedResponse()
    response.ParseFromString(message)
    payload = MessageToDict(response, preserving_proto_field_name=True)
    events: list[dict[str, Any]] = []
    for instrument_key, feed in payload.get("feeds", {}).items():
        ltpc = _find_mapping(feed, "ltpc") or {}
        ltp = ltpc.get("ltp")
        if ltp is None:
            continue
        timestamp = (
            ltpc.get("ltt")
            or payload.get("currentTs")
            or payload.get("current_ts")
        )
        events.append(
            {
                "instrument_key": instrument_key,
                "timestamp": pd.to_datetime(int(timestamp), unit="ms", utc=True),
                "ltp": float(ltp),
                "volume": _find_numeric(feed, ("vtt", "volume")),
                "open": _find_numeric(feed, ("open",)),
                "high": _find_numeric(feed, ("high",)),
                "low": _find_numeric(feed, ("low",)),
                "close": _find_numeric(feed, ("cp", "close")),
            }
        )
    return tuple(events)


def _find_mapping(value: Any, key: str) -> dict[str, Any] | None:
    if isinstance(value, dict):
        found = value.get(key)
        if isinstance(found, dict):
            return found
        for child in value.values():
            result = _find_mapping(child, key)
            if result is not None:
                return result
    elif isinstance(value, list):
        for child in value:
            result = _find_mapping(child, key)
            if result is not None:
                return result
    return None


def authorize_market_feed(access_token: str) -> dict[str, Any]:
    """Authorize the market-data feed with a browser-like HTTP client."""
    token = access_token.strip()
    if not token:
        raise ValueError("UPSTOX_ACCESS_TOKEN is required")
    headers = {
        **DEFAULT_HEADERS,
        "Authorization": f"Bearer {token}",
    }
    try:
        import requests
    except ImportError:
        request = Request(AUTHORIZE_URL, headers=headers)
        with urlopen(request, timeout=20) as response:
            return json.load(response)

    response = requests.get(
        AUTHORIZE_URL,
        headers=headers,
        timeout=20,
        allow_redirects=True,
    )
    try:
        response.raise_for_status()
    except requests.HTTPError as error:
        raise RuntimeError(_format_http_error(response.status_code, response.text)) from error
    try:
        return response.json()
    except ValueError as error:
        raise RuntimeError("Upstox authorize endpoint returned non-JSON content") from error


def _format_http_error(status_code: int, body: str) -> str:
    body = body.strip()
    if not body:
        return f"HTTP {status_code}"
    return f"HTTP {status_code}: {body}"


def _find_numeric(value: Any, keys: tuple[str, ...]) -> float | None:
    if isinstance(value, dict):
        for key in keys:
            if key in value:
                try:
                    return float(value[key])
                except (TypeError, ValueError):
                    pass
        for child in value.values():
            result = _find_numeric(child, keys)
            if result is not None:
                return result
    elif isinstance(value, list):
        for child in value:
            result = _find_numeric(child, keys)
            if result is not None:
                return result
    return None
