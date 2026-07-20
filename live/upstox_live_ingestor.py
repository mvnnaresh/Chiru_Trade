"""Command-line Upstox feed -> completed M5 SQLite ingestor.

Run from the repository root:
    python live/upstox_live_ingestor.py --dry-run
    python live/upstox_live_ingestor.py --universe default

TODO: Add the OAuth login/token-refresh flow. This preparation stage accepts a
manually generated access token through ``UPSTOX_ACCESS_TOKEN``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from live.candle_builder import M5CandleBuilder
from live.instrument_resolver import (
    DEFAULT_UPSTOX_UNIVERSE,
    UpstoxInstrumentMap,
    resolve_instruments,
)
from live.token_loader import load_upstox_access_token
from providers.upstox_provider import UpstoxProvider


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Read-only Upstox M5 ingestor")
    parser.add_argument("--universe", choices=("default",), default="default")
    parser.add_argument("--symbols", nargs="*", metavar="DATABASE")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--cache", type=Path, default=Path("upstox_instruments_cache.json")
    )
    return parser


def select_universe(
    symbols: list[str] | None,
) -> tuple[UpstoxInstrumentMap, ...]:
    if not symbols:
        return DEFAULT_UPSTOX_UNIVERSE
    requested = {Path(symbol).name.lower() for symbol in symbols}
    return tuple(
        item
        for item in DEFAULT_UPSTOX_UNIVERSE
        if item.database_name.lower() in requested
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    universe = select_universe(args.symbols)
    unknown = (
        set(Path(symbol).name.lower() for symbol in args.symbols or ())
        - {item.database_name.lower() for item in universe}
    )
    if unknown:
        print(f"Unknown database mappings: {', '.join(sorted(unknown))}")

    resolved, unresolved = resolve_instruments(
        universe,
        cache_path=args.cache,
        output_path=args.root / "upstox_instrument_map.json",
    )
    print(f"Configured instruments: {len(universe)}")
    print(f"Resolved instruments: {len(resolved)}")
    for item in resolved:
        database = args.root / item.database_name
        state = "exists" if database.exists() else "missing database"
        print(f"  {item.database_name}: {item.instrument_key} ({state})")
    if unresolved:
        print("Unresolved instruments:")
        for item in unresolved:
            print(
                f"  {item.database_name}: "
                f"{item.exchange_segment}/{item.trading_symbol}"
            )
    if args.dry_run:
        print("Dry run complete; no WebSocket connection was attempted.")
        return 0

    token = load_upstox_access_token(root=args.root)
    if not token:
        print(
            "UPSTOX_ACCESS_TOKEN is missing in env and .streamlit/secrets.toml. "
            "Local SQLite/Yahoo modes remain available.",
            file=sys.stderr,
        )
        return 2
    available = [
        item for item in resolved if (args.root / item.database_name).exists()
    ]
    if not available:
        print("No resolved instruments have an existing local database.", file=sys.stderr)
        return 2

    databases = {
        str(item.instrument_key): args.root / item.database_name
        for item in available
    }
    builder = M5CandleBuilder(databases, volume_is_cumulative=True)
    provider = UpstoxProvider(token, mode="ltpc")
    status_path = args.root / "upstox_live_status.json"

    def on_tick(tick: dict[str, object]) -> None:
        builder.process_tick(tick)
        _write_status(
            status_path,
            state="UPSTOX LIVE CONNECTED",
            last_tick_time=builder.last_tick_time,
            last_completed_m5=builder.last_completed_candle_time,
        )

    try:
        _write_status(status_path, state="CONNECTING")
        provider.connect()
        provider.subscribe(databases)
        _write_status(status_path, state="UPSTOX LIVE CONNECTED")
        provider.listen(on_tick)
    except KeyboardInterrupt:
        _write_status(status_path, state="STOPPED")
        return 0
    except Exception as error:
        _write_status(status_path, state="UPSTOX CONNECTION ERROR", error=str(error))
        print(f"Upstox connection error: {error}", file=sys.stderr)
        return 1
    finally:
        provider.close()
    return 0


def _write_status(
    path: Path,
    *,
    state: str,
    last_tick_time: pd.Timestamp | None = None,
    last_completed_m5: pd.Timestamp | None = None,
    error: str | None = None,
) -> None:
    payload = {
        "state": state,
        "last_tick_time": None if last_tick_time is None else last_tick_time.isoformat(),
        "last_completed_m5": (
            None if last_completed_m5 is None else last_completed_m5.isoformat()
        ),
        "error": error,
    }
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


if __name__ == "__main__":
    raise SystemExit(main())
