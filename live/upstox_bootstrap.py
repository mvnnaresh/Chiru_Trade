"""Bootstrap non-secret Upstox live-data assets.

Downloads:
- complete Upstox instrument master JSON
- global instruments JSON
- official Market Data Feed V3 proto

This script does not require or use account credentials.
"""

from __future__ import annotations

import argparse
import gzip
import json
import shutil
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

BOD_COMPLETE_URL = "https://assets.upstox.com/market-quote/instruments/exchange/complete.json.gz"
GLOBAL_INSTRUMENTS_URL = "https://assets.upstox.com/market-quote/instruments/exchange/global.json.gz"
PROTO_URL = "https://assets.upstox.com/feed/market-data-feed/v3/MarketDataFeed.proto"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download Upstox live-data assets")
    parser.add_argument("--root", type=Path, default=Path("."))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)

    complete_records = _download_json_gz(BOD_COMPLETE_URL)
    try:
        global_records = _download_json_gz(GLOBAL_INSTRUMENTS_URL)
    except Exception:
        global_records = []
    combined = _merge_records(complete_records, global_records)

    cache_path = root / "upstox_instruments_cache.json"
    cache_path.write_text(json.dumps(combined, indent=2), encoding="utf-8")

    proto_path = root / "MarketDataFeed.proto"
    _download_file(PROTO_URL, proto_path)

    print(f"Saved instrument cache: {cache_path.name} ({len(combined)} records)")
    print(f"Saved proto file: {proto_path.name}")
    if not global_records:
        print(
            "Global instrument file was not downloaded; Indian market symbols still "
            "remain usable if present in the complete instrument file."
        )
    print(
        "Next step: generate MarketDataFeed_pb2.py from MarketDataFeed.proto "
        "using protoc or grpc_tools if available."
    )
    return 0


def _download_json_gz(url: str) -> list[dict[str, object]]:
    try:
        with urlopen(url, timeout=60) as response:
            with gzip.GzipFile(fileobj=response) as gz:
                payload = json.loads(gz.read().decode("utf-8"))
    except URLError as error:
        raise RuntimeError(f"Unable to download {url}") from error
    if not isinstance(payload, list):
        raise ValueError(f"Expected a JSON list from {url}")
    return [item for item in payload if isinstance(item, dict)]


def _download_file(url: str, path: Path) -> None:
    with urlopen(url, timeout=60) as response:
        with path.open("wb") as handle:
            shutil.copyfileobj(response, handle)


def _merge_records(
    primary: list[dict[str, object]],
    secondary: list[dict[str, object]],
) -> list[dict[str, object]]:
    merged: dict[str, dict[str, object]] = {}
    for record in [*primary, *secondary]:
        key = str(record.get("instrument_key", "")).strip()
        if key:
            merged[key] = record
    return list(merged.values())


if __name__ == "__main__":
    raise SystemExit(main())
