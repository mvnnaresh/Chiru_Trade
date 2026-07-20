"""Safe Upstox token validation for market-data authorization.

This script never prints the token. It only reports whether Upstox accepts it
for the Market Data Feed V3 authorization endpoint.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from live.token_loader import load_upstox_access_token
from providers.upstox_provider import authorize_market_feed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate Upstox access token")
    parser.add_argument("--root", type=Path, default=Path("."))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    token = load_upstox_access_token(root=args.root)
    if not token:
        print("Token check: FAIL")
        print("Reason: UPSTOX_ACCESS_TOKEN not found in env or .streamlit/secrets.toml")
        return 2

    try:
        payload = authorize_market_feed(token)
    except Exception as error:
        print("Token check: FAIL")
        print(f"Reason: {error}")
        return 1

    uri = payload.get("data", {}).get("authorized_redirect_uri")
    if uri:
        print("Token check: PASS")
        print("Feed authorize endpoint accepted the token.")
        return 0

    print("Token check: FAIL")
    print("Response was successful but did not include authorized_redirect_uri.")
    print(json.dumps(payload, indent=2))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
