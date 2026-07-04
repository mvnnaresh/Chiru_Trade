"""Resolve the configured database universe against a cached Upstox instrument master."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class UpstoxInstrumentMap:
    database_name: str
    display_name: str
    exchange_segment: str
    trading_symbol: str
    instrument_key: str | None = None


DEFAULT_UPSTOX_UNIVERSE: tuple[UpstoxInstrumentMap, ...] = (
    UpstoxInstrumentMap("NIFTY_50.db", "NIFTY 50", "NSE_INDEX", "NIFTY"),
    UpstoxInstrumentMap("NIFTY_BANK.db", "NIFTY Bank", "NSE_INDEX", "BANKNIFTY"),
    UpstoxInstrumentMap("RELIANCE.db", "Reliance Industries", "NSE_EQ", "RELIANCE"),
    UpstoxInstrumentMap("TCS.db", "Tata Consultancy Services", "NSE_EQ", "TCS"),
    UpstoxInstrumentMap("HDFC_BANK.db", "HDFC Bank", "NSE_EQ", "HDFCBANK"),
    UpstoxInstrumentMap("ICICI_BANK.db", "ICICI Bank", "NSE_EQ", "ICICIBANK"),
    UpstoxInstrumentMap("INFOSYS.db", "Infosys", "NSE_EQ", "INFY"),
    UpstoxInstrumentMap("SBI.db", "State Bank of India", "NSE_EQ", "SBIN"),
    UpstoxInstrumentMap("BHARTI_AIRTEL.db", "Bharti Airtel", "NSE_EQ", "BHARTIARTL"),
    UpstoxInstrumentMap("LARSEN_TOUBRO.db", "Larsen & Toubro", "NSE_EQ", "LT"),
    UpstoxInstrumentMap("BTC_USD.db", "Bitcoin / US Dollar", "GLOBAL", "BTCUSD"),
    UpstoxInstrumentMap("ETH_USD.db", "Ether / US Dollar", "GLOBAL", "ETHUSD"),
    UpstoxInstrumentMap("SOL_USD.db", "Solana / US Dollar", "GLOBAL", "SOLUSD"),
    UpstoxInstrumentMap("BNB_USD.db", "BNB / US Dollar", "GLOBAL", "BNBUSD"),
    UpstoxInstrumentMap("XRP_USD.db", "XRP / US Dollar", "GLOBAL", "XRPUSD"),
    UpstoxInstrumentMap("SP500.db", "S&P 500", "GLOBAL", "SPX"),
    UpstoxInstrumentMap("NASDAQ_100.db", "Nasdaq 100", "GLOBAL", "NDX"),
    UpstoxInstrumentMap("DOW_JONES.db", "Dow Jones", "GLOBAL", "DJI"),
    UpstoxInstrumentMap("RUSSELL_2000.db", "Russell 2000", "GLOBAL", "RUT"),
    UpstoxInstrumentMap("GOLD.db", "Gold", "MCX_FO", "GOLD"),
    UpstoxInstrumentMap("SILVER.db", "Silver", "MCX_FO", "SILVER"),
    UpstoxInstrumentMap("WTI_CRUDE.db", "Crude Oil", "MCX_FO", "CRUDEOIL"),
    UpstoxInstrumentMap("EUR_USD.db", "EUR / USD", "GLOBAL", "EURUSD"),
    UpstoxInstrumentMap("GBP_USD.db", "GBP / USD", "GLOBAL", "GBPUSD"),
    UpstoxInstrumentMap("USD_INR.db", "USD / INR", "NCD_FO", "USDINR"),
)


def resolve_instruments(
    universe: tuple[UpstoxInstrumentMap, ...] = DEFAULT_UPSTOX_UNIVERSE,
    *,
    cache_path: str | Path = "upstox_instruments_cache.json",
    output_path: str | Path = "upstox_instrument_map.json",
) -> tuple[tuple[UpstoxInstrumentMap, ...], tuple[UpstoxInstrumentMap, ...]]:
    """Resolve exact segment/symbol matches and persist non-secret mapping results."""
    records = _load_records(Path(cache_path))
    lookup = {
        (
            str(record.get("segment", "")).upper(),
            str(record.get("trading_symbol", "")).upper(),
        ): str(record["instrument_key"])
        for record in records
        if record.get("instrument_key")
    }
    resolved: list[UpstoxInstrumentMap] = []
    unresolved: list[UpstoxInstrumentMap] = []
    for item in universe:
        key = lookup.get(
            (item.exchange_segment.upper(), item.trading_symbol.upper())
        )
        mapped = replace(item, instrument_key=key)
        (resolved if key else unresolved).append(mapped)
    Path(output_path).write_text(
        json.dumps([asdict(item) for item in (*resolved, *unresolved)], indent=2),
        encoding="utf-8",
    )
    return tuple(resolved), tuple(unresolved)


def _load_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = payload.get("data", payload.get("instruments", []))
    if not isinstance(payload, list):
        raise ValueError("Upstox instrument cache must contain a JSON list")
    return [item for item in payload if isinstance(item, dict)]
