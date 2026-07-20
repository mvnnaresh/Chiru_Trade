# Elliott Wave Terminal

The application is read-only decision support. It reads canonical completed
five-minute candles from local SQLite databases; it never places orders.

## Using Upstox Live Data

Upstox Live is intended for Indian production market data. Yahoo remains a
test-refresh feed, and Local SQLite mode remains available without credentials.

1. Create an Upstox developer application.
2. Configure credentials. For the current preparation stage, only the access
   token is required at runtime. You can either place it in
   `.streamlit/secrets.toml`:

   ```toml
   UPSTOX_ACCESS_TOKEN="..."
   ```

   or set the environment variables below:

   ```text
   UPSTOX_API_KEY=...
   UPSTOX_API_SECRET=...
   UPSTOX_REDIRECT_URI=...
   UPSTOX_ACCESS_TOKEN=...
   ```

3. Bootstrap the public Upstox assets and verify mappings without connecting:

   ```powershell
   python live/upstox_bootstrap.py
   python live/upstox_live_ingestor.py --dry-run
   ```

4. Generate `MarketDataFeedV3_pb2.py` from the official Upstox Market Data Feed
   V3 proto and start the separate read-only ingestor:

   ```powershell
   protoc --python_out=. MarketDataFeedV3.proto
   python live/upstox_live_ingestor.py --universe default
   ```

   A subset can be selected with:

   ```powershell
   python live/upstox_live_ingestor.py --symbols HDFC_BANK.db RELIANCE.db NIFTY_50.db
   ```

5. In another terminal, start the dashboard:

   ```powershell
   python -m streamlit run app.py
   ```

The ingestor normalizes Upstox V3 Protobuf ticks, builds UTC-aligned M5 candles,
and writes only completed candles through the existing `ingest_m5()` API. The
dashboard continues reading SQLite through the existing aggregation pipeline.

OAuth login and automatic token refresh are intentionally left as a TODO. No
credentials, broker orders, account positions, or auto-trading are implemented.

Official references: [Market Data Feed V3](https://upstox.com/developer/api-documentation/v3/get-market-data-feed/),
[authorized feed URL](https://upstox.com/developer/api-documentation/get-market-data-feed-authorize-v3/),
and [instrument files](https://upstox.com/developer/api-documentation/instruments/).
