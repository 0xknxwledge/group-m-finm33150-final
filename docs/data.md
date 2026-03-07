# data — Storage and Fetchers

## storage.py

Owner: Antonio Braz

Polars-first parquet save/load layer. All data lives in `data/` at the project root.
`DATA_DIR` resolves relative to the source file, so paths are portable regardless of working directory.

### Constants

```python
DATA_DIR: Path  # <project_root>/data/
```

### Save / Load (Polars)

```python
def save_parquet_pl(df: pl.DataFrame, name: str) -> Path
```
Save a polars DataFrame to `data/{name}.parquet`.

```python
def load_parquet_pl(name: str) -> pl.DataFrame
```
Load `data/{name}.parquet` as a polars DataFrame. Raises `FileNotFoundError` if missing.

### Save / Load (Pandas — notebook compat)

```python
save_parquet = _save_parquet_pd   # alias
load_parquet = _load_parquet_pd   # alias
```
Kept for backward compatibility. Prefer the polars variants in new code.

### Typed Loaders

Convenience wrappers that load named datasets. Always returns polars.

```python
def load_funding() -> pl.DataFrame       # funding_rates.parquet
def load_candles() -> pl.DataFrame       # candles.parquet
def load_oi() -> pl.DataFrame            # open_interest.parquet
def load_orderbook_depth() -> pl.DataFrame   # orderbook_depth.parquet
def load_liquidation_volume() -> pl.DataFrame  # liquidation_volume.parquet
```

---

## fetchers.py

Owner: Antonio Braz (stubs), John Beecher (implementations)

Per-venue API fetchers. All return polars DataFrames with unified schemas.

### Unified Schemas

| Dataset | Columns |
|---------|---------|
| Funding rates | `[timestamp, venue, coin, funding_rate]` |
| Candles | `[timestamp, venue, coin, o, h, l, c, v]` |
| Open interest | `[timestamp, venue, coin, oi_usd]` |
| Liquidations | `[timestamp, venue, coin, side, size_usd, price]` |

All timestamps are UTC datetime. Funding rates are per-period values (not annualized).

### Constants

```python
TOKENS = ["BTC", "ETH", "SOL", "HYPE", "DOGE"]
VENUES = ["hyperliquid", "lighter", "okx", "kraken", "binance", "bybit", "dydx"]
VENUE_COINS: dict[str, list[str]]  # every venue supports all 5 coins
```

Symbol maps per venue: `OKX_SYMBOLS`, `KRAKEN_SYMBOLS`, `BINANCE_SYMBOLS`, `BYBIT_SYMBOLS`, `DYDX_SYMBOLS`.

### API Base URLs

| Venue | URL | Auth |
|-------|-----|------|
| Hyperliquid | `api.hyperliquid.xyz/info` | none |
| 0xArchive | `api.0xarchive.io/v1/*` | `X-API-Key` (env: `OXARCHIVE_API_KEY`) |
| OKX | `www.okx.com/api/v5/*` | none |
| Kraken | `futures.kraken.com/derivatives/*` | none |
| Binance | `fapi.binance.com/fapi/v1/*` | none (VPN) |
| Bybit | `api.bybit.com/v5/market/*` | none (VPN) |
| dYdX | `indexer.dydx.trade/v4/*` | none |

### Key Functions

```python
def fetch_hyperliquid_meta() -> tuple[pl.DataFrame, dict[str, dict]]
```
Fetch metadata and asset contexts from Hyperliquid. Returns `(universe_df, contexts)`.
`universe_df` columns: `[timestamp, venue, coin, oi, mark_price, funding_rate, premium, day_ntl_vlm, oi_usd]`.

### Funding Rate Semantics

| Venue | Rate unit | Settlement | Hourly payment |
|-------|-----------|------------|----------------|
| Hyperliquid | 8h rate | every 1h | rate / 8 |
| Lighter | 8h rate | every 1h | rate / 8 |
| dYdX | 1h rate | every 1h | rate |
| Kraken | 1h rate | every 1h | rate |
| Binance | 8h rate | every 8h | forward-fill on 1h grid |
| Bybit | 8h rate | every 8h | forward-fill on 1h grid |
| OKX | 8h rate | every 8h | forward-fill on 1h grid |
