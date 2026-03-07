"""Exchange API fetchers for funding rates, candles, open interest, and liquidations.

Owner: Antonio Braz (stubs), John Beecher (cascade-critical implementations)

Each fetcher returns a polars DataFrame with a unified schema:
- Funding rates:  columns [timestamp, venue, coin, funding_rate]
- Candles:        columns [timestamp, venue, coin, o, h, l, c, v]
- Open interest:  columns [timestamp, venue, coin, oi_usd]
- Liquidations:   columns [timestamp, venue, coin, side, size_usd, price]

All timestamps are UTC datetime. Funding rates are per-period (8h)
values, not annualized.

API Reference:
  Hyperliquid   POST api.hyperliquid.xyz/info          (no auth)
  0xArchive     GET  api.0xarchive.io/v1/*              (X-API-Key)
  OKX           GET  www.okx.com/api/v5/*               (no auth)
  Kraken        GET  futures.kraken.com/derivatives/*    (no auth)
  Binance       GET  fapi.binance.com/fapi/v1/*         (no auth, VPN)
  Bybit         GET  api.bybit.com/v5/market/*          (no auth, VPN)
  dYdX          GET  indexer.dydx.trade/v4/*             (no auth)
"""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timezone
from typing import Any

import polars as pl
import requests

logger = logging.getLogger(__name__)

# ── API base URLs ──
HYPERLIQUID_API = "https://api.hyperliquid.xyz/info"
OXARCHIVE_API = "https://api.0xarchive.io"
OXARCHIVE_KEY = os.environ.get("OXARCHIVE_API_KEY", "")
OKX_API = "https://www.okx.com"
KRAKEN_FUTURES_API = "https://futures.kraken.com"
BINANCE_FAPI = "https://fapi.binance.com/fapi/v1"
BYBIT_API = "https://api.bybit.com/v5"
DYDX_API = "https://indexer.dydx.trade/v4"

# Token universe
TOKENS = ["BTC", "ETH", "SOL", "HYPE", "DOGE"]

# All venues
VENUES = ["hyperliquid", "lighter", "okx", "kraken", "binance", "bybit", "dydx"]

# ── Symbol mappings per venue ──
OKX_SYMBOLS: dict[str, str] = {
    "BTC": "BTC-USDT-SWAP",
    "ETH": "ETH-USDT-SWAP",
    "SOL": "SOL-USDT-SWAP",
    "HYPE": "HYPE-USDT-SWAP",
    "DOGE": "DOGE-USDT-SWAP",
}

KRAKEN_SYMBOLS: dict[str, str] = {
    "BTC": "PF_XBTUSD",
    "ETH": "PF_ETHUSD",
    "SOL": "PF_SOLUSD",
    "HYPE": "PF_HYPEUSD",
    "DOGE": "PF_DOGEUSD",
}

BINANCE_SYMBOLS: dict[str, str] = {
    "BTC": "BTCUSDT",
    "ETH": "ETHUSDT",
    "SOL": "SOLUSDT",
    "HYPE": "HYPEUSDT",
    "DOGE": "DOGEUSDT",
}

BYBIT_SYMBOLS: dict[str, str] = {
    "BTC": "BTCUSDT",
    "ETH": "ETHUSDT",
    "SOL": "SOLUSDT",
    "HYPE": "HYPEUSDT",
    "DOGE": "DOGEUSDT",
}

DYDX_SYMBOLS: dict[str, str] = {
    "BTC": "BTC-USD",
    "ETH": "ETH-USD",
    "SOL": "SOL-USD",
    "HYPE": "HYPE-USD",
    "DOGE": "DOGE-USD",
}

# Every venue supports all 5 coins
VENUE_COINS: dict[str, list[str]] = {v: list(TOKENS) for v in VENUES}

# ── Empty schema helpers ──
_FUNDING_SCHEMA = {
    "timestamp": pl.Datetime("us", "UTC"),
    "venue": pl.Utf8,
    "coin": pl.Utf8,
    "funding_rate": pl.Float64,
}
_CANDLE_SCHEMA = {
    "timestamp": pl.Datetime("us", "UTC"),
    "venue": pl.Utf8,
    "coin": pl.Utf8,
    "o": pl.Float64,
    "h": pl.Float64,
    "l": pl.Float64,
    "c": pl.Float64,
    "v": pl.Float64,
}
_OI_SCHEMA = {
    "timestamp": pl.Datetime("us", "UTC"),
    "venue": pl.Utf8,
    "coin": pl.Utf8,
    "oi_usd": pl.Float64,
}
_LIQ_SCHEMA = {
    "timestamp": pl.Datetime("us", "UTC"),
    "venue": pl.Utf8,
    "coin": pl.Utf8,
    "side": pl.Utf8,
    "size_usd": pl.Float64,
    "price": pl.Float64,
}


def _empty(schema: dict) -> pl.DataFrame:
    return pl.DataFrame(schema=schema)


# ═══════════════════════════════════════════════════════════════════════
# Hyperliquid direct API
# ═══════════════════════════════════════════════════════════════════════


def _hl_post(payload: dict[str, Any]) -> Any:
    """POST to Hyperliquid info API and return parsed JSON."""
    resp = requests.post(HYPERLIQUID_API, json=payload, timeout=30)
    resp.raise_for_status()
    return resp.json()


def fetch_hyperliquid_meta() -> tuple[pl.DataFrame, dict[str, dict]]:
    """Fetch metadata and asset contexts from Hyperliquid.

    Returns (universe_df, contexts) where universe_df has columns:
        [timestamp, venue, coin, oi, mark_price, funding_rate, premium,
         day_ntl_vlm, oi_usd]
    """
    data = _hl_post({"type": "metaAndAssetCtxs"})
    meta, ctxs = data[0], data[1]
    universe = meta["universe"]

    rows = []
    contexts: dict[str, dict] = {}
    for asset_info, ctx in zip(universe, ctxs):
        coin = asset_info["name"]
        contexts[coin] = ctx
        mp = float(ctx.get("markPx") or 0)
        oi = float(ctx.get("openInterest") or 0)
        rows.append(
            {
                "coin": coin,
                "oi": oi,
                "mark_price": mp,
                "funding_rate": float(ctx.get("funding") or 0),
                "premium": float(ctx.get("premium") or 0),
                "day_ntl_vlm": float(ctx.get("dayNtlVlm") or 0),
                "oi_usd": oi * mp,
            }
        )

    now = datetime.now(timezone.utc)
    df = pl.DataFrame(rows).with_columns(
        pl.lit(now).cast(pl.Datetime("us", "UTC")).alias("timestamp"),
        pl.lit("hyperliquid").alias("venue"),
    )
    return df, contexts


def _fetch_candles_hyperliquid(
    coin: str,
    interval: str = "1h",
    days: int = 90,
) -> pl.DataFrame:
    """Fetch OHLCV candles from Hyperliquid (fallback if 0xArchive is down)."""
    end_ms = int(time.time() * 1000)
    start_ms = end_ms - (days * 24 * 3600 * 1000)
    all_candles: list[dict] = []
    cursor = start_ms

    while cursor < end_ms:
        data = _hl_post(
            {
                "type": "candleSnapshot",
                "req": {
                    "coin": coin,
                    "interval": interval,
                    "startTime": cursor,
                    "endTime": end_ms,
                },
            }
        )
        if not data:
            break
        all_candles.extend(data)
        last_t = int(data[-1]["t"])
        if last_t <= cursor:
            break
        cursor = last_t + 1

    if not all_candles:
        return _empty(_CANDLE_SCHEMA)

    df = pl.DataFrame(all_candles)
    df = df.with_columns(
        _epoch_ms_to_dt("t").alias("timestamp"),
        pl.lit("hyperliquid").alias("venue"),
        pl.lit(coin).alias("coin"),
        pl.col("o").cast(pl.Float64),
        pl.col("h").cast(pl.Float64),
        pl.col("l").cast(pl.Float64),
        pl.col("c").cast(pl.Float64),
        pl.col("v").cast(pl.Float64),
    ).select("timestamp", "venue", "coin", "o", "h", "l", "c", "v")
    return df.unique(subset=["timestamp", "coin"]).sort("timestamp")


def _fetch_funding_hyperliquid(coin: str, days: int = 90) -> pl.DataFrame:
    """Fetch historical funding rates from Hyperliquid (fallback)."""
    end_ms = int(time.time() * 1000)
    start_ms = end_ms - (days * 24 * 3600 * 1000)
    all_records: list[dict] = []
    cursor = start_ms

    while cursor < end_ms:
        data = _hl_post(
            {
                "type": "fundingHistory",
                "coin": coin,
                "startTime": cursor,
                "endTime": end_ms,
            }
        )
        if not data:
            break
        all_records.extend(data)
        last_t = int(data[-1]["time"])
        if last_t <= cursor:
            break
        cursor = last_t + 1

    if not all_records:
        return _empty(_FUNDING_SCHEMA)

    df = pl.DataFrame(all_records)
    df = df.with_columns(
        _epoch_ms_to_dt("time").alias("timestamp"),
        pl.col("fundingRate").cast(pl.Float64).alias("funding_rate"),
        pl.lit("hyperliquid").alias("venue"),
        pl.lit(coin).alias("coin"),
    ).select("timestamp", "venue", "coin", "funding_rate")
    return df.unique(subset=["timestamp", "coin"]).sort("timestamp")


def fetch_orderbook_depth(coin: str = "HYPE", n_levels: int = 20) -> dict:
    """Fetch L2 orderbook from Hyperliquid and compute depth metrics."""
    data = _hl_post({"type": "l2Book", "coin": coin, "nSigFigs": 5})
    levels = data.get("levels", [[], []])
    bids_raw = levels[0] if len(levels) > 0 else []
    asks_raw = levels[1] if len(levels) > 1 else []

    bids = [(float(b["px"]), float(b["sz"])) for b in bids_raw[:n_levels]]
    asks = [(float(a["px"]), float(a["sz"])) for a in asks_raw[:n_levels]]

    best_bid = bids[0][0] if bids else 0.0
    best_ask = asks[0][0] if asks else 0.0
    mid = (best_bid + best_ask) / 2 if (best_bid and best_ask) else 0.0
    spread_bps = ((best_ask - best_bid) / mid * 10_000) if mid else 0.0

    return {
        "coin": coin,
        "mid_price": mid,
        "spread_bps": spread_bps,
        "bid_depth_usd": sum(px * sz for px, sz in bids),
        "ask_depth_usd": sum(px * sz for px, sz in asks),
        "bids": bids,
        "asks": asks,
    }


def fetch_predicted_funding() -> pl.DataFrame:
    """Fetch predicted next funding rates from Hyperliquid.

    POST {"type": "predictedFundings"}
    Output columns: [timestamp, venue, coin, predicted_rate]
    """
    data = _hl_post({"type": "predictedFundings"})
    now = datetime.now(timezone.utc)
    rows = []
    # data is a list of [venue_name, [[coin, predicted_rate], ...]]
    for entry in data:
        venue_name = entry[0].lower() if isinstance(entry[0], str) else "unknown"
        pairs = entry[1] if len(entry) > 1 else []
        for pair in pairs:
            coin, rate = pair[0], float(pair[1])
            rows.append(
                {
                    "timestamp": now,
                    "venue": venue_name,
                    "coin": coin,
                    "predicted_rate": rate,
                }
            )
    if not rows:
        return pl.DataFrame(
            schema={
                "timestamp": pl.Datetime("us", "UTC"),
                "venue": pl.Utf8,
                "coin": pl.Utf8,
                "predicted_rate": pl.Float64,
            }
        )
    return pl.DataFrame(rows).cast({"timestamp": pl.Datetime("us", "UTC")})


# ═══════════════════════════════════════════════════════════════════════
# 0xArchive REST API (Hyperliquid + Lighter historical data)
# ═══════════════════════════════════════════════════════════════════════


def _0xa_get(
    path: str,
    params: dict[str, Any] | None = None,
    paginate: bool = False,
) -> list[dict]:
    """GET from 0xArchive with optional auto-pagination.

    Returns list of data dicts. All field values come as strings.
    """
    headers = {}
    if OXARCHIVE_KEY:
        headers["X-API-Key"] = OXARCHIVE_KEY
    params = dict(params or {})
    all_data: list[dict] = []

    while True:
        resp = requests.get(
            f"{OXARCHIVE_API}{path}",
            params=params,
            headers=headers,
            timeout=30,
        )
        resp.raise_for_status()
        body = resp.json()
        data = body.get("data", [])
        all_data.extend(data)

        if not paginate:
            break
        next_cursor = body.get("meta", {}).get("next_cursor")
        if not next_cursor or not data:
            break
        params["cursor"] = next_cursor
        time.sleep(0.1)

    return all_data


# ── 0xArchive funding ──


def _parse_0xa_ts(col: str = "timestamp") -> pl.Expr:
    """Parse 0xArchive ISO-8601 timestamp strings to Datetime."""
    return pl.col(col).str.to_datetime("%+", time_zone="UTC")


def _epoch_ms_to_dt(col: str) -> pl.Expr:
    """Convert epoch-millisecond column to Datetime(us, UTC)."""
    return (
        pl.from_epoch(pl.col(col).cast(pl.Int64), time_unit="ms")
        .dt.cast_time_unit("us")
        .dt.replace_time_zone("UTC")
    )


def _fetch_funding_0xa(
    coin: str,
    venue_path: str,
    venue_name: str,
    days: int = 90,
) -> pl.DataFrame:
    """Fetch funding from 0xArchive for Hyperliquid or Lighter.

    Both report 8h rates but settle hourly (paying rate/8 each hour).
    """
    end_ms = int(time.time() * 1000)
    start_ms = end_ms - (days * 24 * 3600 * 1000)
    data = _0xa_get(
        f"/v1/{venue_path}/funding/{coin}",
        params={
            "start": str(start_ms),
            "end": str(end_ms),
            "interval": "1h",
            "limit": "1000",
        },
        paginate=True,
    )
    if not data:
        return _empty(_FUNDING_SCHEMA)

    df = pl.DataFrame(data)
    rate_expr = pl.col("funding_rate").cast(pl.Float64)
    # Lighter API returns rates in percentage points; convert to decimal
    if venue_name == "lighter":
        rate_expr = rate_expr / 100
    df = df.with_columns(
        _parse_0xa_ts().alias("timestamp"),
        rate_expr.alias("funding_rate"),
        pl.lit(venue_name).alias("venue"),
        pl.lit(coin).alias("coin"),
    ).select("timestamp", "venue", "coin", "funding_rate")
    return df.unique(subset=["timestamp", "coin"]).sort("timestamp")


# ── 0xArchive candles ──


def _fetch_candles_0xa(
    coin: str,
    venue_path: str,
    venue_name: str,
    interval: str = "1h",
    days: int = 90,
) -> pl.DataFrame:
    """Fetch candles from 0xArchive for Hyperliquid or Lighter."""
    end_ms = int(time.time() * 1000)
    start_ms = end_ms - (days * 24 * 3600 * 1000)
    data = _0xa_get(
        f"/v1/{venue_path}/candles/{coin}",
        params={
            "start": str(start_ms),
            "end": str(end_ms),
            "interval": interval,
            "limit": "1000",
        },
        paginate=True,
    )
    if not data:
        return _empty(_CANDLE_SCHEMA)

    df = pl.DataFrame(data)
    df = df.with_columns(
        _parse_0xa_ts().alias("timestamp"),
        pl.lit(venue_name).alias("venue"),
        pl.lit(coin).alias("coin"),
        pl.col("open").cast(pl.Float64).alias("o"),
        pl.col("high").cast(pl.Float64).alias("h"),
        pl.col("low").cast(pl.Float64).alias("l"),
        pl.col("close").cast(pl.Float64).alias("c"),
        pl.col("volume").cast(pl.Float64).alias("v"),
    ).select("timestamp", "venue", "coin", "o", "h", "l", "c", "v")
    return df.unique(subset=["timestamp", "coin"]).sort("timestamp")


# ── 0xArchive open interest ──


def _fetch_oi_0xa(
    coin: str,
    venue_path: str,
    venue_name: str,
    days: int = 90,
) -> pl.DataFrame:
    """Fetch historical OI from 0xArchive for Hyperliquid or Lighter.

    Hyperliquid returns OI in base asset (e.g. BTC) → multiply by mark_price.
    Lighter returns OI already in USD → use directly.
    """
    end_ms = int(time.time() * 1000)
    start_ms = end_ms - (days * 24 * 3600 * 1000)
    data = _0xa_get(
        f"/v1/{venue_path}/openinterest/{coin}",
        params={
            "start": str(start_ms),
            "end": str(end_ms),
            "interval": "1h",
            "limit": "1000",
        },
        paginate=True,
    )
    if not data:
        return _empty(_OI_SCHEMA)

    df = pl.DataFrame(data)

    if venue_name == "lighter":
        # Lighter OI is already denominated in USD (quote currency)
        oi_expr = pl.col("open_interest").cast(pl.Float64).alias("oi_usd")
    else:
        # Hyperliquid OI is in base asset — convert to USD
        oi_expr = (
            pl.col("open_interest").cast(pl.Float64)
            * pl.col("mark_price").cast(pl.Float64)
        ).alias("oi_usd")

    df = df.with_columns(
        _parse_0xa_ts().alias("timestamp"),
        pl.lit(venue_name).alias("venue"),
        pl.lit(coin).alias("coin"),
        oi_expr,
    ).select("timestamp", "venue", "coin", "oi_usd")
    return df.unique(subset=["timestamp", "coin"]).sort("timestamp")


# ── 0xArchive prices (for lending module) ──


def _fetch_prices_0xa(
    coin: str,
    venue_path: str = "hyperliquid",
    days: int = 90,
) -> pl.DataFrame:
    """Fetch price history from 0xArchive. Returns [timestamp, markPrice]."""
    end_ms = int(time.time() * 1000)
    start_ms = end_ms - (days * 24 * 3600 * 1000)
    data = _0xa_get(
        f"/v1/{venue_path}/prices/{coin}",
        params={"start": str(start_ms), "end": str(end_ms), "limit": "1000"},
        paginate=True,
    )
    if not data:
        return pl.DataFrame(
            schema={
                "timestamp": pl.Datetime("us", "UTC"),
                "mark_price": pl.Float64,
            }
        )
    df = pl.DataFrame(data)
    df = df.with_columns(
        _parse_0xa_ts().alias("timestamp"),
        pl.col("mark_price").cast(pl.Float64),
    ).select("timestamp", "mark_price")
    return df.sort("timestamp")


# ═══════════════════════════════════════════════════════════════════════
# OKX (free, no auth, works from US)
# ═══════════════════════════════════════════════════════════════════════


def _okx_get(path: str, params: dict[str, Any] | None = None) -> list:
    """GET from OKX public API."""
    resp = requests.get(f"{OKX_API}{path}", params=params, timeout=30)
    resp.raise_for_status()
    body = resp.json()
    return body.get("data", [])


def _fetch_funding_okx(coin: str, days: int = 90) -> pl.DataFrame:
    """Fetch historical funding rates from OKX (~90 day max)."""
    sym = OKX_SYMBOLS.get(coin)
    if not sym:
        return _empty(_FUNDING_SCHEMA)

    cutoff_ms = int(time.time() * 1000) - (days * 24 * 3600 * 1000)
    all_data: list[dict] = []
    after = ""

    while True:
        params: dict[str, Any] = {"instId": sym, "limit": "100"}
        if after:
            params["after"] = after
        data = _okx_get("/api/v5/public/funding-rate-history", params)
        if not data:
            break
        all_data.extend(data)
        # OKX returns newest first; stop if we've reached our cutoff
        oldest_ts = int(data[-1].get("fundingTime", "0"))
        if oldest_ts < cutoff_ms:
            break
        after = data[-1].get("fundingTime", "")
        time.sleep(0.25)

    if not all_data:
        return _empty(_FUNDING_SCHEMA)

    df = pl.DataFrame(all_data)
    df = df.with_columns(
        _epoch_ms_to_dt("fundingTime").alias("timestamp"),
        pl.col("fundingRate").cast(pl.Float64).alias("funding_rate"),
        pl.lit("okx").alias("venue"),
        pl.lit(coin).alias("coin"),
    ).select("timestamp", "venue", "coin", "funding_rate")
    df = df.filter(
        pl.col("timestamp")
        >= pl.lit(datetime.fromtimestamp(cutoff_ms / 1000, tz=timezone.utc))
    )
    return df.unique(subset=["timestamp", "coin"]).sort("timestamp")


def _fetch_candles_okx(
    coin: str,
    interval: str = "1H",
    days: int = 90,
) -> pl.DataFrame:
    """Fetch candles from OKX."""
    sym = OKX_SYMBOLS.get(coin)
    if not sym:
        return _empty(_CANDLE_SCHEMA)

    cutoff_ms = int(time.time() * 1000) - (days * 24 * 3600 * 1000)
    all_data: list[list] = []
    after = ""

    while True:
        params: dict[str, Any] = {"instId": sym, "bar": interval, "limit": "100"}
        if after:
            params["after"] = after
        data = _okx_get("/api/v5/market/history-candles", params)
        if not data:
            break
        all_data.extend(data)
        oldest_ts = int(data[-1][0])
        if oldest_ts < cutoff_ms:
            break
        after = data[-1][0]
        time.sleep(0.25)

    if not all_data:
        return _empty(_CANDLE_SCHEMA)

    # OKX candle format: [ts, o, h, l, c, vol, volCcy, volCcyQuote, confirm]
    rows = []
    for c in all_data:
        rows.append(
            {
                "ts": int(c[0]),
                "o": float(c[1]),
                "h": float(c[2]),
                "l": float(c[3]),
                "c": float(c[4]),
                "v": float(c[5]),
            }
        )
    df = pl.DataFrame(rows)
    df = df.with_columns(
        _epoch_ms_to_dt("ts").alias("timestamp"),
        pl.lit("okx").alias("venue"),
        pl.lit(coin).alias("coin"),
    ).select("timestamp", "venue", "coin", "o", "h", "l", "c", "v")
    df = df.filter(
        pl.col("timestamp")
        >= pl.lit(datetime.fromtimestamp(cutoff_ms / 1000, tz=timezone.utc))
    )
    return df.unique(subset=["timestamp", "coin"]).sort("timestamp")


def _fetch_oi_okx(coin: str) -> pl.DataFrame:
    """Fetch current OI snapshot from OKX.

    /api/v5/public/open-interest returns oiUsd (already in USD).
    """
    sym = OKX_SYMBOLS.get(coin)
    if not sym:
        return _empty(_OI_SCHEMA)

    data = _okx_get("/api/v5/public/open-interest", {"instType": "SWAP", "instId": sym})
    if not data:
        return _empty(_OI_SCHEMA)

    oi_usd = float(data[0].get("oiUsd", 0))

    now = datetime.now(timezone.utc)
    return pl.DataFrame(
        [
            {
                "timestamp": now,
                "venue": "okx",
                "coin": coin,
                "oi_usd": oi_usd,
            }
        ]
    ).cast({"timestamp": pl.Datetime("us", "UTC")})


# ═══════════════════════════════════════════════════════════════════════
# Kraken Futures (free, no auth, works from US)
# ═══════════════════════════════════════════════════════════════════════


def _fetch_funding_kraken(coin: str, days: int = 90) -> pl.DataFrame:
    """Fetch historical funding rates from Kraken Futures.

    Kraken reports per-hour rates (relativeFundingRate), settled hourly.
    """
    sym = KRAKEN_SYMBOLS.get(coin)
    if not sym:
        return _empty(_FUNDING_SCHEMA)

    resp = requests.get(
        f"{KRAKEN_FUTURES_API}/derivatives/api/v4/historicalfundingrates",
        params={"symbol": sym},
        timeout=30,
    )
    resp.raise_for_status()
    body = resp.json()
    rates = body.get("rates", [])
    if not rates:
        return _empty(_FUNDING_SCHEMA)

    cutoff = datetime.now(timezone.utc).timestamp() - (days * 24 * 3600)
    filtered = [r for r in rates if r.get("timestamp", "") != ""]

    rows = []
    for r in filtered:
        ts_str = r.get("timestamp", "") or r.get("effectiveTime", "")
        try:
            ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        except (ValueError, TypeError):
            continue
        if ts.timestamp() < cutoff:
            continue
        # Use relativeFundingRate (per-hour rate); fundingRate is absolute USD
        rows.append(
            {
                "timestamp": ts,
                "funding_rate": float(r.get("relativeFundingRate", 0)),
            }
        )

    if not rows:
        return _empty(_FUNDING_SCHEMA)

    # Kraken reports 1h rates (relativeFundingRate), settled hourly
    df = pl.DataFrame(rows).cast({"timestamp": pl.Datetime("us", "UTC")})
    df = df.with_columns(
        pl.lit("kraken").alias("venue"),
        pl.lit(coin).alias("coin"),
    ).select("timestamp", "venue", "coin", "funding_rate")
    return df.unique(subset=["timestamp"]).sort("timestamp")


def _fetch_candles_kraken(
    coin: str,
    resolution: str = "1h",
    days: int = 90,
) -> pl.DataFrame:
    """Fetch candles from Kraken Futures."""
    sym = KRAKEN_SYMBOLS.get(coin)
    if not sym:
        return _empty(_CANDLE_SCHEMA)

    resp = requests.get(
        f"{KRAKEN_FUTURES_API}/api/charts/v1/trade/{sym}/{resolution}",
        timeout=30,
    )
    resp.raise_for_status()
    body = resp.json()
    candles = body.get("candles", [])
    if not candles:
        return _empty(_CANDLE_SCHEMA)

    cutoff_s = time.time() - (days * 24 * 3600)
    rows = []
    for c in candles:
        ts = c.get("time", 0) / 1000  # ms → s
        if ts < cutoff_s:
            continue
        rows.append(
            {
                "timestamp": int(c.get("time", 0)),
                "o": float(c.get("open", 0)),
                "h": float(c.get("high", 0)),
                "l": float(c.get("low", 0)),
                "c": float(c.get("close", 0)),
                "v": float(c.get("volume", 0)),
            }
        )
    if not rows:
        return _empty(_CANDLE_SCHEMA)

    df = pl.DataFrame(rows)
    df = df.with_columns(
        _epoch_ms_to_dt("timestamp").alias("timestamp"),
        pl.lit("kraken").alias("venue"),
        pl.lit(coin).alias("coin"),
    ).select("timestamp", "venue", "coin", "o", "h", "l", "c", "v")
    return df.unique(subset=["timestamp", "coin"]).sort("timestamp")


def _fetch_oi_kraken() -> pl.DataFrame:
    """Fetch current OI snapshot from Kraken Futures tickers."""
    resp = requests.get(
        f"{KRAKEN_FUTURES_API}/derivatives/api/v3/tickers",
        timeout=30,
    )
    resp.raise_for_status()
    body = resp.json()
    tickers = body.get("tickers", [])

    # Reverse lookup: symbol → coin
    sym_to_coin = {v: k for k, v in KRAKEN_SYMBOLS.items()}
    now = datetime.now(timezone.utc)
    rows = []
    for t in tickers:
        sym = t.get("symbol", "")
        coin = sym_to_coin.get(sym)
        if coin is None:
            continue
        oi = float(t.get("openInterest", 0))
        mark = float(t.get("markPrice", 0))
        rows.append(
            {
                "timestamp": now,
                "venue": "kraken",
                "coin": coin,
                "oi_usd": oi * mark,
            }
        )
    if not rows:
        return _empty(_OI_SCHEMA)
    return pl.DataFrame(rows).cast({"timestamp": pl.Datetime("us", "UTC")})


# ═══════════════════════════════════════════════════════════════════════
# Binance Futures (VPN required)
# ═══════════════════════════════════════════════════════════════════════


def _fetch_funding_binance(coin: str, days: int = 90) -> pl.DataFrame:
    """Fetch historical funding rates from Binance Futures."""
    sym = BINANCE_SYMBOLS.get(coin)
    if not sym:
        return _empty(_FUNDING_SCHEMA)

    end_ms = int(time.time() * 1000)
    start_ms = end_ms - (days * 24 * 3600 * 1000)
    all_data: list[dict] = []

    while start_ms < end_ms:
        resp = requests.get(
            f"{BINANCE_FAPI}/fundingRate",
            params={"symbol": sym, "startTime": start_ms, "limit": 1000},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        if not data:
            break
        all_data.extend(data)
        start_ms = int(data[-1]["fundingTime"]) + 1
        time.sleep(0.05)

    if not all_data:
        return _empty(_FUNDING_SCHEMA)

    df = pl.DataFrame(all_data)
    df = df.with_columns(
        _epoch_ms_to_dt("fundingTime").alias("timestamp"),
        pl.col("fundingRate").cast(pl.Float64).alias("funding_rate"),
        pl.lit("binance").alias("venue"),
        pl.lit(coin).alias("coin"),
    ).select("timestamp", "venue", "coin", "funding_rate")
    return df.unique(subset=["timestamp", "coin"]).sort("timestamp")


def _fetch_candles_binance(
    coin: str,
    interval: str = "1h",
    days: int = 90,
) -> pl.DataFrame:
    """Fetch candles from Binance Futures."""
    sym = BINANCE_SYMBOLS.get(coin)
    if not sym:
        return _empty(_CANDLE_SCHEMA)

    end_ms = int(time.time() * 1000)
    start_ms = end_ms - (days * 24 * 3600 * 1000)
    all_data: list[list] = []

    while start_ms < end_ms:
        resp = requests.get(
            f"{BINANCE_FAPI}/klines",
            params={
                "symbol": sym,
                "interval": interval,
                "startTime": start_ms,
                "limit": 1500,
            },
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        if not data:
            break
        all_data.extend(data)
        start_ms = int(data[-1][0]) + 1
        time.sleep(0.05)

    if not all_data:
        return _empty(_CANDLE_SCHEMA)

    # Binance kline: [openTime, o, h, l, c, vol, closeTime, ...]
    rows = [
        {
            "ts": int(c[0]),
            "o": float(c[1]),
            "h": float(c[2]),
            "l": float(c[3]),
            "c": float(c[4]),
            "v": float(c[5]),
        }
        for c in all_data
    ]
    df = pl.DataFrame(rows)
    df = df.with_columns(
        _epoch_ms_to_dt("ts").alias("timestamp"),
        pl.lit("binance").alias("venue"),
        pl.lit(coin).alias("coin"),
    ).select("timestamp", "venue", "coin", "o", "h", "l", "c", "v")
    return df.unique(subset=["timestamp", "coin"]).sort("timestamp")


def _fetch_oi_binance(coin: str) -> pl.DataFrame:
    """Fetch current OI from Binance Futures.

    /fapi/v1/openInterest returns OI in base asset (contracts).
    We fetch the mark price from /fapi/v1/premiumIndex to convert to USD.
    """
    sym = BINANCE_SYMBOLS.get(coin)
    if not sym:
        return _empty(_OI_SCHEMA)

    resp = requests.get(
        f"{BINANCE_FAPI}/openInterest",
        params={"symbol": sym},
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    oi_contracts = float(data.get("openInterest", 0))

    # Fetch mark price for USD conversion
    resp2 = requests.get(
        f"{BINANCE_FAPI}/premiumIndex",
        params={"symbol": sym},
        timeout=30,
    )
    resp2.raise_for_status()
    mark_price = float(resp2.json().get("markPrice", 0))

    now = datetime.now(timezone.utc)
    return pl.DataFrame(
        [
            {
                "timestamp": now,
                "venue": "binance",
                "coin": coin,
                "oi_usd": oi_contracts * mark_price,
            }
        ]
    ).cast({"timestamp": pl.Datetime("us", "UTC")})


# ═══════════════════════════════════════════════════════════════════════
# Bybit (VPN required)
# ═══════════════════════════════════════════════════════════════════════


def _fetch_funding_bybit(coin: str, days: int = 90) -> pl.DataFrame:
    """Fetch historical funding rates from Bybit."""
    sym = BYBIT_SYMBOLS.get(coin)
    if not sym:
        return _empty(_FUNDING_SCHEMA)

    end_ms = int(time.time() * 1000)
    start_ms = end_ms - (days * 24 * 3600 * 1000)
    all_data: list[dict] = []
    current_end = end_ms

    for _ in range(50):  # safety limit
        params: dict[str, Any] = {
            "category": "linear",
            "symbol": sym,
            "startTime": str(start_ms),
            "endTime": str(current_end),
            "limit": "200",
        }
        resp = requests.get(
            f"{BYBIT_API}/market/funding/history",
            params=params,
            timeout=30,
        )
        resp.raise_for_status()
        body = resp.json()
        result = body.get("result", {})
        data = result.get("list", [])
        if not data:
            break
        all_data.extend(data)
        # Bybit returns newest first; move endTime to oldest entry - 1
        oldest_ts = int(data[-1].get("fundingRateTimestamp", "0"))
        if oldest_ts <= start_ms:
            break
        current_end = oldest_ts - 1
        time.sleep(0.5)

    if not all_data:
        return _empty(_FUNDING_SCHEMA)

    df = pl.DataFrame(all_data)
    df = df.with_columns(
        _epoch_ms_to_dt("fundingRateTimestamp").alias("timestamp"),
        pl.col("fundingRate").cast(pl.Float64).alias("funding_rate"),
        pl.lit("bybit").alias("venue"),
        pl.lit(coin).alias("coin"),
    ).select("timestamp", "venue", "coin", "funding_rate")
    return df.unique(subset=["timestamp", "coin"]).sort("timestamp")


def _fetch_candles_bybit(
    coin: str,
    interval: str = "60",
    days: int = 90,
) -> pl.DataFrame:
    """Fetch candles from Bybit. interval='60' means 1h."""
    sym = BYBIT_SYMBOLS.get(coin)
    if not sym:
        return _empty(_CANDLE_SCHEMA)

    end_ms = int(time.time() * 1000)
    start_ms = end_ms - (days * 24 * 3600 * 1000)
    all_data: list[list] = []
    current_end = end_ms

    for _ in range(100):  # safety limit
        resp = requests.get(
            f"{BYBIT_API}/market/kline",
            params={
                "category": "linear",
                "symbol": sym,
                "interval": interval,
                "start": start_ms,
                "end": current_end,
                "limit": 200,
            },
            timeout=30,
        )
        resp.raise_for_status()
        body = resp.json()
        data = body.get("result", {}).get("list", [])
        if not data:
            break
        all_data.extend(data)
        # Bybit returns newest first; data[-1] is oldest in this page
        oldest_ts = int(data[-1][0])
        if oldest_ts <= start_ms:
            break
        current_end = oldest_ts - 1
        time.sleep(0.5)

    if not all_data:
        return _empty(_CANDLE_SCHEMA)

    # Bybit kline: [startTime, open, high, low, close, volume, turnover]
    rows = [
        {
            "ts": int(c[0]),
            "o": float(c[1]),
            "h": float(c[2]),
            "l": float(c[3]),
            "c": float(c[4]),
            "v": float(c[5]),
        }
        for c in all_data
    ]
    df = pl.DataFrame(rows)
    df = df.with_columns(
        _epoch_ms_to_dt("ts").alias("timestamp"),
        pl.lit("bybit").alias("venue"),
        pl.lit(coin).alias("coin"),
    ).select("timestamp", "venue", "coin", "o", "h", "l", "c", "v")
    return df.unique(subset=["timestamp", "coin"]).sort("timestamp")


def _fetch_oi_bybit(coin: str) -> pl.DataFrame:
    """Fetch current OI from Bybit.

    /v5/market/open-interest returns OI in base asset (contracts).
    We fetch the mark price from /v5/market/tickers to convert to USD.
    """
    sym = BYBIT_SYMBOLS.get(coin)
    if not sym:
        return _empty(_OI_SCHEMA)

    resp = requests.get(
        f"{BYBIT_API}/market/open-interest",
        params={
            "category": "linear",
            "symbol": sym,
            "intervalTime": "5min",
            "limit": 1,
        },
        timeout=30,
    )
    resp.raise_for_status()
    body = resp.json()
    data = body.get("result", {}).get("list", [])
    if not data:
        return _empty(_OI_SCHEMA)

    oi_contracts = float(data[0].get("openInterest", 0))

    # Fetch mark price for USD conversion
    resp2 = requests.get(
        f"{BYBIT_API}/market/tickers",
        params={"category": "linear", "symbol": sym},
        timeout=30,
    )
    resp2.raise_for_status()
    tickers = resp2.json().get("result", {}).get("list", [])
    mark_price = float(tickers[0].get("markPrice", 0)) if tickers else 0

    now = datetime.now(timezone.utc)
    return pl.DataFrame(
        [
            {
                "timestamp": now,
                "venue": "bybit",
                "coin": coin,
                "oi_usd": oi_contracts * mark_price,
            }
        ]
    ).cast({"timestamp": pl.Datetime("us", "UTC")})


# ═══════════════════════════════════════════════════════════════════════
# dYdX v4 (no auth, works from US)
# ═══════════════════════════════════════════════════════════════════════


def _fetch_funding_dydx(coin: str, days: int = 90) -> pl.DataFrame:
    """Fetch historical funding rates from dYdX v4."""
    sym = DYDX_SYMBOLS.get(coin)
    if not sym:
        return _empty(_FUNDING_SCHEMA)

    cutoff = datetime.now(timezone.utc).timestamp() - (days * 24 * 3600)
    all_data: list[dict] = []
    url = f"{DYDX_API}/historicalFunding/{sym}"

    max_pages = 20  # safety limit
    for _ in range(max_pages):
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        body = resp.json()
        records = body.get("historicalFunding", [])
        if not records:
            break
        all_data.extend(records)
        oldest = records[-1].get("effectiveAt", "")
        try:
            oldest_ts = datetime.fromisoformat(
                oldest.replace("Z", "+00:00")
            ).timestamp()
        except (ValueError, TypeError):
            break
        if oldest_ts < cutoff:
            break
        # dYdX paginates via effectiveAtBeforeOrAt
        url = f"{DYDX_API}/historicalFunding/{sym}?effectiveAtBeforeOrAt={oldest}"
        time.sleep(0.5)

    if not all_data:
        return _empty(_FUNDING_SCHEMA)

    rows = []
    for r in all_data:
        try:
            ts = datetime.fromisoformat(r["effectiveAt"].replace("Z", "+00:00"))
        except (ValueError, TypeError, KeyError):
            continue
        rows.append(
            {
                "timestamp": ts,
                "funding_rate": float(r.get("rate", 0)),
                "venue": "dydx",
                "coin": coin,
            }
        )
    if not rows:
        return _empty(_FUNDING_SCHEMA)

    df = pl.DataFrame(rows).cast({"timestamp": pl.Datetime("us", "UTC")})
    df = df.filter(
        pl.col("timestamp") >= pl.lit(datetime.fromtimestamp(cutoff, tz=timezone.utc))
    )
    # dYdX reports 1h rates, settled hourly
    df = (
        df.select("timestamp", "venue", "coin", "funding_rate")
        .unique(subset=["timestamp"])
        .sort("timestamp")
    )
    return df


def _fetch_candles_dydx(
    coin: str,
    interval: str = "1HOUR",
    days: int = 90,
) -> pl.DataFrame:
    """Fetch candles from dYdX v4."""
    sym = DYDX_SYMBOLS.get(coin)
    if not sym:
        return _empty(_CANDLE_SCHEMA)

    cutoff = datetime.now(timezone.utc).timestamp() - (days * 24 * 3600)
    resp = requests.get(
        f"{DYDX_API}/candles/perpetualMarkets/{sym}",
        params={"resolution": interval, "limit": 1000},
        timeout=30,
    )
    resp.raise_for_status()
    body = resp.json()
    candles = body.get("candles", [])
    if not candles:
        return _empty(_CANDLE_SCHEMA)

    rows = []
    for c in candles:
        try:
            ts = datetime.fromisoformat(c["startedAt"].replace("Z", "+00:00"))
        except (ValueError, TypeError, KeyError):
            continue
        if ts.timestamp() < cutoff:
            continue
        rows.append(
            {
                "timestamp": ts,
                "o": float(c.get("open", 0)),
                "h": float(c.get("high", 0)),
                "l": float(c.get("low", 0)),
                "c": float(c.get("close", 0)),
                "v": float(c.get("baseTokenVolume", 0)),
            }
        )
    if not rows:
        return _empty(_CANDLE_SCHEMA)

    df = pl.DataFrame(rows).cast({"timestamp": pl.Datetime("us", "UTC")})
    df = df.with_columns(
        pl.lit("dydx").alias("venue"),
        pl.lit(coin).alias("coin"),
    ).select("timestamp", "venue", "coin", "o", "h", "l", "c", "v")
    return df.sort("timestamp")


def _fetch_oi_dydx(coin: str) -> pl.DataFrame:
    """Fetch current OI from dYdX v4."""
    sym = DYDX_SYMBOLS.get(coin)
    if not sym:
        return _empty(_OI_SCHEMA)

    resp = requests.get(
        f"{DYDX_API}/perpetualMarkets",
        params={"ticker": sym},
        timeout=30,
    )
    resp.raise_for_status()
    body = resp.json()
    markets = body.get("markets", {})
    info = markets.get(sym, {})
    if not info:
        return _empty(_OI_SCHEMA)

    now = datetime.now(timezone.utc)
    oi = float(info.get("openInterest", 0))
    price = float(info.get("oraclePrice", 0))
    return pl.DataFrame(
        [
            {
                "timestamp": now,
                "venue": "dydx",
                "coin": coin,
                "oi_usd": oi * price,
            }
        ]
    ).cast({"timestamp": pl.Datetime("us", "UTC")})


# ═══════════════════════════════════════════════════════════════════════
# Aggregators — dispatch to all venues, concat results
# ═══════════════════════════════════════════════════════════════════════


def fetch_funding_rates(
    coins: list[str] | None = None,
    venues: list[str] | None = None,
    days: int = 90,
) -> pl.DataFrame:
    """Fetch funding rates for all coins x venues.

    Returns polars DataFrame: [timestamp, venue, coin, funding_rate]
    """
    coins = coins or TOKENS
    venues = venues or VENUES

    frames: list[pl.DataFrame] = []
    for venue in venues:
        for coin in coins:
            try:
                df = _dispatch_funding(venue, coin, days)
                if df is not None and not df.is_empty():
                    frames.append(df)
            except Exception as e:
                logger.warning(f"Funding fetch failed: {venue}/{coin}: {e}")

    if not frames:
        return _empty(_FUNDING_SCHEMA)
    return pl.concat(frames).sort("timestamp")


def _dispatch_funding(venue: str, coin: str, days: int) -> pl.DataFrame | None:
    if venue == "hyperliquid":
        return _fetch_funding_0xa(coin, "hyperliquid", "hyperliquid", days)
    elif venue == "lighter":
        return _fetch_funding_0xa(coin, "lighter", "lighter", days)
    elif venue == "okx":
        return _fetch_funding_okx(coin, days)
    elif venue == "kraken":
        return _fetch_funding_kraken(coin, days)
    elif venue == "binance":
        return _fetch_funding_binance(coin, days)
    elif venue == "bybit":
        return _fetch_funding_bybit(coin, days)
    elif venue == "dydx":
        return _fetch_funding_dydx(coin, days)
    return None


def fetch_candles(
    coins: list[str] | None = None,
    venues: list[str] | None = None,
    interval: str = "1h",
    days: int = 90,
) -> pl.DataFrame:
    """Fetch candle data for all coins x venues.

    Returns polars DataFrame: [timestamp, venue, coin, o, h, l, c, v]
    """
    coins = coins or TOKENS
    venues = venues or VENUES

    frames: list[pl.DataFrame] = []
    for venue in venues:
        for coin in coins:
            try:
                df = _dispatch_candles(venue, coin, interval, days)
                if df is not None and not df.is_empty():
                    frames.append(df)
            except Exception as e:
                logger.warning(f"Candle fetch failed: {venue}/{coin}: {e}")

    if not frames:
        return _empty(_CANDLE_SCHEMA)
    return pl.concat(frames).sort("timestamp")


def _dispatch_candles(
    venue: str,
    coin: str,
    interval: str,
    days: int,
) -> pl.DataFrame | None:
    if venue == "hyperliquid":
        return _fetch_candles_0xa(coin, "hyperliquid", "hyperliquid", interval, days)
    elif venue == "lighter":
        return _fetch_candles_0xa(coin, "lighter", "lighter", interval, days)
    elif venue == "okx":
        # OKX uses "1H" format
        okx_interval = interval.upper() if interval else "1H"
        return _fetch_candles_okx(coin, okx_interval, days)
    elif venue == "kraken":
        return _fetch_candles_kraken(coin, interval, days)
    elif venue == "binance":
        return _fetch_candles_binance(coin, interval, days)
    elif venue == "bybit":
        # Bybit uses minutes: "60" for 1h
        bybit_map = {"1h": "60", "4h": "240", "1d": "D", "15m": "15", "5m": "5"}
        bybit_interval = bybit_map.get(interval.lower(), "60")
        return _fetch_candles_bybit(coin, bybit_interval, days)
    elif venue == "dydx":
        # dYdX uses "1HOUR", "4HOURS", "1DAY"
        dydx_map = {
            "1h": "1HOUR",
            "4h": "4HOURS",
            "1d": "1DAY",
            "15m": "15MINS",
            "5m": "5MINS",
        }
        dydx_interval = dydx_map.get(interval.lower(), "1HOUR")
        return _fetch_candles_dydx(coin, dydx_interval, days)
    return None


def fetch_open_interest(
    coins: list[str] | None = None,
    venues: list[str] | None = None,
    days: int = 90,
) -> pl.DataFrame:
    """Fetch open interest for all coins x venues.

    Returns polars DataFrame: [timestamp, venue, coin, oi_usd]
    Historical from 0xArchive; snapshots from live APIs.
    """
    coins = coins or TOKENS
    venues = venues or VENUES

    frames: list[pl.DataFrame] = []
    for venue in venues:
        try:
            if venue == "hyperliquid":
                for coin in coins:
                    df = _fetch_oi_0xa(coin, "hyperliquid", "hyperliquid", days)
                    if not df.is_empty():
                        frames.append(df)
            elif venue == "lighter":
                for coin in coins:
                    df = _fetch_oi_0xa(coin, "lighter", "lighter", days)
                    if not df.is_empty():
                        frames.append(df)
            elif venue == "okx":
                for coin in coins:
                    df = _fetch_oi_okx(coin)
                    if not df.is_empty():
                        frames.append(df)
            elif venue == "kraken":
                df = _fetch_oi_kraken()
                if not df.is_empty():
                    # Filter to requested coins
                    df = df.filter(pl.col("coin").is_in(coins))
                    if not df.is_empty():
                        frames.append(df)
            elif venue == "binance":
                for coin in coins:
                    df = _fetch_oi_binance(coin)
                    if not df.is_empty():
                        frames.append(df)
            elif venue == "bybit":
                for coin in coins:
                    df = _fetch_oi_bybit(coin)
                    if not df.is_empty():
                        frames.append(df)
            elif venue == "dydx":
                for coin in coins:
                    df = _fetch_oi_dydx(coin)
                    if not df.is_empty():
                        frames.append(df)
        except Exception as e:
            logger.warning(f"OI fetch failed: {venue}: {e}")

    if not frames:
        return _empty(_OI_SCHEMA)
    return pl.concat(frames).sort("timestamp")


# ═══════════════════════════════════════════════════════════════════════
# Orderbook depth (multi-venue via 0xArchive)
# ═══════════════════════════════════════════════════════════════════════


def _bid_depth_1pct(bids: list[dict], mid: float) -> float:
    """Sum bid-side USD within 1% of mid price."""
    cutoff = mid * 0.99
    return sum(
        float(b["px"]) * float(b["sz"]) for b in bids if float(b["px"]) >= cutoff
    )


def fetch_orderbook_depth_all(
    coins: list[str] | None = None,
) -> pl.DataFrame:
    """Fetch 1%-depth from Hyperliquid + Lighter orderbooks via 0xArchive.

    Returns DataFrame: [coin, venue, bid_depth_usd, mid_price].
    Aggregated depth per coin is sum across venues.
    """
    coins = coins or TOKENS
    rows: list[dict] = []

    for coin in coins:
        for venue in ("hyperliquid", "lighter"):
            try:
                path = f"/v1/{venue}/orderbook/{coin}"
                resp = requests.get(
                    f"{OXARCHIVE_API}{path}",
                    params={"depth": "50"},
                    headers={"X-API-Key": OXARCHIVE_KEY} if OXARCHIVE_KEY else {},
                    timeout=30,
                )
                resp.raise_for_status()
                book = resp.json().get("data", {})
                bids = book.get("bids", [])
                mid = float(book.get("mid_price", 0) or book.get("midPrice", 0))
                if mid <= 0:
                    continue
                rows.append(
                    {
                        "coin": coin,
                        "venue": venue,
                        "bid_depth_usd": _bid_depth_1pct(bids, mid),
                        "mid_price": mid,
                    }
                )
            except Exception as e:
                logger.warning(f"Orderbook depth fetch failed: {venue}/{coin}: {e}")

    if not rows:
        return pl.DataFrame(
            schema={
                "coin": pl.Utf8,
                "venue": pl.Utf8,
                "bid_depth_usd": pl.Float64,
                "mid_price": pl.Float64,
            }
        )
    return pl.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════
# Historical liquidation volume (0xArchive — Hyperliquid)
# ═══════════════════════════════════════════════════════════════════════


def fetch_liquidation_volume(
    coins: list[str] | None = None,
    days: int = 90,
    interval: str = "1h",
) -> pl.DataFrame:
    """Fetch hourly liquidation volume from 0xArchive (Hyperliquid).

    Returns DataFrame: [timestamp, coin, total_usd, long_usd, short_usd, count].
    """
    coins = coins or TOKENS
    end_ms = int(time.time() * 1000)
    start_ms = end_ms - (days * 24 * 3600 * 1000)
    frames: list[pl.DataFrame] = []

    for coin in coins:
        try:
            data = _0xa_get(
                f"/v1/hyperliquid/liquidations/volume/{coin}",
                params={
                    "start": str(start_ms),
                    "end": str(end_ms),
                    "interval": interval,
                    "limit": "1000",
                },
                paginate=True,
            )
            if not data:
                continue
            df = (
                pl.DataFrame(data)
                .with_columns(
                    _parse_0xa_ts().alias("timestamp"),
                    pl.lit(coin).alias("coin"),
                    pl.col("totalUsd").cast(pl.Float64).alias("total_usd"),
                    pl.col("longUsd").cast(pl.Float64).alias("long_usd"),
                    pl.col("shortUsd").cast(pl.Float64).alias("short_usd"),
                    pl.col("count").cast(pl.Int64).alias("liq_count"),
                )
                .select(
                    "timestamp",
                    "coin",
                    "total_usd",
                    "long_usd",
                    "short_usd",
                    "liq_count",
                )
            )
            frames.append(df)
        except Exception as e:
            logger.warning(f"Liquidation volume fetch failed: {coin}: {e}")

    if not frames:
        return pl.DataFrame(
            schema={
                "timestamp": pl.Datetime("us", "UTC"),
                "coin": pl.Utf8,
                "total_usd": pl.Float64,
                "long_usd": pl.Float64,
                "short_usd": pl.Float64,
                "liq_count": pl.Int64,
            }
        )
    return pl.concat(frames).sort("timestamp")
