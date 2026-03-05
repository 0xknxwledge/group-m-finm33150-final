#!/usr/bin/env python
"""Pull historical data from all venues and save to data/*.parquet.

Usage:
    # Full historical pull (0xArchive back to Apr 2023 + all venues recent)
    python scripts/pull_data.py

    # Quick pull — recent 90 days only, skip deep history
    python scripts/pull_data.py --quick

    # Single venue / coin for debugging
    python scripts/pull_data.py --venues hyperliquid --coins BTC --days 7

Requires:
    OXARCHIVE_API_KEY env var (or set in .env)
    VPN for Binance/Bybit (script continues gracefully if they fail)

Output:
    data/funding_rates.parquet
    data/candles.parquet
    data/open_interest.parquet
    data/liquidations.parquet
    data/reserve_prices.parquet

Each file has a 'venue' column so you can filter downstream.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Ensure src/ is importable when running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import polars as pl

from funding_the_fall.data.fetchers import (
    TOKENS,
    VENUES,
    _0xa_get,
    _empty,
    _FUNDING_SCHEMA,
    _CANDLE_SCHEMA,
    _OI_SCHEMA,
    _LIQ_SCHEMA,
    _fetch_funding_0xa,
    _fetch_candles_0xa,
    _fetch_oi_0xa,
    _fetch_liquidations_0xa,
    _fetch_funding_kraken,
    _fetch_candles_kraken,
    _fetch_oi_kraken,
    _fetch_funding_okx,
    _fetch_candles_okx,
    _fetch_oi_okx,
    _fetch_liquidations_okx,
    _fetch_funding_binance,
    _fetch_candles_binance,
    _fetch_oi_binance,
    _fetch_liquidations_binance,
    _fetch_funding_bybit,
    _fetch_candles_bybit,
    _fetch_oi_bybit,
    _fetch_funding_dydx,
    _fetch_candles_dydx,
    _fetch_oi_dydx,
)
from funding_the_fall.data.lending import fetch_reserve_prices
from funding_the_fall.data.storage import DATA_DIR, save_parquet_pl

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("pull_data")

# 0xArchive build tier allows up to 364 days of history
DEEP_HISTORY_DAYS = 364


def _safe(fn, *args, label: str = "") -> pl.DataFrame:
    """Call a fetcher, return empty on failure."""
    try:
        df = fn(*args)
        if df is not None and not df.is_empty():
            log.info(f"  OK  {label}: {len(df)} rows")
            return df
        log.info(f"  --  {label}: empty")
    except Exception as e:
        log.warning(f"  ERR {label}: {e}")
    return None


def pull_funding(coins: list[str], venues: list[str], days: int) -> pl.DataFrame:
    """Pull funding rates from all venues."""
    frames: list[pl.DataFrame] = []

    for coin in coins:
        # 0xArchive deep history (Hyperliquid + Lighter)
        if "hyperliquid" in venues:
            df = _safe(_fetch_funding_0xa, coin, "hyperliquid", "hyperliquid", days,
                       label=f"funding/hyperliquid/{coin}")
            if df is not None:
                frames.append(df)

        if "lighter" in venues:
            df = _safe(_fetch_funding_0xa, coin, "lighter", "lighter", days,
                       label=f"funding/lighter/{coin}")
            if df is not None:
                frames.append(df)

        # Kraken — full history available
        if "kraken" in venues:
            df = _safe(_fetch_funding_kraken, coin, days,
                       label=f"funding/kraken/{coin}")
            if df is not None:
                frames.append(df)

        # OKX — ~90 day max
        if "okx" in venues:
            df = _safe(_fetch_funding_okx, coin, min(days, 90),
                       label=f"funding/okx/{coin}")
            if df is not None:
                frames.append(df)

        # VPN venues — may fail
        if "binance" in venues:
            df = _safe(_fetch_funding_binance, coin, days,
                       label=f"funding/binance/{coin}")
            if df is not None:
                frames.append(df)

        if "bybit" in venues:
            df = _safe(_fetch_funding_bybit, coin, days,
                       label=f"funding/bybit/{coin}")
            if df is not None:
                frames.append(df)

        if "dydx" in venues:
            df = _safe(_fetch_funding_dydx, coin, days,
                       label=f"funding/dydx/{coin}")
            if df is not None:
                frames.append(df)

    if not frames:
        return _empty(_FUNDING_SCHEMA)
    return pl.concat(frames).unique(subset=["timestamp", "venue", "coin"]).sort("timestamp")


def pull_candles(coins: list[str], venues: list[str], days: int) -> pl.DataFrame:
    """Pull candles from all venues."""
    frames: list[pl.DataFrame] = []

    for coin in coins:
        if "hyperliquid" in venues:
            df = _safe(_fetch_candles_0xa, coin, "hyperliquid", "hyperliquid", "1h", days,
                       label=f"candles/hyperliquid/{coin}")
            if df is not None:
                frames.append(df)

        if "lighter" in venues:
            df = _safe(_fetch_candles_0xa, coin, "lighter", "lighter", "1h", days,
                       label=f"candles/lighter/{coin}")
            if df is not None:
                frames.append(df)

        if "kraken" in venues:
            df = _safe(_fetch_candles_kraken, coin, "1h", days,
                       label=f"candles/kraken/{coin}")
            if df is not None:
                frames.append(df)

        if "okx" in venues:
            df = _safe(_fetch_candles_okx, coin, "1H", min(days, 90),
                       label=f"candles/okx/{coin}")
            if df is not None:
                frames.append(df)

        if "binance" in venues:
            df = _safe(_fetch_candles_binance, coin, "1h", days,
                       label=f"candles/binance/{coin}")
            if df is not None:
                frames.append(df)

        if "bybit" in venues:
            df = _safe(_fetch_candles_bybit, coin, "60", days,
                       label=f"candles/bybit/{coin}")
            if df is not None:
                frames.append(df)

        if "dydx" in venues:
            df = _safe(_fetch_candles_dydx, coin, "1HOUR", days,
                       label=f"candles/dydx/{coin}")
            if df is not None:
                frames.append(df)

    if not frames:
        return _empty(_CANDLE_SCHEMA)
    return pl.concat(frames).unique(subset=["timestamp", "venue", "coin"]).sort("timestamp")


def pull_oi(coins: list[str], venues: list[str], days: int) -> pl.DataFrame:
    """Pull open interest from all venues."""
    frames: list[pl.DataFrame] = []

    for coin in coins:
        # Historical OI from 0xArchive
        if "hyperliquid" in venues:
            df = _safe(_fetch_oi_0xa, coin, "hyperliquid", "hyperliquid", days,
                       label=f"oi/hyperliquid/{coin}")
            if df is not None:
                frames.append(df)

        if "lighter" in venues:
            df = _safe(_fetch_oi_0xa, coin, "lighter", "lighter", days,
                       label=f"oi/lighter/{coin}")
            if df is not None:
                frames.append(df)

        # Snapshot venues (one call each)
        if "okx" in venues:
            df = _safe(_fetch_oi_okx, coin, label=f"oi/okx/{coin}")
            if df is not None:
                frames.append(df)

        if "binance" in venues:
            df = _safe(_fetch_oi_binance, coin, label=f"oi/binance/{coin}")
            if df is not None:
                frames.append(df)

        if "bybit" in venues:
            df = _safe(_fetch_oi_bybit, coin, label=f"oi/bybit/{coin}")
            if df is not None:
                frames.append(df)

        if "dydx" in venues:
            df = _safe(_fetch_oi_dydx, coin, label=f"oi/dydx/{coin}")
            if df is not None:
                frames.append(df)

    # Kraken OI is a single call for all symbols
    if "kraken" in venues:
        df = _safe(_fetch_oi_kraken, label="oi/kraken/all")
        if df is not None:
            df = df.filter(pl.col("coin").is_in(coins))
            if not df.is_empty():
                frames.append(df)

    if not frames:
        return _empty(_OI_SCHEMA)
    return pl.concat(frames).sort("timestamp")


def pull_liquidations(coins: list[str], days: int) -> pl.DataFrame:
    """Pull liquidations from available sources."""
    frames: list[pl.DataFrame] = []

    for coin in coins:
        df = _safe(_fetch_liquidations_0xa, coin, days,
                   label=f"liq/0xa/{coin}")
        if df is not None:
            frames.append(df)

        df = _safe(_fetch_liquidations_okx, coin, label=f"liq/okx/{coin}")
        if df is not None:
            frames.append(df)

        df = _safe(_fetch_liquidations_binance, coin, days,
                   label=f"liq/binance/{coin}")
        if df is not None:
            frames.append(df)

    if not frames:
        return _empty(_LIQ_SCHEMA)
    return pl.concat(frames).sort("timestamp")


def validate(name: str, df: pl.DataFrame) -> None:
    """Print coverage summary for a dataset."""
    if df.is_empty():
        log.warning(f"  {name}: EMPTY")
        return

    venues = df["venue"].unique().sort().to_list()
    coins = df["coin"].unique().sort().to_list()
    ts_min = df["timestamp"].min()
    ts_max = df["timestamp"].max()
    log.info(f"  {name}: {len(df)} rows | {ts_min} → {ts_max}")
    log.info(f"    venues: {venues}")
    log.info(f"    coins:  {coins}")

    # Per-venue row counts
    counts = df.group_by("venue").len().sort("venue")
    for row in counts.iter_rows(named=True):
        log.info(f"    {row['venue']:15s} {row['len']:>8,d} rows")


def main():
    parser = argparse.ArgumentParser(description="Pull historical data to data/*.parquet")
    parser.add_argument("--quick", action="store_true",
                        help="Recent 90 days only (skip deep 0xArchive history)")
    parser.add_argument("--days", type=int, default=None,
                        help="Override lookback in days")
    parser.add_argument("--venues", nargs="+", default=None,
                        help="Venues to pull (default: all)")
    parser.add_argument("--coins", nargs="+", default=None,
                        help="Coins to pull (default: all 5)")
    parser.add_argument("--skip-candles", action="store_true",
                        help="Skip candle pull (large dataset)")
    args = parser.parse_args()

    coins = args.coins or TOKENS
    venues = args.venues or VENUES

    if args.days is not None:
        days = args.days
    elif args.quick:
        days = 90
    else:
        days = DEEP_HISTORY_DAYS

    log.info(f"=== Data Pull: {len(coins)} coins × {len(venues)} venues, {days} days ===")
    log.info(f"Coins:  {coins}")
    log.info(f"Venues: {venues}")
    log.info(f"Output: {DATA_DIR}/")
    log.info("")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # ── Funding ──
    log.info("── Funding Rates ──")
    funding = pull_funding(coins, venues, days)
    if not funding.is_empty():
        save_parquet_pl(funding, "funding_rates")
    validate("funding_rates", funding)
    log.info("")

    # ── Candles ──
    if not args.skip_candles:
        log.info("── Candles ──")
        candles = pull_candles(coins, venues, days)
        if not candles.is_empty():
            save_parquet_pl(candles, "candles")
        validate("candles", candles)
        log.info("")

    # ── Open Interest ──
    log.info("── Open Interest ──")
    oi = pull_oi(coins, venues, days)
    if not oi.is_empty():
        save_parquet_pl(oi, "open_interest")
    validate("open_interest", oi)
    log.info("")

    # ── Liquidations ──
    log.info("── Liquidations ──")
    liqs = pull_liquidations(coins, days)
    if not liqs.is_empty():
        save_parquet_pl(liqs, "liquidations")
    validate("liquidations", liqs)
    log.info("")

    # ── Reserve Prices ──
    log.info("── Reserve Prices ──")
    try:
        prices = fetch_reserve_prices(days=days)
        if not prices.is_empty():
            log.info(f"  reserve_prices: {len(prices)} rows")
        else:
            log.warning("  reserve_prices: EMPTY")
    except Exception as e:
        log.warning(f"  reserve_prices: {e}")
    log.info("")

    elapsed = time.time() - t0
    log.info(f"=== Done in {elapsed:.0f}s ===")

    # ── Final coverage matrix ──
    log.info("")
    log.info("── Coverage Matrix (funding) ──")
    if not funding.is_empty():
        matrix = funding.group_by("venue", "coin").len().sort("venue", "coin")
        # Pivot for readability
        for venue in sorted(funding["venue"].unique().to_list()):
            v_data = matrix.filter(pl.col("venue") == venue)
            coin_counts = {r["coin"]: r["len"] for r in v_data.iter_rows(named=True)}
            counts_str = "  ".join(f"{c}:{coin_counts.get(c, 0):>5d}" for c in coins)
            log.info(f"  {venue:15s}  {counts_str}")


if __name__ == "__main__":
    main()
