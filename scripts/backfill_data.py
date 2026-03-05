#!/usr/bin/env python
"""Backfill liquidations (0xArchive + OKX) and re-pull dYdX funding with 8h aggregation.

Merges new data into existing parquets without re-pulling everything.

Usage:
    PYTHONPATH=src python scripts/backfill_data.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import polars as pl

from funding_the_fall.data.fetchers import (
    TOKENS,
    _empty,
    _FUNDING_SCHEMA,
    _LIQ_SCHEMA,
    _fetch_liquidations_0xa,
    _fetch_liquidations_okx,
    _fetch_liquidations_binance,
    _fetch_funding_dydx,
)
from funding_the_fall.data.storage import DATA_DIR, save_parquet_pl, load_parquet_pl

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("backfill")


def _safe(fn, *args, label: str = "") -> pl.DataFrame | None:
    try:
        df = fn(*args)
        if df is not None and not df.is_empty():
            log.info(f"  OK  {label}: {len(df)} rows")
            return df
        log.info(f"  --  {label}: empty")
    except Exception as e:
        log.warning(f"  ERR {label}: {e}")
    return None


def backfill_liquidations():
    """Re-pull liquidations from 0xArchive (capped to 300 days) + OKX + Binance."""
    log.info("── Backfilling Liquidations ──")
    frames: list[pl.DataFrame] = []

    for coin in TOKENS:
        # 0xArchive — Hyperliquid liquidations (May 2025+, capped at 300 days)
        df = _safe(_fetch_liquidations_0xa, coin, 300, label=f"liq/0xa/{coin}")
        if df is not None:
            frames.append(df)

        # OKX
        df = _safe(_fetch_liquidations_okx, coin, label=f"liq/okx/{coin}")
        if df is not None:
            frames.append(df)

        # Binance (needs API key + VPN)
        df = _safe(_fetch_liquidations_binance, coin, 30, label=f"liq/binance/{coin}")
        if df is not None:
            frames.append(df)

    if not frames:
        log.warning("No liquidation data retrieved")
        return

    new_liqs = pl.concat(frames).sort("timestamp")

    # Merge with existing — keep OKX data from previous pull, add 0xArchive
    try:
        existing = load_parquet_pl("liquidations")
        combined = pl.concat([existing, new_liqs])
        combined = combined.unique(subset=["timestamp", "venue", "coin", "side", "price"]).sort("timestamp")
    except FileNotFoundError:
        combined = new_liqs

    save_parquet_pl(combined, "liquidations")

    venues = combined["venue"].unique().sort().to_list()
    log.info(f"  Saved {len(combined)} liquidation rows")
    for v in venues:
        count = combined.filter(pl.col("venue") == v).height
        log.info(f"    {v:15s} {count:>8,d} rows")


def backfill_dydx_funding():
    """Re-pull dYdX funding with proper 1h→8h aggregation."""
    log.info("── Backfilling dYdX Funding (1h→8h aggregation) ──")

    frames: list[pl.DataFrame] = []
    for coin in TOKENS:
        df = _safe(_fetch_funding_dydx, coin, 364, label=f"funding/dydx/{coin}")
        if df is not None:
            frames.append(df)

    if not frames:
        log.warning("No dYdX funding data retrieved")
        return

    new_dydx = pl.concat(frames)

    # Replace dYdX rows in existing funding, keep all other venues
    try:
        existing = load_parquet_pl("funding_rates")
        non_dydx = existing.filter(pl.col("venue") != "dydx")
        combined = pl.concat([non_dydx, new_dydx])
        combined = combined.unique(subset=["timestamp", "venue", "coin"]).sort("timestamp")
    except FileNotFoundError:
        combined = new_dydx

    save_parquet_pl(combined, "funding_rates")

    # Summary
    dydx_count = combined.filter(pl.col("venue") == "dydx").height
    total = combined.height
    log.info(f"  Saved {total} total funding rows ({dydx_count} dYdX, now 8h aggregated)")


def main():
    backfill_liquidations()
    log.info("")
    backfill_dydx_funding()
    log.info("")

    # Quick validation
    log.info("── Validation ──")
    for name in ["liquidations", "funding_rates"]:
        try:
            df = load_parquet_pl(name)
            ts_min = df["timestamp"].min()
            ts_max = df["timestamp"].max()
            log.info(f"  {name}: {len(df)} rows, {ts_min} → {ts_max}")
        except FileNotFoundError:
            log.warning(f"  {name}: not found")


if __name__ == "__main__":
    main()
