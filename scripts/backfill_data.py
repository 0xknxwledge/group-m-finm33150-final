#!/usr/bin/env python
"""Backfill dYdX funding (merges into existing parquet).

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
    _fetch_funding_dydx,
)
from funding_the_fall.data.storage import save_parquet_pl, load_parquet_pl

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


def backfill_dydx_funding():
    """Re-pull dYdX funding (native 1h rates)."""
    log.info("── Backfilling dYdX Funding (1h rates) ──")

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

    dydx_count = combined.filter(pl.col("venue") == "dydx").height
    total = combined.height
    log.info(f"  Saved {total} total funding rows ({dydx_count} dYdX, 1h rates)")


def main():
    backfill_dydx_funding()


if __name__ == "__main__":
    main()
