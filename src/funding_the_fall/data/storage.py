"""Local data storage — save/load parquet files with caching.

Owner: Antonio Braz

Convention: all data lives in data/ at the project root.
Files are named: {data_type}.parquet
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import polars as pl

DATA_DIR = Path(__file__).resolve().parents[3] / "data"


def _ensure_dir() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


# ── Pandas helpers (notebook compat) ──


def _save_parquet_pd(df: pd.DataFrame, name: str) -> Path:
    """Save a pandas DataFrame to data/{name}.parquet."""
    _ensure_dir()
    path = DATA_DIR / f"{name}.parquet"
    df.to_parquet(path, index=False)
    return path


def _load_parquet_pd(name: str) -> pd.DataFrame:
    """Load data/{name}.parquet as pandas."""
    path = DATA_DIR / f"{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"No data file: {path}")
    return pd.read_parquet(path)


# Keep old names as aliases for backward compat
save_parquet = _save_parquet_pd
load_parquet = _load_parquet_pd


# ── Polars (primary) ──


def save_parquet_pl(df: pl.DataFrame, name: str) -> Path:
    """Save a polars DataFrame to data/{name}.parquet."""
    _ensure_dir()
    path = DATA_DIR / f"{name}.parquet"
    df.write_parquet(path)
    return path


def load_parquet_pl(name: str) -> pl.DataFrame:
    """Load data/{name}.parquet as polars."""
    path = DATA_DIR / f"{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"No data file: {path}")
    return pl.read_parquet(path)


# ── Typed loaders (polars) ──


def load_funding() -> pl.DataFrame:
    """Load the unified funding rate dataset."""
    return load_parquet_pl("funding_rates")


def load_candles() -> pl.DataFrame:
    """Load the unified candle dataset."""
    return load_parquet_pl("candles")


def load_oi() -> pl.DataFrame:
    """Load the unified open interest dataset."""
    return load_parquet_pl("open_interest")


def load_orderbook_depth() -> pl.DataFrame:
    """Load the orderbook depth snapshot."""
    return load_parquet_pl("orderbook_depth")


def load_liquidation_volume() -> pl.DataFrame:
    """Load the historical liquidation volume dataset."""
    return load_parquet_pl("liquidation_volume")
