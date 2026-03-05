"""Local data storage — save/load parquet files with caching.

Owner: Antonio Braz

Convention: all data lives in data/ at the project root.
Files are named: {data_type}.parquet
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[3] / "data"


def _ensure_dir() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def save_parquet(df: pd.DataFrame, name: str) -> Path:
    """Save a DataFrame to data/{name}.parquet."""
    _ensure_dir()
    path = DATA_DIR / f"{name}.parquet"
    df.to_parquet(path, index=False)
    return path


def load_parquet(name: str) -> pd.DataFrame:
    """Load data/{name}.parquet, raising FileNotFoundError if missing."""
    path = DATA_DIR / f"{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"No data file: {path}")
    return pd.read_parquet(path)


def load_funding() -> pd.DataFrame:
    """Load the unified funding rate dataset."""
    return load_parquet("funding_rates")


def load_candles() -> pd.DataFrame:
    """Load the unified candle dataset."""
    return load_parquet("candles")


def load_oi() -> pd.DataFrame:
    """Load the unified open interest dataset."""
    return load_parquet("open_interest")


def load_liquidations() -> pd.DataFrame:
    """Load the unified liquidation events dataset."""
    return load_parquet("liquidations")


def load_lending() -> pd.DataFrame:
    """Load the unified lending positions dataset."""
    return load_parquet("lending_positions")
