"""Tests for data storage helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import polars as pl
import pytest

from funding_the_fall.data import storage

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _patch_data_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "DATA_DIR", tmp_path)


# ---------------------------------------------------------------------------
# Unit tests — polars round-trip
# ---------------------------------------------------------------------------


class TestPolarsRoundTrip:
    def test_save_load_round_trip(self, tmp_path):
        df = pl.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
        path = storage.save_parquet_pl(df, "test_pl")
        assert path == tmp_path / "test_pl.parquet"
        assert path.exists()

        loaded = storage.load_parquet_pl("test_pl")
        assert loaded.equals(df)

    def test_overwrites_existing(self, tmp_path):
        df1 = pl.DataFrame({"x": [1]})
        df2 = pl.DataFrame({"x": [2]})
        storage.save_parquet_pl(df1, "overwrite")
        storage.save_parquet_pl(df2, "overwrite")
        loaded = storage.load_parquet_pl("overwrite")
        assert loaded.equals(df2)


# ---------------------------------------------------------------------------
# Unit tests — pandas round-trip
# ---------------------------------------------------------------------------


class TestPandasRoundTrip:
    def test_save_load_round_trip(self, tmp_path):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
        path = storage._save_parquet_pd(df, "test_pd")
        assert path == tmp_path / "test_pd.parquet"
        assert path.exists()

        loaded = storage._load_parquet_pd("test_pd")
        pd.testing.assert_frame_equal(loaded, df)


# ---------------------------------------------------------------------------
# Aliases
# ---------------------------------------------------------------------------


class TestAliases:
    def test_save_parquet_is_pandas_helper(self):
        assert storage.save_parquet is storage._save_parquet_pd

    def test_load_parquet_is_pandas_helper(self):
        assert storage.load_parquet is storage._load_parquet_pd


# ---------------------------------------------------------------------------
# FileNotFoundError
# ---------------------------------------------------------------------------


class TestMissingFile:
    def test_load_parquet_pl_raises(self):
        with pytest.raises(FileNotFoundError):
            storage.load_parquet_pl("nonexistent")

    def test_load_parquet_pd_raises(self):
        with pytest.raises(FileNotFoundError):
            storage._load_parquet_pd("nonexistent")


# ---------------------------------------------------------------------------
# _ensure_dir
# ---------------------------------------------------------------------------


class TestEnsureDir:
    def test_creates_directory(self, tmp_path, monkeypatch):
        new_dir = tmp_path / "nested" / "deep"
        monkeypatch.setattr(storage, "DATA_DIR", new_dir)
        storage._ensure_dir()
        assert new_dir.is_dir()

    def test_idempotent(self, tmp_path):
        storage._ensure_dir()
        storage._ensure_dir()
        assert tmp_path.is_dir()


# ---------------------------------------------------------------------------
# DATA_DIR resolution
# ---------------------------------------------------------------------------


class TestDataDir:
    def test_resolves_to_project_root_data(self):
        assert DATA_DIR == Path(__file__).resolve().parent.parent / "data"
        # The module-level constant (before monkeypatch) should match
        project_root = Path(__file__).resolve().parent.parent
        expected = project_root / "data"
        assert expected.name == "data"
        assert expected.parent == project_root


# ---------------------------------------------------------------------------
# Integration tests — real parquet files
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestLoadFunding:
    @pytest.fixture(autouse=True)
    def _use_real_dir(self, monkeypatch):
        monkeypatch.setattr(storage, "DATA_DIR", DATA_DIR)
        if not (DATA_DIR / "funding_rates.parquet").exists():
            pytest.skip("funding_rates.parquet not present")

    def test_returns_dataframe_with_expected_columns(self):
        df = storage.load_funding()
        assert isinstance(df, pl.DataFrame)
        for col in ("timestamp", "venue", "coin", "funding_rate"):
            assert col in df.columns


@pytest.mark.integration
class TestLoadCandles:
    @pytest.fixture(autouse=True)
    def _use_real_dir(self, monkeypatch):
        monkeypatch.setattr(storage, "DATA_DIR", DATA_DIR)
        if not (DATA_DIR / "candles.parquet").exists():
            pytest.skip("candles.parquet not present")

    def test_returns_dataframe_with_expected_columns(self):
        df = storage.load_candles()
        assert isinstance(df, pl.DataFrame)
        for col in ("timestamp", "venue", "coin", "o", "h", "l", "c", "v"):
            assert col in df.columns


@pytest.mark.integration
class TestLoadOI:
    @pytest.fixture(autouse=True)
    def _use_real_dir(self, monkeypatch):
        monkeypatch.setattr(storage, "DATA_DIR", DATA_DIR)
        if not (DATA_DIR / "open_interest.parquet").exists():
            pytest.skip("open_interest.parquet not present")

    def test_returns_dataframe_with_expected_columns(self):
        df = storage.load_oi()
        assert isinstance(df, pl.DataFrame)
        for col in ("timestamp", "venue", "coin", "oi_usd"):
            assert col in df.columns


@pytest.mark.integration
class TestLoadOrderbookDepth:
    @pytest.fixture(autouse=True)
    def _use_real_dir(self, monkeypatch):
        monkeypatch.setattr(storage, "DATA_DIR", DATA_DIR)
        if not (DATA_DIR / "orderbook_depth.parquet").exists():
            pytest.skip("orderbook_depth.parquet not present")

    def test_returns_dataframe_with_expected_columns(self):
        df = storage.load_orderbook_depth()
        assert isinstance(df, pl.DataFrame)
        assert "coin" in df.columns


@pytest.mark.integration
class TestLoadLiquidationVolume:
    @pytest.fixture(autouse=True)
    def _use_real_dir(self, monkeypatch):
        monkeypatch.setattr(storage, "DATA_DIR", DATA_DIR)
        if not (DATA_DIR / "liquidation_volume.parquet").exists():
            pytest.skip("liquidation_volume.parquet not present")

    def test_returns_dataframe_with_expected_columns(self):
        df = storage.load_liquidation_volume()
        assert isinstance(df, pl.DataFrame)
        assert "coin" in df.columns or "total_usd" in df.columns
