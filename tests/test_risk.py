"""Tests for jump-weighted risk scoring."""

from __future__ import annotations

import polars as pl
import pytest

from funding_the_fall.models.merton import MertonParams
from funding_the_fall.models.cascade import Position
from funding_the_fall.models.risk import (
    jump_weighted_risk,
    jump_weighted_risk_all_coins,
)


def _make_params(**overrides) -> MertonParams:
    defaults = dict(sigma=0.01, lam=0.1, mu_j=-0.05, sigma_j=0.02, mu=0.0)
    defaults.update(overrides)
    return MertonParams(**defaults)


def _safe_positions() -> list[Position]:
    return [
        Position(
            collateral_usd=500, debt_usd=500, liquidation_threshold=0.005, layer="perp"
        ),
    ]


def _fragile_positions() -> list[Position]:
    return [
        Position(
            collateral_usd=20, debt_usd=980, liquidation_threshold=0.005, layer="perp"
        )
        for _ in range(20)
    ]


def _make_oi_df(coins: list[str], oi: float = 1_000_000.0) -> pl.DataFrame:
    rows = []
    for coin in coins:
        rows.append(
            {
                "timestamp": "2025-01-01T00:00:00",
                "venue": "binance",
                "coin": coin,
                "oi_usd": oi,
            }
        )
    return pl.DataFrame(rows).with_columns(
        pl.col("timestamp").str.to_datetime(time_zone="UTC")
    )


# ---------------------------------------------------------------------------
# jump_weighted_risk
# ---------------------------------------------------------------------------


class TestJumpWeightedRisk:
    def test_output_keys(self):
        result = jump_weighted_risk(_make_params(), _safe_positions(), n_shocks=20)
        expected_keys = {
            "baseline_loss",
            "amplified_loss",
            "cascade_excess",
            "cascade_multiplier",
            "tail_probability_5pct",
            "amplification_at_5pct",
        }
        assert set(result.keys()) == expected_keys

    def test_amplified_ge_baseline(self):
        result = jump_weighted_risk(_make_params(), _safe_positions(), n_shocks=20)
        assert result["amplified_loss"] >= result["baseline_loss"] - 1e-12

    def test_multiplier_formula(self):
        result = jump_weighted_risk(_make_params(), _safe_positions(), n_shocks=20)
        if result["baseline_loss"] > 0:
            expected = result["amplified_loss"] / result["baseline_loss"]
            assert result["cascade_multiplier"] == pytest.approx(expected)

    def test_excess_formula(self):
        result = jump_weighted_risk(_make_params(), _safe_positions(), n_shocks=20)
        expected = result["amplified_loss"] - result["baseline_loss"]
        assert result["cascade_excess"] == pytest.approx(expected)

    def test_fragile_higher_amplification(self):
        safe = jump_weighted_risk(
            _make_params(), _safe_positions(), orderbook_depth_usd=100_000, n_shocks=20
        )
        fragile = jump_weighted_risk(
            _make_params(),
            _fragile_positions(),
            orderbook_depth_usd=100_000,
            n_shocks=20,
        )
        assert fragile["cascade_multiplier"] >= safe["cascade_multiplier"]

    def test_n_shocks_param(self):
        r1 = jump_weighted_risk(_make_params(), _safe_positions(), n_shocks=10)
        r2 = jump_weighted_risk(_make_params(), _safe_positions(), n_shocks=50)
        # Both should produce valid results
        assert r1["baseline_loss"] > 0
        assert r2["baseline_loss"] > 0


# ---------------------------------------------------------------------------
# jump_weighted_risk_all_coins
# ---------------------------------------------------------------------------


class TestJumpWeightedRiskAllCoins:
    def test_multiple_coins(self):
        params_dict = {"BTC": _make_params(), "ETH": _make_params(sigma=0.02)}
        oi_df = _make_oi_df(["BTC", "ETH"])
        results = jump_weighted_risk_all_coins(params_dict, oi_df)
        assert set(results.keys()) == {"BTC", "ETH"}

    def test_missing_coin_skipped(self):
        params_dict = {"BTC": _make_params(), "DOGE": _make_params()}
        oi_df = _make_oi_df(["BTC"])  # No DOGE OI
        results = jump_weighted_risk_all_coins(params_dict, oi_df)
        assert "BTC" in results
        assert "DOGE" not in results

    def test_tiered_flag(self):
        params_dict = {"BTC": _make_params()}
        oi_df = _make_oi_df(["BTC"])
        r_uniform = jump_weighted_risk_all_coins(params_dict, oi_df, tiered=False)
        r_tiered = jump_weighted_risk_all_coins(params_dict, oi_df, tiered=True)
        # Both should produce valid results; tiered may differ
        assert "BTC" in r_uniform
        assert "BTC" in r_tiered

    def test_depth_per_coin(self):
        params_dict = {"BTC": _make_params()}
        oi_df = _make_oi_df(["BTC"])
        depth = {"BTC": 500_000.0}
        results = jump_weighted_risk_all_coins(
            params_dict,
            oi_df,
            depth_per_coin=depth,
        )
        assert "BTC" in results
