"""Tests for the liquidation cascade simulator."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from funding_the_fall.models.cascade import (
    CascadeResult,
    Position,
    _is_liquidated,
    _price_impact,
    build_positions_from_oi,
    cascade_risk_signal,
    compute_amplification_curve,
    sensitivity_to_depth,
    sensitivity_to_leverage,
    simulate_cascade,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


# ---------------------------------------------------------------------------
# _is_liquidated
# ---------------------------------------------------------------------------


class TestIsLiquidated:
    def test_safe_position(self):
        pos = Position(
            collateral_usd=200, debt_usd=100, liquidation_threshold=1.25, layer="perp"
        )
        assert _is_liquidated(pos, 0.05) is False

    def test_underwater_position(self):
        pos = Position(
            collateral_usd=120, debt_usd=100, liquidation_threshold=1.0, layer="perp"
        )
        assert _is_liquidated(pos, 0.25) is True

    def test_exact_threshold_not_liquidated(self):
        # HF = (100 * (1 - 0) * 1.0) / 100 = 1.0 → not liquidated (check is < 1)
        pos = Position(
            collateral_usd=100, debt_usd=100, liquidation_threshold=1.0, layer="perp"
        )
        assert _is_liquidated(pos, 0.0) is False

    def test_zero_debt_never_liquidated(self):
        pos = Position(
            collateral_usd=1000,
            debt_usd=0,
            liquidation_threshold=1.25,
            layer="hyperlend",
        )
        assert _is_liquidated(pos, 0.99) is False

    def test_full_shock_liquidates(self):
        pos = Position(
            collateral_usd=200, debt_usd=100, liquidation_threshold=1.25, layer="morpho"
        )
        assert _is_liquidated(pos, 1.0) is True


# ---------------------------------------------------------------------------
# _price_impact
# ---------------------------------------------------------------------------


class TestPriceImpact:
    def test_known_value(self):
        # sqrt(1_000_000 / 4_000_000) = sqrt(0.25) = 0.5
        assert _price_impact(1_000_000, 4_000_000) == pytest.approx(0.5)

    def test_zero_volume(self):
        assert _price_impact(0, 5_000_000) == 0.0

    def test_zero_depth(self):
        assert _price_impact(1_000_000, 0) == 0.0

    def test_massive_volume_capped(self):
        assert _price_impact(100_000_000, 1_000) == 1.0


# ---------------------------------------------------------------------------
# simulate_cascade
# ---------------------------------------------------------------------------


class TestSimulateCascade:
    @staticmethod
    def _safe_positions() -> list[Position]:
        return [
            Position(
                collateral_usd=10_000,
                debt_usd=1_000,
                liquidation_threshold=1.25,
                layer="perp",
            ),
            Position(
                collateral_usd=10_000,
                debt_usd=1_000,
                liquidation_threshold=1.25,
                layer="hyperlend",
            ),
        ]

    @staticmethod
    def _fragile_positions() -> list[Position]:
        """Highly leveraged positions that cascade easily."""
        return [
            # HF after 5% drop: 1020*0.95/1000 = 0.969 → liquidated round 1
            Position(
                collateral_usd=1020,
                debt_usd=1000,
                liquidation_threshold=1.0,
                layer="perp",
            ),
            # Survives 5% but not the additional price impact from round 1
            Position(
                collateral_usd=1100,
                debt_usd=1000,
                liquidation_threshold=1.0,
                layer="hyperlend",
            ),
            # Even more cushion — needs round 2 impact
            Position(
                collateral_usd=1200,
                debt_usd=1000,
                liquidation_threshold=1.0,
                layer="morpho",
            ),
        ]

    def test_no_liquidations(self):
        result = simulate_cascade(self._safe_positions(), 100.0, 0.01)
        assert result.amplification == pytest.approx(1.0)
        assert result.rounds == 0
        assert result.total_debt_liquidated == 0.0

    def test_single_round(self):
        positions = [
            Position(
                collateral_usd=1050,
                debt_usd=1000,
                liquidation_threshold=1.0,
                layer="perp",
            ),
        ]
        # 10% shock liquidates this position; only one round needed
        result = simulate_cascade(positions, 100.0, 0.10, orderbook_depth_usd=1e12)
        assert result.rounds == 1
        assert result.total_debt_liquidated == pytest.approx(1000.0)

    def test_multi_round_cascade(self):
        result = simulate_cascade(
            self._fragile_positions(),
            100.0,
            0.05,
            orderbook_depth_usd=500_000,
        )
        assert result.rounds >= 2
        assert result.effective_shock > 0.05

    def test_layer_tracking(self):
        result = simulate_cascade(
            self._fragile_positions(),
            100.0,
            0.10,
            orderbook_depth_usd=500_000,
        )
        assert isinstance(result.liquidations_by_layer, dict)
        total_from_layers = sum(result.liquidations_by_layer.values())
        assert total_from_layers == pytest.approx(result.total_debt_liquidated)

    def test_empty_positions(self):
        result = simulate_cascade([], 100.0, 0.10)
        assert result.rounds == 0
        assert result.total_debt_liquidated == 0.0

    def test_max_rounds_cap(self):
        # Positions that always re-trigger — cap prevents infinite loop
        huge = [
            Position(
                collateral_usd=1e9,
                debt_usd=1e9,
                liquidation_threshold=1.0,
                layer="perp",
            )
            for _ in range(200)
        ]
        result = simulate_cascade(
            huge, 100.0, 0.01, orderbook_depth_usd=1.0, max_rounds=5
        )
        assert result.rounds <= 5


# ---------------------------------------------------------------------------
# compute_amplification_curve
# ---------------------------------------------------------------------------


class TestAmplificationCurve:
    @staticmethod
    def _positions() -> list[Position]:
        return [
            Position(
                collateral_usd=1100,
                debt_usd=1000,
                liquidation_threshold=1.0,
                layer="perp",
            ),
        ]

    def test_default_shocks_count(self):
        results = compute_amplification_curve(self._positions(), 100.0)
        assert len(results) == 99

    def test_custom_shocks(self):
        shocks = np.array([0.01, 0.05, 0.10])
        results = compute_amplification_curve(self._positions(), 100.0, shocks=shocks)
        assert len(results) == 3

    def test_effective_shock_non_decreasing(self):
        results = compute_amplification_curve(self._positions(), 100.0)
        effective = [r.effective_shock for r in results]
        for a, b in zip(effective, effective[1:]):
            assert b >= a - 1e-12


# ---------------------------------------------------------------------------
# cascade_risk_signal
# ---------------------------------------------------------------------------


class TestCascadeRiskSignal:
    def test_low_risk(self):
        safe = [
            Position(
                collateral_usd=10_000,
                debt_usd=100,
                liquidation_threshold=1.25,
                layer="perp",
            ),
        ]
        sig = cascade_risk_signal(safe, 100.0)
        assert sig["risk_score"] == pytest.approx(0.0)
        assert not sig["signal"]
        assert sig["critical_shock"] is None

    def test_high_risk(self):
        fragile = [
            Position(
                collateral_usd=1050,
                debt_usd=1000,
                liquidation_threshold=1.0,
                layer="perp",
            )
            for _ in range(50)
        ]
        sig = cascade_risk_signal(fragile, 100.0, orderbook_depth_usd=100_000)
        assert sig["risk_score"] > 0.5
        assert sig["signal"]
        assert isinstance(sig["critical_shock"], (float, np.floating))

    def test_signal_keys(self):
        sig = cascade_risk_signal([], 100.0)
        assert set(sig.keys()) == {
            "risk_score",
            "critical_shock",
            "amplification_at_5pct",
            "signal",
        }


# ---------------------------------------------------------------------------
# build_positions_from_oi
# ---------------------------------------------------------------------------


class TestBuildPositionsFromOI:
    @staticmethod
    def _make_oi_df(
        timestamps: list[str],
        venues: list[str],
        coins: list[str],
        oi_values: list[float],
    ):
        import polars as pl

        return pl.DataFrame(
            {
                "timestamp": timestamps,
                "venue": venues,
                "coin": coins,
                "oi_usd": oi_values,
            }
        ).with_columns(pl.col("timestamp").str.to_datetime(time_zone="UTC"))

    def test_synthetic_df(self):
        df = self._make_oi_df(
            timestamps=[
                "2025-01-01T00:00:00",
                "2025-01-02T00:00:00",
                "2025-01-01T00:00:00",
            ],
            venues=["binance", "binance", "bybit"],
            coins=["BTC", "BTC", "ETH"],
            oi_values=[1_000_000.0, 2_000_000.0, 500_000.0],
        )
        positions = build_positions_from_oi(df, leverage=5.0)
        assert len(positions) == 2  # latest BTC + ETH
        total_debt = sum(p.debt_usd for p in positions)
        # 2_000_000 * 4/5 + 500_000 * 4/5 = 2_000_000
        assert total_debt == pytest.approx(2_000_000.0)

    def test_custom_params(self):
        df = self._make_oi_df(
            timestamps=["2025-01-01T00:00:00"],
            venues=["hyperliquid"],
            coins=["SOL"],
            oi_values=[100_000.0],
        )
        positions = build_positions_from_oi(
            df, leverage=10.0, liq_threshold=1.25, layer="morpho"
        )
        assert len(positions) == 1
        assert positions[0].collateral_usd == pytest.approx(10_000.0)
        assert positions[0].debt_usd == pytest.approx(90_000.0)
        assert positions[0].liquidation_threshold == 1.25
        assert positions[0].layer == "morpho"

    def test_zero_oi_skipped(self):
        df = self._make_oi_df(
            timestamps=["2025-01-01T00:00:00"],
            venues=["binance"],
            coins=["BTC"],
            oi_values=[0.0],
        )
        assert build_positions_from_oi(df) == []

    def test_empty_df(self):
        import polars as pl

        df = pl.DataFrame(
            {
                "timestamp": pl.Series([], dtype=pl.Datetime("us", "UTC")),
                "venue": pl.Series([], dtype=pl.Utf8),
                "coin": pl.Series([], dtype=pl.Utf8),
                "oi_usd": pl.Series([], dtype=pl.Float64),
            }
        )
        assert build_positions_from_oi(df) == []

    def test_multiple_snapshots_picks_latest(self):
        df = self._make_oi_df(
            timestamps=[
                "2025-01-01T00:00:00",
                "2025-01-02T00:00:00",
                "2025-01-03T00:00:00",
            ],
            venues=["binance", "binance", "binance"],
            coins=["BTC", "BTC", "BTC"],
            oi_values=[1_000_000.0, 2_000_000.0, 3_000_000.0],
        )
        positions = build_positions_from_oi(df, leverage=5.0)
        assert len(positions) == 1
        assert positions[0].collateral_usd == pytest.approx(3_000_000.0 / 5.0)

    def test_negative_oi_skipped(self):
        df = self._make_oi_df(
            timestamps=["2025-01-01T00:00:00"],
            venues=["binance"],
            coins=["BTC"],
            oi_values=[-500_000.0],
        )
        assert build_positions_from_oi(df) == []

    def test_null_oi_skipped(self):
        import polars as pl

        df = pl.DataFrame(
            {
                "timestamp": ["2025-01-01T00:00:00"],
                "venue": ["binance"],
                "coin": ["BTC"],
                "oi_usd": [None],
            }
        ).with_columns(
            pl.col("timestamp").str.to_datetime(time_zone="UTC"),
            pl.col("oi_usd").cast(pl.Float64),
        )
        assert build_positions_from_oi(df) == []

    def test_leverage_math(self):
        for lev in [2.0, 10.0, 20.0]:
            notional = 1_000_000.0
            df = self._make_oi_df(
                timestamps=["2025-01-01T00:00:00"],
                venues=["binance"],
                coins=["BTC"],
                oi_values=[notional],
            )
            positions = build_positions_from_oi(df, leverage=lev)
            assert len(positions) == 1
            expected_collateral = notional / lev
            expected_debt = notional - expected_collateral
            assert positions[0].collateral_usd == pytest.approx(expected_collateral)
            assert positions[0].debt_usd == pytest.approx(expected_debt)

    def test_all_positions_have_correct_layer(self):
        df = self._make_oi_df(
            timestamps=["2025-01-01T00:00:00", "2025-01-01T00:00:00"],
            venues=["binance", "bybit"],
            coins=["BTC", "ETH"],
            oi_values=[1_000_000.0, 500_000.0],
        )
        positions = build_positions_from_oi(df, layer="morpho")
        assert all(p.layer == "morpho" for p in positions)


# ---------------------------------------------------------------------------
# Integration test — uses real parquet data
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestCascadeWithRealOI:
    OI_PATH = DATA_DIR / "open_interest.parquet"
    LIQ_PATH = DATA_DIR / "liquidations.parquet"

    @pytest.fixture(autouse=True)
    def _require_data(self):
        if not self.OI_PATH.exists() or not self.LIQ_PATH.exists():
            pytest.skip("parquet data files not present")

    def test_cascade_with_real_oi(self):
        import polars as pl

        oi = pl.read_parquet(self.OI_PATH)
        positions = build_positions_from_oi(oi, leverage=5.0)

        if not positions:
            pytest.skip("no valid OI rows found")

        result = simulate_cascade(positions, 1.0, 0.10)
        assert isinstance(result, CascadeResult)
        assert result.effective_shock >= 0.10
        assert result.amplification >= 1.0

    def test_end_to_end_oi_to_risk_signal(self):
        import polars as pl

        oi = pl.read_parquet(self.OI_PATH)
        positions = build_positions_from_oi(oi, leverage=5.0)

        if not positions:
            pytest.skip("no valid OI rows found")

        sig = cascade_risk_signal(positions, 1.0)
        assert set(sig.keys()) == {
            "risk_score",
            "critical_shock",
            "amplification_at_5pct",
            "signal",
        }
        assert 0.0 <= sig["risk_score"] <= 1.0
        assert isinstance(sig["signal"], bool)
        assert sig["amplification_at_5pct"] >= 1.0
        if sig["critical_shock"] is not None:
            assert 0.0 < sig["critical_shock"] <= 0.5


# ---------------------------------------------------------------------------
# Interface contract tests
# ---------------------------------------------------------------------------


class TestCascadeInterfaces:
    """Verify cascade output contracts match downstream consumers."""

    def test_risk_signal_dict_compatible_with_allocation(self):
        """cascade_risk_signal() output has keys/types allocation.allocate_positions expects."""
        positions = [
            Position(
                collateral_usd=1050,
                debt_usd=1000,
                liquidation_threshold=1.0,
                layer="perp",
            )
            for _ in range(20)
        ]
        sig = cascade_risk_signal(positions, 100.0, orderbook_depth_usd=100_000)

        assert isinstance(sig["risk_score"], float)
        assert 0.0 <= sig["risk_score"] <= 1.0
        assert isinstance(sig["signal"], bool)
        assert sig["critical_shock"] is None or isinstance(
            sig["critical_shock"], (float, np.floating)
        )
        assert isinstance(sig["amplification_at_5pct"], (float, np.floating))
        assert sig["amplification_at_5pct"] >= 1.0

    def test_models_init_reexports(self):
        """Public API re-exports from models.__init__."""
        from funding_the_fall.models import (
            build_positions_from_oi,
            cascade_risk_signal,
            compute_amplification_curve,
            sensitivity_to_depth,
            sensitivity_to_leverage,
            simulate_cascade,
        )

        assert callable(simulate_cascade)
        assert callable(cascade_risk_signal)
        assert callable(build_positions_from_oi)
        assert callable(compute_amplification_curve)
        assert callable(sensitivity_to_leverage)
        assert callable(sensitivity_to_depth)


# ---------------------------------------------------------------------------
# sensitivity_to_leverage
# ---------------------------------------------------------------------------


class TestSensitivityToLeverage:
    @staticmethod
    def _make_oi_df():
        import polars as pl

        return pl.DataFrame(
            {
                "timestamp": ["2025-01-01T00:00:00", "2025-01-01T00:00:00"],
                "venue": ["binance", "bybit"],
                "coin": ["BTC", "ETH"],
                "oi_usd": [5_000_000.0, 2_000_000.0],
            }
        ).with_columns(pl.col("timestamp").str.to_datetime(time_zone="UTC"))

    def test_default_leverages(self):
        result = sensitivity_to_leverage(self._make_oi_df())
        assert set(result.keys()) == {3.0, 5.0, 10.0, 20.0}

    def test_custom_leverages(self):
        result = sensitivity_to_leverage(self._make_oi_df(), leverages=[2.0, 7.0])
        assert set(result.keys()) == {2.0, 7.0}

    def test_values_are_cascade_result_lists(self):
        result = sensitivity_to_leverage(
            self._make_oi_df(),
            shocks=np.array([0.05, 0.10]),
        )
        for curves in result.values():
            assert isinstance(curves, list)
            assert all(isinstance(r, CascadeResult) for r in curves)
            assert len(curves) == 2

    def test_higher_leverage_more_debt_liquidated(self):
        shocks = np.array([0.10])
        result = sensitivity_to_leverage(
            self._make_oi_df(),
            leverages=[3.0, 20.0],
            shocks=shocks,
            orderbook_depth_usd=1e9,
        )
        debt_low = result[3.0][0].total_debt_liquidated
        debt_high = result[20.0][0].total_debt_liquidated
        assert debt_high >= debt_low


# ---------------------------------------------------------------------------
# sensitivity_to_depth
# ---------------------------------------------------------------------------


class TestSensitivityToDepth:
    @staticmethod
    def _positions() -> list[Position]:
        return [
            Position(
                collateral_usd=1100,
                debt_usd=1000,
                liquidation_threshold=1.0,
                layer="perp",
            )
            for _ in range(20)
        ]

    def test_default_depths(self):
        result = sensitivity_to_depth(self._positions())
        assert set(result.keys()) == {1e6, 5e6, 10e6, 50e6}

    def test_custom_depths(self):
        result = sensitivity_to_depth(self._positions(), depths_usd=[100_000, 1e6])
        assert set(result.keys()) == {100_000, 1e6}

    def test_values_are_cascade_result_lists(self):
        result = sensitivity_to_depth(
            self._positions(),
            shocks=np.array([0.05, 0.10]),
        )
        for curves in result.values():
            assert isinstance(curves, list)
            assert all(isinstance(r, CascadeResult) for r in curves)
            assert len(curves) == 2

    def test_deeper_book_lower_amplification(self):
        shocks = np.array([0.10, 0.20])
        result = sensitivity_to_depth(
            self._positions(),
            depths_usd=[100_000, 50e6],
            shocks=shocks,
        )
        amp_shallow = max(r.amplification for r in result[100_000])
        amp_deep = max(r.amplification for r in result[50e6])
        assert amp_shallow >= amp_deep
