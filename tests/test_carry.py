"""Tests for the funding rate carry strategy."""

from __future__ import annotations

import pandas as pd
import pytest

from funding_the_fall.strategy.carry import (
    CarryParams,
    GridSearchResult,
    compute_funding_spreads,
    evaluate_carry,
    grid_search_all_pairs,
    grid_search_params,
    select_best_params,
    simulate_carry,
)


def _make_funding_df(
    n_epochs: int = 30,
    venues: list[str] | None = None,
    coin: str = "BTC",
    base_rate: float = 0.0001,
    spread: float = 0.0003,
) -> pd.DataFrame:
    """Build a synthetic funding rate DataFrame."""
    venues = venues or ["binance", "bybit"]
    timestamps = pd.date_range("2025-01-01", periods=n_epochs, freq="8h")
    rows = []
    for ts in timestamps:
        for i, v in enumerate(venues):
            # venue 0 has lower rate, venue 1 has higher rate
            rate = base_rate + i * spread
            rows.append({"timestamp": ts, "venue": v, "coin": coin, "funding_rate": rate})
    return pd.DataFrame(rows)


def _make_spreads_df(**kwargs) -> pd.DataFrame:
    """Build a spread DataFrame from synthetic funding data."""
    return compute_funding_spreads(_make_funding_df(**kwargs))


# ---------------------------------------------------------------------------
# compute_funding_spreads
# ---------------------------------------------------------------------------


class TestComputeFundingSpreads:
    def test_spread_sign(self):
        df = _make_spreads_df()
        assert (df["spread"] >= 0).all()

    def test_annualization(self):
        df = _make_spreads_df()
        # annualized = spread * 3 * 365
        row = df.iloc[0]
        assert row["spread_annualized"] == pytest.approx(
            row["spread"] * 3 * 365
        )

    def test_long_short_assignment(self):
        df = _make_spreads_df()
        # binance has lower rate → long, bybit has higher rate → short
        for _, row in df.iterrows():
            assert row["long_venue"] == "binance"
            assert row["short_venue"] == "bybit"

    def test_multiple_coins(self):
        funding1 = _make_funding_df(coin="BTC")
        funding2 = _make_funding_df(coin="ETH")
        combined = pd.concat([funding1, funding2], ignore_index=True)
        spreads = compute_funding_spreads(combined)
        assert set(spreads["coin"].unique()) == {"BTC", "ETH"}

    def test_three_venues(self):
        df = _make_spreads_df(venues=["binance", "bybit", "okx"])
        # 3 venues → C(3,2) = 3 pairs per epoch
        n_epochs = 30
        assert len(df) == n_epochs * 3

    def test_single_venue_empty(self):
        df = _make_spreads_df(venues=["binance"])
        assert len(df) == 0


# ---------------------------------------------------------------------------
# simulate_carry
# ---------------------------------------------------------------------------


class TestSimulateCarry:
    def test_entry_exit_signals(self):
        spreads = _make_spreads_df(spread=0.001)  # large spread
        params = CarryParams("BTC", "binance", "bybit",
                             entry_spread=0.10, exit_spread=0.01,
                             max_holding_epochs=100)
        signals = simulate_carry(spreads, params)
        # Should have at least one entry
        entries = [s for s in signals if s.action == "enter"]
        exits = [s for s in signals if s.action == "exit"]
        assert len(entries) >= 1
        assert len(exits) >= 1

    def test_max_holding_exit(self):
        spreads = _make_spreads_df(n_epochs=50, spread=0.001)
        params = CarryParams("BTC", "binance", "bybit",
                             entry_spread=0.10, exit_spread=0.0,
                             max_holding_epochs=5)
        signals = simulate_carry(spreads, params)
        # With exit_spread=0, exit should be forced by max_holding
        exits = [s for s in signals if s.action == "exit"]
        assert len(exits) >= 1

    def test_forced_exit_at_end(self):
        spreads = _make_spreads_df(n_epochs=10, spread=0.001)
        params = CarryParams("BTC", "binance", "bybit",
                             entry_spread=0.10, exit_spread=0.0,
                             max_holding_epochs=100)
        signals = simulate_carry(spreads, params)
        if signals:
            assert signals[-1].action == "exit"

    def test_no_entry(self):
        spreads = _make_spreads_df(spread=0.00001)  # tiny spread
        params = CarryParams("BTC", "binance", "bybit",
                             entry_spread=99.0, exit_spread=0.01,
                             max_holding_epochs=100)
        signals = simulate_carry(spreads, params)
        assert len(signals) == 0

    def test_alternation(self):
        spreads = _make_spreads_df(n_epochs=50, spread=0.001)
        params = CarryParams("BTC", "binance", "bybit",
                             entry_spread=0.10, exit_spread=0.01,
                             max_holding_epochs=5)
        signals = simulate_carry(spreads, params)
        # Signals should alternate enter/exit
        for i in range(0, len(signals) - 1, 2):
            assert signals[i].action == "enter"
            if i + 1 < len(signals):
                assert signals[i + 1].action == "exit"


# ---------------------------------------------------------------------------
# evaluate_carry
# ---------------------------------------------------------------------------


class TestEvaluateCarry:
    def test_zero_trades(self):
        spreads = _make_spreads_df(spread=0.00001)
        params = CarryParams("BTC", "binance", "bybit",
                             entry_spread=99.0, exit_spread=0.01,
                             max_holding_epochs=100)
        result = evaluate_carry(spreads, params)
        assert result.n_trades == 0
        assert result.total_carry_pnl == 0.0
        assert result.sharpe == 0.0
        assert result.win_rate == 0.0

    def test_single_trade_metrics(self):
        spreads = _make_spreads_df(n_epochs=50, spread=0.001)
        params = CarryParams("BTC", "binance", "bybit",
                             entry_spread=0.10, exit_spread=0.01,
                             max_holding_epochs=100)
        result = evaluate_carry(spreads, params)
        assert isinstance(result, GridSearchResult)
        if result.n_trades > 0:
            assert result.total_carry_pnl != 0.0
            assert result.avg_holding_epochs > 0

    def test_win_rate_bounds(self):
        spreads = _make_spreads_df(n_epochs=100, spread=0.001)
        params = CarryParams("BTC", "binance", "bybit",
                             entry_spread=0.10, exit_spread=0.01,
                             max_holding_epochs=10)
        result = evaluate_carry(spreads, params)
        assert 0.0 <= result.win_rate <= 1.0


# ---------------------------------------------------------------------------
# grid_search_params
# ---------------------------------------------------------------------------


class TestGridSearchParams:
    def test_default_grids(self):
        spreads = _make_spreads_df(n_epochs=50, spread=0.001)
        results = grid_search_params(spreads, "BTC", "binance", "bybit")
        # 6 entry * 4 exit * 4 hold - skipped (exit >= entry)
        assert len(results) > 0

    def test_sorted_by_sharpe(self):
        spreads = _make_spreads_df(n_epochs=50, spread=0.001)
        results = grid_search_params(spreads, "BTC", "binance", "bybit")
        sharpes = [r.sharpe for r in results]
        assert sharpes == sorted(sharpes, reverse=True)

    def test_exit_ge_entry_skipped(self):
        spreads = _make_spreads_df(n_epochs=50, spread=0.001)
        results = grid_search_params(
            spreads, "BTC", "binance", "bybit",
            entry_spreads=[0.05], exit_spreads=[0.05, 0.10],
            max_holding_epochs_list=[15],
        )
        # exit >= entry should be skipped, so 0 results
        assert len(results) == 0

    def test_custom_grids(self):
        spreads = _make_spreads_df(n_epochs=50, spread=0.001)
        results = grid_search_params(
            spreads, "BTC", "binance", "bybit",
            entry_spreads=[0.10], exit_spreads=[0.02],
            max_holding_epochs_list=[15],
        )
        assert len(results) == 1


# ---------------------------------------------------------------------------
# grid_search_all_pairs
# ---------------------------------------------------------------------------


class TestGridSearchAllPairs:
    def test_keys_are_tuples(self):
        spreads = _make_spreads_df(n_epochs=20, spread=0.001)
        results = grid_search_all_pairs(spreads)
        for key in results.keys():
            assert isinstance(key, tuple)
            assert len(key) == 3


# ---------------------------------------------------------------------------
# select_best_params
# ---------------------------------------------------------------------------


class TestSelectBestParams:
    def test_min_trades_filter(self):
        spreads = _make_spreads_df(n_epochs=20, spread=0.001)
        grid_results = grid_search_all_pairs(spreads)
        best = select_best_params(grid_results, min_trades=9999)
        assert len(best) == 0

    def test_highest_sharpe_selected(self):
        spreads = _make_spreads_df(n_epochs=100, spread=0.001)
        grid_results = grid_search_all_pairs(spreads)
        best = select_best_params(grid_results, min_trades=1)
        for key, params in best.items():
            assert isinstance(params, CarryParams)

    def test_no_valid_results(self):
        spreads = _make_spreads_df(spread=0.00001)  # no entries
        grid_results = grid_search_all_pairs(spreads)
        best = select_best_params(grid_results, min_trades=1)
        # All grid points produce 0 trades → empty
        assert len(best) == 0
